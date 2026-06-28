#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

from contracts import (
    RUN_METADATA_SCHEMA_NAME,
    RUN_METADATA_SCHEMA_VERSION,
    SCORES_COLUMNS,
    SCORES_SCHEMA_NAME,
    SCORES_SCHEMA_VERSION,
    USAGE_COLUMNS,
    USAGE_SCHEMA_NAME,
    USAGE_SCHEMA_VERSION,
    write_schema_manifest,
)


def _canon(s: str) -> str:
    return (s or "").strip().lower()


def _migrate_csv_with_aliases(
    path: Path,
    *,
    target_columns: list[str],
    aliases: dict[str, str],
) -> bool:
    """
    Normalize CSV header/column order to `target_columns`.
    Returns True if file content was rewritten.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False

    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    if not rows:
        return False

    header = [h.strip() for h in rows[0]]
    if header == target_columns:
        return False

    mapped = [aliases.get(_canon(h), "") for h in header]
    if any(not m for m in mapped):
        raise ValueError(f"Unrecognized header columns in {path}: {header}")
    if len(set(mapped)) != len(mapped):
        raise ValueError(f"Duplicate mapped columns in {path}: {header}")

    index_by_col = {c: i for i, c in enumerate(mapped)}
    if any(c not in index_by_col for c in target_columns):
        raise ValueError(f"Missing required columns for {path}: have={mapped}, need={target_columns}")

    migrated: list[list[str]] = [target_columns]
    for row in rows[1:]:
        migrated.append(
            [row[index_by_col[c]] if index_by_col[c] < len(row) else "" for c in target_columns]
        )

    with path.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows(migrated)
    return True


def migrate_scores_csv(path: Path) -> bool:
    aliases = {
        "date_utc": "date_utc",
        "dateutc": "date_utc",
        "date": "date_utc",
        "dataset": "dataset",
        "model": "model",
        "norm_hamming": "norm_hamming",
        "norm": "norm_hamming",
        "exact_pct": "exact_pct",
        "exact": "exact_pct",
        "total_cost_usd": "total_cost_usd",
        "total_cost": "total_cost_usd",
        "pred_file": "pred_file",
        "preds": "pred_file",
    }
    changed = _migrate_csv_with_aliases(path, target_columns=SCORES_COLUMNS, aliases=aliases)
    if path.exists():
        write_schema_manifest(
            path,
            schema_name=SCORES_SCHEMA_NAME,
            schema_version=SCORES_SCHEMA_VERSION,
            fmt="csv",
            columns=SCORES_COLUMNS,
            notes="Migrated/validated against canonical scores schema.",
        )
    return changed


def migrate_usage_csv(path: Path) -> bool:
    aliases = {
        "ts": "ts",
        "timestamp": "ts",
        "model": "model",
        "prompt_tok": "prompt_tok",
        "prompt_tokens": "prompt_tok",
        "completion_tok": "completion_tok",
        "completion_tokens": "completion_tok",
        "total_tok": "total_tok",
        "total_tokens": "total_tok",
        "usd": "usd",
        "usd_cost": "usd",
    }
    changed = _migrate_csv_with_aliases(path, target_columns=USAGE_COLUMNS, aliases=aliases)
    if path.exists():
        write_schema_manifest(
            path,
            schema_name=USAGE_SCHEMA_NAME,
            schema_version=USAGE_SCHEMA_VERSION,
            fmt="csv",
            columns=USAGE_COLUMNS,
            notes="Migrated/validated against canonical usage schema.",
        )
    return changed


def migrate_run_metadata(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    changed = False
    out_lines: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object record")

        if obj.get("schema_name") != RUN_METADATA_SCHEMA_NAME:
            obj["schema_name"] = RUN_METADATA_SCHEMA_NAME
            changed = True
        if obj.get("schema_version") != RUN_METADATA_SCHEMA_VERSION:
            obj["schema_version"] = RUN_METADATA_SCHEMA_VERSION
            changed = True
        if "total_cost_usd" in obj and "cost_usd" not in obj:
            obj["cost_usd"] = obj.pop("total_cost_usd")
            changed = True

        out_lines.append(json.dumps(obj, separators=(",", ":")))

    if changed:
        path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    write_schema_manifest(
        path,
        schema_name=RUN_METADATA_SCHEMA_NAME,
        schema_version=RUN_METADATA_SCHEMA_VERSION,
        fmt="jsonl",
        notes="Migrated/validated against canonical run metadata schema.",
    )
    return changed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Migrate legacy CABench artifacts to canonical schemas.")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument("--scores", type=Path, help="Optional explicit scores.csv path")
    p.add_argument("--master-usage", type=Path, help="Optional explicit master_usage.csv path")
    p.add_argument("--run-metadata", type=Path, help="Optional explicit run_metadata.jsonl path")
    p.add_argument("--usage-glob", default="*_usage.csv", help="Glob for per-run usage files")
    return p


def _run_migration(name: str, fn: Callable[[Path], bool], path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"{name}: skipped (missing {path})"
    changed = fn(path)
    return changed, f"{name}: {'updated' if changed else 'ok'} ({path})"


def main() -> None:
    args = build_parser().parse_args()

    scores_path = args.scores or (args.results_dir / "scores.csv")
    metadata_path = args.run_metadata or (args.results_dir / "run_metadata.jsonl")
    master_usage = args.master_usage or (args.log_dir / "master_usage.csv")
    usage_paths = sorted(args.log_dir.glob(args.usage_glob))

    msgs: list[str] = []
    total_updates = 0

    for name, fn, path in (
        ("scores", migrate_scores_csv, scores_path),
        ("master_usage", migrate_usage_csv, master_usage),
        ("run_metadata", migrate_run_metadata, metadata_path),
    ):
        changed, msg = _run_migration(name, fn, path)
        msgs.append(msg)
        if changed:
            total_updates += 1

    for usage in usage_paths:
        changed, msg = _run_migration(f"usage:{usage.name}", migrate_usage_csv, usage)
        msgs.append(msg)
        if changed:
            total_updates += 1

    print("\n".join(msgs))
    print(f"migration_complete updates={total_updates}")


if __name__ == "__main__":
    main()
