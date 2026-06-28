#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _norm_row(row: dict[str, str]) -> dict[str, str]:
    return {(k or "").strip(): (v or "").strip() for k, v in row.items()}


def load_scores(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            return rows
        for raw in reader:
            r = _norm_row(raw)
            dataset = r.get("dataset", "")
            model = r.get("model", "")
            if not dataset or not model:
                continue
            norm = _to_float(r.get("norm_hamming"))
            exact = _to_float(r.get("exact_pct"))
            cost = _to_float(r.get("total_cost_usd")) or 0.0
            if norm is None or exact is None:
                # Skip dry-run or malformed metric rows in aggregates.
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "norm_hamming": norm,
                    "exact_pct": exact,
                    "cost_usd": cost,
                    "date_utc": r.get("date_utc", ""),
                    "pred_file": r.get("pred_file", ""),
                }
            )
    return rows


def load_scores_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_scores(path)


def load_run_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object record")
        rows.append(obj)
    return rows


def metric_rows_from_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        metrics = row.get("metrics") or {}
        dataset = row.get("dataset") or {}
        model = row.get("model") or {}
        norm = _to_float(metrics.get("norm_hamming"))
        exact = _to_float(metrics.get("exact_pct"))
        cost = _to_float(row.get("cost_usd")) or 0.0
        num_cases = _to_int(dataset.get("num_cases"))
        if norm is None or exact is None:
            continue
        if not dataset.get("name") or not model.get("id"):
            continue
        out.append(
            {
                "dataset": str(dataset["name"]),
                "model": str(model["id"]),
                "norm_hamming": norm,
                "exact_pct": exact,
                "cost_usd": cost,
                "num_cases": num_cases,
                "date_utc": str(row.get("run_finished_utc", "")),
            }
        )
    return out


def aggregate_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    out: list[dict[str, Any]] = []
    for model, items in by_model.items():
        n = len(items)
        avg_norm = sum(x["norm_hamming"] for x in items) / n
        avg_exact = sum(x["exact_pct"] for x in items) / n
        total_cost = sum(x["cost_usd"] for x in items)
        total_cases = sum(_to_int(x.get("num_cases")) or 0 for x in items)
        out.append(
            {
                "model": model,
                "runs": n,
                "avg_norm_hamming": avg_norm,
                "avg_exact_pct": avg_exact,
                "total_cost_usd": total_cost,
                "cost_per_case": (total_cost / total_cases) if total_cases else None,
            }
        )
    out.sort(key=lambda x: (-x["avg_norm_hamming"], -x["avg_exact_pct"], x["total_cost_usd"]))
    return out


def aggregate_by_model_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_key[(r["dataset"], r["model"])].append(r)

    out: list[dict[str, Any]] = []
    for (dataset, model), items in by_key.items():
        n = len(items)
        total_cost = sum(x["cost_usd"] for x in items)
        total_cases = sum(_to_int(x.get("num_cases")) or 0 for x in items)
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "runs": n,
                "avg_norm_hamming": sum(x["norm_hamming"] for x in items) / n,
                "avg_exact_pct": sum(x["exact_pct"] for x in items) / n,
                "total_cost_usd": total_cost,
                "cost_per_case": (total_cost / total_cases) if total_cases else None,
            }
        )
    out.sort(key=lambda x: (x["dataset"], -x["avg_norm_hamming"], -x["avg_exact_pct"], x["total_cost_usd"]))
    return out


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(vals: list[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    sep = "-+-".join("-" * w for w in widths)
    parts = [fmt_row(headers), sep]
    parts.extend(fmt_row(r) for r in rows)
    return "\n".join(parts)


def _render_kv_block(items: list[tuple[str, str]]) -> str:
    width = max(len(k) for k, _ in items) if items else 0
    return "\n".join(f"{k.ljust(width)} : {v}" for k, v in items)


def _invocation_sort_key(record: dict[str, Any]) -> str:
    return (
        str(record.get("invocation_started_utc", "")).strip()
        or str(record.get("run_started_utc", "")).strip()
        or str(record.get("run_finished_utc", "")).strip()
    )


def latest_invocation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = str(row.get("invocation_id", "")).strip() or f"legacy:{idx}:{_invocation_sort_key(row)}"
        grouped[key].append(row)

    return max(
        grouped.values(),
        key=lambda items: max(_invocation_sort_key(item) for item in items),
    )


def _safe_metric(value: Any, fmt: str) -> str:
    num = _to_float(value)
    if num is None:
        return "-"
    return format(num, fmt)


def render_latest_invocation(rows: list[dict[str, Any]]) -> str:
    latest = latest_invocation_rows(rows)
    if not latest:
        return "Latest invocation\n\nNo run metadata available."

    latest.sort(
        key=lambda r: (
            str((r.get("dataset") or {}).get("name", "")),
            str((r.get("model") or {}).get("id", "")),
        )
    )
    total_cost = 0.0
    total_cases = 0
    finished_times: list[str] = []
    spend_cap = None
    table_rows: list[list[str]] = []

    for row in latest:
        dataset = row.get("dataset") or {}
        model = row.get("model") or {}
        metrics = row.get("metrics") or {}
        status = str(row.get("status", "")).strip() or "unknown"
        cost = _to_float(row.get("cost_usd")) or 0.0
        n_cases = _to_int(dataset.get("num_cases")) or 0
        total_cost += cost
        total_cases += n_cases
        if row.get("run_finished_utc"):
            finished_times.append(str(row["run_finished_utc"]))

        cli_overrides = (row.get("config") or {}).get("cli_overrides") or {}
        if spend_cap is None:
            spend_cap = _to_float(cli_overrides.get("spend_cap_usd"))

        cost_per_case = cost / n_cases if n_cases else None
        table_rows.append(
            [
                str(dataset.get("name", "")),
                str(model.get("id", "")),
                str(dataset.get("dim", "")),
                str(n_cases),
                _safe_metric(metrics.get("norm_hamming"), ".4f"),
                _safe_metric(metrics.get("exact_pct"), ".2f"),
                f"{cost:.4f}",
                "-" if cost_per_case is None else f"{cost_per_case:.4f}",
                status,
            ]
        )

    first = latest[0]
    invocation_id = str(first.get("invocation_id", "")).strip() or "legacy"
    started_utc = str(first.get("invocation_started_utc", "")).strip() or _invocation_sort_key(first)
    finished_utc = max(finished_times) if finished_times else "-"
    summary_items = [
        ("invocation_id", invocation_id),
        ("started_utc", started_utc),
        ("finished_utc", finished_utc),
        ("runs", str(len(latest))),
        ("total_cases", str(total_cases)),
        ("total_cost_usd", f"{total_cost:.4f}"),
    ]
    if spend_cap is not None:
        summary_items.append(("spend_cap_usd", f"{spend_cap:.2f}"))

    return "\n".join(
        [
            "Latest invocation",
            "",
            _render_kv_block(summary_items),
            "",
            _format_table(
                [
                    "dataset",
                    "model",
                    "dim",
                    "cases",
                    "norm_hamming",
                    "exact_pct",
                    "cost_usd",
                    "cost_per_case",
                    "status",
                ],
                table_rows,
            ),
        ]
    )


def render_report(
    rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]] | None = None,
    *,
    latest_only: bool = False,
) -> str:
    sections: list[str] = ["CABench Results Report"]
    aggregate_rows = rows

    if metadata_rows:
        sections.extend(["", render_latest_invocation(metadata_rows)])
        if latest_only:
            return "\n".join(sections)
        metadata_metric_rows = metric_rows_from_metadata(metadata_rows)
        if metadata_metric_rows:
            aggregate_rows = metadata_metric_rows

    if not aggregate_rows:
        if len(sections) == 1:
            sections.extend(["", "No scored runs available."])
        return "\n".join(sections)

    by_model = aggregate_by_model(aggregate_rows)
    by_md = aggregate_by_model_dataset(aggregate_rows)
    best_model = by_model[0]
    cheapest_model = min(
        by_model,
        key=lambda r: (
            float("inf") if r["cost_per_case"] is None else r["cost_per_case"],
            -r["avg_norm_hamming"],
        ),
    )

    sections.extend(
        [
            "",
            "Topline Summary",
            _render_kv_block(
                [
                    ("scored_runs", str(len(aggregate_rows))),
                    ("models_compared", str(len(by_model))),
                    ("datasets_covered", str(len({r['dataset'] for r in aggregate_rows}))),
                    (
                        "best_norm_hamming",
                        f"{best_model['model']} ({best_model['avg_norm_hamming']:.4f})",
                    ),
                    (
                        "lowest_cost_per_case",
                        (
                            f"{cheapest_model['model']} ({cheapest_model['cost_per_case']:.4f})"
                            if cheapest_model["cost_per_case"] is not None
                            else "n/a"
                        ),
                    ),
                ]
            ),
        ]
    )

    sections.extend(
        [
            "",
            "Headline Metrics By Model",
            _format_table(
                ["model", "runs", "avg_norm_hamming", "avg_exact_pct", "total_cost_usd", "cost_per_case"],
                [
                    [
                        r["model"],
                        str(r["runs"]),
                        f"{r['avg_norm_hamming']:.4f}",
                        f"{r['avg_exact_pct']:.2f}",
                        f"{r['total_cost_usd']:.4f}",
                        "-" if r["cost_per_case"] is None else f"{r['cost_per_case']:.4f}",
                    ]
                    for r in by_model
                ],
            ),
            "",
            "Headline Metrics By Dataset",
            _format_table(
                ["dataset", "model", "runs", "avg_norm_hamming", "avg_exact_pct", "total_cost_usd", "cost_per_case"],
                [
                    [
                        r["dataset"],
                        r["model"],
                        str(r["runs"]),
                        f"{r['avg_norm_hamming']:.4f}",
                        f"{r['avg_exact_pct']:.2f}",
                        f"{r['total_cost_usd']:.4f}",
                        "-" if r["cost_per_case"] is None else f"{r['cost_per_case']:.4f}",
                    ]
                    for r in by_md
                ],
            ),
        ]
    )
    return "\n".join(sections)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate human-readable CABench reports.")
    p.add_argument("--scores", type=Path, default=Path("results/scores.csv"))
    p.add_argument("--metadata", type=Path, default=Path("results/run_metadata.jsonl"))
    p.add_argument("--out", type=Path, help="Optional path to write report text")
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="Render only the latest orchestrator invocation summary.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = load_scores_if_present(args.scores) if args.latest_only else load_scores(args.scores)
    metadata_rows = load_run_metadata(args.metadata)
    report = render_report(rows, metadata_rows, latest_only=args.latest_only)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
