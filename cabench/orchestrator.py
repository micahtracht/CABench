#!/usr/bin/env python
"""
cabench.orchestrator
--------------------
Run CA-Bench end-to-end:

1. For each dataset/model in bench.yaml:
    - (optional) generate the dataset if a "gen" block is present
    - run the in-process LLM runner (cabench.llm.runner) for structured
      JSONL predictions
    - convert JSONL → plain-text preds (cabench.convert)
    - evaluate (cabench.scoring)
    - append metrics & cost to results/scores.csv
    - merge per-call usage into logs/master_usage.csv
2. Enforce a hard $1.00 USD ceiling on actual spend
3. In --dry-run mode, skip API calls and API-dependent postprocessing
   (conversion/evaluation), and write explicit dry-run markers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
import yaml
from cabench.contracts import (
    PRED_JSONL_SCHEMA_NAME,
    PRED_JSONL_SCHEMA_VERSION,
    RUN_METADATA_SCHEMA_NAME,
    RUN_METADATA_SCHEMA_VERSION,
    SCORES_COLUMNS,
    SCORES_SCHEMA_NAME,
    SCORES_SCHEMA_VERSION,
    USAGE_COLUMNS,
    USAGE_SCHEMA_NAME,
    USAGE_SCHEMA_VERSION,
    ensure_csv_header,
    write_schema_manifest,
)
from cabench.convert import convert_file
from cabench.scoring import score
from cabench.llm.runner import SpendCapError, run_batch


class OrchestratorError(Exception):
    """Raised for unrecoverable orchestration/config/data errors. Caught at the CLI boundary."""

# Paths & Globals
ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
LOG_DIR    = ROOT / "logs"
RESULT_DIR = ROOT / "results"

for d in (DATA_DIR, LOG_DIR, RESULT_DIR):
    d.mkdir(exist_ok=True)

HARD_SPEND_CEILING = 1.00  # USD per invocation


def read_usage_csv(path: Path) -> tuple[float, list[list[str]]]:
    """
    Return (total_usd, all_rows) from a usage CSV.
    all_rows includes the header row if present.
    """
    if not path.exists():
        return 0.0, []

    with path.open(newline="") as fp:
        rows = list(csv.reader(fp))

    if not rows:
        return 0.0, []

    header = rows[0]
    if "usd" not in header:
        raise OrchestratorError(f"Usage CSV missing 'usd' column: {path}")
    cost_idx = header.index("usd")

    total = 0.0
    for row in rows[1:]:
        if len(row) <= cost_idx:
            continue
        val = row[cost_idx].strip()
        if not val:
            continue
        try:
            total += float(val)
        except ValueError as exc:
            raise OrchestratorError(f"Invalid usd value '{val}' in {path}: {exc}") from exc

    return total, rows


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_metadata(repo_root: Path) -> dict[str, Any]:
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return {"commit": None, "dirty": None}

    commit = rev.stdout.strip() if rev.returncode == 0 else None
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    return {"commit": commit, "dirty": dirty}


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, separators=(",", ":")) + "\n")


def parse_csv_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def print_run_summary(rows: List[dict[str, str]]) -> None:
    if not rows:
        print("Run summary: no model/dataset runs were executed.")
        return

    print("\nRun summary:")
    print("dataset | model | norm_hamming | exact_pct | cost_usd | pred_file")
    print("-" * 88)
    total_cost = 0.0
    for r in rows:
        cost = float(r["cost_usd"])
        total_cost += cost
        print(
            f"{r['dataset']} | {r['model']} | {r['norm_hamming']} | "
            f"{r['exact_pct']} | {cost:.4f} | {r['pred_file']}"
        )
    print("-" * 88)
    print(f"runs={len(rows)} total_cost_usd={total_cost:.4f}")


def resolve_dataset_artifact_path(ds: dict[str, Any], data_dir: Path) -> Path:
    """
    Resolve the dataset artifact location for generated datasets under `data_dir`.
    Non-generated datasets keep their configured path.
    """
    configured = Path(ds["path"])
    if "gen" in ds:
        return data_dir / configured.name
    return configured


def _apply_cli_overrides(
    spec: dict[str, Any],
    *,
    cli_dim: Optional[int],
    dataset_names: Optional[List[str]],
    file: Optional[Path],
    num_questions: Optional[int],
) -> bool:
    """
    Mutate ``spec["datasets"]`` in place per the CLI overrides. Returns True if any
    datasets remain, or False (after printing why) if a filter emptied the list.
    """
    if cli_dim is not None:
        spec["datasets"] = [ds for ds in spec["datasets"] if ds.get("dim", 1) == cli_dim]
        if not spec["datasets"]:
            print(f"No datasets matching dimension {cli_dim}", file=sys.stderr)
            return False
    if dataset_names:
        names = set(dataset_names)
        spec["datasets"] = [ds for ds in spec["datasets"] if ds.get("name") in names]
        if not spec["datasets"]:
            print(f"No datasets matching --datasets {sorted(names)}", file=sys.stderr)
            return False

    if file is not None:
        for ds in spec["datasets"]:
            ds["path"] = str(file)
            ds["name"] = Path(file).stem

    if num_questions is not None:
        for ds in spec["datasets"]:
            if "gen" in ds:
                ds["gen"]["n"] = num_questions
    return True


def _setup_manifests(scores_path: Path, master: Path, metadata_path: Path) -> None:
    """Write schema sidecar manifests and validate existing CSV headers."""
    try:
        ensure_csv_header(scores_path, SCORES_COLUMNS)
    except ValueError as exc:
        raise OrchestratorError(str(exc)) from exc
    write_schema_manifest(
        scores_path,
        schema_name=SCORES_SCHEMA_NAME,
        schema_version=SCORES_SCHEMA_VERSION,
        fmt="csv",
        columns=SCORES_COLUMNS,
        notes="One row per model/dataset run. Cost is run-level USD delta for this invocation.",
    )
    write_schema_manifest(
        master,
        schema_name=USAGE_SCHEMA_NAME,
        schema_version=USAGE_SCHEMA_VERSION,
        fmt="csv",
        columns=USAGE_COLUMNS,
        notes="Per-call token/cost usage ledger merged across runs.",
    )
    write_schema_manifest(
        metadata_path,
        schema_name=RUN_METADATA_SCHEMA_NAME,
        schema_version=RUN_METADATA_SCHEMA_VERSION,
        fmt="jsonl",
        notes=(
            "One metadata record per model/dataset orchestrator run. "
            "Includes config snapshot, git state, dataset hash, and run artifacts."
        ),
    )
    if master.exists() and master.stat().st_size > 0:
        try:
            ensure_csv_header(master, USAGE_COLUMNS)
        except ValueError as exc:
            raise OrchestratorError(str(exc)) from exc


def _prepare_dataset(
    ds: dict[str, Any],
    *,
    data_dir: Path,
    force_new: bool,
    num_questions: Optional[int],
) -> Optional[tuple[Path, int, str, int]]:
    """
    Resolve (generating if needed) the dataset artifact and an optional truncated
    head copy. Returns (gold_path, n_cases, dataset_sha256, dataset_bytes) or None
    if the dataset file is missing.
    """
    dataset_path = resolve_dataset_artifact_path(ds, data_dir)

    if "gen" in ds:
        gen_cfg = ds["gen"].copy()
        mode = gen_cfg.pop("mode", "1d")
        if force_new or not dataset_path.exists():
            from cabench.dataset import dispatch_main as gen_cli
            argv = ["--mode", mode, "--outfile", str(dataset_path)]
            for k, v in gen_cfg.items():
                argv += [f"--{k}", str(v)]
            print(f"Generating dataset {ds['name']} → {dataset_path}")
            gen_cli(argv)
        else:
            print(f"Dataset {ds['name']} already exists; skipping generation")

    gold_path = dataset_path
    if not gold_path.exists():
        print("Dataset not found:", gold_path, file=sys.stderr)
        return None

    n_total = sum(1 for _ in gold_path.open())
    n_cases = n_total

    if num_questions is not None and num_questions < n_total:
        head_path = data_dir / f"{gold_path.stem}_head{num_questions}{gold_path.suffix}"
        if force_new or not head_path.exists() or sum(1 for _ in head_path.open()) != num_questions:
            with gold_path.open() as src, head_path.open("w", encoding="utf-8") as dst:
                for idx, line in enumerate(src):
                    if idx >= num_questions:
                        break
                    dst.write(line)
        gold_path = head_path
        n_cases = num_questions

    return gold_path, n_cases, file_sha256(gold_path), gold_path.stat().st_size


def _build_metadata_record(
    *,
    invocation_id: str,
    invocation_started: str,
    run_started: str,
    run_finished: str,
    dry_run: bool,
    ds: dict[str, Any],
    gold_path: Path,
    dataset_hash: str,
    dataset_bytes: int,
    n_cases: int,
    mdl: dict[str, Any],
    model_id_run: str,
    price_per_1k: float,
    max_calls: int,
    cfg_path: Path,
    cfg_hash: str,
    cli_overrides: dict[str, Any],
    git_meta: dict[str, Any],
    artifacts: dict[str, Any],
    metrics: dict[str, Any],
    cost_usd: float,
    status: str,
) -> dict[str, Any]:
    """Build one run-metadata record. Used for both dry-run and completed runs."""
    return {
        "schema_name": RUN_METADATA_SCHEMA_NAME,
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "invocation_id": invocation_id,
        "invocation_started_utc": invocation_started,
        "run_started_utc": run_started,
        "run_finished_utc": run_finished,
        "dry_run": dry_run,
        "dataset": {
            "name": ds["name"],
            "path": str(gold_path),
            "sha256": dataset_hash,
            "bytes": dataset_bytes,
            "num_cases": n_cases,
            "dim": ds.get("dim", 1),
        },
        "model": {
            "id": model_id_run,
            "provider": mdl.get("provider"),
            "price_per_1k_tokens": price_per_1k,
            "max_calls_per_min": max_calls,
            "temperature": mdl.get("temperature", 0),
        },
        "config": {
            "path": str(cfg_path),
            "sha256": cfg_hash,
            "dataset_spec": ds,
            "model_spec": mdl,
            "cli_overrides": cli_overrides,
        },
        "git": git_meta,
        "artifacts": artifacts,
        "metrics": metrics,
        "cost_usd": cost_usd,
        "status": status,
    }


def _merge_usage_into_master(
    master: Path,
    usage_before_rows: list[list[str]],
    usage_after_rows: list[list[str]],
    usage_csv: Path,
) -> None:
    """Append only the per-call usage rows added by this run to the master ledger."""
    if not usage_after_rows:
        return
    if usage_after_rows[0] != USAGE_COLUMNS:
        raise OrchestratorError(
            f"Unexpected usage header in {usage_csv}: {usage_after_rows[0]}"
        )
    before_n = max(0, len(usage_before_rows) - 1)
    new_rows = usage_after_rows[1 + before_n :]
    if new_rows:
        try:
            ensure_csv_header(master, USAGE_COLUMNS)
        except ValueError as exc:
            raise OrchestratorError(str(exc)) from exc
        with master.open("a", newline="") as dst:
            csv.writer(dst).writerows(new_rows)


# Main orchestration
def run(
    cfg_path: Path,
    dry_run: bool = False,
    model_id: Optional[str] = None,
    model_ids: Optional[List[str]] = None,
    dataset_names: Optional[List[str]] = None,
    dim: Optional[int] = None,
    force_new: bool = False,
    file: Optional[Path] = None,
    num_questions: Optional[int] = None,
    force_preds: bool = False,
    data_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    summarize: bool = True,
    spend_cap_usd: Optional[float] = None,
) -> None:
    data_dir = data_dir or DATA_DIR
    log_dir = log_dir or LOG_DIR
    results_dir = results_dir or RESULT_DIR
    spend_cap_usd = HARD_SPEND_CEILING if spend_cap_usd is None else spend_cap_usd
    for d in (data_dir, log_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)
    if spend_cap_usd <= 0:
        raise OrchestratorError("Spend cap must be greater than 0 USD.")

    spec = yaml.safe_load(cfg_path.read_text())
    cli_dim = dim
    run_summary_rows: List[dict[str, str]] = []
    invocation_id = uuid.uuid4().hex
    invocation_started = now_utc_iso()

    selected_models: Optional[set[str]] = None
    if model_id is not None:
        selected_models = {model_id}
    if model_ids:
        ids = set(model_ids)
        selected_models = ids if selected_models is None else (selected_models | ids)

    if not _apply_cli_overrides(
        spec,
        cli_dim=cli_dim,
        dataset_names=dataset_names,
        file=file,
        num_questions=num_questions,
    ):
        return

    cli_overrides = {
        "model_id": model_id,
        "dim": cli_dim,
        "force_new": force_new,
        "file": str(file) if file is not None else None,
        "num_questions": num_questions,
        "force_preds": force_preds,
        "spend_cap_usd": spend_cap_usd,
    }

    scores_path     = results_dir / "scores.csv"
    metadata_path   = results_dir / "run_metadata.jsonl"
    actual_spend_total = 0.0
    master = log_dir / "master_usage.csv"
    git_meta = get_git_metadata(ROOT)
    cfg_hash = file_sha256(cfg_path)

    _setup_manifests(scores_path, master, metadata_path)

    with scores_path.open("a", newline="") as fp_scores:
        writer = csv.writer(fp_scores)

        for ds in spec["datasets"]:
            prepared = _prepare_dataset(
                ds, data_dir=data_dir, force_new=force_new, num_questions=num_questions
            )
            if prepared is None:
                continue
            gold_path, n_cases, dataset_hash, dataset_bytes = prepared

            print(f"Dataset {ds['name']} — {n_cases} cases")

            for mdl in spec["models"]:
                # skip models not matching --model-id (if provided)
                if selected_models is not None and mdl["id"] not in selected_models:
                    continue

                model_id_run = mdl["id"]
                price_per_1k = mdl["price_per_1k_tokens"]
                max_calls = mdl.get("max_calls_per_min", 60)
                run_started = now_utc_iso()

                if dry_run:
                    print(f"Model {model_id_run}: dry run — skipping API calls")
                else:
                    remaining = spend_cap_usd - actual_spend_total
                    if remaining <= 0:
                        raise SpendCapError(
                            f"Actual spend cap reached (${actual_spend_total:.2f} >= "
                            f"${spend_cap_usd:.2f}). Aborting before {model_id_run}."
                        )
                    print(
                        f"Model {model_id_run}: actual spend so far ${actual_spend_total:.2f}; "
                        f"remaining budget ${remaining:.2f}"
                    )

                # pick the correct runner for this dataset's dimensionality
                ds_dim = ds.get("dim", 1)
                if ds_dim not in (1, 2):
                    print(f"Unsupported dimension {ds_dim} in dataset '{ds['name']}'", file=sys.stderr)
                    continue

                # 1) call LLM for structured JSONL predictions
                jsonl_preds = data_dir / f"{model_id_run}_{ds['name']}.jsonl"
                usage_csv   = log_dir  / f"{model_id_run}_{ds['name']}_usage.csv"
                preds_txt   = data_dir / f"{model_id_run}_{ds['name']}.preds"

                if force_preds:
                    for p in (jsonl_preds, usage_csv, preds_txt):
                        if p.exists():
                            p.unlink()

                if dry_run:
                    marker_path = data_dir / f"{model_id_run}_{ds['name']}.dryrun.json"
                    marker = {
                        "mode": "dry-run",
                        "dataset": ds["name"],
                        "model": model_id_run,
                        "gold_path": str(gold_path),
                        "n_cases": n_cases,
                        "note": "API call, conversion, and evaluation were skipped.",
                    }
                    marker_path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
                    run_finished = now_utc_iso()

                    writer.writerow([
                        run_finished,
                        ds["name"],
                        model_id_run,
                        "",
                        "",
                        "0.0000",
                        f"DRY_RUN:{marker_path}",
                    ])
                    run_summary_rows.append(
                        {
                            "dataset": ds["name"],
                            "model": model_id_run,
                            "norm_hamming": "",
                            "exact_pct": "",
                            "cost_usd": "0.0000",
                            "pred_file": f"DRY_RUN:{marker_path}",
                        }
                    )
                    fp_scores.flush()
                    print(
                        f"[DRY-RUN] {model_id_run} on {ds['name']} — "
                        "skipped API call, conversion, and evaluation"
                    )
                    append_jsonl_record(
                        metadata_path,
                        _build_metadata_record(
                            invocation_id=invocation_id,
                            invocation_started=invocation_started,
                            run_started=run_started,
                            run_finished=run_finished,
                            dry_run=True,
                            ds=ds,
                            gold_path=gold_path,
                            dataset_hash=dataset_hash,
                            dataset_bytes=dataset_bytes,
                            n_cases=n_cases,
                            mdl=mdl,
                            model_id_run=model_id_run,
                            price_per_1k=price_per_1k,
                            max_calls=max_calls,
                            cfg_path=cfg_path,
                            cfg_hash=cfg_hash,
                            cli_overrides=cli_overrides,
                            git_meta=git_meta,
                            artifacts={
                                "dry_run_marker": str(marker_path),
                                "scores_csv": str(scores_path),
                            },
                            metrics={"norm_hamming": None, "exact_pct": None},
                            cost_usd=0.0,
                            status="dry_run",
                        ),
                    )
                    continue

                write_schema_manifest(
                    jsonl_preds,
                    schema_name=PRED_JSONL_SCHEMA_NAME,
                    schema_version=PRED_JSONL_SCHEMA_VERSION,
                    fmt="jsonl",
                    notes=(
                        "One JSON object per sample from model output. "
                        "Expected answer payload in 'answer' or 'final_state'."
                    ),
                )
                write_schema_manifest(
                    usage_csv,
                    schema_name=USAGE_SCHEMA_NAME,
                    schema_version=USAGE_SCHEMA_VERSION,
                    fmt="csv",
                    columns=USAGE_COLUMNS,
                    notes="Per-call usage for this model/dataset run file.",
                )

                usage_before_usd, usage_before_rows = read_usage_csv(usage_csv)

                # 1) call LLM for structured JSONL predictions. The runner
                # enforces a per-run cap derived from the remaining global
                # budget so spending control stays centralized here.
                run_batch(
                    model=model_id_run,
                    input_path=gold_path,
                    output_path=jsonl_preds,
                    usage_path=usage_csv,
                    dim=ds_dim,
                    price_per_1k=price_per_1k,
                    hard_cap=max(0.0, spend_cap_usd - actual_spend_total),
                    tpm=max_calls,
                    temperature=0.0,
                )

                # 2) convert structured JSONL -> plain-text preds
                convert_file(jsonl_preds, preds_txt)

                # 3) evaluate (EvalError propagates to the CLI boundary)
                metrics = score(gold_path, preds_txt)

                norm_acc  = metrics["norm_hamming"]
                exact_pct = metrics["exact_pct"]

                # 4) read actual usage and compute this run's spend delta
                usage_after_usd, usage_after_rows = read_usage_csv(usage_csv)
                run_cost_usd = usage_after_usd - usage_before_usd
                if run_cost_usd < -1e-9:
                    print(
                        f"[warn] usage cost decreased for {usage_csv} "
                        f"(before={usage_before_usd:.5f}, after={usage_after_usd:.5f}); "
                        "treating run delta as $0.00 (ledger likely repaired).",
                        file=sys.stderr,
                    )
                    run_cost_usd = 0.0
                run_cost_usd = max(0.0, run_cost_usd)
                actual_spend_total += run_cost_usd
                total_cost_usd = run_cost_usd
                if actual_spend_total > spend_cap_usd + 1e-9:
                    raise SpendCapError(
                        f"Actual spend ${actual_spend_total:.2f} exceeded "
                        f"hard ceiling ${spend_cap_usd:.2f}. Aborting."
                    )

                # append run-level metrics
                run_finished = now_utc_iso()
                writer.writerow([
                    run_finished,
                    ds["name"],
                    model_id_run,
                    f"{norm_acc:.4f}",
                    f"{exact_pct:.2f}",
                    f"{total_cost_usd:.4f}",
                    str(preds_txt),
                ])
                run_summary_rows.append(
                    {
                        "dataset": ds["name"],
                        "model": model_id_run,
                        "norm_hamming": f"{norm_acc:.4f}",
                        "exact_pct": f"{exact_pct:.2f}",
                        "cost_usd": f"{total_cost_usd:.4f}",
                        "pred_file": str(preds_txt),
                    }
                )
                fp_scores.flush()

                print(
                    f"Finished {model_id_run} on {ds['name']} — "
                    f"norm {norm_acc:.4f}, exact {exact_pct:.2f}%, cost ${total_cost_usd:.2f}"
                )
                append_jsonl_record(
                    metadata_path,
                    _build_metadata_record(
                        invocation_id=invocation_id,
                        invocation_started=invocation_started,
                        run_started=run_started,
                        run_finished=run_finished,
                        dry_run=False,
                        ds=ds,
                        gold_path=gold_path,
                        dataset_hash=dataset_hash,
                        dataset_bytes=dataset_bytes,
                        n_cases=n_cases,
                        mdl=mdl,
                        model_id_run=model_id_run,
                        price_per_1k=price_per_1k,
                        max_calls=max_calls,
                        cfg_path=cfg_path,
                        cfg_hash=cfg_hash,
                        cli_overrides=cli_overrides,
                        git_meta=git_meta,
                        artifacts={
                            "gold_path": str(gold_path),
                            "jsonl_preds": str(jsonl_preds),
                            "preds_txt": str(preds_txt),
                            "usage_csv": str(usage_csv),
                            "scores_csv": str(scores_path),
                            "master_usage_csv": str(master),
                        },
                        metrics={"norm_hamming": norm_acc, "exact_pct": exact_pct},
                        cost_usd=total_cost_usd,
                        status="completed",
                    ),
                )

                # 5) append only newly-added per-call usage rows to master ledger
                _merge_usage_into_master(
                    master, usage_before_rows, usage_after_rows, usage_csv
                )

    if not dry_run:
        print(
            f"Run complete. Actual spend this invocation: "
            f"${actual_spend_total:.2f} / ${spend_cap_usd:.2f}"
        )
    if summarize:
        print_run_summary(run_summary_rows)
