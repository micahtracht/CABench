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

import argparse
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
from cabench.scoring import EvalError, score
from cabench.llm.runner import run_batch

# Paths & Globals
ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR   = ROOT / "data"
LOG_DIR    = ROOT / "logs"
RESULT_DIR = ROOT / "results"

for d in (DATA_DIR, LOG_DIR, RESULT_DIR):
    d.mkdir(exist_ok=True)

HARD_SPEND_CEILING = 1.00  # USD per invocation

# System prompt for structured responses
SYSTEM_PROMPT = (
    "Return only a valid JSON object with exactly one key: "
    "'answer'. The value must be a binary array (for 1-D) or nested binary arrays "
    "(for 2-D). No markdown, no prose, no code fences."
)


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
        sys.exit(f"Usage CSV missing 'usd' column: {path}")
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
            sys.exit(f"Invalid usd value '{val}' in {path}: {exc}")

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
        sys.exit("Spend cap must be greater than 0 USD.")

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

    # Filter datasets by dimension if requested
    if cli_dim is not None:
        spec["datasets"] = [ds for ds in spec["datasets"] if ds.get("dim", 1) == cli_dim]
        if not spec["datasets"]:
            print(f"No datasets matching dimension {cli_dim}", file=sys.stderr)
            return
    if dataset_names:
        names = set(dataset_names)
        spec["datasets"] = [ds for ds in spec["datasets"] if ds.get("name") in names]
        if not spec["datasets"]:
            print(f"No datasets matching --datasets {sorted(names)}", file=sys.stderr)
            return

    # Override dataset path if a custom file is provided
    if file is not None:
        for ds in spec["datasets"]:
            ds["path"] = str(file)
            ds["name"] = Path(file).stem

    # Override number of questions for dataset generation
    if num_questions is not None:
        for ds in spec["datasets"]:
            if "gen" in ds:
                ds["gen"]["n"] = num_questions

    scores_path     = results_dir / "scores.csv"
    metadata_path   = results_dir / "run_metadata.jsonl"
    actual_spend_total = 0.0
    master = log_dir / "master_usage.csv"
    git_meta = get_git_metadata(ROOT)
    cfg_hash = file_sha256(cfg_path)

    try:
        ensure_csv_header(scores_path, SCORES_COLUMNS)
    except ValueError as exc:
        sys.exit(str(exc))
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
            sys.exit(str(exc))

    with scores_path.open("a", newline="") as fp_scores:
        writer = csv.writer(fp_scores)

        for ds in spec["datasets"]:
            dataset_path = resolve_dataset_artifact_path(ds, data_dir)

            # Optional: auto-generate dataset if a "gen" block is present
            if "gen" in ds:
                gen_cfg = ds["gen"].copy()
                mode    = gen_cfg.pop("mode", "1d")
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
                continue

            n_total = sum(1 for _ in gold_path.open())
            n_cases = n_total

            # Create a truncated copy if num_questions is set and dataset is larger
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

            print(f"Dataset {ds['name']} — {n_cases} cases")
            dataset_hash = file_sha256(gold_path)
            dataset_bytes = gold_path.stat().st_size

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
                        sys.exit(
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
                        {
                            "schema_name": RUN_METADATA_SCHEMA_NAME,
                            "schema_version": RUN_METADATA_SCHEMA_VERSION,
                            "invocation_id": invocation_id,
                            "invocation_started_utc": invocation_started,
                            "run_started_utc": run_started,
                            "run_finished_utc": run_finished,
                            "dry_run": True,
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
                                "cli_overrides": {
                                    "model_id": model_id,
                                    "dim": cli_dim,
                                    "force_new": force_new,
                                    "file": str(file) if file is not None else None,
                                    "num_questions": num_questions,
                                    "force_preds": force_preds,
                                    "spend_cap_usd": spend_cap_usd,
                                },
                            },
                            "git": git_meta,
                            "artifacts": {
                                "dry_run_marker": str(marker_path),
                                "scores_csv": str(scores_path),
                            },
                            "metrics": {"norm_hamming": None, "exact_pct": None},
                            "cost_usd": 0.0,
                            "status": "dry_run",
                        },
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
                    system=SYSTEM_PROMPT,
                )

                # 2) convert structured JSONL -> plain-text preds
                convert_file(jsonl_preds, preds_txt)

                # 3) evaluate
                try:
                    metrics = score(gold_path, preds_txt)
                except EvalError as exc:
                    print(str(exc), file=sys.stderr)
                    sys.exit(1)

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
                    sys.exit(
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
                    {
                        "schema_name": RUN_METADATA_SCHEMA_NAME,
                        "schema_version": RUN_METADATA_SCHEMA_VERSION,
                        "invocation_id": invocation_id,
                        "invocation_started_utc": invocation_started,
                        "run_started_utc": run_started,
                        "run_finished_utc": run_finished,
                        "dry_run": False,
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
                            "cli_overrides": {
                                "model_id": model_id,
                                "dim": cli_dim,
                                "force_new": force_new,
                                "file": str(file) if file is not None else None,
                                "num_questions": num_questions,
                                "force_preds": force_preds,
                                "spend_cap_usd": spend_cap_usd,
                            },
                        },
                        "git": git_meta,
                        "artifacts": {
                            "gold_path": str(gold_path),
                            "jsonl_preds": str(jsonl_preds),
                            "preds_txt": str(preds_txt),
                            "usage_csv": str(usage_csv),
                            "scores_csv": str(scores_path),
                            "master_usage_csv": str(master),
                        },
                        "metrics": {"norm_hamming": norm_acc, "exact_pct": exact_pct},
                        "cost_usd": total_cost_usd,
                        "status": "completed",
                    },
                )

                # 5) append only newly-added per-call usage rows to master ledger
                if usage_after_rows:
                    if usage_after_rows[0] != USAGE_COLUMNS:
                        sys.exit(
                            f"Unexpected usage header in {usage_csv}: {usage_after_rows[0]}"
                        )
                    before_n = max(0, len(usage_before_rows) - 1)
                    new_rows = usage_after_rows[1 + before_n :]
                    if new_rows:
                        try:
                            ensure_csv_header(master, USAGE_COLUMNS)
                        except ValueError as exc:
                            sys.exit(str(exc))
                        with master.open("a", newline="") as dst:
                            wtr = csv.writer(dst)
                            wtr.writerows(new_rows)

    if not dry_run:
        print(
            f"Run complete. Actual spend this invocation: "
            f"${actual_spend_total:.2f} / ${spend_cap_usd:.2f}"
        )
    if summarize:
        print_run_summary(run_summary_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CA-Bench orchestrator")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "bench.yaml",
        help="Path to bench.yaml",
    )
    parser.add_argument(
        "--model-id",
        help="If set, only run this model (must match one of the IDs in bench.yaml)",
    )
    parser.add_argument(
        "--models",
        help="Comma-separated model IDs to run (e.g. gpt-5.4-mini,gpt-5.4).",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset names to run (from bench.yaml).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=[1, 2],
        help="Number of dimensions in the question (1 or 2)",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Generate new questions even if dataset file exists",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Use questions from this specific file",
    )
    parser.add_argument(
        "--numquestions",
        type=int,
        help="Number of questions to use in the benchmark",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build artifacts without calling the API",
    )
    parser.add_argument(
        "--force-preds",
        action="store_true",
        help="Overwrite prediction & usage files before model call",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory for model JSONL/.preds artifacts.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        help="Directory for usage/cost logs.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULT_DIR,
        help="Directory for score and run-metadata outputs.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable end-of-run summary table.",
    )
    parser.add_argument(
        "--spend-cap",
        type=float,
        default=HARD_SPEND_CEILING,
        help="Hard USD cap for this orchestrator invocation.",
    )
    args = parser.parse_args()
    run(
        args.config,
        dry_run=args.dry_run,
        model_id=args.model_id,
        model_ids=parse_csv_arg(args.models),
        dataset_names=parse_csv_arg(args.datasets),
        dim=args.dim,
        force_new=args.new,
        file=args.file,
        num_questions=args.numquestions,
        force_preds=args.force_preds,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        summarize=not args.no_summary,
        spend_cap_usd=args.spend_cap,
    )
