#!/usr/bin/env python
"""
orchestrator.py
---------------
Run CA-Bench end-to-end:

1. For each dataset/model in bench.yaml:
     - (optional) generate the dataset if a "gen" block is present
     - invoke run_llm_json.py to get structured JSONL predictions
     - convert JSONL → plain-text preds
     - evaluate with eval.py
     - append metrics & cost to results/scores.csv
     - merge per-call usage into logs/master_usage.csv
2. Enforce a hard $5.00 USD ceiling on projected spend
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import yaml

# Paths & Globals
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
LOG_DIR    = ROOT / "logs"
RESULT_DIR = ROOT / "results"

for d in (DATA_DIR, LOG_DIR, RESULT_DIR):
    d.mkdir(exist_ok=True)

HARD_SPEND_CEILING = 5.00  # USD

# Helpers
def shell(cmd: List[str]) -> None:
    """Run a subprocess and exit if it fails."""
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode:
        sys.exit(result.returncode)


def projected_cost(n_prompts: int, price_per_1k: float, avg_tok: int = 120) -> float:
    """Estimate cost in USD."""
    return n_prompts * avg_tok / 1000 * price_per_1k


# Main orchestration
def run(cfg_path: Path, dry_run: bool = False) -> None:
    spec = yaml.safe_load(cfg_path.read_text())

    scores_path     = RESULT_DIR / "scores.csv"
    first_write     = not scores_path.exists()
    projected_total = 0.0

    with scores_path.open("a", newline="") as fp_scores:
        writer = csv.writer(fp_scores)
        if first_write:
            writer.writerow([
                "date_utc",
                "dataset",
                "model",
                "norm_hamming",
                "exact_pct",
                "total_cost_usd",
                "pred_file",
            ])

        for ds in spec["datasets"]:
            # Optional: auto-generate dataset if a "gen" block is present
            if "gen" in ds:
                gen_cfg = ds["gen"].copy()
                mode    = gen_cfg.pop("mode", "1d")
                outfile = Path(ds["path"])
                if not outfile.exists():
                    from generate_dataset import dispatch_main as gen_cli
                    argv = ["--mode", mode, "--outfile", str(outfile)]
                    for k, v in gen_cfg.items():
                        argv += [f"--{k}", str(v)]
                    print(f"Generating dataset {ds['name']} → {outfile}")
                    gen_cli(argv)
                else:
                    print(f"Dataset {ds['name']} already exists; skipping generation")

            gold_path = Path(ds["path"])
            if not gold_path.exists():
                print("Dataset not found:", gold_path, file=sys.stderr)
                continue

            n_cases = sum(1 for _ in gold_path.open())
            print(f"Dataset {ds['name']} — {n_cases} cases")

            for mdl in spec["models"]:
                model_id     = mdl["id"]
                price_per_1k = mdl["price_per_1k_tokens"]
                max_calls    = mdl["max_calls_per_min"]

                # projected cost guard
                if not dry_run:
                    est_usd = projected_cost(n_cases, price_per_1k)
                    projected_total += est_usd
                    if projected_total > HARD_SPEND_CEILING:
                        sys.exit(
                            f"Projected total (${projected_total:.2f}) exceeds "
                            f"hard ceiling ${HARD_SPEND_CEILING:.2f}. Aborting."
                        )
                    print(
                        f"Model {model_id}: projected cost ${est_usd:.2f} "
                        f"({price_per_1k}/1K tokens)"
                    )
                else:
                    est_usd = 0.0
                    print(f"Model {model_id}: dry run — skipping API calls")

                # pick the correct runner for this dataset's dimensionality
                dim = ds.get("dim", 1)
                if dim == 1:
                    runner_mod = "wrappers.run_llm_json"
                elif dim == 2:
                    runner_mod = "wrappers.run_llm_json_2d"
                else:
                    print(f"Unsupported dimension {dim} in dataset '{ds['name']}'", file=sys.stderr)
                    continue

                # 1) call LLM for structured JSONL predictions
                jsonl_preds = DATA_DIR / f"{model_id}_{ds['name']}.jsonl"
                usage_csv   = LOG_DIR   / f"{model_id}_{ds['name']}_usage.csv"

                cmd = [
                    sys.executable, "-m", runner_mod,
                    "--model", model_id,
                    "--input", str(gold_path),
                    "--output", str(jsonl_preds),
                    "--usage", str(usage_csv),
                    "--tpm", str(max_calls),
                ]
                if dry_run:
                    cmd.append("--dry-run")
                shell(cmd)

                # 2) convert structured JSONL -> plain-text preds
                preds_txt = DATA_DIR / f"{model_id}_{ds['name']}.preds"
                shell([
                    sys.executable, "convert_predictions.py",
                    "--input",  str(jsonl_preds),
                    "--output", str(preds_txt),
                ])

                # 3) evaluate
                eval_proc = subprocess.run(
                    [
                        sys.executable,
                        "eval.py",
                        "--gold", str(gold_path),
                        "--pred", str(preds_txt),
                    ],
                    capture_output=True,
                    text=True,
                )
                if eval_proc.returncode:
                    print(eval_proc.stderr, file=sys.stderr)
                    sys.exit(eval_proc.returncode)

                # parse metrics
                try:
                    norm_line  = next(l for l in eval_proc.stdout.splitlines() if "Normalized" in l)
                    exact_line = next(l for l in eval_proc.stdout.splitlines() if "Exact-match" in l)
                except StopIteration:
                    print("Unexpected eval.py output:", file=sys.stderr)
                    print(eval_proc.stdout, file=sys.stderr)
                    sys.exit(1)

                norm_acc  = float(norm_line.split()[-1])
                exact_pct = float(exact_line.split("=")[-1].split("%")[0].strip())

                # 4) sum real cost from usage CSV (zero in dry-run)
                if not dry_run:
                    with usage_csv.open() as f_u:
                        rows     = list(csv.reader(f_u))
                        cost_idx = rows[0].index("usd")
                        total_cost_usd = sum(float(r[cost_idx]) for r in rows[1:])
                else:
                    total_cost_usd = 0.0

                # append run-level metrics
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ds["name"],
                    model_id,
                    f"{norm_acc:.4f}",
                    f"{exact_pct:.2f}",
                    f"{total_cost_usd:.4f}",
                    str(preds_txt),
                ])
                fp_scores.flush()

                print(
                    f"Finished {model_id} on {ds['name']} — "
                    f"norm {norm_acc:.4f}, exact {exact_pct:.2f}%, cost ${total_cost_usd:.2f}"
                )

                # 5) append per-call usage rows to master ledger
                master = LOG_DIR / "master_usage.csv"
                add_header = not master.exists()
                with usage_csv.open() as src, master.open("a", newline="") as dst:
                    rdr = csv.reader(src)
                    wtr = csv.writer(dst)
                    header = next(rdr)
                    if add_header:
                        wtr.writerow(header)
                    wtr.writerows(rdr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CA-Bench orchestrator")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "bench.yaml",
        help="Path to bench.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build artifacts without calling the API",
    )
    args = parser.parse_args()
    run(args.config, dry_run=args.dry_run)
