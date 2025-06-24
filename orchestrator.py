#!/usr/bin/env python
"""
orchestrator.py
---------------
High-level driver for CA-Bench:
-reads bench.yaml
-for each <dataset,model> pair:
    calls run_llm_json to get structured JSON replies
    converts JSONL → plain-text preds
    evaluates with eval.py
    appends a row to results/scores.csv (including norm_hamming, exact_pct, cost)
    aggregates per-call usage into logs/master_usage.csv
-enforces a hard $5.00 USD ceiling on total projected cost
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
    """Run a subprocess, echo it, and abort if it fails."""
    print(" ".join(cmd))
    res = subprocess.run(cmd, text=True)
    if res.returncode:
        sys.exit(res.returncode)


def projected_cost(n_prompts: int, price_per_1k: float, avg_tok: int = 120) -> float:
    """Estimate cost = (#prompts x avg_tok) / 1000 x price_per_1k."""
    return n_prompts * avg_tok / 1000 * price_per_1k

# Main orchestration
def run(cfg_path: Path) -> None:
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
            gold_path = Path(ds["path"])
            if not gold_path.exists():
                print(f"‼Dataset not found: {gold_path}", file=sys.stderr)
                continue

            n_cases = sum(1 for _ in gold_path.open())
            print(f"📄 Dataset {ds['name']} — {n_cases} cases")

            for mdl in spec["models"]:
                model_id     = mdl["id"]
                price_per_1k = mdl["price_per_1k_tokens"]
                max_calls    = mdl["max_calls_per_min"]

                # cost guard (projected)
                est_usd = projected_cost(n_cases, price_per_1k)
                projected_total += est_usd
                if projected_total > HARD_SPEND_CEILING:
                    print(
                        f"Aborting: projected total "
                        f"${projected_total:.2f} exceeds ceiling "
                        f"${HARD_SPEND_CEILING:.2f}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                print(f"→ {model_id}: projected cost ${est_usd:.2f} (@{price_per_1k}/1K)")

                # 1) Call LLM batch (structured JSON)
                jsonl_preds = DATA_DIR / f"{model_id}_{ds['name']}.jsonl"
                usage_csv   = LOG_DIR / f"{model_id}_{ds['name']}_usage.csv"

                shell([
                    sys.executable, "-m", "wrappers.run_llm_json",
                    "--model",       model_id,
                    "--input",       str(gold_path),
                    "--output",      str(jsonl_preds),
                    "--usage",       str(usage_csv),
                    "--tpm",         str(max_calls),
                ])

                # 2) Convert JSONL > plain-text preds
                preds_txt = DATA_DIR / f"{model_id}_{ds['name']}.preds"
                shell([
                    sys.executable, "convert_preds.py",
                    "--input",  str(jsonl_preds),
                    "--output", str(preds_txt),
                ])

                # 3) Evaluate
                result = subprocess.run([
                    sys.executable, "eval.py",
                    "--gold", str(gold_path),
                    "--pred", str(preds_txt),
                ], capture_output=True, text=True)

                if result.returncode:
                    print(result.stderr, file=sys.stderr)
                    sys.exit(result.returncode)

                # parse metrics
                try:
                    norm_line  = next(l for l in result.stdout.splitlines() if "Normalized" in l)
                    exact_line = next(l for l in result.stdout.splitlines() if "Exact-match" in l)
                except StopIteration:
                    print("eval.py output format unexpected", file=sys.stderr)
                    print(result.stdout, file=sys.stderr)
                    sys.exit(1)

                norm_acc  = float(norm_line.split()[-1])
                exact_pct = float(exact_line.split("=")[-1].split("%")[0].strip())

                # 4) Sum real cost from per-call CSV
                with usage_csv.open() as f_usage:
                    rows = list(csv.reader(f_usage))
                    cost_idx = rows[0].index("usd")  # header may vary
                    total_cost_usd = sum(float(r[cost_idx]) for r in rows[1:])

                # 5) Append run‐level row
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
                    f"{model_id} on {ds['name']} — "
                    f"norm {norm_acc:.4f}, exact {exact_pct:.2f}%, "
                    f"cost ${total_cost_usd:.2f}"
                )

                # 6) Append per-call usage to master ledger
                master = LOG_DIR / "master_usage.csv"
                append_header = not master.exists()
                with usage_csv.open() as src, master.open("a", newline="") as dst:
                    rdr = csv.reader(src)
                    wtr = csv.writer(dst)
                    header = next(rdr)
                    if append_header:
                        wtr.writerow(header)
                    for row in rdr:
                        wtr.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CA-Bench orchestrator")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "bench.yaml",
        help="Path to bench.yaml",
    )
    args = parser.parse_args()
    run(args.config)
