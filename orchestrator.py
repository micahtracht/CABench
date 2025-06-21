#!/usr/bin/env python
"""
orchestrator.py
---------------
Drive CA-Bench end-to-end.

 * reads bench.yaml
 * for every <dataset, model> pair:
       - calls run_llm.py   → predictions
       - calls eval.py      → metrics
       - appends row to     → results/scores.csv
 * enforces a **hard $5.00 USD spend ceiling** per invocation
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
from typing import List

import yaml

# project paths
ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "results"

for p in (DATA_DIR, LOG_DIR, RESULT_DIR):
    p.mkdir(exist_ok=True)

# helpers
HARD_SPEND_CEILING = 5.00  # USD, do NOT exceed

def shell(cmd: List[str]) -> None:
    """Run a subprocess and abort on failure."""
    print("Working: ", " ".join(cmd))
    res = subprocess.run(cmd, text=True)
    if res.returncode:
        sys.exit(res.returncode)


def projected_cost(
    n_prompts: int, price_per_1k: float, avg_tok: int = 120
) -> float:
    """Rough token-based cost estimate."""
    return n_prompts * avg_tok / 1000 * price_per_1k


# main driver
def run(cfg_path: pathlib.Path) -> None:
    spec = yaml.safe_load(cfg_path.read_text())

    scores_path = RESULT_DIR / "scores.csv"
    first_write = not scores_path.exists()

    projected_run_total = 0.0  # keep running tally

    with scores_path.open("a", newline="") as fp_csv:
        writer = csv.writer(fp_csv)
        if first_write:
            writer.writerow(
                ["date_utc", "dataset", "model", "norm_hamming", "exact_pct", "pred_file"]
            )

        # --------------------- iterate datasets ---------------------- #
        for ds in spec["datasets"]:
            gold_path = pathlib.Path(ds["path"])
            if not gold_path.exists():
                print(f"!!ALERT!! Dataset not found: {gold_path}", file=sys.stderr)
                continue

            n_cases = sum(1 for _ in gold_path.open())
            print(f"📄  Dataset {ds['name']} — {n_cases} cases")

            # ------------------- iterate models ---------------------- #
            for mdl in spec["models"]:
                model_id = mdl["id"]
                price_per_1k = mdl["price_per_1k_tokens"]

                est_usd = projected_cost(n_cases, price_per_1k)
                projected_run_total += est_usd

                # ------ global hard-cap guard ----------------------- #
                if projected_run_total > HARD_SPEND_CEILING:
                    print(
                        f"ABORTING: projected total (${projected_run_total:.2f}) "
                        f"exceeds hard ceiling ${HARD_SPEND_CEILING:.2f}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                print(
                    f"→ {model_id}: projected cost ≈ ${est_usd:.2f} "
                    f"({price_per_1k}/1K tokens)"
                )

                preds_file = DATA_DIR / f"{model_id}_{ds['name']}.preds"

                # -------- 1. run_llm.py -------- #
                shell(
                    [
                        sys.executable,
                        "run_llm.py",
                        "--model",
                        model_id,
                        "--input",
                        str(gold_path),
                        "--output",
                        str(preds_file),
                        "--temperature",
                        str(mdl.get("temperature", 0)),
                        "--sleep",
                        "0",
                    ]
                )

                # -------- 2. eval.py ----------- #
                result = subprocess.run(
                    [
                        sys.executable,
                        "eval.py",
                        "--gold",
                        str(gold_path),
                        "--pred",
                        str(preds_file),
                    ],
                    capture_output=True,
                    text=True,
                )
                if result.returncode:
                    print(result.stderr, file=sys.stderr)
                    sys.exit(result.returncode)

                try:
                    norm_line = next(
                        l for l in result.stdout.splitlines() if "Normalized" in l
                    )
                    exact_line = next(
                        l for l in result.stdout.splitlines() if "Exact" in l
                    )
                except StopIteration:
                    print("!ALERT! eval.py output format unexpected:", file=sys.stderr)
                    print(result.stdout, file=sys.stderr)
                    sys.exit(1)

                norm_acc = float(norm_line.split()[-1])
                exact_pct = float(exact_line.split("=")[-1].split("%")[0].strip())

                # -------- 3. log row ---------- #
                writer.writerow(
                    [
                        _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                        ds["name"],
                        model_id,
                        f"{norm_acc:.4f}",
                        f"{exact_pct:.2f}",
                        str(preds_file),
                    ]
                )
                fp_csv.flush()

                print(
                    f"✅  {model_id} on {ds['name']} — norm {norm_acc:.4f}, "
                    f"exact {exact_pct:.2f}%"
                )


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run CA-Bench orchestrator.")
    ap.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOT / "bench.yaml",
        help="Path to bench.yaml (default = ./bench.yaml)",
    )
    args = ap.parse_args()
    run(args.config)
