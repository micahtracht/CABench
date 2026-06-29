#!/usr/bin/env python
"""
cabench.llm.runner
------------------
Batch-query a single model on a 1-D or 2-D CA JSONL dataset and record
structured (JSON-mode) responses plus per-call usage. A single ``run_batch``
handles both dimensionalities; the only per-dimension difference is how a
dataset record is turned into a prompt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from typing import Callable, Dict, List

from openai import OpenAI, RateLimitError, APIError

from cabench.env import load_project_env
from .rate_limit import wait_one_second, set_tpm
from .response_logger import log_response

from cabench.generate import (
    CAProblemGenerator2D,
    ECAProblemGenerator,
    Problem1D,
    Problem2D,
)
from cabench.rules import Rule1D, Rule2D
from cabench.json_extract import extract_answer_json
from cabench.contracts import (
    PRED_JSONL_SCHEMA_NAME,
    PRED_JSONL_SCHEMA_VERSION,
    USAGE_COLUMNS,
    USAGE_SCHEMA_NAME,
    USAGE_SCHEMA_VERSION,
    ensure_csv_header,
    write_schema_manifest,
)

load_project_env()

DEFAULT_PRICE_PER_1K = os.getenv("CABENCH_PRICE_PER_1K")
_hard_cap_env = os.getenv("CABENCH_HARD_CAP", "1.0")
try:
    DEFAULT_HARD_CAP_USD = float(_hard_cap_env)
except ValueError:
    DEFAULT_HARD_CAP_USD = 1.0

# Module-level OpenAI client; set lazily by run_batch for live runs and
# monkeypatched directly in tests.
client = None

DEFAULT_SYSTEM_PROMPT = (
    "Return only a valid JSON object with exactly one key: "
    "'answer'. The value must be a binary array (for 1-D) or nested binary arrays "
    "(for 2-D). No markdown, no prose, no code fences."
)

MAX_SAMPLE_ATTEMPTS = 5
MAX_API_RETRY_SLEEP = 8.0


# --------------------------------------------------------------------------- #
# Per-dimension prompt construction
# --------------------------------------------------------------------------- #
def _problem_1d(record: Dict) -> Problem1D:
    init = [int(ch) for ch in record["init"]]
    rule_bits = record["rule"] if "rule" in record else record["rule_bits"]
    return Problem1D(start_state=init, rule=Rule1D(rule_bits), timesteps=record["timesteps"])


def _problem_2d(record: Dict) -> Problem2D:
    return Problem2D(
        start_grid=record["init"],
        rule=Rule2D(record["rule"]),
        timesteps=record["timesteps"],
    )


def _prompt_builder(dim: int) -> Callable[[Dict], str]:
    """Return a function mapping a dataset record to a prompt string."""
    if dim == 1:
        gen = ECAProblemGenerator(state_size=0)  # size unused for prompt building
        return lambda rec: gen.generate_prompt_1D(_problem_1d(rec))
    if dim == 2:
        gen = CAProblemGenerator2D(height=1, width=1)  # dims unused for prompt building
        return lambda rec: gen.generate_prompt_2D(_problem_2d(rec))
    raise ValueError(f"unsupported dimension: {dim}")


# --------------------------------------------------------------------------- #
# Resume reconciliation
# --------------------------------------------------------------------------- #
def _write_jsonl_lines(path: pathlib.Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for line in lines:
            fp.write(line + "\n")


def _write_usage_rows(path: pathlib.Path, rows: List[List[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows(rows)


def _reconcile_resume_state(
    output_path: pathlib.Path,
    usage_path: pathlib.Path,
    total_records: int,
) -> int:
    """
    Align prediction and usage files to a consistent prefix for safe resume.
    Keeps only valid rows and truncates both files to min(valid_pred, valid_usage).
    """
    valid_preds: List[str] = []
    pred_total = 0
    pred_corrupt = False
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fp:
            for raw in fp:
                pred_total += 1
                line = raw.strip()
                if not line:
                    pred_corrupt = True
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pred_corrupt = True
                    break
                valid_preds.append(line)

    if len(valid_preds) > total_records:
        valid_preds = valid_preds[:total_records]
        pred_corrupt = True

    with usage_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))

    header = USAGE_COLUMNS
    valid_usage_rows: List[List[str]] = [header]
    usage_corrupt = False
    usage_data = rows[1:] if rows else []
    for row in usage_data:
        if len(row) != len(USAGE_COLUMNS):
            usage_corrupt = True
            break
        try:
            int(row[2])
            int(row[3])
            int(row[4])
            float(row[5])
        except ValueError:
            usage_corrupt = True
            break
        valid_usage_rows.append(row)

    if (len(valid_usage_rows) - 1) > total_records:
        valid_usage_rows = [header] + valid_usage_rows[1 : total_records + 1]
        usage_corrupt = True

    pred_done = len(valid_preds)
    usage_done = len(valid_usage_rows) - 1
    done = min(pred_done, usage_done)

    if pred_done != done or pred_corrupt or pred_total != pred_done:
        _write_jsonl_lines(output_path, valid_preds[:done])
        print(
            f"[resume] repaired predictions file to {done} consistent rows: {output_path}",
            file=sys.stderr,
        )

    if usage_done != done or usage_corrupt:
        _write_usage_rows(usage_path, [header] + valid_usage_rows[1 : done + 1])
        print(
            f"[resume] repaired usage file to {done} consistent rows: {usage_path}",
            file=sys.stderr,
        )

    return done


# --------------------------------------------------------------------------- #
# Single sample
# --------------------------------------------------------------------------- #
def chat_json(
    model: str,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
):
    """One sample with retries -> (record_dict, token_usage, raw_text)."""
    REQUIRED_KEYS = ("answer", "final_state")
    total_usage = {"prompt": 0, "completion": 0, "total": 0}
    raw = ""
    last_error = ""
    error_type = ""

    for attempt in range(1, MAX_SAMPLE_ATTEMPTS + 1):
        wait_one_second()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},  # JSON mode
                temperature=temperature,
                max_tokens=1000,
            )
        except (RateLimitError, APIError) as exc:
            last_error = str(exc)
            error_type = type(exc).__name__
            if attempt < MAX_SAMPLE_ATTEMPTS:
                sleep_s = min(2 ** (attempt - 1), MAX_API_RETRY_SLEEP)
                print(
                    f"[warning] API error ({error_type}) on attempt {attempt}/{MAX_SAMPLE_ATTEMPTS}; "
                    f"retrying in {sleep_s:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
                continue
            return (
                {
                    "answer": [],
                    "status": "error",
                    "error_type": error_type,
                    "error": last_error,
                    "attempts": attempt,
                },
                total_usage,
                raw,
            )
        except Exception as exc:  # keep run alive on unexpected SDK/runtime errors
            return (
                {
                    "answer": [],
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "attempts": attempt,
                },
                total_usage,
                raw,
            )

        raw = resp.choices[0].message.content or ""
        log_response(model, raw)
        u = resp.usage
        total_usage["prompt"] += u.prompt_tokens
        total_usage["completion"] += u.completion_tokens
        total_usage["total"] += u.total_tokens

        finish_reason = resp.choices[0].finish_reason
        parsed = extract_answer_json(raw)
        if parsed is None or not any(k in parsed for k in REQUIRED_KEYS):
            last_error = "failed to parse JSON or missing answer/final_state key"
            if attempt < MAX_SAMPLE_ATTEMPTS:
                print(
                    f"[warning] parse failure on attempt {attempt}/{MAX_SAMPLE_ATTEMPTS}; retrying",
                    file=sys.stderr,
                )
                continue
            return (
                {
                    "answer": [],
                    "status": "invalid",
                    "error_type": "ParseError",
                    "error": last_error,
                    "attempts": attempt,
                },
                total_usage,
                raw,
            )
        if "answer" not in parsed and "final_state" in parsed:
            parsed["answer"] = parsed["final_state"]

        parsed["status"] = "truncated" if finish_reason == "length" else "success"
        parsed["attempts"] = attempt
        if finish_reason:
            parsed["finish_reason"] = finish_reason
        return parsed, total_usage, raw

    # Defensive fallback; loop always returns.
    return (
        {
            "answer": [],
            "status": "error",
            "error_type": "UnknownError",
            "error": "exhausted attempts without result",
            "attempts": MAX_SAMPLE_ATTEMPTS,
        },
        total_usage,
        raw,
    )


# --------------------------------------------------------------------------- #
# Batch run
# --------------------------------------------------------------------------- #
def run_batch(
    *,
    model: str,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    usage_path: pathlib.Path,
    dim: int = 1,
    price_per_1k: float | None = None,
    hard_cap: float | None = None,
    temperature: float = 0.0,
    system: str = DEFAULT_SYSTEM_PROMPT,
    tpm: int = 60,
    dry_run: bool = False,
    raw_log: pathlib.Path | None = None,
) -> Dict[str, int]:
    """
    Query ``model`` over every record in ``input_path`` and append structured
    predictions to ``output_path`` and per-call usage to ``usage_path``.
    Returns the status counts for the calls performed.
    """
    global client

    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    usage_path = pathlib.Path(usage_path)

    set_tpm(tpm)

    if not dry_run and client is None:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if client.api_key is None:
            sys.exit("OPENAI_API_KEY env var not set")

    total_records = sum(1 for _ in input_path.open())

    if price_per_1k is None and DEFAULT_PRICE_PER_1K is not None:
        try:
            price_per_1k = float(DEFAULT_PRICE_PER_1K)
        except ValueError:
            price_per_1k = None

    if hard_cap is None:
        hard_cap = DEFAULT_HARD_CAP_USD
    if hard_cap is not None and hard_cap <= 0:
        hard_cap = None

    if not dry_run and price_per_1k is None:
        print("[warning] price_per_1k not set; usage USD will be reported as 0.0", file=sys.stderr)

    write_schema_manifest(
        output_path,
        schema_name=PRED_JSONL_SCHEMA_NAME,
        schema_version=PRED_JSONL_SCHEMA_VERSION,
        fmt="jsonl",
        notes=(
            "One JSON object per sample. Expected answer payload in "
            "'answer' or 'final_state'."
        ),
    )
    write_schema_manifest(
        usage_path,
        schema_name=USAGE_SCHEMA_NAME,
        schema_version=USAGE_SCHEMA_VERSION,
        fmt="csv",
        columns=USAGE_COLUMNS,
        notes="Per-call token/cost usage for this wrapper invocation.",
    )
    try:
        ensure_csv_header(usage_path, USAGE_COLUMNS)
    except ValueError as exc:
        sys.exit(str(exc))

    done = _reconcile_resume_state(output_path, usage_path, total_records)
    if done:
        print(f"Resuming — {done} predictions already exist")

    build_prompt = _prompt_builder(dim)

    out_jsonl = output_path.open("a", encoding="utf-8")
    raw_log_path = raw_log if raw_log else output_path.with_suffix(".raw")
    raw_log_fp = raw_log_path.open("a", encoding="utf-8")
    usage_fp = usage_path.open("a", newline="")
    writer = csv.writer(usage_fp)

    running_cost = 0.0
    status_counts = {"success": 0, "invalid": 0, "error": 0, "truncated": 0}
    idx = done
    with input_path.open() as fp_in:
        for idx, line in enumerate(fp_in, 1):
            if idx <= done:
                continue

            rec: Dict = json.loads(line.strip())
            prompt = build_prompt(rec)

            if dry_run:
                out_jsonl.write(json.dumps({"answer": [], "status": "dry_run"}) + "\n")
                continue

            data, usage, raw_txt = chat_json(model, prompt, system, temperature)
            status = data.get("status", "error")
            if status in status_counts:
                status_counts[status] += 1

            usd = 0.0 if price_per_1k is None else (usage["total"] / 1000 * price_per_1k)
            running_cost += usd
            if hard_cap is not None and running_cost > hard_cap:
                sys.exit(f"cost {running_cost:.2f} > hard cap ${hard_cap}")

            out_jsonl.write(json.dumps(data, separators=(",", ":")) + "\n")
            raw_log_fp.write(raw_txt + "\n")
            raw_log_fp.flush()
            writer.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                model,
                usage["prompt"],
                usage["completion"],
                usage["total"],
                f"{usd:.5f}",
            ])
            out_jsonl.flush()
            usage_fp.flush()
            extra_wait = max(0, 60 / tpm - 1)
            if extra_wait:
                time.sleep(extra_wait)

    out_jsonl.close()
    raw_log_fp.close()
    usage_fp.close()

    print(
        f"Finished {idx - done} calls; total spent ≈ ${running_cost:.2f}. "
        f"status: success={status_counts['success']}, truncated={status_counts['truncated']}, "
        f"invalid={status_counts['invalid']}, error={status_counts['error']}. "
        f"Preds → {output_path}"
    )
    return status_counts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch-query a model on a CA JSONL dataset.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--usage", type=pathlib.Path, required=True)
    ap.add_argument("--dim", type=int, choices=[1, 2], default=1)
    ap.add_argument("--price-per-1k", type=float, default=None,
                    help="USD price per 1K tokens (overrides CABENCH_PRICE_PER_1K)")
    ap.add_argument("--hard-cap", type=float, default=None,
                    help="Abort if spend exceeds this USD (overrides CABENCH_HARD_CAP)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--tpm", type=int, default=60, help="max calls / minute")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--raw-log", type=pathlib.Path, help="File to append raw model responses")
    return ap


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_batch(
        model=args.model,
        input_path=args.input,
        output_path=args.output,
        usage_path=args.usage,
        dim=args.dim,
        price_per_1k=args.price_per_1k,
        hard_cap=args.hard_cap,
        temperature=args.temperature,
        system=args.system,
        tpm=args.tpm,
        dry_run=args.dry_run,
        raw_log=args.raw_log,
    )


if __name__ == "__main__":
    main()
