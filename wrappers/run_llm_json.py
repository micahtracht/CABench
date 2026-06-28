#!/usr/bin/env python
"""
run_llm_json.py
---------------
Batch-query a single model on a JSONL dataset and record structured (JSON-mode)
responses plus per-call usage.
"""

from __future__ import annotations
import argparse, csv, json, os, pathlib, sys, time
from typing import Dict, List

from openai import OpenAI, RateLimitError, APIError
from cabench.env import load_project_env
from .rate_limit import wait_one_second, set_tpm
from .response_logger import log_response

from generate import ECAProblemGenerator, Problem1D
from rules import Rule1D
from json_extract import extract_first_json_object
from contracts import (
    PRED_JSONL_SCHEMA_NAME,
    PRED_JSONL_SCHEMA_VERSION,
    USAGE_COLUMNS,
    USAGE_SCHEMA_NAME,
    USAGE_SCHEMA_VERSION,
    ensure_csv_header,
    write_schema_manifest,
)

load_project_env()

# constants & client
DEFAULT_PRICE_PER_1K = os.getenv("CABENCH_PRICE_PER_1K")
_hard_cap_env = os.getenv("CABENCH_HARD_CAP", "1.0")
try:
    DEFAULT_HARD_CAP_USD = float(_hard_cap_env)
except ValueError:
    DEFAULT_HARD_CAP_USD = 5.0

client = None

# generator instance just for its prompt-builder method
gen = ECAProblemGenerator(state_size=0)  # size unused for prompt building

# default system prompt enforcing structured JSON
DEFAULT_SYSTEM_PROMPT = (
    "Return only a valid JSON object with exactly one key: "
    "'answer'. The value must be a binary array (for 1-D) or nested binary arrays "
    "(for 2-D). No markdown, no prose, no code fences."
)

MAX_SAMPLE_ATTEMPTS = 5
MAX_API_RETRY_SLEEP = 8.0


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
        parsed = extract_first_json_object(raw)
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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--usage",  type=pathlib.Path, required=True)
    ap.add_argument("--price-per-1k", type=float, default=None,
                    help="USD price per 1K tokens (overrides CABENCH_PRICE_PER_1K)")
    ap.add_argument("--hard-cap", type=float, default=None,
                    help="Abort if spend exceeds this USD (overrides CABENCH_HARD_CAP)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--tpm", type=int, default=60, help="max calls / minute")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--raw-log", type=pathlib.Path,
                    help="File to append raw model responses")
    args = ap.parse_args()

    # adjust internal rate limiter
    set_tpm(args.tpm)

    # only initialize the client if we're doing a real run
    global client
    if not args.dry_run:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if client.api_key is None:
            sys.exit("OPENAI_API_KEY env var not set")

    total_records = sum(1 for _ in args.input.open())

    # resolve pricing and cap
    price_per_1k = args.price_per_1k
    if price_per_1k is None and DEFAULT_PRICE_PER_1K is not None:
        try:
            price_per_1k = float(DEFAULT_PRICE_PER_1K)
        except ValueError:
            price_per_1k = None

    hard_cap = args.hard_cap if args.hard_cap is not None else DEFAULT_HARD_CAP_USD
    if hard_cap is not None and hard_cap <= 0:
        hard_cap = None

    if not args.dry_run and price_per_1k is None:
        print("[warning] price_per_1k not set; usage USD will be reported as 0.0", file=sys.stderr)

    write_schema_manifest(
        args.output,
        schema_name=PRED_JSONL_SCHEMA_NAME,
        schema_version=PRED_JSONL_SCHEMA_VERSION,
        fmt="jsonl",
        notes=(
            "One JSON object per sample. Expected answer payload in "
            "'answer' or 'final_state'."
        ),
    )
    write_schema_manifest(
        args.usage,
        schema_name=USAGE_SCHEMA_NAME,
        schema_version=USAGE_SCHEMA_VERSION,
        fmt="csv",
        columns=USAGE_COLUMNS,
        notes="Per-call token/cost usage for this wrapper invocation.",
    )
    try:
        ensure_csv_header(args.usage, USAGE_COLUMNS)
    except ValueError as exc:
        sys.exit(str(exc))

    done = _reconcile_resume_state(args.output, args.usage, total_records)
    if done:
        print(f"Resuming — {done} predictions already exist")

    out_jsonl = args.output.open("a", encoding="utf-8")
    raw_log_path = args.raw_log if args.raw_log else args.output.with_suffix(".raw")
    raw_log = raw_log_path.open("a", encoding="utf-8")
    usage_csv = args.usage.open("a", newline="")
    writer = csv.writer(usage_csv)

    running_cost = 0.0
    status_counts = {"success": 0, "invalid": 0, "error": 0, "truncated": 0}
    with args.input.open() as fp_in:
        for idx, line in enumerate(fp_in, 1):
            if idx <= done:
                continue

            rec: Dict = json.loads(line.strip())

            # ----- reconstruct Problem1D --------------------------- #
            init = [int(ch) for ch in rec["init"]]  # assumes "init" is "0101…" string
            rule_bits = rec["rule"] if "rule" in rec else rec["rule_bits"]
            rule_obj = Rule1D(rule_bits)
            prob = Problem1D(start_state=init, rule=rule_obj, timesteps=rec["timesteps"])

            prompt = gen.generate_prompt_1D(prob)

            # -------- dry-run branch --------------------------------
            if args.dry_run:
                out_jsonl.write(json.dumps({"answer": [], "status": "dry_run"}) + "\n")
                continue

            # -------- live call -------------------------------------
            data, usage, raw_txt = chat_json(
                args.model, prompt, args.system, args.temperature
            )
            status = data.get("status", "error")
            if status in status_counts:
                status_counts[status] += 1

            usd = 0.0 if price_per_1k is None else (usage["total"] / 1000 * price_per_1k)
            running_cost += usd
            if hard_cap is not None and running_cost > hard_cap:
                sys.exit(f"cost {running_cost:.2f} > hard cap ${hard_cap}")

            out_jsonl.write(json.dumps(data, separators=(",", ":")) + "\n")
            raw_log.write(raw_txt + "\n")
            raw_log.flush()
            writer.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                args.model,
                usage["prompt"],
                usage["completion"],
                usage["total"],
                f"{usd:.5f}",
            ])
            out_jsonl.flush()
            usage_csv.flush()
            extra_wait = max(0, 60 / args.tpm - 1)
            if extra_wait:
                time.sleep(extra_wait)

    print(
        f"Finished {idx-done} calls; total spent ≈ ${running_cost:.2f}. "
        f"status: success={status_counts['success']}, truncated={status_counts['truncated']}, "
        f"invalid={status_counts['invalid']}, error={status_counts['error']}. "
        f"Preds → {args.output}"
    )

if __name__ == "__main__":
    main()
