#!/usr/bin/env python
"""
wrappers/run_llm_json_2d.py
---------------------------
Batch-query a single model on a **2-D** CA JSONL dataset and store
JSON-mode replies + per-call token usage.

Usage (orchestrator invokes this):
    python -m wrappers.run_llm_json_2d \
           --model gpt-4o-mini \
           --input  data/val2d_public.jsonl \
           --output data/gpt4o_val2d_public.jsonl \
           --usage  logs/gpt4o_val2d_public_usage.csv
"""
from __future__ import annotations
import argparse, csv, json, os, pathlib, re, sys, time
from typing import Dict, List

from openai import OpenAI, RateLimitError, APIError
import backoff
from .rate_limit import wait_one_second, set_tpm
from .response_logger import log_response

from generate import CAProblemGenerator2D, Problem2D
from rules import Rule2D
from simulate import simulate_2d   # only used for optional sanity checks

PRICE_PER_1K = 0.003  # USD
HARD_CAP     = 5.00   # abort if projected spend exceeds this

client = None  # will be set in main()

# dummy generator solely for its prompt builder
_prompt_gen = CAProblemGenerator2D(height=1, width=1)

# default system prompt enforcing structured JSON
DEFAULT_SYSTEM_PROMPT = (
    "Respond with a JSON object containing keys 'initial_state' and 'final_state' "
    "(alias 'answer'). Do not include any extra text outside the JSON."
)

def extract_json_from_string(s: str) -> dict | None:
    """Return first JSON object embedded in a string (or None)."""
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_time=60)
def chat_json(
    model: str,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
):
    """One JSON-mode chat completion → (python_dict, usage_dict, raw_text)."""
    REQUIRED_KEYS = ("answer", "final_state")
    total_usage = {"prompt": 0, "completion": 0, "total": 0}
    raw = ""

    for attempt in range(5):
        wait_one_second()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=1000,
        )

        raw = resp.choices[0].message.content
        log_response(model, raw)
        u = resp.usage
        total_usage["prompt"] += u.prompt_tokens
        total_usage["completion"] += u.completion_tokens
        total_usage["total"] += u.total_tokens

        parsed = extract_json_from_string(raw)
        if parsed is None or not any(k in parsed for k in REQUIRED_KEYS):
            warn = "[warning] failed to parse JSON or missing keys;"
            if attempt < 4:
                print(warn + " retrying", file=sys.stderr)
                continue
            print(warn + " logging empty answer", file=sys.stderr)
            parsed = {"answer": []}
        elif "answer" not in parsed and "final_state" in parsed:
            parsed["answer"] = parsed["final_state"]
        break

    return parsed, total_usage, raw

def reconstruct_problem(record: Dict) -> Problem2D:
    """Rebuild a Problem2D from a JSON dict in the dataset."""
    init_grid = record["init"]              # nested list
    rule_bits = record["rule"]
    rule      = Rule2D(rule_bits)
    return Problem2D(
        start_grid=init_grid,
        rule=rule,
        timesteps=record["timesteps"],
    )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input",  type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--usage",  type=pathlib.Path, required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--tpm", type=int, default=60, help="max calls / minute")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--raw-log", type=pathlib.Path,
                    help="File to append raw model responses")
    args = ap.parse_args()

    # adjust rate limiter according to target RPM
    set_tpm(args.tpm)

    # only initialize the client if this is not a dry run
    global client
    if not args.dry_run:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if client.api_key is None:
            sys.exit("OPENAI_API_KEY env var not set")

    total_records = sum(1 for _ in args.input.open())

    # resume capability
    done = 0
    out_mode = "a"
    if args.output.exists():
        done = sum(1 for _ in args.output.open())
        if done > total_records:
            print(
                f"[info] existing prediction file has {done} lines; dataset has {total_records}. Starting fresh"
            )
            done = 0
            out_mode = "w"
    out_jsonl = args.output.open(out_mode, encoding="utf-8")
    raw_log_path = args.raw_log if args.raw_log else args.output.with_suffix(".raw")
    raw_log = raw_log_path.open("a", encoding="utf-8")
    if done:
        print(f"Resuming — {done} predictions already exist")

    usage_csv = args.usage.open("a", newline="")
    writer = csv.writer(usage_csv)
    if usage_csv.tell() == 0:
        writer.writerow(["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"])

    running_cost = 0.0

    with args.input.open() as fp_in:
        for idx, line in enumerate(fp_in, 1):
            if idx <= done:
                continue

            rec: Dict = json.loads(line.strip())
            prob = reconstruct_problem(rec)
            prompt = _prompt_gen.generate_prompt_2D(prob)

            if args.dry_run:
                out_jsonl.write(json.dumps({"answer": []}) + "\n")
                continue

            data, usage, raw_txt = chat_json(
                args.model, prompt, args.system, args.temperature
            )
            usd = usage["total"] / 1000 * PRICE_PER_1K
            running_cost += usd
            if running_cost > HARD_CAP:
                sys.exit(f"💸 cost {running_cost:.2f} > hard cap ${HARD_CAP}")

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

    print(f"Finished {idx-done} calls; total spent ≈ ${running_cost:.2f}. "
          f"Preds → {args.output}")

if __name__ == "__main__":
    main()
