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
import backoff
import re
from .rate_limit import wait_one_second, set_tpm
from .response_logger import log_response

from generate import ECAProblemGenerator, Problem1D
from rules import Rule1D

# constants & client
PRICE_PER_1K = 0.003
HARD_CAP     = 5.00

client = None

# generator instance just for its prompt-builder method
gen = ECAProblemGenerator(state_size=0)  # size unused for prompt building

# default system prompt enforcing structured JSON
DEFAULT_SYSTEM_PROMPT = (
    "Respond with a JSON object containing keys 'initial_state' and 'final_state' "
    "(alias 'answer'). Do not include any extra text outside the JSON."
)

def extract_json_from_string(s: str) -> dict | None:
    """
    Finds and parses the first valid JSON object from a string.
    """
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None

@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_time=60)
def chat_json(
    model: str,
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
):
    """One JSON-mode call → (python_dict, token_usage, raw_text)."""
    wait_one_second()
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

    raw = resp.choices[0].message.content
    log_response(model, raw)
    parsed = extract_json_from_string(raw)
    if parsed is None:
        print(
            "[warning] failed to parse JSON from model response; "
            "logging empty answer",
            file=sys.stderr,
        )
        parsed = {"answer": []}

    return parsed, {
        "prompt": resp.usage.prompt_tokens,
        "completion": resp.usage.completion_tokens,
        "total": resp.usage.total_tokens,
    }, raw

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--usage",  type=pathlib.Path, required=True)
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
        else:
            print(f"Resuming — {done} predictions already exist")

    out_jsonl = args.output.open(out_mode, encoding="utf-8")
    raw_log_path = args.raw_log if args.raw_log else args.output.with_suffix(".raw")
    raw_log = raw_log_path.open("a", encoding="utf-8")
    usage_csv = args.usage.open("a", newline="")
    writer = csv.writer(usage_csv)
    if usage_csv.tell() == 0:
        writer.writerow(
            ["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"]
        )

    running_cost = 0.0
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
                out_jsonl.write(json.dumps({"answer": []}) + "\n")
                continue

            # -------- live call -------------------------------------
            data, usage, raw_txt = chat_json(
                args.model, prompt, args.system, args.temperature
            )
            usd = usage["total"] / 1000 * PRICE_PER_1K
            running_cost += usd
            if running_cost > HARD_CAP:
                sys.exit(f"cost {running_cost:.2f} > hard cap ${HARD_CAP}")

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
        f"Preds → {args.output}"
    )

if __name__ == "__main__":
    main()
