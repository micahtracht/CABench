# single_call_test_v1.py   (works with openai 1.x)

import csv, os, pathlib, re, sys, time
import openai
from openai import OpenAI
from cabench.env import load_project_env
from .rate_limit import wait_one_second, set_tpm
from .response_logger import log_response

load_project_env()

MODEL = "gpt-4o-mini"
DEFAULT_PRICE_PER_1K = os.getenv("CABENCH_PRICE_PER_1K")
try:
    PRICE_PER_1K = float(DEFAULT_PRICE_PER_1K) if DEFAULT_PRICE_PER_1K else 0.003
except ValueError:
    PRICE_PER_1K = 0.003
DEFAULT_HARD_LIMIT_USD = os.getenv("CABENCH_HARD_CAP")
try:
    HARD_LIMIT_USD = float(DEFAULT_HARD_LIMIT_USD) if DEFAULT_HARD_LIMIT_USD else 0.05
except ValueError:
    HARD_LIMIT_USD = 0.05

prompt_text = (
    "You are given the following 1-D CA state:\n"
    "1,0,1,0\n"
    "Every cell stays the same. Return the state after 1 step.\n"
    "Answer only as a comma-separated list."
)

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    sys.exit("OPENAI_API_KEY env var missing")

client = OpenAI(api_key=api_key)
set_tpm(60)

wait_one_second()
resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt_text}],
    temperature=0,
    max_tokens=1000,
)

raw_txt = resp.choices[0].message.content
log_response(MODEL, raw_txt)
reply_text = raw_txt.strip()

usage = resp.usage
usd_cost = usage.total_tokens / 1000 * PRICE_PER_1K
if usd_cost > HARD_LIMIT_USD:
    sys.exit("💸 exceeded demo hard cap")

print("MODEL REPLY :", reply_text)
print("TOKEN USAGE :", usage)
print(f"APPROX USD  : ${usd_cost:.4f}")

# ---- append to CSV -------------------------------------------------------- #
log_path = pathlib.Path("logs") / "usage_hello.csv"
log_path.parent.mkdir(exist_ok=True)

header = ["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd_cost"]
row    = [
    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    MODEL,
    usage.prompt_tokens,
    usage.completion_tokens,
    usage.total_tokens,
    f"{usd_cost:.5f}",
]

write_header = not log_path.exists()
with log_path.open("a", newline="") as fp:
    w = csv.writer(fp)
    if write_header:
        w.writerow(header)
    w.writerow(row)

print("Logged to", log_path)
