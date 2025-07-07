from __future__ import annotations
from openai import OpenAI
import os, backoff, openai
from .rate_limit import wait_one_second, set_tpm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
set_tpm(60)

@backoff.on_exception(backoff.expo,
                    (openai.RateLimitError, openai.APIError),
                    max_time=60)

def chat_once(model: str, prompt: str, temperature: float = 0.0):
    """
    Returns (reply_text, usage_dict)
    """
    wait_one_second()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1000,
    )
    msg = resp.choices[0].message.content.strip()
    u = resp.usage
    usage = {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
    }
    return msg, usage
