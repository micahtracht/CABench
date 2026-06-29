from __future__ import annotations

import json


def extract_first_json_object(text: str) -> dict | None:
    """
    Parse and return the first JSON object found in `text`.
    This avoids regex-based extraction and handles:
    - exact JSON object strings
    - prose before/after JSON
    - code-fenced JSON blocks
    """
    if not text:
        return None

    s = text.strip()
    if not s:
        return None

    # Fast path: the whole payload is JSON.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Handle common markdown fences.
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            inner = "\n".join(lines[1:-1]).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

    # Robust fallback: scan for the first decodable JSON object start.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(s[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    return None
