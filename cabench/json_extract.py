from __future__ import annotations

import json

_ANSWER_KEYS = ("answer", "final_state")


def extract_answer_json(text: str) -> dict | None:
    """
    Parse and return the answer-bearing JSON object from `text`.

    The prompts instruct the model to emit its answer object on the very last
    line, after any reasoning. So when several JSON objects are present we
    prefer the LAST one that carries an answer key, falling back to the last
    decodable object. This handles:
    - exact JSON object strings
    - prose / reasoning before the JSON
    - code-fenced JSON blocks
    - an echoed example object followed by the real answer
    """
    if not text:
        return None

    s = text.strip()
    if not s:
        return None

    # Collect every decodable top-level JSON object in document order.
    objects: list[dict] = []

    # Fast path: the whole payload is a single JSON object.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Markdown fence: decode the fenced body as a single object if possible.
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

    # Robust scan: gather all decodable JSON objects, in order.
    decoder = json.JSONDecoder()
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(s[i:])
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        i += end  # skip past the object we just decoded

    if not objects:
        return None

    for obj in reversed(objects):
        if any(k in obj for k in _ANSWER_KEYS):
            return obj
    return objects[-1]
