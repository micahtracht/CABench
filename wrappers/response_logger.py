from __future__ import annotations

import json
import pathlib
import time

# Path to the global response log file
LOG_PATH = pathlib.Path("logs") / "responses.log"


def log_response(model: str, text: str, *, log_file: pathlib.Path = LOG_PATH) -> None:
    """Append a record of a model's raw response to the log file.

    Each line is a JSON object with the keys:
      - ts: ISO timestamp (UTC)
      - model: model identifier
      - response: the raw text returned by the model
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "response": text,
    }
    with log_file.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

