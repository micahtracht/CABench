from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

SCORES_SCHEMA_NAME = "cabench.results.scores"
SCORES_SCHEMA_VERSION = "1.0.0"
SCORES_COLUMNS = [
    "date_utc",
    "dataset",
    "model",
    "norm_hamming",
    "exact_pct",
    "total_cost_usd",
    "pred_file",
]

USAGE_SCHEMA_NAME = "cabench.logs.usage"
USAGE_SCHEMA_VERSION = "1.0.0"
USAGE_COLUMNS = ["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"]

PRED_JSONL_SCHEMA_NAME = "cabench.predictions.jsonl"
PRED_JSONL_SCHEMA_VERSION = "1.0.0"

PREDS_TEXT_SCHEMA_NAME = "cabench.predictions.bits"
PREDS_TEXT_SCHEMA_VERSION = "1.0.0"

RUN_METADATA_SCHEMA_NAME = "cabench.results.run_metadata"
RUN_METADATA_SCHEMA_VERSION = "1.0.0"


def ensure_csv_header(path: Path, expected_header: Sequence[str]) -> None:
    """
    Ensure a CSV exists with the exact expected header.
    Creates the file with header if absent/empty.
    Raises ValueError on mismatch.
    """
    expected = list(expected_header)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(expected)
        return

    with path.open(newline="", encoding="utf-8") as fp:
        reader = csv.reader(fp)
        current = next(reader, None)

    if current != expected:
        raise ValueError(
            f"CSV header mismatch for {path}. expected={expected} got={current}"
        )


def write_schema_manifest(
    artifact_path: Path,
    *,
    schema_name: str,
    schema_version: str,
    fmt: str,
    columns: Sequence[str] | None = None,
    notes: str | None = None,
) -> Path:
    """
    Write/refresh a sidecar schema manifest:
    <artifact_name>.<suffix>.schema.json (or <name>.schema.json if no suffix).
    """
    if artifact_path.suffix:
        manifest_path = artifact_path.with_suffix(f"{artifact_path.suffix}.schema.json")
    else:
        manifest_path = artifact_path.with_name(f"{artifact_path.name}.schema.json")

    payload: dict[str, object] = {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "format": fmt,
        "artifact": artifact_path.name,
    }
    if columns is not None:
        payload["columns"] = list(columns)
    if notes:
        payload["notes"] = notes

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path
