import csv
from pathlib import Path

import pytest

from cabench.contracts import (
    SCORES_COLUMNS,
    SCORES_SCHEMA_NAME,
    SCORES_SCHEMA_VERSION,
    ensure_csv_header,
    write_schema_manifest,
)


def test_ensure_csv_header_creates_file_with_expected_header(tmp_path: Path):
    path = tmp_path / "scores.csv"
    ensure_csv_header(path, SCORES_COLUMNS)

    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))

    assert rows == [SCORES_COLUMNS]


def test_ensure_csv_header_raises_on_mismatch(tmp_path: Path):
    path = tmp_path / "scores.csv"
    path.write_text("a,b,c\n", encoding="utf-8")

    with pytest.raises(ValueError):
        ensure_csv_header(path, SCORES_COLUMNS)


def test_write_schema_manifest_with_suffix(tmp_path: Path):
    artifact = tmp_path / "scores.csv"
    manifest = write_schema_manifest(
        artifact,
        schema_name=SCORES_SCHEMA_NAME,
        schema_version=SCORES_SCHEMA_VERSION,
        fmt="csv",
        columns=SCORES_COLUMNS,
    )

    assert manifest == tmp_path / "scores.csv.schema.json"
    payload = manifest.read_text(encoding="utf-8")
    assert SCORES_SCHEMA_NAME in payload
    assert SCORES_SCHEMA_VERSION in payload


def test_write_schema_manifest_without_suffix(tmp_path: Path):
    artifact = tmp_path / "master_usage"
    manifest = write_schema_manifest(
        artifact,
        schema_name="demo.schema",
        schema_version="1.0.0",
        fmt="csv",
    )
    assert manifest == tmp_path / "master_usage.schema.json"
