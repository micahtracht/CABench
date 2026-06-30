import csv
import json
from pathlib import Path

import pytest

from cabench.contracts import (
    RUN_METADATA_SCHEMA_NAME,
    RUN_METADATA_SCHEMA_VERSION,
    SCORES_COLUMNS,
    USAGE_COLUMNS,
)
from cabench.migrate import (
    main as migrate_main,
)
from cabench.migrate import (
    migrate_run_metadata,
    migrate_scores_csv,
    migrate_usage_csv,
)


def test_migrate_scores_csv_legacy_header(tmp_path: Path):
    p = tmp_path / "scores.csv"
    with p.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["date", "dataset", "model", "norm", "exact", "total_cost", "preds"])
        w.writerow(["t", "d", "m", "0.5", "10.0", "0.1", "x"])

    changed = migrate_scores_csv(p)
    assert changed

    with p.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert rows[0] == SCORES_COLUMNS
    assert rows[1][3] == "0.5"

    assert (tmp_path / "scores.csv.schema.json").exists()


def test_migrate_usage_csv_aliases(tmp_path: Path):
    p = tmp_path / "usage.csv"
    with p.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            ["timestamp", "model", "prompt_tokens", "completion_tokens", "total_tokens", "usd_cost"]
        )
        w.writerow(["t", "m", "1", "2", "3", "0.01"])

    changed = migrate_usage_csv(p)
    assert changed

    with p.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert rows[0] == USAGE_COLUMNS
    assert rows[1][5] == "0.01"


def test_migrate_run_metadata_adds_schema_and_cost_key(tmp_path: Path):
    p = tmp_path / "run_metadata.jsonl"
    p.write_text(
        json.dumps({"total_cost_usd": 0.2, "status": "completed"}) + "\n", encoding="utf-8"
    )

    changed = migrate_run_metadata(p)
    assert changed
    rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert rec["schema_name"] == RUN_METADATA_SCHEMA_NAME
    assert rec["schema_version"] == RUN_METADATA_SCHEMA_VERSION
    assert rec["cost_usd"] == 0.2
    assert "total_cost_usd" not in rec

    assert (tmp_path / "run_metadata.jsonl.schema.json").exists()


# --- no-op / empty-file paths --------------------------------------------- #


def test_migrate_scores_already_canonical_is_noop(tmp_path: Path):
    p = tmp_path / "scores.csv"
    with p.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(SCORES_COLUMNS)
        w.writerow(["t", "d", "m", "0.5", "10.0", "0.1", "x"])

    assert migrate_scores_csv(p) is False  # already canonical
    assert (tmp_path / "scores.csv.schema.json").exists()  # manifest still refreshed


def test_migrate_empty_usage_file_is_noop(tmp_path: Path):
    p = tmp_path / "usage.csv"
    p.write_text("", encoding="utf-8")
    assert migrate_usage_csv(p) is False


def test_migrate_run_metadata_missing_file_is_noop(tmp_path: Path):
    assert migrate_run_metadata(tmp_path / "nope.jsonl") is False


def test_migrate_run_metadata_already_canonical_is_noop(tmp_path: Path):
    p = tmp_path / "run_metadata.jsonl"
    rec = {
        "schema_name": RUN_METADATA_SCHEMA_NAME,
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "cost_usd": 0.1,
    }
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert migrate_run_metadata(p) is False


def test_migrate_run_metadata_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "run_metadata.jsonl"
    p.write_text(
        "\n" + json.dumps({"total_cost_usd": 0.2}) + "\n\n",
        encoding="utf-8",
    )
    assert migrate_run_metadata(p) is True
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


# --- error branches ------------------------------------------------------- #


def test_migrate_unrecognized_header_raises(tmp_path: Path):
    p = tmp_path / "scores.csv"
    with p.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(["date", "dataset", "model", "norm", "exact", "wat", "preds"])
    with pytest.raises(ValueError, match="Unrecognized header"):
        migrate_scores_csv(p)


def test_migrate_duplicate_mapped_columns_raises(tmp_path: Path):
    p = tmp_path / "usage.csv"
    # both "timestamp" and "ts" canonicalize to "ts"
    with p.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(
            ["timestamp", "ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"]
        )
    with pytest.raises(ValueError, match="Duplicate mapped"):
        migrate_usage_csv(p)


def test_migrate_missing_required_columns_raises(tmp_path: Path):
    p = tmp_path / "scores.csv"
    # all recognized, no dups, but missing pred_file
    with p.open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerow(["date", "dataset", "model", "norm", "exact", "total_cost"])
    with pytest.raises(ValueError, match="Missing required columns"):
        migrate_scores_csv(p)


def test_migrate_run_metadata_invalid_json_raises(tmp_path: Path):
    p = tmp_path / "run_metadata.jsonl"
    p.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        migrate_run_metadata(p)


def test_migrate_run_metadata_non_dict_record_raises(tmp_path: Path):
    p = tmp_path / "run_metadata.jsonl"
    p.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        migrate_run_metadata(p)


# --- CLI end-to-end ------------------------------------------------------- #


def test_migrate_cli_end_to_end(tmp_path: Path, capsys):
    results_dir = tmp_path / "results"
    log_dir = tmp_path / "logs"
    results_dir.mkdir()
    log_dir.mkdir()

    scores = results_dir / "scores.csv"
    with scores.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["date", "dataset", "model", "norm", "exact", "total_cost", "preds"])
        w.writerow(["t", "d", "m", "0.5", "10.0", "0.1", "x"])

    per_run_usage = log_dir / "gpt_d_usage.csv"
    with per_run_usage.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            ["timestamp", "model", "prompt_tokens", "completion_tokens", "total_tokens", "usd_cost"]
        )
        w.writerow(["t", "m", "1", "2", "3", "0.01"])
    # run_metadata.jsonl intentionally absent -> exercises the "skipped (missing)" branch

    migrate_main(["--results-dir", str(results_dir), "--log-dir", str(log_dir)])

    out = capsys.readouterr().out
    assert "migration_complete" in out
    assert "run_metadata: skipped" in out

    with scores.open(newline="", encoding="utf-8") as fp:
        assert next(csv.reader(fp)) == SCORES_COLUMNS
    with per_run_usage.open(newline="", encoding="utf-8") as fp:
        assert next(csv.reader(fp)) == USAGE_COLUMNS
