import csv
import json
from pathlib import Path

from cabench.contracts import RUN_METADATA_SCHEMA_NAME, RUN_METADATA_SCHEMA_VERSION, SCORES_COLUMNS, USAGE_COLUMNS
from cabench.migrate import migrate_run_metadata, migrate_scores_csv, migrate_usage_csv


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
        w.writerow(["timestamp", "model", "prompt_tokens", "completion_tokens", "total_tokens", "usd_cost"])
        w.writerow(["t", "m", "1", "2", "3", "0.01"])

    changed = migrate_usage_csv(p)
    assert changed

    with p.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert rows[0] == USAGE_COLUMNS
    assert rows[1][5] == "0.01"


def test_migrate_run_metadata_adds_schema_and_cost_key(tmp_path: Path):
    p = tmp_path / "run_metadata.jsonl"
    p.write_text(json.dumps({"total_cost_usd": 0.2, "status": "completed"}) + "\n", encoding="utf-8")

    changed = migrate_run_metadata(p)
    assert changed
    rec = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert rec["schema_name"] == RUN_METADATA_SCHEMA_NAME
    assert rec["schema_version"] == RUN_METADATA_SCHEMA_VERSION
    assert rec["cost_usd"] == 0.2
    assert "total_cost_usd" not in rec

    assert (tmp_path / "run_metadata.jsonl.schema.json").exists()
