import csv
from pathlib import Path

import pytest

from cabench.orchestrator import OrchestratorError, read_usage_csv


def test_read_usage_csv_missing_file_returns_zero(tmp_path: Path):
    total, rows = read_usage_csv(tmp_path / "missing.csv")
    assert total == 0.0
    assert rows == []


def test_read_usage_csv_sums_usd(tmp_path: Path):
    path = tmp_path / "usage.csv"
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"])
        w.writerow(["t1", "m", "1", "1", "2", "0.01000"])
        w.writerow(["t2", "m", "1", "1", "2", "0.02000"])

    total, _ = read_usage_csv(path)
    assert total == 0.03


def test_read_usage_csv_invalid_usd_exits(tmp_path: Path):
    path = tmp_path / "usage.csv"
    with path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"])
        w.writerow(["t1", "m", "1", "1", "2", "not-a-number"])

    with pytest.raises(OrchestratorError):
        read_usage_csv(path)
