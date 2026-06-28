import csv
import json
from pathlib import Path

from wrappers.run_llm_json import _reconcile_resume_state


def _usage_row(usd: str = "0.01000"):
    return [
        "2026-01-01T00:00:00Z",
        "gpt-4.1-mini",
        "10",
        "20",
        "30",
        usd,
    ]


def test_reconcile_truncates_corrupt_tail_and_aligns_files(tmp_path: Path):
    out_path = tmp_path / "preds.jsonl"
    usage_path = tmp_path / "usage.csv"

    out_lines = [
        json.dumps({"answer": [0, 1]}),
        "{bad-json",
        json.dumps({"answer": [1, 0]}),
    ]
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    with usage_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"])
        w.writerow(_usage_row("0.00100"))
        w.writerow(["bad", "row"])

    done = _reconcile_resume_state(out_path, usage_path, total_records=10)
    assert done == 1

    repaired_preds = out_path.read_text(encoding="utf-8").splitlines()
    assert repaired_preds == [json.dumps({"answer": [0, 1]})]

    with usage_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 2  # header + 1 good row


def test_reconcile_caps_to_dataset_size(tmp_path: Path):
    out_path = tmp_path / "preds.jsonl"
    usage_path = tmp_path / "usage.csv"

    out_path.write_text(
        "\n".join(
            [
                json.dumps({"answer": [0]}),
                json.dumps({"answer": [1]}),
                json.dumps({"answer": [0]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with usage_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["ts", "model", "prompt_tok", "completion_tok", "total_tok", "usd"])
        w.writerow(_usage_row("0.00100"))
        w.writerow(_usage_row("0.00200"))
        w.writerow(_usage_row("0.00300"))

    done = _reconcile_resume_state(out_path, usage_path, total_records=2)
    assert done == 2

    assert len(out_path.read_text(encoding="utf-8").splitlines()) == 2
    with usage_path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 3  # header + 2 rows
