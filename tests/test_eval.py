import json, pathlib, tempfile
from eval import _flatten, normalized_hamming_accuracy, main as eval_cli
import pytest

def test_metric_exact():
    assert normalized_hamming_accuracy("1010", "1010") == 1.0
    assert normalized_hamming_accuracy("1010", "0000") == 0.5
    assert normalized_hamming_accuracy("0101", "1010") == 0.0
    assert normalized_hamming_accuracy("0000", "1010") == 0.5
    assert normalized_hamming_accuracy("", "1001") == 0.0
    assert normalized_hamming_accuracy("00", "0011") == 0.5
    
def test_eval_cli_gold_equals_pred(tmp_path, monkeypatch):
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.txt"
    
    # fake 3-line file
    data = [{"target": t} for t in ("01","11","00")]
    gold.write_text("\n".join(json.dumps(d) for d in data))
    preds.write_text("\n".join(d["target"] for d in data))
    
    # run eval CLI, expect exit 0
    with pytest.raises(SystemExit) as exc:
        eval_cli(["--gold", str(gold), "--pred", str(preds)])
    assert exc.value.code == 0

def test_flatten_nested_list():
    nested = [[1, 0], [0, 1]]
    assert _flatten(nested) == "1001"


def test_flatten_invalid_symbol():
    nested_bad = [[1, 2], [0]]
    with pytest.raises(ValueError):
        _flatten(nested_bad)


def test_eval_cli_length_mismatch(tmp_path):
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.txt"

    # two gold lines, one prediction line
    gold.write_text("\n".join(json.dumps({"target": "0"}) for _ in range(2)))
    preds.write_text("0\n")

    with pytest.raises(SystemExit) as exc:
        eval_cli(["--gold", str(gold), "--pred", str(preds)])
    assert exc.value.code == 1