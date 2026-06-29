import json, pathlib, tempfile
from cabench.scoring import (
    EvalError,
    _flatten,
    normalized_hamming_accuracy,
    score,
    main as eval_cli,
)
import pytest


def _write_gold(path, targets):
    path.write_text("\n".join(json.dumps({"target": t}) for t in targets))

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


# --- Bug #2: blank/empty predictions count as invalid --------------------- #

def test_blank_pred_line_counts_invalid(tmp_path):
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.txt"
    _write_gold(gold, ["01", "11", "00"])
    # middle prediction is blank (model errored / unparseable)
    preds.write_text("01\n\n00")

    result = score(gold, preds)
    assert result["total"] == 3
    assert result["invalid"] == 1          # the blank row is flagged
    assert result["exact_match"] == 2      # the two good rows still match
    # blank row scored as fully wrong (0.0), so mean = (1 + 0 + 1) / 3
    assert result["norm_hamming"] == pytest.approx(2 / 3)


def test_all_blank_preds_all_invalid(tmp_path):
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.txt"
    _write_gold(gold, ["01", "11", "00"])
    preds.write_text("\n\n\n")             # three blank rows

    result = score(gold, preds)
    assert result["invalid"] == result["total"] == 3
    assert result["exact_match"] == 0
    assert result["norm_hamming"] == 0.0


# --- Bug #3: score() robust on empty / malformed gold --------------------- #

def test_score_empty_inputs_raises(tmp_path):
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.txt"
    gold.write_text("")
    preds.write_text("")

    with pytest.raises(EvalError):
        score(gold, preds)


def test_score_malformed_gold_line_raises(tmp_path):
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.txt"
    gold.write_text("not json\n")
    preds.write_text("01\n")

    with pytest.raises(EvalError):
        score(gold, preds)


def test_eval_cli_empty_inputs_exits_1(tmp_path):
    gold = tmp_path / "gold.jsonl"
    preds = tmp_path / "preds.txt"
    gold.write_text("")
    preds.write_text("")

    with pytest.raises(SystemExit) as exc:
        eval_cli(["--gold", str(gold), "--pred", str(preds)])
    assert exc.value.code == 1
