import json, pathlib, tempfile
from eval import normalized_hamming_accuracy, main as eval_cli

def test_metric_exact():
    assert normalized_hamming_accuracy("1010", "1010") == 1.0
    assert normalized_hamming_accuracy("1010", "0000") == 0.25
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
    eval_cli(["--gold", str(gold), "--pred", str(preds)])
