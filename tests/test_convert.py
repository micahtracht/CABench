import json

import pytest

from cabench.convert import convert_file, json_to_bits
from cabench.convert import main as convert_main


def test_json_to_bits_flat():
    assert json_to_bits({"answer": [0, 1, 0, 1]}) == "0101"


def test_json_to_bits_nested():
    assert json_to_bits({"answer": [[0, 1], [1, 0]]}) == "0110"


def test_json_to_bits_final_state_alias():
    assert json_to_bits({"final_state": [1, 0]}) == "10"


def test_json_to_bits_missing_answer_raises():
    with pytest.raises(ValueError, match="missing 'answer'"):
        json_to_bits({"foo": 1})


def test_json_to_bits_invalid_bit_raises():
    with pytest.raises(ValueError, match="invalid bit"):
        json_to_bits({"answer": [0, 2]})


def test_json_to_bits_empty_answer_raises():
    with pytest.raises(ValueError, match="empty"):
        json_to_bits({"answer": []})


def test_convert_file_ok_bad_and_blank(tmp_path, capsys):
    inp = tmp_path / "in.jsonl"
    inp.write_text(
        "\n".join(
            [
                json.dumps({"answer": [0, 1]}),  # ok -> "01"
                "not json at all",  # unparseable -> blank
                json.dumps({"answer": []}),  # empty answer -> blank
                "",  # blank input line -> skipped (no output)
                json.dumps({"final_state": [1, 1]}),  # ok via alias -> "11"
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.preds"

    convert_file(inp, out)

    # blank input line produces no output row; the two bad rows produce blanks.
    assert out.read_text(encoding="utf-8").splitlines() == ["01", "", "", "11"]
    assert "[warn]" in capsys.readouterr().err
    assert (tmp_path / "out.preds.schema.json").exists()


def test_convert_main_cli(tmp_path):
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"answer": [1, 0]}) + "\n", encoding="utf-8")
    out = tmp_path / "out.preds"

    convert_main(["--input", str(inp), "--output", str(out)])
    assert out.read_text(encoding="utf-8").splitlines() == ["10"]
