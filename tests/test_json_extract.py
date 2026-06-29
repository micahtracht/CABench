from cabench.json_extract import extract_answer_json


def test_extract_exact_json_object():
    assert extract_answer_json('{"answer":[0,1]}') == {"answer": [0, 1]}


def test_extract_with_prose_before_and_after():
    s = 'Reasoning...\n{"answer":[1,0]}\nDone.'
    assert extract_answer_json(s) == {"answer": [1, 0]}


def test_extract_code_fenced_json():
    s = '```json\n{"answer":[0,0,1]}\n```'
    assert extract_answer_json(s) == {"answer": [0, 0, 1]}


def test_extract_returns_none_when_no_object():
    assert extract_answer_json("no json here") is None


def test_extract_returns_none_for_non_object_json():
    assert extract_answer_json("[1,2,3]") is None


def test_extract_prefers_last_answer_object():
    # A model that echoes the prompt's example object first, then emits the
    # real answer on the final line. The last answer-bearing object wins.
    s = 'Example: {"answer":[0,0,0]}\nWorking...\n{"answer":[1,0,1]}'
    assert extract_answer_json(s) == {"answer": [1, 0, 1]}


def test_extract_reasoning_trace_answer_on_last_line():
    s = 'Step 1: the left cell flips.\nStep 2: the right cell stays.\n{"answer":[1,1,0,0]}'
    assert extract_answer_json(s) == {"answer": [1, 1, 0, 0]}


def test_extract_skips_non_answer_objects():
    # A leading metadata object without an answer key must be ignored in favor
    # of the later object that actually carries the answer.
    s = '{"note":"thinking"}\n{"final_state":[0,1]}'
    assert extract_answer_json(s) == {"final_state": [0, 1]}


def test_extract_falls_back_to_last_dict_without_answer_key():
    # No object carries an answer key -> return the last decodable dict.
    s = '{"a":1}\n{"b":2}'
    assert extract_answer_json(s) == {"b": 2}
