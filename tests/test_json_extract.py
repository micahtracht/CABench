from json_extract import extract_first_json_object


def test_extract_exact_json_object():
    assert extract_first_json_object('{"answer":[0,1]}') == {"answer": [0, 1]}


def test_extract_with_prose_before_and_after():
    s = 'Reasoning...\n{"answer":[1,0]}\nDone.'
    assert extract_first_json_object(s) == {"answer": [1, 0]}


def test_extract_code_fenced_json():
    s = "```json\n{\"answer\":[0,0,1]}\n```"
    assert extract_first_json_object(s) == {"answer": [0, 0, 1]}


def test_extract_returns_none_when_no_object():
    assert extract_first_json_object("no json here") is None


def test_extract_returns_none_for_non_object_json():
    assert extract_first_json_object("[1,2,3]") is None
