import types

import pytest

import wrappers.run_llm_json as w1
import wrappers.run_llm_json_2d as w2


class _Usage:
    def __init__(self, prompt=10, completion=5, total=15):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _Choice:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage()


class _FakeClient:
    def __init__(self, events):
        self._events = list(events)
        self._idx = 0
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        event = self._events[self._idx]
        self._idx += 1
        if isinstance(event, Exception):
            raise event
        return event


@pytest.mark.parametrize("mod", [w1, w2])
def test_chat_json_status_success(mod, monkeypatch):
    monkeypatch.setattr(mod, "wait_one_second", lambda: None)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "client", _FakeClient([_Resp('{"answer":[0,1]}')]))

    data, usage, _raw = mod.chat_json("m", "p")
    assert data["status"] == "success"
    assert data["attempts"] == 1
    assert usage["total"] == 15


@pytest.mark.parametrize("mod", [w1, w2])
def test_chat_json_status_truncated(mod, monkeypatch):
    monkeypatch.setattr(mod, "wait_one_second", lambda: None)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "client", _FakeClient([_Resp('{"answer":[1]}', "length")]))

    data, _usage, _raw = mod.chat_json("m", "p")
    assert data["status"] == "truncated"
    assert data["finish_reason"] == "length"


@pytest.mark.parametrize("mod", [w1, w2])
def test_chat_json_parse_retry_then_success(mod, monkeypatch):
    monkeypatch.setattr(mod, "wait_one_second", lambda: None)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "MAX_SAMPLE_ATTEMPTS", 3)
    monkeypatch.setattr(
        mod,
        "client",
        _FakeClient([_Resp("not json"), _Resp('{"final_state":[1,0]}')]),
    )

    data, _usage, _raw = mod.chat_json("m", "p")
    assert data["status"] == "success"
    assert data["attempts"] == 2
    assert data["answer"] == [1, 0]


@pytest.mark.parametrize("mod", [w1, w2])
def test_chat_json_invalid_after_exhausting_attempts(mod, monkeypatch):
    monkeypatch.setattr(mod, "wait_one_second", lambda: None)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "MAX_SAMPLE_ATTEMPTS", 2)
    monkeypatch.setattr(mod, "client", _FakeClient([_Resp("bad"), _Resp("still bad")]))

    data, _usage, _raw = mod.chat_json("m", "p")
    assert data["status"] == "invalid"
    assert data["attempts"] == 2
    assert data["answer"] == []


@pytest.mark.parametrize("mod", [w1, w2])
def test_chat_json_error_status_on_exception(mod, monkeypatch):
    monkeypatch.setattr(mod, "wait_one_second", lambda: None)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(mod, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mod, "client", _FakeClient([RuntimeError("boom")]))

    data, _usage, _raw = mod.chat_json("m", "p")
    assert data["status"] == "error"
    assert data["error_type"] == "RuntimeError"
