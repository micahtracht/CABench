import types

import cabench.llm.runner as runner


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
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        event = self._events[self._idx]
        self._idx += 1
        if isinstance(event, Exception):
            raise event
        return event


def test_chat_json_status_success(monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "client", _FakeClient([_Resp('{"answer":[0,1]}')]))

    data, usage, _raw = runner.chat_json("m", "p")
    assert data["status"] == "success"
    assert data["attempts"] == 1
    assert usage["total"] == 15


def test_chat_json_status_truncated(monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "client", _FakeClient([_Resp('{"answer":[1]}', "length")]))

    data, _usage, _raw = runner.chat_json("m", "p")
    assert data["status"] == "truncated"
    assert data["finish_reason"] == "length"


def test_chat_json_parse_retry_then_success(monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "MAX_SAMPLE_ATTEMPTS", 3)
    monkeypatch.setattr(
        runner,
        "client",
        _FakeClient([_Resp("not json"), _Resp('{"final_state":[1,0]}')]),
    )

    data, _usage, _raw = runner.chat_json("m", "p")
    assert data["status"] == "success"
    assert data["attempts"] == 2
    assert data["answer"] == [1, 0]


def test_chat_json_invalid_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "MAX_SAMPLE_ATTEMPTS", 2)
    monkeypatch.setattr(runner, "client", _FakeClient([_Resp("bad"), _Resp("still bad")]))

    data, _usage, _raw = runner.chat_json("m", "p")
    assert data["status"] == "invalid"
    assert data["attempts"] == 2
    assert data["answer"] == []


def test_chat_json_error_status_on_exception(monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(runner, "log_response", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "client", _FakeClient([RuntimeError("boom")]))

    data, _usage, _raw = runner.chat_json("m", "p")
    assert data["status"] == "error"
    assert data["error_type"] == "RuntimeError"
