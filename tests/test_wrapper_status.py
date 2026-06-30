import json
import types

import pytest

import cabench.llm.runner as runner


def _write_1d_dataset(path, n=2):
    rec = {"rule": "000000", "timesteps": 1, "init": "01", "target": "00"}
    path.write_text("\n".join(json.dumps(rec) for _ in range(n)) + "\n", encoding="utf-8")


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


class _FakeRateLimit(runner.RateLimitError):
    def __init__(self):
        Exception.__init__(self, "rate limited")


def test_chat_json_api_error_retries_then_terminal(monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner, "log_response", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "MAX_SAMPLE_ATTEMPTS", 2)
    monkeypatch.setattr(runner, "client", _FakeClient([_FakeRateLimit(), _FakeRateLimit()]))

    data, _usage, _raw = runner.chat_json("m", "p")
    assert data["status"] == "error"
    assert data["error_type"] == "_FakeRateLimit"
    assert data["attempts"] == 2


# --- run_batch level ------------------------------------------------------ #


def test_run_batch_records_usage_and_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner, "log_response", lambda *_a, **_k: None)
    monkeypatch.setattr(
        runner, "client", _FakeClient([_Resp('{"answer":[0,0]}'), _Resp('{"answer":[0,0]}')])
    )
    ds = tmp_path / "d.jsonl"
    _write_1d_dataset(ds, 2)

    counts = runner.run_batch(
        model="m",
        input_path=ds,
        output_path=tmp_path / "out.jsonl",
        usage_path=tmp_path / "usage.csv",
        dim=1,
        price_per_1k=0.01,
        hard_cap=None,
        tpm=30,  # forces the inter-call extra_wait sleep branch
    )
    assert counts["success"] == 2
    rows = (tmp_path / "usage.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3  # header + 2 calls


def test_run_batch_warns_when_no_price(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner, "log_response", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "DEFAULT_PRICE_PER_1K", None)
    monkeypatch.setattr(runner, "client", _FakeClient([_Resp('{"answer":[0,0]}')]))
    ds = tmp_path / "d.jsonl"
    _write_1d_dataset(ds, 1)

    runner.run_batch(
        model="m",
        input_path=ds,
        output_path=tmp_path / "o.jsonl",
        usage_path=tmp_path / "u.csv",
        dim=1,
        price_per_1k=None,
        hard_cap=None,
        tpm=60,
    )
    assert "price_per_1k not set" in capsys.readouterr().err


def test_run_batch_malformed_input_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner, "client", _FakeClient([]))
    ds = tmp_path / "d.jsonl"
    ds.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(runner.RunnerError, match="malformed dataset line 1"):
        runner.run_batch(
            model="m",
            input_path=ds,
            output_path=tmp_path / "o.jsonl",
            usage_path=tmp_path / "u.csv",
            dim=1,
            price_per_1k=0.01,
            tpm=60,
        )


def test_runner_main_dry_run(tmp_path):
    ds = tmp_path / "d.jsonl"
    _write_1d_dataset(ds, 2)
    out = tmp_path / "o.jsonl"

    runner.main(
        [
            "--model",
            "m",
            "--input",
            str(ds),
            "--output",
            str(out),
            "--usage",
            str(tmp_path / "u.csv"),
            "--dim",
            "1",
            "--dry-run",
        ]
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["status"] == "dry_run" for line in lines)


def test_run_batch_hard_cap_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner, "log_response", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "client", _FakeClient([_Resp('{"answer":[0,0]}')]))
    ds = tmp_path / "d.jsonl"
    _write_1d_dataset(ds, 1)

    with pytest.raises(runner.SpendCapError):
        runner.run_batch(
            model="m",
            input_path=ds,
            output_path=tmp_path / "o.jsonl",
            usage_path=tmp_path / "u.csv",
            dim=1,
            price_per_1k=1.0,  # 15 tokens -> $0.015, over the tiny cap
            hard_cap=0.0001,
            tpm=60,
        )


def test_run_batch_zero_hard_cap_disables_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "wait_one_second", lambda: None)
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runner, "log_response", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "client", _FakeClient([_Resp('{"answer":[0,0]}')]))
    ds = tmp_path / "d.jsonl"
    _write_1d_dataset(ds, 1)

    counts = runner.run_batch(
        model="m",
        input_path=ds,
        output_path=tmp_path / "o.jsonl",
        usage_path=tmp_path / "u.csv",
        dim=1,
        price_per_1k=1.0,
        hard_cap=0,  # <= 0 disables the cap
        tpm=60,
    )
    assert counts["success"] == 1


def test_run_batch_bad_usage_header_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "client", _FakeClient([]))
    ds = tmp_path / "d.jsonl"
    _write_1d_dataset(ds, 1)
    usage = tmp_path / "u.csv"
    usage.write_text("wrong,header\n", encoding="utf-8")

    with pytest.raises(runner.RunnerError):
        runner.run_batch(
            model="m",
            input_path=ds,
            output_path=tmp_path / "o.jsonl",
            usage_path=usage,
            dim=1,
            price_per_1k=0.01,
            tpm=60,
        )
