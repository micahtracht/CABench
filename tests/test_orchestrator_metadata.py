import json
from pathlib import Path

import cabench.orchestrator as orchestrator
def test_file_sha256_is_stable(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello\n", encoding="utf-8")

    h1 = orchestrator.file_sha256(p)
    h2 = orchestrator.file_sha256(p)
    assert h1 == h2
    assert len(h1) == 64


def test_append_jsonl_record_writes_one_line(tmp_path: Path):
    p = tmp_path / "meta.jsonl"
    orchestrator.append_jsonl_record(p, {"a": 1, "b": "x"})
    orchestrator.append_jsonl_record(p, {"a": 2})

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1, "b": "x"}
    assert json.loads(lines[1]) == {"a": 2}


def test_get_git_metadata_success(monkeypatch):
    class _Proc:
        def __init__(self, stdout: str, returncode: int = 0):
            self.stdout = stdout
            self.returncode = returncode

    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Proc("abc123\n", 0)
        return _Proc(" M orchestrator.py\n", 0)

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    meta = orchestrator.get_git_metadata(Path("."))
    assert meta == {"commit": "abc123", "dirty": True}


def test_get_git_metadata_failure(monkeypatch):
    def fake_run(*args, **kwargs):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    meta = orchestrator.get_git_metadata(Path("."))
    assert meta == {"commit": None, "dirty": None}
