import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import orchestrator
import wrappers.run_llm_json as run1d
import wrappers.run_llm_json_2d as run2d
import convert_predictions


def _run_main(module, argv: list[str], prog: str) -> None:
    old = sys.argv
    try:
        sys.argv = [prog, *argv]
        module.main()
    finally:
        sys.argv = old


class _FakeUsage:
    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.total_tokens = 15


class _FakeChoice:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)
        self.finish_reason = "stop"


class _FakeResp:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeOpenAI:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        # Deterministic mocked model output for both 1-D and 2-D tests.
        return _FakeResp('{"answer":[0,0]}')


def test_orchestrator_full_pipeline_with_mocked_openai(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    cfg_path = tmp_path / "bench.yaml"
    ds1 = tmp_path / "tiny1d.jsonl"
    ds2 = tmp_path / "tiny2d.jsonl"

    ds1.write_text(
        json.dumps({"rule": "000000", "timesteps": 1, "init": "01", "target": "00"}) + "\n",
        encoding="utf-8",
    )
    ds2.write_text(
        json.dumps({"rule": "0" * 18, "timesteps": 1, "init": [[0, 1]], "target": [[0, 0]]}) + "\n",
        encoding="utf-8",
    )
    cfg_path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: tiny1d",
                f"    path: {ds1}",
                "    dim: 1",
                "  - name: tiny2d",
                f"    path: {ds2}",
                "    dim: 2",
                "models:",
                "  - id: mocked-model",
                "    provider: openai",
                "    max_calls_per_min: 60",
                "    price_per_1k_tokens: 0.001",
                "    temperature: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(run1d, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(run2d, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(run1d, "wait_one_second", lambda: None)
    monkeypatch.setattr(run2d, "wait_one_second", lambda: None)
    monkeypatch.setattr(run1d.time, "sleep", lambda _s: None)
    monkeypatch.setattr(run2d.time, "sleep", lambda _s: None)
    monkeypatch.setattr(orchestrator, "get_git_metadata", lambda _root: {"commit": "abc", "dirty": False})

    def fake_shell(cmd):
        if len(cmd) >= 3 and cmd[1:3] == ["-m", "wrappers.run_llm_json"]:
            _run_main(run1d, cmd[3:], "wrappers.run_llm_json")
            return
        if len(cmd) >= 3 and cmd[1:3] == ["-m", "wrappers.run_llm_json_2d"]:
            _run_main(run2d, cmd[3:], "wrappers.run_llm_json_2d")
            return
        if len(cmd) >= 2 and Path(cmd[1]).name == "convert_predictions.py":
            _run_main(convert_predictions, cmd[2:], "convert_predictions.py")
            return
        raise AssertionError(f"Unexpected command in fake shell: {cmd}")

    monkeypatch.setattr(orchestrator, "shell", fake_shell)

    orchestrator.run(
        cfg_path=cfg_path,
        data_dir=data_dir,
        log_dir=log_dir,
        results_dir=results_dir,
        summarize=False,
    )

    with (results_dir / "scores.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 3  # header + 2 dataset rows
    assert rows[1][3] == "1.0000"
    assert rows[1][4] == "100.00"
    assert rows[2][3] == "1.0000"
    assert rows[2][4] == "100.00"

    metadata_lines = (results_dir / "run_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metadata_lines) == 2
