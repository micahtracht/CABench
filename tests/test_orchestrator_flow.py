import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import orchestrator
from contracts import USAGE_COLUMNS
from convert_predictions import convert_file


def _write_dataset(path: Path, *, target: str = "00") -> None:
    rec = {"rule": "000000", "timesteps": 1, "init": "01", "target": target}
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def _write_config(path: Path, dataset_path: Path, model_ids: list[str]) -> None:
    lines = [
        "datasets:",
        "  - name: tiny",
        f"    path: {dataset_path}",
        "    dim: 1",
        "models:",
    ]
    for mid in model_ids:
        lines.extend(
            [
                f"  - id: {mid}",
                "    provider: openai",
                "    max_calls_per_min: 60",
                "    price_per_1k_tokens: 0.001",
                "    temperature: 0",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_eval_run(cmd, capture_output=False, text=False, check=False):
    if isinstance(cmd, list) and len(cmd) >= 2 and cmd[1] == "eval.py":
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "-Evaluated 1 cases\n"
                "-Normalized Hamming accuracy: 1.0000\n"
                "-Exact-match accuracy: 1/1 = 100.00%\n"
            ),
            stderr="",
        )
    raise AssertionError(f"Unexpected subprocess.run call in test: {cmd}")


def _shell_factory(cost_by_model: dict[str, float], *, assert_fresh=False):
    def _shell(cmd):
        if len(cmd) >= 4 and cmd[1] == "-m" and cmd[2].startswith("wrappers.run_llm_json"):
            model = cmd[cmd.index("--model") + 1]
            out_jsonl = Path(cmd[cmd.index("--output") + 1])
            usage_csv = Path(cmd[cmd.index("--usage") + 1])
            if assert_fresh:
                assert not out_jsonl.exists()
                assert not usage_csv.exists()

            out_jsonl.parent.mkdir(parents=True, exist_ok=True)
            usage_csv.parent.mkdir(parents=True, exist_ok=True)
            out_jsonl.write_text('{"answer":[0,0]}\n', encoding="utf-8")

            exists = usage_csv.exists() and usage_csv.stat().st_size > 0
            with usage_csv.open("a", newline="", encoding="utf-8") as fp:
                w = csv.writer(fp)
                if not exists:
                    w.writerow(USAGE_COLUMNS)
                w.writerow(
                    [
                        "2026-01-01T00:00:00Z",
                        model,
                        "10",
                        "10",
                        "20",
                        f"{cost_by_model.get(model, 0.01):.5f}",
                    ]
                )
            return

        if len(cmd) >= 2 and Path(cmd[1]).name == "convert_predictions.py":
            in_path = Path(cmd[cmd.index("--input") + 1])
            out_path = Path(cmd[cmd.index("--output") + 1])
            convert_file(in_path, out_path)
            return

        raise AssertionError(f"Unexpected shell call in test: {cmd}")

    return _shell


def test_orchestrator_dry_run_skips_shell_and_eval(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "gold.jsonl"
    cfg_path = tmp_path / "bench.yaml"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    _write_dataset(dataset_path)
    _write_config(cfg_path, dataset_path, ["m1"])

    monkeypatch.setattr(orchestrator, "get_git_metadata", lambda _root: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(orchestrator, "shell", lambda _cmd: (_ for _ in ()).throw(AssertionError("shell should not run")))
    monkeypatch.setattr(orchestrator.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("eval should not run")))

    orchestrator.run(
        cfg_path=cfg_path,
        dry_run=True,
        data_dir=data_dir,
        log_dir=log_dir,
        results_dir=results_dir,
        summarize=False,
    )

    marker = data_dir / "m1_tiny.dryrun.json"
    assert marker.exists()
    scores = (results_dir / "scores.csv").read_text(encoding="utf-8")
    assert "DRY_RUN:" in scores


def test_orchestrator_master_usage_dedupes_on_rerun(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "gold.jsonl"
    cfg_path = tmp_path / "bench.yaml"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    _write_dataset(dataset_path)
    _write_config(cfg_path, dataset_path, ["m1"])

    monkeypatch.setattr(orchestrator, "get_git_metadata", lambda _root: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(orchestrator, "shell", _shell_factory({"m1": 0.01}))
    monkeypatch.setattr(orchestrator.subprocess, "run", _fake_eval_run)

    orchestrator.run(cfg_path, data_dir=data_dir, log_dir=log_dir, results_dir=results_dir, summarize=False)
    orchestrator.run(cfg_path, data_dir=data_dir, log_dir=log_dir, results_dir=results_dir, summarize=False)

    with (log_dir / "master_usage.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 3  # header + 2 runs (not duplicated historical rows)


def test_orchestrator_force_preds_restarts_usage_and_preds(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "gold.jsonl"
    cfg_path = tmp_path / "bench.yaml"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    _write_dataset(dataset_path)
    _write_config(cfg_path, dataset_path, ["m1"])

    stale_jsonl = data_dir / "m1_tiny.jsonl"
    stale_preds = data_dir / "m1_tiny.preds"
    stale_usage = log_dir / "m1_tiny_usage.csv"
    stale_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stale_preds.parent.mkdir(parents=True, exist_ok=True)
    stale_usage.parent.mkdir(parents=True, exist_ok=True)
    stale_jsonl.write_text('{"answer":[1,1]}\n', encoding="utf-8")
    stale_preds.write_text("11\n", encoding="utf-8")
    with stale_usage.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(USAGE_COLUMNS)
        w.writerow(["old", "m1", "1", "1", "2", "0.50000"])

    monkeypatch.setattr(orchestrator, "get_git_metadata", lambda _root: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(orchestrator, "shell", _shell_factory({"m1": 0.01}, assert_fresh=True))
    monkeypatch.setattr(orchestrator.subprocess, "run", _fake_eval_run)

    orchestrator.run(
        cfg_path=cfg_path,
        force_preds=True,
        data_dir=data_dir,
        log_dir=log_dir,
        results_dir=results_dir,
        summarize=False,
    )

    with stale_usage.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 2  # header + fresh row only


def test_orchestrator_budget_cap_uses_actual_usage(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "gold.jsonl"
    cfg_path = tmp_path / "bench.yaml"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    _write_dataset(dataset_path)
    _write_config(cfg_path, dataset_path, ["m1", "m2"])

    monkeypatch.setattr(orchestrator, "HARD_SPEND_CEILING", 0.05)
    monkeypatch.setattr(orchestrator, "get_git_metadata", lambda _root: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(orchestrator, "shell", _shell_factory({"m1": 0.03, "m2": 0.03}))
    monkeypatch.setattr(orchestrator.subprocess, "run", _fake_eval_run)

    with pytest.raises(SystemExit) as exc:
        orchestrator.run(
            cfg_path=cfg_path,
            data_dir=data_dir,
            log_dir=log_dir,
            results_dir=results_dir,
            summarize=False,
        )
    assert "exceeded hard ceiling" in str(exc.value)


def test_orchestrator_convert_and_eval_integration(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "gold.jsonl"
    cfg_path = tmp_path / "bench.yaml"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    _write_dataset(dataset_path, target="00")
    _write_config(cfg_path, dataset_path, ["m1"])

    monkeypatch.setattr(orchestrator, "get_git_metadata", lambda _root: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(orchestrator, "shell", _shell_factory({"m1": 0.01}))

    # Use real subprocess.run for eval.py invocation.
    orchestrator.run(
        cfg_path=cfg_path,
        data_dir=data_dir,
        log_dir=log_dir,
        results_dir=results_dir,
        summarize=False,
    )

    with (results_dir / "scores.csv").open(newline="", encoding="utf-8") as fp:
        rows = list(csv.reader(fp))
    assert len(rows) == 2
    assert rows[1][3] == "1.0000"  # norm_hamming
    assert rows[1][4] == "100.00"  # exact_pct
