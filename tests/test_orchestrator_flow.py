import csv
import json
from pathlib import Path

import pytest

import cabench.orchestrator as orchestrator
from cabench.contracts import USAGE_COLUMNS


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


def _fake_run_batch_factory(cost_by_model: dict[str, float], *, assert_fresh=False):
    """Stand in for the in-process LLM runner: write a prediction and a usage row."""

    def _run_batch(*, model, output_path, usage_path, **_kwargs):
        out_jsonl = Path(output_path)
        usage_csv = Path(usage_path)
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
        return {"success": 1, "invalid": 0, "error": 0, "truncated": 0}

    return _run_batch


def test_orchestrator_dry_run_skips_runner_and_eval(tmp_path: Path, monkeypatch):
    dataset_path = tmp_path / "gold.jsonl"
    cfg_path = tmp_path / "bench.yaml"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    results_dir = tmp_path / "results"
    _write_dataset(dataset_path)
    _write_config(cfg_path, dataset_path, ["m1"])

    monkeypatch.setattr(orchestrator, "get_git_metadata", lambda _root: {"commit": "abc", "dirty": False})
    monkeypatch.setattr(
        orchestrator,
        "run_batch",
        lambda **_k: (_ for _ in ()).throw(AssertionError("run_batch should not run")),
    )
    monkeypatch.setattr(
        orchestrator,
        "score",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("score should not run")),
    )

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
    monkeypatch.setattr(orchestrator, "run_batch", _fake_run_batch_factory({"m1": 0.01}))

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
    monkeypatch.setattr(orchestrator, "run_batch", _fake_run_batch_factory({"m1": 0.01}, assert_fresh=True))

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
    monkeypatch.setattr(orchestrator, "run_batch", _fake_run_batch_factory({"m1": 0.03, "m2": 0.03}))

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
    # Only the LLM call is faked; convert + score run for real in-process.
    monkeypatch.setattr(orchestrator, "run_batch", _fake_run_batch_factory({"m1": 0.01}))

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
