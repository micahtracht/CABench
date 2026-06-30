import json
from pathlib import Path

import pytest

import cabench.cli as cabench_main


def _write_gen_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: g1",
                "    gen:",
                "      mode: 1d",
                "      n: 4",
                "      size: 8",
                "      timesteps: 2",
                "      density: 0.5",
                "      seed: 1",
                "    path: data/g1.jsonl",
                "    dim: 1",
                "models:",
                "  - id: m1",
                "    provider: openai",
                "    max_calls_per_min: 60",
                "    price_per_1k_tokens: 0.001",
                "    temperature: 0",
                "presets:",
                "  - name: quick",
                "    datasets: [g1]",
                "    models: [m1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: d1",
                "    path: data/d1.jsonl",
                "    dim: 1",
                "  - name: d2",
                "    path: data/d2.jsonl",
                "    dim: 2",
                "models:",
                "  - id: m1",
                "    provider: openai",
                "    max_calls_per_min: 60",
                "    price_per_1k_tokens: 0.001",
                "    temperature: 0",
                "  - id: m2",
                "    provider: openai",
                "    max_calls_per_min: 60",
                "    price_per_1k_tokens: 0.001",
                "    temperature: 0",
                "presets:",
                "  - name: quick",
                "    description: Small run",
                "    datasets: [d1]",
                "    models: [m1]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_command_uses_preset_and_default_spend_cap(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "bench.yaml"
    _write_config(cfg)
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cabench_main.orchestrator, "run", fake_run)

    rc = cabench_main.main(
        [
            "run",
            "--config",
            str(cfg),
            "--preset",
            "quick",
            "--results-dir",
            str(tmp_path / "results"),
            "--data-dir",
            str(tmp_path / "data"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--no-report",
        ]
    )

    assert rc == 0
    assert captured["model_ids"] == ["m1"]
    assert captured["dataset_names"] == ["d1"]
    assert captured["spend_cap_usd"] == 1.0


def test_list_presets_includes_default_and_custom(tmp_path: Path, capsys):
    cfg = tmp_path / "bench.yaml"
    _write_config(cfg)

    rc = cabench_main.main(["list-presets", "--config", str(cfg)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "default:" in out
    assert "quick:" in out


def _run_dirs(tmp_path: Path) -> list[str]:
    return [
        "--data-dir",
        str(tmp_path / "data"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--results-dir",
        str(tmp_path / "results"),
    ]


def test_run_command_dry_run_generates_summarizes_and_reports(tmp_path, capsys, monkeypatch):
    cfg = tmp_path / "bench.yaml"
    _write_gen_config(cfg)
    monkeypatch.setattr(
        cabench_main.orchestrator, "get_git_metadata", lambda _r: {"commit": "x", "dirty": False}
    )

    rc = cabench_main.main(
        ["run", "--config", str(cfg), "--preset", "quick", "--dry-run", *_run_dirs(tmp_path)]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Run summary" in out  # print_run_summary path (no --no-summary)
    assert (tmp_path / "data" / "g1.jsonl").exists()  # generation happened
    assert (tmp_path / "results" / "latest_report.txt").exists()  # report path


def test_run_command_unknown_preset_exits(tmp_path):
    cfg = tmp_path / "bench.yaml"
    _write_config(cfg)
    with pytest.raises(SystemExit):
        cabench_main.main(["run", "--config", str(cfg), "--preset", "ghost", "--no-report"])


def test_run_command_merges_model_id_and_models(tmp_path, monkeypatch):
    cfg = tmp_path / "bench.yaml"
    _write_config(cfg)
    captured: dict = {}
    monkeypatch.setattr(cabench_main.orchestrator, "run", lambda **k: captured.update(k))

    cabench_main.main(
        [
            "run",
            "--config",
            str(cfg),
            "--preset",
            "quick",
            "--model-id",
            "m2",
            "--models",
            "m1,m2",
            "--no-report",
            *_run_dirs(tmp_path),
        ]
    )
    assert captured["model_ids"] == ["m1", "m2"]  # merged + deduped + sorted


def test_main_catches_domain_error_and_returns_1(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "bench.yaml"
    _write_config(cfg)

    def _boom(**_k):
        raise cabench_main.OrchestratorError("nope")

    monkeypatch.setattr(cabench_main.orchestrator, "run", _boom)

    rc = cabench_main.main(
        ["run", "--config", str(cfg), "--preset", "quick", "--no-report", *_run_dirs(tmp_path)]
    )
    assert rc == 1
    assert "nope" in capsys.readouterr().err


def test_main_passthrough_convert(tmp_path):
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps({"answer": [1, 0]}) + "\n", encoding="utf-8")
    out = tmp_path / "out.preds"

    rc = cabench_main.main(["convert", "--input", str(inp), "--output", str(out)])
    assert rc == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["10"]


def test_report_command_renders_and_writes(tmp_path):
    scores = tmp_path / "scores.csv"
    scores.write_text(
        "date_utc,dataset,model,norm_hamming,exact_pct,total_cost_usd,pred_file\n"
        "2026-01-01T00:00:00+00:00,d1,m1,0.5000,50.00,0.1000,p\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.txt"

    rc = cabench_main.main(
        ["report", "--scores", str(scores), "--out", str(out), "--results-dir", str(tmp_path)]
    )
    assert rc == 0
    assert out.exists()
    assert "m1" in out.read_text(encoding="utf-8")
