from pathlib import Path

import cabench.cli as cabench_main


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
