import json
from pathlib import Path

import cabench.orchestrator as orchestrator


def test_parse_csv_arg():
    assert orchestrator.parse_csv_arg(None) is None
    assert orchestrator.parse_csv_arg("") is None
    assert orchestrator.parse_csv_arg("a,b , c") == ["a", "b", "c"]


def test_run_dry_with_custom_dirs(tmp_path: Path):
    data_dir = tmp_path / "out_data"
    log_dir = tmp_path / "out_logs"
    results_dir = tmp_path / "out_results"
    dataset_path = tmp_path / "gold.jsonl"
    cfg_path = tmp_path / "bench.yaml"

    dataset_path.write_text(
        json.dumps({"rule": "000000", "timesteps": 1, "init": "01", "target": "00"}) + "\n",
        encoding="utf-8",
    )

    cfg_path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: tiny",
                f"    path: {dataset_path}",
                "    dim: 1",
                "models:",
                "  - id: gpt-4.1-mini",
                "    provider: openai",
                "    max_calls_per_min: 60",
                "    price_per_1k_tokens: 0.001",
                "    temperature: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    orchestrator.run(
        cfg_path=cfg_path,
        dry_run=True,
        model_ids=["gpt-4.1-mini"],
        dataset_names=["tiny"],
        data_dir=data_dir,
        log_dir=log_dir,
        results_dir=results_dir,
        summarize=False,
    )

    assert (results_dir / "scores.csv").exists()
    assert (results_dir / "run_metadata.jsonl").exists()
    assert (data_dir / "gpt-4.1-mini_tiny.dryrun.json").exists()


def test_generated_dataset_uses_data_dir(tmp_path: Path):
    data_dir = tmp_path / "out_data"
    log_dir = tmp_path / "out_logs"
    results_dir = tmp_path / "out_results"
    cfg_path = tmp_path / "bench.yaml"
    configured_dataset_path = tmp_path / "configured" / "tiny.jsonl"

    cfg_path.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: tiny",
                "    gen:",
                "      mode: 1d",
                "      n: 4",
                "      size: 8",
                "      timesteps: 1",
                "      density: 0.5",
                "      seed: 1",
                f"    path: {configured_dataset_path}",
                "    dim: 1",
                "models:",
                "  - id: gpt-4.1-mini",
                "    provider: openai",
                "    max_calls_per_min: 60",
                "    price_per_1k_tokens: 0.001",
                "    temperature: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    orchestrator.run(
        cfg_path=cfg_path,
        dry_run=True,
        model_ids=["gpt-4.1-mini"],
        dataset_names=["tiny"],
        data_dir=data_dir,
        log_dir=log_dir,
        results_dir=results_dir,
        summarize=False,
    )

    assert (data_dir / "tiny.jsonl").exists()
    assert not configured_dataset_path.exists()
    assert (data_dir / "gpt-4.1-mini_tiny.dryrun.json").exists()
