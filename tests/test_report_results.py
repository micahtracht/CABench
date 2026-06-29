from pathlib import Path

from cabench.report import (
    aggregate_by_model,
    load_run_metadata,
    load_scores,
    metric_rows_from_metadata,
    render_report,
)


def test_load_scores_skips_dry_run_rows(tmp_path: Path):
    p = tmp_path / "scores.csv"
    p.write_text(
        "\n".join(
            [
                "date_utc,dataset,model,norm_hamming,exact_pct,total_cost_usd,pred_file",
                "2026-01-01T00:00:00+00:00,quick1d_32,gpt-4.1-mini,0.5000,10.00,0.1000,a.preds",
                "2026-01-01T00:00:01+00:00,quick1d_32,gpt-4.1-mini,,,0.0000,DRY_RUN:x",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_scores(p)
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-4.1-mini"


def test_aggregate_by_model(tmp_path: Path):
    p = tmp_path / "scores.csv"
    p.write_text(
        "\n".join(
            [
                "date_utc,dataset,model,norm_hamming,exact_pct,total_cost_usd,pred_file",
                "t,d1,m1,0.5,10,0.1,x",
                "t,d2,m1,0.7,30,0.2,x",
                "t,d1,m2,0.6,20,0.4,x",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_scores(p)
    agg = aggregate_by_model(rows)
    assert agg[0]["model"] == "m1"
    assert agg[0]["runs"] == 2
    assert round(agg[0]["avg_norm_hamming"], 4) == 0.6
    assert round(agg[0]["avg_exact_pct"], 2) == 20.0
    assert round(agg[0]["total_cost_usd"], 4) == 0.3


def test_render_report_contains_sections(tmp_path: Path):
    p = tmp_path / "scores.csv"
    p.write_text(
        "\n".join(
            [
                "date_utc,dataset,model,norm_hamming,exact_pct,total_cost_usd,pred_file",
                "t,d1,m1,0.5,10,0.1,x",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = render_report(load_scores(p))
    assert "Topline Summary" in report
    assert "Headline Metrics By Model" in report
    assert "Headline Metrics By Dataset" in report
    assert "m1" in report


def test_render_report_includes_latest_invocation(tmp_path: Path):
    scores = tmp_path / "scores.csv"
    metadata = tmp_path / "run_metadata.jsonl"
    scores.write_text(
        "\n".join(
            [
                "date_utc,dataset,model,norm_hamming,exact_pct,total_cost_usd,pred_file",
                "2026-01-01T00:00:00+00:00,d1,m1,0.5,10,0.1,a.preds",
                "2026-01-01T00:00:01+00:00,d2,m1,0.7,30,0.2,b.preds",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    metadata.write_text(
        "\n".join(
            [
                '{"invocation_id":"run-1","invocation_started_utc":"2026-01-01T00:00:00+00:00","run_finished_utc":"2026-01-01T00:00:10+00:00","dataset":{"name":"d1","dim":1,"num_cases":16},"model":{"id":"m1"},"metrics":{"norm_hamming":0.5,"exact_pct":10.0},"cost_usd":0.1,"status":"completed","config":{"cli_overrides":{"spend_cap_usd":1.0}}}',
                '{"invocation_id":"run-1","invocation_started_utc":"2026-01-01T00:00:00+00:00","run_finished_utc":"2026-01-01T00:00:12+00:00","dataset":{"name":"d2","dim":2,"num_cases":8},"model":{"id":"m1"},"metrics":{"norm_hamming":0.7,"exact_pct":30.0},"cost_usd":0.2,"status":"completed","config":{"cli_overrides":{"spend_cap_usd":1.0}}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = render_report(load_scores(scores), load_run_metadata(metadata))
    assert "Latest invocation" in report
    assert "spend_cap_usd" in report
    assert "cost_per_case" in report
    assert "Topline Summary" in report
    assert "Headline Metrics By Model" in report
    assert "d1" in report
    assert "d2" in report


def test_metric_rows_from_metadata_enables_cost_per_case():
    rows = metric_rows_from_metadata(
        [
            {
                "dataset": {"name": "d1", "num_cases": 10},
                "model": {"id": "m1"},
                "metrics": {"norm_hamming": 0.5, "exact_pct": 20.0},
                "cost_usd": 0.1,
                "run_finished_utc": "t1",
            },
            {
                "dataset": {"name": "d2", "num_cases": 20},
                "model": {"id": "m1"},
                "metrics": {"norm_hamming": 0.7, "exact_pct": 40.0},
                "cost_usd": 0.2,
                "run_finished_utc": "t2",
            },
        ]
    )

    agg = aggregate_by_model(rows)
    assert round(agg[0]["cost_per_case"], 4) == 0.01
