from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from cabench import convert, dataset, migrate, orchestrator, report, scoring
from cabench.llm.runner import RunnerError
from cabench.orchestrator import OrchestratorError
from cabench.scoring import EvalError


DEFAULT_CONFIG = Path("bench.yaml")


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return data


def available_presets(spec: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_names = [str(ds.get("name", "")).strip() for ds in spec.get("datasets", []) if ds.get("name")]
    model_ids = [str(m.get("id", "")).strip() for m in spec.get("models", []) if m.get("id")]
    default_preset = {
        "name": "default",
        "description": "Run all datasets and models from the config.",
        "datasets": dataset_names,
        "models": model_ids,
    }

    seen = {"default"}
    presets = [default_preset]
    for raw in spec.get("presets", []) or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name or name in seen:
            continue
        presets.append(
            {
                "name": name,
                "description": str(raw.get("description", "")).strip(),
                "datasets": [str(x).strip() for x in raw.get("datasets", []) if str(x).strip()],
                "models": [str(x).strip() for x in raw.get("models", []) if str(x).strip()],
            }
        )
        seen.add(name)
    return presets


def get_preset(spec: dict[str, Any], preset_name: str) -> dict[str, Any]:
    for preset in available_presets(spec):
        if preset["name"] == preset_name:
            return preset
    valid = ", ".join(p["name"] for p in available_presets(spec))
    raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {valid}")


def _merge_model_selection(model_id: str | None, model_ids_csv: str | None) -> list[str] | None:
    selected = orchestrator.parse_csv_arg(model_ids_csv)
    if model_id:
        selected = (selected or []) + [model_id]
    if not selected:
        return None
    return sorted(set(selected))


def _resolve_run_selection(
    spec: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], list[str] | None, list[str] | None]:
    preset = get_preset(spec, args.preset)
    selected_models = _merge_model_selection(args.model_id, args.models)
    selected_datasets = orchestrator.parse_csv_arg(args.datasets)

    if selected_models is None:
        selected_models = preset["models"] or None
    if selected_datasets is None:
        selected_datasets = preset["datasets"] or None

    return preset, selected_models, selected_datasets


def _latest_report_text(results_dir: Path) -> str:
    scores_path = results_dir / "scores.csv"
    metadata_path = results_dir / "run_metadata.jsonl"
    score_rows = report.load_scores_if_present(scores_path)
    metadata_rows = report.load_run_metadata(metadata_path)
    return report.render_report(score_rows, metadata_rows, latest_only=True)


def _write_latest_report(results_dir: Path, report_text: str) -> Path:
    out_path = results_dir / "latest_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text + "\n", encoding="utf-8")
    return out_path


def run_command(args: argparse.Namespace) -> int:
    spec = load_config(args.config)
    try:
        preset, selected_models, selected_datasets = _resolve_run_selection(spec, args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Running preset '{preset['name']}' with spend cap ${args.spend_cap:.2f}. "
        f"datasets={selected_datasets or 'all'} models={selected_models or 'all'}"
    )
    orchestrator.run(
        cfg_path=args.config,
        dry_run=args.dry_run,
        model_id=args.model_id,
        model_ids=selected_models,
        dataset_names=selected_datasets,
        dim=args.dim,
        force_new=args.new,
        file=args.file,
        num_questions=args.numquestions,
        force_preds=args.force_preds,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        results_dir=args.results_dir,
        summarize=not args.no_summary,
        spend_cap_usd=args.spend_cap,
    )

    if args.no_report:
        return 0

    report_text = _latest_report_text(args.results_dir)
    out_path = _write_latest_report(args.results_dir, report_text)
    print()
    print(report_text)
    print(f"\nWrote latest report to {out_path}")
    return 0


def report_command(args: argparse.Namespace) -> int:
    scores_path = args.scores or (args.results_dir / "scores.csv")
    metadata_path = args.metadata or (args.results_dir / "run_metadata.jsonl")
    score_rows = (
        report.load_scores_if_present(scores_path)
        if args.latest_only
        else report.load_scores(scores_path)
    )
    metadata_rows = report.load_run_metadata(metadata_path)
    report_text = report.render_report(
        score_rows,
        metadata_rows,
        latest_only=args.latest_only,
    )
    print(report_text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report_text + "\n", encoding="utf-8")
        print(f"\nWrote report to {args.out}")
    return 0


def list_presets_command(args: argparse.Namespace) -> int:
    spec = load_config(args.config)
    presets = available_presets(spec)
    print("Available presets:")
    for preset in presets:
        desc = f" - {preset['description']}" if preset["description"] else ""
        print(
            f"{preset['name']}: datasets={len(preset['datasets'])} "
            f"models={len(preset['models'])}{desc}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CABench CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a benchmark preset and emit a latest report.")
    run_p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to bench.yaml")
    run_p.add_argument("--preset", default="default", help="Preset name from bench.yaml")
    run_p.add_argument("--model-id", help="Run a single model ID")
    run_p.add_argument("--models", help="Comma-separated model IDs")
    run_p.add_argument("--datasets", help="Comma-separated dataset names")
    run_p.add_argument("--dim", type=int, choices=[1, 2], help="Filter to one dimension")
    run_p.add_argument("--new", action="store_true", help="Regenerate datasets if config includes gen blocks")
    run_p.add_argument("--file", type=Path, help="Override dataset path")
    run_p.add_argument("--numquestions", type=int, help="Limit question count")
    run_p.add_argument("--dry-run", action="store_true", help="Skip API calls")
    run_p.add_argument("--force-preds", action="store_true", help="Overwrite prediction and usage files")
    run_p.add_argument("--data-dir", type=Path, default=orchestrator.DATA_DIR)
    run_p.add_argument("--log-dir", type=Path, default=orchestrator.LOG_DIR)
    run_p.add_argument("--results-dir", type=Path, default=orchestrator.RESULT_DIR)
    run_p.add_argument("--spend-cap", type=float, default=orchestrator.HARD_SPEND_CEILING)
    run_p.add_argument("--no-summary", action="store_true", help="Disable orchestrator summary table")
    run_p.add_argument("--no-report", action="store_true", help="Skip writing and printing the latest report")
    run_p.set_defaults(func=run_command)

    report_p = sub.add_parser("report", help="Render benchmark reports from saved results.")
    report_p.add_argument("--results-dir", type=Path, default=orchestrator.RESULT_DIR)
    report_p.add_argument("--scores", type=Path, help="Explicit scores.csv path")
    report_p.add_argument("--metadata", type=Path, help="Explicit run_metadata.jsonl path")
    report_p.add_argument("--latest-only", action="store_true", help="Show only the latest invocation summary")
    report_p.add_argument("--out", type=Path, help="Optional output path for the rendered report")
    report_p.set_defaults(func=report_command)

    list_p = sub.add_parser("list-presets", help="List benchmark presets from the config.")
    list_p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to bench.yaml")
    list_p.set_defaults(func=list_presets_command)

    # The PASSTHROUGH tools are registered for discoverability; main()
    # forwards their raw args to each tool's own parser before argparse runs.
    sub.add_parser("generate", help="Generate a 1-D/2-D CA dataset (see --mode).", add_help=False)
    sub.add_parser("eval", help="Score predictions against a gold JSONL.", add_help=False)
    sub.add_parser("convert", help="Flatten structured JSONL predictions to bit strings.", add_help=False)
    sub.add_parser("migrate", help="Migrate legacy artifacts to canonical schemas.", add_help=False)

    return parser


# Subcommands whose flags are owned by the underlying tool's own argument
# parser. main() dispatches these before argparse so their options are not
# intercepted by the top-level parser.
PASSTHROUGH = {
    "generate": dataset.dispatch_main,
    "eval": scoring.main,
    "convert": convert.main,
    "migrate": migrate.main,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in PASSTHROUGH:
        PASSTHROUGH[argv[0]](argv[1:])
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (OrchestratorError, RunnerError, EvalError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
