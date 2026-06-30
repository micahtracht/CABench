# CABench Agent Notes

## What This Repo Is

CABench benchmarks LLM spatial/working-memory reasoning using cellular automata (CA).
Each benchmark item gives a model:
- an initial CA state,
- transition rules in plain language,
- a timestep count,
and asks for the final state.

Primary benchmark modes:
- 1D outer-totalistic binary CA (left/right neighbors)
- 2D outer-totalistic binary CA (Moore neighborhood)

Core objective:
- measure prediction quality with objective metrics,
- while logging token/cost usage per call and per run.

## Package Layout

All code lives in the installable `cabench` package, driven by the single `cabench`
CLI (`python -m cabench`). There are no loose top-level modules and no `wrappers/`
directory anymore.

- `cabench/cli.py`: CLI entry — subcommands `run`, `report`, `list-presets`,
  `generate`, `eval`, `convert`, `migrate`
- `cabench/orchestrator.py`: one-command runner (generate -> predict -> convert ->
  eval -> log), all **in-process** (no subprocess shelling)
- `cabench/dataset.py`: CLI for 1D/2D dataset generation (JSONL)
- `cabench/generate.py`: CA problem generators + prompt builders
- `cabench/rules.py`: CA rule encodings (`Rule1D`, `Rule2D`)
- `cabench/simulate.py`: CA update/simulation logic
- `cabench/llm/runner.py`: unified 1D/2D OpenAI runner — one `run_batch(dim=...)`
  with retries, JSON-mode parsing, usage logging, and resume reconciliation
- `cabench/llm/rate_limit.py`, `cabench/llm/response_logger.py`: per-call pacing and
  raw-response logging helpers
- `cabench/json_extract.py`: `extract_answer_json` — pulls the answer object from a
  reply, preferring the **last** object carrying an `answer`/`final_state` key
- `cabench/convert.py`: structured model JSONL -> flattened `.preds`
- `cabench/scoring.py`: normalized Hamming + exact-match scoring (`score`, `EvalError`)
- `cabench/report.py`: per-model and per-dataset comparison reports
- `cabench/contracts.py`: canonical artifact schema names/versions + manifest writer
- `cabench/migrate.py`: legacy-artifact migration to current schemas
- `bench.yaml`: dataset/model config
- `tests/`: unit + orchestration + mocked end-to-end tests
- `.github/workflows/`: CI test and dry-run workflows

## High-Level Pipeline

Entry point: `python -m cabench run`. For each dataset/model pair (from `bench.yaml`
or a `--preset`):
1. Generate dataset JSONL if needed (`cabench.dataset`, via the `gen` block)
2. Query the model: `cabench.llm.runner.run_batch(model=..., dim=1|2, ...)`
3. Convert structured JSON responses to flat prediction lines: `cabench.convert.convert_file`
4. Score predictions against gold targets: `cabench.scoring.score`
5. Append run metrics to `results/scores.csv`
6. Merge per-call usage into `logs/master_usage.csv`
7. Append a reproducibility record to `results/run_metadata.jsonl`

## Data Contracts

Canonical schema/version definitions live in `cabench/contracts.py`. Producers write
sidecar manifests next to each artifact (`<artifact>.schema.json` /
`<artifact>.<suffix>.schema.json`). See `SCHEMA_POLICY.md` for the evolution rules.

Current schema IDs (v1): `cabench.results.scores`, `cabench.logs.usage`,
`cabench.predictions.jsonl`, `cabench.predictions.bits`, `cabench.results.run_metadata`.

### Dataset JSONL
- 1D records: `rule` (6-bit string), `timesteps` (int), `init` (binary string),
  `target` (binary string, same length as `init`)
- 2D records: `rule` (18-bit string), `timesteps` (int), `init` (nested grid),
  `target` (nested grid, same shape as `init`)

### Prediction Files
- Runner output JSONL: one JSON object per sample, with `answer` (or `final_state` alias)
- Converted predictions: one flat binary string per line in `.preds`
- Line alignment matters: conversion writes a blank line for bad/empty rows to preserve
  index alignment with gold

## Scoring (`cabench.scoring`)
- Normalized Hamming accuracy (mean over examples) + exact-match accuracy
- Gold `target` may be a string or nested sequences (flattened row-major)
- Blank/non-binary prediction lines are counted as **invalid** and scored fully wrong
- Empty inputs or malformed/blank gold lines raise `EvalError` (CLI exits 1)
- Prediction/gold line-count mismatch raises `EvalError`

## Triviality (`cabench.generate`)
`is_trivial` judges over the **full** timestep horizon: a problem is trivial if its
start is all-dead/all-alive, or if it evolves back to the start after `timesteps`
(the answer would just copy the visible input).

## Config (`bench.yaml`)
- `datasets`: named datasets with optional generation params (`gen`) and dimensionality
  (`dim`)
- `models`: model IDs, provider, `max_calls_per_min`, blended `price_per_1k_tokens`,
  temperature. Current models are the `gpt-5.4` family (`gpt-5.4-nano`, `gpt-5.4-mini`,
  `gpt-5.4`). `price_per_1k_tokens` is a 50/50 input+output blended approximation.
- `presets`: named bundles of datasets + models (run with `--preset <name>`)

`cabench run` flags include `--config`, `--preset`, `--model-id`, `--models`,
`--datasets`, `--dim`, `--new`, `--file`, `--numquestions`, `--dry-run`,
`--force-preds`, `--spend-cap` (default $1.00), `--data-dir`/`--log-dir`/`--results-dir`,
`--no-summary`, `--no-report`.

## Environment & Dependencies
- Python target: 3.14 (`pyproject.toml` requires `>=3.14`; CI runs 3.14)
- Runtime deps (`requirements.txt` / `pyproject.toml`): `numpy`, `openai`, `pyyaml`,
  `backoff`, `python-dotenv`; `pytest` for tests
- `.env` is auto-loaded if present (never overrides already-exported vars);
  `OPENAI_API_KEY` is required for live runs

## Practical Run Commands
```bash
python -m pip install -r requirements.txt          # or: pip install -e .
python -m cabench run --config bench.yaml           # full run
python -m cabench run --dry-run --numquestions 8    # fast smoke, no API
python -m pytest -q                                 # tests
python -m cabench report --scores results/scores.csv --out results/report.txt
```

## Operational Notes for Agents
- Keep dataset/prediction line alignment intact; downstream scoring depends on it.
- Prefer preserving existing run artifacts unless explicitly forcing regeneration.
- Keep cost logging intact; this benchmark is both quality- and cost-sensitive.
- Resume is checkpointed on the consistent prefix of `(predictions, usage)` rows; the
  runner auto-repairs corrupted tails before continuing.
- The orchestrator merges only newly added usage rows into `logs/master_usage.csv` to
  avoid duplicate ledger entries, and appends one reproducibility record per run to
  `results/run_metadata.jsonl` (dataset/config hashes, git state, model params,
  timestamps, artifact paths).
- The runner is OpenAI-only today (JSON mode). Adding a provider means generalizing
  `run_batch`/`chat_json` in `cabench/llm/runner.py` (no abstraction seam yet).
