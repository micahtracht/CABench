# CABench
*A lightweight benchmark for spatial reasoning in LLMs using 1D and 2D cellular automata.*

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.14-blue.svg)

---

## Why CA Bench?
LLMs still fail to reason spatially in an effective way. Furthermore, coming up with easily verifiable, mass produced questions that isolate these abilities is also not trivial. But by using cellular automata, we can:
- Isolate short term "working memory" & multi-step inference challenges,
- Vary dimensionality without changing scoring,
- Generate millions of unique tasks on the fly, removing contamination or data leakage concerns.

---

## Repo Layout

All code lives in the installable `cabench` package, driven by the single `cabench` CLI (`python -m cabench`).

- `cabench/cli.py`: the `cabench` CLI — `run`, `report`, `list-presets`, `generate`, `eval`, `convert`, `migrate`
- `cabench/orchestrator.py`: one-command benchmark runner (generate -> predict -> convert -> eval -> log), in-process
- `bench.yaml`: dataset/model configuration
- `cabench/dataset.py`: generate 1D/2D CA datasets (JSONL)
- `cabench/generate.py`, `cabench/rules.py`, `cabench/simulate.py`: CA problem/rule/simulation logic
- `cabench/llm/runner.py`: unified 1D/2D OpenAI runner with retries/status logging
- `cabench/convert.py`: structured model JSONL -> flattened `.preds`
- `cabench/scoring.py`: normalized Hamming + exact-match scoring
- `cabench/report.py`: per-model and per-dataset comparison reports
- `cabench/contracts.py`: canonical artifact schema names/versions
- `results/`: benchmark outputs (`scores.csv`, `run_metadata.jsonl`)
- `logs/`: per-call usage and master usage ledger
- `tests/`: unit + orchestration + mocked end-to-end tests

---

## Quick start

```bash
# clone and install dependencies
git clone https://github.com/micahtracht/CABench.git
cd CABench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set your OpenAI API key in a local .env file (recommended)
cp .env.example .env
# edit .env and replace the placeholder key

# or export it in your shell
export OPENAI_API_KEY=sk-...

# run the default benchmark
python -m cabench run
```
The `cabench` console script is also installed (`pip install -e .`), so `cabench run` works too.
CABench automatically loads a repo-local `.env` file if present and will not override variables you already exported in your shell. `.env` is gitignored; `.env.example` is the committed template.

Expected artifacts:
 - data/<model>_<dataset>.jsonl – model replies (JSON)

 - data/<model>_<dataset>.preds – flattened predictions

 - logs/<model>_<dataset>_usage.csv – tokens + cost

 - results/scores.csv – one row per model/dataset run

 - results/run_metadata.jsonl – reproducibility metadata per run

 - logs/master_usage.csv – merged per-call usage ledger across runs

 - *.schema.json sidecars – schema/version manifests for key artifacts

## Run CLI

`cabench run` is the primary interface.

```bash
python -m cabench run \
  --config bench.yaml \
  --models gpt-5.4-mini,gpt-5.4 \
  --datasets bench1d_32,bench2d_32x32 \
  --dim 1 \
  --numquestions 128 \
  --data-dir out/data \
  --log-dir out/logs \
  --results-dir out/results
```

Useful flags:
- `--preset <name>`: run a named preset from `bench.yaml` (e.g. `quick`, `presentation`)
- `--model-id <id>`: run one model
- `--models a,b,c`: run multiple models
- `--datasets a,b,c`: run selected datasets
- `--dim 1|2`: select dimensionality
- `--numquestions N`: limit question count
- `--new`: force dataset regeneration
- `--force-preds`: reset existing prediction/usage files for fresh run
- `--dry-run`: skip API calls and API-dependent postprocessing (conversion/eval); writes dry-run markers
- `--spend-cap <usd>`: cap total actual spend for the run (default $1.00)
- `--data-dir`, `--log-dir`, `--results-dir`: custom output roots
- `--no-summary`: disable end-of-run summary table
- `--no-report`: skip writing the end-of-run report

## Bench.yaml cheat sheet
```yaml
datasets:
  - name: bench1d_32
    gen:                 # auto-generate if file absent
      mode: 1d
      n: 128             # number of tasks
      size: 32           # 1-D length
      timesteps: 4
      density: 0.5
      seed: 123
    path: data/bench1d_32.jsonl
    dim: 1               # picks the 1-D wrapper

models:
  - id: gpt-5.4-mini
    provider: openai
    max_calls_per_min: 60
    price_per_1k_tokens: 0.002625  # blended 50/50 input+output estimate
    temperature: 0
```

## Common Tweaks

- Fast smoke test (no API):
```bash
python -m cabench run --dry-run --numquestions 8 --models gpt-5.4-mini
```

- Run only 2D datasets:
```bash
python -m cabench run --dim 2
```

- Run one model on one dataset:
```bash
python -m cabench run --model-id gpt-5.4-mini --datasets bench1d_32
```

- Keep outputs separate per experiment:
```bash
python -m cabench run --data-dir runs/r1/data --log-dir runs/r1/logs --results-dir runs/r1/results
```

## Results

Generate comparison tables from accumulated scores:

```bash
python -m cabench report --scores results/scores.csv --out results/report.txt
```

Report flags:
- `--results-dir <dir>`: results root the other paths derive from (default `results/`)
- `--scores <path>`: scores CSV to read (default `<results-dir>/scores.csv`)
- `--metadata <path>`: run-metadata JSONL to enrich the report (default `<results-dir>/run_metadata.jsonl`)
- `--latest-only`: show only the latest invocation summary
- `--out <path>`: optional path to write the rendered report

Report includes:
- per-model aggregate ranking (`avg_norm_hamming`, `avg_exact_pct`, `total_cost_usd`)
- per-dataset model comparison table

## Schema Evolution & Migration

Schema/version contracts are defined in `cabench/contracts.py` and governed by `SCHEMA_POLICY.md`.

If you have legacy artifacts, migrate them before running strict schema writers:

```bash
python -m cabench migrate --results-dir results --log-dir logs
```

This upgrades/normalizes:
- `results/scores.csv`
- `logs/master_usage.csv` and per-run `*_usage.csv`
- `results/run_metadata.jsonl`

## Citation
@misc{CABench2025,
  title  = {CA-Bench: Evaluating Spatial & Working-Memory Reasoning in LLMs},
  author = {Micah Tracht},
  year   = {2025},
  url    = {https://github.com/micahtracht/CABench}
}

## License
This project is licensed under the MIT License—see LICENSE for details.
