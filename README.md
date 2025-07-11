# CABench
*A lightweight benchmark for spatial reasoning in LLMs using 1D and 2D cellular automata.*

![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)

---

## Why CA Bench?
LLMs still fail to reason spatially in an effective way. Furthermore, coming up with easily verifiable, mass produced questions that isolate these abilities is also not trivial. But by using cellular automata, we can:
-Isolate short term "working memory" & multi-step inference challenges,
-Vary dimensionality without changing scoring,
-Generate millions of unique tasks on the fly, removing contamination or data leakage concerns.

---

## Repo Layout

TBD

---

## Quick start

```bash
# clone and install dependencies
git clone https://github.com/micahtracht/CABench.git
cd CABench
python -m pip install requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# run the default benchmark
python orchestrator.py # can add --dry-run to avoid API calls
```
Expected artifacts:
 - data/<model>_<dataset>.jsonl – model replies (JSON)

 - data/<model>_<dataset>.preds – flattened predictions

 - logs/<model>_<dataset>_usage.csv – tokens + cost

 - results/scores.csv – one row per model/dataset run

## Bench.yaml cheat sheet
```yaml
datasets:
  - name: quick1d_32
    gen:                 # auto-generate if file absent
      mode: 1d
      n: 256             # number of tasks
      size: 32           # 1-D length
      timesteps: 4
      density: 0.4
      seed: 123
    path: data/quick1d_32.jsonl
    dim: 1               # picks the 1-D wrapper

models:
  - id: gpt-4.1-mini
    provider: openai
    max_calls_per_min: 60
    price_per_1k_tokens: 0.001
    temperature: 0
```

## Common Tweaks

TBD

## Results

TBD (results not obtained yet)

## Citation
@misc{CABench2025,
  title  = {CA-Bench: Evaluating Spatial & Working-Memory Reasoning in LLMs},
  author = {Micah Tracht},
  year   = {2025},
  url    = {https://github.com/micahtracht/CABench}
}

## License
This project is licensed under the MIT License—see LICENSE for details.
