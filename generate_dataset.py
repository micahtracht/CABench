"""
generate_dataset.py

Create a JSONL file of 1-D outer-totalistic ECA tasks.

Example
-------
python generate_dataset.py
       --n 2048 --size 32 --timesteps 4
       --density 0.4 --seed 123
       --outfile data/train.jsonl
"""

from __future__ import annotations
import argparse, json, pathlib, sys
from typing import List
import numpy as np
from generate import ECAProblemGenerator, Problem1D
from simulate import simulate

def problem_to_jsonl(problem: Problem1D) -> str:
    """
    Serialize a Problem1D (+ ground truth) as one JSON line.
    """
    target = "".join(map(str, simulate(problem.start_state, problem.rule, problem.timesteps)))
    return json.dumps(
        {
            "rule": problem.rule.rule,
            "timesteps": problem.timesteps,
            "init": "".join(map(str, problem.start_state)),
            "target": target,
        },
        separators=(",", ":"),
    )

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a JSONL file of 1-D ECA tasks.")
    
    p.add_argument("--n", type=int, required=True, help="Number of problems to generate.")
    p.add_argument("--size", type=int, default=32, help="Length of the 1-D lattice.")
    p.add_argument("--timesteps", type=int, default=4, help="Evolution steps per problem.")
    p.add_argument("--density", type=float, default=0.5, help="Probability a cell is alive in the initial state.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    p.add_argument("--outfile", type=pathlib.Path, required=True, help="Where to write the JSONL.")
    return p

def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    gen = ECAProblemGenerator(
        state_size=args.size,
        seed=args.seed,
        density=args.density,
    )
    batch = gen.generate_batch(
        num_problems=args.n,
        timesteps=args.timesteps, # int allowed
        trim_trivial=True,
    )
    
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as f:
        for prob in batch:
            f.write(problem_to_jsonl(prob) + "\n")

    print(f"Wrote {len(batch):,} problems to {args.outfile}")

if __name__ == "__main__":
    main()