"""
generate_dataset.py

Create a JSONL file of 1-D or 2-D outer-totalistic CA tasks.

Example (1-D)
-------
python generate_dataset.py --mode 1d \
       --n 2048 --size 32 --timesteps 4 \
       --density 0.4 --seed 123 \
       --outfile data/train1d.jsonl

Example (2-D)
-------
python generate_dataset.py --mode 2d \
       --n 128 --height 16 --width 16 --timesteps 1 \
       --density 0.5 --seed 42 \
       --outfile data/val2d_public.jsonl
"""

from __future__ import annotations
import argparse, json, pathlib, sys
from typing import List
import numpy as np
from generate import CAProblemGenerator2D, ECAProblemGenerator, Problem1D, Problem2D
from simulate import simulate, simulate_2d

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


def problem2d_to_jsonl(problem: Problem2D) -> str:
    """
    Serialize a 2-D CA task (with ground truth) into one compact JSON record.
    Grids are kept as nested lists to preserve shape information.
    """
    target = simulate_2d(problem.start_grid, problem.rule, problem.timesteps)
    return json.dumps(
        {
            "rule": problem.rule.rule_bits,
            "timesteps": problem.timesteps,
            "init": problem.start_grid,
            "target": target,
        },
        separators=(",", ":"),
    )


def build_parser_2d() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a JSONL file of 2-D outer-totalistic CA tasks."
    )
    p.add_argument("--n", type=int, required=True, help="Number of problems.")
    p.add_argument("--height", type=int, default=16, help="Grid height.")
    p.add_argument("--width", type=int, default=16, help="Grid width.")
    p.add_argument("--timesteps", type=int, default=1, help="Steps per problem.")
    p.add_argument(
        "--density", type=float, default=0.5, help="Probability a cell starts alive."
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument("--outfile", type=pathlib.Path, required=True, help="Output JSONL.")
    return p


def main_2d(argv: List[str] | None = None) -> None:
    """
    CLI entry point for 2-D dataset generation.
    """
    args = build_parser_2d().parse_args(argv)

    gen = CAProblemGenerator2D(
        height=args.height,
        width=args.width,
        seed=args.seed,
        density=args.density,
    )
    batch = gen.generate_batch(
        num_problems=args.n, timesteps=args.timesteps, trim_trivial=True
    )

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with args.outfile.open("w", encoding="utf-8") as f:
        for prob in batch:
            f.write(problem2d_to_jsonl(prob) + "\n")

    print(f"Wrote {len(batch):,} 2-D problems to {args.outfile}")


def dispatch_main(argv: List[str] | None = None) -> None:
    """
    Dispatcher that invokes either the 1-D or 2-D generator based on --mode.
    """
    # parse only --mode, leave the rest of arguments for the specific main
    top = argparse.ArgumentParser(add_help=False)
    top.add_argument(
        "--mode",
        choices=["1d", "2d"],
        default="1d",
        help="Generate 1-D tasks or 2-D tasks",
    )
    args, remaining = top.parse_known_args(argv)
    if args.mode == "2d":
        main_2d(remaining)
    else:
        main(remaining)


if __name__ == "__main__":
    dispatch_main()
