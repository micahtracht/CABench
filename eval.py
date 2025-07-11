from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List, Sequence, Union

def _flatten(obj: Union[str, Sequence]) -> str:
    """
    Convert the target—whether a binary string, a 1-D list, or a 2-D nested
    list—into a flat binary string (row-major order).  Raises ValueError on
    invalid symbols.
    """
    if isinstance(obj, str):
        flat = obj
    elif isinstance(obj, Sequence):
        # recurse over arbitrary depth; collect scalars
        bits: List[str] = []
        def _collect(x):
            if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
                for y in x:
                    _collect(y)
            else:
                bits.append(str(x))
        _collect(obj)
        flat = "".join(bits)
    else:
        raise ValueError("target type not recognised")

    if not all(c in "01" for c in flat):
        raise ValueError("non-binary symbol found")
    return flat

def normalized_hamming_accuracy(correct: str, response: str) -> float:
    '''
    1.0 means perfect match, 0.0 means nothing was correct. Lower = proportion of mismatches."
    '''
    if not correct:
        return 0.0
    
    hamming_distance = sum(c != r for c, r in zip(correct, response))
    hamming_distance += abs(len(correct) - len(response)) # responses too long or too short have all excess/nonexistent characters counted as incorrect
    
    return 1.0 - (hamming_distance/max(len(correct), len(response))) # in [0, 1]

def evaluate(gold_path: pathlib.Path, pred_path: pathlib.Path) -> None:
    gold_lines = gold_path.read_text(encoding="utf-8").splitlines()
    pred_lines = pred_path.read_text(encoding="utf-8").splitlines()

    if len(gold_lines) != len(pred_lines):
        print(
            f"Mismatch: {len(pred_lines)} predictions vs {len(gold_lines)} gold.",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(gold_lines)
    sum_acc = 0.0
    exact_match = 0
    invalid = 0

    for idx, (gline, pred_raw) in enumerate(zip(gold_lines, pred_lines), 1):
        correct_obj = json.loads(gline)["target"]
        try:
            gold_flat = _flatten(correct_obj)
        except ValueError as e:
            print(f"Gold line {idx}: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            pred_flat = _flatten(pred_raw.strip())     # pred already string
        except ValueError:
            invalid += 1
            pred_flat = ""                             # treat as fully wrong

        acc = normalized_hamming_accuracy(gold_flat, pred_flat)
        sum_acc += acc
        if pred_flat == gold_flat:
            exact_match += 1

    if invalid:
        print(f"-Found {invalid} invalid prediction lines (non-binary symbols).", file=sys.stderr)

    print(f"-Evaluated {total} cases")
    print(f"-Normalized Hamming accuracy: {sum_acc / total:.4f}")
    print(f"-Exact-match accuracy: {exact_match}/{total} = {(exact_match / total) * 100:.2f}%")
    sys.exit(0)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score CA predictions (1-D or 2-D) against a gold JSONL."
    )
    p.add_argument("--gold", type=pathlib.Path, required=True, help="Gold JSONL with a 'target' field.")
    p.add_argument("--pred", type=pathlib.Path, required=True, help="Predictions file (one line per case).")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args.gold, args.pred)


if __name__ == "__main__":
    main()
