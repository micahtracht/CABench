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


class EvalError(ValueError):
    """Raised when gold/prediction inputs cannot be scored."""


def score(gold_path: pathlib.Path, pred_path: pathlib.Path) -> dict:
    """
    Score predictions against gold and return aggregate metrics.

    Returns a dict with keys: total, norm_hamming, exact_match, exact_pct, invalid.
    Raises EvalError on length mismatch or malformed gold targets.
    """
    gold_lines = gold_path.read_text(encoding="utf-8").splitlines()
    pred_lines = pred_path.read_text(encoding="utf-8").splitlines()

    if len(gold_lines) != len(pred_lines):
        raise EvalError(
            f"Mismatch: {len(pred_lines)} predictions vs {len(gold_lines)} gold."
        )

    total = len(gold_lines)
    if total == 0:
        raise EvalError("no rows to score (empty gold/prediction files).")

    sum_acc = 0.0
    exact_match = 0
    invalid = 0

    for idx, (gline, pred_raw) in enumerate(zip(gold_lines, pred_lines), 1):
        try:
            correct_obj = json.loads(gline)["target"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise EvalError(f"Gold line {idx}: invalid JSON or missing 'target'") from e
        try:
            gold_flat = _flatten(correct_obj)
        except ValueError as e:
            raise EvalError(f"Gold line {idx}: {e}") from e

        pred_str = pred_raw.strip()
        if pred_str == "":
            invalid += 1
            pred_flat = ""                             # blank/missing prediction
        else:
            try:
                pred_flat = _flatten(pred_str)         # pred already string
            except ValueError:
                invalid += 1
                pred_flat = ""                         # treat as fully wrong

        acc = normalized_hamming_accuracy(gold_flat, pred_flat)
        sum_acc += acc
        if pred_flat == gold_flat:
            exact_match += 1

    return {
        "total": total,
        "norm_hamming": sum_acc / total,
        "exact_match": exact_match,
        "exact_pct": (exact_match / total) * 100,
        "invalid": invalid,
    }


def evaluate(gold_path: pathlib.Path, pred_path: pathlib.Path) -> None:
    try:
        result = score(gold_path, pred_path)
    except EvalError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if result["invalid"]:
        print(
            f"-Found {result['invalid']} invalid prediction lines (blank or non-binary symbols).",
            file=sys.stderr,
        )

    print(f"-Evaluated {result['total']} cases")
    print(f"-Normalized Hamming accuracy: {result['norm_hamming']:.4f}")
    print(
        f"-Exact-match accuracy: {result['exact_match']}/{result['total']} = "
        f"{result['exact_pct']:.2f}%"
    )
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
