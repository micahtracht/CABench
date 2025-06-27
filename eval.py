from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import List

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
    gold_lines = gold_path.read_text(encoding='utf-8').splitlines()
    pred_lines = pred_path.read_text(encoding='utf-8').splitlines()
    
    if len(gold_lines) != len(pred_lines):
        print(f'Mismatch: there are {len(pred_lines)} predictions but {len(gold_lines)} gold cases.',
            file=sys.stderr,
        )
        sys.exit(1)
    
    total = len(gold_lines)
    sum_acc = 0.0
    exact_match_count = 0
    invalid_preds = 0
    
    for idx, (gline, pred) in enumerate(zip(gold_lines, pred_lines), start=1):
        data = json.loads(gline)
        correct = data['target']

        # Validate that the prediction is a binary string
        if not all(c in '01' for c in pred):
            print(f"Warning: Invalid character(s) in prediction on line {idx}. Treating as incorrect.", file=sys.stderr)
            acc = 0.0
            invalid_preds += 1
        else:
            acc = normalized_hamming_accuracy(correct, pred)

        sum_acc += acc
        
        if pred == correct:
            exact_match_count += 1
            
    if invalid_preds > 0:
        print(f"-Found {invalid_preds} predictions with invalid characters.", file=sys.stderr)
        
    average_acc = sum_acc / total
    exact_match_percent = exact_match_count / total
    print(f"-Evaluated {total} cases")
    print(f"-Normalized Hamming accuracy: {average_acc:.4f}")
    print(f"-Exact-match accuracy: {exact_match_count}/{total} = {exact_match_percent * 100:.2f}%")
    sys.exit(0)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score binary-string predictions against a gold JSONL of ECA targets.")

    p.add_argument("--gold",type=pathlib.Path,required=True,help="Path to gold JSONL (with a ‘target’ field).",
    )
    p.add_argument("--pred",type=pathlib.Path,required=True,help="Path to predictions file (one binary string per line).",)
    return p

def main(argv: List[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args.gold, args.pred)

if __name__ == "__main__":
    main()