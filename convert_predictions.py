#!/usr/bin/env python
"""
convert_predictions.py

Convert structured-output JSONL (one JSON object per line, each with
`{"answer": [0,1,0,…]}`) into a flat text file where every line is a binary string like `0100110`.

Usage

python convert_predictions.py \
    --input  data/gpt4o_val_public.jsonl \
    --output data/gpt4o_val_public.predictions
"""

from __future__ import annotations
import argparse, json, pathlib, sys


def json_to_bits(obj: dict) -> str:
    """
    Extract the answer array and return it as a '0101…' string.
    Raises ValueError if not valid.
    """
    if "answer" not in obj:
        raise ValueError("missing 'answer' key")

    ans = obj["answer"]
    if not isinstance(ans, list):
        raise ValueError("'answer' must be a list")

    bits = []
    for x in ans:
        if x not in (0, 1):
            raise ValueError(f"invalid bit value: {x!r}")
        bits.append(str(x))
    return "".join(bits)


def convert_file(in_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """
    Convert whole JSONL file → plain-text predictions file.
    """
    n_ok, n_bad = 0, 0
    with in_path.open() as fin, out_path.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                bits = json_to_bits(obj)
                fout.write(bits + "\n")
                n_ok += 1
            except Exception as exc:
                n_bad += 1
                sys.stderr.write(
                    f"[warn] line {lineno}: {exc}; writing blank line\n"
                )
                fout.write("\n")  # keep alignment with gold file

    print(f"wrote {n_ok} predictions. {n_bad} lines had errors → blank")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True, help="JSONL file")
    ap.add_argument("--output", type=pathlib.Path, required=True, help="plain-text predictions")
    args = ap.parse_args()
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
