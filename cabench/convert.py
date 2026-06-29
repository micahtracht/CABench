#!/usr/bin/env python
"""
cabench.convert

Convert structured-output JSONL (one JSON object per line, each with
`{"answer": [0,1,0,…]}`) into a flat text file where every line is a binary string like `0100110`.

Usage

python -m cabench convert \
    --input  data/gpt4o_val_public.jsonl \
    --output data/gpt4o_val_public.predictions
"""

from __future__ import annotations
import argparse, json, pathlib, sys
from typing import List
from cabench.contracts import (
    PREDS_TEXT_SCHEMA_NAME,
    PREDS_TEXT_SCHEMA_VERSION,
    write_schema_manifest,
)
from cabench.json_extract import extract_first_json_object

def json_to_bits(obj: dict) -> str:
    """
    Extract the answer array and return it as a '0101…' string.
    Raises ValueError if not valid.
    """

    if "answer" not in obj:
        if "final_state" in obj:
            obj["answer"] = obj["final_state"]
        else:
            raise ValueError("missing 'answer' key")

    def _collect(x, sink):
        if isinstance(x, list):
            for y in x:
                _collect(y, sink)
        elif x in (0, 1):
            sink.append(str(x))
        else:
            raise ValueError(f"invalid bit value: {x!r}")

    bits: List[str] = []
    _collect(obj["answer"], bits)
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
                # Use the new function to extract the JSON object
                obj = extract_first_json_object(line)
                if obj is None:
                    raise ValueError("No JSON object found in line")
                
                bits = json_to_bits(obj)
                fout.write(bits + "\n")
                n_ok += 1
            except Exception as exc:
                n_bad += 1
                sys.stderr.write(
                    f"[warn] line {lineno}: {exc}; writing blank line\n"
                )
                fout.write("\n")  # keep alignment with gold file

    write_schema_manifest(
        out_path,
        schema_name=PREDS_TEXT_SCHEMA_NAME,
        schema_version=PREDS_TEXT_SCHEMA_VERSION,
        fmt="text",
        notes="One flattened binary prediction string per line; blank lines mark invalid rows.",
    )
    print(f"wrote {n_ok} predictions. {n_bad} lines had errors → blank")

 
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path, required=True, help="JSONL file")
    ap.add_argument("--output", type=pathlib.Path, required=True, help="plain-text predictions")
    args = ap.parse_args(argv)
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
