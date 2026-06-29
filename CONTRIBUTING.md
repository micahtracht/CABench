# Contributing to CABench

Thanks for your interest in improving CABench.

## Development setup

```bash
git clone https://github.com/micahtracht/CABench.git
cd CABench
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.13+ is required.

## Before you push

Run the same checks CI runs:

```bash
ruff check cabench tests       # lint
ruff format cabench tests      # format
mypy cabench                   # type-check
pytest                         # tests (with coverage gate)
```

Or install the pre-commit hooks so lint/format/type-check run automatically:

```bash
pre-commit install
```

A fast, network-free smoke test of the full pipeline:

```bash
python -m cabench run --dry-run --numquestions 8 --models gpt-5.4-mini
```

## Commit conventions

Tag-first, short one-liners: `fix: ...`, `feat: ...`, `refactor: ...`, `chore: ...`,
`docs: ...`, `test: ...`. Prefer several small commits over one large one.

## Changing artifact schemas

Any change to a versioned artifact (scores CSV, usage ledger, run metadata, prediction
files) must follow `SCHEMA_POLICY.md` — bump the version in `cabench/contracts.py`, update
producers/consumers and `cabench/migrate.py`, and add tests for both fresh writes and the
migration path.

## Tests

Tests live in `tests/` (xUnit-style functions, pytest). Add or update tests alongside any
behavior change; keep the suite green and coverage at or above the CI floor.
