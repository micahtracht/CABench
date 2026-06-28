# CABench Schema Evolution Policy

This policy governs artifact contracts defined in `contracts.py`.

## Scope

Versioned artifact schemas:
- `cabench.results.scores`
- `cabench.logs.usage`
- `cabench.results.run_metadata`
- `cabench.predictions.jsonl`
- `cabench.predictions.bits`

## Versioning Rules

Use semantic versioning for each schema ID:
- `MAJOR`: backward-incompatible changes (rename/remove required columns/fields, type changes).
- `MINOR`: backward-compatible additions (new optional fields/columns).
- `PATCH`: non-structural clarifications (docs/manifests/notes only).

## Change Process

When changing an artifact contract:
1. Update schema constants in `contracts.py`.
2. Update producers/consumers to read/write the new contract.
3. Add/update migration logic in `migrate_artifacts.py`.
4. Add/adjust tests for both fresh writes and migration paths.
5. Update docs (`README.md`, `INFO.md`, this file).

## Compatibility Guarantees

- Writers in this repo emit only the current schema versions in `contracts.py`.
- Readers should remain tolerant for one previous compatible generation where practical.
- Incompatible historical files should be upgraded using `migrate_artifacts.py` before running strict writers.

## Migration Story

`migrate_artifacts.py` is the supported compatibility tool for legacy artifacts:
- `scores.csv`: normalizes legacy headers to canonical `SCORES_COLUMNS`.
- usage CSV (`*_usage.csv`, `master_usage.csv`): normalizes headers to canonical `USAGE_COLUMNS` (including legacy aliases like `usd_cost`).
- `run_metadata.jsonl`: injects missing schema identifiers and normalizes legacy key aliases.

After migration, sidecar manifests are refreshed (`*.schema.json`).

## Operational Guidance

- Run migration once before enabling new strict schema checks on old repositories:
  - `python migrate_artifacts.py --results-dir results --log-dir logs`
- Prefer migration + commit over hand-editing historical artifacts.
- Treat schema/version bumps as release-level changes.
