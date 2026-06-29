# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Single installable `cabench` package with one CLI entry point
  (`run`, `report`, `list-presets`, `generate`, `eval`, `convert`, `migrate`).
- Tooling: ruff (lint + format), mypy, coverage gate, and a pre-commit config.
- Packaging metadata: SPDX license, project URLs, classifiers, `py.typed`, and a
  single-sourced dynamic version.
- `CONTRIBUTING.md`, `SECURITY.md`, and this changelog.

### Changed
- Refreshed model/pricing config to the current OpenAI lineup (gpt-5.4 family).
- Orchestration runs fully in-process; the 1D/2D model runners are unified behind one
  `run_batch(dim=...)`; the CA generators and rule classes share a common base.
- Library errors raise typed exceptions caught at the CLI boundary instead of calling
  `sys.exit`; CI installs the package and runs lint/format/type/coverage across
  Python 3.13 and 3.14.

### Fixed
- Answer extraction prefers the last answer-bearing JSON object (matches the prompt
  contract).
- Blank/empty predictions are counted as invalid; scoring is robust to empty or malformed
  inputs.
- Triviality detection now considers the full timestep horizon.

## [0.1.0]
- Initial baseline: 1D/2D cellular-automata benchmark with dataset generation, OpenAI
  querying, scoring, usage/cost logging, and versioned artifact schemas.
