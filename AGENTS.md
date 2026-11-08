# AGENTS.md

Working notes for coding agents and contributors in **CRF Sequence Labeler**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

A linear-chain conditional random field for sequence labeling, built from scratch: forward-backward training with log-likelihood gradients, Viterbi decoding, and span-level evaluation.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/crf_sequence_labeler/__init__.py`: CRF Sequence Labeler - a linear-chain conditional random field from scratch.
- `src/crf_sequence_labeler/cli.py`: Command line interface for the CRF sequence labeler.
- `src/crf_sequence_labeler/crf.py`: A linear-chain conditional random field, trained from scratch.
- `src/crf_sequence_labeler/data.py`: A bundled named-entity dataset with BIO tags for demonstration and evaluation.
- `src/crf_sequence_labeler/evaluation.py`: Span-level evaluation for sequence labeling.
- `src/crf_sequence_labeler/features.py`: Observation feature extraction for the linear-chain CRF.
- `src/crf_sequence_labeler/observability.py`: In-process request metrics for the HTTP service (Prometheus text exposition).
- `src/crf_sequence_labeler/service.py`: A standard-library HTTP service exposing the trained CRF tagger.
- `src/crf_sequence_labeler/training.py`: Train the CRF by stochastic gradient ascent on the log-likelihood.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make evaluation` | Reproduce the results table |
| `make run` | Run the application entrypoint |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
