# Contributing

## Dev setup

```bash
uv venv
uv sync --locked --dev
```

Install the local git hooks:

```bash
uv run pre-commit install
```

Run them on the whole repo:

```bash
uv run pre-commit run --all-files
```

## Canonical commands

- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`
- Tests (fast, default): `uv run pytest`
- Coverage: `uv run pytest --cov=exp_harness --cov-report=term-missing`

## Pytest markers

The test suite is mostly hermetic; Docker and slow/concurrency tests are gated.

- Docker tests:
  - `RUN_EXPERIMENT_TEST_DOCKER=1 uv run pytest -m docker`
- Slow tests:
  - `RUN_EXPERIMENT_TEST_SLOW=1 uv run pytest -m slow`

## Run/artifacts layout

By default, runs and artifacts go under:

- `experiment_results/runs/<name>/<run_key>/`
- `experiment_results/artifacts/<name>/<run_key>/`

It’s safe to delete `experiment_results/` locally if you want to clean up old runs.

## Do not commit

- Generated run outputs (`experiment_results/`)
- Virtualenvs and caches (`.venv/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/`)
- Secrets (API keys, private keys, tokens)
- Large binaries (the repo has a pre-commit size guardrail)
