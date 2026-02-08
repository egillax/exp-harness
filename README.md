# exp-harness

`exp-harness` provides a local experiments harness with a CLI (`run-experiment`) for running multi-step experiments reproducibly and capturing provenance, logs, and (optionally) Docker execution metadata.

## Install (editable)

```bash
cd exp-harness
pip install -e .
```

## Dev setup (uv + Ruff)

```bash
cd exp-harness
uv venv
uv sync --dev
```

Repro / CI-style install:

```bash
uv sync --dev --frozen
```

Common commands:

```bash
uv run ruff format .
uv run ruff check . --fix
uv run pytest
uv run pytest --cov=exp_harness --cov-report=term-missing
```

## Quick start

```bash
run-experiment run examples/toy.yaml
run-experiment status
```

Default outputs go under:

- `experiment_results/runs/<name>/<run_key>/`
- `experiment_results/artifacts/<name>/<run_key>/`

Override roots via CLI or env vars:

- `--runs-root` / `RUN_EXPERIMENT_RUNS_ROOT`
- `--artifacts-root` / `RUN_EXPERIMENT_ARTIFACTS_ROOT`

## Overrides

- `--set params.x=...` parses the RHS as YAML (so `3`, `true`, `[1,2]` work).
- `--set-str params.x=...` forces the RHS to be a string.

## Docker notes (MVP)

- `env.kind: docker` requires `env.docker.image`.
- `env.offline: true` sets Hugging Face offline env vars; for docker it also defaults `env.docker.network: none` if unspecified.
- `env.docker.mounts` is tri-state:
  - omitted / null → auto-mount runs+artifacts to `/workspace/runs` and `/workspace/artifacts`
  - provided list → use exactly that list
  - provided empty list → no mounts
- By default, docker image identity is verified via `docker image inspect` and recorded (Id + RepoDigests/RepoTags). If inspect fails, the run fails unless `env.docker.allow_unverified_image: true`.

## GPU locks

Locks live under `experiment_results/runs/_locks/` (or your `--runs-root` equivalent).

- `run-experiment status` shows alive/orphaned GPU locks.
- `run-experiment locks gc` lists orphan locks eligible for removal (use `--force` to delete).
