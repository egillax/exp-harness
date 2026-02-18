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

CI / reproducible install (ensures `uv.lock` matches `pyproject.toml`):

```bash
uv sync --dev --locked
```

Install git hooks (recommended):

```bash
uv run pre-commit install
uv run pre-commit run --all-files
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
run-experiment run examples/toy_overlay.yaml
run-experiment status
```

Default outputs go under:

- `experiment_results/runs/<name>/<run_key>/`
- `experiment_results/artifacts/<name>/<run_key>/`

Override roots via CLI or env vars:

- `--runs-root` / `RUN_EXPERIMENT_RUNS_ROOT`
- `--artifacts-root` / `RUN_EXPERIMENT_ARTIFACTS_ROOT`

## CLI overview

```bash
run-experiment run spec.yaml \
  [--set params.x=3] [--set-str params.tag=001] \
  [--salt myrun] [--enforce-clean] \
  [--follow-steps] [--stderr-tail-lines 120] \
  [--runs-root /path/to/runs] [--artifacts-root /path/to/artifacts]

run-experiment status [--name toy] [--limit 20] [--runs-root ...]
run-experiment logs <name> <run_key> [--step train] [-f] [--runs-root ...]
run-experiment inspect <name> <run_key> [--runs-root ...]
run-experiment locks gc [--grace-seconds 600] [--force] [--runs-root ...]
```

## Overrides

- `--set params.x=...` parses the RHS as YAML (so `3`, `true`, `[1,2]` work).
- `--set-str params.x=...` forces the RHS to be a string.
- For many sweeps, prefer spec layering with `extends:` so you can keep a stable base spec and make small per-run spec files.

## Spec schema (minimal reference)

- `extends` (optional): path (or list of paths) to a base spec to merge in first
- `name` (required): experiment name
- `env.kind`: `local|docker` (default: `local`)
- `env.workdir`: working directory (default: project root)
- `env.env`: environment variables dict (default: `{}`)
- `env.offline`: `true|false` (default: `false`)
- `env.docker.image` (required if `env.kind: docker`)
- `env.docker.runtime` (optional, e.g. `nvidia`)
- `env.docker.gpu_mode` (optional): `auto|docker_gpus_device|nvidia_visible_devices|none` (default: `auto`)
- `env.docker.mounts`: tri-state (see below)
- `resources.gpus`: `0|N|[ids...]` (default: `0`)
- `inputs.<key>.path`: input path
- `inputs.<key>.fingerprint`: `{kind, include, exclude}` (optional)
- `vars`: freeform YAML variables (optional)
- `params`: freeform YAML (optional)
- `steps[]`: `{id, needs?, cmd, outputs.artifacts_dir?, timeout_seconds?}`

Minimal Docker example:

```yaml
name: toy_docker
env:
  kind: docker
  offline: true
  docker:
    image: python:3.12-slim
steps:
  - id: hello
    cmd: ["python", "-c", "print('hi')"]
```

## Interpolation contract

Interpolation is resolved before execution and written to `resolved_spec.json`.

Supported references:

- `${params.*}`
- `${vars.*}`
- `${<key>}` (shorthand for simple keys in `vars` / `params`)
- `${inputs.<key>.path}`
- `${env.workdir}`, `${env.kind}`
- `${run.id}` (the run key)
- `${run.runs}` (run directory path)
- `${run.artifacts}` (artifacts directory path)

## Outputs / provenance

Each run writes a small durable run record under `.../runs/<name>/<run_key>/`:

- `run.json`: state + pointers
- `spec.yaml`: original spec
- `resolved_spec.json`: defaults + overrides + interpolation resolved
- `provenance/`:
  - `git.json` (+ diff hash; `git_diff.patch` optional)
  - `inputs.json` (fingerprints)
  - `host.json`, `nvidia_smi.txt` (best-effort)
  - `environment.json` (resolved roots + selected env)
  - `docker.json` (if docker)
  - `python.txt`, `pip_freeze.txt` (from inside the execution env)
- `steps/00_<step_id>/`:
  - `command.txt`, `exec.json`, `stdout.log`, `stderr.log` (and optional `metrics.json`)
  - On failure, `run-experiment run` prints a tail of `stderr.log` (configurable with `--stderr-tail-lines`).

Large outputs land under `.../artifacts/<name>/<run_key>/<step_id>/` by default.

## Docker notes (MVP)

- `env.kind: docker` requires `env.docker.image`.
- `env.offline: true` sets `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` and defaults `HF_HOME`:
  - local: `<artifacts_root>/hf_home` (best-effort; network is not sandboxed)
  - docker: `/workspace/artifacts/hf_home` and (by default) `env.docker.network: none`
- `env.docker.mounts` is tri-state:
  - omitted / null → auto-mount runs+artifacts to `/workspace/runs` and `/workspace/artifacts`
  - provided list → use exactly that list
  - provided empty list → no mounts
- By default, docker image identity is verified via `docker image inspect` and recorded (Id + RepoDigests/RepoTags). If inspect fails, the run fails unless `env.docker.allow_unverified_image: true`.

## GPU locks

Locks live under `experiment_results/runs/_locks/` (or your `--runs-root` equivalent).

- `resources.gpus: 2` requests any 2 free GPUs; `resources.gpus: [2,3]` requests exact GPU ids.
- If insufficient GPUs are available, the run fails before steps start.
- Docker runs are constrained via `docker run --gpus device=...` by default.
  - If your Docker/NVIDIA runtime combination errors with "cannot set both Count and DeviceIDs on device request",
    set `env.docker.gpu_mode: nvidia_visible_devices` (and optionally `env.docker.runtime: nvidia`).
- `run-experiment status` shows alive/orphaned GPU locks.
- `run-experiment locks gc` lists orphan locks eligible for removal (use `--force` to delete).

## Run discoverability

Runs are identified by a stable `run_key` (hash). To make it easier to find runs on disk by time,
each run also creates a symlink under:

- `.../runs/<name>/_by_time/<timestamp>_<run_key>` -> `../<run_key>`
- `.../artifacts/<name>/_by_time/<timestamp>_<run_key>` -> `../<run_key>`

## Troubleshooting

- No artifacts from Docker runs: check `env.docker.mounts` (and that it’s not `[]`).
- Offline mode still downloads: local offline is best-effort; verify `HF_HOME` + offline env vars in `provenance/environment.json`.
- Stuck GPU locks: `run-experiment locks gc` and then `run-experiment locks gc --force` after verifying no related processes/containers are running.

## Developing / testing

- Default test run (fast, no docker/slow): `uv run pytest`
- Docker tests (requires Docker): `RUN_EXPERIMENT_TEST_DOCKER=1 uv run pytest -m docker`
- Slow/concurrency tests: `RUN_EXPERIMENT_TEST_SLOW=1 uv run pytest -m slow`
