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
uv run basedpyright
uv run pytest
uv run pytest --cov=exp_harness --cov-report=term-missing
```

## Quick start

```bash
run-experiment run "name=toy_hydra"
run-experiment status
```

Default outputs go under:

- `experiment_results/runs/<name>/<run_id>/`
- `experiment_results/artifacts/<name>/<run_id>/`

Where `run_id` is timestamp-first and human-readable:
- `<YYYYMMDD-HHMMSSZ>__<run_label>__<short_hash>`

Override roots via CLI or env vars:

- `--runs-root` / `RUN_EXPERIMENT_RUNS_ROOT`
- `--artifacts-root` / `RUN_EXPERIMENT_ARTIFACTS_ROOT`

## CLI overview

```bash
run-experiment run "name=myexp" "env=local" "++params.lr=1e-4" \
  [--config-name config] \
  [--salt myrun] [--run-label phase2_lr_micro_up] [--enforce-clean] \
  [--no-follow-steps] [--stderr-tail-lines 120] \
  [--runs-root /path/to/runs] [--artifacts-root /path/to/artifacts]

run-experiment status [--name toy] [--limit 20] [--runs-root ...]
run-experiment logs <name> <run_key> [--step train] [-f] [--runs-root ...]
run-experiment inspect <name> <run_key> [--runs-root ...]
run-experiment locks gc [--grace-seconds 600] [--force] [--runs-root ...]
run-experiment sweep "name=myexp" "++params.lr=1e-3,1e-4" "resources=default,gpu1"
```

`run-experiment sweep` uses Hydra override syntax and composition, then executes members through
the harness runner sequentially so runs/provenance stay in the canonical harness layout.

`run-experiment run` streams step logs by default. Use `--no-follow-steps` to disable live streaming.
Standard local experiments and sweeps do not require any bash wrapper scripts.

## Python API

```python
from exp_harness.run.api import run_experiment

result = run_experiment(
    overrides=["name=myexp", "env=local", "++params.lr=1e-4"],
    follow_steps=False,
)
print(result["run_key"])
```

Hydra composition API:

```python
from exp_harness.run.api import compose_experiment_config

cfg = compose_experiment_config(
    overrides=["env=docker", "resources=gpu1", "name=my_experiment"]
)
print(cfg["env"]["kind"])  # docker
```

The harness remains the source of truth for run directories/provenance; Hydra is used only for config
composition and overrides.

Hydra sweep API:

```python
from exp_harness.run.api import run_hydra_sweep

summary = run_hydra_sweep(
    overrides=["name=myexp", "++params.lr=1e-3,1e-4", "resources=default,gpu1"],
)
print(summary["total"], summary["succeeded"], summary["failed"])
```

Sweep semantics:

- Hydra is used for override parsing and per-member composition.
- exp-harness expands and executes sweep members sequentially.
- This keeps canonical harness run/provenance directories.
- Hydra launcher/sweeper plugin execution (native Hydra multirun infrastructure) is not invoked.

## Overrides

- `run` / `sweep` use Hydra override syntax (`group=value`, `+key=value`, `++key=value`, comma sweeps).

## Spec schema (minimal reference)

- `name` (required): experiment name
- `run_label` (optional): human label used in run directory naming
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

Each run writes a small durable run record under `.../runs/<name>/<run_id>/`:

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

Large outputs land under `.../artifacts/<name>/<run_id>/<step_id>/` by default.

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

Runs keep a stable `run_key` (hash) for identity/deduping and use a human-friendly `run_id` for paths:

- `run_key`: canonical identity (full hash)
- `run_id`: `<YYYYMMDD-HHMMSSZ>__<run_label>__<short_hash>`

`run_label` precedence:
1) CLI `--run-label`
2) spec `run_label`
3) spec filename stem

## Package layout

Current core package topology:

- `exp_harness/config/`: roots/env-path resolution (`Roots`, `resolve_roots`)
- `exp_harness/run/`: run orchestration API, identity, phases, state, and execution helpers
- `exp_harness/executors/`: local and docker step executors
- `exp_harness/provenance/`: git/host/env/provenance writers and git metadata collection
- `exp_harness/resources/`: GPU lock pool + allocation helpers
- `exp_harness/cli.py`: Typer CLI entrypoint

## Troubleshooting

- No artifacts from Docker runs: check `env.docker.mounts` (and that it’s not `[]`).
- Offline mode still downloads: local offline is best-effort; verify `HF_HOME` + offline env vars in `provenance/environment.json`.
- Stuck GPU locks: `run-experiment locks gc` and then `run-experiment locks gc --force` after verifying no related processes/containers are running.

## Developing / testing

- Default test run (fast, no docker/slow): `uv run pytest`
- Docker tests (requires Docker): `RUN_EXPERIMENT_TEST_DOCKER=1 uv run pytest -m docker`
- Slow/concurrency tests: `RUN_EXPERIMENT_TEST_SLOW=1 uv run pytest -m slow`

## Runner container (Docker-outside-of-Docker)

If you don't want to install exp-harness on the host (e.g. air-gapped servers), you can run it from a small
"runner" container that mounts the host Docker socket and launches study containers via the host Docker
daemon.

Important: mounting `/var/run/docker.sock` gives the container root-equivalent control of the host via Docker.

- Runner image Dockerfile: `docker/exp-harness-runner/Dockerfile`
- Build helper: `scripts/docker_build_runner_image.sh`
- Host wrapper template: `scripts/run-experiment-dood` (copy into a project as `run-experiment`)
- Script audit and classification: `scripts/README.md`
