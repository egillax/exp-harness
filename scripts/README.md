# scripts/

This directory contains **operational wrappers** only.

Standard local experiments and sweeps are Python/CLI-first:

- `run-experiment run ...`
- `run-experiment sweep ...`
- `python -c "from exp_harness.run.api import run_experiment, run_hydra_sweep; ..."`

Script audit:

- `run-experiment-dood`:
  Keep. Optional Docker-outside-of-Docker wrapper for environments where host Python install is undesirable.
- `docker_build_runner_image.sh`:
  Keep. Optional runner image build helper used for containerized operational workflows.

No bash script in this directory is required for standard local experiment or sweep execution.
