# exp-harness runner image (Docker-outside-of-Docker)

This folder builds a small “runner” image that contains:

- the `run-experiment` CLI (from `exp-harness`)
- the Docker CLI (`docker`) so it can launch study containers via the host Docker daemon

The runner container mounts the host Docker socket (`/var/run/docker.sock`) and invokes `docker run ...`
to execute your *study* image(s). This is Docker-outside-of-Docker (DoOD).

Security note: mounting the Docker socket gives the container effectively root-equivalent control of the host.

## Build

This Dockerfile expects one or more wheels under `docker/exp-harness-runner/wheels/`.

On a connected machine:

1) Build an `exp-harness` wheel:

```bash
python -m pip install -U build
python -m build --wheel --outdir docker/exp-harness-runner/wheels .
```

2) Build the runner image:

```bash
docker build -t exp-harness-runner:local -f docker/exp-harness-runner/Dockerfile docker/exp-harness-runner
```

3) Smoke test:

```bash
docker run --rm exp-harness-runner:local --help
```

## Host wrapper

To run exp-harness without installing it on the host, copy `scripts/run-experiment-dood` into your project
root as `run-experiment`, then:

```bash
./run-experiment run path/to/spec.yaml
```

## Air-gapped transfer

On a connected machine:

```bash
docker save -o exp-harness-runner.tar exp-harness-runner:local
```

On the air-gapped server:

```bash
docker load -i exp-harness-runner.tar
```
