# syntax=docker/dockerfile:1.7

FROM python:3.12.12-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Docker CLI (NOT dockerd). This container talks to the host Docker daemon via /var/run/docker.sock.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      docker.io \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/exp-harness

# Build context is expected to be `docker/exp-harness-runner/`.
# Place an exp-harness wheel under `docker/exp-harness-runner/wheels/` before building.
COPY wheels/ /opt/wheels/

RUN python -m pip install --no-cache-dir -U pip \
 && shopt -s nullglob \
 && wheels=(/opt/wheels/*.whl) \
 && if (( ${#wheels[@]} == 0 )); then echo "error: no wheels found under /opt/wheels" >&2; exit 2; fi \
 && python -m pip install --no-cache-dir "${wheels[@]}" \
 && run-experiment --help >/dev/null

ENTRYPOINT ["run-experiment"]
