#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Build the exp-harness runner image (DoOD) without requiring a host Python environment.

This script:
  1) builds an exp-harness wheel using a temporary python container
  2) writes it to docker/exp-harness-runner/wheels/
  3) builds the runner image

Usage:
  scripts/docker_build_runner_image.sh [--tag exp-harness-runner:local]
USAGE
}

TAG="exp-harness-runner:local"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WHEELS_DIR="${ROOT}/docker/exp-harness-runner/wheels"

mkdir -p "${WHEELS_DIR}"
rm -f "${WHEELS_DIR}"/*.whl

echo "[runner-build] building exp-harness wheel from: ${ROOT}" >&2
docker run --rm \
  -v "${ROOT}:/src" \
  -v "${WHEELS_DIR}:/wheels" \
  -w /src \
  python:3.12-slim-bookworm \
  bash -lc "python -m pip install -U pip build >/dev/null && python -m build --wheel --outdir /wheels"

echo "[runner-build] built wheel(s):" >&2
ls -la "${WHEELS_DIR}"/*.whl >&2

echo "[runner-build] building image: ${TAG}" >&2
docker build -t "${TAG}" -f "${ROOT}/docker/exp-harness-runner/Dockerfile" "${ROOT}/docker/exp-harness-runner"

echo "[runner-build] done: ${TAG}" >&2
