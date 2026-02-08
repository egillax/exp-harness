from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.utils import resolve_relpath, shell_out

RUNS_ROOT_PLACEHOLDER = "<RUNS_DIR>"
ARTIFACTS_ROOT_PLACEHOLDER = "<ARTIFACTS_DIR>"


def inspect_image(*, image: str, cwd: Path) -> dict[str, Any] | None:
    """
    Best-effort image metadata for provenance/run_key.
    """
    rc, out, _ = shell_out(["docker", "image", "inspect", image], cwd=cwd)
    if rc != 0:
        return None
    try:
        data = json.loads(out)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    img = data[0] if isinstance(data[0], dict) else None
    if not img:
        return None
    return {
        "image": image,
        "image_id": img.get("Id"),
        "repo_digests": img.get("RepoDigests") or [],
        "repo_tags": img.get("RepoTags") or [],
    }


def resolve_mounts(
    mounts: list[dict[str, Any]] | None,
    *,
    roots: Roots,
    project_root: Path,
    for_hashing: bool,
) -> list[dict[str, str]]:
    """
    Tri-state mounts resolution:
      - mounts is None (missing/null) => auto minimal mounts for runs/artifacts
      - mounts is [] => no mounts
      - mounts is non-empty => resolve host paths to absolute
    """
    if mounts is None:
        if for_hashing:
            return [
                {"host": RUNS_ROOT_PLACEHOLDER, "container": "/workspace/runs"},
                {"host": ARTIFACTS_ROOT_PLACEHOLDER, "container": "/workspace/artifacts"},
            ]
        return [
            {"host": str(roots.runs_root.resolve()), "container": "/workspace/runs"},
            {"host": str(roots.artifacts_root.resolve()), "container": "/workspace/artifacts"},
        ]
    if mounts == []:
        return []
    out: list[dict[str, str]] = []
    for m in mounts:
        host = str(m.get("host"))
        container = str(m.get("container"))
        if for_hashing:
            out.append({"host": host, "container": container})
        else:
            host_abs = resolve_relpath(host, base_dir=project_root)
            out.append({"host": str(host_abs), "container": container})
    return out


def resolve_docker_runtime(
    docker: dict[str, Any],
    *,
    offline: bool,
    roots: Roots,
    project_root: Path,
    for_hashing: bool,
) -> dict[str, Any]:
    effective = dict(docker)
    mounts_val = effective.get("mounts")
    effective["mounts"] = resolve_mounts(
        mounts_val if mounts_val is None else list(mounts_val),
        roots=roots,
        project_root=project_root,
        for_hashing=for_hashing,
    )
    if offline and effective.get("network") is None:
        effective["network"] = "none"
    return effective
