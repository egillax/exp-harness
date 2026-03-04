from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict

from exp_harness.config import Roots
from exp_harness.docker_utils import ARTIFACTS_ROOT_PLACEHOLDER, RUNS_ROOT_PLACEHOLDER
from exp_harness.errors import DockerConfigurationError, DockerImageInspectionError
from exp_harness.fingerprints import fingerprint_path
from exp_harness.git_info import GitInfo
from exp_harness.resolve import resolve_for_hashing
from exp_harness.run.naming import format_run_id
from exp_harness.run.runtime_env import prepare_effective_docker, resolved_offline_env
from exp_harness.run_key import compute_run_key
from exp_harness.utils import resolve_relpath, utc_now_iso


class InspectImageFn(Protocol):
    def __call__(self, *, image: str, cwd: Path) -> dict[str, Any] | None: ...


class RunIdentity(TypedDict):
    run_key: str
    run_id: str
    started_at: str
    run_key_material: dict[str, Any]
    input_fps: dict[str, Any]
    docker_meta: dict[str, Any] | None


def _placeholder_run(*, kind: str, name: str) -> dict[str, str]:
    placeholder_id = "<RUN_ID>"
    if kind == "docker":
        return {
            "id": placeholder_id,
            "runs": f"/workspace/runs/{name}/{placeholder_id}",
            "artifacts": f"/workspace/artifacts/{name}/{placeholder_id}",
        }
    return {
        "id": placeholder_id,
        "runs": f"{RUNS_ROOT_PLACEHOLDER}/{name}/{placeholder_id}",
        "artifacts": f"{ARTIFACTS_ROOT_PLACEHOLDER}/{name}/{placeholder_id}",
    }


def _collect_input_fingerprints(
    *, resolved_hashing: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    # Best-effort, only if configured and parseable.
    input_fps: dict[str, Any] = {}
    inputs = resolved_hashing.get("inputs") or {}
    for key, value in inputs.items():
        if not isinstance(value, dict):
            continue
        path_s = value.get("path")
        fp = value.get("fingerprint") or {}
        if not isinstance(path_s, str):
            continue
        p = resolve_relpath(path_s, base_dir=project_root)
        kind_s = str(fp.get("kind") or "none")
        include = list(fp.get("include") or [])
        exclude = list(fp.get("exclude") or [])
        res = fingerprint_path(p, kind=kind_s, include=include, exclude=exclude)
        input_fps[key] = {
            "path": str(p),
            "fingerprint": {"kind": res.kind, "value": res.value, "files_hashed": res.files_hashed},
            "include": include,
            "exclude": exclude,
        }
    return input_fps


def _resolve_docker_meta(
    *,
    kind: str,
    resolved_hashing: dict[str, Any],
    project_root: Path,
    inspect_image_fn: InspectImageFn,
) -> dict[str, Any] | None:
    if kind != "docker":
        return None
    docker_cfg = (resolved_hashing.get("env") or {}).get("docker") or {}
    image = docker_cfg.get("image")
    if not image:
        raise DockerConfigurationError("env.docker.image is required for docker runs")
    allow_unverified = bool(docker_cfg.get("allow_unverified_image", False))
    docker_meta = inspect_image_fn(image=str(image), cwd=project_root)
    if docker_meta is None and not allow_unverified:
        raise DockerImageInspectionError(
            f"docker image inspect failed for image={image!r}; set env.docker.allow_unverified_image: true to proceed"
        )
    if docker_meta is None:
        return {
            "image": str(image),
            "image_id": None,
            "repo_digests": [],
            "repo_tags": [],
            "unverified": True,
        }
    return docker_meta


def build_run_identity(
    *,
    raw_hash: dict[str, Any],
    roots: Roots,
    name: str,
    kind: str,
    effective_run_label: str,
    salt: str | None,
    git: GitInfo,
    inspect_image_fn: InspectImageFn,
) -> RunIdentity:
    # Apply effective defaults that affect run_key hashing.
    env_block_hash = dict(raw_hash.get("env") or {})
    env_vars_hash = dict(env_block_hash.get("env") or {})
    if kind == "docker":
        hf_home_hash = "/workspace/artifacts/hf_home"
    else:
        hf_home_hash = f"{ARTIFACTS_ROOT_PLACEHOLDER}/hf_home"
    env_vars_hash = resolved_offline_env(
        offline=bool(env_block_hash.get("offline")),
        hf_home=hf_home_hash,
        existing=env_vars_hash,
    )
    env_block_hash["env"] = env_vars_hash
    raw_hash["env"] = env_block_hash

    if kind == "docker":
        prepare_effective_docker(raw_hash, roots=roots, for_hashing=True)

    resolved_hashing = resolve_for_hashing(
        raw_hash,
        project_root=roots.project_root,
        placeholder_run=_placeholder_run(kind=kind, name=name),
    )
    input_fps = _collect_input_fingerprints(
        resolved_hashing=resolved_hashing, project_root=roots.project_root
    )
    docker_meta = _resolve_docker_meta(
        kind=kind,
        resolved_hashing=resolved_hashing,
        project_root=roots.project_root,
        inspect_image_fn=inspect_image_fn,
    )

    run_key_material: dict[str, Any] = {
        "spec": resolved_hashing,
        "git": {
            "commit": git.commit,
            "dirty": git.dirty,
            "diff_hash": git.diff_hash,
        },
        "docker_image": docker_meta,
        "inputs_fingerprints": input_fps,
        "salt": salt,
    }
    run_key = compute_run_key(run_key_material)
    started_at = utc_now_iso()
    run_id = format_run_id(
        started_at_utc=started_at,
        run_label=effective_run_label,
        run_key=run_key,
    )
    return {
        "run_key": run_key,
        "run_id": run_id,
        "started_at": started_at,
        "run_key_material": run_key_material,
        "input_fps": input_fps,
        "docker_meta": docker_meta,
    }
