from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.git_info import GitInfo
from exp_harness.provenance import (
    write_env_provenance,
    write_git_provenance,
    write_host_provenance,
    write_nvidia_smi,
    write_python_and_freeze,
)
from exp_harness.run.naming import proc_start_ticks_linux, sanitize_run_label
from exp_harness.store import RunPaths
from exp_harness.utils import write_json


def initial_run_json(
    *,
    name: str,
    effective_run_label: str,
    run_id: str,
    run_key: str,
    started_at: str,
    paths: RunPaths,
    run_ctx: dict[str, str],
    kind: str,
    run_key_material: dict[str, Any],
) -> dict[str, Any]:
    pid = os.getpid()
    return {
        "name": name,
        "run_label": sanitize_run_label(effective_run_label),
        "run_id": run_id,
        "run_key": run_key,
        "created_at_utc": started_at,
        "started_at_utc": started_at,
        "pid": pid,
        "proc_start_ticks": proc_start_ticks_linux(pid),
        "state": "running",
        "paths": {
            "host": {"run_dir": str(paths.run_dir), "artifacts_dir": str(paths.artifacts_dir)},
            "container": run_ctx if kind == "docker" else None,
        },
        "allocated_gpus_host": [],
        "steps": [],
        "run_key_material": run_key_material,
    }


def write_run_provenance(
    *,
    paths: RunPaths,
    roots: Roots,
    spec_path: Path,
    git: GitInfo,
    env_vars: dict[str, str],
    kind: str,
    input_fps: dict[str, Any],
    resolved_final: dict[str, Any],
    docker_meta: dict[str, Any] | None,
) -> None:
    write_git_provenance(paths.provenance_dir, git)
    write_nvidia_smi(paths.provenance_dir)
    write_host_provenance(
        paths.provenance_dir,
        extra={
            "project_root": str(roots.project_root),
            "runs_root": str(roots.runs_root),
            "artifacts_root": str(roots.artifacts_root),
            "spec_path": str(spec_path.resolve()),
            "resolved_paths": {
                "run_dir": str(paths.run_dir),
                "artifacts_dir": str(paths.artifacts_dir),
            },
        },
    )
    write_env_provenance(
        paths.provenance_dir,
        env_allow=["PATH", "CUDA_VISIBLE_DEVICES"],
        explicit_env=env_vars,
    )
    if kind != "docker":
        write_python_and_freeze(paths.provenance_dir)
    if input_fps:
        write_json(paths.provenance_dir / "inputs.json", input_fps)
    if kind == "docker":
        docker_effective = (resolved_final.get("env") or {}).get("docker")
        write_json(
            paths.provenance_dir / "docker.json",
            {
                "effective": docker_effective,
                "image": docker_meta,
                "offline_network_enforced": (docker_effective or {}).get("network") == "none"
                and bool((resolved_final.get("env") or {}).get("offline")),
            },
        )
