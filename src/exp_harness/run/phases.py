from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.errors import RunConflictError
from exp_harness.provenance.git import GitInfo
from exp_harness.resolve import resolve_final
from exp_harness.resources.gpu_pool import Allocation, GpuPool, allocate_gpus
from exp_harness.run.execution import ExecutorFactory, build_executor_and_context, run_ordered_steps
from exp_harness.run.provenance import initial_run_json, write_run_provenance
from exp_harness.run.runtime_env import prepare_effective_docker, prepare_effective_env
from exp_harness.run.step_graph import validate_and_toposort_steps
from exp_harness.spec import ExperimentSpec
from exp_harness.store import (
    RunPaths,
    get_run_paths,
    init_run_dirs,
    resolve_run_dir,
    write_run_json,
    write_run_key_index,
)
from exp_harness.utils import ensure_dir, write_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedRun:
    paths: RunPaths
    run_ctx: dict[str, str]
    resolved_final: dict[str, Any]
    env_vars: dict[str, str]


@dataclass(frozen=True)
class ResourceAllocation:
    pool: GpuPool
    allocation: Allocation


def phase_prepare_run(
    *,
    roots: Roots,
    spec_path: Path,
    raw_runtime: dict[str, Any],
    name: str,
    kind: str,
    run_key: str,
    run_id: str,
) -> PreparedRun:
    existing_run_dir = resolve_run_dir(roots=roots, name=name, run_key=run_key)
    if existing_run_dir is not None:
        raise RunConflictError(f"Run already exists: {existing_run_dir} (use --salt for a new run)")

    paths = get_run_paths(roots=roots, name=name, run_id=run_id)
    if paths.run_dir.exists():
        raise RunConflictError(
            f"Run directory already exists: {paths.run_dir} (use --salt or a different --run-label)"
        )

    init_run_dirs(paths)
    ensure_dir(paths.artifacts_dir)
    write_run_key_index(roots=roots, name=name, run_key=run_key, run_id=run_id)
    shutil.copyfile(spec_path, paths.run_dir / "spec.yaml")

    if kind == "docker":
        run_ctx = {
            "id": run_key,
            "runs": f"/workspace/runs/{name}/{run_id}",
            "artifacts": f"/workspace/artifacts/{name}/{run_id}",
        }
    else:
        run_ctx = {"id": run_key, "runs": str(paths.run_dir), "artifacts": str(paths.artifacts_dir)}

    resolved_final = resolve_final(raw_runtime, project_root=roots.project_root, run_ctx=run_ctx)
    env_vars = prepare_effective_env(resolved_final, roots=roots, kind=kind)
    if kind == "docker":
        prepare_effective_docker(resolved_final, roots=roots, for_hashing=False)

    env_block = resolved_final.get("env") or {}
    if kind == "local" and bool(env_block.get("offline")):
        logger.warning("offline mode for local runs is best-effort; network is not sandboxed")

    ExperimentSpec.model_validate(resolved_final)
    write_json(paths.run_dir / "resolved_spec.json", resolved_final)
    return PreparedRun(
        paths=paths,
        run_ctx=run_ctx,
        resolved_final=resolved_final,
        env_vars=env_vars,
    )


def phase_initialize_metadata(
    *,
    prepared: PreparedRun,
    roots: Roots,
    spec_path: Path,
    git: GitInfo,
    name: str,
    kind: str,
    effective_run_label: str,
    run_id: str,
    run_key: str,
    started_at: str,
    run_key_material: dict[str, Any],
    input_fps: dict[str, Any],
    docker_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    run_json = initial_run_json(
        name=name,
        effective_run_label=effective_run_label,
        run_id=run_id,
        run_key=run_key,
        started_at=started_at,
        paths=prepared.paths,
        run_ctx=prepared.run_ctx,
        kind=kind,
        run_key_material=run_key_material,
    )
    write_run_json(prepared.paths, run_json)

    write_run_provenance(
        paths=prepared.paths,
        roots=roots,
        spec_path=spec_path,
        git=git,
        env_vars=prepared.env_vars,
        kind=kind,
        input_fps=input_fps,
        resolved_final=prepared.resolved_final,
        docker_meta=docker_meta,
    )
    return run_json


def phase_allocate_resources(
    *,
    roots: Roots,
    prepared: PreparedRun,
    run_json: dict[str, Any],
    name: str,
    run_key: str,
) -> ResourceAllocation:
    pool = GpuPool(locks_dir=(roots.runs_root / "_locks"))
    allocation = allocate_gpus(
        pool,
        (prepared.resolved_final.get("resources") or {}).get("gpus", 0),
        run_path=str(prepared.paths.run_dir),
        run_key=run_key,
        name=name,
    )
    try:
        run_json["allocated_gpus_host"] = allocation.gpu_ids
        write_run_json(prepared.paths, run_json)
    except Exception:
        for gpu_id in allocation.gpu_ids:
            pool.release(gpu_id, expected_pid=allocation.pid)
        raise
    return ResourceAllocation(pool=pool, allocation=allocation)


def phase_execute_run(
    *,
    roots: Roots,
    prepared: PreparedRun,
    resources: ResourceAllocation,
    run_json: dict[str, Any],
    kind: str,
    name: str,
    run_key: str,
    follow_steps: bool,
    stderr_tail_lines: int,
    docker_executor_factory: ExecutorFactory,
    local_executor_factory: ExecutorFactory,
    resume_mode: bool = False,
) -> None:
    executor, ctx = build_executor_and_context(
        kind=kind,
        name=name,
        run_key=run_key,
        project_root=roots.project_root,
        paths=prepared.paths,
        resolved_final=prepared.resolved_final,
        env_vars=prepared.env_vars,
        allocated_gpus_host=resources.allocation.gpu_ids,
        follow_steps=follow_steps,
        docker_executor_factory=docker_executor_factory,
        local_executor_factory=local_executor_factory,
    )
    executor.prepare_run(ctx)

    steps = validate_and_toposort_steps(list(prepared.resolved_final.get("steps") or []))
    run_ordered_steps(
        executor=executor,
        ctx=ctx,
        ordered_steps=steps,
        paths=prepared.paths,
        run_json=run_json,
        run_ctx=prepared.run_ctx,
        stderr_tail_lines=stderr_tail_lines,
        write_run_json_fn=write_run_json,
        resume_mode=resume_mode,
    )
    executor.finalize_run(ctx)


def phase_release_resources(*, resources: ResourceAllocation) -> None:
    for gpu_id in resources.allocation.gpu_ids:
        resources.pool.release(gpu_id, expected_pid=resources.allocation.pid)
