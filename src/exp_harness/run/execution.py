from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from exp_harness.errors import StepExecutionError
from exp_harness.executors.base import RunContext, StepResult
from exp_harness.store import RunPaths
from exp_harness.utils import ensure_dir, tail_text_lines

logger = logging.getLogger(__name__)


class RunnerExecutor(Protocol):
    def prepare_run(self, ctx: RunContext, /) -> None: ...
    def run_step(
        self,
        ctx: RunContext,
        /,
        *,
        step_index: int,
        step_id: str,
        cmd: list[str],
        step_dir: Path,
        timeout_seconds: int | None,
        step_artifacts_dir: str | None,
    ) -> StepResult: ...
    def finalize_run(self, ctx: RunContext, /) -> None: ...


class ExecutorFactory(Protocol):
    def __call__(self) -> RunnerExecutor: ...


def build_executor_and_context(
    *,
    kind: str,
    name: str,
    run_key: str,
    project_root: Path,
    paths: RunPaths,
    resolved_final: dict[str, Any],
    env_vars: dict[str, str],
    allocated_gpus_host: list[int],
    follow_steps: bool,
    docker_executor_factory: ExecutorFactory,
    local_executor_factory: ExecutorFactory,
) -> tuple[RunnerExecutor, RunContext]:
    env_block = resolved_final.get("env") or {}
    workdir = str(env_block.get("workdir"))
    offline = bool(env_block.get("offline"))

    if kind == "docker":
        executor = docker_executor_factory()
        docker_block = env_block.get("docker") or {}
        ctx = RunContext(
            name=name,
            run_key=run_key,
            project_root=project_root,
            run_dir=paths.run_dir,
            artifacts_dir=paths.artifacts_dir,
            workdir=workdir,
            env=env_vars,
            offline=offline,
            kind=kind,
            docker=docker_block,
            allocated_gpus_host=allocated_gpus_host,
            stream_logs=bool(follow_steps),
        )
        return executor, ctx

    executor = local_executor_factory()
    ctx = RunContext(
        name=name,
        run_key=run_key,
        project_root=project_root,
        run_dir=paths.run_dir,
        artifacts_dir=paths.artifacts_dir,
        workdir=workdir,
        env=env_vars,
        offline=offline,
        kind=kind,
        docker=None,
        allocated_gpus_host=allocated_gpus_host,
        stream_logs=bool(follow_steps),
    )
    return executor, ctx


def run_ordered_steps(
    *,
    executor: RunnerExecutor,
    ctx: RunContext,
    ordered_steps: list[dict[str, Any]],
    paths: RunPaths,
    run_json: dict[str, Any],
    run_ctx: dict[str, str],
    stderr_tail_lines: int,
    write_run_json_fn: Callable[[RunPaths, dict[str, Any]], None],
) -> None:
    for idx, step in enumerate(ordered_steps):
        step_id = str(step["id"])
        step_dir = paths.steps_dir / f"{idx:02d}_{step_id}"
        ensure_dir(step_dir)
        cmd = list(step.get("cmd") or [])
        timeout_s = step.get("timeout_seconds")
        step_artifacts_dir = ((step.get("outputs") or {}) or {}).get("artifacts_dir")
        if step_artifacts_dir == f"{run_ctx['artifacts']}/{step_id}":
            ensure_dir(paths.artifacts_dir / step_id)

        result = executor.run_step(
            ctx,
            step_index=idx,
            step_id=step_id,
            cmd=cmd,
            step_dir=step_dir,
            timeout_seconds=timeout_s,
            step_artifacts_dir=step_artifacts_dir,
        )
        run_json["steps"].append({"step_id": step_id, "rc": result.rc, "dir": str(step_dir)})
        write_run_json_fn(paths, run_json)
        if result.rc != 0:
            stderr_fp = step_dir / "stderr.log"
            stdout_fp = step_dir / "stdout.log"
            tail = tail_text_lines(stderr_fp, n=int(stderr_tail_lines))
            logger.error("step failed: %s (rc=%s)", step_id, result.rc)
            logger.error("command: %s", step_dir / "command.txt")
            logger.error("stderr: %s", stderr_fp)
            logger.error("stdout: %s", stdout_fp)
            if tail:
                logger.error(
                    "stderr tail (last %s lines):\n%s",
                    stderr_tail_lines,
                    tail.rstrip("\n"),
                )
            run_json.setdefault("error", {})
            run_json["error"] = {
                "message": f"Step failed: {step_id} (rc={result.rc})",
                "step_id": step_id,
                "rc": int(result.rc),
                "stderr_tail": tail,
                "stderr_log": str(stderr_fp),
                "stdout_log": str(stdout_fp),
            }
            write_run_json_fn(paths, run_json)
            raise StepExecutionError(step_id=step_id, rc=int(result.rc))
