from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from exp_harness.errors import StepExecutionError
from exp_harness.executors.base import RunContext, StepResult
from exp_harness.store import RunPaths
from exp_harness.utils import ensure_dir, tail_text_lines, utc_now_iso

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


def _to_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_existing_step_record(
    *,
    existing: dict[str, Any] | None,
    step_id: str,
    step_dir: Path,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": step_id,
        "state": "pending",
        "started_at_utc": None,
        "finished_at_utc": None,
        "rc": None,
        "attempt": 0,
        "error": None,
        "dir": str(step_dir),
    }

    if not existing:
        return step

    step["dir"] = str(existing.get("dir") or step_dir)
    step["attempt"] = _to_int(existing.get("attempt"), default=0)
    step["started_at_utc"] = existing.get("started_at_utc")
    step["finished_at_utc"] = existing.get("finished_at_utc")
    step["error"] = existing.get("error")

    state = existing.get("state")
    if isinstance(state, str) and state:
        step["state"] = state
        step["rc"] = existing.get("rc")
        return step

    rc = existing.get("rc")
    if isinstance(rc, int):
        step["rc"] = rc
        step["state"] = "succeeded" if rc == 0 else "failed"
        if step["attempt"] <= 0:
            step["attempt"] = 1
    return step


def ensure_step_records(
    *,
    ordered_steps: list[dict[str, Any]],
    paths: RunPaths,
    run_json: dict[str, Any],
    write_run_json_fn: Callable[[RunPaths, dict[str, Any]], None],
) -> list[dict[str, Any]]:
    existing_steps = run_json.get("steps")
    existing_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(existing_steps, list):
        for item in existing_steps:
            if isinstance(item, dict):
                step_id = item.get("step_id")
                if isinstance(step_id, str) and step_id:
                    existing_by_id[step_id] = item

    normalized: list[dict[str, Any]] = []
    for idx, step in enumerate(ordered_steps):
        step_id = str(step["id"])
        step_dir = paths.steps_dir / f"{idx:02d}_{step_id}"
        normalized.append(
            _normalize_existing_step_record(
                existing=existing_by_id.get(step_id),
                step_id=step_id,
                step_dir=step_dir,
            )
        )

    run_json["steps"] = normalized
    write_run_json_fn(paths, run_json)
    return normalized


def _mark_remaining_steps_skipped(
    *,
    step_records: list[dict[str, Any]],
    start_idx: int,
    reason: str,
    paths: RunPaths,
    run_json: dict[str, Any],
    write_run_json_fn: Callable[[RunPaths, dict[str, Any]], None],
) -> None:
    finished_at = utc_now_iso()
    for rec in step_records[start_idx:]:
        if rec.get("state") == "pending":
            rec["state"] = "skipped"
            rec["finished_at_utc"] = finished_at
            rec["error"] = {"message": reason}
    write_run_json_fn(paths, run_json)


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
    resume_mode: bool = False,
) -> None:
    step_records = ensure_step_records(
        ordered_steps=ordered_steps,
        paths=paths,
        run_json=run_json,
        write_run_json_fn=write_run_json_fn,
    )

    for idx, step in enumerate(ordered_steps):
        step_record = step_records[idx]
        step_id = str(step["id"])
        step_dir = paths.steps_dir / f"{idx:02d}_{step_id}"
        ensure_dir(step_dir)
        cmd = list(step.get("cmd") or [])
        timeout_s = step.get("timeout_seconds")
        step_artifacts_dir = ((step.get("outputs") or {}) or {}).get("artifacts_dir")
        if step_artifacts_dir == f"{run_ctx['artifacts']}/{step_id}":
            ensure_dir(paths.artifacts_dir / step_id)

        if resume_mode and step_record.get("state") == "succeeded":
            continue

        step_record["state"] = "running"
        step_record["started_at_utc"] = utc_now_iso()
        step_record["finished_at_utc"] = None
        step_record["attempt"] = _to_int(step_record.get("attempt"), default=0) + 1
        step_record["error"] = None
        write_run_json_fn(paths, run_json)

        try:
            result = executor.run_step(
                ctx,
                step_index=idx,
                step_id=step_id,
                cmd=cmd,
                step_dir=step_dir,
                timeout_seconds=timeout_s,
                step_artifacts_dir=step_artifacts_dir,
            )
        except KeyboardInterrupt:
            step_record["state"] = "interrupted"
            step_record["finished_at_utc"] = utc_now_iso()
            step_record["error"] = {"message": "Step interrupted by keyboard signal"}
            write_run_json_fn(paths, run_json)
            raise
        except Exception as exc:
            step_record["state"] = "failed"
            step_record["finished_at_utc"] = utc_now_iso()
            step_record["error"] = {"message": str(exc)}
            write_run_json_fn(paths, run_json)
            _mark_remaining_steps_skipped(
                step_records=step_records,
                start_idx=idx + 1,
                reason=f"Skipped because step {step_id} failed",
                paths=paths,
                run_json=run_json,
                write_run_json_fn=write_run_json_fn,
            )
            raise

        if result.rc == 0:
            step_record["state"] = "succeeded"
            step_record["finished_at_utc"] = utc_now_iso()
            step_record["rc"] = 0
            write_run_json_fn(paths, run_json)
            continue

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

        step_record["state"] = "failed"
        step_record["finished_at_utc"] = utc_now_iso()
        step_record["rc"] = int(result.rc)
        step_record["error"] = {
            "message": f"Step failed: {step_id} (rc={result.rc})",
            "stderr_tail": tail,
            "stderr_log": str(stderr_fp),
            "stdout_log": str(stdout_fp),
        }
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
        _mark_remaining_steps_skipped(
            step_records=step_records,
            start_idx=idx + 1,
            reason=f"Skipped because step {step_id} failed",
            paths=paths,
            run_json=run_json,
            write_run_json_fn=write_run_json_fn,
        )
        raise StepExecutionError(step_id=step_id, rc=int(result.rc))
