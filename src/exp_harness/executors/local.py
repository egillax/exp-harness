from __future__ import annotations

import os
import shlex
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from exp_harness.executors.base import RunContext, StepResult
from exp_harness.utils import utc_now_iso, write_json, write_text


class LocalExecutor:
    def prepare_run(self, _ctx: RunContext) -> None:
        return

    def run_step(
        self,
        ctx: RunContext,
        *,
        step_index: int,
        step_id: str,
        cmd: list[str],
        step_dir: Path,
        timeout_seconds: int | None,
        step_artifacts_dir: str | None,
    ) -> StepResult:
        write_text(step_dir / "command.txt", " ".join(shlex.quote(x) for x in cmd) + "\n")

        env = os.environ.copy()
        env.update(ctx.env)

        allocated_host = list(ctx.allocated_gpus_host)
        allocated_visible = list(ctx.allocated_gpus_host)
        if allocated_host:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in allocated_host)

        env["EXP_HARNESS_RUN_KEY"] = ctx.run_key
        env["EXP_HARNESS_RUN_DIR"] = str(ctx.run_dir)
        env["EXP_HARNESS_ARTIFACTS_DIR"] = str(ctx.artifacts_dir)
        if step_artifacts_dir:
            env["EXP_HARNESS_STEP_ARTIFACTS_DIR"] = str(step_artifacts_dir)

        started = utc_now_iso()
        t0 = time.time()
        timed_out = False
        stdout_fp = (step_dir / "stdout.log").open("wb")
        stderr_fp = (step_dir / "stderr.log").open("wb")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=ctx.workdir,
                env=env,
                stdout=stdout_fp,
                stderr=stderr_fp,
            )
            try:
                rc = proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                rc = 124
                with suppress(Exception):
                    proc.wait(timeout=5)
        finally:
            stdout_fp.close()
            stderr_fp.close()
        finished = utc_now_iso()
        dt = time.time() - t0

        result = StepResult(
            step_id=step_id,
            rc=int(rc),
            started_at_utc=started,
            finished_at_utc=finished,
            duration_seconds=float(dt),
            allocated_gpus_host=allocated_host,
            allocated_gpus_visible=allocated_visible,
            extra={
                "cwd": ctx.workdir,
                "step_index": step_index,
                "timeout_seconds": timeout_seconds,
                "timed_out": timed_out,
            },
        )
        write_json(step_dir / "exec.json", result.__dict__)
        return result

    def finalize_run(self, _ctx: RunContext) -> None:
        return
