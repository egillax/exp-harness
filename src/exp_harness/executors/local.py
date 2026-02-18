from __future__ import annotations

import os
import selectors
import shlex
import subprocess
import sys
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
            if not ctx.stream_logs:
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
            else:
                proc = subprocess.Popen(
                    cmd,
                    cwd=ctx.workdir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                rc = _stream_process(
                    proc, stdout_fp=stdout_fp, stderr_fp=stderr_fp, timeout_seconds=timeout_seconds
                )
                timed_out = rc == 124
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


def _stream_process(
    proc: subprocess.Popen[bytes],
    *,
    stdout_fp,
    stderr_fp,
    timeout_seconds: int | None,
) -> int:
    sel = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    sel.register(proc.stdout, selectors.EVENT_READ, data=("stdout", stdout_fp, sys.stdout.buffer))
    sel.register(proc.stderr, selectors.EVENT_READ, data=("stderr", stderr_fp, sys.stderr.buffer))

    t0 = time.time()
    killed = False
    while sel.get_map():
        if timeout_seconds is not None and (time.time() - t0) > timeout_seconds and not killed:
            killed = True
            with suppress(Exception):
                proc.kill()

        timeout = 1.0
        if timeout_seconds is not None:
            remaining = timeout_seconds - (time.time() - t0)
            timeout = max(0.0, min(1.0, remaining))

        events = sel.select(timeout)
        if not events:
            continue

        for key, _ in events:
            _, file_fp, live_fp = key.data
            try:
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
            except Exception:
                chunk = b""
            if not chunk:
                with suppress(Exception):
                    sel.unregister(key.fileobj)
                continue
            file_fp.write(chunk)
            file_fp.flush()
            live_fp.write(chunk)
            live_fp.flush()

    with suppress(Exception):
        proc.wait(timeout=5)

    if killed:
        return 124
    return int(proc.returncode or 0)
