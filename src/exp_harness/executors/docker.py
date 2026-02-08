from __future__ import annotations

import shlex
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from exp_harness.docker_utils import inspect_image
from exp_harness.executors.base import RunContext, StepResult
from exp_harness.utils import resolve_relpath, utc_now_iso, write_json, write_text


class DockerExecutor:
    def __init__(self) -> None:
        self._image_meta: dict[str, Any] | None = None

    def prepare_run(self, ctx: RunContext) -> None:
        docker = ctx.docker or {}
        image = docker.get("image")
        if image:
            self._image_meta = inspect_image(image=str(image), cwd=ctx.project_root)
        # Capture python/pip provenance from inside the container environment (best-effort).
        try:
            prov_dir = ctx.run_dir / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)
            py = self._probe(ctx, ["python", "-V"])
            if py:
                write_text(prov_dir / "python.txt", py.strip() + "\n")
            freeze = self._probe(ctx, ["python", "-m", "pip", "freeze"])
            if freeze is not None:
                write_text(prov_dir / "pip_freeze.txt", freeze)
        except Exception:
            return

    def _probe(self, ctx: RunContext, cmd: list[str]) -> str | None:
        docker = ctx.docker or {}
        image = docker.get("image")
        if not image:
            return None
        argv: list[str] = ["docker", "run", "--rm"]
        network = docker.get("network")
        if network:
            argv += ["--network", str(network)]
        ipc = docker.get("ipc")
        if ipc:
            argv += [f"--ipc={ipc}"]
        shm_size = docker.get("shm_size")
        if shm_size:
            argv += [f"--shm-size={shm_size}"]
        mounts = docker.get("mounts")
        if mounts is None:
            raise RuntimeError(
                "Resolved docker mounts missing; expected env.docker.mounts to be a list"
            )
        for m in mounts:
            host_p = resolve_relpath(str(m["host"]), base_dir=ctx.project_root)
            argv += ["-v", f"{host_p}:{m['container']}"]
        for k, v in (ctx.env or {}).items():
            argv += ["-e", f"{k}={v}"]
        argv += ["-w", ctx.workdir, str(image)]
        argv += cmd
        proc = subprocess.run(
            argv,
            cwd=str(ctx.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.stdout

    def _docker_run_argv(
        self,
        ctx: RunContext,
        *,
        step_id: str,
        cmd: list[str],
        step_artifacts_dir: str | None,
    ) -> list[str]:
        docker = ctx.docker or {}
        image = docker.get("image")
        if not image:
            raise RuntimeError("env.docker.image is required for docker runs")

        argv: list[str] = ["docker", "run", "--rm"]

        network = docker.get("network")
        if network:
            argv += ["--network", str(network)]

        ipc = docker.get("ipc")
        if ipc:
            argv += [f"--ipc={ipc}"]

        shm_size = docker.get("shm_size")
        if shm_size:
            argv += [f"--shm-size={shm_size}"]

        # Labels for lock gc / observability.
        argv += ["--label", f"exp-harness.name={ctx.name}"]
        argv += ["--label", f"exp-harness.run_key={ctx.run_key}"]
        argv += ["--label", f"exp-harness.step_id={step_id}"]

        mounts = docker.get("mounts")
        if mounts is None:
            raise RuntimeError(
                "Resolved docker mounts missing; expected env.docker.mounts to be a list"
            )
        for m in mounts:
            host_p = resolve_relpath(str(m["host"]), base_dir=ctx.project_root)
            argv += ["-v", f"{host_p}:{m['container']}"]

        env = dict(ctx.env)
        env["EXP_HARNESS_RUN_KEY"] = ctx.run_key
        env["EXP_HARNESS_RUN_DIR"] = f"/workspace/runs/{ctx.name}/{ctx.run_key}"
        env["EXP_HARNESS_ARTIFACTS_DIR"] = f"/workspace/artifacts/{ctx.name}/{ctx.run_key}"
        if step_artifacts_dir:
            env["EXP_HARNESS_STEP_ARTIFACTS_DIR"] = str(step_artifacts_dir)

        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]

        allocated_host = list(ctx.allocated_gpus_host)
        if allocated_host:
            argv += ["--gpus", "device=" + ",".join(str(x) for x in allocated_host)]
            argv += ["-e", "EXP_HARNESS_HOST_GPU_IDS=" + ",".join(str(x) for x in allocated_host)]

        argv += ["-w", ctx.workdir]
        argv.append(str(image))
        argv += cmd
        return argv

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
        docker_cmd = self._docker_run_argv(
            ctx, step_id=step_id, cmd=cmd, step_artifacts_dir=step_artifacts_dir
        )
        write_text(step_dir / "command.txt", " ".join(shlex.quote(x) for x in docker_cmd) + "\n")

        started = utc_now_iso()
        t0 = time.time()
        timed_out = False
        stdout_fp = (step_dir / "stdout.log").open("wb")
        stderr_fp = (step_dir / "stderr.log").open("wb")
        try:
            proc = subprocess.Popen(
                docker_cmd, cwd=str(ctx.project_root), stdout=stdout_fp, stderr=stderr_fp
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

        allocated_host = list(ctx.allocated_gpus_host)
        allocated_visible = list(allocated_host)
        result = StepResult(
            step_id=step_id,
            rc=int(rc),
            started_at_utc=started,
            finished_at_utc=finished,
            duration_seconds=float(dt),
            allocated_gpus_host=allocated_host,
            allocated_gpus_visible=allocated_visible,
            extra={
                "docker_image_id": (self._image_meta or {}).get("image_id"),
                "docker_image": (ctx.docker or {}).get("image"),
                "step_index": step_index,
                "timeout_seconds": timeout_seconds,
                "timed_out": timed_out,
            },
        )
        write_json(step_dir / "exec.json", result.__dict__)
        return result

    def finalize_run(self, _ctx: RunContext) -> None:
        return

    def docker_metadata(self) -> dict[str, Any]:
        return self._image_meta or {}
