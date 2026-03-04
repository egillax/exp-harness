from __future__ import annotations

import os
import selectors
import shlex
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

from exp_harness.docker_utils import inspect_image
from exp_harness.errors import DockerConfigurationError
from exp_harness.executors.base import RunContext, StepResult
from exp_harness.utils import resolve_relpath, utc_now_iso, write_json, write_text


class DockerExecutor:
    def __init__(self) -> None:
        self._image_meta: dict[str, Any] | None = None
        self._gpu_mode_selected: str | None = None

    def prepare_run(self, ctx: RunContext) -> None:
        docker = ctx.docker or {}
        image = docker.get("image")
        if image:
            self._image_meta = inspect_image(image=str(image), cwd=ctx.project_root)
        self._write_mount_provenance(ctx)
        self._probe_mounts(ctx)

        # Cache a GPU mode decision for this run (best-effort).
        if ctx.allocated_gpus_host and (docker.get("gpu_mode") or "auto") == "auto":
            _ = self._resolve_gpu_mode(ctx)
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

    def _write_mount_provenance(self, ctx: RunContext) -> None:
        try:
            docker = ctx.docker or {}
            mounts = docker.get("mounts")
            if mounts is None:
                return
            prov_dir = ctx.run_dir / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)

            resolved: list[dict[str, Any]] = []
            for m in mounts:
                host_p = resolve_relpath(str(m["host"]), base_dir=ctx.project_root)
                resolved.append(
                    {
                        "host": str(host_p),
                        "host_exists": bool(host_p.exists()),
                        "container": str(m["container"]),
                    }
                )
            write_json(prov_dir / "docker_mounts.json", resolved)
        except Exception:
            return

    def _probe_mounts(self, ctx: RunContext) -> None:
        try:
            prov_dir = ctx.run_dir / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)
            out = self._probe(
                ctx,
                [
                    "sh",
                    "-lc",
                    "pwd; ls -la /workspace || true; "
                    "echo '--- /workspace/runs'; ls -la /workspace/runs || true; "
                    "echo '--- /workspace/artifacts'; ls -la /workspace/artifacts || true; "
                    "echo '--- /workspace/artifacts/tensorized'; ls -la /workspace/artifacts/tensorized || true",
                ],
            )
            if out:
                write_text(prov_dir / "docker_mount_check.txt", out)
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
            raise DockerConfigurationError(
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

    def _gpu_mode_requested(self, ctx: RunContext) -> str:
        docker = ctx.docker or {}
        mode = str(docker.get("gpu_mode") or "auto").strip().lower()
        if mode in {"auto", "docker_gpus_device", "nvidia_visible_devices", "none"}:
            return mode
        raise DockerConfigurationError(f"invalid env.docker.gpu_mode: {mode!r}")

    def _resolve_gpu_mode(self, ctx: RunContext) -> str:
        mode = self._gpu_mode_requested(ctx)
        if mode != "auto":
            return mode
        if not ctx.allocated_gpus_host:
            return "none"
        if self._gpu_mode_selected:
            return self._gpu_mode_selected

        # Default preference: the native docker GPU device selector.
        supported = self._supports_docker_gpus_device(ctx)
        self._gpu_mode_selected = "docker_gpus_device" if supported else "nvidia_visible_devices"
        return self._gpu_mode_selected

    def _supports_docker_gpus_device(self, ctx: RunContext) -> bool:
        """
        Detect whether `docker run --gpus device=...` works on this host/runtime.

        Some Docker/NVIDIA runtime combinations error with:
          "cannot set both Count and DeviceIDs on device request"
        In that case, we fall back to NVIDIA_VISIBLE_DEVICES.
        """
        docker = ctx.docker or {}
        image = docker.get("image")
        if not image:
            return True
        # Probe with the full requested list. Some runtimes accept a single device but fail on a list.
        gpu_list = ",".join(str(x) for x in list(ctx.allocated_gpus_host))
        argv = ["docker", "run", "--rm", "--gpus", f"device={gpu_list}", str(image), "true"]
        proc = subprocess.run(
            argv,
            cwd=str(ctx.project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode == 0:
            return True
        out = proc.stdout or ""
        # Unknown failure: keep the default behavior so the real error is visible later.
        return "cannot set both Count and DeviceIDs on device request" not in out

    def _should_retry_gpu_mode_fallback(self, *, stderr_text: str, stdout_text: str) -> bool:
        msg = "cannot set both Count and DeviceIDs on device request"
        return (msg in (stderr_text or "")) or (msg in (stdout_text or ""))

    def _run_docker_once(
        self,
        *,
        docker_cmd: list[str],
        cwd: Path,
        step_dir: Path,
        timeout_seconds: int | None,
        stream_logs: bool,
    ) -> tuple[int, bool]:
        timed_out = False
        stdout_fp = (step_dir / "stdout.log").open("wb")
        stderr_fp = (step_dir / "stderr.log").open("wb")
        try:
            if not stream_logs:
                proc = subprocess.Popen(
                    docker_cmd, cwd=str(cwd), stdout=stdout_fp, stderr=stderr_fp
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
                    docker_cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                rc = self._stream_process(
                    proc,
                    stdout_fp=stdout_fp,
                    stderr_fp=stderr_fp,
                    timeout_seconds=timeout_seconds,
                )
                timed_out = rc == 124
        finally:
            stdout_fp.close()
            stderr_fp.close()
        return int(rc), bool(timed_out)

    def _archive_attempt_files(self, *, step_dir: Path, attempt: int) -> None:
        # Best-effort: keep first attempt logs/command for debugging.
        suffix = f".attempt{attempt}"
        for src_name, dst_name in [
            ("command.txt", f"command{suffix}.txt"),
            ("stdout.log", f"stdout{suffix}.log"),
            ("stderr.log", f"stderr{suffix}.log"),
            ("exec.json", f"exec{suffix}.json"),
        ]:
            src = step_dir / src_name
            dst = step_dir / dst_name
            if src.exists() and not dst.exists():
                with suppress(Exception):
                    src.rename(dst)

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
            raise DockerConfigurationError("env.docker.image is required for docker runs")

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
            raise DockerConfigurationError(
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

        allocated_host = list(ctx.allocated_gpus_host)
        if allocated_host:
            mode = self._resolve_gpu_mode(ctx)
            gpu_list = ",".join(str(x) for x in allocated_host)
            if mode == "docker_gpus_device":
                argv += ["--gpus", "device=" + gpu_list]
                env["EXP_HARNESS_HOST_GPU_IDS"] = gpu_list
            elif mode == "nvidia_visible_devices":
                runtime = str(docker.get("runtime") or "nvidia").strip()
                if runtime:
                    argv += [f"--runtime={runtime}"]

                # Constrain which host GPUs are exposed to the container.
                # Note: inside the container, these typically get re-indexed to 0..N-1.
                env.setdefault("NVIDIA_VISIBLE_DEVICES", gpu_list)
                env.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")
                env.setdefault(
                    "CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(len(allocated_host)))
                )
                env["EXP_HARNESS_HOST_GPU_IDS"] = gpu_list
            elif mode == "none":
                pass
            else:
                raise DockerConfigurationError(f"unexpected gpu mode: {mode!r}")

        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]

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
        started = utc_now_iso()
        t0 = time.time()
        attempt = 1

        docker_cmd = self._docker_run_argv(
            ctx, step_id=step_id, cmd=cmd, step_artifacts_dir=step_artifacts_dir
        )
        write_text(step_dir / "command.txt", " ".join(shlex.quote(x) for x in docker_cmd) + "\n")
        rc, timed_out = self._run_docker_once(
            docker_cmd=docker_cmd,
            cwd=ctx.project_root,
            step_dir=step_dir,
            timeout_seconds=timeout_seconds,
            stream_logs=ctx.stream_logs,
        )

        # Best-effort auto-fallback for hosts where `--gpus device=...` fails at runtime.
        # Some runtimes only fail when passing a device list, so we also retry here even if
        # the earlier probe passed.
        if (
            rc != 0
            and not timed_out
            and self._gpu_mode_requested(ctx) == "auto"
            and "--gpus" in docker_cmd
            and any(x.startswith("device=") for x in docker_cmd)
        ):
            stderr_text = ""
            stdout_text = ""
            with suppress(Exception):
                stderr_text = (step_dir / "stderr.log").read_text(
                    encoding="utf-8", errors="replace"
                )
            with suppress(Exception):
                stdout_text = (step_dir / "stdout.log").read_text(
                    encoding="utf-8", errors="replace"
                )
            if self._should_retry_gpu_mode_fallback(
                stderr_text=stderr_text, stdout_text=stdout_text
            ):
                self._gpu_mode_selected = "nvidia_visible_devices"
                attempt = 2
                self._archive_attempt_files(step_dir=step_dir, attempt=1)
                ctx2 = replace(ctx, env=dict(ctx.env))
                docker_cmd = self._docker_run_argv(
                    ctx2, step_id=step_id, cmd=cmd, step_artifacts_dir=step_artifacts_dir
                )
                write_text(
                    step_dir / "command.txt", " ".join(shlex.quote(x) for x in docker_cmd) + "\n"
                )
                rc, timed_out = self._run_docker_once(
                    docker_cmd=docker_cmd,
                    cwd=ctx.project_root,
                    step_dir=step_dir,
                    timeout_seconds=timeout_seconds,
                    stream_logs=ctx.stream_logs,
                )

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
                "attempt": attempt,
            },
        )
        write_json(step_dir / "exec.json", result.__dict__)
        return result

    def _stream_process(
        self,
        proc: subprocess.Popen[bytes],
        *,
        stdout_fp,
        stderr_fp,
        timeout_seconds: int | None,
    ) -> int:
        sel = selectors.DefaultSelector()
        assert proc.stdout is not None
        assert proc.stderr is not None
        sel.register(
            proc.stdout, selectors.EVENT_READ, data=("stdout", stdout_fp, sys.stdout.buffer)
        )
        sel.register(
            proc.stderr, selectors.EVENT_READ, data=("stderr", stderr_fp, sys.stderr.buffer)
        )

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
                if proc.poll() is not None and not sel.get_map():
                    break
                continue

            for key, _ in events:
                _, file_fp, live_fp = key.data
                try:
                    chunk = os.read(_selector_fd(key.fileobj), 64 * 1024)
                except Exception:
                    chunk = b""
                if not chunk:
                    with suppress(Exception):
                        sel.unregister(key.fileobj)
                    continue

                file_fp.write(chunk)
                file_fp.flush()
                # Always stream when ctx.stream_logs is true; stdout/stderr keep their destinations.
                live_fp.write(chunk)
                live_fp.flush()

        # Drain any remaining output quickly after process exit/kill.
        with suppress(Exception):
            proc.wait(timeout=5)

        if killed:
            return 124
        return int(proc.returncode or 0)

    def finalize_run(self, _ctx: RunContext) -> None:
        return

    def docker_metadata(self) -> dict[str, Any]:
        return self._image_meta or {}


def _selector_fd(fileobj: object) -> int:
    if isinstance(fileobj, int):
        return fileobj
    if hasattr(fileobj, "fileno"):
        return cast(_HasFileno, fileobj).fileno()
    raise TypeError("Selector file object does not provide fileno()")


class _HasFileno(Protocol):
    def fileno(self) -> int: ...
