from __future__ import annotations

import subprocess
from pathlib import Path

from exp_harness.executors.base import RunContext
from exp_harness.executors.docker import DockerExecutor


class _FakePopen:
    def __init__(self, argv, *, returncode: int):
        self.argv = list(argv)
        self.returncode = int(returncode)

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        return None


def test_docker_executor_retries_with_nvidia_visible_devices(monkeypatch, tmp_path: Path) -> None:
    # Avoid the probe; pretend the host supports docker_gpus_device so we hit the retry path.
    monkeypatch.setattr(DockerExecutor, "_supports_docker_gpus_device", lambda _self, _ctx: True)

    popen_calls: list[list[str]] = []

    def fake_popen(argv, cwd=None, stdout=None, stderr=None):
        popen_calls.append(list(argv))
        # First run: fail with the exact daemon error message written to stderr.log.
        if "--gpus" in argv:
            assert stderr is not None
            stderr.write(
                b"docker: Error response from daemon: cannot set both Count and DeviceIDs on device request\n"
            )
            stderr.flush()
            return _FakePopen(argv, returncode=125)
        # Second run: succeed.
        return _FakePopen(argv, returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    step_dir = tmp_path / "step"
    step_dir.mkdir()
    run_dir = tmp_path / "runs"
    run_dir.mkdir()

    ctx = RunContext(
        name="n",
        run_key="k",
        project_root=tmp_path,
        run_dir=run_dir,
        artifacts_dir=tmp_path / "artifacts",
        workdir="/workspace",
        env={},
        offline=False,
        kind="docker",
        docker={
            "image": "img:tag",
            "network": "none",
            "runtime": "nvidia",
            "mounts": [{"host": str(tmp_path), "container": "/workspace"}],
            "gpu_mode": "auto",
        },
        allocated_gpus_host=[2, 3],
        stream_logs=False,
    )

    ex = DockerExecutor()
    res = ex.run_step(
        ctx,
        step_index=0,
        step_id="s",
        cmd=["true"],
        step_dir=step_dir,
        timeout_seconds=10,
        step_artifacts_dir=None,
    )

    assert res.rc == 0
    # First attempt used --gpus device=...
    assert any("--gpus" in call for call in popen_calls)
    # Second attempt should include NVIDIA_VISIBLE_DEVICES env var.
    cmd_text = (step_dir / "command.txt").read_text(encoding="utf-8")
    assert "NVIDIA_VISIBLE_DEVICES=2,3" in cmd_text
    assert (step_dir / "command.attempt1.txt").exists()
    assert (step_dir / "stderr.attempt1.log").exists()
