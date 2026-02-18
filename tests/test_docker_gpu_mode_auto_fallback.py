from __future__ import annotations

from pathlib import Path

from exp_harness.executors.base import RunContext
from exp_harness.executors.docker import DockerExecutor


class _FakeProc:
    def __init__(self, *, returncode: int, stdout: str):
        self.returncode = returncode
        self.stdout = stdout


def test_gpu_mode_auto_falls_back_to_nvidia_visible_devices(monkeypatch, tmp_path: Path) -> None:
    def fake_run(argv, cwd=None, stdout=None, stderr=None, text=None):
        # Called by _supports_docker_gpus_device
        if argv[:5] == ["docker", "run", "--rm", "--gpus", "device=2"]:
            return _FakeProc(
                returncode=1,
                stdout="docker: Error response from daemon: cannot set both Count and DeviceIDs on device request\n",
            )
        return _FakeProc(returncode=0, stdout="")

    monkeypatch.setattr("exp_harness.executors.docker.subprocess.run", fake_run)

    ctx = RunContext(
        name="n",
        run_key="k",
        project_root=tmp_path,
        run_dir=tmp_path / "runs" / "n" / "k",
        artifacts_dir=tmp_path / "artifacts" / "n" / "k",
        workdir="/workspace",
        env={},
        offline=False,
        kind="docker",
        docker={
            "image": "img:tag",
            "network": "none",
            "runtime": "nvidia",
            "mounts": [{"host": str(tmp_path), "container": "/workspace"}],
        },
        allocated_gpus_host=[2, 3],
    )

    ex = DockerExecutor()
    argv = ex._docker_run_argv(ctx, step_id="s", cmd=["true"], step_artifacts_dir=None)
    s = "\n".join(argv)

    assert "--gpus" not in argv
    assert "--runtime=nvidia" in argv
    assert "NVIDIA_VISIBLE_DEVICES=2,3" in s
    assert "CUDA_VISIBLE_DEVICES=0,1" in s
    assert "EXP_HARNESS_HOST_GPU_IDS=2,3" in s


def test_gpu_mode_auto_uses_docker_gpus_device_when_supported(monkeypatch, tmp_path: Path) -> None:
    # Avoid real docker calls; pretend the probe succeeds.
    monkeypatch.setattr(DockerExecutor, "_supports_docker_gpus_device", lambda _self, _ctx: True)

    ctx = RunContext(
        name="n",
        run_key="k",
        project_root=tmp_path,
        run_dir=tmp_path / "runs" / "n" / "k",
        artifacts_dir=tmp_path / "artifacts" / "n" / "k",
        workdir="/workspace",
        env={},
        offline=False,
        kind="docker",
        docker={
            "image": "img:tag",
            "network": "none",
            "mounts": [{"host": str(tmp_path), "container": "/workspace"}],
        },
        allocated_gpus_host=[2, 3],
    )

    ex = DockerExecutor()
    argv = ex._docker_run_argv(ctx, step_id="s", cmd=["true"], step_artifacts_dir=None)
    s = "\n".join(argv)
    assert "--gpus" in argv
    assert "device=2,3" in s
    assert "NVIDIA_VISIBLE_DEVICES" not in s
    assert "CUDA_VISIBLE_DEVICES" not in s
