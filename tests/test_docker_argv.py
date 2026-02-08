from __future__ import annotations

from pathlib import Path

import pytest

from exp_harness.executors.base import RunContext
from exp_harness.executors.docker import DockerExecutor


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        name="toy",
        run_key="rk",
        project_root=tmp_path,
        run_dir=tmp_path / "runs" / "toy" / "rk",
        artifacts_dir=tmp_path / "artifacts" / "toy" / "rk",
        workdir="/workspace",
        env={},
        offline=False,
        kind="docker",
        docker={
            "image": "img:tag",
            "network": "none",
            "mounts": [{"host": str(tmp_path / "runs"), "container": "/workspace/runs"}],
        },
        allocated_gpus_host=[2, 3],
    )


def test_docker_argv_does_not_force_cuda_visible_devices(tmp_path: Path) -> None:
    ex = DockerExecutor()
    ctx = _ctx(tmp_path)
    argv = ex._docker_run_argv(ctx, step_id="s", cmd=["echo", "hi"], step_artifacts_dir=None)
    s = "\n".join(argv)
    assert "--gpus" in argv
    assert "device=2,3" in s
    assert "CUDA_VISIBLE_DEVICES" not in s
    assert "EXP_HARNESS_HOST_GPU_IDS=2,3" in s


def test_docker_argv_requires_resolved_mounts(tmp_path: Path) -> None:
    ex = DockerExecutor()
    ctx = _ctx(tmp_path)
    ctx = ctx.__class__(**{**ctx.__dict__, "docker": {**(ctx.docker or {}), "mounts": None}})
    with pytest.raises(RuntimeError, match="Resolved docker mounts missing"):
        ex._docker_run_argv(ctx, step_id="s", cmd=["echo", "hi"], step_artifacts_dir=None)
