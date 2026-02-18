from __future__ import annotations

from pathlib import Path

from exp_harness.executors.base import RunContext
from exp_harness.executors.docker import DockerExecutor


def test_docker_argv_includes_runtime_knobs_and_env(monkeypatch, tmp_path: Path) -> None:
    ctx = RunContext(
        name="n",
        run_key="k",
        project_root=tmp_path,
        run_dir=tmp_path / "runs" / "n" / "k",
        artifacts_dir=tmp_path / "artifacts" / "n" / "k",
        workdir="/workspace",
        env={"FOO": "BAR"},
        offline=False,
        kind="docker",
        docker={
            "image": "img:tag",
            "network": "bridge",
            "ipc": "host",
            "shm_size": "2g",
            "mounts": [{"host": str(tmp_path), "container": "/workspace"}],
        },
        allocated_gpus_host=[],
    )
    ex = DockerExecutor()
    # Avoid the docker CLI even if future tests add allocated GPUs.
    monkeypatch.setattr(ex, "_supports_docker_gpus_device", lambda _ctx: True)
    argv = ex._docker_run_argv(
        ctx, step_id="s", cmd=["echo", "hi"], step_artifacts_dir="/workspace/artifacts/n/k/s"
    )
    s = "\n".join(argv)
    assert "--network" in argv and "bridge" in argv
    assert "--ipc=host" in argv
    assert "--shm-size=2g" in argv
    assert "-e" in argv and "FOO=BAR" in s
    assert "EXP_HARNESS_RUN_DIR=/workspace/runs/n/k" in s
    assert "EXP_HARNESS_ARTIFACTS_DIR=/workspace/artifacts/n/k" in s
    assert "EXP_HARNESS_STEP_ARTIFACTS_DIR=/workspace/artifacts/n/k/s" in s
