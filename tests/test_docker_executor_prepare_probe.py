from __future__ import annotations

import json
from pathlib import Path

from exp_harness.executors.base import RunContext
from exp_harness.executors.docker import DockerExecutor


def test_docker_prepare_run_writes_python_and_freeze(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "n" / "k"
    run_dir.mkdir(parents=True)

    ctx = RunContext(
        name="n",
        run_key="k",
        project_root=tmp_path,
        run_dir=run_dir,
        artifacts_dir=tmp_path / "artifacts" / "n" / "k",
        workdir="/workspace",
        env={"FOO": "BAR"},
        offline=False,
        kind="docker",
        docker={
            "image": "img:tag",
            "network": "none",
            "mounts": [{"host": str(tmp_path), "container": "/workspace"}],
        },
        allocated_gpus_host=[],
    )

    # Avoid touching docker.
    monkeypatch.setattr(
        "exp_harness.executors.docker.inspect_image",
        lambda image, cwd: {
            "image": image,
            "image_id": "sha256:x",
            "repo_digests": [],
            "repo_tags": [],
        },
    )

    calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, stdout: str):
            self.stdout = stdout

    def fake_run(argv, cwd=None, stdout=None, stderr=None, text=None):
        calls.append(list(argv))
        if argv[-2:] == ["python", "-V"]:
            return FakeProc("Python 3.12.0\n")
        if argv[-4:] == ["python", "-m", "pip", "freeze"]:
            return FakeProc("a==1\n")
        return FakeProc("")

    monkeypatch.setattr("exp_harness.executors.docker.subprocess.run", fake_run)

    ex = DockerExecutor()
    ex.prepare_run(ctx)

    prov = run_dir / "provenance"
    mounts = json.loads((prov / "docker_mounts.json").read_text(encoding="utf-8"))
    assert mounts and mounts[0]["host_exists"] is True
    assert (prov / "python.txt").read_text(encoding="utf-8").startswith("Python")
    assert (prov / "pip_freeze.txt").read_text(encoding="utf-8") == "a==1\n"
    # Ensure we attempted docker run for both probes.
    assert any(cmd[:3] == ["docker", "run", "--rm"] for cmd in calls)
