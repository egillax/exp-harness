from __future__ import annotations

import json
from pathlib import Path

from exp_harness.provenance import (
    write_env_provenance,
    write_git_provenance,
    write_host_provenance,
    write_nvidia_smi,
    write_python_and_freeze,
)
from exp_harness.provenance.git import GitInfo


def test_write_git_provenance(tmp_path: Path) -> None:
    gi = GitInfo(
        commit="c",
        branch="b",
        dirty=True,
        origin_url="o",
        diff_patch="diff",
        diff_hash="h",
    )
    write_git_provenance(tmp_path, gi)
    data = json.loads((tmp_path / "git.json").read_text(encoding="utf-8"))
    assert data["commit"] == "c"
    assert (tmp_path / "git_diff.patch").read_text(encoding="utf-8") == "diff"


def test_write_env_and_host_provenance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PATH", "X")
    write_env_provenance(tmp_path, env_allow=["PATH"], explicit_env={"A": "B"})
    env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    assert env["PATH"] == "X"
    assert env["A"] == "B"

    write_host_provenance(tmp_path, extra={"k": "v"})
    host = json.loads((tmp_path / "host.json").read_text(encoding="utf-8"))
    assert host["k"] == "v"


def test_write_nvidia_smi_branches(tmp_path: Path, monkeypatch) -> None:
    # No nvidia-smi => no file.
    monkeypatch.setattr("exp_harness.provenance.which", lambda _: None)
    write_nvidia_smi(tmp_path)
    assert not (tmp_path / "nvidia_smi.txt").exists()

    # With nvidia-smi => file.
    monkeypatch.setattr("exp_harness.provenance.which", lambda _: "nvidia-smi")
    monkeypatch.setattr("exp_harness.provenance.shell_out", lambda argv, cwd=None: (0, "OK", ""))
    write_nvidia_smi(tmp_path)
    assert (tmp_path / "nvidia_smi.txt").read_text(encoding="utf-8") == "OK"


def test_write_python_and_freeze_monkeypatched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "exp_harness.provenance.shell_out", lambda argv, cwd=None, env=None: (0, "PythonX", "")
    )

    class FakeProc:
        def __init__(self):
            self.stdout = "p==1\n"

    monkeypatch.setattr("exp_harness.provenance.subprocess.run", lambda *a, **k: FakeProc())
    write_python_and_freeze(tmp_path)
    assert (tmp_path / "python.txt").read_text(encoding="utf-8").strip() == "PythonX"
    assert (tmp_path / "pip_freeze.txt").read_text(encoding="utf-8") == "p==1\n"
