from __future__ import annotations

import subprocess

from exp_harness.utils import shell_out


def test_shell_out_missing_executable_returns_127(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("no such file or directory: git")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, out, err = shell_out(["git", "rev-parse", "HEAD"], cwd=None)
    assert rc == 127
    assert out == ""
    assert "no such file or directory" in err.lower()
