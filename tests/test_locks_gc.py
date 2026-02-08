from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp_harness.config import Roots
from exp_harness.locks import gc_locks


def test_locks_gc_lists_and_removes_orphans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    locks_dir = roots.runs_root / "_locks"
    locks_dir.mkdir(parents=True)
    lock_fp = locks_dir / "gpu0.lock"
    lock_fp.write_text(
        json.dumps(
            {
                "pid": 999999,
                "proc_start_ticks": 1,
                "started_at_utc": "2000-01-01T00:00:00Z",
                "run_key": "rk",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("exp_harness.locks.which", lambda _: None)

    gc_locks(roots=roots, grace_seconds=0, force=False)
    out = capsys.readouterr().out
    assert "eligible for removal" in out

    gc_locks(roots=roots, grace_seconds=0, force=True)
    assert not lock_fp.exists()


def test_locks_gc_respects_grace_period(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    locks_dir = roots.runs_root / "_locks"
    locks_dir.mkdir(parents=True)
    lock_fp = locks_dir / "gpu0.lock"
    lock_fp.write_text(
        json.dumps(
            {
                "pid": 999999,
                "proc_start_ticks": 1,
                "started_at_utc": "2099-01-01T00:00:00Z",
                "run_key": "rk",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("exp_harness.locks.which", lambda _: None)
    gc_locks(roots=roots, grace_seconds=600, force=True)
    assert lock_fp.exists()


def test_locks_gc_skips_when_docker_container_running_unless_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    locks_dir = roots.runs_root / "_locks"
    locks_dir.mkdir(parents=True)
    lock_fp = locks_dir / "gpu0.lock"
    lock_fp.write_text(
        json.dumps(
            {
                "pid": 999999,
                "proc_start_ticks": 1,
                "started_at_utc": "2000-01-01T00:00:00Z",
                "run_key": "rk",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("exp_harness.locks.which", lambda _: "docker")

    def fake_shell_out(argv, cwd=None):
        if argv[:2] == ["docker", "ps"]:
            return 0, "abc123\n", ""
        return 1, "", "unexpected"

    monkeypatch.setattr("exp_harness.locks.shell_out", fake_shell_out)

    gc_locks(roots=roots, grace_seconds=0, force=False)
    assert lock_fp.exists()

    gc_locks(roots=roots, grace_seconds=0, force=True)
    assert not lock_fp.exists()
