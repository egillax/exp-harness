from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp_harness.config import Roots
from exp_harness.logs import show_logs
from exp_harness.status import print_status


def test_status_marks_running_run_stale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    run_dir = roots.runs_root / "n" / "rk"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "name": "n",
                "run_key": "rk",
                "state": "running",
                "pid": 123,
                "proc_start_ticks": 1,
                "created_at_utc": "x",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("exp_harness.status._pid_alive", lambda pid, start_ticks: False)
    print_status(roots=roots, name=None, limit=10)
    out = capsys.readouterr().out
    assert "(stale: pid not alive)" in out


def test_logs_tail_and_default_last_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    run_dir = roots.runs_root / "n" / "rk"
    step0 = run_dir / "steps" / "00_a"
    step1 = run_dir / "steps" / "01_b"
    step0.mkdir(parents=True)
    step1.mkdir(parents=True)

    many = "\n".join([f"line{i}" for i in range(300)]) + "\n"
    (step0 / "stdout.log").write_text("hello\n", encoding="utf-8")
    (step0 / "stderr.log").write_text("", encoding="utf-8")
    (step1 / "stdout.log").write_text(many, encoding="utf-8")
    (step1 / "stderr.log").write_text("", encoding="utf-8")

    show_logs(roots=roots, name="n", run_key="rk", step=None, follow=False)
    out = capsys.readouterr().out
    assert "line299" in out
    assert "line0" not in out  # tail should drop early lines


def test_logs_missing_step_raises(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    (roots.runs_root / "n" / "rk").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        show_logs(roots=roots, name="n", run_key="rk", step="nope", follow=False)
