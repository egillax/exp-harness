from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from exp_harness.cli import app
from tests.helpers import write_spec


def test_cli_run_status_logs_inspect_and_locks_gc(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"

    spec_fp = write_spec(
        project_root,
        {
            "name": "cli",
            "env": {"kind": "local"},
            "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('hi')"]}],
        },
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "run",
            str(spec_fp),
            "--runs-root",
            str(runs_root),
            "--artifacts-root",
            str(artifacts_root),
            "--salt",
            "s",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    parts = res.stdout.splitlines()[0].split()
    assert parts[0] == "cli"
    run_key = parts[1]

    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        res2 = runner.invoke(app, ["status", "--runs-root", str(runs_root), "--limit", "5"])
    finally:
        os.chdir(old_cwd)
    assert res2.exit_code == 0
    assert run_key in res2.stdout

    res3 = runner.invoke(
        app, ["logs", "cli", run_key, "--runs-root", str(runs_root), "--step", "a"]
    )
    assert res3.exit_code == 0
    assert "stdout.log" in res3.stdout

    res4 = runner.invoke(app, ["inspect", "cli", run_key, "--runs-root", str(runs_root)])
    assert res4.exit_code == 0
    assert f"run_key: {run_key}" in res4.stdout

    # locks gc: create an orphan lock and ensure it is listed/removed.
    lock_fp = runs_root / "_locks" / "gpu0.lock"
    lock_fp.parent.mkdir(parents=True, exist_ok=True)
    lock_fp.write_text(
        json.dumps(
            {"pid": 999999, "proc_start_ticks": 1, "started_at_utc": "2000-01-01T00:00:00Z"}
        ),
        encoding="utf-8",
    )
    res5 = runner.invoke(
        app, ["locks", "gc", "--runs-root", str(runs_root), "--grace-seconds", "0"]
    )
    assert res5.exit_code == 0
    assert "eligible for removal" in res5.stdout
    assert lock_fp.exists()

    res6 = runner.invoke(
        app, ["locks", "gc", "--runs-root", str(runs_root), "--grace-seconds", "0", "--force"]
    )
    assert res6.exit_code == 0
    assert not lock_fp.exists()
