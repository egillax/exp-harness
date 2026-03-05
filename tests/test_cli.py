from __future__ import annotations

import json
import os
from pathlib import Path

from typer.testing import CliRunner

from exp_harness.cli import app
from tests.helpers import write_spec


def test_cli_run_status_logs_inspect_and_locks_gc(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"

    runner = CliRunner()
    old_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        res = runner.invoke(
            app,
            [
                "run",
                "name=cli",
                "--runs-root",
                str(runs_root),
                "--artifacts-root",
                str(artifacts_root),
                "--salt",
                "s",
                "--follow-steps",
                "--stderr-tail-lines",
                "0",
            ],
        )
        assert res.exit_code == 0, res.stdout + res.stderr
        lines = res.stdout.splitlines()
        summary = next((ln for ln in lines if ln.startswith("cli ")), "")
        assert summary, f"missing summary line in stdout:\n{res.stdout}"
        parts = summary.split()
        assert parts[0] == "cli"
        run_key = parts[1]

        res2 = runner.invoke(app, ["status", "--runs-root", str(runs_root), "--limit", "5"])
        assert res2.exit_code == 0
        assert run_key in res2.stdout
    finally:
        os.chdir(old_cwd)

    res3 = runner.invoke(
        app, ["logs", "cli", run_key, "--runs-root", str(runs_root), "--step", "main"]
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


def test_cli_run_defaults_follow_steps_true_and_allows_opt_out(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"

    captured: list[bool] = []

    def _fake_run_experiment(**kwargs):
        captured.append(bool(kwargs["follow_steps"]))
        return {
            "name": "cli",
            "run_id": "20260224-120000Z__cli__abc12300",
            "run_key": "abc123",
            "run_dir": str(runs_root / "cli" / "abc123"),
            "artifacts_dir": str(artifacts_root / "cli" / "abc123"),
        }

    monkeypatch.setattr("exp_harness.run.api.run_experiment", _fake_run_experiment)

    runner = CliRunner()
    res_default = runner.invoke(
        app,
        [
            "run",
            "name=cli",
            "--runs-root",
            str(runs_root),
            "--artifacts-root",
            str(artifacts_root),
        ],
    )
    assert res_default.exit_code == 0, res_default.stdout + res_default.stderr
    assert captured[-1] is True

    res_no_follow = runner.invoke(
        app,
        [
            "run",
            "name=cli",
            "--runs-root",
            str(runs_root),
            "--artifacts-root",
            str(artifacts_root),
            "--no-follow-steps",
        ],
    )
    assert res_no_follow.exit_code == 0, res_no_follow.stdout + res_no_follow.stderr
    assert captured[-1] is False


def test_cli_passes_run_label_override(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"

    captured: list[str | None] = []

    def _fake_run_experiment(**kwargs):
        captured.append(kwargs.get("run_label"))
        return {
            "name": "cli",
            "run_id": "20260224-120000Z__from-cli__abc12300",
            "run_key": "abc123",
            "run_dir": str(runs_root / "cli" / "abc123"),
            "artifacts_dir": str(artifacts_root / "cli" / "abc123"),
        }

    monkeypatch.setattr("exp_harness.run.api.run_experiment", _fake_run_experiment)
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "run",
            "name=cli",
            "--runs-root",
            str(runs_root),
            "--artifacts-root",
            str(artifacts_root),
            "--run-label",
            "from-cli",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    assert captured == ["from-cli"]


def test_cli_run_invokes_hydra_api_with_overrides(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"
    captured: dict[str, object] = {}

    def _fake_run_experiment(**kwargs):
        captured.update(kwargs)
        return {
            "name": "hydra_cli",
            "run_id": "20260224-120000Z__hydra__abc12300",
            "run_key": "abc123",
            "run_dir": str(runs_root / "hydra_cli" / "abc123"),
            "artifacts_dir": str(artifacts_root / "hydra_cli" / "abc123"),
        }

    monkeypatch.setattr("exp_harness.run.api.run_experiment", _fake_run_experiment)
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "run",
            "name=hydra_cli",
            "resources=default",
            "--runs-root",
            str(runs_root),
            "--artifacts-root",
            str(artifacts_root),
            "--no-follow-steps",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    assert "hydra_cli abc123" in res.stdout
    assert captured["overrides"] == ["name=hydra_cli", "resources=default"]
    assert captured["follow_steps"] is False


def test_cli_run_spec_executes_spec_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"

    spec_fp = write_spec(
        project_root,
        {
            "name": "spec_cli",
            "env": {"kind": "local"},
            "steps": [{"id": "main", "cmd": ["python", "-c", "print('hi')"]}],
        },
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "run-spec",
            str(spec_fp),
            "--runs-root",
            str(runs_root),
            "--artifacts-root",
            str(artifacts_root),
            "--no-follow-steps",
        ],
    )

    assert res.exit_code == 0, res.stdout + res.stderr
    assert "spec_cli " in res.stdout


def test_cli_resume_invokes_resume_api(tmp_path: Path, monkeypatch) -> None:
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"
    captured: dict[str, object] = {}

    def _fake_resume_experiment(**kwargs):
        captured.update(kwargs)
        return {
            "name": "resume_cli",
            "run_id": "20260305-120000Z__resume__abc12300",
            "run_key": "abc123",
            "run_dir": str(runs_root / "resume_cli" / "abc123"),
            "artifacts_dir": str(artifacts_root / "resume_cli" / "abc123"),
        }

    monkeypatch.setattr("exp_harness.run.api.resume_experiment", _fake_resume_experiment)
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "resume",
            "resume_cli",
            "abc123",
            "--runs-root",
            str(runs_root),
            "--artifacts-root",
            str(artifacts_root),
            "--allow-spec-drift",
            "--force",
            "--no-follow-steps",
        ],
    )
    assert res.exit_code == 0, res.stdout + res.stderr
    assert "resume_cli abc123" in res.stdout
    assert captured["name"] == "resume_cli"
    assert captured["run_key"] == "abc123"
    assert captured["allow_spec_drift"] is True
    assert captured["force"] is True
    assert captured["follow_steps"] is False
