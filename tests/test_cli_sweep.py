from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from exp_harness.cli import app


def test_cli_sweep_success_exit_zero(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"

    def _fake_run_hydra_sweep(**kwargs):
        return {
            "total": 2,
            "succeeded": 2,
            "failed": 0,
            "summary_path": str(runs_root / "_sweeps" / "summary.json"),
            "runs": [
                {
                    "index": 1,
                    "overrides": ["name=sweep", "params.x=1"],
                    "status": "succeeded",
                    "result": {
                        "name": "sweep",
                        "run_id": "run-1",
                        "run_key": "key-1",
                        "run_dir": str(runs_root / "run-1"),
                        "artifacts_dir": str(artifacts_root / "run-1"),
                    },
                    "error": None,
                    "attempts": [
                        {
                            "attempt": 1,
                            "status": "succeeded",
                            "error": None,
                            "run_id": "run-1",
                            "run_key": "key-1",
                        }
                    ],
                },
                {
                    "index": 2,
                    "overrides": ["name=sweep", "params.x=2"],
                    "status": "succeeded",
                    "result": {
                        "name": "sweep",
                        "run_id": "run-2",
                        "run_key": "key-2",
                        "run_dir": str(runs_root / "run-2"),
                        "artifacts_dir": str(artifacts_root / "run-2"),
                    },
                    "error": None,
                    "attempts": [
                        {
                            "attempt": 1,
                            "status": "succeeded",
                            "error": None,
                            "run_id": "run-2",
                            "run_key": "key-2",
                        }
                    ],
                },
            ],
        }

    monkeypatch.setattr("exp_harness.run.api.run_hydra_sweep", _fake_run_hydra_sweep)

    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        runner = CliRunner()
        res = runner.invoke(app, ["sweep", "name=sweep", "params.x=1,2"])
    finally:
        os.chdir(old_cwd)

    assert res.exit_code == 0, res.stdout + res.stderr
    assert "sweep total=2 succeeded=2 failed=0" in res.stdout
    assert "[1/2] ok sweep key-1" in res.stdout
    assert "sweep summary:" in res.stdout


def test_cli_sweep_failure_exit_nonzero(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()

    def _fake_run_hydra_sweep(**kwargs):
        return {
            "total": 2,
            "succeeded": 1,
            "failed": 1,
            "summary_path": "/tmp/sweeps/summary.json",
            "runs": [
                {
                    "index": 1,
                    "overrides": ["name=sweep", "params.x=1"],
                    "status": "succeeded",
                    "result": {
                        "name": "sweep",
                        "run_id": "run-1",
                        "run_key": "key-1",
                        "run_dir": "/tmp/runs/run-1",
                        "artifacts_dir": "/tmp/artifacts/run-1",
                    },
                    "error": None,
                    "attempts": [
                        {
                            "attempt": 1,
                            "status": "succeeded",
                            "error": None,
                            "run_id": "run-1",
                            "run_key": "key-1",
                        }
                    ],
                },
                {
                    "index": 2,
                    "overrides": ["name=sweep", "params.x=2"],
                    "status": "failed",
                    "result": None,
                    "error": "boom",
                    "attempts": [
                        {
                            "attempt": 1,
                            "status": "failed",
                            "error": "boom",
                            "run_id": None,
                            "run_key": None,
                        }
                    ],
                },
            ],
        }

    monkeypatch.setattr("exp_harness.run.api.run_hydra_sweep", _fake_run_hydra_sweep)

    import os

    old_cwd = os.getcwd()
    try:
        os.chdir(project_root)
        runner = CliRunner()
        res = runner.invoke(app, ["sweep", "name=sweep", "params.x=1,2"])
    finally:
        os.chdir(old_cwd)

    assert res.exit_code == 1
    out = res.stdout + res.stderr
    assert "[2/2] failed (name=sweep params.x=2): boom" in out
    assert "sweep total=2 succeeded=1 failed=1" in out
