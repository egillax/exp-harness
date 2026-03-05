from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.config import Roots
from exp_harness.errors import StepExecutionError
from exp_harness.runner import run_experiment


def _read_run_json(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def test_run_json_persists_step_lifecycle_success(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "lifecycle_ok",
                "env": {"kind": "local"},
                "steps": [
                    {"id": "a", "cmd": [sys.executable, "-c", "print('a')"]},
                    {"id": "b", "cmd": [sys.executable, "-c", "print('b')"]},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
        follow_steps=False,
    )

    run_json = _read_run_json(Path(res["run_dir"]))
    steps = run_json["steps"]
    assert [s["step_id"] for s in steps] == ["a", "b"]
    assert all(s["state"] == "succeeded" for s in steps)
    assert all(s["attempt"] == 1 for s in steps)
    assert all(s["rc"] == 0 for s in steps)
    assert all(isinstance(s["started_at_utc"], str) and s["started_at_utc"] for s in steps)
    assert all(isinstance(s["finished_at_utc"], str) and s["finished_at_utc"] for s in steps)


def test_run_json_persists_step_lifecycle_failure_and_skips_remaining(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "lifecycle_fail",
                "env": {"kind": "local"},
                "steps": [
                    {"id": "a", "cmd": [sys.executable, "-c", "print('ok')"]},
                    {"id": "b", "cmd": [sys.executable, "-c", "import sys; sys.exit(3)"]},
                    {"id": "c", "cmd": [sys.executable, "-c", "print('should_skip')"]},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(StepExecutionError):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_fp = next((roots.runs_root / "lifecycle_fail").glob("*/run.json"))
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    steps = run_json["steps"]
    assert [s["step_id"] for s in steps] == ["a", "b", "c"]

    assert steps[0]["state"] == "succeeded"
    assert steps[0]["rc"] == 0

    assert steps[1]["state"] == "failed"
    assert steps[1]["rc"] == 3
    assert isinstance((steps[1].get("error") or {}).get("message"), str)

    assert steps[2]["state"] == "skipped"
    assert steps[2]["rc"] is None
