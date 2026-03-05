from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.config import Roots
from exp_harness.errors import RunResumeError, StepExecutionError
from exp_harness.runner import resume_experiment, run_experiment


def _write_spec(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_resume_continues_from_first_incomplete_step(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_it",
            "env": {"kind": "local"},
            "steps": [
                {
                    "id": "a",
                    "cmd": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('a.done').write_text('ok')",
                    ],
                },
                {
                    "id": "b",
                    "cmd": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; import os,sys; "
                            "flag=Path(os.environ['EXP_HARNESS_RUN_DIR'])/'resume.flag'; "
                            "sys.exit(0 if flag.exists() else 2)"
                        ),
                    ],
                },
                {"id": "c", "cmd": [sys.executable, "-c", "print('c')"]},
            ],
        },
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

    run_json_fp = next((roots.runs_root / "resume_it").glob("*/run.json"))
    run_dir = run_json_fp.parent
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    run_key = str(run_json["run_key"])
    assert run_json["steps"][0]["state"] == "succeeded"
    assert run_json["steps"][1]["state"] == "failed"
    assert run_json["steps"][2]["state"] == "skipped"

    (run_dir / "resume.flag").write_text("ok", encoding="utf-8")

    resumed = resume_experiment(
        roots=roots,
        name="resume_it",
        run_key=run_key,
        enforce_clean=False,
        follow_steps=False,
    )
    assert resumed["run_key"] == run_key
    run_json_after = json.loads(run_json_fp.read_text(encoding="utf-8"))
    assert run_json_after["state"] == "succeeded"

    steps = run_json_after["steps"]
    assert steps[0]["state"] == "succeeded"
    assert steps[0]["attempt"] == 1
    assert steps[1]["state"] == "succeeded"
    assert steps[1]["attempt"] >= 2
    assert steps[2]["state"] == "succeeded"
    assert steps[2]["attempt"] == 1
    attempts = run_json_after.get("resume_attempts") or []
    assert attempts
    assert attempts[-1]["resumed_from_step"] == "b"


def test_resume_refuses_succeeded_run_without_force(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_done",
            "env": {"kind": "local"},
            "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('ok')"]}],
        },
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
    with pytest.raises(RunResumeError, match="already succeeded"):
        resume_experiment(
            roots=roots,
            name="resume_done",
            run_key=res["run_key"],
            enforce_clean=False,
        )


def test_resume_refuses_spec_drift_without_override(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_drift",
            "env": {"kind": "local"},
            "steps": [
                {"id": "a", "cmd": [sys.executable, "-c", "print('ok')"]},
                {"id": "b", "cmd": [sys.executable, "-c", "import sys; sys.exit(1)"]},
            ],
        },
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

    run_json_fp = next((roots.runs_root / "resume_drift").glob("*/run.json"))
    run_dir = run_json_fp.parent
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    run_key = str(run_json["run_key"])

    spec_copy = run_dir / "spec.yaml"
    spec_copy.write_text(spec_copy.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")

    with pytest.raises(RunResumeError, match="Spec drift detected"):
        resume_experiment(
            roots=roots,
            name="resume_drift",
            run_key=run_key,
            enforce_clean=False,
        )
