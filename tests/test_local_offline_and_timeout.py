from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from exp_harness.errors import StepExecutionError
from exp_harness.runner import run_experiment
from tests.helpers import find_single_run_dir, tmp_roots, write_spec


def test_local_offline_injects_env_and_provenance(tmp_path: Path) -> None:
    roots = tmp_roots(tmp_path)
    spec_fp = write_spec(
        roots.project_root,
        {
            "name": "off",
            "env": {"kind": "local", "offline": True},
            "steps": [
                {
                    "id": "a",
                    "cmd": [
                        sys.executable,
                        "-c",
                        "import os; print(os.environ.get('TRANSFORMERS_OFFLINE')); print(os.environ.get('HF_HOME')); print(os.environ.get('EXP_HARNESS_STEP_ARTIFACTS_DIR'))",
                    ],
                }
            ],
        },
    )
    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
    )
    run_dir = Path(res["run_dir"])
    out = (run_dir / "steps" / "00_a" / "stdout.log").read_text(encoding="utf-8")
    assert "1\n" in out
    assert str((roots.artifacts_root / "hf_home").resolve()) in out
    assert str((roots.artifacts_root / "off" / run_dir.name / "a").resolve()) in out

    env_json = json.loads((run_dir / "provenance" / "environment.json").read_text(encoding="utf-8"))
    assert env_json["TRANSFORMERS_OFFLINE"] == "1"
    assert env_json["HF_HUB_OFFLINE"] == "1"
    assert env_json["HF_DATASETS_OFFLINE"] == "1"
    assert env_json["HF_HOME"] == str((roots.artifacts_root / "hf_home").resolve())


def test_timeout_marks_step_timed_out_and_run_failed(tmp_path: Path) -> None:
    roots = tmp_roots(tmp_path)
    spec_fp = write_spec(
        roots.project_root,
        {
            "name": "to",
            "env": {"kind": "local"},
            "steps": [
                {
                    "id": "sleepy",
                    "timeout_seconds": 1,
                    "cmd": [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(2)",
                    ],
                }
            ],
        },
    )

    with pytest.raises(StepExecutionError, match="Step failed"):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
        )

    run_dir = find_single_run_dir(roots, "to")
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["state"] == "failed"

    exec_json = json.loads(
        (run_dir / "steps" / "00_sleepy" / "exec.json").read_text(encoding="utf-8")
    )
    assert exec_json["rc"] == 124
    assert exec_json["extra"]["timed_out"] is True
