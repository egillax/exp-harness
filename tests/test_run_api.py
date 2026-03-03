from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from exp_harness.run.api import OverrideParseError, parse_set_overrides, run_experiment
from tests.helpers import write_spec


def test_parse_set_overrides_parses_key_value_pairs() -> None:
    assert parse_set_overrides(["params.x=3", "params.y = abc"]) == [
        ("params.x", "3"),
        ("params.y", " abc"),
    ]


def test_parse_set_overrides_errors_on_invalid_assignments() -> None:
    with pytest.raises(OverrideParseError, match="Expected KEY=VALUE"):
        parse_set_overrides(["params.x"])
    with pytest.raises(OverrideParseError, match="Empty key"):
        parse_set_overrides(["=1"])


def test_run_experiment_api_runs_spec_and_applies_overrides(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"
    spec_fp = write_spec(
        project_root,
        {
            "name": "api",
            "env": {"kind": "local"},
            "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('hi')"]}],
        },
    )

    result = run_experiment(
        spec_path=spec_fp,
        set_overrides=[("params.x", "7")],
        set_string_overrides=[("params.tag", "001")],
        runs_root=runs_root,
        artifacts_root=artifacts_root,
        salt="from-api",
        follow_steps=False,
    )

    run_dir = Path(result["run_dir"])
    assert run_dir.exists()
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["state"] == "succeeded"

    resolved = json.loads((run_dir / "resolved_spec.json").read_text(encoding="utf-8"))
    assert resolved["params"]["x"] == 7
    assert resolved["params"]["tag"] == "001"
