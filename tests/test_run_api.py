from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from exp_harness.run.api import (
    OverrideParseError,
    compose_experiment_config,
    parse_set_overrides,
    run_experiment,
)
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


def test_compose_experiment_config_defaults_are_valid() -> None:
    cfg = compose_experiment_config()
    assert cfg["name"] == "default_experiment"
    assert cfg["env"]["kind"] == "local"
    assert cfg["resources"]["gpus"] == 0
    assert "hydra" not in cfg


def test_compose_experiment_config_supports_group_overrides() -> None:
    cfg = compose_experiment_config(overrides=["env=docker", "resources=gpu1", "name=hydra_demo"])
    assert cfg["name"] == "hydra_demo"
    assert cfg["env"]["kind"] == "docker"
    assert cfg["resources"]["gpus"] == 1
    assert cfg["env"]["docker"]["image"] == "python:3.12-slim"


def test_compose_experiment_config_does_not_create_hydra_outputs(tmp_path: Path) -> None:
    old_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        _ = compose_experiment_config()
    finally:
        os.chdir(old_cwd)
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / ".hydra").exists()


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
