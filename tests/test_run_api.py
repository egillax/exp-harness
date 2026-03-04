from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.run.api import (
    OverrideParseError,
    compose_experiment_config,
    expand_hydra_sweep_overrides,
    parse_set_overrides,
    run_experiment,
    run_hydra_experiment,
    run_hydra_sweep,
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


def test_expand_hydra_sweep_overrides_builds_cartesian_product() -> None:
    combos = expand_hydra_sweep_overrides(
        ["name=demo", "params.lr=1e-3,1e-4", "resources=default,gpu1"]
    )
    assert len(combos) == 4
    assert ["name=demo", "params.lr=0.001", "resources=default"] in combos
    assert ["name=demo", "params.lr=0.0001", "resources=gpu1"] in combos


def test_run_hydra_sweep_collects_partial_failures(tmp_path: Path, monkeypatch) -> None:
    def _fake_run_experiment(**kwargs):
        spec_path = Path(kwargs["spec_path"])
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        kind = str(spec.get("env", {}).get("kind"))
        if kind == "docker":
            raise RuntimeError("docker boom")
        return {
            "name": str(spec.get("name")),
            "run_id": f"run-{kind}",
            "run_key": f"key-{kind}",
            "run_dir": str(tmp_path / "runs" / f"run-{kind}"),
            "artifacts_dir": str(tmp_path / "artifacts" / f"run-{kind}"),
        }

    monkeypatch.setattr("exp_harness.run.api.run_experiment", _fake_run_experiment)
    result = run_hydra_sweep(
        overrides=["name=sweep_api", "env=local,docker"],
        project_root=tmp_path,
    )

    assert result["total"] == 2
    assert result["succeeded"] == 1
    assert result["failed"] == 1
    failed = [item for item in result["runs"] if item["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["error"] == "docker boom"


def test_run_hydra_experiment_composes_config_and_invokes_runner(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_experiment(**kwargs):
        spec_path = Path(kwargs["spec_path"])
        spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return {
            "name": str(spec.get("name")),
            "run_id": "run-local",
            "run_key": "key-local",
            "run_dir": str(tmp_path / "runs" / "run-local"),
            "artifacts_dir": str(tmp_path / "artifacts" / "run-local"),
        }

    monkeypatch.setattr("exp_harness.run.api.run_experiment", _fake_run_experiment)
    result = run_hydra_experiment(
        overrides=["name=hydra_single", "++params.lr=1e-4"],
        project_root=tmp_path,
        run_label="from-cli",
    )

    assert result["name"] == "hydra_single"
    spec = captured["spec"]
    assert isinstance(spec, dict)
    assert spec["name"] == "hydra_single"
    assert spec["params"]["lr"] == 0.0001
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["run_label"] == "from-cli"


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
