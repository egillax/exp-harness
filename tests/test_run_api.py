from __future__ import annotations

import json
import os
from pathlib import Path

from exp_harness.run.api import (
    compose_experiment_config,
    expand_hydra_sweep_overrides,
    resume_experiment,
    run_experiment,
    run_hydra_sweep,
    run_spec_experiment,
)
from tests.helpers import write_spec


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
    def _fake_run_composed_experiment(**kwargs):
        cfg = kwargs["cfg"]
        assert isinstance(cfg, dict)
        kind = str((cfg.get("env") or {}).get("kind"))
        if kind == "docker":
            raise RuntimeError("docker boom")
        return {
            "name": str(cfg.get("name")),
            "run_id": f"run-{kind}",
            "run_key": f"key-{kind}",
            "run_dir": str(tmp_path / "runs" / f"run-{kind}"),
            "artifacts_dir": str(tmp_path / "artifacts" / f"run-{kind}"),
        }

    monkeypatch.setattr(
        "exp_harness.run.api._run_composed_experiment", _fake_run_composed_experiment
    )
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


def test_run_experiment_composes_config_and_invokes_runner(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_composed_experiment(**kwargs):
        captured.update(kwargs)
        cfg = kwargs["cfg"]
        assert isinstance(cfg, dict)
        return {
            "name": str(cfg.get("name")),
            "run_id": "run-local",
            "run_key": "key-local",
            "run_dir": str(tmp_path / "runs" / "run-local"),
            "artifacts_dir": str(tmp_path / "artifacts" / "run-local"),
        }

    monkeypatch.setattr(
        "exp_harness.run.api._run_composed_experiment", _fake_run_composed_experiment
    )
    result = run_experiment(
        overrides=["name=hydra_single", "++params.lr=1e-4"],
        project_root=tmp_path,
        run_label="from-cli",
    )

    assert result["name"] == "hydra_single"
    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    assert cfg["name"] == "hydra_single"
    assert cfg["params"]["lr"] == 0.0001
    assert captured["run_label"] == "from-cli"


def test_run_experiment_api_runs_hydra_config(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"

    result = run_experiment(
        overrides=["name=api", "++params.x=7", "++params.tag='001'"],
        project_root=tmp_path,
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


def test_run_spec_experiment_runs_yaml_spec(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runs_root = tmp_path / "runs"
    artifacts_root = tmp_path / "artifacts"
    spec_fp = write_spec(
        project_root,
        {
            "name": "spec_api",
            "env": {"kind": "local"},
            "steps": [{"id": "main", "cmd": ["python", "-c", "print('hi')"]}],
        },
    )

    result = run_spec_experiment(
        spec_path=spec_fp,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
        follow_steps=False,
    )
    assert result["name"] == "spec_api"
    assert Path(result["run_dir"]).exists()


def test_resume_experiment_delegates_to_runner(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_resume_experiment(**kwargs):
        captured.update(kwargs)
        return {
            "name": str(kwargs["name"]),
            "run_id": "20260305-120000Z__resume__abc12300",
            "run_key": str(kwargs["run_key"]),
            "run_dir": str(tmp_path / "runs" / "resume"),
            "artifacts_dir": str(tmp_path / "artifacts" / "resume"),
        }

    monkeypatch.setattr("exp_harness.runner.resume_experiment", _fake_resume_experiment)
    result = resume_experiment(
        name="resume_api",
        run_key="abc123",
        project_root=tmp_path,
        allow_spec_drift=True,
        force=True,
        follow_steps=False,
    )

    assert result["name"] == "resume_api"
    assert result["run_key"] == "abc123"
    assert captured["allow_spec_drift"] is True
    assert captured["force"] is True
    assert captured["follow_steps"] is False
