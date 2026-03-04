from __future__ import annotations

import sys
from pathlib import Path

import pytest

from exp_harness.interp import InterpError
from exp_harness.resolve import apply_computed_defaults, resolve_final


def test_apply_computed_defaults_sets_workdir_gpu_and_step_artifacts(tmp_path: Path) -> None:
    raw = {
        "name": "defaults",
        "env": {"kind": "local"},
        "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('hi')"]}],
    }
    with_defaults = apply_computed_defaults(raw, project_root=tmp_path, kind="local")
    assert with_defaults["env"]["workdir"] == str(tmp_path)
    assert with_defaults["resources"]["gpus"] == 0
    assert with_defaults["steps"][0]["outputs"]["artifacts_dir"] == "${run.artifacts}/a"


def test_vars_shorthand_precedence_over_params(tmp_path: Path) -> None:
    raw = {
        "name": "v",
        "vars": {"batch_size": 128},
        "params": {"batch_size": 64},
        "env": {"kind": "local", "workdir": "x=${batch_size}"},
        "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('batch=${batch_size}')"]}],
    }
    with_defaults = apply_computed_defaults(raw, project_root=tmp_path, kind="local")
    resolved = resolve_final(
        with_defaults,
        project_root=tmp_path,
        run_ctx={
            "id": "rk",
            "runs": str(tmp_path / "runs"),
            "artifacts": str(tmp_path / "artifacts"),
        },
    )
    assert resolved["env"]["workdir"] == "x=128"
    assert resolved["steps"][0]["cmd"][-1] == "print('batch=128')"


def test_vars_cannot_shadow_reserved_keys(tmp_path: Path) -> None:
    raw = {
        "name": "v2",
        "vars": {"params": "nope"},
        "params": {},
        "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('hi')"]}],
    }
    with_defaults = apply_computed_defaults(raw, project_root=tmp_path, kind="local")
    with pytest.raises(InterpError, match="reserved key"):
        resolve_final(
            with_defaults,
            project_root=tmp_path,
            run_ctx={
                "id": "rk",
                "runs": str(tmp_path / "runs"),
                "artifacts": str(tmp_path / "artifacts"),
            },
        )
