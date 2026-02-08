from __future__ import annotations

from pathlib import Path

import pytest

from exp_harness.config import ENV_ARTIFACTS_ROOT, ENV_RUNS_ROOT, resolve_roots


def test_defaults_relative_to_project_root(tmp_path: Path) -> None:
    roots = resolve_roots(project_root=tmp_path, runs_root=None, artifacts_root=None)
    assert roots.runs_root == (tmp_path / "experiment_results" / "runs").resolve()
    assert roots.artifacts_root == (tmp_path / "experiment_results" / "artifacts").resolve()


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_RUNS_ROOT, str(tmp_path / "runs_env"))
    monkeypatch.setenv(ENV_ARTIFACTS_ROOT, str(tmp_path / "art_env"))
    roots = resolve_roots(project_root=tmp_path, runs_root=None, artifacts_root=None)
    assert roots.runs_root == (tmp_path / "runs_env").resolve()
    assert roots.artifacts_root == (tmp_path / "art_env").resolve()


def test_cli_overrides_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_RUNS_ROOT, str(tmp_path / "runs_env"))
    roots = resolve_roots(
        project_root=tmp_path, runs_root=tmp_path / "runs_cli", artifacts_root=None
    )
    assert roots.runs_root == (tmp_path / "runs_cli").resolve()
