from __future__ import annotations

from pathlib import Path

from exp_harness.config import Roots
from exp_harness.docker_utils import resolve_docker_runtime


def test_offline_defaults_network_none(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    eff = resolve_docker_runtime(
        {"image": "x:y"}, offline=True, roots=roots, project_root=tmp_path, for_hashing=True
    )
    assert eff["network"] == "none"


def test_mounts_tri_state(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )

    eff_auto = resolve_docker_runtime(
        {"image": "x:y"}, offline=False, roots=roots, project_root=tmp_path, for_hashing=True
    )
    assert eff_auto["mounts"] and eff_auto["mounts"][0]["container"] == "/workspace/runs"

    eff_none = resolve_docker_runtime(
        {"image": "x:y", "mounts": []},
        offline=False,
        roots=roots,
        project_root=tmp_path,
        for_hashing=True,
    )
    assert eff_none["mounts"] == []

    eff_explicit = resolve_docker_runtime(
        {"image": "x:y", "mounts": [{"host": "./foo", "container": "/foo"}]},
        offline=False,
        roots=roots,
        project_root=tmp_path,
        for_hashing=False,
    )
    assert eff_explicit["mounts"][0]["host"].startswith(str(tmp_path))
