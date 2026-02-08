from __future__ import annotations

import sys
from pathlib import Path

import yaml

from exp_harness.config import Roots
from exp_harness.runner import run_experiment


def test_run_key_does_not_depend_on_roots_for_local_runs(tmp_path: Path) -> None:
    project_root = tmp_path / "proj"
    project_root.mkdir()
    spec_fp = project_root / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "portable",
                "env": {"kind": "local"},
                "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('x')"]}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    roots_a = Roots(
        project_root=project_root, runs_root=tmp_path / "runsA", artifacts_root=tmp_path / "artA"
    )
    roots_b = Roots(
        project_root=project_root, runs_root=tmp_path / "runsB", artifacts_root=tmp_path / "artB"
    )

    res_a = run_experiment(
        spec_path=spec_fp,
        roots=roots_a,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
    )
    res_b = run_experiment(
        spec_path=spec_fp,
        roots=roots_b,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
    )
    assert res_a["run_key"] == res_b["run_key"]
