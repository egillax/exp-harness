from __future__ import annotations

from pathlib import Path

import pytest

from exp_harness.runner import run_experiment
from tests.helpers import tmp_roots, write_spec


def test_run_already_exists_and_salt_changes_run_key(tmp_path: Path) -> None:
    roots = tmp_roots(tmp_path)
    spec_fp = write_spec(
        roots.project_root,
        {"name": "x", "env": {"kind": "local"}, "steps": [{"id": "a", "cmd": ["echo", "hi"]}]},
    )

    res1 = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="same",
        enforce_clean=False,
    )
    with pytest.raises(RuntimeError, match="Run already exists"):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="same",
            enforce_clean=False,
        )

    res2 = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="different",
        enforce_clean=False,
    )
    assert res1["run_key"] != res2["run_key"]
