from __future__ import annotations

from pathlib import Path

from exp_harness.runner import run_experiment
from tests.helpers import tmp_roots, write_spec


def test_run_id_uses_spec_run_label_when_present(tmp_path: Path) -> None:
    roots = tmp_roots(tmp_path)
    spec_fp = write_spec(
        roots.project_root,
        {
            "name": "x",
            "run_label": "My Batch 01",
            "env": {"kind": "local"},
            "steps": [{"id": "a", "cmd": ["echo", "hi"]}],
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
    run_id = res["run_id"]
    assert "__my-batch-01__" in run_id
    assert run_id.endswith(res["run_key"][:8])
    assert (Path(res["run_dir"]).parent / "_by_time").exists() is False


def test_run_id_prefers_cli_run_label_override(tmp_path: Path) -> None:
    roots = tmp_roots(tmp_path)
    spec_fp = write_spec(
        roots.project_root,
        {
            "name": "x",
            "run_label": "from-spec",
            "env": {"kind": "local"},
            "steps": [{"id": "a", "cmd": ["echo", "hi"]}],
        },
    )
    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s2",
        run_label="from-cli",
        enforce_clean=False,
    )
    assert "__from-cli__" in res["run_id"]
