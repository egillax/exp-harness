from __future__ import annotations

import pytest

from exp_harness.run.phases import PreparedRun, phase_allocate_resources
from exp_harness.store import get_run_paths, init_run_dirs
from tests.helpers import tmp_roots


def test_phase_allocate_releases_lock_if_run_json_write_fails(tmp_path, monkeypatch) -> None:
    roots = tmp_roots(tmp_path)
    name = "alloc_cleanup"
    run_id = "run-001"
    run_key = "rk-001"

    paths = get_run_paths(roots=roots, name=name, run_id=run_id)
    init_run_dirs(paths)

    prepared = PreparedRun(
        paths=paths,
        run_ctx={"id": run_key, "runs": str(paths.run_dir), "artifacts": str(paths.artifacts_dir)},
        resolved_final={"resources": {"gpus": [0]}},
        env_vars={},
    )
    run_json: dict[str, object] = {}

    def _fail_write_run_json(*_args, **_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("exp_harness.run.phases.write_run_json", _fail_write_run_json)

    with pytest.raises(OSError, match="disk full"):
        phase_allocate_resources(
            roots=roots,
            prepared=prepared,
            run_json=run_json,
            name=name,
            run_key=run_key,
        )

    lock_fp = roots.runs_root / "_locks" / "gpu0.lock"
    assert not lock_fp.exists()
