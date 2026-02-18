from __future__ import annotations

import sys
from pathlib import Path

from exp_harness.executors.base import RunContext
from exp_harness.executors.local import LocalExecutor


def test_local_executor_stream_logs_writes_files_and_live_output(tmp_path: Path, capfd) -> None:
    run_dir = tmp_path / "runs" / "n" / "k"
    artifacts_dir = tmp_path / "artifacts" / "n" / "k"
    step_dir = run_dir / "steps" / "00_a"
    step_dir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        name="n",
        run_key="k",
        project_root=tmp_path,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        workdir=str(tmp_path),
        env={},
        offline=False,
        kind="local",
        docker=None,
        allocated_gpus_host=[],
        stream_logs=True,
    )

    ex = LocalExecutor()
    res = ex.run_step(
        ctx,
        step_index=0,
        step_id="a",
        cmd=[
            sys.executable,
            "-c",
            "import sys; print('OUT'); print('ERR', file=sys.stderr)",
        ],
        step_dir=step_dir,
        timeout_seconds=10,
        step_artifacts_dir=None,
    )

    assert res.rc == 0
    out = (step_dir / "stdout.log").read_text(encoding="utf-8")
    err = (step_dir / "stderr.log").read_text(encoding="utf-8")
    assert "OUT" in out
    assert "ERR" in err

    live = capfd.readouterr()
    assert "OUT" in live.out
    assert "ERR" in live.err
