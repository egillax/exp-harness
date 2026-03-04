from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.config import Roots
from exp_harness.errors import StepExecutionError
from exp_harness.runner import run_experiment


def test_step_failure_records_stderr_tail_and_uses_human_readable_run_id(
    tmp_path: Path, caplog
) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )

    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "fail",
                "env": {"kind": "local"},
                "steps": [
                    {
                        "id": "boom",
                        "cmd": [
                            sys.executable,
                            "-c",
                            (
                                "import sys; "
                                "[print(f'line{i}', file=sys.stderr) for i in range(200)]; "
                                "sys.exit(2)"
                            ),
                        ],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with (
        caplog.at_level(logging.ERROR, logger="exp_harness.run.execution"),
        pytest.raises(StepExecutionError, match=r"Step failed: boom \(rc=2\)") as exc_info,
    ):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            stderr_tail_lines=5,
        )
    assert exc_info.value.step_id == "boom"
    assert exc_info.value.rc == 2

    assert "stderr tail (last 5 lines)" in caplog.text
    assert "line199" in caplog.text

    run_dirs = sorted((roots.runs_root / "fail").glob("*/run.json"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0].parent
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert run_json["state"] == "failed"
    assert run_json["error"]["step_id"] == "boom"
    assert run_json["error"]["rc"] == 2
    assert "line199" in (run_json["error"]["stderr_tail"] or "")

    # Run directories are timestamp-first and include a short hash suffix.
    assert "__spec__" in run_dir.name
    assert run_json["run_id"] == run_dir.name

    artifacts_dir = roots.artifacts_root / "fail" / run_dir.name
    assert artifacts_dir.exists()
