from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.config import Roots
from exp_harness.runner import run_experiment


def test_step_failure_records_stderr_tail_and_creates_by_time_symlinks(
    tmp_path: Path, capsys
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

    with pytest.raises(RuntimeError, match=r"Step failed: boom \(rc=2\)"):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            stderr_tail_lines=5,
        )

    err = capsys.readouterr().err
    assert "[exp-harness] stderr tail (last 5 lines)" in err
    assert "line199" in err

    run_dirs = sorted((roots.runs_root / "fail").glob("*/run.json"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0].parent
    run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert run_json["state"] == "failed"
    assert run_json["error"]["step_id"] == "boom"
    assert run_json["error"]["rc"] == 2
    assert "line199" in (run_json["error"]["stderr_tail"] or "")

    by_time = roots.runs_root / "fail" / "_by_time"
    links = list(by_time.iterdir())
    assert len(links) == 1
    assert links[0].is_symlink()
    assert links[0].resolve() == run_dir.resolve()

    artifacts_dir = roots.artifacts_root / "fail" / run_dir.name
    by_time_a = roots.artifacts_root / "fail" / "_by_time"
    links_a = list(by_time_a.iterdir())
    assert len(links_a) == 1
    assert links_a[0].is_symlink()
    assert links_a[0].resolve() == artifacts_dir.resolve()
