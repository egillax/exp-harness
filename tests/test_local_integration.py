from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.config import Roots
from exp_harness.errors import StepExecutionError
from exp_harness.runner import run_experiment


def test_local_run_writes_layout_and_artifacts(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "it",
                "env": {"kind": "local"},
                "steps": [
                    {"id": "a", "cmd": [sys.executable, "-c", "print('hi')"]},
                    {
                        "id": "b",
                        "needs": ["a"],
                        "cmd": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; import os; p=Path(os.environ['EXP_HARNESS_STEP_ARTIFACTS_DIR']); p.mkdir(parents=True, exist_ok=True); (p/'x.txt').write_text('ok')",
                        ],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
    )

    run_dir = Path(res["run_dir"])
    assert (run_dir / "run.json").exists()
    assert (run_dir / "resolved_spec.json").exists()
    assert (Path(res["artifacts_dir"]) / "b" / "x.txt").exists()


def test_failed_step_stops_downstream(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "it2",
                "env": {"kind": "local"},
                "steps": [
                    {"id": "a", "cmd": [sys.executable, "-c", "print('ok')"]},
                    {
                        "id": "b",
                        "needs": ["a"],
                        "cmd": [sys.executable, "-c", "import sys; sys.exit(2)"],
                    },
                    {"id": "c", "needs": ["b"], "cmd": [sys.executable, "-c", "print('no')"]},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(StepExecutionError):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
        )

    # Find the single run dir.
    run_jsons = list((roots.runs_root / "it2").glob("*/run.json"))
    assert len(run_jsons) == 1
    run_dir = run_jsons[0].parent
    assert (run_dir / "steps" / "00_a").exists()
    assert (run_dir / "steps" / "01_b").exists()
    assert not (run_dir / "steps" / "02_c").exists()
