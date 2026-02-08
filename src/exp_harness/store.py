from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.utils import ensure_dir, write_json


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    artifacts_dir: Path
    provenance_dir: Path
    steps_dir: Path


def get_run_paths(*, roots: Roots, name: str, run_key: str) -> RunPaths:
    run_dir = roots.runs_root / name / run_key
    artifacts_dir = roots.artifacts_root / name / run_key
    return RunPaths(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        provenance_dir=run_dir / "provenance",
        steps_dir=run_dir / "steps",
    )


def init_run_dirs(paths: RunPaths) -> None:
    ensure_dir(paths.run_dir)
    ensure_dir(paths.provenance_dir)
    ensure_dir(paths.steps_dir)
    ensure_dir(paths.artifacts_dir)


def write_run_json(paths: RunPaths, run_json: dict[str, Any]) -> None:
    write_json(paths.run_dir / "run.json", run_json)
