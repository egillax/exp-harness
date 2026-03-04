from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_RUNS_ROOT = "RUN_EXPERIMENT_RUNS_ROOT"
ENV_ARTIFACTS_ROOT = "RUN_EXPERIMENT_ARTIFACTS_ROOT"

__all__ = ["ENV_RUNS_ROOT", "ENV_ARTIFACTS_ROOT", "Roots", "resolve_roots"]


@dataclass(frozen=True)
class Roots:
    project_root: Path
    runs_root: Path
    artifacts_root: Path


def _env_path(key: str) -> Path | None:
    v = os.environ.get(key)
    if not v:
        return None
    return Path(v).expanduser()


def resolve_roots(
    *,
    project_root: Path,
    runs_root: Path | None,
    artifacts_root: Path | None,
) -> Roots:
    runs_root_final = (
        runs_root or _env_path(ENV_RUNS_ROOT) or (project_root / "experiment_results" / "runs")
    )
    artifacts_root_final = (
        artifacts_root
        or _env_path(ENV_ARTIFACTS_ROOT)
        or (project_root / "experiment_results" / "artifacts")
    )
    return Roots(
        project_root=project_root,
        runs_root=runs_root_final.resolve(),
        artifacts_root=artifacts_root_final.resolve(),
    )
