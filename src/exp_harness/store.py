from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.utils import ensure_dir, write_json


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    run_dir: Path
    artifacts_dir: Path
    provenance_dir: Path
    steps_dir: Path


def get_run_paths(*, roots: Roots, name: str, run_id: str) -> RunPaths:
    run_dir = roots.runs_root / name / run_id
    artifacts_dir = roots.artifacts_root / name / run_id
    return RunPaths(
        run_id=run_id,
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


def _run_key_index_dir(*, roots: Roots, name: str) -> Path:
    return roots.runs_root / name / "_by_key"


def write_run_key_index(*, roots: Roots, name: str, run_key: str, run_id: str) -> None:
    idx_dir = _run_key_index_dir(roots=roots, name=name)
    ensure_dir(idx_dir)
    (idx_dir / f"{run_key}.txt").write_text(run_id, encoding="utf-8")


def resolve_run_dir(*, roots: Roots, name: str, run_key: str) -> Path | None:
    # Backward-compat: legacy layout used run_key as directory name.
    legacy = roots.runs_root / name / run_key
    if legacy.is_dir():
        return legacy

    # Preferred fast path: run_key -> run_id index.
    idx = _run_key_index_dir(roots=roots, name=name) / f"{run_key}.txt"
    if idx.is_file():
        try:
            run_id = idx.read_text(encoding="utf-8").strip()
        except Exception:
            run_id = ""
        if run_id:
            candidate = roots.runs_root / name / run_id
            if (candidate / "run.json").is_file():
                return candidate

    # Fallback scan: robust against missing/corrupt index.
    base = roots.runs_root / name
    if not base.is_dir():
        return None
    for fp in sorted(base.glob("*/run.json")):
        try:
            payload = fp.read_text(encoding="utf-8")
        except Exception:
            continue
        if f'"run_key": "{run_key}"' in payload:
            return fp.parent
    return None
