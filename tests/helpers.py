from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from exp_harness.config import Roots
from exp_harness.store import resolve_run_dir


def tmp_roots(tmp_path: Path) -> Roots:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)
    return Roots(
        project_root=project_root,
        runs_root=tmp_path / "runs",
        artifacts_root=tmp_path / "artifacts",
    )


def write_spec(project_root: Path, data: dict[str, Any], *, name: str = "spec.yaml") -> Path:
    fp = project_root / name
    fp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return fp


def read_run_json(roots: Roots, name: str, run_key: str) -> dict[str, Any]:
    run_dir = resolve_run_dir(roots=roots, name=name, run_key=run_key)
    if run_dir is None:
        raise FileNotFoundError(f"run not found: name={name} run_key={run_key}")
    fp = run_dir / "run.json"
    return json.loads(fp.read_text(encoding="utf-8"))


def find_single_run_dir(roots: Roots, name: str) -> Path:
    base = roots.runs_root / name
    run_jsons = sorted(base.glob("*/run.json"))
    if len(run_jsons) != 1:
        raise AssertionError(f"Expected 1 run.json under {base}, found {len(run_jsons)}")
    return run_jsons[0].parent


def docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


def clear_offline_env(monkeypatch) -> None:
    for k in list(os.environ.keys()):
        if k.startswith("HF_") or k in {
            "TRANSFORMERS_OFFLINE",
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
        }:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
