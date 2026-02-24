from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.store import resolve_run_dir


def _load_json(fp: Path) -> dict[str, Any] | None:
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def inspect_run(*, roots: Roots, name: str, run_key: str) -> None:
    run_dir = resolve_run_dir(roots=roots, name=name, run_key=run_key)
    if not run_dir:
        raise FileNotFoundError(f"Run not found: name={name} run_key={run_key}")
    run_json = _load_json(run_dir / "run.json") or {}
    prov_dir = run_dir / "provenance"
    git = _load_json(prov_dir / "git.json") or {}
    docker = _load_json(prov_dir / "docker.json") or {}
    inputs = _load_json(prov_dir / "inputs.json") or {}

    print(f"name: {run_json.get('name')}")
    print(f"run_id: {run_json.get('run_id')}")
    print(f"run_label: {run_json.get('run_label')}")
    print(f"run_key: {run_json.get('run_key')}")
    print(f"state: {run_json.get('state')}")
    print(f"gpus_host: {run_json.get('allocated_gpus_host')}")
    print(f"git: {git.get('commit')} dirty={git.get('dirty')} diff_hash={git.get('diff_hash')}")
    if docker:
        img = ((docker.get("image") or {}) or {}).get("image")
        img_id = ((docker.get("image") or {}) or {}).get("image_id")
        repo_digests = ((docker.get("image") or {}) or {}).get("repo_digests")
        if repo_digests:
            print(f"docker: image={img} image_id={img_id} repo_digests={repo_digests}")
        else:
            print(f"docker: image={img} image_id={img_id}")
    if inputs:
        print(f"inputs: {list(inputs.keys())}")
