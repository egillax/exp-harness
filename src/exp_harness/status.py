from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.gpu_pool import GpuPool
from exp_harness.utils import normalize_space


def _proc_start_ticks_linux(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            stat = f.read()
        return int(stat.split()[21])
    except Exception:
        return None


def _pid_alive(pid: int, start_ticks: int | None) -> bool:
    try:
        import os

        os.kill(pid, 0)
    except OSError:
        return False
    if start_ticks is None:
        return True
    cur = _proc_start_ticks_linux(pid)
    return cur is not None and cur == start_ticks


def _iter_run_json_files(runs_root: Path, *, name: str | None) -> list[Path]:
    if name:
        base = runs_root / name
        if not base.exists():
            return []
        return sorted(base.glob("*/run.json"))
    return sorted(runs_root.glob("*/*/run.json"))


def _load_json(fp: Path) -> dict[str, Any] | None:
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception:
        return None


def print_status(*, roots: Roots, name: str | None, limit: int) -> None:
    run_json_fps = _iter_run_json_files(roots.runs_root, name=name)
    runs: list[dict[str, Any]] = []
    for fp in run_json_fps:
        r = _load_json(fp)
        if not r:
            continue
        r["_path"] = str(fp.parent)
        runs.append(r)

    def key(r: dict[str, Any]) -> str:
        return str(r.get("created_at_utc") or "")

    runs = sorted(runs, key=key, reverse=True)[: max(0, limit)]
    for r in runs:
        nm = r.get("name", "?")
        rid = r.get("run_id", "?")
        rk = r.get("run_key", "?")
        st = r.get("state", "?")
        g = r.get("allocated_gpus_host") or []
        created = r.get("created_at_utc") or ""
        pid = r.get("pid")
        start_ticks = r.get("proc_start_ticks")
        stale = ""
        if (
            st == "running"
            and isinstance(pid, int)
            and not _pid_alive(pid, start_ticks if isinstance(start_ticks, int) else None)
        ):
            stale = " (stale: pid not alive)"
        print(f"{nm} {rid} {rk} {st}{stale} gpus={g} created={created} path={r.get('_path')}")

    pool = GpuPool(locks_dir=(roots.runs_root / "_locks"))
    locks = pool.list_locks()
    if not locks:
        return
    alive = [lk for lk in locks if lk.is_alive()]
    dead = [lk for lk in locks if not lk.is_alive()]
    if alive:
        print(f"\nGPU locks (alive): {normalize_space(str([lk.gpu_id for lk in alive]))}")
    if dead:
        print(f"GPU locks (orphaned): {normalize_space(str([lk.gpu_id for lk in dead]))}")
        print("hint: run-experiment locks gc --force")
