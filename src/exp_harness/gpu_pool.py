from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exp_harness.utils import hostname, shell_out, utc_now_iso, which


def _proc_start_ticks_linux(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            stat = f.read()
        parts = stat.split()
        # Field 22: starttime in clock ticks since boot.
        return int(parts[21])
    except Exception:
        return None


def _pid_alive(pid: int, expected_start_ticks: int | None) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    if expected_start_ticks is None:
        return True
    cur = _proc_start_ticks_linux(pid)
    return cur is not None and cur == expected_start_ticks


@dataclass(frozen=True)
class LockInfo:
    gpu_id: int
    path: Path
    data: dict[str, Any]

    @property
    def pid(self) -> int | None:
        v = self.data.get("pid")
        return int(v) if isinstance(v, (int, str)) and str(v).isdigit() else None

    @property
    def start_ticks(self) -> int | None:
        v = self.data.get("proc_start_ticks")
        return int(v) if isinstance(v, int) else None

    def is_alive(self) -> bool:
        pid = self.pid
        if pid is None:
            return False
        return _pid_alive(pid, self.start_ticks)


class GpuPool:
    def __init__(self, *, locks_dir: Path) -> None:
        self.locks_dir = locks_dir
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    def detect_gpu_count(self) -> int:
        if not which("nvidia-smi"):
            return 0
        rc, out, _ = shell_out(["nvidia-smi", "-L"], cwd=None)
        if rc != 0:
            return 0
        lines = [ln for ln in out.splitlines() if ln.strip().startswith("GPU ")]
        return len(lines)

    def lock_path(self, gpu_id: int) -> Path:
        return self.locks_dir / f"gpu{gpu_id}.lock"

    def try_acquire(self, gpu_id: int, *, lock_data: dict[str, Any]) -> bool:
        lp = self.lock_path(gpu_id)
        try:
            fd = os.open(str(lp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(lock_data, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            with suppress(Exception):
                os.unlink(lp)
            return False
        return True

    def release(self, gpu_id: int, *, expected_pid: int | None = None) -> None:
        lp = self.lock_path(gpu_id)
        if not lp.exists():
            return
        if expected_pid is not None:
            try:
                data = json.loads(lp.read_text(encoding="utf-8"))
                if int(data.get("pid", -1)) != expected_pid:
                    return
            except Exception:
                return
        try:
            os.unlink(lp)
        except FileNotFoundError:
            return

    def list_locks(self) -> list[LockInfo]:
        out: list[LockInfo] = []
        for fp in sorted(self.locks_dir.glob("gpu*.lock")):
            name = fp.stem
            if not name.startswith("gpu"):
                continue
            try:
                gpu_id = int(name[3:])
            except Exception:
                continue
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            out.append(LockInfo(gpu_id=gpu_id, path=fp, data=data))
        return out


@dataclass(frozen=True)
class Allocation:
    gpu_ids: list[int]
    pid: int


def allocate_gpus(
    pool: GpuPool,
    request: int | list[int],
    *,
    run_path: str,
    run_key: str,
    name: str,
) -> Allocation:
    count = pool.detect_gpu_count()
    if isinstance(request, int):
        if request <= 0:
            return Allocation(gpu_ids=[], pid=os.getpid())
        if count == 0:
            raise RuntimeError(
                "No GPUs detected (nvidia-smi not available), but GPUs were requested"
            )
        want = request
        candidates = list(range(count))
    else:
        want = len(request)
        candidates = list(request)
        if want == 0:
            return Allocation(gpu_ids=[], pid=os.getpid())
        if count == 0:
            raise RuntimeError(
                "No GPUs detected (nvidia-smi not available), but explicit GPU ids were requested"
            )
        bad = [x for x in candidates if x < 0 or x >= count]
        if bad:
            raise RuntimeError(f"Requested GPU ids out of range (0..{count - 1}): {bad}")

    acquired: list[int] = []
    pid = os.getpid()
    lock_data_base = {
        "pid": pid,
        "proc_start_ticks": _proc_start_ticks_linux(pid),
        "hostname": hostname(),
        "started_at_utc": utc_now_iso(),
        "run_path": run_path,
        "run_key": run_key,
        "name": name,
    }

    for gpu_id in candidates:
        if pool.try_acquire(gpu_id, lock_data=lock_data_base | {"gpu_id": gpu_id}):
            acquired.append(gpu_id)
            if len(acquired) == want:
                break

    if len(acquired) != want:
        for g in acquired:
            pool.release(g, expected_pid=pid)
        if isinstance(request, int):
            raise RuntimeError(
                f"Insufficient free GPUs: requested {want}, acquired {len(acquired)}"
            )
        raise RuntimeError(f"Could not acquire requested GPUs: {request}")

    return Allocation(gpu_ids=acquired, pid=pid)
