from __future__ import annotations

from datetime import UTC, datetime

from exp_harness.config import Roots
from exp_harness.resources.gpu_pool import GpuPool, LockInfo
from exp_harness.utils import shell_out, which


def _parse_utc(s: str) -> datetime | None:
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(UTC)
    except Exception:
        return None


def _docker_containers_for_run(run_key: str) -> list[str]:
    if not which("docker"):
        return []
    rc, out, _ = shell_out(
        [
            "docker",
            "ps",
            "--filter",
            f"label=exp-harness.run_key={run_key}",
            "--format",
            "{{.ID}}",
        ],
        cwd=None,
    )
    if rc != 0:
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def gc_locks(*, roots: Roots, grace_seconds: int, force: bool) -> None:
    pool = GpuPool(locks_dir=(roots.runs_root / "_locks"))
    locks = pool.list_locks()
    if not locks:
        print("No GPU locks found.")
        return

    now = datetime.now(UTC)
    eligible: list[LockInfo] = []
    for lk in locks:
        if lk.is_alive():
            continue
        started = _parse_utc(str(lk.data.get("started_at_utc") or ""))
        if started is not None:
            age_s = (now - started).total_seconds()
            if age_s < grace_seconds:
                continue
        run_key = str(lk.data.get("run_key") or "")
        if run_key:
            containers = _docker_containers_for_run(run_key)
            if containers and not force:
                print(
                    f"Skipping gpu{lk.gpu_id}: docker containers still running for run_key={run_key}: {containers}"
                )
                continue
        eligible.append(lk)

    if not eligible:
        print("No orphan locks eligible for removal.")
        return

    if not force:
        print("Orphan locks eligible for removal:")
        for lk in eligible:
            print(f"  - gpu{lk.gpu_id}: {lk.path}")
        print("Re-run with --force to remove them.")
        return

    for lk in eligible:
        try:
            lk.path.unlink(missing_ok=True)
            print(f"Removed: gpu{lk.gpu_id} ({lk.path})")
        except Exception as e:
            print(f"Failed to remove gpu{lk.gpu_id} ({lk.path}): {e}")
