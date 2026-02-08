from __future__ import annotations

import time
from pathlib import Path

from exp_harness.config import Roots


def _find_step_dir(run_dir: Path, step_id: str | None) -> Path | None:
    steps_dir = run_dir / "steps"
    if not steps_dir.exists():
        return None
    if step_id is None:
        all_steps = sorted([p for p in steps_dir.iterdir() if p.is_dir()])
        return all_steps[-1] if all_steps else None
    for p in sorted([p for p in steps_dir.iterdir() if p.is_dir()]):
        if p.name.endswith(f"_{step_id}"):
            return p
    return None


def _tail(fp: Path, n: int = 200) -> str:
    if not fp.exists():
        return ""
    try:
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:]) + ("\n" if lines else "")
    except Exception:
        return fp.read_text(encoding="utf-8", errors="replace")


def _follow(fp: Path) -> None:
    pos = 0
    while True:
        if fp.exists():
            with fp.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            if chunk:
                print(chunk, end="")
        time.sleep(0.5)


def show_logs(*, roots: Roots, name: str, run_key: str, step: str | None, follow: bool) -> None:
    run_dir = roots.runs_root / name / run_key
    step_dir = _find_step_dir(run_dir, step)
    if not step_dir:
        raise FileNotFoundError(f"Step not found under: {run_dir}")
    stdout_fp = step_dir / "stdout.log"
    stderr_fp = step_dir / "stderr.log"
    if follow:
        print(f"==> {stdout_fp}")
        _follow(stdout_fp)
        return
    if stdout_fp.exists():
        print(f"==> {stdout_fp}")
        print(_tail(stdout_fp), end="")
    if stderr_fp.exists():
        print(f"==> {stderr_fp}")
        print(_tail(stderr_fp), end="")
