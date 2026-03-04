from __future__ import annotations

import re


def sanitize_run_label(value: str, *, fallback: str = "run", max_len: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip().lower())
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    if not s:
        return fallback
    return s[:max_len]


def format_run_id(*, started_at_utc: str, run_label: str, run_key: str) -> str:
    # 2026-02-24T12:48:53Z -> 20260224-124853Z
    ts = started_at_utc.replace("-", "").replace(":", "")
    ts = ts.replace("T", "-")
    label = sanitize_run_label(run_label)
    return f"{ts}__{label}__{run_key[:8]}"


def proc_start_ticks_linux(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            stat = f.read()
        return int(stat.split()[21])
    except Exception:
        return None
