from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_WS_RE = re.compile(r"\s+")


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json_bytes(obj: Any) -> bytes:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")


def hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def shell_out(
    argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    proc = subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def which(cmd: str) -> str | None:
    from shutil import which as _which

    return _which(cmd)


def resolve_relpath(path: str, *, base_dir: Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def normalize_space(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def safe_makedirs(p: Path) -> None:
    os.makedirs(p, exist_ok=True)


def discover_project_root(spec_path: Path) -> Path:
    start_dir = spec_path.resolve().parent
    return discover_project_root_from_dir(start_dir)


def discover_project_root_from_dir(start_dir: Path) -> Path:
    start_dir = start_dir.resolve()

    # Prefer git root if available.
    try:
        rc, out, _ = shell_out(["git", "rev-parse", "--show-toplevel"], cwd=start_dir)
        if rc == 0:
            p = Path(out.strip())
            if p.exists():
                return p.resolve()
    except Exception:
        pass

    # Otherwise, walk upwards looking for a pyproject.toml.
    cur = start_dir
    while True:
        if (cur / "pyproject.toml").exists():
            return cur.resolve()
        if cur.parent == cur:
            return start_dir.resolve()
        cur = cur.parent
