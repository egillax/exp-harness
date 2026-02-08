from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from exp_harness.git_info import GitInfo
from exp_harness.utils import hostname, shell_out, utc_now_iso, which, write_json, write_text


def write_git_provenance(prov_dir: Path, git: GitInfo) -> None:
    write_json(
        prov_dir / "git.json",
        {
            "commit": git.commit,
            "branch": git.branch,
            "dirty": git.dirty,
            "origin_url": git.origin_url,
            "diff_hash": git.diff_hash,
        },
    )
    if git.diff_patch:
        write_text(prov_dir / "git_diff.patch", git.diff_patch)


def write_host_provenance(prov_dir: Path, *, extra: dict[str, Any]) -> None:
    info: dict[str, Any] = {
        "captured_at_utc": utc_now_iso(),
        "hostname": hostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "uname": {},
    }
    try:
        u = platform.uname()
        info["uname"] = {
            "system": u.system,
            "node": u.node,
            "release": u.release,
            "version": u.version,
            "machine": u.machine,
            "processor": u.processor,
        }
    except Exception:
        pass
    info.update(extra)
    write_json(prov_dir / "host.json", info)


def write_env_provenance(
    prov_dir: Path, *, env_allow: list[str], explicit_env: dict[str, str]
) -> None:
    env: dict[str, str] = {}
    for k in env_allow:
        if k in os.environ:
            env[k] = os.environ[k]
    for k, v in explicit_env.items():
        env[k] = v
    write_json(prov_dir / "environment.json", env)


def write_nvidia_smi(prov_dir: Path) -> None:
    if not which("nvidia-smi"):
        return
    try:
        rc, out, err = shell_out(["nvidia-smi"], cwd=None)
        if rc == 0:
            write_text(prov_dir / "nvidia_smi.txt", out)
        else:
            write_text(prov_dir / "nvidia_smi.txt", out + "\n" + err)
    except Exception:
        return


def write_python_and_freeze(
    prov_dir: Path, *, env: dict[str, str] | None = None, cwd: Path | None = None
) -> None:
    try:
        rc, out, err = shell_out([sys.executable, "-V"], cwd=cwd, env=env)
        write_text(prov_dir / "python.txt", (out or err).strip() + "\n")
    except Exception:
        pass
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        write_text(prov_dir / "pip_freeze.txt", proc.stdout)
    except Exception:
        pass
