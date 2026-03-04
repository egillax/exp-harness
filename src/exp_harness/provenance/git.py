from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from exp_harness.utils import shell_out


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    branch: str | None
    dirty: bool
    origin_url: str | None
    diff_patch: str | None
    diff_hash: str | None


def collect_git_info(*, project_root: Path) -> GitInfo:
    rc, commit, _ = shell_out(["git", "rev-parse", "HEAD"], cwd=project_root)
    commit_s = commit.strip() if rc == 0 else None

    rc, branch, _ = shell_out(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
    branch_s = branch.strip() if rc == 0 else None

    rc, status, _ = shell_out(["git", "status", "--porcelain=v1"], cwd=project_root)
    dirty = bool(status.strip()) if rc == 0 else False

    rc, origin, _ = shell_out(["git", "remote", "get-url", "origin"], cwd=project_root)
    origin_s = origin.strip() if rc == 0 else None

    rc, diff, _ = shell_out(["git", "diff"], cwd=project_root)
    diff_s = diff if rc == 0 and diff else None
    diff_hash = None
    if diff_s:
        diff_hash = hashlib.sha256(diff_s.encode("utf-8")).hexdigest()

    return GitInfo(
        commit=commit_s,
        branch=branch_s,
        dirty=dirty,
        origin_url=origin_s,
        diff_patch=diff_s,
        diff_hash=diff_hash,
    )
