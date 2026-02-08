from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FingerprintResult:
    kind: str
    value: str | None
    files_hashed: int


def _match_any(path: str, globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, g) for g in globs)


def fingerprint_path(
    root: Path,
    *,
    kind: str,
    include: list[str],
    exclude: list[str],
) -> FingerprintResult:
    if kind == "none":
        return FingerprintResult(kind=kind, value=None, files_hashed=0)

    p = root
    if p.is_file():
        files = [p]
        base = p.parent
    else:
        base = p
        files = [x for x in sorted(base.rglob("*")) if x.is_file()]

    h = hashlib.sha256()
    n = 0
    for fp in files:
        rel = fp.relative_to(base).as_posix()
        if include and not _match_any(rel, include):
            continue
        if exclude and _match_any(rel, exclude):
            continue
        n += 1
        if kind == "sha256_files":
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            h.update(str(fp.stat().st_size).encode("utf-8"))
            h.update(b"\0")
            h.update(str(int(fp.stat().st_mtime)).encode("utf-8"))
            h.update(b"\0")
        elif kind == "sha256_tree":
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            with fp.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            h.update(b"\0")
        else:
            raise ValueError(f"Unknown fingerprint kind: {kind}")

    return FingerprintResult(kind=kind, value=h.hexdigest(), files_hashed=n)
