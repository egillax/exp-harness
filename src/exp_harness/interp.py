from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"\$\{([^}]+)\}")


@dataclass(frozen=True)
class InterpError(Exception):
    msg: str

    def __str__(self) -> str:  # pragma: no cover
        return self.msg


def _lookup(expr: str, ctx: dict[str, Any]) -> Any:
    cur: Any = ctx
    for part in expr.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise InterpError(f"Unknown interpolation key: {expr}")
    return cur


def resolve_obj(obj: Any, *, ctx: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: resolve_obj(v, ctx=ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_obj(v, ctx=ctx) for v in obj]
    if isinstance(obj, tuple):
        return tuple(resolve_obj(v, ctx=ctx) for v in obj)
    if not isinstance(obj, str):
        return obj

    m = _TOKEN_RE.fullmatch(obj)
    if m:
        return _lookup(m.group(1), ctx)

    def repl(mm: re.Match[str]) -> str:
        v = _lookup(mm.group(1), ctx)
        return str(v)

    return _TOKEN_RE.sub(repl, obj)
