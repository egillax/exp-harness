from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from exp_harness.interp import InterpError, resolve_obj
from exp_harness.spec import ExperimentSpec
from exp_harness.utils import resolve_relpath


@dataclass(frozen=True)
class ResolvedSpec:
    spec: ExperimentSpec
    raw: dict[str, Any]
    resolved_dict: dict[str, Any]


def _set_in_dict(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = root
    for part in parts[:-1]:
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set {path}: {part} is not a dict")
        if part not in cur or cur[part] is None:
            cur[part] = {}
        cur = cur[part]
    last = parts[-1]
    if not isinstance(cur, dict):
        raise ValueError(f"Cannot set {path}: parent is not a dict")
    cur[last] = value


def _parse_yaml_value(s: str) -> Any:
    v = yaml.safe_load(s)
    # If user passes nothing/blank, keep it as empty string (avoid surprising null).
    if v is None and s.strip() == "":
        return ""
    return v


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Spec YAML must be a mapping at the top level: {path}")
    return raw


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if k == "extends":
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            # Lists (e.g. steps) are replaced rather than merged.
            out[k] = v
    return out


def _apply_extends(
    raw: dict[str, Any], *, spec_path: Path, _stack: tuple[Path, ...] = ()
) -> dict[str, Any]:
    """
    Support a lightweight "extends" preprocessor for specs.

    - `extends: base.yaml` (or list of paths)
    - Base specs are deep-merged, then the current spec overlays them.
    - Dicts are merged recursively; lists are replaced.
    - `extends` is removed from the effective spec so it doesn't affect run identity hashing.
    """
    rp = spec_path.resolve()
    if rp in _stack:
        chain = " -> ".join(str(x) for x in (_stack + (rp,)))
        raise ValueError(f"extends cycle detected: {chain}")
    stack = _stack + (rp,)

    extends = raw.get("extends")
    if extends is None:
        out = dict(raw)
        out.pop("extends", None)
        return out

    if isinstance(extends, str):
        extend_paths = [extends]
    elif isinstance(extends, list) and all(isinstance(x, str) for x in extends):
        extend_paths = list(extends)
    else:
        raise ValueError("extends must be a string path or a list of string paths")

    merged: dict[str, Any] = {}

    for rel in extend_paths:
        p = resolve_relpath(rel, base_dir=spec_path.parent)
        base_raw = _load_yaml_mapping(p)
        base_spec = _apply_extends(base_raw, spec_path=p, _stack=stack)
        merged = _deep_merge(merged, base_spec)

    merged = _deep_merge(merged, raw)
    merged.pop("extends", None)
    return merged


def load_and_validate(
    *,
    spec_path: Path,
    set_overrides: list[tuple[str, str]],
    set_string_overrides: list[tuple[str, str]],
) -> dict[str, Any]:
    raw = _load_yaml_mapping(spec_path)
    raw = _apply_extends(raw, spec_path=spec_path)

    for k, v in set_overrides:
        _set_in_dict(raw, k, _parse_yaml_value(v))
    for k, v in set_string_overrides:
        _set_in_dict(raw, k, v)

    # Validate structure early (types may still include interpolation placeholders).
    ExperimentSpec.model_validate(raw)
    return raw


def apply_computed_defaults(
    raw: dict[str, Any], *, project_root: Path, kind: str
) -> dict[str, Any]:
    data = dict(raw)
    env = dict(data.get("env") or {})
    data["env"] = env
    data.setdefault("vars", {})

    if not env.get("workdir"):
        env["workdir"] = str(project_root) if kind == "local" else "/workspace"
    if "env" not in env or env["env"] is None:
        env["env"] = {}

    resources = dict(data.get("resources") or {})
    data["resources"] = resources
    if "gpus" not in resources:
        resources["gpus"] = 0

    steps = data.get("steps") or []
    for step in steps:
        if not isinstance(step, dict):
            continue
        outputs = step.get("outputs")
        if outputs is None:
            outputs = {}
            step["outputs"] = outputs
        if isinstance(outputs, dict) and not outputs.get("artifacts_dir"):
            step_id = step.get("id", "step")
            outputs["artifacts_dir"] = f"${{run.artifacts}}/{step_id}"

    # Docker tri-state mounts is handled at execution time; keep raw.
    return data


def _build_interp_ctx(
    raw_with_defaults: dict[str, Any], *, run_ctx: dict[str, Any]
) -> dict[str, Any]:
    env = raw_with_defaults.get("env") or {}
    inputs = raw_with_defaults.get("inputs") or {}
    params = raw_with_defaults.get("params") or {}
    vars_ = raw_with_defaults.get("vars") or {}

    if not isinstance(params, dict):
        raise ValueError("params must be a mapping")
    if not isinstance(vars_, dict):
        raise ValueError("vars must be a mapping")

    ctx: dict[str, Any] = {
        "env": env,
        "inputs": {k: v for k, v in inputs.items()},
        "params": params,
        "vars": vars_,
        "run": run_ctx,
    }

    reserved = set(ctx.keys())
    flat: dict[str, Any] = {}

    def _add_flat(source: str, d: dict[str, Any], *, overwrite: bool) -> None:
        for k, v in d.items():
            if not isinstance(k, str) or not k:
                continue
            if k in reserved:
                raise InterpError(
                    f"Invalid interpolation var {k!r} in {source}: reserved key; use ${{{source}.{k}}} instead"
                )
            if k in flat and not overwrite:
                continue
            flat[k] = v

    # Provide a shorthand so `${batch_size}` can resolve to either vars/params.
    # If a key exists in both, vars wins (use `${params.<key>}` to force params).
    _add_flat("vars", vars_, overwrite=True)
    _add_flat("params", params, overwrite=False)
    ctx.update(flat)
    return ctx


def resolve_for_hashing(
    raw_with_defaults: dict[str, Any],
    *,
    project_root: Path,
    placeholder_run: dict[str, Any],
) -> dict[str, Any]:
    """
    Resolve interpolation excluding run-key dependent fields.
    """
    ctx = _build_interp_ctx(raw_with_defaults, run_ctx=placeholder_run)
    resolved = resolve_obj(raw_with_defaults, ctx=ctx)
    # Normalize any input paths to absolute for fingerprinting stability.
    if isinstance(resolved, dict) and "inputs" in resolved and isinstance(resolved["inputs"], dict):
        for _k, v in resolved["inputs"].items():
            if isinstance(v, dict) and "path" in v and isinstance(v["path"], str):
                v["path"] = str(resolve_relpath(v["path"], base_dir=project_root))
    return resolved


def resolve_final(
    raw_with_defaults: dict[str, Any],
    *,
    project_root: Path,
    run_ctx: dict[str, Any],
) -> dict[str, Any]:
    ctx = _build_interp_ctx(raw_with_defaults, run_ctx=run_ctx)
    resolved = resolve_obj(raw_with_defaults, ctx=ctx)
    if isinstance(resolved, dict) and "inputs" in resolved and isinstance(resolved["inputs"], dict):
        for _k, v in resolved["inputs"].items():
            if isinstance(v, dict) and "path" in v and isinstance(v["path"], str):
                v["path"] = str(resolve_relpath(v["path"], base_dir=project_root))
    return resolved
