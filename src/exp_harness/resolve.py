from __future__ import annotations

from pathlib import Path
from typing import Any

from exp_harness.interp import InterpError, resolve_obj
from exp_harness.spec import StepSpec
from exp_harness.utils import resolve_relpath


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

    for step in _iter_step_dicts(data.get("steps")):
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


def _iter_step_dicts(steps: Any) -> list[dict[str, Any]]:
    if not isinstance(steps, list):
        return []
    out: list[dict[str, Any]] = []
    for step in steps:
        if isinstance(step, StepSpec):
            out.append(step.model_dump(mode="python"))
        elif isinstance(step, dict):
            out.append(step)
    return out


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
