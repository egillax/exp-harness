from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from exp_harness.interp import resolve_obj
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


def load_and_validate(
    *,
    spec_path: Path,
    set_overrides: list[tuple[str, str]],
    set_string_overrides: list[tuple[str, str]],
) -> dict[str, Any]:
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Spec YAML must be a mapping at the top level")

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


def resolve_for_hashing(
    raw_with_defaults: dict[str, Any],
    *,
    project_root: Path,
    placeholder_run: dict[str, Any],
) -> dict[str, Any]:
    """
    Resolve interpolation excluding run-key dependent fields.
    """
    env = raw_with_defaults.get("env") or {}
    inputs = raw_with_defaults.get("inputs") or {}
    params = raw_with_defaults.get("params") or {}

    ctx = {
        "env": env,
        "inputs": {k: v for k, v in inputs.items()},
        "params": params,
        "run": placeholder_run,
    }
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
    env = raw_with_defaults.get("env") or {}
    inputs = raw_with_defaults.get("inputs") or {}
    params = raw_with_defaults.get("params") or {}

    ctx = {
        "env": env,
        "inputs": {k: v for k, v in inputs.items()},
        "params": params,
        "run": run_ctx,
    }
    resolved = resolve_obj(raw_with_defaults, ctx=ctx)
    if isinstance(resolved, dict) and "inputs" in resolved and isinstance(resolved["inputs"], dict):
        for _k, v in resolved["inputs"].items():
            if isinstance(v, dict) and "path" in v and isinstance(v["path"], str):
                v["path"] = str(resolve_relpath(v["path"], base_dir=project_root))
    return resolved
