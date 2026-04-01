from __future__ import annotations

import hashlib
import json
from typing import Any

from exp_harness.errors import RunResumeError

ALLOWED_CONFIG_DRIFT_STATES = {"failed", "pending", "interrupted"}


def parse_allowed_config_drift(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    values = {part.strip() for part in raw.split(",") if part.strip()}
    if not values:
        raise RunResumeError("--allow-config-drift must list at least one state")
    invalid = sorted(values - ALLOWED_CONFIG_DRIFT_STATES)
    if invalid:
        joined = ", ".join(invalid)
        raise RunResumeError(
            f"Invalid --allow-config-drift state(s): {joined}; "
            "expected failed, pending, interrupted"
        )
    return values


def normalize_resume_step_state(state: Any) -> str:
    value = str(state or "").strip().lower()
    if value == "succeeded":
        return "succeeded"
    if value in {"failed"}:
        return "failed"
    if value in {"interrupted", "running"}:
        return "interrupted"
    if value in {"pending", "skipped", ""}:
        return "pending"
    return value or "pending"


def build_step_contract(step: dict[str, Any], *, resolved_final: dict[str, Any]) -> dict[str, Any]:
    env_block = resolved_final.get("env") or {}
    contract: dict[str, Any] = {
        "step": {
            "id": step.get("id"),
            "needs": list(step.get("needs") or []),
            "cmd": list(step.get("cmd") or []),
            "timeout_seconds": step.get("timeout_seconds"),
            "outputs": dict(step.get("outputs") or {}),
            "resume_policy": step.get("resume_policy"),
            "success_markers": list(step.get("success_markers") or []),
        },
        "run": {
            "kind": env_block.get("kind"),
            "workdir": env_block.get("workdir"),
            "offline": bool(env_block.get("offline")),
            "env": dict(env_block.get("env") or {}),
            "resources": dict(resolved_final.get("resources") or {}),
            "docker": dict(env_block.get("docker") or {})
            if env_block.get("kind") == "docker"
            else None,
        },
    }
    return contract


def contract_hash(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _diff_values(old: Any, new: Any, *, path: str) -> list[dict[str, Any]]:
    if isinstance(old, dict) and isinstance(new, dict):
        diffs: list[dict[str, Any]] = []
        keys = sorted(set(old.keys()) | set(new.keys()))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in old:
                diffs.append({"field": child, "old": None, "new": new[key]})
                continue
            if key not in new:
                diffs.append({"field": child, "old": old[key], "new": None})
                continue
            diffs.extend(_diff_values(old[key], new[key], path=child))
        return diffs

    if isinstance(old, list) and isinstance(new, list):
        if old == new:
            return []
        return [{"field": path, "old": old, "new": new}]

    if old == new:
        return []
    return [{"field": path, "old": old, "new": new}]


def contract_diff(
    *, old_contract: dict[str, Any], new_contract: dict[str, Any]
) -> list[dict[str, Any]]:
    return _diff_values(old_contract, new_contract, path="")


def format_diff_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def format_diff_line(diff: dict[str, Any]) -> str:
    field = str(diff.get("field") or "contract")
    return f"{field}: {format_diff_value(diff.get('old'))} -> {format_diff_value(diff.get('new'))}"
