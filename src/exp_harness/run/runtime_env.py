from __future__ import annotations

from typing import Any

from exp_harness.config import Roots
from exp_harness.docker_utils import resolve_docker_runtime

HF_OFFLINE_VARS: dict[str, str] = {
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def resolved_offline_env(
    *, offline: bool, hf_home: str, existing: dict[str, str]
) -> dict[str, str]:
    env = dict(existing)
    if not offline:
        return env
    env.update(HF_OFFLINE_VARS)
    env.setdefault("HF_HOME", hf_home)
    return env


def prepare_effective_env(spec_dict: dict[str, Any], *, roots: Roots, kind: str) -> dict[str, str]:
    env_block = dict(spec_dict.get("env") or {})
    offline = bool(env_block.get("offline"))
    env_vars = dict(env_block.get("env") or {})
    if kind == "docker":
        hf_home = "/workspace/artifacts/hf_home"
    else:
        hf_home = str((roots.artifacts_root / "hf_home").resolve())
    env_vars = resolved_offline_env(offline=offline, hf_home=hf_home, existing=env_vars)
    env_block["env"] = env_vars
    spec_dict["env"] = env_block
    return env_vars


def prepare_effective_docker(
    spec_dict: dict[str, Any], *, roots: Roots, for_hashing: bool
) -> dict[str, Any]:
    env_block = dict(spec_dict.get("env") or {})
    docker_block = dict(env_block.get("docker") or {})
    offline = bool(env_block.get("offline"))
    effective = resolve_docker_runtime(
        docker_block,
        offline=offline,
        roots=roots,
        project_root=roots.project_root,
        for_hashing=for_hashing,
    )
    env_block["docker"] = effective
    spec_dict["env"] = env_block
    return effective
