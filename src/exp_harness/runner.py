from __future__ import annotations

import copy
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

from exp_harness.config import Roots
from exp_harness.docker_utils import (
    ARTIFACTS_ROOT_PLACEHOLDER,
    RUNS_ROOT_PLACEHOLDER,
    inspect_image,
    resolve_docker_runtime,
)
from exp_harness.executors.base import RunContext
from exp_harness.executors.docker import DockerExecutor
from exp_harness.executors.local import LocalExecutor
from exp_harness.fingerprints import fingerprint_path
from exp_harness.git_info import collect_git_info
from exp_harness.gpu_pool import GpuPool, allocate_gpus
from exp_harness.provenance import (
    write_env_provenance,
    write_git_provenance,
    write_host_provenance,
    write_nvidia_smi,
    write_python_and_freeze,
)
from exp_harness.resolve import (
    apply_computed_defaults,
    load_and_validate,
    resolve_final,
    resolve_for_hashing,
)
from exp_harness.run_key import compute_run_key
from exp_harness.spec import ExperimentSpec
from exp_harness.store import get_run_paths, init_run_dirs, write_run_json
from exp_harness.utils import (
    ensure_dir,
    iso_to_compact_utc,
    resolve_relpath,
    safe_symlink,
    tail_text_lines,
    utc_now_iso,
    write_json,
)

HF_OFFLINE_VARS: dict[str, str] = {
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}


def _proc_start_ticks_linux(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            stat = f.read()
        return int(stat.split()[21])
    except Exception:
        return None


def _resolved_offline_env(
    *, offline: bool, hf_home: str, existing: dict[str, str]
) -> dict[str, str]:
    env = dict(existing)
    if not offline:
        return env
    env.update(HF_OFFLINE_VARS)
    env.setdefault("HF_HOME", hf_home)
    return env


def _prepare_effective_env(spec_dict: dict[str, Any], *, roots: Roots, kind: str) -> dict[str, str]:
    env_block = dict(spec_dict.get("env") or {})
    offline = bool(env_block.get("offline"))
    env_vars = dict(env_block.get("env") or {})
    if kind == "docker":
        hf_home = "/workspace/artifacts/hf_home"
    else:
        hf_home = str((roots.artifacts_root / "hf_home").resolve())
    env_vars = _resolved_offline_env(offline=offline, hf_home=hf_home, existing=env_vars)
    env_block["env"] = env_vars
    spec_dict["env"] = env_block
    return env_vars


def _prepare_effective_docker(
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


def _validate_and_toposort_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for s in steps:
        sid = s.get("id")
        if not isinstance(sid, str) or not sid:
            raise ValueError("Step has missing/empty id")
        if sid in by_id:
            raise ValueError(f"Duplicate step id: {sid}")
        by_id[sid] = s

    for sid, s in by_id.items():
        needs = s.get("needs") or []
        for dep in needs:
            if dep not in by_id:
                raise ValueError(f"Step {sid} needs missing dependency: {dep}")

    incoming: dict[str, set[str]] = {sid: set(s.get("needs") or []) for sid, s in by_id.items()}
    outgoing: dict[str, set[str]] = {}
    for sid, deps in incoming.items():
        for dep in deps:
            outgoing.setdefault(dep, set()).add(sid)

    ready = sorted([sid for sid, deps in incoming.items() if not deps])
    order: list[str] = []
    while ready:
        sid = ready.pop(0)
        order.append(sid)
        for nxt in sorted(outgoing.get(sid, set())):
            incoming[nxt].discard(sid)
            if not incoming[nxt] and nxt not in order and nxt not in ready:
                ready.append(nxt)
                ready.sort()

    if len(order) != len(by_id):
        remaining = [sid for sid in by_id if sid not in order]
        raise ValueError(f"Cycle detected in steps (remaining): {remaining}")

    return [by_id[sid] for sid in order]


def run_experiment(
    *,
    spec_path: Path,
    roots: Roots,
    set_overrides: list[tuple[str, str]],
    set_string_overrides: list[tuple[str, str]],
    salt: str | None,
    enforce_clean: bool,
    follow_steps: bool = False,
    stderr_tail_lines: int = 120,
) -> dict[str, str]:
    raw_base = load_and_validate(
        spec_path=spec_path,
        set_overrides=set_overrides,
        set_string_overrides=set_string_overrides,
    )
    kind = (raw_base.get("env") or {}).get("kind", "local")
    raw_base = apply_computed_defaults(raw_base, project_root=roots.project_root, kind=kind)
    name = str(raw_base.get("name"))

    raw_hash = copy.deepcopy(raw_base)
    raw_runtime = copy.deepcopy(raw_base)

    # Apply effective defaults that affect run_key hashing.
    env_block_hash = dict(raw_hash.get("env") or {})
    env_vars_hash = dict(env_block_hash.get("env") or {})
    if kind == "docker":
        hf_home_hash = "/workspace/artifacts/hf_home"
    else:
        hf_home_hash = f"{ARTIFACTS_ROOT_PLACEHOLDER}/hf_home"
    env_vars_hash = _resolved_offline_env(
        offline=bool(env_block_hash.get("offline")), hf_home=hf_home_hash, existing=env_vars_hash
    )
    env_block_hash["env"] = env_vars_hash
    raw_hash["env"] = env_block_hash

    if kind == "docker":
        _prepare_effective_docker(raw_hash, roots=roots, for_hashing=True)

    git = collect_git_info(project_root=roots.project_root)
    if enforce_clean and git.dirty:
        raise RuntimeError("Git working tree is dirty (--enforce-clean is set)")

    placeholder_id = "<RUN_ID>"
    if kind == "docker":
        placeholder_run = {
            "id": placeholder_id,
            "runs": f"/workspace/runs/{name}/{placeholder_id}",
            "artifacts": f"/workspace/artifacts/{name}/{placeholder_id}",
        }
    else:
        placeholder_run = {
            "id": placeholder_id,
            "runs": f"{RUNS_ROOT_PLACEHOLDER}/{name}/{placeholder_id}",
            "artifacts": f"{ARTIFACTS_ROOT_PLACEHOLDER}/{name}/{placeholder_id}",
        }
    resolved_hashing = resolve_for_hashing(
        raw_hash, project_root=roots.project_root, placeholder_run=placeholder_run
    )

    # Inputs fingerprints (best-effort, only if configured).
    input_fps: dict[str, Any] = {}
    inputs = resolved_hashing.get("inputs") or {}
    for k, v in inputs.items():
        if not isinstance(v, dict):
            continue
        path_s = v.get("path")
        fp = v.get("fingerprint") or {}
        if not isinstance(path_s, str):
            continue
        p = resolve_relpath(path_s, base_dir=roots.project_root)
        kind_s = str(fp.get("kind") or "none")
        include = list(fp.get("include") or [])
        exclude = list(fp.get("exclude") or [])
        res = fingerprint_path(p, kind=kind_s, include=include, exclude=exclude)
        input_fps[k] = {
            "path": str(p),
            "fingerprint": {"kind": res.kind, "value": res.value, "files_hashed": res.files_hashed},
            "include": include,
            "exclude": exclude,
        }

    docker_meta: dict[str, Any] | None = None
    if kind == "docker":
        docker_cfg = (resolved_hashing.get("env") or {}).get("docker") or {}
        image = docker_cfg.get("image")
        if not image:
            raise RuntimeError("env.docker.image is required for docker runs")
        allow_unverified = bool(docker_cfg.get("allow_unverified_image", False))
        docker_meta = inspect_image(image=str(image), cwd=roots.project_root)
        if docker_meta is None and not allow_unverified:
            raise RuntimeError(
                f"docker image inspect failed for image={image!r}; set env.docker.allow_unverified_image: true to proceed"
            )
        if docker_meta is None:
            docker_meta = {
                "image": str(image),
                "image_id": None,
                "repo_digests": [],
                "repo_tags": [],
                "unverified": True,
            }

    run_key_material = {
        "spec": resolved_hashing,
        "git": {
            "commit": git.commit,
            "dirty": git.dirty,
            "diff_hash": git.diff_hash,
        },
        "docker_image": docker_meta,
        "inputs_fingerprints": input_fps,
        "salt": salt,
    }
    run_key = compute_run_key(run_key_material)

    paths = get_run_paths(roots=roots, name=name, run_key=run_key)
    if paths.run_dir.exists():
        raise RuntimeError(f"Run already exists: {paths.run_dir} (use --salt for a new run)")

    init_run_dirs(paths)
    ensure_dir(paths.artifacts_dir)
    shutil.copyfile(spec_path, paths.run_dir / "spec.yaml")

    if kind == "docker":
        run_ctx = {
            "id": run_key,
            "runs": f"/workspace/runs/{name}/{run_key}",
            "artifacts": f"/workspace/artifacts/{name}/{run_key}",
        }
    else:
        run_ctx = {"id": run_key, "runs": str(paths.run_dir), "artifacts": str(paths.artifacts_dir)}

    resolved_final = resolve_final(raw_runtime, project_root=roots.project_root, run_ctx=run_ctx)
    env_vars = _prepare_effective_env(resolved_final, roots=roots, kind=kind)
    if kind == "docker":
        _prepare_effective_docker(resolved_final, roots=roots, for_hashing=False)

    env_block = resolved_final.get("env") or {}
    if kind == "local" and bool(env_block.get("offline")):
        print(
            "warning: offline mode for local runs is best-effort; network is not sandboxed",
            file=sys.stderr,
        )

    ExperimentSpec.model_validate(resolved_final)
    write_json(paths.run_dir / "resolved_spec.json", resolved_final)

    started_at = utc_now_iso()
    started_at_compact = iso_to_compact_utc(started_at)

    # Create a time-indexed alias to make it easier to find runs on disk.
    # This does not affect the stable run_key (hash) used for identity/deduping.
    safe_symlink(
        link_path=roots.runs_root / name / "_by_time" / f"{started_at_compact}_{run_key}",
        target_path=paths.run_dir,
    )
    safe_symlink(
        link_path=roots.artifacts_root / name / "_by_time" / f"{started_at_compact}_{run_key}",
        target_path=paths.artifacts_dir,
    )
    run_json: dict[str, Any] = {
        "name": name,
        "run_key": run_key,
        "created_at_utc": started_at,
        "started_at_utc": started_at,
        "pid": os.getpid(),
        "proc_start_ticks": _proc_start_ticks_linux(os.getpid()),
        "state": "running",
        "paths": {
            "host": {"run_dir": str(paths.run_dir), "artifacts_dir": str(paths.artifacts_dir)},
            "container": run_ctx if kind == "docker" else None,
        },
        "allocated_gpus_host": [],
        "steps": [],
        "run_key_material": run_key_material,
    }
    write_run_json(paths, run_json)

    write_git_provenance(paths.provenance_dir, git)
    write_nvidia_smi(paths.provenance_dir)
    write_host_provenance(
        paths.provenance_dir,
        extra={
            "project_root": str(roots.project_root),
            "runs_root": str(roots.runs_root),
            "artifacts_root": str(roots.artifacts_root),
            "spec_path": str(spec_path.resolve()),
            "resolved_paths": {
                "run_dir": str(paths.run_dir),
                "artifacts_dir": str(paths.artifacts_dir),
            },
        },
    )
    write_env_provenance(
        paths.provenance_dir,
        env_allow=["PATH", "CUDA_VISIBLE_DEVICES"],
        explicit_env=env_vars,
    )
    if kind != "docker":
        write_python_and_freeze(paths.provenance_dir)
    if input_fps:
        write_json(paths.provenance_dir / "inputs.json", input_fps)
    if kind == "docker":
        docker_effective = (resolved_final.get("env") or {}).get("docker")
        write_json(
            paths.provenance_dir / "docker.json",
            {
                "effective": docker_effective,
                "image": docker_meta,
                "offline_network_enforced": (docker_effective or {}).get("network") == "none"
                and bool((resolved_final.get("env") or {}).get("offline")),
            },
        )

    pool = GpuPool(locks_dir=(roots.runs_root / "_locks"))
    allocation = None
    try:
        allocation = allocate_gpus(
            pool,
            (resolved_final.get("resources") or {}).get("gpus", 0),
            run_path=str(paths.run_dir),
            run_key=run_key,
            name=name,
        )
        run_json["allocated_gpus_host"] = allocation.gpu_ids
        write_run_json(paths, run_json)
    except Exception as e:
        run_json["state"] = "failed"
        run_json["finished_at_utc"] = utc_now_iso()
        existing = run_json.get("error")
        if isinstance(existing, dict):
            existing.setdefault("message", str(e))
            existing["traceback"] = traceback.format_exc()
            run_json["error"] = existing
        else:
            run_json["error"] = {"message": str(e), "traceback": traceback.format_exc()}
        write_run_json(paths, run_json)
        raise

    try:
        if kind == "docker":
            executor = DockerExecutor()
            docker_block = (resolved_final.get("env") or {}).get("docker") or {}
            ctx = RunContext(
                name=name,
                run_key=run_key,
                project_root=roots.project_root,
                run_dir=paths.run_dir,
                artifacts_dir=paths.artifacts_dir,
                workdir=str((resolved_final.get("env") or {}).get("workdir")),
                env=env_vars,
                offline=bool((resolved_final.get("env") or {}).get("offline")),
                kind=kind,
                docker=docker_block,
                allocated_gpus_host=allocation.gpu_ids,
                stream_logs=bool(follow_steps),
            )
        else:
            executor = LocalExecutor()
            ctx = RunContext(
                name=name,
                run_key=run_key,
                project_root=roots.project_root,
                run_dir=paths.run_dir,
                artifacts_dir=paths.artifacts_dir,
                workdir=str((resolved_final.get("env") or {}).get("workdir")),
                env=env_vars,
                offline=bool((resolved_final.get("env") or {}).get("offline")),
                kind=kind,
                docker=None,
                allocated_gpus_host=allocation.gpu_ids,
                stream_logs=bool(follow_steps),
            )

        executor.prepare_run(ctx)

        steps = _validate_and_toposort_steps(list(resolved_final.get("steps") or []))
        for idx, step in enumerate(steps):
            step_id = step["id"]
            step_dir = paths.steps_dir / f"{idx:02d}_{step_id}"
            ensure_dir(step_dir)
            cmd = list(step.get("cmd") or [])
            timeout_s = step.get("timeout_seconds")
            step_artifacts_dir = ((step.get("outputs") or {}) or {}).get("artifacts_dir")
            if step_artifacts_dir == f"{run_ctx['artifacts']}/{step_id}":
                ensure_dir(paths.artifacts_dir / step_id)

            result = executor.run_step(
                ctx,
                step_index=idx,
                step_id=step_id,
                cmd=cmd,
                step_dir=step_dir,
                timeout_seconds=timeout_s,
                step_artifacts_dir=step_artifacts_dir,
            )
            run_json["steps"].append({"step_id": step_id, "rc": result.rc, "dir": str(step_dir)})
            write_run_json(paths, run_json)
            if result.rc != 0:
                stderr_fp = step_dir / "stderr.log"
                stdout_fp = step_dir / "stdout.log"
                tail = tail_text_lines(stderr_fp, n=int(stderr_tail_lines))
                if tail:
                    print("", file=sys.stderr)
                    print(f"[exp-harness] step failed: {step_id} (rc={result.rc})", file=sys.stderr)
                    print(f"[exp-harness] command: {step_dir / 'command.txt'}", file=sys.stderr)
                    print(f"[exp-harness] stderr:  {stderr_fp}", file=sys.stderr)
                    print(f"[exp-harness] stdout:  {stdout_fp}", file=sys.stderr)
                    print(
                        f"[exp-harness] stderr tail (last {stderr_tail_lines} lines):",
                        file=sys.stderr,
                    )
                    print(tail.rstrip("\n"), file=sys.stderr)
                    print("", file=sys.stderr)
                run_json.setdefault("error", {})
                run_json["error"] = {
                    "message": f"Step failed: {step_id} (rc={result.rc})",
                    "step_id": step_id,
                    "rc": int(result.rc),
                    "stderr_tail": tail,
                    "stderr_log": str(stderr_fp),
                    "stdout_log": str(stdout_fp),
                }
                write_run_json(paths, run_json)
                raise RuntimeError(f"Step failed: {step_id} (rc={result.rc})")

        executor.finalize_run(ctx)
        run_json["state"] = "succeeded"
        run_json["finished_at_utc"] = utc_now_iso()
        write_run_json(paths, run_json)
    except KeyboardInterrupt:
        run_json["state"] = "interrupted"
        run_json["finished_at_utc"] = utc_now_iso()
        write_run_json(paths, run_json)
        raise
    except Exception as e:
        run_json["state"] = "failed"
        run_json["finished_at_utc"] = utc_now_iso()
        existing = run_json.get("error")
        if isinstance(existing, dict):
            existing.setdefault("message", str(e))
            existing["traceback"] = traceback.format_exc()
            run_json["error"] = existing
        else:
            run_json["error"] = {"message": str(e), "traceback": traceback.format_exc()}
        write_run_json(paths, run_json)
        raise
    finally:
        if allocation is not None:
            for g in allocation.gpu_ids:
                pool.release(g, expected_pid=allocation.pid)

    return {
        "name": name,
        "run_key": run_key,
        "run_dir": str(paths.run_dir),
        "artifacts_dir": str(paths.artifacts_dir),
    }
