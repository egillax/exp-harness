from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from exp_harness.config import Roots
from exp_harness.docker_utils import inspect_image
from exp_harness.errors import GitDirtyWorktreeError, RunResumeError
from exp_harness.executors.docker import DockerExecutor
from exp_harness.executors.local import LocalExecutor
from exp_harness.provenance.git import collect_git_info
from exp_harness.resolve import apply_computed_defaults
from exp_harness.run.execution import ensure_step_records
from exp_harness.run.identity import build_run_identity
from exp_harness.run.phases import (
    PreparedRun,
    ResourceAllocation,
    phase_allocate_resources,
    phase_execute_run,
    phase_initialize_metadata,
    phase_prepare_run,
    phase_release_resources,
)
from exp_harness.run.provenance import proc_start_ticks_linux
from exp_harness.run.runtime_env import prepare_effective_docker, prepare_effective_env
from exp_harness.run.state import mark_failed_with_traceback, mark_interrupted, mark_succeeded
from exp_harness.run.step_graph import validate_and_toposort_steps
from exp_harness.spec import ExperimentSpec
from exp_harness.store import get_run_paths, resolve_run_dir, write_run_json
from exp_harness.utils import utc_now_iso


def _load_spec_mapping(spec_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Spec YAML must be a mapping at the top level: {spec_path}")
    if raw.get("extends") is not None:
        raise ValueError(
            "`extends` is no longer supported; use Hydra config groups and overrides instead"
        )
    ExperimentSpec.model_validate(raw)
    return raw


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _record_spec_fingerprints(*, run_json: dict[str, Any], run_dir: Path) -> None:
    spec_fp = run_dir / "spec.yaml"
    resolved_fp = run_dir / "resolved_spec.json"
    if not spec_fp.is_file() or not resolved_fp.is_file():
        return
    run_json["spec"] = {
        "spec_sha256": _sha256_file(spec_fp),
        "resolved_spec_sha256": _sha256_file(resolved_fp),
    }


def _verify_resume_spec_fingerprints(
    *,
    run_json: dict[str, Any],
    run_dir: Path,
    allow_spec_drift: bool,
) -> None:
    spec_meta = run_json.get("spec")
    if not isinstance(spec_meta, dict):
        if allow_spec_drift:
            return
        raise RunResumeError(
            "Run metadata does not contain spec fingerprints; pass --allow-spec-drift to resume"
        )

    expected_spec = spec_meta.get("spec_sha256")
    expected_resolved = spec_meta.get("resolved_spec_sha256")
    if not isinstance(expected_spec, str) or not isinstance(expected_resolved, str):
        if allow_spec_drift:
            return
        raise RunResumeError(
            "Run metadata spec fingerprints are incomplete; pass --allow-spec-drift to resume"
        )

    cur_spec = _sha256_file(run_dir / "spec.yaml")
    cur_resolved = _sha256_file(run_dir / "resolved_spec.json")
    if (cur_spec != expected_spec or cur_resolved != expected_resolved) and not allow_spec_drift:
        raise RunResumeError(
            "Spec drift detected for run artifacts; pass --allow-spec-drift to resume anyway"
        )


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RunResumeError(f"Could not parse {label}: {path}") from exc
    if not isinstance(raw, dict):
        raise RunResumeError(f"{label} must be a JSON mapping: {path}")
    return raw


def run_experiment(
    *,
    spec_path: Path,
    roots: Roots,
    set_overrides: list[tuple[str, str]],
    set_string_overrides: list[tuple[str, str]],
    salt: str | None,
    run_label: str | None = None,
    enforce_clean: bool,
    follow_steps: bool = False,
    stderr_tail_lines: int = 120,
) -> dict[str, str]:
    if set_overrides or set_string_overrides:
        raise ValueError(
            "`--set`/`--set-str` overrides are no longer supported; use Hydra overrides"
        )

    raw_base = _load_spec_mapping(spec_path)
    kind = (raw_base.get("env") or {}).get("kind", "local")
    raw_base = apply_computed_defaults(raw_base, project_root=roots.project_root, kind=kind)
    name = str(raw_base.get("name"))
    spec_run_label = raw_base.get("run_label")
    label_from_spec = str(spec_run_label) if isinstance(spec_run_label, str) else None
    effective_run_label = run_label or label_from_spec or spec_path.stem

    raw_hash = copy.deepcopy(raw_base)
    raw_runtime = copy.deepcopy(raw_base)

    # run_label is UX metadata and should not affect run identity/hash.
    raw_hash.pop("run_label", None)

    git = collect_git_info(project_root=roots.project_root)
    if enforce_clean and git.dirty:
        raise GitDirtyWorktreeError("Git working tree is dirty (--enforce-clean is set)")

    identity = build_run_identity(
        raw_hash=raw_hash,
        roots=roots,
        name=name,
        kind=kind,
        effective_run_label=effective_run_label,
        salt=salt,
        git=git,
        inspect_image_fn=inspect_image,
    )
    run_key = identity["run_key"]
    run_id = identity["run_id"]
    started_at = identity["started_at"]
    run_key_material = identity["run_key_material"]
    input_fps = identity["input_fps"]
    docker_meta = identity["docker_meta"]

    prepared = phase_prepare_run(
        roots=roots,
        spec_path=spec_path,
        raw_runtime=raw_runtime,
        name=name,
        kind=kind,
        run_key=run_key,
        run_id=run_id,
    )
    run_json = phase_initialize_metadata(
        prepared=prepared,
        roots=roots,
        spec_path=spec_path,
        git=git,
        name=name,
        kind=kind,
        effective_run_label=effective_run_label,
        run_id=run_id,
        run_key=run_key,
        started_at=started_at,
        run_key_material=run_key_material,
        input_fps=input_fps,
        docker_meta=docker_meta,
    )
    _record_spec_fingerprints(run_json=run_json, run_dir=prepared.paths.run_dir)
    write_run_json(prepared.paths, run_json)

    resources: ResourceAllocation | None = None
    try:
        resources = phase_allocate_resources(
            roots=roots,
            prepared=prepared,
            run_json=run_json,
            name=name,
            run_key=run_key,
        )
    except Exception as e:
        mark_failed_with_traceback(run_json, error=e)
        write_run_json(prepared.paths, run_json)
        raise

    try:
        assert resources is not None
        phase_execute_run(
            roots=roots,
            prepared=prepared,
            resources=resources,
            run_json=run_json,
            kind=kind,
            name=name,
            run_key=run_key,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
            docker_executor_factory=DockerExecutor,
            local_executor_factory=LocalExecutor,
            resume_mode=False,
        )
        mark_succeeded(run_json)
        write_run_json(prepared.paths, run_json)
    except KeyboardInterrupt:
        mark_interrupted(run_json)
        write_run_json(prepared.paths, run_json)
        raise
    except Exception as e:
        mark_failed_with_traceback(run_json, error=e)
        write_run_json(prepared.paths, run_json)
        raise
    finally:
        if resources is not None:
            phase_release_resources(resources=resources)

    return {
        "name": name,
        "run_id": run_id,
        "run_key": run_key,
        "run_dir": str(prepared.paths.run_dir),
        "artifacts_dir": str(prepared.paths.artifacts_dir),
    }


def resume_experiment(
    *,
    roots: Roots,
    name: str,
    run_key: str,
    enforce_clean: bool,
    allow_spec_drift: bool = False,
    force: bool = False,
    follow_steps: bool = False,
    stderr_tail_lines: int = 120,
) -> dict[str, str]:
    run_dir = resolve_run_dir(roots=roots, name=name, run_key=run_key)
    if run_dir is None:
        raise RunResumeError(f"Run not found: name={name} run_key={run_key}")

    run_json = _load_json_mapping(run_dir / "run.json", label="run.json")
    resolved_final = _load_json_mapping(run_dir / "resolved_spec.json", label="resolved_spec.json")
    ExperimentSpec.model_validate(resolved_final)

    run_state = str(run_json.get("state") or "")
    if run_state == "succeeded" and not force:
        raise RunResumeError("Run already succeeded; pass --force to resume anyway")

    _verify_resume_spec_fingerprints(
        run_json=run_json,
        run_dir=run_dir,
        allow_spec_drift=allow_spec_drift,
    )

    git = collect_git_info(project_root=roots.project_root)
    if enforce_clean and git.dirty:
        raise GitDirtyWorktreeError("Git working tree is dirty (--enforce-clean is set)")

    run_id = str(run_json.get("run_id") or run_dir.name)
    kind = str((resolved_final.get("env") or {}).get("kind") or "local")
    paths = get_run_paths(roots=roots, name=name, run_id=run_id)
    run_ctx_raw = ((run_json.get("paths") or {}).get("container")) if kind == "docker" else None
    if isinstance(run_ctx_raw, dict):
        run_ctx = {
            "id": str(run_ctx_raw.get("id") or run_key),
            "runs": str(run_ctx_raw.get("runs") or f"/workspace/runs/{name}/{run_id}"),
            "artifacts": str(
                run_ctx_raw.get("artifacts") or f"/workspace/artifacts/{name}/{run_id}"
            ),
        }
    elif kind == "docker":
        run_ctx = {
            "id": run_key,
            "runs": f"/workspace/runs/{name}/{run_id}",
            "artifacts": f"/workspace/artifacts/{name}/{run_id}",
        }
    else:
        run_ctx = {"id": run_key, "runs": str(paths.run_dir), "artifacts": str(paths.artifacts_dir)}

    env_vars = prepare_effective_env(resolved_final, roots=roots, kind=kind)
    if kind == "docker":
        prepare_effective_docker(resolved_final, roots=roots, for_hashing=False)
    prepared = PreparedRun(
        paths=paths, run_ctx=run_ctx, resolved_final=resolved_final, env_vars=env_vars
    )

    ordered_steps = validate_and_toposort_steps(list(resolved_final.get("steps") or []))
    step_records = ensure_step_records(
        ordered_steps=ordered_steps,
        paths=paths,
        run_json=run_json,
        write_run_json_fn=write_run_json,
    )
    first_incomplete = next(
        (str(rec.get("step_id")) for rec in step_records if rec.get("state") != "succeeded"),
        None,
    )
    if first_incomplete is None and not force:
        raise RunResumeError(
            "All steps are already marked succeeded; pass --force to rerun according to step policies"
        )

    pid = os.getpid()
    run_json["pid"] = pid
    run_json["proc_start_ticks"] = proc_start_ticks_linux(pid)
    run_json["state"] = "running"
    run_json.pop("error", None)

    attempts = run_json.get("resume_attempts")
    attempt_list: list[dict[str, Any]]
    if isinstance(attempts, list):
        attempt_list = [a for a in attempts if isinstance(a, dict)]
    else:
        attempt_list = []
    resume_entry = {
        "attempt": len(attempt_list) + 1,
        "resumed_at_utc": utc_now_iso(),
        "resumed_from_step": first_incomplete,
    }
    attempt_list.append(resume_entry)
    run_json["resume_attempts"] = attempt_list
    write_run_json(paths, run_json)

    resources: ResourceAllocation | None = None
    try:
        resources = phase_allocate_resources(
            roots=roots,
            prepared=prepared,
            run_json=run_json,
            name=name,
            run_key=run_key,
        )
    except Exception as e:
        mark_failed_with_traceback(run_json, error=e)
        write_run_json(paths, run_json)
        raise

    try:
        assert resources is not None
        phase_execute_run(
            roots=roots,
            prepared=prepared,
            resources=resources,
            run_json=run_json,
            kind=kind,
            name=name,
            run_key=run_key,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
            docker_executor_factory=DockerExecutor,
            local_executor_factory=LocalExecutor,
            resume_mode=True,
        )
        mark_succeeded(run_json)
        write_run_json(paths, run_json)
    except KeyboardInterrupt:
        mark_interrupted(run_json)
        write_run_json(paths, run_json)
        raise
    except Exception as e:
        mark_failed_with_traceback(run_json, error=e)
        write_run_json(paths, run_json)
        raise
    finally:
        if resources is not None:
            phase_release_resources(resources=resources)

    return {
        "name": name,
        "run_id": run_id,
        "run_key": run_key,
        "run_dir": str(paths.run_dir),
        "artifacts_dir": str(paths.artifacts_dir),
    }
