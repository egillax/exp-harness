from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from exp_harness.config import Roots
from exp_harness.docker_utils import inspect_image
from exp_harness.errors import GitDirtyWorktreeError, RunResumeError
from exp_harness.executors.docker import DockerExecutor
from exp_harness.executors.local import LocalExecutor
from exp_harness.provenance.git import collect_git_info
from exp_harness.resolve import apply_computed_defaults, resolve_final
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
from exp_harness.run.repair import (
    build_step_contract,
    contract_diff,
    contract_hash,
    format_diff_line,
    normalize_resume_step_state,
    parse_allowed_config_drift,
)
from exp_harness.run.runtime_env import prepare_effective_docker, prepare_effective_env
from exp_harness.run.state import mark_failed_with_traceback, mark_interrupted, mark_succeeded
from exp_harness.run.step_graph import validate_and_toposort_steps
from exp_harness.spec import ExperimentSpec
from exp_harness.store import get_run_paths, resolve_run_dir, write_run_json
from exp_harness.utils import ensure_dir, utc_now_iso, write_json


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


def _record_initial_spec_metadata(*, run_json: dict[str, Any], run_dir: Path) -> None:
    spec_fp = run_dir / "spec.yaml"
    resolved_fp = run_dir / "resolved_spec.json"
    if not spec_fp.is_file() or not resolved_fp.is_file():
        return
    spec_meta = {
        "spec_sha256": _sha256_file(spec_fp),
        "resolved_spec_sha256": _sha256_file(resolved_fp),
    }
    run_json["spec"] = spec_meta
    run_json["active_spec"] = {
        "source": "original",
        "spec_path": "spec.yaml",
        "resolved_spec_path": "resolved_spec.json",
        "spec_sha256": spec_meta["spec_sha256"],
        "resolved_spec_sha256": spec_meta["resolved_spec_sha256"],
        "updated_at_utc": utc_now_iso(),
    }


def _path_from_run_dir(run_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else run_dir / path


def _relative_to_run_dir(run_dir: Path, path: Path) -> str:
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _resolve_active_spec_paths(
    *, run_json: dict[str, Any], run_dir: Path
) -> tuple[dict[str, Any], Path, Path]:
    active_spec = run_json.get("active_spec")
    if isinstance(active_spec, dict):
        spec_path = active_spec.get("spec_path")
        resolved_path = active_spec.get("resolved_spec_path")
        if isinstance(spec_path, str) and isinstance(resolved_path, str):
            return (
                active_spec,
                _path_from_run_dir(run_dir, spec_path),
                _path_from_run_dir(run_dir, resolved_path),
            )

    legacy_meta = run_json.get("spec")
    meta = legacy_meta if isinstance(legacy_meta, dict) else {}
    return meta, run_dir / "spec.yaml", run_dir / "resolved_spec.json"


def _set_active_spec_metadata(
    *,
    run_json: dict[str, Any],
    run_dir: Path,
    spec_path: Path,
    resolved_path: Path,
    source: str,
    repair_event_index: int | None = None,
) -> None:
    active_spec: dict[str, Any] = {
        "source": source,
        "spec_path": _relative_to_run_dir(run_dir, spec_path),
        "resolved_spec_path": _relative_to_run_dir(run_dir, resolved_path),
        "spec_sha256": _sha256_file(spec_path),
        "resolved_spec_sha256": _sha256_file(resolved_path),
        "updated_at_utc": utc_now_iso(),
    }
    if repair_event_index is not None:
        active_spec["repair_event_index"] = repair_event_index
    run_json["active_spec"] = active_spec


def _verify_active_spec_fingerprints(*, run_json: dict[str, Any], run_dir: Path) -> None:
    spec_meta, spec_fp, resolved_fp = _resolve_active_spec_paths(run_json=run_json, run_dir=run_dir)
    if not spec_fp.is_file() or not resolved_fp.is_file():
        raise RunResumeError("Active resume spec artifacts are missing from the run directory")

    expected_spec = spec_meta.get("spec_sha256") if isinstance(spec_meta, dict) else None
    expected_resolved = (
        spec_meta.get("resolved_spec_sha256") if isinstance(spec_meta, dict) else None
    )
    if not isinstance(expected_spec, str) or not isinstance(expected_resolved, str):
        return

    cur_spec = _sha256_file(spec_fp)
    cur_resolved = _sha256_file(resolved_fp)
    if cur_spec != expected_spec or cur_resolved != expected_resolved:
        raise RunResumeError("Spec drift detected for active resume spec artifacts")


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RunResumeError(f"Could not parse {label}: {path}") from exc
    if not isinstance(raw, dict):
        raise RunResumeError(f"{label} must be a JSON mapping: {path}")
    return raw


def _build_run_ctx(
    *,
    kind: str,
    run_json: dict[str, Any],
    name: str,
    run_key: str,
    run_id: str,
    paths,
) -> dict[str, str]:
    run_ctx_raw = ((run_json.get("paths") or {}).get("container")) if kind == "docker" else None
    if isinstance(run_ctx_raw, dict):
        return {
            "id": str(run_ctx_raw.get("id") or run_key),
            "runs": str(run_ctx_raw.get("runs") or f"/workspace/runs/{name}/{run_id}"),
            "artifacts": str(
                run_ctx_raw.get("artifacts") or f"/workspace/artifacts/{name}/{run_id}"
            ),
        }
    if kind == "docker":
        return {
            "id": run_key,
            "runs": f"/workspace/runs/{name}/{run_id}",
            "artifacts": f"/workspace/artifacts/{name}/{run_id}",
        }
    return {"id": run_key, "runs": str(paths.run_dir), "artifacts": str(paths.artifacts_dir)}


def _resolve_runtime_spec(
    *,
    spec_path: Path,
    roots: Roots,
    run_ctx: dict[str, str],
    expected_name: str,
    expected_kind: str | None = None,
) -> tuple[dict[str, Any], str]:
    raw_base = _load_spec_mapping(spec_path)
    spec_name = str(raw_base.get("name") or "")
    if spec_name != expected_name:
        raise RunResumeError(
            f"Repair spec name mismatch: expected {expected_name!r}, got {spec_name!r}"
        )
    kind = str((raw_base.get("env") or {}).get("kind", "local"))
    if expected_kind is not None and kind != expected_kind:
        raise RunResumeError(f"Repair spec env.kind must stay {expected_kind!r}; got {kind!r}")
    raw_base = apply_computed_defaults(raw_base, project_root=roots.project_root, kind=kind)
    resolved_final = resolve_final(raw_base, project_root=roots.project_root, run_ctx=run_ctx)
    env_vars = prepare_effective_env(resolved_final, roots=roots, kind=kind)
    if kind == "docker":
        prepare_effective_docker(resolved_final, roots=roots, for_hashing=False)
    resolved_final["env"]["env"] = env_vars
    ExperimentSpec.model_validate(resolved_final)
    return resolved_final, kind


def _validate_repair_shape(
    *, stored_steps: list[dict[str, Any]], candidate_steps: list[dict[str, Any]]
) -> None:
    stored_ids = [str(step["id"]) for step in stored_steps]
    candidate_ids = [str(step["id"]) for step in candidate_steps]
    if set(stored_ids) != set(candidate_ids):
        raise RunResumeError("Repair spec must preserve the exact step id set")

    candidate_by_id = {str(step["id"]): step for step in candidate_steps}
    for step_id in stored_ids:
        old_needs = list(
            (next(step for step in stored_steps if str(step["id"]) == step_id)).get("needs") or []
        )
        new_needs = list(candidate_by_id[step_id].get("needs") or [])
        if old_needs != new_needs:
            raise RunResumeError(f"Repair spec must preserve dependencies for step {step_id!r}")


def _analyze_repair_drift(
    *,
    stored_resolved: dict[str, Any],
    candidate_resolved: dict[str, Any],
    step_records: list[dict[str, Any]],
    allowed_states: set[str],
) -> dict[str, Any]:
    stored_steps = validate_and_toposort_steps(list(stored_resolved.get("steps") or []))
    candidate_steps = validate_and_toposort_steps(list(candidate_resolved.get("steps") or []))
    _validate_repair_shape(stored_steps=stored_steps, candidate_steps=candidate_steps)

    candidate_by_id = {str(step["id"]): step for step in candidate_steps}
    records_by_id = {str(rec.get("step_id")): rec for rec in step_records}

    changed_steps: list[dict[str, Any]] = []
    disallowed_steps: list[dict[str, Any]] = []
    unchanged_incomplete: list[dict[str, str]] = []
    frozen_succeeded: list[str] = []

    for step in stored_steps:
        step_id = str(step["id"])
        candidate_step = candidate_by_id[step_id]
        record = records_by_id.get(step_id, {"step_id": step_id, "state": "pending"})
        step_state = normalize_resume_step_state(record.get("state"))
        if step_state == "succeeded":
            frozen_succeeded.append(step_id)

        old_contract = build_step_contract(step, resolved_final=stored_resolved)
        new_contract = build_step_contract(candidate_step, resolved_final=candidate_resolved)
        old_hash = contract_hash(old_contract)
        new_hash = contract_hash(new_contract)
        if old_hash == new_hash:
            if step_state != "succeeded":
                unchanged_incomplete.append({"id": step_id, "state": step_state})
            continue

        diff_payload = contract_diff(old_contract=old_contract, new_contract=new_contract)
        entry = {
            "id": step_id,
            "state": step_state,
            "old_hash": old_hash,
            "new_hash": new_hash,
            "diff": diff_payload,
            "diff_summary": [format_diff_line(item) for item in diff_payload],
        }
        if step_state == "succeeded" or step_state not in allowed_states:
            disallowed_steps.append(entry)
        else:
            changed_steps.append(entry)

    return {
        "changed_steps": changed_steps,
        "disallowed_steps": disallowed_steps,
        "unchanged_incomplete": unchanged_incomplete,
        "frozen_succeeded": frozen_succeeded,
    }


def _raise_disallowed_drift(*, disallowed_steps: list[dict[str, Any]], repair_mode: bool) -> None:
    if not disallowed_steps:
        return
    details: list[str] = []
    for entry in disallowed_steps[:3]:
        summary = entry.get("diff_summary") or ["contract changed"]
        details.append(f"{entry['id']} (state={entry['state']}): {summary[0]}")
    prefix = (
        "Disallowed config drift detected for repair resume"
        if repair_mode
        else "Config drift detected for resume"
    )
    raise RunResumeError(f"{prefix}: {'; '.join(details)}")


def _print_resume_summary(
    *,
    mode: str,
    allowed_states: set[str],
    frozen_succeeded: list[str],
    changed_steps: list[dict[str, Any]],
    unchanged_incomplete: list[dict[str, str]],
    dry_run: bool,
) -> None:
    suffix = f" allowed_states={','.join(sorted(allowed_states))}" if allowed_states else ""
    print(f"[resume] mode={mode}{suffix}")
    print(f"[resume] succeeded step contracts frozen: {len(frozen_succeeded)}")
    if changed_steps:
        print(f"[resume] accepted config drift for {len(changed_steps)} step(s):")
        for entry in changed_steps:
            print(f"  - {entry['id']} (state={entry['state']})")
            for diff_line in entry.get("diff_summary", [])[:3]:
                print(f"    {diff_line}")
    if unchanged_incomplete:
        print(f"[resume] unchanged incomplete steps: {len(unchanged_incomplete)}")
    if dry_run:
        print("[resume] dry-run only; no execution started")


def _write_repair_spec_artifacts(
    *,
    run_dir: Path,
    repair_spec: Path,
    resolved_final: dict[str, Any],
    repair_event_index: int,
) -> tuple[Path, Path]:
    repair_dir = run_dir / "provenance" / "repairs" / f"repair_{repair_event_index:04d}"
    ensure_dir(repair_dir)
    spec_copy = repair_dir / "spec.yaml"
    resolved_copy = repair_dir / "resolved_spec.json"
    shutil.copyfile(repair_spec, spec_copy)
    write_json(resolved_copy, resolved_final)
    return spec_copy, resolved_copy


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
    _record_initial_spec_metadata(run_json=run_json, run_dir=prepared.paths.run_dir)
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
    repair_spec: Path | None = None,
    allow_config_drift: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    follow_steps: bool = False,
    stderr_tail_lines: int = 120,
) -> dict[str, str]:
    run_dir = resolve_run_dir(roots=roots, name=name, run_key=run_key)
    if run_dir is None:
        raise RunResumeError(f"Run not found: name={name} run_key={run_key}")

    run_json = _load_json_mapping(run_dir / "run.json", label="run.json")
    allowed_drift_states = parse_allowed_config_drift(allow_config_drift)
    repair_mode = repair_spec is not None or bool(allowed_drift_states)
    if repair_spec is None and allowed_drift_states:
        raise RunResumeError("--allow-config-drift requires --repair-spec")
    if repair_spec is not None and not allowed_drift_states:
        raise RunResumeError("--repair-spec requires --allow-config-drift")

    run_state = str(run_json.get("state") or "")
    if run_state == "succeeded" and not force:
        raise RunResumeError("Run already succeeded; pass --force to resume anyway")

    _verify_active_spec_fingerprints(run_json=run_json, run_dir=run_dir)
    _, _, active_resolved_fp = _resolve_active_spec_paths(run_json=run_json, run_dir=run_dir)
    active_resolved = _load_json_mapping(active_resolved_fp, label="active resolved spec")
    ExperimentSpec.model_validate(active_resolved)

    git = collect_git_info(project_root=roots.project_root)
    if enforce_clean and git.dirty:
        raise GitDirtyWorktreeError("Git working tree is dirty (--enforce-clean is set)")

    run_id = str(run_json.get("run_id") or run_dir.name)
    kind = str((active_resolved.get("env") or {}).get("kind") or "local")
    paths = get_run_paths(roots=roots, name=name, run_id=run_id)
    run_ctx = _build_run_ctx(
        kind=kind,
        run_json=run_json,
        name=name,
        run_key=run_key,
        run_id=run_id,
        paths=paths,
    )

    resolved_final = active_resolved
    repair_analysis = {
        "changed_steps": [],
        "disallowed_steps": [],
        "unchanged_incomplete": [],
        "frozen_succeeded": [],
    }
    if repair_mode:
        assert repair_spec is not None
        resolved_final, repair_kind = _resolve_runtime_spec(
            spec_path=repair_spec,
            roots=roots,
            run_ctx=run_ctx,
            expected_name=name,
            expected_kind=kind,
        )
        kind = repair_kind

    ordered_steps = validate_and_toposort_steps(list(active_resolved.get("steps") or []))
    step_records = ensure_step_records(
        ordered_steps=ordered_steps,
        paths=paths,
        run_json=copy.deepcopy(run_json),
        write_run_json_fn=lambda _paths, _run_json: None,
    )
    first_incomplete = next(
        (str(rec.get("step_id")) for rec in step_records if rec.get("state") != "succeeded"),
        None,
    )
    if first_incomplete is None and not force:
        raise RunResumeError(
            "All steps are already marked succeeded; pass --force to rerun according to step policies"
        )

    if repair_mode:
        repair_analysis = _analyze_repair_drift(
            stored_resolved=active_resolved,
            candidate_resolved=resolved_final,
            step_records=step_records,
            allowed_states=allowed_drift_states,
        )
        _raise_disallowed_drift(
            disallowed_steps=repair_analysis["disallowed_steps"],
            repair_mode=True,
        )
    else:
        repair_analysis["frozen_succeeded"] = [
            str(rec.get("step_id"))
            for rec in step_records
            if normalize_resume_step_state(rec.get("state")) == "succeeded"
        ]
        repair_analysis["unchanged_incomplete"] = [
            {
                "id": str(rec.get("step_id")),
                "state": normalize_resume_step_state(rec.get("state")),
            }
            for rec in step_records
            if normalize_resume_step_state(rec.get("state")) != "succeeded"
        ]

    env_vars = prepare_effective_env(resolved_final, roots=roots, kind=kind)
    if kind == "docker":
        prepare_effective_docker(resolved_final, roots=roots, for_hashing=False)
    prepared = PreparedRun(
        paths=paths, run_ctx=run_ctx, resolved_final=resolved_final, env_vars=env_vars
    )

    _print_resume_summary(
        mode="repair" if repair_mode else "strict",
        allowed_states=allowed_drift_states,
        frozen_succeeded=repair_analysis["frozen_succeeded"],
        changed_steps=repair_analysis["changed_steps"],
        unchanged_incomplete=repair_analysis["unchanged_incomplete"],
        dry_run=dry_run,
    )
    if dry_run:
        return {
            "name": name,
            "run_id": run_id,
            "run_key": run_key,
            "run_dir": str(paths.run_dir),
            "artifacts_dir": str(paths.artifacts_dir),
        }

    if repair_mode:
        repair_events_raw = run_json.get("repair_events")
        repair_events = (
            [event for event in repair_events_raw if isinstance(event, dict)]
            if isinstance(repair_events_raw, list)
            else []
        )
        repair_event_index = len(repair_events) + 1
        assert repair_spec is not None
        spec_copy, resolved_copy = _write_repair_spec_artifacts(
            run_dir=run_dir,
            repair_spec=repair_spec,
            resolved_final=resolved_final,
            repair_event_index=repair_event_index,
        )
        repair_event = {
            "index": repair_event_index,
            "timestamp": utc_now_iso(),
            "source_spec_path": str(repair_spec.resolve()),
            "stored_spec_path": _relative_to_run_dir(run_dir, spec_copy),
            "stored_resolved_spec_path": _relative_to_run_dir(run_dir, resolved_copy),
            "allowed_states": sorted(allowed_drift_states),
            "changed_steps": [
                {
                    "id": entry["id"],
                    "state": entry["state"],
                    "old_hash": entry["old_hash"],
                    "new_hash": entry["new_hash"],
                    "diff": entry["diff"],
                    "diff_summary": entry["diff_summary"],
                }
                for entry in repair_analysis["changed_steps"]
            ],
        }
        repair_events.append(repair_event)
        run_json["repair_events"] = repair_events
        _set_active_spec_metadata(
            run_json=run_json,
            run_dir=run_dir,
            spec_path=spec_copy,
            resolved_path=resolved_copy,
            source="repair",
            repair_event_index=repair_event_index,
        )

    pid = os.getpid()
    run_json["pid"] = pid
    run_json["proc_start_ticks"] = proc_start_ticks_linux(pid)
    run_json["state"] = "running"
    run_json.pop("error", None)
    run_json["resume_mode"] = "repair" if repair_mode else "strict"

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
        "mode": run_json["resume_mode"],
        "active_spec_source": str((run_json.get("active_spec") or {}).get("source") or "original"),
    }
    if repair_mode:
        resume_entry["repair_event_index"] = int(
            (run_json.get("active_spec") or {}).get("repair_event_index") or 0
        )
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
