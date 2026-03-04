from __future__ import annotations

import copy
from pathlib import Path

from exp_harness.config import Roots
from exp_harness.docker_utils import inspect_image
from exp_harness.executors.docker import DockerExecutor
from exp_harness.executors.local import LocalExecutor
from exp_harness.git_info import collect_git_info
from exp_harness.resolve import apply_computed_defaults, load_and_validate
from exp_harness.run.identity import build_run_identity
from exp_harness.run.phases import (
    ResourceAllocation,
    phase_allocate_resources,
    phase_execute_run,
    phase_initialize_metadata,
    phase_prepare_run,
    phase_release_resources,
)
from exp_harness.run.state import mark_failed_with_traceback, mark_interrupted, mark_succeeded
from exp_harness.store import write_run_json


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
    raw_base = load_and_validate(
        spec_path=spec_path,
        set_overrides=set_overrides,
        set_string_overrides=set_string_overrides,
    )
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
        raise RuntimeError("Git working tree is dirty (--enforce-clean is set)")

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
