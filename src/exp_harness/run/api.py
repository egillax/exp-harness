from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypedDict, cast

import yaml
from hydra import compose, initialize_config_module
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from exp_harness.config import resolve_roots
from exp_harness.spec import ExperimentSpec
from exp_harness.utils import (
    discover_project_root,
    discover_project_root_from_dir,
    utc_now_iso,
    write_json,
)


class ComposeConfigError(ValueError):
    """Raised when Hydra composition does not produce a valid experiment mapping."""


class RunResult(TypedDict):
    name: str
    run_id: str
    run_key: str
    run_dir: str
    artifacts_dir: str


class SweepAttemptResult(TypedDict):
    attempt: int
    status: Literal["succeeded", "failed"]
    error: str | None
    run_id: str | None
    run_key: str | None


class SweepMemberResult(TypedDict):
    index: int
    overrides: list[str]
    status: Literal["succeeded", "failed"]
    result: RunResult | None
    error: str | None
    attempts: list[SweepAttemptResult]


class SweepResult(TypedDict):
    total: int
    succeeded: int
    failed: int
    runs: list[SweepMemberResult]
    summary_path: str


def expand_hydra_sweep_overrides(overrides: Sequence[str] | None = None) -> list[list[str]]:
    parser = OverridesParser.create()
    parsed = parser.parse_overrides(list(overrides or []))

    dimensions: list[list[str]] = []
    for override in parsed:
        if override.is_sweep_override():
            key = override.get_key_element()
            values = [
                f"{key}={value}" for value in override.sweep_string_iterator() if value is not None
            ]
            dimensions.append(values)
        else:
            line = override.input_line
            if line is None:
                raise ComposeConfigError("Hydra override is missing input text")
            dimensions.append([line])

    if not dimensions:
        return [[]]
    return [list(combo) for combo in product(*dimensions)]


def compose_experiment_config(
    *,
    overrides: Sequence[str] | None = None,
    config_name: str = "config",
    config_module: str = "exp_harness.conf",
) -> dict[str, Any]:
    with initialize_config_module(version_base=None, config_module=config_module):
        cfg = compose(
            config_name=config_name,
            overrides=list(overrides or []),
            return_hydra_config=False,
        )
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ComposeConfigError("Hydra composition must produce a top-level mapping")
    ExperimentSpec.model_validate(data)
    return cast(dict[str, Any], data)


def _write_composed_spec(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    suffix: str,
) -> Path:
    name = str(cfg.get("name") or "experiment")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "experiment"
    spec_path = output_dir / f"{safe_name}__{suffix}.yaml"
    spec_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return spec_path


def _run_spec_file(
    *,
    spec_path: Path,
    project_root: Path,
    runs_root: Path | None,
    artifacts_root: Path | None,
    salt: str | None,
    run_label: str | None,
    enforce_clean: bool,
    follow_steps: bool,
    stderr_tail_lines: int,
) -> RunResult:
    from exp_harness.runner import run_experiment as _run_experiment

    roots = resolve_roots(
        project_root=project_root,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
    )
    return cast(
        RunResult,
        _run_experiment(
            spec_path=spec_path,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt=salt,
            run_label=run_label,
            enforce_clean=enforce_clean,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
        ),
    )


def _run_composed_experiment(
    *,
    cfg: dict[str, Any],
    project_root: Path,
    runs_root: Path | None,
    artifacts_root: Path | None,
    salt: str | None,
    run_label: str | None,
    enforce_clean: bool,
    follow_steps: bool,
    stderr_tail_lines: int,
    suffix: str,
) -> RunResult:
    with TemporaryDirectory(prefix="exp-harness-hydra-run-") as tmp_dir:
        spec_path = _write_composed_spec(
            cfg=cfg,
            output_dir=Path(tmp_dir),
            suffix=suffix,
        )
        return _run_spec_file(
            spec_path=spec_path,
            project_root=project_root,
            runs_root=runs_root,
            artifacts_root=artifacts_root,
            salt=salt,
            run_label=run_label,
            enforce_clean=enforce_clean,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
        )


def run_experiment(
    *,
    overrides: Sequence[str] | None = None,
    config_name: str = "config",
    config_module: str = "exp_harness.conf",
    project_root: Path | None = None,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
    salt: str | None = None,
    run_label: str | None = None,
    enforce_clean: bool = False,
    follow_steps: bool = True,
    stderr_tail_lines: int = 120,
) -> RunResult:
    """
    Run a single experiment composed from Hydra config groups and overrides.
    """
    root = project_root or discover_project_root_from_dir(Path.cwd())
    cfg = compose_experiment_config(
        overrides=overrides,
        config_name=config_name,
        config_module=config_module,
    )
    return _run_composed_experiment(
        cfg=cfg,
        project_root=root,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
        salt=salt,
        run_label=run_label,
        enforce_clean=enforce_clean,
        follow_steps=follow_steps,
        stderr_tail_lines=stderr_tail_lines,
        suffix="hydra_run",
    )


def run_spec_experiment(
    *,
    spec_path: Path,
    project_root: Path | None = None,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
    salt: str | None = None,
    run_label: str | None = None,
    enforce_clean: bool = False,
    follow_steps: bool = True,
    stderr_tail_lines: int = 120,
) -> RunResult:
    """
    Run a YAML spec file directly (compatibility path for spec-based wrappers).
    """
    root = project_root or discover_project_root(spec_path)
    return _run_spec_file(
        spec_path=spec_path,
        project_root=root,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
        salt=salt,
        run_label=run_label,
        enforce_clean=enforce_clean,
        follow_steps=follow_steps,
        stderr_tail_lines=stderr_tail_lines,
    )


def resume_experiment(
    *,
    name: str,
    run_key: str,
    project_root: Path | None = None,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
    enforce_clean: bool = False,
    allow_spec_drift: bool = False,
    force: bool = False,
    follow_steps: bool = False,
    stderr_tail_lines: int = 120,
) -> RunResult:
    """
    Resume an existing run from the first step not yet marked succeeded.
    """
    from exp_harness.runner import resume_experiment as _resume_experiment

    root = project_root or discover_project_root_from_dir(Path.cwd())
    roots = resolve_roots(
        project_root=root,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
    )
    return cast(
        RunResult,
        _resume_experiment(
            roots=roots,
            name=name,
            run_key=run_key,
            enforce_clean=enforce_clean,
            allow_spec_drift=allow_spec_drift,
            force=force,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
        ),
    )


def run_hydra_sweep(
    *,
    overrides: Sequence[str] | None = None,
    config_name: str = "config",
    config_module: str = "exp_harness.conf",
    project_root: Path | None = None,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
    salt: str | None = None,
    enforce_clean: bool = False,
    follow_steps: bool = False,
    stderr_tail_lines: int = 120,
    continue_on_error: bool = True,
    retry_failed: int = 0,
) -> SweepResult:
    """
    Compose and execute a sweep using Hydra override syntax and exp-harness execution.

    Sweep members are expanded from Hydra overrides, composed one-by-one, and run through
    the canonical harness runner. This preserves the harness run/provenance model and does
    not invoke Hydra launcher/sweeper plugins.
    """
    root = project_root or discover_project_root_from_dir(Path.cwd())
    resolved_roots = resolve_roots(
        project_root=root,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
    )
    members = expand_hydra_sweep_overrides(overrides)
    runs: list[SweepMemberResult] = []

    for idx, member_overrides in enumerate(members, start=1):
        member_attempts: list[SweepAttemptResult] = []
        final_result: RunResult | None = None
        final_error: str | None = None
        final_status: Literal["succeeded", "failed"] = "failed"

        max_attempts = max(1, int(retry_failed) + 1)
        for attempt in range(1, max_attempts + 1):
            try:
                cfg = compose_experiment_config(
                    overrides=member_overrides,
                    config_name=config_name,
                    config_module=config_module,
                )
                run_res = _run_composed_experiment(
                    cfg=cfg,
                    project_root=root,
                    runs_root=runs_root,
                    artifacts_root=artifacts_root,
                    salt=salt,
                    run_label=f"sweep-{idx:03d}",
                    enforce_clean=enforce_clean,
                    follow_steps=follow_steps,
                    stderr_tail_lines=stderr_tail_lines,
                    suffix=f"sweep_{idx:03d}",
                )
                member_attempts.append(
                    {
                        "attempt": attempt,
                        "status": "succeeded",
                        "error": None,
                        "run_id": run_res["run_id"],
                        "run_key": run_res["run_key"],
                    }
                )
                final_result = run_res
                final_status = "succeeded"
                final_error = None
                break
            except Exception as exc:
                final_error = str(exc)
                member_attempts.append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "error": final_error,
                        "run_id": None,
                        "run_key": None,
                    }
                )
                if attempt >= max_attempts:
                    break

        runs.append(
            {
                "index": idx,
                "overrides": list(member_overrides),
                "status": final_status,
                "result": final_result,
                "error": final_error,
                "attempts": member_attempts,
            }
        )
        if final_status == "failed" and not continue_on_error:
            break

    succeeded = sum(1 for item in runs if item["status"] == "succeeded")
    failed = len(runs) - succeeded
    summary_dir = resolved_roots.runs_root / "_sweeps"
    summary_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    summary_path = summary_dir / f"sweep_{stamp}.json"
    summary_payload = {
        "total": len(runs),
        "succeeded": succeeded,
        "failed": failed,
        "runs": runs,
    }
    write_json(summary_path, summary_payload)

    return {
        "total": len(runs),
        "succeeded": succeeded,
        "failed": failed,
        "runs": runs,
        "summary_path": str(summary_path),
    }
