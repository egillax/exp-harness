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
from exp_harness.utils import discover_project_root, discover_project_root_from_dir


class OverrideParseError(ValueError):
    """Raised when a --set/--set-str assignment is not in KEY=VALUE form."""


class ComposeConfigError(ValueError):
    """Raised when Hydra composition does not produce a valid experiment mapping."""


class RunResult(TypedDict):
    name: str
    run_id: str
    run_key: str
    run_dir: str
    artifacts_dir: str


class SweepMemberResult(TypedDict):
    index: int
    overrides: list[str]
    status: Literal["succeeded", "failed"]
    result: RunResult | None
    error: str | None


class SweepResult(TypedDict):
    total: int
    succeeded: int
    failed: int
    runs: list[SweepMemberResult]


def parse_set_overrides(values: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise OverrideParseError("Expected KEY=VALUE")
        key, rhs = value.split("=", 1)
        key = key.strip()
        if not key:
            raise OverrideParseError("Empty key")
        parsed.append((key, rhs))
    return parsed


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


def run_hydra_experiment(
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
    root = project_root or discover_project_root_from_dir(Path.cwd())
    cfg = compose_experiment_config(
        overrides=overrides,
        config_name=config_name,
        config_module=config_module,
    )
    with TemporaryDirectory(prefix="exp-harness-hydra-run-") as tmp_dir:
        spec_path = _write_composed_spec(
            cfg=cfg,
            output_dir=Path(tmp_dir),
            suffix="hydra_run",
        )
        return run_experiment(
            spec_path=spec_path,
            set_overrides=[],
            set_string_overrides=[],
            project_root=root,
            runs_root=runs_root,
            artifacts_root=artifacts_root,
            salt=salt,
            run_label=run_label,
            enforce_clean=enforce_clean,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
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
) -> SweepResult:
    root = project_root or discover_project_root_from_dir(Path.cwd())
    members = expand_hydra_sweep_overrides(overrides)
    runs: list[SweepMemberResult] = []

    with TemporaryDirectory(prefix="exp-harness-hydra-sweep-") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, member_overrides in enumerate(members, start=1):
            try:
                cfg = compose_experiment_config(
                    overrides=member_overrides,
                    config_name=config_name,
                    config_module=config_module,
                )
                spec_path = _write_composed_spec(
                    cfg=cfg,
                    output_dir=tmp_root,
                    suffix=f"sweep_{idx:03d}",
                )

                run_res = run_experiment(
                    spec_path=spec_path,
                    set_overrides=[],
                    set_string_overrides=[],
                    project_root=root,
                    runs_root=runs_root,
                    artifacts_root=artifacts_root,
                    salt=salt,
                    run_label=f"sweep-{idx:03d}",
                    enforce_clean=enforce_clean,
                    follow_steps=follow_steps,
                    stderr_tail_lines=stderr_tail_lines,
                )
                runs.append(
                    {
                        "index": idx,
                        "overrides": list(member_overrides),
                        "status": "succeeded",
                        "result": run_res,
                        "error": None,
                    }
                )
            except Exception as exc:
                runs.append(
                    {
                        "index": idx,
                        "overrides": list(member_overrides),
                        "status": "failed",
                        "result": None,
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    break

    succeeded = sum(1 for item in runs if item["status"] == "succeeded")
    failed = len(runs) - succeeded
    return {
        "total": len(runs),
        "succeeded": succeeded,
        "failed": failed,
        "runs": runs,
    }


def run_experiment(
    *,
    spec_path: Path,
    set_overrides: Sequence[tuple[str, str]] | None = None,
    set_string_overrides: Sequence[tuple[str, str]] | None = None,
    project_root: Path | None = None,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
    salt: str | None = None,
    run_label: str | None = None,
    enforce_clean: bool = False,
    follow_steps: bool = True,
    stderr_tail_lines: int = 120,
) -> RunResult:
    from exp_harness.runner import run_experiment as _run_experiment

    resolved_project_root = project_root or discover_project_root(spec_path)
    roots = resolve_roots(
        project_root=resolved_project_root,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
    )
    return cast(
        RunResult,
        _run_experiment(
            spec_path=spec_path,
            roots=roots,
            set_overrides=list(set_overrides or []),
            set_string_overrides=list(set_string_overrides or []),
            salt=salt,
            run_label=run_label,
            enforce_clean=enforce_clean,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
        ),
    )
