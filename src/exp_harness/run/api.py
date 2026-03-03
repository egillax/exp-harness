from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from exp_harness.config import resolve_roots
from exp_harness.runner import run_experiment as _run_experiment
from exp_harness.spec import ExperimentSpec
from exp_harness.utils import discover_project_root


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


def run_experiment(
    *,
    spec_path: Path,
    set_overrides: Sequence[tuple[str, str]] | None = None,
    set_string_overrides: Sequence[tuple[str, str]] | None = None,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
    salt: str | None = None,
    run_label: str | None = None,
    enforce_clean: bool = False,
    follow_steps: bool = True,
    stderr_tail_lines: int = 120,
) -> RunResult:
    project_root = discover_project_root(spec_path)
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
            set_overrides=list(set_overrides or []),
            set_string_overrides=list(set_string_overrides or []),
            salt=salt,
            run_label=run_label,
            enforce_clean=enforce_clean,
            follow_steps=follow_steps,
            stderr_tail_lines=stderr_tail_lines,
        ),
    )
