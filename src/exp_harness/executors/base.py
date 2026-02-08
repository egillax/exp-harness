from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class StepResult:
    step_id: str
    rc: int
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    allocated_gpus_host: list[int]
    allocated_gpus_visible: list[int]
    extra: dict[str, Any]


@dataclass(frozen=True)
class RunContext:
    name: str
    run_key: str
    project_root: Path
    run_dir: Path
    artifacts_dir: Path
    workdir: str
    env: dict[str, str]
    offline: bool
    kind: str
    docker: dict[str, Any] | None
    allocated_gpus_host: list[int]


class Executor(Protocol):
    def prepare_run(self, ctx: RunContext) -> None: ...
    def run_step(
        self,
        ctx: RunContext,
        *,
        step_index: int,
        step_id: str,
        cmd: list[str],
        step_dir: Path,
        timeout_seconds: int | None,
        step_artifacts_dir: str | None,
    ) -> StepResult: ...
    def finalize_run(self, ctx: RunContext) -> None: ...
