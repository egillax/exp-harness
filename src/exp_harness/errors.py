from __future__ import annotations


class ExpHarnessError(RuntimeError):
    """Base class for domain-specific harness failures."""


class GitDirtyWorktreeError(ExpHarnessError):
    """Raised when enforce-clean is enabled and the git worktree is dirty."""


class RunConflictError(ExpHarnessError):
    """Raised when a run key or run directory already exists."""


class DockerConfigurationError(ExpHarnessError):
    """Raised when docker configuration is invalid or incomplete."""


class DockerImageInspectionError(ExpHarnessError):
    """Raised when docker image inspection fails and strict verification is enabled."""


class GpuRequestError(ExpHarnessError):
    """Raised when requested GPU ids are invalid."""


class GpuAllocationError(ExpHarnessError):
    """Raised when requested GPUs cannot be allocated."""


class StepExecutionError(ExpHarnessError):
    """Raised when a step exits with a non-zero return code."""

    def __init__(self, *, step_id: str, rc: int) -> None:
        self.step_id = step_id
        self.rc = int(rc)
        super().__init__(f"Step failed: {step_id} (rc={rc})")


class RunResumeError(ExpHarnessError):
    """Raised when a run cannot be resumed safely under current constraints."""
