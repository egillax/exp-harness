from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DockerMount(_Model):
    host: str
    container: str


class DockerConfig(_Model):
    image: str
    mounts: list[DockerMount] | None = None  # tri-state: None => auto, [] => no mounts
    shm_size: str | None = None
    ipc: Literal["host", "private"] | None = None
    network: Literal["none", "bridge"] | None = None
    runtime: str | None = None
    gpu_mode: Literal["auto", "docker_gpus_device", "nvidia_visible_devices", "none"] = "auto"
    allow_unverified_image: bool = False


class EnvSpec(_Model):
    kind: Literal["local", "docker"] = "local"
    workdir: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    offline: bool = False
    docker: DockerConfig | None = None


class ResourcesSpec(_Model):
    gpus: int | list[int] = 0
    cpu: int | None = None


class FingerprintSpec(_Model):
    kind: Literal["none", "sha256_files", "sha256_tree"] = "none"
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class InputSpec(_Model):
    path: str
    fingerprint: FingerprintSpec = Field(default_factory=FingerprintSpec)


class StepOutputsSpec(_Model):
    artifacts_dir: str | None = None


class StepSpec(_Model):
    id: str
    needs: list[str] = Field(default_factory=list)
    cmd: list[str]
    outputs: StepOutputsSpec = Field(default_factory=StepOutputsSpec)
    timeout_seconds: int | None = None
    resume_policy: Literal["rerun", "skip_if_succeeded", "skip_if_marker"] = "skip_if_succeeded"
    success_markers: list[str] = Field(default_factory=list)


class ExperimentSpec(_Model):
    name: str
    run_label: str | None = None
    env: EnvSpec = Field(default_factory=EnvSpec)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    vars: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepSpec]
