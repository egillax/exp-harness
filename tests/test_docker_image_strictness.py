from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.config import Roots
from exp_harness.runner import run_experiment


class _FakeDockerExecutor:
    def prepare_run(self, ctx) -> None:
        return

    def run_step(
        self, ctx, *, step_index, step_id, cmd, step_dir, timeout_seconds, step_artifacts_dir
    ):
        from exp_harness.executors.base import StepResult
        from exp_harness.utils import utc_now_iso, write_json

        started = utc_now_iso()
        finished = utc_now_iso()
        res = StepResult(
            step_id=step_id,
            rc=0,
            started_at_utc=started,
            finished_at_utc=finished,
            duration_seconds=0.0,
            allocated_gpus_host=list(ctx.allocated_gpus_host),
            allocated_gpus_visible=list(ctx.allocated_gpus_host),
            extra={},
        )
        write_json(step_dir / "exec.json", res.__dict__)
        return res

    def finalize_run(self, ctx) -> None:
        return


def test_docker_image_inspect_fails_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("exp_harness.runner.inspect_image", lambda image, cwd: None)

    project_root = tmp_path / "proj"
    project_root.mkdir()
    spec_fp = project_root / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "d",
                "env": {"kind": "docker", "docker": {"image": "img:tag"}},
                "steps": [{"id": "a", "cmd": ["echo", "hi"]}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    roots = Roots(
        project_root=project_root, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "art"
    )

    with pytest.raises(RuntimeError, match="allow_unverified_image"):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
        )


def test_docker_allow_unverified_image_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("exp_harness.runner.inspect_image", lambda image, cwd: None)
    monkeypatch.setattr("exp_harness.runner.DockerExecutor", _FakeDockerExecutor)

    project_root = tmp_path / "proj"
    project_root.mkdir()
    spec_fp = project_root / "spec.yaml"
    spec_fp.write_text(
        yaml.safe_dump(
            {
                "name": "d2",
                "env": {
                    "kind": "docker",
                    "docker": {"image": "img:tag", "allow_unverified_image": True},
                },
                "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('x')"]}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    roots = Roots(
        project_root=project_root, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "art"
    )

    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
    )
    assert res["run_key"]
