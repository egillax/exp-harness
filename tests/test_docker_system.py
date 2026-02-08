from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp_harness.runner import run_experiment
from tests.helpers import tmp_roots, write_spec


@pytest.mark.docker
def test_docker_minimal_step_writes_artifact(tmp_path: Path) -> None:
    import subprocess

    roots = tmp_roots(tmp_path)
    # Ensure image is available; if pull fails, skip.
    pull = subprocess.run(
        ["docker", "pull", "python:3.12-slim"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if pull.returncode != 0:
        pytest.skip("docker image python:3.12-slim not available and pull failed")

    spec_fp = write_spec(
        roots.project_root,
        {
            "name": "dock",
            "env": {"kind": "docker", "docker": {"image": "python:3.12-slim"}},
            "steps": [
                {
                    "id": "a",
                    "cmd": [
                        "python",
                        "-c",
                        "from pathlib import Path; import os; p=Path(os.environ['EXP_HARNESS_STEP_ARTIFACTS_DIR']); p.mkdir(parents=True, exist_ok=True); (p/'x.txt').write_text('ok')",
                    ],
                }
            ],
        },
    )

    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
    )
    assert (Path(res["artifacts_dir"]) / "a" / "x.txt").exists()


@pytest.mark.docker
def test_docker_offline_defaults_network_none_and_records_provenance(tmp_path: Path) -> None:
    import subprocess

    roots = tmp_roots(tmp_path)
    pull = subprocess.run(
        ["docker", "pull", "python:3.12-slim"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if pull.returncode != 0:
        pytest.skip("docker image python:3.12-slim not available and pull failed")

    spec_fp = write_spec(
        roots.project_root,
        {
            "name": "dockoff",
            "env": {"kind": "docker", "offline": True, "docker": {"image": "python:3.12-slim"}},
            "steps": [{"id": "a", "cmd": ["python", "-c", "print('hi')"]}],
        },
    )
    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
    )
    run_dir = Path(res["run_dir"])
    resolved = json.loads((run_dir / "resolved_spec.json").read_text(encoding="utf-8"))
    assert resolved["env"]["docker"]["network"] == "none"
    docker_json = json.loads((run_dir / "provenance" / "docker.json").read_text(encoding="utf-8"))
    assert docker_json["offline_network_enforced"] is True
