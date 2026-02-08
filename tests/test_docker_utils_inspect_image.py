from __future__ import annotations

import json
from pathlib import Path

from exp_harness.docker_utils import inspect_image


def test_inspect_image_parses_selected_fields(monkeypatch, tmp_path: Path) -> None:
    payload = [
        {
            "Id": "sha256:abc",
            "RepoDigests": ["repo@sha256:def"],
            "RepoTags": ["repo:tag"],
        }
    ]

    def fake_shell_out(argv, cwd=None):
        assert argv[:3] == ["docker", "image", "inspect"]
        return 0, json.dumps(payload), ""

    monkeypatch.setattr("exp_harness.docker_utils.shell_out", fake_shell_out)

    meta = inspect_image(image="repo:tag", cwd=tmp_path)
    assert meta is not None
    assert meta["image_id"] == "sha256:abc"
    assert meta["repo_digests"] == ["repo@sha256:def"]
    assert meta["repo_tags"] == ["repo:tag"]
