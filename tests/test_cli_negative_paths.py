from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

from exp_harness.cli import app
from tests.helpers import write_spec


def test_cli_set_requires_equals(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    spec_fp = write_spec(
        project_root,
        {
            "name": "cli2",
            "env": {"kind": "local"},
            "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('hi')"]}],
        },
    )
    runner = CliRunner()
    res = runner.invoke(app, ["run", str(spec_fp), "--set", "params.x"])
    assert res.exit_code != 0
    assert "Expected KEY=VALUE" in (res.stdout + res.stderr)


def test_cli_set_empty_key(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    spec_fp = write_spec(
        project_root,
        {"name": "cli3", "env": {"kind": "local"}, "steps": [{"id": "a", "cmd": ["echo", "hi"]}]},
    )
    runner = CliRunner()
    res = runner.invoke(app, ["run", str(spec_fp), "--set", "=1"])
    assert res.exit_code != 0
    assert "Empty key" in (res.stdout + res.stderr)
