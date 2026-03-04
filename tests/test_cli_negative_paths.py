from __future__ import annotations

import re

from typer.testing import CliRunner

from exp_harness.cli import app


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_cli_run_rejects_invalid_hydra_override_syntax() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["run", "params.x"])
    assert res.exit_code != 0
    assert res.exception is not None
    assert "Error parsing override" in str(res.exception)


def test_cli_run_rejects_removed_set_flag() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["run", "name=demo", "--set", "params.x=1"])
    assert res.exit_code != 0
    out = _strip_ansi(res.stdout + res.stderr)
    assert "No such option" in out
    assert "--set" in out
