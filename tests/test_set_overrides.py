from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from exp_harness.resolve import load_and_validate


def _write(fp: Path, data: dict) -> None:
    fp.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_set_parses_yaml_scalars(tmp_path: Path) -> None:
    spec_fp = tmp_path / "s.yaml"
    _write(spec_fp, {"name": "x", "steps": [{"id": "a", "cmd": ["echo", "hi"]}]})

    raw = load_and_validate(
        spec_path=spec_fp, set_overrides=[("params.x", "3")], set_string_overrides=[]
    )
    assert raw["params"]["x"] == 3


def test_set_parses_yaml_lists(tmp_path: Path) -> None:
    spec_fp = tmp_path / "s.yaml"
    _write(spec_fp, {"name": "x", "steps": [{"id": "a", "cmd": ["echo", "hi"]}]})

    raw = load_and_validate(
        spec_path=spec_fp, set_overrides=[("params.x", "[1,2]")], set_string_overrides=[]
    )
    assert raw["params"]["x"] == [1, 2]


def test_set_str_preserves_string(tmp_path: Path) -> None:
    spec_fp = tmp_path / "s.yaml"
    _write(spec_fp, {"name": "x", "steps": [{"id": "a", "cmd": ["echo", "hi"]}]})

    raw = load_and_validate(
        spec_path=spec_fp, set_overrides=[], set_string_overrides=[("params.x", "001")]
    )
    assert raw["params"]["x"] == "001"


def test_set_nested_paths(tmp_path: Path) -> None:
    spec_fp = tmp_path / "s.yaml"
    _write(spec_fp, {"name": "x", "steps": [{"id": "a", "cmd": ["echo", "hi"]}]})

    raw = load_and_validate(
        spec_path=spec_fp, set_overrides=[("params.a.b", "1")], set_string_overrides=[]
    )
    assert raw["params"]["a"]["b"] == 1


def test_set_type_conflict_errors(tmp_path: Path) -> None:
    spec_fp = tmp_path / "s.yaml"
    _write(spec_fp, {"name": "x", "steps": [{"id": "a", "cmd": ["echo", "hi"]}]})

    with pytest.raises(ValueError, match="not a dict"):
        load_and_validate(
            spec_path=spec_fp,
            set_overrides=[("params.a", "3"), ("params.a.b", "1")],
            set_string_overrides=[],
        )
