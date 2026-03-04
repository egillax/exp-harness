from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.interp import InterpError
from exp_harness.resolve import apply_computed_defaults, load_and_validate, resolve_final


def _write_yaml(p: Path, data: dict) -> None:
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_extends_deep_merge_and_list_replace(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"

    _write_yaml(
        base,
        {
            "name": "base",
            "params": {"a": 1, "b": 2, "nested": {"x": "keep", "y": "base"}},
            "steps": [{"id": "base_step", "cmd": [sys.executable, "-c", "print('base')"]}],
        },
    )
    _write_yaml(
        overlay,
        {
            "extends": "base.yaml",
            "name": "final",
            "params": {"b": 9, "nested": {"y": "overlay", "z": "new"}},
            "steps": [{"id": "overlay_step", "cmd": [sys.executable, "-c", "print('overlay')"]}],
        },
    )

    raw = load_and_validate(spec_path=overlay, set_overrides=[], set_string_overrides=[])
    assert raw.get("extends") is None
    assert raw["name"] == "final"
    assert raw["params"]["a"] == 1
    assert raw["params"]["b"] == 9
    assert raw["params"]["nested"] == {"x": "keep", "y": "overlay", "z": "new"}
    assert [s["id"] for s in raw["steps"]] == ["overlay_step"]


def test_extends_emits_compatibility_warning(tmp_path: Path, caplog) -> None:
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"
    _write_yaml(
        base,
        {
            "name": "base",
            "steps": [{"id": "base_step", "cmd": [sys.executable, "-c", "print('base')"]}],
        },
    )
    _write_yaml(
        overlay,
        {
            "extends": "base.yaml",
            "name": "final",
            "steps": [{"id": "overlay_step", "cmd": [sys.executable, "-c", "print('overlay')"]}],
        },
    )

    with caplog.at_level(logging.WARNING, logger="exp_harness.resolve"):
        _ = load_and_validate(spec_path=overlay, set_overrides=[], set_string_overrides=[])
    assert "legacy `extends` layering is compatibility mode" in caplog.text


def test_extends_list_of_bases(tmp_path: Path) -> None:
    base1 = tmp_path / "base1.yaml"
    base2 = tmp_path / "base2.yaml"
    overlay = tmp_path / "overlay.yaml"

    _write_yaml(
        base1,
        {
            "name": "base1",
            "params": {"a": 1},
            "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('a')"]}],
        },
    )
    _write_yaml(
        base2,
        {
            "name": "base2",
            "params": {"b": 2},
            "steps": [{"id": "b", "cmd": [sys.executable, "-c", "print('b')"]}],
        },
    )
    _write_yaml(
        overlay,
        {
            "extends": ["base1.yaml", "base2.yaml"],
            "name": "final",
            "params": {"c": 3},
            # no steps override => should come from last base in the list
        },
    )

    raw = load_and_validate(spec_path=overlay, set_overrides=[], set_string_overrides=[])
    assert raw["params"] == {"a": 1, "b": 2, "c": 3}
    assert [s["id"] for s in raw["steps"]] == ["b"]


def test_extends_cycle_detected(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"

    _write_yaml(
        a,
        {
            "extends": "b.yaml",
            "name": "a",
            "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('a')"]}],
        },
    )
    _write_yaml(
        b,
        {
            "extends": "a.yaml",
            "name": "b",
            "steps": [{"id": "b", "cmd": [sys.executable, "-c", "print('b')"]}],
        },
    )

    with pytest.raises(ValueError, match="extends cycle detected"):
        load_and_validate(spec_path=a, set_overrides=[], set_string_overrides=[])


def test_vars_shorthand_precedence_over_params(tmp_path: Path) -> None:
    raw = {
        "name": "v",
        "vars": {"batch_size": 128},
        "params": {"batch_size": 64},
        "env": {"kind": "local", "workdir": "x=${batch_size}"},
        "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('batch=${batch_size}')"]}],
    }
    with_defaults = apply_computed_defaults(raw, project_root=tmp_path, kind="local")
    resolved = resolve_final(
        with_defaults,
        project_root=tmp_path,
        run_ctx={
            "id": "rk",
            "runs": str(tmp_path / "runs"),
            "artifacts": str(tmp_path / "artifacts"),
        },
    )
    assert resolved["env"]["workdir"] == "x=128"
    assert resolved["steps"][0]["cmd"][-1] == "print('batch=128')"


def test_vars_cannot_shadow_reserved_keys(tmp_path: Path) -> None:
    raw = {
        "name": "v2",
        "vars": {"params": "nope"},
        "params": {},
        "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('hi')"]}],
    }
    with_defaults = apply_computed_defaults(raw, project_root=tmp_path, kind="local")
    with pytest.raises(InterpError, match="reserved key"):
        resolve_final(
            with_defaults,
            project_root=tmp_path,
            run_ctx={
                "id": "rk",
                "runs": str(tmp_path / "runs"),
                "artifacts": str(tmp_path / "artifacts"),
            },
        )
