from __future__ import annotations

import pytest

from exp_harness.run.step_graph import validate_and_toposort_steps


def test_duplicate_step_ids_error() -> None:
    with pytest.raises(ValueError, match="Duplicate step id"):
        validate_and_toposort_steps(
            [
                {"id": "a", "cmd": ["echo", "a"]},
                {"id": "a", "cmd": ["echo", "b"]},
            ]
        )


def test_missing_dependency_error() -> None:
    with pytest.raises(ValueError, match="needs missing dependency"):
        validate_and_toposort_steps(
            [
                {"id": "a", "needs": ["missing"], "cmd": ["echo", "a"]},
            ]
        )


def test_cycle_error() -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        validate_and_toposort_steps(
            [
                {"id": "a", "needs": ["b"], "cmd": ["echo", "a"]},
                {"id": "b", "needs": ["a"], "cmd": ["echo", "b"]},
            ]
        )


def test_ordering_stable_for_small_dag() -> None:
    ordered = validate_and_toposort_steps(
        [
            {"id": "d", "needs": ["b", "c"], "cmd": ["echo", "d"]},
            {"id": "b", "needs": ["a"], "cmd": ["echo", "b"]},
            {"id": "c", "needs": ["a"], "cmd": ["echo", "c"]},
            {"id": "a", "cmd": ["echo", "a"]},
        ]
    )
    assert [s["id"] for s in ordered] == ["a", "b", "c", "d"]
