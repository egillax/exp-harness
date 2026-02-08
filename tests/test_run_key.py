from __future__ import annotations

from exp_harness.run_key import compute_run_key


def test_run_key_is_stable_for_same_material() -> None:
    m = {"b": 2, "a": 1}
    assert compute_run_key(m) == compute_run_key({"a": 1, "b": 2})
