from __future__ import annotations

import pytest

from exp_harness.interp import InterpError, resolve_obj


def test_full_token_can_be_non_string() -> None:
    ctx = {"params": {"a": 3}, "inputs": {}, "env": {}, "run": {}}
    assert resolve_obj("${params.a}", ctx=ctx) == 3


def test_partial_token_stringifies() -> None:
    ctx = {"params": {"a": 3}, "inputs": {}, "env": {}, "run": {}}
    assert resolve_obj("a=${params.a}", ctx=ctx) == "a=3"


def test_unknown_key_raises() -> None:
    ctx = {"params": {}, "inputs": {}, "env": {}, "run": {}}
    with pytest.raises(InterpError):
        resolve_obj("${params.missing}", ctx=ctx)
