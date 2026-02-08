from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp_harness.gpu_pool import GpuPool, allocate_gpus


def test_lockinfo_pid_parsing_does_not_crash(tmp_path: Path) -> None:
    locks_dir = tmp_path / "locks"
    locks_dir.mkdir()
    fp = locks_dir / "gpu0.lock"
    fp.write_text(json.dumps({"pid": "123"}), encoding="utf-8")

    pool = GpuPool(locks_dir=locks_dir)
    locks = pool.list_locks()
    assert locks and locks[0].pid == 123


def test_allocate_gpus_validates_explicit_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = GpuPool(locks_dir=tmp_path / "locks")
    monkeypatch.setattr(pool, "detect_gpu_count", lambda: 2)

    with pytest.raises(RuntimeError, match=r"out of range"):
        allocate_gpus(pool, [7], run_path="x", run_key="y", name="z")


def test_allocate_gpus_explicit_ids_requires_detection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = GpuPool(locks_dir=tmp_path / "locks")
    monkeypatch.setattr(pool, "detect_gpu_count", lambda: 0)

    with pytest.raises(RuntimeError, match=r"No GPUs detected"):
        allocate_gpus(pool, [0], run_path="x", run_key="y", name="z")
