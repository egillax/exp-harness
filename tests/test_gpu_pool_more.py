from __future__ import annotations

from pathlib import Path

import pytest

from exp_harness.errors import GpuAllocationError
from exp_harness.resources.gpu_pool import GpuPool, allocate_gpus


def test_try_acquire_and_release(tmp_path: Path) -> None:
    pool = GpuPool(locks_dir=tmp_path / "locks")
    assert pool.try_acquire(0, lock_data={"pid": 1})
    assert not pool.try_acquire(0, lock_data={"pid": 2})
    pool.release(0, expected_pid=999)  # should not remove
    assert pool.lock_path(0).exists()
    pool.release(0, expected_pid=1)
    assert not pool.lock_path(0).exists()
    assert pool.try_acquire(0, lock_data={"pid": 3})


def test_list_locks_invalid_json(tmp_path: Path) -> None:
    pool = GpuPool(locks_dir=tmp_path / "locks")
    lp = pool.lock_path(0)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text("not json", encoding="utf-8")
    locks = pool.list_locks()
    assert locks and locks[0].data == {}


def test_allocate_gpus_skips_locked_candidates_and_cleans_up_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = GpuPool(locks_dir=tmp_path / "locks")
    monkeypatch.setattr(pool, "detect_gpu_count", lambda: 2)

    # Pre-lock GPU0.
    assert pool.try_acquire(0, lock_data={"pid": 1})

    alloc = allocate_gpus(pool, 1, run_path="x", run_key="y", name="z")
    assert alloc.gpu_ids == [1]

    # Now request 2 (should fail; and should clean up any partial acquisition).
    with pytest.raises(GpuAllocationError, match="Insufficient free GPUs"):
        allocate_gpus(pool, 2, run_path="x", run_key="y", name="z")

    # GPU1 lock from the failed attempt should not exist (only the earlier allocation still holds).
    # Release the successful one to check cleanup deterministically.
    pool.release(1, expected_pid=alloc.pid)
    assert not pool.lock_path(1).exists()


def test_detect_gpu_count_parses_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pool = GpuPool(locks_dir=tmp_path / "locks")
    monkeypatch.setattr("exp_harness.resources.gpu_pool.which", lambda _: "nvidia-smi")

    def fake_shell_out(argv, cwd=None):
        assert argv == ["nvidia-smi", "-L"]
        return 0, "GPU 0: X\nGPU 1: Y\n", ""

    monkeypatch.setattr("exp_harness.resources.gpu_pool.shell_out", fake_shell_out)
    assert pool.detect_gpu_count() == 2
