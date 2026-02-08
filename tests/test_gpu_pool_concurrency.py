from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import pytest


@pytest.mark.slow
def test_gpu_lock_conflict_across_processes(tmp_path: Path) -> None:
    locks_dir = tmp_path / "locks"

    def worker(req, sleep_s, q):
        from exp_harness.gpu_pool import GpuPool, allocate_gpus

        GpuPool.detect_gpu_count = lambda self: 4  # type: ignore[method-assign]
        pool = GpuPool(locks_dir=locks_dir)
        try:
            alloc = allocate_gpus(pool, req, run_path="x", run_key="y", name="z")
            q.put(("ok", alloc.gpu_ids))
            time.sleep(sleep_s)
        except Exception as e:
            q.put(("err", str(e)))

    q = mp.Queue()
    p1 = mp.Process(target=worker, args=([0, 1], 1.0, q))
    p2 = mp.Process(target=worker, args=([1], 0.1, q))
    p1.start()
    time.sleep(0.2)
    p2.start()
    p1.join(timeout=5)
    p2.join(timeout=5)
    assert p1.exitcode == 0
    assert p2.exitcode == 0
    results = [q.get(timeout=1) for _ in range(2)]
    assert ("ok", [0, 1]) in results
    errs = [r for r in results if r[0] == "err"]
    assert errs, f"expected one error, got: {results}"
