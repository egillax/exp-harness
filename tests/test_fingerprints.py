from __future__ import annotations

import os
from pathlib import Path

from exp_harness.fingerprints import fingerprint_path


def test_sha256_tree_changes_on_content_change(tmp_path: Path) -> None:
    fp = tmp_path / "a.txt"
    fp.write_text("hello", encoding="utf-8")
    r1 = fingerprint_path(fp, kind="sha256_tree", include=[], exclude=[])
    fp.write_text("hello2", encoding="utf-8")
    r2 = fingerprint_path(fp, kind="sha256_tree", include=[], exclude=[])
    assert r1.value != r2.value


def test_sha256_files_changes_on_mtime_or_size(tmp_path: Path) -> None:
    fp = tmp_path / "a.txt"
    fp.write_text("hello", encoding="utf-8")
    r1 = fingerprint_path(fp, kind="sha256_files", include=[], exclude=[])
    st = fp.stat()
    os.utime(fp, (st.st_atime + 10, st.st_mtime + 10))
    r2 = fingerprint_path(fp, kind="sha256_files", include=[], exclude=[])
    assert r1.value != r2.value

    fp.write_text("hello!!!", encoding="utf-8")
    r3 = fingerprint_path(fp, kind="sha256_files", include=[], exclude=[])
    assert r2.value != r3.value


def test_include_exclude_globs(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    r_all = fingerprint_path(tmp_path, kind="sha256_tree", include=[], exclude=[])
    r_a_only = fingerprint_path(tmp_path, kind="sha256_tree", include=["a*"], exclude=[])
    r_exclude_b = fingerprint_path(tmp_path, kind="sha256_tree", include=[], exclude=["b*"])
    assert r_all.files_hashed == 2
    assert r_a_only.files_hashed == 1
    assert r_exclude_b.files_hashed == 1
    assert r_a_only.value == r_exclude_b.value
