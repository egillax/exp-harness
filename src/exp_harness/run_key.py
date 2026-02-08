from __future__ import annotations

import hashlib
from typing import Any

from exp_harness.utils import canonical_json_bytes


def compute_run_key(material: dict[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    return digest[:12]
