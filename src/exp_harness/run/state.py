from __future__ import annotations

import logging
import traceback
from typing import Any

from exp_harness.utils import utc_now_iso

logger = logging.getLogger(__name__)


def mark_succeeded(run_json: dict[str, Any]) -> None:
    run_json["state"] = "succeeded"
    run_json["finished_at_utc"] = utc_now_iso()


def mark_interrupted(run_json: dict[str, Any]) -> None:
    run_json["state"] = "interrupted"
    run_json["finished_at_utc"] = utc_now_iso()


def mark_failed_with_traceback(run_json: dict[str, Any], *, error: Exception) -> None:
    logger.exception("run failed: %s", error)
    run_json["state"] = "failed"
    run_json["finished_at_utc"] = utc_now_iso()
    existing = run_json.get("error")
    if isinstance(existing, dict):
        existing.setdefault("message", str(error))
        existing["traceback"] = traceback.format_exc()
        run_json["error"] = existing
    else:
        run_json["error"] = {"message": str(error), "traceback": traceback.format_exc()}
