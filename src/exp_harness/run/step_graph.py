from __future__ import annotations

from typing import Any


def validate_and_toposort_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for step in steps:
        step_id = step.get("id")
        if not isinstance(step_id, str) or not step_id:
            raise ValueError("Step has missing/empty id")
        if step_id in by_id:
            raise ValueError(f"Duplicate step id: {step_id}")
        by_id[step_id] = step

    for step_id, step in by_id.items():
        needs = step.get("needs") or []
        for dep in needs:
            if dep not in by_id:
                raise ValueError(f"Step {step_id} needs missing dependency: {dep}")

    incoming: dict[str, set[str]] = {
        step_id: set(step.get("needs") or []) for step_id, step in by_id.items()
    }
    outgoing: dict[str, set[str]] = {}
    for step_id, deps in incoming.items():
        for dep in deps:
            outgoing.setdefault(dep, set()).add(step_id)

    ready = sorted([step_id for step_id, deps in incoming.items() if not deps])
    order: list[str] = []
    while ready:
        step_id = ready.pop(0)
        order.append(step_id)
        for nxt in sorted(outgoing.get(step_id, set())):
            incoming[nxt].discard(step_id)
            if not incoming[nxt] and nxt not in order and nxt not in ready:
                ready.append(nxt)
                ready.sort()

    if len(order) != len(by_id):
        remaining = [step_id for step_id in by_id if step_id not in order]
        raise ValueError(f"Cycle detected in steps (remaining): {remaining}")

    return [by_id[step_id] for step_id in order]
