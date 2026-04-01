from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from exp_harness.config import Roots
from exp_harness.errors import RunResumeError, StepExecutionError
from exp_harness.runner import resume_experiment, run_experiment


def _write_spec(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _repairable_spec(
    *,
    b_flag: str = "resume.flag",
    b_value: str = "orig-b",
    c_value: str = "orig-c",
    a_value: str = "orig-a",
    c_needs: list[str] | None = None,
) -> dict:
    return {
        "name": "repairable",
        "env": {"kind": "local"},
        "steps": [
            {
                "id": "a",
                "cmd": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import os; "
                        "run_dir=Path(os.environ['EXP_HARNESS_RUN_DIR']); "
                        f"(run_dir/'a.txt').write_text('{a_value}')"
                    ),
                ],
            },
            {
                "id": "b",
                "cmd": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import os,sys; "
                        "run_dir=Path(os.environ['EXP_HARNESS_RUN_DIR']); "
                        f"flag=run_dir/'{b_flag}'; "
                        "flag.exists() or sys.exit(2); "
                        f"(run_dir/'b.txt').write_text('{b_value}')"
                    ),
                ],
            },
            {
                "id": "c",
                "needs": list(c_needs or ["b"]),
                "cmd": [
                    sys.executable,
                    "-c",
                    (
                        "from pathlib import Path; import os; "
                        "run_dir=Path(os.environ['EXP_HARNESS_RUN_DIR']); "
                        f"(run_dir/'c.txt').write_text('{c_value}')"
                    ),
                ],
            },
        ],
    }


def _create_failed_repairable_run(tmp_path: Path) -> tuple[Roots, Path, Path, Path, str]:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(spec_fp, _repairable_spec())

    with pytest.raises(StepExecutionError):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_fp = next((roots.runs_root / "repairable").glob("*/run.json"))
    run_dir = run_json_fp.parent
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    return roots, spec_fp, run_json_fp, run_dir, str(run_json["run_key"])


def test_resume_continues_from_first_incomplete_step(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_it",
            "env": {"kind": "local"},
            "steps": [
                {
                    "id": "a",
                    "cmd": [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; Path('a.done').write_text('ok')",
                    ],
                },
                {
                    "id": "b",
                    "cmd": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; import os,sys; "
                            "flag=Path(os.environ['EXP_HARNESS_RUN_DIR'])/'resume.flag'; "
                            "sys.exit(0 if flag.exists() else 2)"
                        ),
                    ],
                },
                {"id": "c", "cmd": [sys.executable, "-c", "print('c')"]},
            ],
        },
    )

    with pytest.raises(StepExecutionError):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_fp = next((roots.runs_root / "resume_it").glob("*/run.json"))
    run_dir = run_json_fp.parent
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    run_key = str(run_json["run_key"])
    assert run_json["steps"][0]["state"] == "succeeded"
    assert run_json["steps"][1]["state"] == "failed"
    assert run_json["steps"][2]["state"] == "skipped"

    (run_dir / "resume.flag").write_text("ok", encoding="utf-8")

    resumed = resume_experiment(
        roots=roots,
        name="resume_it",
        run_key=run_key,
        enforce_clean=False,
        follow_steps=False,
    )
    assert resumed["run_key"] == run_key
    run_json_after = json.loads(run_json_fp.read_text(encoding="utf-8"))
    assert run_json_after["state"] == "succeeded"

    steps = run_json_after["steps"]
    assert steps[0]["state"] == "succeeded"
    assert steps[0]["attempt"] == 1
    assert steps[1]["state"] == "succeeded"
    assert steps[1]["attempt"] >= 2
    assert steps[2]["state"] == "succeeded"
    assert steps[2]["attempt"] == 1
    attempts = run_json_after.get("resume_attempts") or []
    assert attempts
    assert attempts[-1]["resumed_from_step"] == "b"


def test_resume_refuses_succeeded_run_without_force(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_done",
            "env": {"kind": "local"},
            "steps": [{"id": "a", "cmd": [sys.executable, "-c", "print('ok')"]}],
        },
    )
    res = run_experiment(
        spec_path=spec_fp,
        roots=roots,
        set_overrides=[],
        set_string_overrides=[],
        salt="s",
        enforce_clean=False,
        follow_steps=False,
    )
    with pytest.raises(RunResumeError, match="already succeeded"):
        resume_experiment(
            roots=roots,
            name="resume_done",
            run_key=res["run_key"],
            enforce_clean=False,
        )


def test_resume_refuses_spec_drift_without_override(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_drift",
            "env": {"kind": "local"},
            "steps": [
                {"id": "a", "cmd": [sys.executable, "-c", "print('ok')"]},
                {"id": "b", "cmd": [sys.executable, "-c", "import sys; sys.exit(1)"]},
            ],
        },
    )

    with pytest.raises(StepExecutionError):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_fp = next((roots.runs_root / "resume_drift").glob("*/run.json"))
    run_dir = run_json_fp.parent
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    run_key = str(run_json["run_key"])

    spec_copy = run_dir / "spec.yaml"
    spec_copy.write_text(spec_copy.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")

    with pytest.raises(RunResumeError, match="Spec drift detected"):
        resume_experiment(
            roots=roots,
            name="resume_drift",
            run_key=run_key,
            enforce_clean=False,
        )


def test_resume_policy_rerun_forces_successful_step_rerun(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_policy_rerun",
            "env": {"kind": "local"},
            "steps": [
                {
                    "id": "a",
                    "resume_policy": "rerun",
                    "cmd": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; import os; "
                            "p=Path(os.environ['EXP_HARNESS_RUN_DIR'])/'a.count'; "
                            "n=(int(p.read_text()) if p.exists() else 0)+1; "
                            "p.write_text(str(n))"
                        ),
                    ],
                },
                {
                    "id": "b",
                    "cmd": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; import os,sys; "
                            "flag=Path(os.environ['EXP_HARNESS_RUN_DIR'])/'resume.flag'; "
                            "sys.exit(0 if flag.exists() else 2)"
                        ),
                    ],
                },
            ],
        },
    )
    with pytest.raises(StepExecutionError):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_fp = next((roots.runs_root / "resume_policy_rerun").glob("*/run.json"))
    run_dir = run_json_fp.parent
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    run_key = str(run_json["run_key"])
    (run_dir / "resume.flag").write_text("ok", encoding="utf-8")

    resume_experiment(
        roots=roots,
        name="resume_policy_rerun",
        run_key=run_key,
        enforce_clean=False,
        follow_steps=False,
    )

    run_json_after = json.loads(run_json_fp.read_text(encoding="utf-8"))
    assert run_json_after["steps"][0]["attempt"] >= 2
    assert (run_dir / "a.count").read_text(encoding="utf-8").strip() == "2"


def test_resume_policy_skip_if_marker_reruns_when_marker_missing(tmp_path: Path) -> None:
    roots = Roots(
        project_root=tmp_path, runs_root=tmp_path / "runs", artifacts_root=tmp_path / "artifacts"
    )
    spec_fp = tmp_path / "spec.yaml"
    _write_spec(
        spec_fp,
        {
            "name": "resume_policy_marker",
            "env": {"kind": "local"},
            "steps": [
                {
                    "id": "a",
                    "resume_policy": "skip_if_marker",
                    "success_markers": ["a.ok"],
                    "cmd": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; import os; "
                            "rd=Path(os.environ['EXP_HARNESS_RUN_DIR']); "
                            "count=rd/'a.count'; n=(int(count.read_text()) if count.exists() else 0)+1; "
                            "count.write_text(str(n)); (rd/'a.ok').write_text('ok')"
                        ),
                    ],
                },
                {
                    "id": "b",
                    "cmd": [
                        sys.executable,
                        "-c",
                        (
                            "from pathlib import Path; import os,sys; "
                            "flag=Path(os.environ['EXP_HARNESS_RUN_DIR'])/'resume.flag'; "
                            "sys.exit(0 if flag.exists() else 2)"
                        ),
                    ],
                },
            ],
        },
    )
    with pytest.raises(StepExecutionError):
        run_experiment(
            spec_path=spec_fp,
            roots=roots,
            set_overrides=[],
            set_string_overrides=[],
            salt="s",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_fp = next((roots.runs_root / "resume_policy_marker").glob("*/run.json"))
    run_dir = run_json_fp.parent
    run_json = json.loads(run_json_fp.read_text(encoding="utf-8"))
    run_key = str(run_json["run_key"])
    (run_dir / "a.ok").unlink(missing_ok=True)
    (run_dir / "resume.flag").write_text("ok", encoding="utf-8")

    resume_experiment(
        roots=roots,
        name="resume_policy_marker",
        run_key=run_key,
        enforce_clean=False,
        follow_steps=False,
    )

    run_json_after = json.loads(run_json_fp.read_text(encoding="utf-8"))
    step_a = run_json_after["steps"][0]
    assert step_a["attempt"] >= 2
    assert (run_dir / "a.count").read_text(encoding="utf-8").strip() == "2"
    assert "missing_markers" in (step_a.get("resume_checks") or {})


def test_repair_resume_allows_failed_and_pending_drift(tmp_path: Path) -> None:
    roots, _, run_json_fp, run_dir, run_key = _create_failed_repairable_run(tmp_path)
    repair_spec_fp = tmp_path / "repair.yaml"
    _write_spec(
        repair_spec_fp,
        _repairable_spec(b_value="repair-b", c_value="repair-c"),
    )
    (run_dir / "resume.flag").write_text("ok", encoding="utf-8")

    resumed = resume_experiment(
        roots=roots,
        name="repairable",
        run_key=run_key,
        repair_spec=repair_spec_fp,
        allow_config_drift="failed,pending",
        enforce_clean=False,
        follow_steps=False,
    )

    assert resumed["run_key"] == run_key
    run_json_after = json.loads(run_json_fp.read_text(encoding="utf-8"))
    assert run_json_after["state"] == "succeeded"
    assert (run_dir / "b.txt").read_text(encoding="utf-8") == "repair-b"
    assert (run_dir / "c.txt").read_text(encoding="utf-8") == "repair-c"
    assert run_json_after["resume_mode"] == "repair"
    assert run_json_after["active_spec"]["source"] == "repair"
    repair_events = run_json_after.get("repair_events") or []
    assert len(repair_events) == 1
    changed_ids = {entry["id"] for entry in repair_events[0]["changed_steps"]}
    assert changed_ids == {"b", "c"}
    repair_spec_copy = run_dir / run_json_after["active_spec"]["spec_path"]
    assert repair_spec_copy.exists()


def test_repair_resume_rejects_succeeded_step_drift(tmp_path: Path) -> None:
    roots, _, run_json_fp, _, run_key = _create_failed_repairable_run(tmp_path)
    repair_spec_fp = tmp_path / "repair_bad.yaml"
    _write_spec(
        repair_spec_fp,
        _repairable_spec(a_value="changed-a", b_value="repair-b"),
    )

    with pytest.raises(RunResumeError, match="Disallowed config drift"):
        resume_experiment(
            roots=roots,
            name="repairable",
            run_key=run_key,
            repair_spec=repair_spec_fp,
            allow_config_drift="failed,pending",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_after = json.loads(run_json_fp.read_text(encoding="utf-8"))
    assert not run_json_after.get("repair_events")


def test_repair_resume_dry_run_is_non_mutating(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roots, _, run_json_fp, _, run_key = _create_failed_repairable_run(tmp_path)
    repair_spec_fp = tmp_path / "repair_dry_run.yaml"
    _write_spec(repair_spec_fp, _repairable_spec(b_value="repair-b"))
    before = json.loads(run_json_fp.read_text(encoding="utf-8"))

    resumed = resume_experiment(
        roots=roots,
        name="repairable",
        run_key=run_key,
        repair_spec=repair_spec_fp,
        allow_config_drift="failed",
        dry_run=True,
        enforce_clean=False,
        follow_steps=False,
    )

    assert resumed["run_key"] == run_key
    after = json.loads(run_json_fp.read_text(encoding="utf-8"))
    assert before == after
    captured = capsys.readouterr()
    assert "[resume] mode=repair allowed_states=failed" in captured.out
    assert "[resume] dry-run only; no execution started" in captured.out


def test_repair_resume_updates_active_spec_for_later_strict_resume(tmp_path: Path) -> None:
    roots, _, run_json_fp, run_dir, run_key = _create_failed_repairable_run(tmp_path)
    repair_spec_fp = tmp_path / "repair_active.yaml"
    _write_spec(
        repair_spec_fp,
        _repairable_spec(b_flag="repair.flag", b_value="repair-b"),
    )

    with pytest.raises(StepExecutionError):
        resume_experiment(
            roots=roots,
            name="repairable",
            run_key=run_key,
            repair_spec=repair_spec_fp,
            allow_config_drift="failed,pending",
            enforce_clean=False,
            follow_steps=False,
        )

    run_json_mid = json.loads(run_json_fp.read_text(encoding="utf-8"))
    assert run_json_mid["active_spec"]["source"] == "repair"
    assert len(run_json_mid.get("repair_events") or []) == 1

    (run_dir / "repair.flag").write_text("ok", encoding="utf-8")
    resumed = resume_experiment(
        roots=roots,
        name="repairable",
        run_key=run_key,
        enforce_clean=False,
        follow_steps=False,
    )

    assert resumed["run_key"] == run_key
    assert (run_dir / "b.txt").read_text(encoding="utf-8") == "repair-b"
