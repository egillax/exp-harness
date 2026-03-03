from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from exp_harness.config import ENV_ARTIFACTS_ROOT, ENV_RUNS_ROOT, resolve_roots
from exp_harness.run.api import OverrideParseError, parse_set_overrides
from exp_harness.utils import discover_project_root_from_dir

app = typer.Typer(add_completion=False, no_args_is_help=True)
locks_app = typer.Typer(no_args_is_help=True)
app.add_typer(locks_app, name="locks")


@app.command()
def run(
    spec: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override values (YAML-parsed), e.g. params.x=3"),
    ] = None,
    set_str: Annotated[
        list[str] | None,
        typer.Option("--set-str", "--set-string", help="Override values as strings"),
    ] = None,
    runs_root: Annotated[
        Path | None,
        typer.Option(
            "--runs-root",
            help=f"Override runs root (env: {ENV_RUNS_ROOT})",
        ),
    ] = None,
    artifacts_root: Annotated[
        Path | None,
        typer.Option(
            "--artifacts-root",
            help=f"Override artifacts root (env: {ENV_ARTIFACTS_ROOT})",
        ),
    ] = None,
    salt: Annotated[
        str | None, typer.Option("--salt", help="Run-key salt (forces new run identity)")
    ] = None,
    run_label: Annotated[
        str | None,
        typer.Option(
            "--run-label",
            help="Optional human label used in run directory naming (timestamp__label__hash).",
        ),
    ] = None,
    enforce_clean: Annotated[
        bool, typer.Option("--enforce-clean", help="Fail if git working tree is dirty")
    ] = False,
    follow_steps: Annotated[
        bool,
        typer.Option(
            "--follow-steps/--no-follow-steps",
            "--follow/--no-follow",
            help="Stream step stdout/stderr to the terminal while running (still writes stdout.log/stderr.log).",
        ),
    ] = True,
    stderr_tail_lines: Annotated[
        int,
        typer.Option(
            "--stderr-tail-lines",
            help="On step failure, print the last N lines of stderr.log (0 disables).",
        ),
    ] = 120,
) -> None:
    """
    Run a multi-step experiment from a YAML spec.
    """
    from exp_harness.run.api import run_experiment

    try:
        set_kv = parse_set_overrides(set_ or [])
        set_str_kv = parse_set_overrides(set_str or [])
    except OverrideParseError as e:
        raise typer.BadParameter(str(e)) from e

    res = run_experiment(
        spec_path=spec,
        set_overrides=set_kv,
        set_string_overrides=set_str_kv,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
        salt=salt,
        run_label=run_label,
        enforce_clean=enforce_clean,
        follow_steps=follow_steps,
        stderr_tail_lines=stderr_tail_lines,
    )
    typer.echo(f"{res['name']} {res['run_key']}")
    typer.echo(f"run_id: {res['run_id']}")
    typer.echo(f"run_dir: {res['run_dir']}")
    typer.echo(f"artifacts_dir: {res['artifacts_dir']}")


@app.command()
def status(
    name: Annotated[str | None, typer.Option("--name", help="Filter by experiment name")] = None,
    runs_root: Annotated[
        Path | None, typer.Option("--runs-root", help=f"Override runs root (env: {ENV_RUNS_ROOT})")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max runs to show")] = 20,
) -> None:
    from exp_harness.status import print_status

    project_root = discover_project_root_from_dir(Path.cwd())
    roots = resolve_roots(project_root=project_root, runs_root=runs_root, artifacts_root=None)
    print_status(roots=roots, name=name, limit=limit)


@app.command()
def logs(
    name: Annotated[str, typer.Argument(help="Experiment name")],
    run_key: Annotated[str, typer.Argument(help="Run key")],
    step: Annotated[str | None, typer.Option("--step", help="Step id")] = None,
    follow: Annotated[bool, typer.Option("-f", "--follow", help="Follow log output")] = False,
    runs_root: Annotated[
        Path | None, typer.Option("--runs-root", help=f"Override runs root (env: {ENV_RUNS_ROOT})")
    ] = None,
) -> None:
    from exp_harness.logs import show_logs

    project_root = discover_project_root_from_dir(Path.cwd())
    roots = resolve_roots(project_root=project_root, runs_root=runs_root, artifacts_root=None)
    show_logs(roots=roots, name=name, run_key=run_key, step=step, follow=follow)


@app.command()
def inspect(
    name: Annotated[str, typer.Argument(help="Experiment name")],
    run_key: Annotated[str, typer.Argument(help="Run key")],
    runs_root: Annotated[
        Path | None, typer.Option("--runs-root", help=f"Override runs root (env: {ENV_RUNS_ROOT})")
    ] = None,
) -> None:
    from exp_harness.inspect_run import inspect_run

    project_root = discover_project_root_from_dir(Path.cwd())
    roots = resolve_roots(project_root=project_root, runs_root=runs_root, artifacts_root=None)
    inspect_run(roots=roots, name=name, run_key=run_key)


@locks_app.command("gc")
def locks_gc(
    grace_seconds: Annotated[
        int, typer.Option("--grace-seconds", help="Orphan grace period before proposing removal")
    ] = 600,
    force: Annotated[
        bool, typer.Option("--force", help="Actually remove eligible orphan locks")
    ] = False,
    runs_root: Annotated[
        Path | None, typer.Option("--runs-root", help=f"Override runs root (env: {ENV_RUNS_ROOT})")
    ] = None,
) -> None:
    from exp_harness.locks import gc_locks

    project_root = discover_project_root_from_dir(Path.cwd())
    roots = resolve_roots(project_root=project_root, runs_root=runs_root, artifacts_root=None)
    gc_locks(roots=roots, grace_seconds=grace_seconds, force=force)


def main(_argv: list[str] | None = None) -> Any:
    return app(standalone_mode=True, prog_name="run-experiment")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
