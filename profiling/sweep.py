"""Run several profiling submit scripts as one sweep, in one process.

Each submit script (e.g. ``examples/Poisson/gaussian_cube/submit_gaussian_cube.py``) builds a
:class:`profiling_job.ProfilingCase`, launches its runs and calls ``finalize_run``. Started as separate processes,
the scripts would each prepare (delete or re-clone) the shared packaging folder and push to it. A sweep runs them all
in this process between :func:`profiling_job.begin_sweep` and :func:`profiling_job.end_sweep`: every script submits
its jobs right away (SLURM runs them as the partitions allow) and only registers its case, and
:func:`profiling_job.finalize_all` then packages (and with ``upload`` pushes) every run as its job finishes.

Usage, e.g. ``examples/Poisson/sweep_poisson.py``::

    from sweep import run_sweep, variant_entries

    entries = variant_entries(Path("examples/Poisson/gaussian_cube/submit_gaussian_cube.py"))
    n_failed = run_sweep(entries, upload=False)
"""

import importlib.util
import runpy
import shlex
import sys
import traceback
from pathlib import Path

import profiling_job


def _load(script: Path):
    """Import a submit script as a module (its ``main`` does not run)."""
    name = f"_sweep_{script.stem}_{abs(hash(str(script)))}"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def variant_entries(script) -> list:
    """The entries of a submit script for :func:`run_sweep`: one per key of its ``VARIANTS`` dict, with
    ``--variant <key>``; a script without ``VARIANTS`` gives one entry without arguments.

    Parameters
    ----------
    script : str | Path
        Path of the submit script.

    Returns
    -------
    list[tuple[Path, list[str]]]
        ``(script, arguments)`` per run of the script.
    """
    script = Path(script).resolve()
    variants = getattr(_load(script), "VARIANTS", None)
    if not variants:
        return [(script, [])]
    return [(script, ["--variant", key]) for key in variants]


def _command(script: Path, args: list) -> str:
    return " ".join(shlex.quote(part) for part in ["python", str(script), *args])


def run_sweep(entries, upload: bool = False, dry_run: bool = False, poll_interval: float = 30.0) -> int:
    """Run the submit scripts of ``entries`` in this process, then package (and push) all runs as they finish.

    Every script runs as ``__main__`` with its arguments (plus ``--upload``), inside
    :func:`profiling_job.begin_sweep`, so its ``finalize_run`` only registers its case; a script that fails is
    reported and the sweep goes on. :func:`profiling_job.finalize_all` then waits on the runs of all registered cases
    together.

    Parameters
    ----------
    entries : list[tuple[str | Path, list[str]]]
        ``(script, arguments)`` per submit command, e.g. from :func:`variant_entries`.
    upload : bool
        Pass ``--upload`` to every script (push the packaged runs to the profiling-data repo).
    dry_run : bool
        Only print the commands.
    poll_interval : float
        Seconds between polls of the SLURM queue in :func:`profiling_job.finalize_all`.

    Returns
    -------
    int
        Number of scripts that failed.
    """
    entries = [(Path(script).resolve(), list(args) + (["--upload"] if upload else [])) for script, args in entries]
    print(f"Sweep of {len(entries)} submit command(s):")
    for script, args in entries:
        print(f"  {_command(script, args)}")
    if dry_run:
        return 0

    failed = []
    profiling_job.begin_sweep()
    try:
        for script, args in entries:
            print(f"\n=== {_command(script, args)}", flush=True)
            argv = sys.argv
            sys.argv = [str(script), *args]
            try:
                runpy.run_path(str(script), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    failed.append((script, args))
                    print(f"FAILED (exit code {exc.code}): {_command(script, args)}")
            except Exception:
                failed.append((script, args))
                traceback.print_exc()
                print(f"FAILED: {_command(script, args)}")
            finally:
                sys.argv = argv
    finally:
        cases = profiling_job.end_sweep()

    profiling_job.finalize_all(cases, poll_interval=poll_interval)

    if failed:
        print(f"\n{len(failed)} of {len(entries)} submit command(s) failed:")
        for script, args in failed:
            print(f"  {_command(script, args)}")
    return len(failed)
