"""Package (and push) the runs of profiling cases whose driver stopped before packaging them.

The driver (a ``submit_*.py`` or ``sweep_*.py`` script) submits the SLURM jobs and then waits for them in the
foreground, packaging and pushing each run as its job finishes. If that process ends early (the login session was
closed, the process was killed), the jobs still run to completion, but their results are neither packaged nor pushed.
This script does that afterwards, from the ``profiling_case_info.json`` the driver wrote into every case folder before
it started waiting:

    python profiling/package_case.py <folder> [<folder> ...] [--upload]

Each ``<folder>`` is a case folder ``<profiling folder>/<timestamp>-<commit>/<label>`` (the one with
``profiling_case_info.json`` and the ``sim_<id>`` folders), or any folder above it (e.g. the run folder
``<timestamp>-<commit>``), in which all case folders are found. On Raven the profiling folder is
``/ptmp/<user>/struphy_profiling`` (``clusters.SCRATCH_DIRS``).

Every run with output is packaged; runs whose job is still queued or running are skipped (run this again once they
are done). The packaged folder keeps the name the driver gave it, so with ``--upload`` it replaces the incomplete
folder an interrupted ``--upload`` run already pushed. Without ``--upload`` the cases are packaged into
``--output-dir`` only, for ``upload.py`` later. Runs whose scripts were written by ``write_scripts.py`` have no
``profiling_case_info.json`` and cannot be packaged this way, nor can those of a driver stopped before it submitted
all its jobs (the file is written after the last submission).
"""

import argparse
import getpass
import shutil
import sys
from pathlib import Path

from profiling_job import ProfilingCase
from upload import upload_packaged_cases

CASE_INFO_FILE = "profiling_case_info.json"


def find_case_folders(folders: list[Path]) -> list[Path]:
    """The case folders (with a ``profiling_case_info.json``) in or below ``folders``, sorted and without repeats."""
    cases = set()
    for folder in folders:
        folder = folder.resolve()
        if not folder.is_dir():
            raise SystemExit(f"{folder} is not a folder.")
        if (folder / CASE_INFO_FILE).exists():
            cases.add(folder)
        else:
            cases.update(path.parent for path in folder.rglob(CASE_INFO_FILE))
    return sorted(cases)


def active_job_ids() -> set:
    """The ids of the user's SLURM jobs that are still queued or running (empty without SLURM)."""
    if shutil.which("squeue") is None:
        return set()
    from slurm_script_generator.squeue import SQueue

    queue = SQueue(user=getpass.getuser())
    queue.refresh()
    return {str(job.job_id) for job in queue.jobs() if job.is_active}


def package_case(case_folder: Path, output_dir: Path, active: set) -> Path | None:
    """Package the finished runs of one case into ``output_dir``; the packaged folder, or None if nothing was packaged."""
    case = ProfilingCase.from_case_info(case_folder, output_dir)
    print(f"\n=== {case.label} ({case_folder})")
    case.package_case_metadata(case._case_info)

    n_pending = 0
    for job_info in case.job_infos:
        if job_info.get("job_id") is not None and str(job_info["job_id"]) in active:
            print(f"Job {job_info['job_id']} (run {job_info['launch_id']:02d}) is still queued or running; skipped.")
            n_pending += 1
            continue
        case.package_finished_run(job_info)

    if n_pending:
        print(f"{n_pending} run(s) of '{case.label}' still pending; run this script again once they are done.")
    if not case._packaged_launch_ids:
        print(f"No finished run of '{case.label}' has output; nothing packaged.")
        shutil.rmtree(case.destination_dir, ignore_errors=True)
        return None
    print(f"Packaged {len(case._packaged_launch_ids)} of {len(case.job_infos)} run(s) into {case.destination_dir}")
    return case.destination_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package (and push) the runs of profiling cases whose driver stopped before packaging them.",
    )
    parser.add_argument("folders", type=Path, nargs="+", help="Case folders, or folders to search for case folders.")
    parser.add_argument("--upload", action="store_true", help="Push the packaged cases to the profiling-data repo.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to package the cases (default: a folder 'packaged' next to each case's run folder).",
    )
    args = parser.parse_args()

    case_folders = find_case_folders(args.folders)
    if not case_folders:
        raise SystemExit(f"No {CASE_INFO_FILE} found in {', '.join(str(folder) for folder in args.folders)}.")
    print(f"Found {len(case_folders)} case(s):")
    for folder in case_folders:
        print(f"  {folder}")

    active = active_job_ids()
    packaged = []
    for folder in case_folders:
        output_dir = args.output_dir if args.output_dir is not None else folder.parent / "packaged"
        destination = package_case(folder, output_dir, active)
        if destination is not None:
            packaged.append(destination)

    if not packaged:
        print("\nNothing packaged.")
        sys.exit(1)
    print(f"\nPackaged {len(packaged)} case(s):")
    for destination in packaged:
        print(f"  {destination}")

    if args.upload:
        print(f"\nUploading {len(packaged)} case(s) ...")
        upload_packaged_cases(packaged)
    else:
        print("\nPush them with --upload, or later with")
        print(f"    python {Path(__file__).resolve().parent / 'upload.py'} <packaged case folder> [...]")


if __name__ == "__main__":
    main()
