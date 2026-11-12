import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def walltime_seconds(walltime: str) -> int:
    """Seconds of a SLURM time limit ``[D-]HH:MM:SS`` (e.g. ``"00:45:00"`` or ``"1-12:00:00"``)."""
    days, _, clock = walltime.rpartition("-")
    h, m, s = (int(part) for part in clock.split(":"))
    return 86400 * int(days or 0) + 3600 * h + 60 * m + s


def scaled_walltime(walltime: str, factor: float, minimum: str = "00:15:00") -> str:
    """The SLURM time limit ``walltime`` times ``factor`` (not below ``minimum``), as ``HH:MM:SS``.

    For example the time limit of a case measured for a number of time steps, scaled to the steps of a run:
    ``scaled_walltime("00:45:00", steps / 50)``.
    """
    seconds = max(walltime_seconds(minimum), round(walltime_seconds(walltime) * factor))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def node_hours(job_ids, timeout: float = 30.0) -> dict[int, float | None]:
    """The node hours used by finished SLURM jobs: elapsed time times allocated nodes, from ``sacct``.

    Parameters
    ----------
    job_ids : int | str | list
        The job IDs (one ``sacct`` call for all).
    timeout : float
        Seconds to wait for ``sacct``.

    Returns
    -------
    dict[int, float | None]
        Node hours per job ID; None where they cannot be determined (no ``sacct``, no accounting, job unknown).
    """
    if not isinstance(job_ids, (list, tuple, set)):
        job_ids = [job_ids]
    ids = []
    for job_id in job_ids:
        match = re.match(r"\s*(\d+)", str(job_id))
        if match:
            ids.append(int(match.group(1)))
    usage: dict[int, float | None] = dict.fromkeys(ids)
    if not ids or not shutil.which("sacct"):
        return usage
    try:
        result = subprocess.run(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--allocations",
                "--jobs",
                ",".join(map(str, ids)),
                "--format=JobIDRaw,ElapsedRaw,NNodes",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return usage
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) == 3 and fields[0].isdigit() and int(fields[0]) in usage:
            try:
                usage[int(fields[0])] = int(fields[1]) * int(fields[2]) / 3600
            except ValueError:
                pass
    return usage


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "unknown"


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError) as exc:
        # e.g. `lscpu` or `scontrol` are not available outside a Linux/SLURM machine.
        return {"command": command, "returncode": 127, "stdout": "", "stderr": str(exc)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _make_unique_results_root(base_dir: Path, run_token: str) -> Path:
    candidate = base_dir / run_token
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = base_dir / f"{run_token}-{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _git_commit_short(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "--short=8", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_commit(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
