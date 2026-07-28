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


def latest_run_root(base_dir: Path) -> Path | None:
    """The most recent profiling run directory directly under `base_dir`, if any.

    Run directories are created by `_make_unique_results_root` as
    `<timestamp>-<commit>[-<n>]`, so the lexicographically greatest name is also the
    most recently created one — no separate "latest run" marker file needed.
    """
    if not base_dir.exists():
        return None
    run_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.name)


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
