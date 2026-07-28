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


def write_latest_run_root(marker_path: Path, run_root: Path) -> None:
    """Record `run_root` as the most recently produced profiling run, for later discovery."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(str(run_root), encoding="utf-8")


def read_latest_run_root(marker_path: Path) -> Path | None:
    """The run root last recorded via `write_latest_run_root`, if the marker still resolves."""
    if not marker_path.exists():
        return None
    marker_root = Path(marker_path.read_text(encoding="utf-8").strip())
    if not marker_root.is_absolute():
        marker_root = (marker_path.parent / marker_root).resolve()
    return marker_root if marker_root.exists() else None


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
