import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _push_profiling_data(packaged_dirs: list[Path], run_commit: str) -> None:
    """Push newly packaged profiling data folders to the struphy-hub/profiling-data repo."""
    if not packaged_dirs:
        print("No packaged profiling data to push; skipping profiling-data repo push.")
        return

    repo_url = os.environ.get(
        "PROFILING_DATA_REPO_URL", "git@github.com:struphy-hub/profiling-data.git"
    )

    with tempfile.TemporaryDirectory(prefix="profiling-data-") as clone_dir_str:
        clone_dir = Path(clone_dir_str)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            check=True,
        )

        for packaged_dir in packaged_dirs:
            destination = clone_dir / packaged_dir.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(packaged_dir, destination)

        subprocess.run(["git", "-C", str(clone_dir), "add", "."], check=True)
        status = subprocess.run(
            ["git", "-C", str(clone_dir), "diff", "--cached", "--quiet"]
        )
        if status.returncode == 0:
            print("No changes to push to profiling-data repo.")
            return

        subprocess.run(
            [
                "git",
                "-C",
                str(clone_dir),
                "commit",
                "-m",
                f"Add profiling data for struphy commit {run_commit[:8]}",
            ],
            check=True,
        )

        for attempt in range(5):
            push_result = subprocess.run(["git", "-C", str(clone_dir), "push"])
            if push_result.returncode == 0:
                print(
                    f"Pushed {len(packaged_dirs)} profiling data folder(s) to {repo_url}."
                )
                return
            print(
                f"Push attempt {attempt + 1} failed; fetching and rebasing before retrying."
            )
            subprocess.run(["git", "-C", str(clone_dir), "fetch", "origin"], check=True)
            subprocess.run(
                ["git", "-C", str(clone_dir), "rebase", "origin/HEAD"], check=True
            )
            time.sleep(random.randint(1, 5))

        raise RuntimeError(
            "Failed to push profiling data to profiling-data repo after retries."
        )
