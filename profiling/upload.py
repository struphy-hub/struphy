import os
import random
import shutil
import subprocess
import time
from pathlib import Path


def _profiling_data_repo_url() -> str:
    """URL of the profiling-data repo results are pushed to."""
    return os.environ.get(
        "PROFILING_DATA_REPO_URL",
        "git@github.com:struphy-hub/profiling-data.git",
    )


def _clone_profiling_data(clone_dir: Path) -> Path:
    """Clone the profiling-data repo into `clone_dir`, replacing whatever is there.

    The packaged results are written straight into this working tree, so pushing them
    is just a commit away — nothing is staged in a separate export folder first.
    """
    repo_url = _profiling_data_repo_url()
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
        check=True,
    )
    return clone_dir


def _push_profiling_data(clone_dir: Path, run_commit: str) -> None:
    """Commit and push whatever has been packaged into the profiling-data clone.

    `clone_dir` is the clone created by `_clone_profiling_data`, i.e. the directory the
    packaged folders are written into directly. Called after every packaging step, so
    each run reaches the repo as soon as it finishes; a call that finds nothing new is
    a no-op.
    """
    subprocess.run(["git", "-C", str(clone_dir), "add", "."], check=True)
    status = subprocess.run(
        ["git", "-C", str(clone_dir), "diff", "--cached", "--quiet"],
        check=False,
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
        push_result = subprocess.run(["git", "-C", str(clone_dir), "push"], check=False)
        if push_result.returncode == 0:
            print(f"Pushed profiling data to {_profiling_data_repo_url()}.")
            return
        print(
            f"Push attempt {attempt + 1} failed; fetching and rebasing before retrying.",
        )
        subprocess.run(["git", "-C", str(clone_dir), "fetch", "origin"], check=True)
        subprocess.run(
            ["git", "-C", str(clone_dir), "rebase", "origin/HEAD"],
            check=True,
        )
        time.sleep(random.randint(1, 5))

    raise RuntimeError(
        "Failed to push profiling data to profiling-data repo after retries.",
    )
