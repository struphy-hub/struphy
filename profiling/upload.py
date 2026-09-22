"""Pushing packaged profiling results to the profiling-data repo.

Used by `ProfilingCase` during a `--upload` run, and standalone afterwards: a run
without `--upload` packages its case folder locally, and

    python upload.py <packaged case folder>

pushes that folder as it stands, without re-running the case. `finalize_run` prints
this command with the folder filled in.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
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


def _packaged_case_commit(case_dir: Path) -> str:
    """The Struphy commit a packaged case was produced with, for the commit message.

    Read from the case's own `case_metadata.json`; falls back to the commit token in
    the folder name (`<timestamp>-<commit>-<testcase>`) if it is not recorded there.
    """
    metadata = json.loads((case_dir / "case_metadata.json").read_text(encoding="utf-8"))
    commit = metadata.get("software_information", {}).get("struphy_commit")
    if commit:
        return commit

    name_parts = case_dir.name.split("-")
    return name_parts[1] if len(name_parts) > 1 else "unknown"


def upload_packaged_case(case_dir: Path) -> None:
    """Push one already-packaged case folder to the profiling-data repo.

    For a case packaged by a run without `--upload`: the folder is copied into a fresh
    clone as it stands (replacing an earlier upload of the same folder name) and pushed.
    Nothing is re-packaged, so a case can be reviewed locally first and uploaded later.
    """
    case_dir = case_dir.resolve()
    if not (case_dir / "case_metadata.json").exists():
        raise SystemExit(
            f"{case_dir} is not a packaged profiling case folder: no case_metadata.json in it.",
        )

    with tempfile.TemporaryDirectory(prefix="profiling-data-") as temporary_dir:
        clone_dir = _clone_profiling_data(Path(temporary_dir) / "profiling-data")
        destination = clone_dir / case_dir.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(case_dir, destination)
        _push_profiling_data(clone_dir, _packaged_case_commit(case_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push an already-packaged profiling case folder to the profiling-data repo.",
    )
    parser.add_argument(
        "case_dir",
        type=Path,
        help="The packaged case folder to push, as printed by a run made without --upload.",
    )
    upload_packaged_case(parser.parse_args().case_dir)


if __name__ == "__main__":
    main()
