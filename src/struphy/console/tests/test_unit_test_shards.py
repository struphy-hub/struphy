import os

import pytest

import struphy
from struphy.console.unit_test_shards import OTHER_TEST_PATHS, UNIT_TEST_SHARDS, unit_test_paths

LIBPATH = struphy.__path__[0]


def _test_files():
    """All test_*.py files of the package, relative to the package folder."""
    files = []
    for root, dirs, names in os.walk(LIBPATH):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        rel = os.path.relpath(root, LIBPATH).replace(os.sep, "/")
        files += [f"{rel}/{f}" for f in names if f.startswith("test_") and f.endswith(".py")]
    return sorted(files)


@pytest.mark.mpi_skip
def test_every_test_file_in_one_shard():
    """Every test file is run by CI: it is covered by exactly one entry of a shard or of OTHER_TEST_PATHS
    (CI does not discover test folders automatically)."""
    listed = unit_test_paths() + OTHER_TEST_PATHS
    duplicates = sorted({p for p in listed if listed.count(p) > 1})
    assert not duplicates, f"test paths listed more than once: {duplicates}"

    for path in listed:
        assert os.path.exists(os.path.join(LIBPATH, path)), f"listed test path does not exist: {path}"

    missing, outside, twice = [], [], []
    for f in _test_files():
        if "/tests/" not in f"/{f}":
            outside.append(f)
        n = sum(f == p or (p.endswith("/") and f.startswith(p)) for p in listed)
        if n == 0:
            missing.append(f)
        elif n > 1:
            twice.append(f)

    assert not outside, f"test files must be in a tests/ folder: {outside}"
    assert not missing, f"test files not run by CI, add them to struphy/console/unit_test_shards.py: {missing}"
    assert not twice, f"test files covered by more than one entry: {twice}"


@pytest.mark.mpi_skip
def test_shard_names():
    assert list(UNIT_TEST_SHARDS) == [f"shard-{n}" for n in range(1, len(UNIT_TEST_SHARDS) + 1)]
