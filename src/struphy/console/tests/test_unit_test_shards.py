import os

import pytest

import struphy
from struphy.console.unit_test_shards import OTHER_TEST_PATHS, UNIT_TEST_SHARDS, unit_test_paths

LIBPATH = struphy.__path__[0]


@pytest.mark.mpi_skip
def test_every_test_path_in_one_shard():
    """Every tests/ folder and every models/tests/test_*.py file is run by CI: it is in exactly one shard or in
    OTHER_TEST_PATHS (CI does not discover test folders automatically)."""
    listed = unit_test_paths() + OTHER_TEST_PATHS
    duplicates = sorted({p for p in listed if listed.count(p) > 1})
    assert not duplicates, f"test paths listed more than once: {duplicates}"

    for path in listed:
        assert os.path.exists(os.path.join(LIBPATH, path)), f"listed test path does not exist: {path}"

    # tests/ folders containing test files (models/tests/ is listed file by file, see below)
    missing = []
    for root, dirs, files in os.walk(LIBPATH):
        rel = os.path.relpath(root, LIBPATH).replace(os.sep, "/")
        if os.path.basename(root) != "tests" or rel == "models/tests":
            continue
        if any(f.startswith("test_") and f.endswith(".py") for f in files) and rel + "/" not in listed:
            missing.append(rel + "/")

    models_tests = os.path.join(LIBPATH, "models", "tests")
    for f in sorted(os.listdir(models_tests)):
        if f.startswith("test_") and f.endswith(".py") and f"models/tests/{f}" not in listed:
            missing.append(f"models/tests/{f}")

    assert not missing, f"test paths not run by CI, add them to struphy/console/unit_test_shards.py: {missing}"


@pytest.mark.mpi_skip
def test_shard_names():
    assert list(UNIT_TEST_SHARDS) == [f"shard-{n}" for n in range(1, len(UNIT_TEST_SHARDS) + 1)]
