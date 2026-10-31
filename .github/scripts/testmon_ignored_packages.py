"""Print the installed packages that testmon should not fingerprint, separated by spaces.

testmon re-runs all tests when the major.minor version of any installed package changes. The CI image is rebuilt
(with the latest dependencies) on every push to devel, so that happened in most PR runs. Only the packages in
``FINGERPRINTED`` are kept in the fingerprint; pass the output to pytest as
``-o "testmon_ignore_dependencies=$(python testmon_ignored_packages.py)"``.
"""

import importlib.metadata
import re

# packages whose (major.minor) update should re-run all tests
FINGERPRINTED = {
    "struphy",
    "feectools",
    "numpy",
    "scipy",
    "cunumpy",
    "pyccel",
    "maybempi",
    "mpi4py",
    "h5py",
    "pytest",
    "pytest-mpi",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


names = {dist.metadata["Name"] for dist in importlib.metadata.distributions() if dist.metadata["Name"]}
print(" ".join(sorted(name for name in names if _normalize(name) not in FINGERPRINTED)))
