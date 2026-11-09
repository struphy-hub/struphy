"""Test shards of the Struphy unit tests.

This is the single list of test paths used by ``struphy test unit`` and by the CI workflows
(``reusable-unit-testing.yml``, ``reusable-scheduled.yml``), which read it with::

    python -m struphy.console.unit_test_shards shard-2

The paths are relative to the ``struphy`` package folder. Every ``tests/`` folder of the package and every
``models/tests/test_*.py`` file must be in exactly one shard or in :data:`OTHER_TEST_PATHS`; this is checked by
``console/tests/test_unit_test_shards.py``.
"""

import sys

UNIT_TEST_SHARDS = {
    "shard-1": [
        "bsplines/tests/",
        "console/tests/",
        "fields_background/tests/",
        "geometry/tests/",
        "initial/tests/",
        "kinetic_background/tests/",
        "linear_algebra/tests/",
        "models/tests/test_diagnostics_on_device.py",
        "models/tests/test_estimate_mem.py",
        "models/tests/test_hasegawa_wakatani.py",
        "models/tests/test_variational_dissipation.py",
        "models/tests/test_variational_divdiv.py",
    ],
    "shard-2": [
        "api/tests/",
        "propagators/tests/",
        "ode/tests/",
        "polar/tests/",
        "post_processing/tests/",
        "dispersion_relations/tests/",
        "io/tests/",
        "simulation/tests/",
        # end-to-end GPU tests (NumPy reference, and the CuPy comparison on a GPU or the fake CuPy)
        "models/tests/test_gpu_LinearVlasovAmpereOneSpecies.py",
        "models/tests/test_gpu_VlasovAmpereOneSpecies.py",
        "models/tests/test_scalars.py",
        "models/tests/test_splitting_order.py",
    ],
    "shard-3": [
        "feec/tests/",
    ],
    "shard-4": [
        "pic/tests/",
    ],
}

# tests run by their own CI jobs (struphy test models / verification, the examples workflow)
OTHER_TEST_PATHS = [
    "models/tests/regression/",
    "models/tests/verification/",
    "models/tests/test_examples.py",
]


def unit_test_paths() -> list[str]:
    """All unit test paths, in shard order."""
    return [path for paths in UNIT_TEST_SHARDS.values() for path in paths]


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in UNIT_TEST_SHARDS:
        sys.exit(f"usage: python -m struphy.console.unit_test_shards {{{'|'.join(UNIT_TEST_SHARDS)}}}")
    print(" ".join(UNIT_TEST_SHARDS[sys.argv[1]]))
