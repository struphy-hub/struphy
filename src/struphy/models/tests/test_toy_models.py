import inspect

import pytest

from struphy.models.tests.util import call_model


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "map_and_equil", [("Cuboid", "HomogenSlab"), ("HollowTorus", "AdhocTorus"), ("Tokamak", "EQDSKequilibrium")]
)
def test_toy(map_and_equil, fast, verbose, model=None, Tend=None):
    """Tests all models and all possible model.options (except solvers without preconditioner) in models/toy.py.

    If model is not None, tests the specified model.

    The argument "fast" is a pytest option that can be specified at the command line (see conftest.py)."""

    from mpi4py import MPI

    from struphy.models import toy

    comm = MPI.COMM_WORLD

    if model is None:
        for key, val in inspect.getmembers(toy):
            if inspect.isclass(val) and key not in {"StruphyModel", "Propagator"}:
                if fast:
                    if "Cuboid" not in map_and_equil[0]:
                        print(f"Fast is enabled, mapping {map_and_equil[0]} skipped ...")
                        continue

                call_model(key, val, map_and_equil, Tend=Tend, verbose=verbose, comm=comm)
    else:
        val = getattr(toy, model)
        call_model(model, val, map_and_equil, Tend=Tend, verbose=verbose, comm=comm)


if __name__ == "__main__":
    test_toy(("Cuboid", "HomogenSlab"), True, True)
    test_toy(("HollowTorus", "AdhocTorus"), True, True)
    test_toy(("Tokamak", "EQDSKequilibrium"), True, True)
