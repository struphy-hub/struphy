import inspect

import pytest

from struphy.models.tests.util import call_model


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize(
    "map_and_equil", [("Cuboid", "HomogenSlab"), ("HollowTorus", "AdhocTorus"), ("Tokamak", "EQDSKequilibrium")]
)
def test_fluid(map_and_equil, fast, vrbose, model=None, Tend=None, verbose=False):
    """Tests all models and all possible model.options (except solvers without preconditioner) in models/fluid.py.

    If model is not None, tests the specified model.

    The argument "fast" is a pytest option that can be specified at the command line (see conftest.py)."""

    from mpi4py import MPI

    from struphy.models import fluid

    comm = MPI.COMM_WORLD

    if model is None:
        for key, val in inspect.getmembers(fluid):
            if inspect.isclass(val) and key not in {"StruphyModel", "Propagator"}:
                # TODO: remove if-clauses
                if "LinearExtendedMHD" in key and "HomogenSlab" not in map_and_equil[1]:
                    print(f"Model {key} is currently excluded from tests with mhd_equil other than HomogenSlab.")
                    continue

                if fast:
                    if "Cuboid" not in map_and_equil[0]:
                        print(f"Fast is enabled, mapping {map_and_equil[0]} skipped ...")
                        continue

                call_model(key, val, map_and_equil, Tend=Tend, verbose=vrbose, comm=comm)
    else:
        val = getattr(fluid, model)

        # TODO: remove if-clause
        if "LinearExtendedMHD" in model and "HomogenSlab" not in map_and_equil[1]:
            print(f"Model {model} is currently excluded from tests with mhd_equil other than HomogenSlab.")
            exit()

        call_model(model, val, map_and_equil, Tend=Tend, verbose=vrbose, comm=comm)


if __name__ == "__main__":
    test_fluid(("Cuboid", "HomogenSlab"), False, model="ViscoresistiveMHD")
    test_fluid(("HollowTorus", "AdhocTorus"), False, model="ViscoresistiveMHD")
    test_fluid(("Tokamak", "EQDSKequilibrium"), False, model="ViscoresistiveMHD")
