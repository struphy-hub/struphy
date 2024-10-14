import pytest
import inspect
from struphy.models.tests.util import call_model


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('map_and_equil', [('Cuboid', 'HomogenSlab'),
                                           ('HollowTorus', 'AdhocTorus'),
                                           ('Tokamak', 'EQDSKequilibrium')
                                           ])
def test_toy(map_and_equil, fast, model=None, Tend=None):
    '''Tests all models and all possible model.options (except solvers without preconditioner) in models/toy.py.

    If model is not None, tests the specified model.

    The argument "fast" is a pytest option that can be specified at the command line (see conftest.py).'''

    from struphy.models import toy

    if model is None:
        for key, val in inspect.getmembers(toy):
            if inspect.isclass(val) and key not in {'StruphyModel', 'Propagator'}:

                if fast:
                    if 'Cuboid' not in map_and_equil[0]:
                        print(
                            f'Fast is enabled, mapping {map_and_equil[0]} skipped ...')
                        continue

                call_model(key, val, map_and_equil, Tend=Tend)
    else:
        val = getattr(toy, model)
        call_model(model, val, map_and_equil, Tend=Tend)


if __name__ == '__main__':

    test_toy(('Cuboid', 'HomogenSlab'), True)
    test_toy(('HollowTorus', 'AdhocTorus'), True)
    test_toy(('Tokamak', 'EQDSKequilibrium'), True)
