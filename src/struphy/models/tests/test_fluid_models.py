import pytest
import inspect
from struphy.models.tests.util import call_model


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('map_and_equil', [('Cuboid', 'HomogenSlab'),
                                           ('HollowTorus', 'AdhocTorus'),
                                           ('Tokamak', 'EQDSKequilibrium')])
def test_fluid(map_and_equil, fast, model=None, Tend=None):
    '''Tests all models and all possible model.options (except solvers without preconditioner) in models/fluid.py.

    If model is not None, tests the specified model.

    The argument "fast" is a pytest option that can be specified at the command line (see conftest.py).'''

    from struphy.models import fluid

    if model is None:
        for key, val in inspect.getmembers(fluid):
            if inspect.isclass(val) and 'StruphyModel' not in key:

                # TODO: remove if-clause
                if 'LinearExtendedMHD' in key and 'HomogenSlab' not in map_and_equil[1]:
                    print(
                        f'Model {key} is currently excluded from tests with mhd_equil other than HomogenSlab.')
                    continue

                if fast:
                    if 'Cuboid' not in map_and_equil[0]:
                        print(
                            f'Fast is enabled, mapping {map_and_equil[0]} skipped ...')
                        continue

                call_model(key, val, map_and_equil, Tend=Tend)
    else:
        val = getattr(fluid, model)

        # TODO: remove if-clause
        if 'LinearExtendedMHD' in model and 'HomogenSlab' not in map_and_equil[1]:
            print(
                f'Model {model} is currently excluded from tests with mhd_equil other than HomogenSlab.')
            exit()

        call_model(model, val, map_and_equil, Tend=Tend)
        
if __name__ == '__main__':
    
    test_fluid(True, ('Cuboid', 'HomogenSlab'), model=None)
    test_fluid(True, ('HollowTorus', 'AdhocTorus'), model=None)
    test_fluid(True, ('Tokamak', 'EQDSKequilibrium'), model=None)