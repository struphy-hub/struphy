
import inspect
from struphy.models.tests.util import call_model
from struphy.models import toy

from struphy.profiling.profiling import (
    # ProfileRegion,
    set_likwid,
    pylikwid_markerinit,
    pylikwid_markerclose,
    set_simulation_label,
)

def test_performance():
    map_and_equil = ('Cuboid', 'HomogenSlab')
    fast = True
    Tend = 0.005
    set_likwid(True)
    pylikwid_markerinit()
    for key, val in inspect.getmembers(toy):
        if inspect.isclass(val) and key not in {'StruphyModel', 'Propagator'}:

            if fast:
                if 'Cuboid' not in map_and_equil[0]:
                    print(
                        f'Fast is enabled, mapping {map_and_equil[0]} skipped ...')
                    continue
            set_simulation_label(f'{key}_')
            call_model(key, val, map_and_equil, Tend=Tend)
            
    pylikwid_markerclose()
    

if __name__ == '__main__':
    test_performance()
    