import inspect

import pytest

from struphy.models import toy
from struphy.models.tests.util import call_model


@pytest.mark.performance
def profile_performance():
    from struphy.profiling.profiling import (
        ProfilingConfig,
        pylikwid_markerclose,
        pylikwid_markerinit,
    )

    map_and_equil = ("Cuboid", "HomogenSlab")
    fast = True
    Tend = 0.005

    # Enable profiling if likwid == True
    config = ProfilingConfig()
    config.likwid = True
    config.sample_duration = 0.0001
    config.sample_interval = 1

    pylikwid_markerinit()
    for key, val in inspect.getmembers(toy):
        if inspect.isclass(val) and key not in {"StruphyModel", "Propagator"}:
            if fast:
                if "Cuboid" not in map_and_equil[0]:
                    print(
                        f"Fast is enabled, mapping {map_and_equil[0]} skipped ...",
                    )
                    continue
            config.simulation_label = f"{key}_"
            call_model(key, val, map_and_equil, Tend=Tend)

    pylikwid_markerclose()


if __name__ == "__main__":
    profile_performance()
