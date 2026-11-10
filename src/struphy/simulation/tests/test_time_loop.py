import math

import pytest

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, grids
from struphy.models import Maxwell


def _run(out_folder: str, sim_folder: str, dt: float, Tend: float) -> Simulation:
    sim = Simulation(
        model=Maxwell(),
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=sim_folder),
        time_opts=Time(dt=dt, Tend=Tend),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 4)),
        derham_opts=DerhamOptions(degree=(1, 1, 1)),
    )
    sim.run()
    return sim


@pytest.mark.parametrize("n", [25, 50, 7])
def test_number_of_time_steps(mpi_out_folder: str, n: int):
    """Tend = n dt gives n steps, also when the sum of the time steps ends a round-off below Tend
    (for dt = 2 pi / 25 or 2 pi / 50 an extra step was done before)."""
    Tend = 2 * math.pi
    sim = _run(mpi_out_folder, f"n{n}", dt=Tend / n, Tend=Tend)
    assert int(sim.time_state["index"][0]) == n
    assert math.isclose(float(sim.time_state["value"][0]), Tend, rel_tol=1e-12)


def test_tend_not_a_multiple_of_dt(mpi_out_folder: str):
    """The run goes at least to Tend: 3.5 steps are 4 steps."""
    sim = _run(mpi_out_folder, "partial", dt=0.1, Tend=0.35)
    assert int(sim.time_state["index"][0]) == 4
