import math

import pytest

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, grids
from struphy.models import Maxwell


def _run(out_folder: str, sim_folder: str, dt: float, Tend: float, one_time_step: bool = False) -> Simulation:
    sim = Simulation(
        model=Maxwell(),
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=sim_folder),
        time_opts=Time(dt=dt, Tend=Tend),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 4)),
        derham_opts=DerhamOptions(degree=(1, 1, 1)),
    )
    sim.run(one_time_step=one_time_step)
    return sim


@pytest.mark.parametrize(
    "Tend, n", [(2 * math.pi, 25), (2 * math.pi, 50), (2 * math.pi, 7), (2e-2 * math.pi, 1000), (1e-6, 3)]
)
def test_number_of_time_steps(mpi_out_folder: str, Tend: float, n: int):
    """Tend = n dt gives n steps, also when the sum of the time steps ends a round-off below Tend
    (for dt = 2 pi / 25 or 2 pi / 50 an extra step was done before, also for small Tend and many steps)."""
    sim = _run(mpi_out_folder, f"n{n}", dt=Tend / n, Tend=Tend)
    assert int(sim.time_state["index"][0]) == n
    assert math.isclose(float(sim.time_state["value"][0]), Tend, rel_tol=1e-12)


def test_tend_not_a_multiple_of_dt(mpi_out_folder: str):
    """The run goes at least to Tend: 3.5 steps are 4 steps."""
    sim = _run(mpi_out_folder, "partial", dt=0.1, Tend=0.35)
    assert int(sim.time_state["index"][0]) == 4


def test_one_time_step(mpi_out_folder: str):
    """run(one_time_step=True) does one step, also for a small dt (round(dt, 14) < dt gave two steps)."""
    sim = _run(mpi_out_folder, "one", dt=1e-6 / 3, Tend=1.0, one_time_step=True)
    assert int(sim.time_state["index"][0]) == 1
