import logging
from pathlib import Path

import numpy as np
import pytest
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import Simulation, set_logging_level
from struphy.io.setup import import_parameters_py

set_logging_level(logging.WARNING)
logger = logging.getLogger("struphy")

PARAMS_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples"
    / "VlasovAmpereOneSpecies"
    / "weak_Landau_damping"
    / "params_weak_Landau_damping.py"
)


@pytest.mark.mpi(min_size=2)
def test_pproc_mpi():

    def do_plotting(sim: Simulation, from_parallel=False):
        sim.load_plotting_data()

        t_grid = sim.t_grid
        eta1 = sim.grids_log[0]
        e_field = sim.spline_values.em_fields.e_field_log
        phi = sim.spline_values.em_fields.phi_log

        f = sim.f.kinetic_ions.e1_v1_density
        print(f.__dict__.keys())
        bins_e1 = f.grid_e1
        bins_v1 = f.grid_v1
        f_binned = f.f_binned
        df_binned = f.delta_f_binned
        print(f"{f_binned.shape=}")

        if from_parallel:
            extra = " (from parallel pproc)"
        else:
            extra = ""

        n = 0  # time index

        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.plot(eta1, e_field.data[t_grid[n]][0][:, 0, 0], label="Ex")
        plt.title(f"Ex at t={t_grid[n]} on rank 0{extra}")
        plt.xlabel("$\\eta1$")
        plt.ylabel("Ex")
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(eta1, phi.data[t_grid[n]][0][:, 0, 0], label="phi")
        plt.title(f"phi at t={t_grid[n]} on rank 0{extra}")
        plt.xlabel("$\\eta1$")
        plt.ylabel("phi")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.pcolor(bins_e1, bins_v1, f_binned[n].T, shading="auto")
        plt.title(f"full f at t={t_grid[n]} on rank 0{extra}")
        plt.xlabel("$\\eta1$")
        plt.ylabel("$v_x$")

        plt.subplot(2, 2, 4)
        plt.pcolor(bins_e1, bins_v1, df_binned[n].T, shading="auto")
        plt.title(f"delta f at t={t_grid[n]} on rank 0{extra}")
        plt.xlabel("$\\eta1$")
        plt.ylabel("$v_x$")

        return (
            e_field.data[t_grid[n]][0][:, 0, 0],
            phi.data[t_grid[n]][0][:, 0, 0],
            f_binned[n].T,
            df_binned[n].T,
        )

    params = import_parameters_py(str(PARAMS_PATH), name="weak_Landau_damping")

    sim: Simulation = params.sim

    sim.run(one_time_step=True)

    if MPI.COMM_WORLD.Get_rank() == 0:
        # serial pproc
        sim.pproc()
        r1, r2, r3, r4 = do_plotting(sim)
    MPI.COMM_WORLD.Barrier()

    # parallel pproc
    sim.pproc(parallel_pproc=True)

    # plot and compare results from serial and parallel pproc
    if MPI.COMM_WORLD.Get_rank() == 0:
        r1_mpi, r2_mpi, r3_mpi, r4_mpi = do_plotting(sim, from_parallel=True)
        plt.show()

        assert np.allclose(r1, r1_mpi)
        assert np.allclose(r2, r2_mpi)
        assert np.allclose(r3, r3_mpi)
        assert np.allclose(r4, r4_mpi)
        print("All checks passed for parallel pproc vs serial pproc.")


if __name__ == "__main__":
    test_pproc_mpi()
