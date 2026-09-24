import logging
from pathlib import Path

import numpy as np
import pytest
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import Output, Simulation, set_logging_level
from struphy.io.setup import import_parameters_py

set_logging_level(logging.WARNING)
logger = logging.getLogger("struphy")

PARAMS_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "tests" / "verification" / "test_verif_VlasovAmpereOneSpecies.py"
)


@pytest.mark.mpi(min_size=2)
def test_pproc_mpi(show_plot=False):

    def do_plotting(run: Output, from_parallel=False):
        e_field = run.fields.em_fields.e_field.isel(t=0, component=0, e2=0, e3=0)
        phi = run.fields.em_fields.phi.isel(t=0, e2=0, e3=0)
        f = run.distributions.kinetic_ions.e1_v1_density
        f_binned = f.f.isel(t=0)
        df_binned = f.delta_f.isel(t=0)

        if show_plot:
            extra = " (from parallel pproc)" if from_parallel else ""
            plt.figure(figsize=(12, 12))
            for index, (data, title) in enumerate(((e_field, "Ex"), (phi, "phi")), 1):
                plt.subplot(2, 2, index)
                data.plot(label=title)
                plt.title(f"{title} at t={float(data.t)} on rank 0{extra}")
                plt.legend()
            for index, (data, title) in enumerate(((f_binned, "full f"), (df_binned, "delta f")), 3):
                plt.subplot(2, 2, index)
                data.plot(x="e1", y="v1")
                plt.title(f"{title} at t={float(data.t)} on rank 0{extra}")

        return tuple(np.asarray(data) for data in (e_field, phi, f_binned, df_binned))

    test_mod = import_parameters_py(str(PARAMS_PATH), name="weak_Landau_damping")

    sim: Simulation = test_mod.test_weak_Landau(do_plot=False, exit_before_run=True)

    sim.run(one_time_step=True)
    run = Output(sim.env.path_out)

    # serial pproc
    run.pproc(create_vtk=True)
    if sim.rank == 0:
        serial = do_plotting(run)

    # parallel pproc
    run.pproc(create_vtk=True, parallel=True, force=True)

    # plot and compare results from serial and parallel pproc
    if sim.rank == 0:
        parallel = do_plotting(run, from_parallel=True)
        if show_plot:
            plt.show()

        for expected, actual in zip(serial, parallel):
            assert np.allclose(expected, actual)
        print("All checks passed for parallel pproc vs serial pproc.")
    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    test_pproc_mpi(show_plot=True)
