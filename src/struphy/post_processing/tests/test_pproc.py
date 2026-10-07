import logging
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from maybempi import MPI

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
        e_field = run.fields.em_fields.e_field.isel(t=0, component=0, eta2=0, eta3=0)
        phi = run.fields.em_fields.phi.isel(t=0, eta2=0, eta3=0)
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
                data.plot(x="eta1", y="v1")
                plt.title(f"{title} at t={float(data.t)} on rank 0{extra}")

        return tuple(np.asarray(data) for data in (e_field, phi, f_binned, df_binned))

    test_mod = import_parameters_py(str(PARAMS_PATH), name="weak_Landau_damping")

    sim: Simulation = test_mod.test_weak_Landau(do_plot=False, exit_before_run=True)

    sim.run(one_time_step=True)
    run = Output(sim.env.path_out)

    # serial pproc
    run.pproc(create_vtk=True, parallel=False)
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


@pytest.mark.mpi(min_size=2)
def test_products_are_processed_in_parallel_on_first_use():
    """Without pproc(), evaluate() processes on every rank and gives the serial products."""
    import shutil

    comm = MPI.COMM_WORLD
    test_mod = import_parameters_py(str(PARAMS_PATH), name="weak_Landau_damping")
    sim: Simulation = test_mod.test_weak_Landau(do_plot=False, exit_before_run=True)
    out = sim.run(one_time_step=True)
    if comm.Get_rank() == 0:
        shutil.rmtree(Path(sim.env.path_out) / "post_processing", ignore_errors=True)
    comm.Barrier()

    modes = []
    setup = Output._setup_processing
    Output._setup_processing = lambda self, parallel: modes.append(parallel) or setup(self, parallel)
    try:
        f = np.asarray(out.evaluate("kinetic_ions/f"))
        e = np.asarray(out.fields.em_fields.e_field)
    finally:
        Output._setup_processing = setup
    assert modes == [True]

    serial = Output(sim.env.path_out).pproc(parallel=False, force=True)
    assert np.allclose(np.asarray(serial.evaluate("kinetic_ions/f")), f)
    assert np.allclose(np.asarray(serial.fields.em_fields.e_field), e)
    comm.Barrier()


if __name__ == "__main__":
    test_pproc_mpi(show_plot=True)
