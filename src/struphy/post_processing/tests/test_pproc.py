import logging
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import (
    BinningPlot,
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    SortingParameters,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
    perturbations,
    set_logging_level,
)
from struphy.models import VlasovAmpereOneSpecies

set_logging_level(logging.WARNING)
logger = logging.getLogger("struphy")


@pytest.mark.mpi(min_size=2)
def test_pproc_mpi(tmp_path):

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

    # Keep the weak Landau setup local: examples are not installed with the package.
    model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
    model.em_fields.e_field.save_data = True
    model.em_fields.phi.save_data = True
    model.kinetic_ions.var.save_data = True

    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=1000),
        weights_params=WeightsParameters(control_variate=True),
        boundary_params=BoundaryParameters(),
        sorting_params=SortingParameters(boxes_per_dim=(16, 1, 1), do_sort=True),
        saving_params=SavingParameters(
            binning_plots=(BinningPlot(slice="e1_v1", n_bins=(128, 128), ranges=((0.0, 1.0), (-5.0, 5.0))),),
        ),
        bufsize=0.4,
    )
    model.propagators.push_eta.options = model.propagators.push_eta.Options()
    model.propagators.coupling_va.options = model.propagators.coupling_va.Options()
    model.initial_poisson.options = model.initial_poisson.Options(stab_mat="M0")
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    perturbation = perturbations.ModesCos(amps=(0.001,), ls=(1,))
    model.kinetic_ions.var.add_initial_condition(maxwellians.Maxwellian3D(n=(1.0, perturbation)))

    # pytest creates a separate temporary directory on each rank; share rank 0's.
    out_folders = MPI.COMM_WORLD.bcast(str(tmp_path), root=0)
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folders, sim_folder="weak_Landau"),
        time_opts=Time(dt=0.05, Tend=20.0, split_algo="LieTrotter"),
        domain=domains.Cuboid(r1=12.56),
        grid=grids.TensorProductGrid(num_elements=(32, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )

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
    with TemporaryDirectory() as tmp_dir:
        test_pproc_mpi(tmp_dir)
