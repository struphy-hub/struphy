import os
import pickle
from pathlib import Path

import h5py
import yaml
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from psydac.ddm.mpi import mpi as MPI
from scipy.special import jv, yn

import struphy
from struphy.post_processing import pproc_struphy
from struphy.utils.arrays import xp


def VlasovAmpereOneSpecies_weakLandau(
    path_out: str,
    rank: int,
    show_plots: bool = False,
):
    """Verification test for weak Landau damping. The computed damping rate is compared to the analytical rate.

    Parameters
    ----------
    path_out : str
        Simulation output folder (absolute path).

    rank : int
        MPI rank.

    show_plots: bool
        Whether to show plots."""

    gamma = -0.1533

    def E_exact(t):
        eps = 0.001
        k = 0.5
        r = 0.3677
        omega = 1.4156
        phi = 0.5362
        return 2 * eps**2 * xp.pi / k**2 * r**2 * xp.exp(2 * gamma * t) * xp.cos(omega * t - phi) ** 2

    # get parameters
    with open(os.path.join(path_out, "parameters.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    dt = params["time"]["dt"]
    algo = params["time"]["split_algo"]
    Nel = params["grid"]["Nel"][0]
    p = params["grid"]["p"][0]
    ppc = params["kinetic"]["species1"]["markers"]["ppc"]

    # get scalar data
    pa_data = os.path.join(path_out, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        E = f["scalar"]["en_E"][()]
    logE = xp.log10(E)

    # find where time derivative of E is zero
    dEdt = (xp.roll(logE, -1) - xp.roll(logE, 1))[1:-1] / (2.0 * dt)
    zeros = dEdt * xp.roll(dEdt, -1) < 0.0
    maxima_inds = xp.logical_and(zeros, dEdt > 0.0)
    maxima = logE[1:-1][maxima_inds]
    t_maxima = time[1:-1][maxima_inds]

    # linear fit
    linfit = xp.polyfit(t_maxima[:5], maxima[:5], 1)
    gamma_num = linfit[0]

    # plot
    if show_plots and rank == 0:
        plt.figure(figsize=(18, 12))
        plt.plot(time, logE, label="numerical")
        plt.plot(time, xp.log10(E_exact(time)), label="exact")
        plt.legend()
        plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
        plt.xlabel("time [m/c]")
        plt.plot(t_maxima[:5], maxima[:5], "r")
        plt.plot(t_maxima[:5], maxima[:5], "or", markersize=10)
        plt.ylim([-10, -4])

        plt.show()

    # assert
    rel_error = xp.abs(gamma_num - gamma) / xp.abs(gamma)
    assert rel_error < 0.25, f"{rank = }: Assertion for weak Landau damping failed: {gamma_num = } vs. {gamma = }."
    print(f"{rank = }: Assertion for weak Landau damping passed ({rel_error = }).")


def LinearVlasovAmpereOneSpecies_weakLandau(
    path_out: str,
    rank: int,
    show_plots: bool = False,
):
    """Verification test for weak Landau damping. The computed damping rate is compared to the analytical rate.

    Parameters
    ----------
    path_out : str
        Simulation output folder (absolute path).

    rank : int
        MPI rank.

    show_plots: bool
        Whether to show plots."""

    gamma = -0.1533

    def E_exact(t):
        eps = 0.001
        k = 0.5
        r = 0.3677
        omega = 1.4156
        phi = 0.5362
        return 2 * eps**2 * xp.pi / k**2 * r**2 * xp.exp(2 * gamma * t) * xp.cos(omega * t - phi) ** 2

    # get parameters
    with open(os.path.join(path_out, "parameters.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    dt = params["time"]["dt"]
    algo = params["time"]["split_algo"]
    Nel = params["grid"]["Nel"][0]
    p = params["grid"]["p"][0]
    ppc = params["kinetic"]["species1"]["markers"]["ppc"]

    # get scalar data
    pa_data = os.path.join(path_out, "data")
    with h5py.File(os.path.join(pa_data, "data_proc0.hdf5"), "r") as f:
        time = f["time"]["value"][()]
        E = f["scalar"]["en_E"][()]
    logE = xp.log10(E)

    # find where time derivative of E is zero
    dEdt = (xp.roll(logE, -1) - xp.roll(logE, 1))[1:-1] / (2.0 * dt)
    zeros = dEdt * xp.roll(dEdt, -1) < 0.0
    maxima_inds = xp.logical_and(zeros, dEdt > 0.0)
    maxima = logE[1:-1][maxima_inds]
    t_maxima = time[1:-1][maxima_inds]

    # linear fit
    linfit = xp.polyfit(t_maxima[:5], maxima[:5], 1)
    gamma_num = linfit[0]

    # plot
    if show_plots and rank == 0:
        plt.figure(figsize=(18, 12))
        plt.plot(time, logE, label="numerical")
        plt.plot(time, xp.log10(E_exact(time)), label="exact")
        plt.legend()
        plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
        plt.xlabel("time [m/c]")
        plt.plot(t_maxima[:5], maxima[:5], "r")
        plt.plot(t_maxima[:5], maxima[:5], "or", markersize=10)
        plt.ylim([-10, -4])
        plt.show()

        # plt.show()

    # assert
    rel_error = xp.abs(gamma_num - gamma) / xp.abs(gamma)
    assert rel_error < 0.25, f"{rank = }: Assertion for weak Landau damping failed: {gamma_num = } vs. {gamma = }."
    print(f"{rank = }: Assertion for weak Landau damping passed ({rel_error = }).")


def IsothermalEulerSPH_soundwave(
    path_out: str,
    rank: int,
    show_plots: bool = False,
):
    """Verification test for SPH discretization of isthermal Euler equations.
    A standing sound wave with c_s=1 traveserses the domain once.

    Parameters
    ----------
    path_out : str
        Simulation output folder (absolute path).

    rank : int
        MPI rank.

    show_plots: bool
        Whether to show plots."""

    path_pp = os.path.join(path_out, "post_processing/")
    if rank == 0:
        pproc_struphy.main(path_out)
    MPI.COMM_WORLD.Barrier()
    path_n_sph = os.path.join(path_pp, "kinetic_data/euler_fluid/n_sph/view_0/")

    ee1, ee2, ee3 = xp.load(os.path.join(path_n_sph, "grid_n_sph.npy"))
    n_sph = xp.load(os.path.join(path_n_sph, "n_sph.npy"))
    # print(f'{ee1.shape = }, {n_sph.shape = }')

    if show_plots and rank == 0:
        ppb = 8
        nx = 16
        end_time = 2.5
        dt = 0.0625
        Nt = int(end_time // dt)
        x = ee1 * 2.5

        plt.figure(figsize=(10, 8))
        interval = Nt / 10
        plot_ct = 0
        for i in range(0, Nt + 1):
            if i % interval == 0:
                print(f"{i = }")
                plot_ct += 1
                ax = plt.gca()

                if plot_ct <= 6:
                    style = "-"
                else:
                    style = "."
                plt.plot(x.squeeze(), n_sph[i, :, 0, 0], style, label=f"time={i * dt:4.2f}")
                plt.xlim(0, 2.5)
                plt.legend()
                ax.set_xticks(xp.linspace(0, 2.5, nx + 1))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                plt.grid(c="k")
                plt.xlabel("x")
                plt.ylabel(r"$\rho$")

                plt.title(f"standing sound wave ($c_s = 1$) for {nx = } and {ppb = }")
            if plot_ct == 11:
                break

        plt.show()

    # assert
    error = xp.max(xp.abs(n_sph[0] - n_sph[-1]))
    print(f"{rank = }: Assertion for SPH sound wave passed ({error = }).")
    assert error < 1.3e-3


def Maxwell_coaxial(
    path_out: str,
    show_plots: bool = False,
):
    """Verification test for coaxial cable with Maxwell equations. Comparison w.r.t analytic solution.

    Solutions taken from TUM master thesis of Alicia Robles Pérez:
    "Development of a Geometric Particle-in-Cell Method for Cylindrical Coordinate Systems", 2024

    Parameters
    ----------
    path_out : str
        Simulation output folder (absolute path).

    show_plots: bool
        Whether to show plots."""

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        pproc_struphy.main(path_out, physical=True)
    MPI.COMM_WORLD.Barrier()

    def B_z(X, Y, Z, m, t):
        """Magnetic field in z direction of coaxial cabel"""
        r = (X**2 + Y**2) ** 0.5
        theta = xp.arctan2(Y, X)
        return (jv(m, r) - 0.28 * yn(m, r)) * xp.cos(m * theta - t)

    def E_r(X, Y, Z, m, t):
        """Electrical field in radial direction of coaxial cabel"""
        r = (X**2 + Y**2) ** 0.5
        theta = xp.arctan2(Y, X)
        return -m / r * (jv(m, r) - 0.28 * yn(m, r)) * xp.cos(m * theta - t)

    def E_theta(X, Y, Z, m, t):
        """Electrical field in azimuthal direction of coaxial cabel"""
        r = (X**2 + Y**2) ** 0.5
        theta = xp.arctan2(Y, X)
        return ((m / r * jv(m, r) - jv(m + 1, r)) - 0.28 * (m / r * yn(m, r) - yn(m + 1, r))) * xp.sin(m * theta - t)

    def to_E_r(X, Y, E_x, E_y):
        r = (X**2 + Y**2) ** 0.5
        theta = xp.arctan2(Y, X)
        return xp.cos(theta) * E_x + xp.sin(theta) * E_y

    def to_E_theta(X, Y, E_x, E_y):
        r = (X**2 + Y**2) ** 0.5
        theta = xp.arctan2(Y, X)
        return -xp.sin(theta) * E_x + xp.cos(theta) * E_y

    # get parameters
    with open(os.path.join(path_out, "parameters.yml")) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    dt = params["time"]["dt"]
    algo = params["time"]["split_algo"]
    Nel = params["grid"]["Nel"][0]
    modes = params["em_fields"]["perturbation"]["e_field"]["CoaxialWaveguideElectric_r"]["m"]

    pproc_path = os.path.join(path_out, "post_processing/")
    em_fields_path = os.path.join(pproc_path, "fields_data/em_fields/")
    t_grid = xp.load(os.path.join(pproc_path, "t_grid.npy"))
    grids_phy = pickle.loads(Path(os.path.join(pproc_path, "fields_data/grids_phy.bin")).read_bytes())
    b_field_phy = pickle.loads(Path(os.path.join(em_fields_path, "b_field_phy.bin")).read_bytes())
    e_field_phy = pickle.loads(Path(os.path.join(em_fields_path, "e_field_phy.bin")).read_bytes())

    X = grids_phy[0][:, :, 0]
    Y = grids_phy[1][:, :, 0]

    # plot
    if show_plots and rank == 0:
        vmin = E_theta(X, Y, grids_phy[0], modes, 0).min()
        vmax = E_theta(X, Y, grids_phy[0], modes, 0).max()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        plot_exac = ax1.contourf(
            X, Y, E_theta(X, Y, grids_phy[0], modes, t_grid[-1]), cmap="plasma", levels=100, vmin=vmin, vmax=vmax
        )
        ax2.contourf(
            X,
            Y,
            to_E_theta(X, Y, e_field_phy[t_grid[-1]][0][:, :, 0], e_field_phy[t_grid[-1]][1][:, :, 0]),
            cmap="plasma",
            levels=100,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(plot_exac, ax=[ax1, ax2], orientation="vertical", shrink=0.9)
        ax1.set_xlabel("Exact")
        ax2.set_xlabel("Numerical")
        fig.suptitle(f"Exact and Simulated $E_\\theta$ Field {dt=}, {algo=}, {Nel=}", fontsize=14)
        plt.show()

    # assert
    Ex_tend = e_field_phy[t_grid[-1]][0][:, :, 0]
    Ey_tend = e_field_phy[t_grid[-1]][1][:, :, 0]
    Er_exact = E_r(X, Y, grids_phy[0], modes, t_grid[-1])
    Etheta_exact = E_theta(X, Y, grids_phy[0], modes, t_grid[-1])
    Bz_tend = b_field_phy[t_grid[-1]][2][:, :, 0]
    Bz_exact = B_z(X, Y, grids_phy[0], modes, t_grid[-1])

    error_Er = xp.max(xp.abs((to_E_r(X, Y, Ex_tend, Ey_tend) - Er_exact)))
    error_Etheta = xp.max(xp.abs((to_E_theta(X, Y, Ex_tend, Ey_tend) - Etheta_exact)))
    error_Bz = xp.max(xp.abs((Bz_tend - Bz_exact)))

    rel_err_Er = error_Er / xp.max(xp.abs(Er_exact))
    rel_err_Etheta = error_Etheta / xp.max(xp.abs(Etheta_exact))
    rel_err_Bz = error_Bz / xp.max(xp.abs(Bz_exact))

    print(f"{rel_err_Er = }")
    print(f"{rel_err_Etheta = }")
    print(f"{rel_err_Bz = }")

    assert rel_err_Bz < 0.0021, f"{rank = }: Assertion for magnetic field Maxwell failed: {rel_err_Bz = }"
    print(f"{rank = }: Assertion for magnetic field Maxwell passed ({rel_err_Bz = }).")
    assert rel_err_Etheta < 0.0021, (
        f"{rank = }: Assertion for electric (E_theta) field Maxwell failed: {rel_err_Etheta = }"
    )
    print(f"{rank = }: Assertion for electric field Maxwell passed ({rel_err_Etheta = }).")
    assert rel_err_Er < 0.0021, f"{rank = }: Assertion for electric (E_r) field Maxwell failed: {rel_err_Er = }"
    print(f"{rank = }: Assertion for electric field Maxwell passed ({rel_err_Er = }).")


if __name__ == "__main__":
    libpath = struphy.__path__[0]
    # model_name = "LinearVlasovAmpereOneSpecies"
    model_name = "Maxwell"
    path_out = os.path.join(libpath, "io", "out", "verification", model_name, "1")
    # LinearVlasovAmpereOneSpecies_weakLandau(path_out, 0, show_plots=True)
    Maxwell_coaxial(path_out, 0, show_plots=True)
