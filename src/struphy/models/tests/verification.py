import os

import h5py
import numpy as np
import yaml
from matplotlib import pyplot as plt

import struphy


def VlasovAmpereOneSpecies_weakLandau(path_out: str, rank: int, show_plots: bool = False):
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
        return 2 * eps**2 * np.pi / k**2 * r**2 * np.exp(2 * gamma * t) * np.cos(omega * t - phi) ** 2

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
    logE = np.log10(E)

    # find where time derivative of E is zero
    dEdt = (np.roll(logE, -1) - np.roll(logE, 1))[1:-1] / (2.0 * dt)
    zeros = dEdt * np.roll(dEdt, -1) < 0.0
    maxima_inds = np.logical_and(zeros, dEdt > 0.0)
    maxima = logE[1:-1][maxima_inds]
    t_maxima = time[1:-1][maxima_inds]

    # linear fit
    linfit = np.polyfit(t_maxima[:5], maxima[:5], 1)
    gamma_num = linfit[0]

    # plot
    if show_plots and rank == 0:
        plt.figure(figsize=(18, 12))
        plt.plot(time, logE, label="numerical")
        plt.plot(time, np.log10(E_exact(time)), label="exact")
        plt.legend()
        plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
        plt.xlabel("time [m/c]")
        plt.plot(t_maxima[:5], maxima[:5], "r")
        plt.plot(t_maxima[:5], maxima[:5], "or", markersize=10)
        plt.ylim([-10, -4])

        plt.show()

    # assert
    rel_error = np.abs(gamma_num - gamma) / np.abs(gamma)
    assert rel_error < 0.25, f"{rank = }: Assertion for weak Landau damping failed: {gamma_num = } vs. {gamma = }."
    print(f"{rank = }: Assertion for weak Landau damping passed ({rel_error = }).")


def LinearVlasovAmpereOneSpecies_weakLandau(path_out, rank, show_plots=False):
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
        return 2 * eps**2 * np.pi / k**2 * r**2 * np.exp(2 * gamma * t) * np.cos(omega * t - phi) ** 2

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
    logE = np.log10(E)

    # find where time derivative of E is zero
    dEdt = (np.roll(logE, -1) - np.roll(logE, 1))[1:-1] / (2.0 * dt)
    zeros = dEdt * np.roll(dEdt, -1) < 0.0
    maxima_inds = np.logical_and(zeros, dEdt > 0.0)
    maxima = logE[1:-1][maxima_inds]
    t_maxima = time[1:-1][maxima_inds]

    # linear fit
    linfit = np.polyfit(t_maxima[:5], maxima[:5], 1)
    gamma_num = linfit[0]

    # plot
    if show_plots and rank == 0:
        plt.figure(figsize=(18, 12))
        plt.plot(time, logE, label="numerical")
        plt.plot(time, np.log10(E_exact(time)), label="exact")
        plt.legend()
        plt.title(f"{dt=}, {algo=}, {Nel=}, {p=}, {ppc=}")
        plt.xlabel("time [m/c]")
        plt.plot(t_maxima[:5], maxima[:5], "r")
        plt.plot(t_maxima[:5], maxima[:5], "or", markersize=10)
        plt.ylim([-10, -4])

        plt.show()

    # assert
    rel_error = np.abs(gamma_num - gamma) / np.abs(gamma)
    assert rel_error < 0.25, f"{rank = }: Assertion for weak Landau damping failed: {gamma_num = } vs. {gamma = }."
    print(f"{rank = }: Assertion for weak Landau damping passed ({rel_error = }).")


if __name__ == "__main__":
    libpath = struphy.__path__[0]
    model_name = "LinearVlasovAmpereOneSpecies"
    path_out = os.path.join(libpath, "io", "out", "verification", model_name, "1")
    LinearVlasovAmpereOneSpecies_weakLandau(path_out, 0, show_plot=True)
