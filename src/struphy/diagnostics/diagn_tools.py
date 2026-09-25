#!/usr/bin/env python3
"""Spectral diagnostics for labeled xarray output."""

import logging

import cunumpy as xp
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import xarray as xr
from scipy.fft import fftfreq, fftn
from scipy.signal import argrelextrema

from struphy.dispersion_relations import analytic

logger = logging.getLogger("struphy")

def power_spectrum_2d(
    field: xr.DataArray,
    component: int = 0,
    slice_at: tuple = (None, 0, 0),
    physical: bool = False,
    do_plot: bool = False,
    disp_name: str = None,
    disp_params: dict = {},
    fit_branches: int = 0,
    noise_level: float = 0.1,
    extr_order: int = 10,
    fit_degree: tuple = (1,),
    save_plot: bool = False,
    save_name: str = None,
    file_format: str = "png",
):
    """Perform fft in space-time, (t, x) -> (omega, k), where x can be a logical or physical coordinate.
    Returns values if plot=False.

    Parameters
    ----------
    field : xarray.DataArray
        An evaluated FEEC field of a :class:`~struphy.Output`, with dims ``(t, [component,] e1, e2, e3)``,
        e.g. ``run.fields.em_fields.e_field_log``. Its time coordinate must be uniform; use
        ``run.with_time_units("normalized")`` to compare with normalized dispersion relations.

    component : int
        Which component of the field to consider; ignored for fields without a component dimension.

    slice_at : 3-tuple
        At which indices i, j the 1d slice data (t, eta)_(i, j) should be obtained.
        One entry must be "None"; this is the direction of the fft.
        Default: [None, 0, 0] performs the eta1-fft at (eta2[0], eta3[0]).

    physical : boolean
        Perform the fft on the physical coordinate (X, Y or Z) along the fft direction instead of
        on the logical one. The field must carry physical coordinates.

    do_plot : boolean
        Plot result if True, otherwise return things.

    disp_name : str
        The name of the dispersion relation class in struphy.dispersion_relations.analytic to be used for analytic
        comparison. If None, only the computed spectrum is drawn.

    disp_params : dict
        Parameters needed for analytical dispersion relation, see struphy.dispersion_relations.analytic.

    fit_branches: int
        How many branches to fit in the dispersion relation.
        Default=0 means no fits are made.

    noise_level: float
        Sets the threshold above which local maxima in the power spectrum are taken into account.
        Computed as threshold = max(spectrum) * noise_level.

    extr_oder: int
        Order given to argrelextrema.

    fit_degree: tuple[int]
        Degree of fitting polynomial for each branch (fit_branches) of power spectrum.

    save_plot : boolean
        Save figure if True. Then a path has to be given.

    save_name : str
        Name under which the plot of the result should be saved.

    file_format : str
        Type of file which the plot of the result should be saved.

    Returns
    -------
    omega : xp.array
        1d array of angular frequency.

    kvec : xp.array
        1d array of wave vector.

    dispersion : xp.array
        2d array of shape (omega.size, kvec.size) holding the fft.

    coeffs : list[list]
        List of fitting coefficients (lenght is fit_branches).
    """
    assert list(slice_at).count(None) == 1, 'Exactly one entry of slice_at must be "None".'
    name = str(field.name)
    if "component" in field.dims:
        field = field.isel(component=component)

    # extract 2d data (t, eta) for fft
    axis = list(slice_at).index(None)
    along = ("e1", "e2", "e3")[axis]
    fixed = {dim: index for dim, index in zip(("e1", "e2", "e3"), slice_at) if index is not None}
    sliced = field.isel(fixed).transpose("t", along)
    data = xp.asarray(sliced)

    # check uniform grid in time
    time = xp.asarray(sliced.t)
    dt = time[1] - time[0]
    assert xp.allclose(time[1:] - time[:-1], dt, rtol=0.0, atol=1e-12 * max(1.0, abs(dt))), "time grid is not uniform"

    if physical:
        grid = xp.asarray(sliced[("X", "Y", "Z")[axis]])
    else:
        grid = xp.asarray(sliced[along])

    # extract uniform grid in space
    Nt = data.shape[0]
    Nx = grid.size
    dx = grid[1] - grid[0]
    assert xp.allclose(grid[1:] - grid[:-1], dx * xp.ones_like(grid[:-1]))

    dispersion = (2.0 / Nt) * (2.0 / Nx) * xp.abs(fftn(data))[: Nt // 2, : Nx // 2]
    kvec = 2 * xp.pi * fftfreq(Nx, dx)[: Nx // 2]
    omega = 2 * xp.pi * fftfreq(Nt, dt)[: Nt // 2]

    coeffs = None
    if fit_branches > 0:
        assert len(fit_degree) == fit_branches
        # determine maxima for each k
        k_start = kvec.size // 8  # take only first half of k-vector
        k_end = kvec.size // 2  # take only first half of k-vector
        k_fit = []
        omega_fit = {}
        for n in range(fit_branches):
            omega_fit[n] = []
        for k, f_of_omega in zip(kvec[k_start:k_end], dispersion[:, k_start:k_end].T):
            threshold = xp.max(f_of_omega) * noise_level
            extrms = argrelextrema(f_of_omega, xp.greater, order=extr_order)[0]
            above_noise = xp.nonzero(f_of_omega > threshold)[0]
            intersec = list(set(extrms) & set(above_noise))
            # intersec = list(set(extrms))
            if not intersec:
                continue
            intersec.sort()
            # logger.info(f"{intersec = }")
            # logger.info(f"{[omega[intersec[n]] for n in range(fit_branches)]}")
            assert len(intersec) == fit_branches, (
                f"Number of found branches {len(intersec)} is not {fit_branches =}! \
                Try to lower 'noise_level' or increase 'extr_order'."
            )
            k_fit += [k]
            for n in range(fit_branches):
                omega_fit[n] += [omega[intersec[n]]]

        # fit
        coeffs = []
        for m, om in omega_fit.items():
            coeffs += [xp.polyfit(k_fit, om, deg=fit_degree[n])]
        logger.info(f"\nFitted {coeffs =}")

    if do_plot:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
        colormap = "plasma"
        K, W = xp.meshgrid(kvec, omega)
        lvls = xp.logspace(-15, -1, 27)
        disp_plot = ax.contourf(
            K,
            W,
            dispersion**2 / (dispersion**2).max(),
            cmap=colormap,
            norm=colors.LogNorm(),
            levels=lvls,
        )
        plt.colorbar(
            ticks=[1e-12, 1e-9, 1e-6, 1e-3],
            mappable=disp_plot,
            format="%.0e",
        )
        title = name + ", component " + str(component + 1)
        ax.set_title(title)
        ax.set_xlabel("$k$ [a.u.]")
        ax.set_ylabel(r"$\omega$ [a.u.]")

        if fit_branches > 0:
            for n, cs in enumerate(coeffs):

                def fun(k):
                    out = k * 0.0
                    for i, c in enumerate(xp.flip(cs)):
                        out += c * k**i
                    return out

                ax.plot(kvec, fun(kvec), "r:", label=f"fit_{n + 1}")

        # analytic solution, when a dispersion relation is given
        set_min = set_max = 0.0
        if disp_name is not None:
            disp = getattr(analytic, disp_name)(**disp_params)

            branches = disp(kvec)
            for key, branch in branches.items():
                vals = xp.real(branch)
                ax.plot(kvec, vals, "--", label=key)
                set_min = min(set_min, xp.min(vals))
                set_max = max(set_max, xp.max(vals))
        else:
            set_min, set_max = 0.0, omega[-1]

        ax.legend()
        ax.set_xlim(0, kvec[-1])
        ax.set_ylim(set_min * 1.1, set_max * 1.1)

        if save_plot:
            assert save_name is not None, "When wanting to save the plot a path has to be given!"
            plt.savefig(save_name + "." + file_format)
        else:
            plt.show()

    return omega, kvec, dispersion, coeffs

