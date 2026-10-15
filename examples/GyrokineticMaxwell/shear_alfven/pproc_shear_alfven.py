"""Post-process and plot the shear Alfvén waves of GyrokineticMaxwell with plasma-plots.

Run as ``python pproc_shear_alfven.py [flags of params_shear_alfven.py] [--save-results]``; the flags must be those
of the run. The figures:

* ``energies``: the energy budget (fields, ions, electrons), the relative drift of the total energy and the energy
  exchanged between fields and particles (``out.plot.energies``);
* ``space_time``: :math:`B_y(z, t)` and :math:`E_x(z, t)`, the standing waves as space-time maps;
* ``dispersion``: the :math:`(k, \\omega)` power spectrum of :math:`B_y(z, t)` with the shear Alfvén branch
  :math:`\\omega = k_\\parallel / \\sqrt{1 + \\sum_s \\alpha_s^2 n_0 / B_0^2}` of the model (light units);
* ``frequencies``: the measured frequency of each seeded mode against that branch, with the relative error;
* ``orbits``: the paths of the saved markers, streaming along the perturbed field lines (magnetic flutter);
* for a run with ``--oblique`` (3D, :math:`k_\\perp \\neq 0`): PyVista views ``slices_3d``, ``isosurface_3d`` of
  :math:`B_y`, ``field_lines_3d`` of :math:`\\mathbf B_0 + \\tilde{\\mathbf B}`, and the movie ``slices_3d.gif``;
* ``waves.gif``: an animation of the :math:`B_y` and :math:`E_x` profiles with the field energies alongside.

See Remmerswaal, Hatzky, Sonnendrücker, J. Plasma Phys. 91, E127 (2025) for the model.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import plasma_plots
import xarray as xr
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent

# The setup of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
import params_shear_alfven as params  # noqa: E402

DEFAULT_OUTPUT = Path(params.args.out_folder) / params.args.sim_folder

ENERGIES = ("en_E", "en_B", "en_ions", "en_electrons")
GROUPS = {"fields": ["en_E", "en_B"], "ions": ["en_ions"], "electrons": ["en_electrons"]}


def alfven_branch():
    """The shear Alfvén branch omega(k) of the run, k per unit eta3 (light units, k_par = k / L_z)."""
    model = params.model
    polarization = sum(
        species.var.species.equation_params.alpha**2 for species in (model.kinetic_ions, model.kinetic_electrons)
    )
    n0_over_B02 = params.equil.params["n0"] / params.equil.params["B0z"] ** 2
    # the Darwin model drops the vacuum displacement current (quasi-neutrality)
    vacuum = 0.0 if params.args.model == "darwin" else 1.0
    v_alfven = 1.0 / np.sqrt(vacuum + polarization * n0_over_B02)
    L_z = params.domain.params["r3"]
    return lambda k: np.abs(k) / L_z * v_alfven


def fit_cosine_frequency(time, values, omega_guess):
    """Angular frequency of ``values(time)`` from a least-squares fit of ``c0 + c1 cos(omega t + phi)``; NaN if it fails."""
    from scipy.optimize import curve_fit

    def model(t, c0, c1, omega, phi):
        return c0 + c1 * np.cos(omega * t + phi)

    try:
        popt, _ = curve_fit(model, time, values, p0=(0.0, np.max(np.abs(values)), omega_guess, 0.0))
    except RuntimeError:
        return np.nan
    return abs(popt[2])


def main(path_out=DEFAULT_OUTPUT, save_results=False):
    run = Output(path_out).pproc().with_time_units("normalized")
    results_dir = run.path_out / "results"

    def finish(result, name):
        """Save the figure as ``<name>.png`` into the results folder (``save_results``), or keep it for ``plt.show()``."""
        if save_results:
            results_dir.mkdir(exist_ok=True)
            result.save(results_dir / f"{name}.png")
            plt.close(result.fig)

    # energy budget: the fields gain what the particles lose (mu B_parallel through the magnetisation current)
    finish(run.plot.energies(parts=ENERGIES, groups=GROUPS), "energies")
    en_tot = run.evaluate("scalars", variables="en_tot")["en_tot"].values
    print(f"Relative energy error at the end: {(en_tot[-1] - en_tot[0]) / en_tot[0]:.3e}")

    # the waves along z (eta3) at the first x, y point, as (eta3, t) arrays
    b_y = run.evaluate("em_fields/b_field", component=1).isel(eta1=0, eta2=0)
    e_x = run.evaluate("em_fields/e_field", component=0).isel(eta1=0, eta2=0)
    b_y.name, e_x.name = "B_y", "E_x"
    b_y.attrs["label"], e_x.attrs["label"] = r"$B_y$", r"$E_x$"

    # space-time maps of the standing waves
    with plasma_plots.figure(1, 2, figsize=(12, 4.5), title="Standing shear Alfvén waves") as fig:
        b_y.plasma.plot.slice(x="eta3", y="t", symmetric=True, cmap="RdBu_r", ax=fig[0])
        e_x.plasma.plot.slice(x="eta3", y="t", symmetric=True, cmap="RdBu_r", ax=fig[1])
    finish(fig, "space_time")

    omega_of_k = alfven_branch()
    # the same branch as the Crank-Nicolson field solver (dt of the run) reproduces it: omega_CN = 2/dt atan(omega dt/2)
    dt = params.args.dt
    omega_cn_of_k = lambda k: 2.0 / dt * np.arctan(omega_of_k(k) * dt / 2.0)  # noqa: E731
    n_times = b_y.sizes["t"]
    k_max = 2 * np.pi * (max(params.args.modes) + 1)
    if n_times >= 8:
        # dispersion relation: the power of B_y(z, t) over (k, omega) with the shear Alfvén branch of the model
        omega_max = 1.5 * omega_of_k(k_max)
        finish(
            b_y.plasma.plot.dispersion(
                dim="eta3",
                branches={"shear Alfvén branch": omega_of_k, f"Crank-Nicolson, dt = {dt:g}": omega_cn_of_k},
                kmin=0.0,
                kmax=k_max,
                omega_max=omega_max,
                title=r"Dispersion of $B_y(z, t)$",
            ),
            "dispersion",
        )

        # the frequency of each seeded mode: a cosine fit of its Fourier coefficient along z (a standing wave),
        # more precise than a frequency bin of the spectrum, against the branch
        coefficients = np.fft.rfft(b_y.transpose("t", "eta3").values, axis=1) / b_y.sizes["eta3"]
        modes = np.asarray(params.args.modes, dtype=float)
        omegas = [
            fit_cosine_frequency(b_y.t.values, coefficients[:, n].real, omega_cn_of_k(2 * np.pi * n))
            for n in params.args.modes
        ]
        omega = xr.DataArray(
            omegas,
            dims="k",
            coords={"k": 2 * np.pi * modes},
            name="omega",
            attrs={"label": r"$\omega$"},
        ).dropna("k")
        if omega.sizes["k"] > 0:
            finish(
                omega.plasma.plot.against_theory(
                    omega_cn_of_k,
                    xlabel=r"$k$ (per unit $\eta_3$)",
                    ylabel=r"$\omega$",
                    title=f"Shear Alfvén frequencies: measured vs. model (Crank-Nicolson, dt = {dt:g})",
                ),
                "frequencies",
            )
            for k, w in zip(omega.k.values, omega.values):
                w_th, w_cn = omega_of_k(k), omega_cn_of_k(k)
                print(
                    f"n = {k / (2 * np.pi):.0f}: omega = {w:.5f}, model {w_th:.5f} ({w / w_th - 1:+.3f}), "
                    f"Crank-Nicolson {w_cn:.5f} ({w / w_cn - 1:+.3f})"
                )
    else:
        print(f"Only {n_times} saved times: no dispersion plots (run longer).")

    # 3D views (PyVista) of a run with more than one cell in x and y (``--oblique``): slices of B_y through the
    # slab, its isosurfaces, the field lines of B_0 + B_tilde, and a movie of the slices
    b_field = run.evaluate("em_fields/b_field")
    if b_field.sizes["eta1"] > 1 and b_field.sizes["eta2"] > 1:
        import pyvista as pv

        pv.OFF_SCREEN = save_results
        b_y_3d = b_field.sel(component=1)
        b_y_3d.name = "B_y"
        b_y_3d.attrs["label"] = r"$B_y$"
        b_full = b_field.copy()
        b_full[dict(component=2)] = b_full.sel(component=2) + params.equil.params["B0z"]
        b_full.name = "B"
        cuts = {"eta3": [0.0, 0.25, 0.5, 0.75], "eta1": 0.0}
        # the snapshot: the time of the largest magnetic energy after the first quarter period of the n = 1 mode
        # (the oblique waves are damped, so the end of the run is mostly noise)
        en_B = run.evaluate("scalars", variables="en_B")["en_B"]
        t_quarter = 0.5 * np.pi / omega_of_k(2 * np.pi * min(params.args.modes))
        later = en_B.where(en_B.t >= t_quarter, drop=True)
        t_snap = float(later.t[int(np.argmax(later.values))]) if later.sizes["t"] > 0 else float(en_B.t[-1])
        b_snap = b_y_3d.sel(t=t_snap, method="nearest")
        level = 0.3 * float(np.abs(b_snap).max())
        views = {
            "slices_3d": b_y_3d.plasma.plot.slices_3d(cuts=cuts, cmap="RdBu_r", symmetric=True, t=t_snap),
            "isosurface_3d": b_y_3d.plasma.plot.isosurface(
                values=[-level, level], cmap="RdBu_r", symmetric=True, t=t_snap
            ),
            "field_lines_3d": b_full.plasma.plot.streamlines(n_points=40, cmap="plasma", t=t_snap),
        }
        print(f"3D views at t = {t_snap:g}")
        for name, plotter in views.items():
            if save_results:
                results_dir.mkdir(exist_ok=True)
                plotter.screenshot(results_dir / f"{name}.png", window_size=(1000, 900))
                plotter.close()
            else:
                plotter.show()
        # movie of the slices over the first two periods of the n = 1 mode
        if save_results and n_times >= 2:
            first_periods = b_y_3d.where(b_y_3d.t <= 8 * t_quarter, drop=True)
            step = max(1, first_periods.sizes["t"] // 40)
            first_periods.plasma.plot.movie(
                results_dir / "slices_3d.gif", kind="slices", step=step, cuts=cuts, cmap="RdBu_r", symmetric=True
            )

    # saved marker orbits: streaming along the perturbed field lines
    for species in ("kinetic_ions", "kinetic_electrons"):
        try:
            orbits = run.evaluate(f"{species}/orbits")
        except (KeyError, ValueError):
            continue
        finish(orbits.plasma.plot.trajectories(), f"orbits_{species}")

    # animation of the profiles with the field energies alongside (saved as GIF, or shown interactively)
    if n_times >= 2 and (save_results or matplotlib.get_backend().lower() != "agg"):
        en_E = run.evaluate("scalars", variables="en_E")["en_E"]
        en_B = run.evaluate("scalars", variables="en_B")["en_B"]
        animation = b_y.plasma.plot.line_animation(
            x="eta3",
            alongside=[e_x, [en_E, en_B]],
            max_frames=120,
            interval=60,
            title=r"$B_y$ along $z$",
        )
        if save_results:
            results_dir.mkdir(exist_ok=True)
            animation.save(results_dir / "waves.gif", writer="pillow")
            plt.close("all")

    if not save_results:
        plt.show()


if __name__ == "__main__":
    main(save_results="--save-results" in sys.argv)
