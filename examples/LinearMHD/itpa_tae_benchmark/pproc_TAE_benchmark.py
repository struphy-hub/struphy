"""Plots and movies for the ITPA TAE benchmark runs (LinearMHD).

Choose the run and the plots in ``main()`` and run ``python pproc_TAE_benchmark.py``.
Figures are saved to ``<run folder>/plots/``.
"""

import os

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from plasma_plots.spectral import filter_time, mode_spectrum

from struphy import Output

HERE = os.path.dirname(os.path.abspath(__file__))

# field key -> (species, variable, label); vector fields have components "r", "pol", "tor" (logical directions)
FIELDS = {
    "v": ("mhd", "velocity", "U"),
    "b": ("em_fields", "b_field", "B"),
    "density": ("mhd", "density", "n"),
    "pressure": ("mhd", "pressure", "p"),
}
COMPONENTS = {"r": 0, "pol": 1, "tor": 2}


def load_output(sim_folder, run_pproc=False):
    """Open saved run metadata without executing its parameter file."""
    out = Output(os.path.join(HERE, sim_folder))
    if run_pproc or not out.is_processed:
        out.pproc(physical=True, force=run_pproc)
    return out


def field_series(out, field, *, t=None):
    """Evaluate a field in its original FEEC representation on the plotting grid."""
    species, var, _ = FIELDS[field]
    representation = "2" if field in ("v", "b") else "3"
    return out.evaluate(
        f"{species}/{var}",
        t=t,
        representation=representation,
        **dict(zip(("eta1", "eta2", "eta3"), out.grids_log)),
    )


def filter_field(out, field, *, pad_bins=0, omega_min=1e-8):
    """Return plasma-plots' filtered field and dominant-band spectrum.

    Filtering is performed per component over the saved time coordinate. It does
    not modify the run or its stored post-processing products.
    """
    return filter_time(field_series(out, field), pad_bins=pad_bins, omega_min=omega_min)


def get_field(out, field, comp, t):
    """One labeled field component at saved time ``t``."""
    data = field_series(out, field, t=float(t)).isel(t=0, drop=True)
    label = FIELDS[field][2]
    if "component" in data.dims:
        data = data.isel(component=COMPONENTS[comp], drop=True)
        label = f"{label}_{comp}"
    return data, label


def minor_radius(out):
    """Minor radius along the radial grid direction."""
    x, y, z = (g[:, 0, 0] for g in out.grids_phy)
    return np.sqrt((np.sqrt(x**2 + y**2) - out.domain.params["R0"]) ** 2 + z**2)


def out_path(out, name):
    """Path ``<run folder>/plots/name`` (folders are created)."""
    path = os.path.join(out.path_out, "plots", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save(fig, out, name):
    """Save a figure to ``<run folder>/plots/name`` and close it."""
    path = out_path(out, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def draw_snapshot(ax_pol, ax_top, out, arr, phi_idx, levels):
    """Draw ``arr`` on a poloidal cross-section (at toroidal index ``phi_idx``) and on the midplane Z=0."""
    x, y, z = out.grids_phy
    arr = np.asarray(arr)
    R = np.sqrt(x[:, :, phi_idx] ** 2 + y[:, :, phi_idx] ** 2)
    kw = dict(levels=levels, cmap="RdBu_r", extend="both")

    im = ax_pol.contourf(R, z[:, :, phi_idx], arr[:, :, phi_idx], **kw)
    ax_pol.plot(R[-1], z[-1, :, phi_idx], "k")
    ax_pol.set(aspect="equal", xlabel="R", ylabel="Z")

    # midplane: outboard (theta=0) and inboard (theta=pi) side of the torus sector
    for th in (0, (x.shape[1] - 1) // 2):
        ax_top.contourf(x[:, th, :], y[:, th, :], arr[:, th, :], **kw)
    ax_top.set(aspect="equal", xlabel="x", ylabel="y")
    return im


def plot_field_vs_r(out, field, comp, t_indices, theta_idx=0, phi_idx=4):
    """Radial profile of one field component at fixed (theta, phi) grid index, one curve per time."""
    r = minor_radius(out)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for n in t_indices:
        t = out.time[n]
        arr, label = get_field(out, field, comp, t)
        ax.plot(r, arr[:, theta_idx, phi_idx], label=f"t={t:.2f}")
    ax.set(xlabel="minor radius r", ylabel=label, title=f"{label} vs r (theta idx {theta_idx}, phi idx {phi_idx})")
    ax.legend()
    save(fig, out, f"{label}_vs_r.png")


def plot_snapshot(out, field, comp, t_index, phi_idx=4):
    """Poloidal cross-section and midplane view of one field component at one time."""
    t = out.time[t_index]
    arr, label = get_field(out, field, comp, t)
    vmax = np.abs(arr).max() or 1.0
    fig, (ax_pol, ax_top) = plt.subplots(2, 1, figsize=(7, 12), layout="constrained")
    im = draw_snapshot(ax_pol, ax_top, out, arr, phi_idx, np.linspace(-vmax, vmax, 51))
    ax_pol.set_title(f"{label} at phi idx {phi_idx}, t={t:.2f}")
    ax_top.set_title(f"{label} at Z=0, t={t:.2f}")
    fig.colorbar(im, ax=[ax_pol, ax_top])
    save(fig, out, f"{label}_snapshot_t{t:.2f}.png")


def make_movie(out, field, comp, phi_idx=4, every=1, fps=10):
    """GIF of the snapshot over time (every ``every``-th saved time), with one colour scale for all frames."""
    times = out.time[::every]
    frames = [get_field(out, field, comp, t)[0] for t in times]
    label = get_field(out, field, comp, times[0])[1]
    vmax = np.percentile(np.abs(frames), 99) or 1.0  # clip rare extremes
    levels = np.linspace(-vmax, vmax, 51)

    fig, (ax_pol, ax_top) = plt.subplots(2, 1, figsize=(7, 12), layout="constrained")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax)), ax=[ax_pol, ax_top])

    def draw(i):
        ax_pol.clear()
        ax_top.clear()
        draw_snapshot(ax_pol, ax_top, out, frames[i], phi_idx, levels)
        ax_pol.set_title(f"{label}, t={times[i]:.2f}")

    path = out_path(out, f"{label}_movie.gif")
    animation.FuncAnimation(fig, draw, frames=len(frames)).save(path, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved {path}")


def plot_energies(out, keys=("en_U", "en_B", "en_thermal", "en_tot")):
    """Energy scalars saved during the run, and the relative drift of en_tot (should stay ~flat)."""
    scalars = out.scalars
    t = scalars.t.values
    en = {k: scalars[k].values for k in set(keys) | {"en_tot"}}

    fig, (ax, ax_drift) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for k in keys:
        ax.plot(t, en[k], label=k, **(dict(color="k", lw=2) if k == "en_tot" else {}))
    ax.set(xlabel="t", ylabel="energy", title="Perturbed energy components")
    ax.legend()
    ax_drift.plot(t, (en["en_tot"] - en["en_tot"][0]) / en["en_tot"][0])
    ax_drift.set(xlabel="t", ylabel="relative drift in en_tot", title="Should stay ~flat")
    save(fig, out, "energies.png")


def plot_modes(out, t_index, fields=(("v", "r"), ("b", "r")), m_list=(9, 10, 11, 12), n_list=(-1,)):
    """Radial profiles of the Fourier modes (m, n) at one time, one panel per field component.

    n is the logical toroidal mode number (physical n = n * tor_period).
    """
    t = out.time[t_index]
    r = minor_radius(out)
    tor_period = out.domain.params["tor_period"]
    fig, axs = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5), squeeze=False, layout="constrained")
    for ax, (field, comp) in zip(axs[0], fields):
        arr, label = get_field(out, field, comp, t)
        # plasma-plots handles periodic endpoints and labels signed mode numbers.
        amps = abs(mode_spectrum(arr, dims=("eta2", "eta3"), periods=(1.0, 1.0)))
        norm = max(float(amps.sel(m=m, n=n).max()) for m in m_list for n in n_list) or 1.0
        for i, (m, n) in enumerate((m, n) for m in m_list for n in n_list):
            ax.plot(r, amps.sel(m=m, n=n) / norm, ls=("-", "--", "-.", ":")[i % 4], label=f"m={m}, n={n * tor_period}")
        ax.set(xlabel="minor radius r", ylabel="normalised amplitude", title=f"{label}, t={t:.2f}")
        ax.legend(fontsize=8)
    save(fig, out, f"modes/modes_t{t:.2f}.png")


def main():
    # ------------------------- settings -------------------------
    sim_folder = "sim4_higherResolution"
    run_pproc = False  # True the first time for a new run

    field, comp = "v", "pol"  # fields: "v", "b", "density", "pressure"; components: "r", "pol", "tor"
    phi_idx = 4  # toroidal grid index of the poloidal cross-section

    do_field_vs_r = True
    t_indices_vs_r = (0, 10, 20)

    do_snapshot = True
    t_index_snapshot = 10

    do_energies = True

    do_modes = True
    t_indices_modes = (0, 50, 100)

    do_movie = False
    movie_field, movie_comp = "v", "r"
    movie_every = 1  # use every n-th saved time as a frame
    # -------------------------------------------------------------

    out = load_output(sim_folder, run_pproc)

    if do_field_vs_r:
        plot_field_vs_r(out, field, comp, t_indices_vs_r, phi_idx=phi_idx)
    if do_snapshot:
        plot_snapshot(out, field, comp, t_index_snapshot, phi_idx)
    if do_energies:
        plot_energies(out)
    if do_modes:
        for n in t_indices_modes:
            plot_modes(out, n)
    if do_movie:
        make_movie(out, movie_field, movie_comp, phi_idx, every=movie_every)


if __name__ == "__main__":
    main()
