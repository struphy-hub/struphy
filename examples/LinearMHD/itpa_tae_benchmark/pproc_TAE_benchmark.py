"""Plots and movies for the ITPA TAE benchmark runs (LinearMHD).

Choose the run and the plots in ``main()`` and run ``python pproc_TAE_benchmark.py``.
Figures are saved to ``<run folder>/plots/``.
"""

import importlib.util
import os

import h5py
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# field key -> (species, variable, label); vector fields have components "r", "pol", "tor" (logical directions)
FIELDS = {
    "v": ("mhd", "velocity", "U"),
    "b": ("em_fields", "b_field", "B"),
    "density": ("mhd", "density", "n"),
    "pressure": ("mhd", "pressure", "p"),
}
COMPONENTS = {"r": 0, "pol": 1, "tor": 2}


def load_sim(sim_folder, run_pproc=False):
    """Load the Simulation of a run folder (from its own parameters.py) and its post-processed data.

    Set ``run_pproc=True`` the first time, to evaluate the fields on a grid (writes ``post_processing/``).
    """
    spec = importlib.util.spec_from_file_location("params", os.path.join(HERE, sim_folder, "parameters.py"))
    params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(params)
    sim = params.sim
    if run_pproc:
        sim.pproc()
    sim.load_plotting_data()
    return sim


def get_field(sim, field, comp, t):
    """Values of one field component at time ``t``, shape (n_r, n_theta, n_phi), and its label."""
    species, var, label = FIELDS[field]
    comps = getattr(getattr(sim.spline_values, species), var + "_log").data[t]
    if len(comps) == 1:  # scalar field
        return comps[0], label
    return comps[COMPONENTS[comp]], f"{label}_{comp}"


def minor_radius(sim):
    """Minor radius along the radial grid direction."""
    x, y, z = (g[:, 0, 0] for g in sim.grids_phy)
    return np.sqrt((np.sqrt(x**2 + y**2) - sim.domain.params["R0"]) ** 2 + z**2)


def out_path(sim, name):
    """Path ``<run folder>/plots/name`` (folders are created)."""
    path = os.path.join(sim.env.path_out, "plots", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save(fig, sim, name):
    """Save a figure to ``<run folder>/plots/name`` and close it."""
    path = out_path(sim, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def draw_snapshot(ax_pol, ax_top, sim, arr, phi_idx, levels):
    """Draw ``arr`` on a poloidal cross-section (at toroidal index ``phi_idx``) and on the midplane Z=0."""
    x, y, z = sim.grids_phy
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


def plot_field_vs_r(sim, field, comp, t_indices, theta_idx=0, phi_idx=4):
    """Radial profile of one field component at fixed (theta, phi) grid index, one curve per time."""
    r = minor_radius(sim)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for n in t_indices:
        t = sim.t_grid[n]
        arr, label = get_field(sim, field, comp, t)
        ax.plot(r, arr[:, theta_idx, phi_idx], label=f"t={t:.2f}")
    ax.set(xlabel="minor radius r", ylabel=label, title=f"{label} vs r (theta idx {theta_idx}, phi idx {phi_idx})")
    ax.legend()
    save(fig, sim, f"{label}_vs_r.png")


def plot_snapshot(sim, field, comp, t_index, phi_idx=4):
    """Poloidal cross-section and midplane view of one field component at one time."""
    t = sim.t_grid[t_index]
    arr, label = get_field(sim, field, comp, t)
    vmax = np.abs(arr).max() or 1.0
    fig, (ax_pol, ax_top) = plt.subplots(2, 1, figsize=(7, 12), layout="constrained")
    im = draw_snapshot(ax_pol, ax_top, sim, arr, phi_idx, np.linspace(-vmax, vmax, 51))
    ax_pol.set_title(f"{label} at phi idx {phi_idx}, t={t:.2f}")
    ax_top.set_title(f"{label} at Z=0, t={t:.2f}")
    fig.colorbar(im, ax=[ax_pol, ax_top])
    save(fig, sim, f"{label}_snapshot_t{t:.2f}.png")


def make_movie(sim, field, comp, phi_idx=4, every=1, fps=10):
    """GIF of the snapshot over time (every ``every``-th saved time), with one colour scale for all frames."""
    times = sim.t_grid[::every]
    frames = [get_field(sim, field, comp, t)[0] for t in times]
    label = get_field(sim, field, comp, times[0])[1]
    vmax = np.percentile(np.abs(frames), 99) or 1.0  # clip rare extremes
    levels = np.linspace(-vmax, vmax, 51)

    fig, (ax_pol, ax_top) = plt.subplots(2, 1, figsize=(7, 12), layout="constrained")
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax)), ax=[ax_pol, ax_top])

    def draw(i):
        ax_pol.clear()
        ax_top.clear()
        draw_snapshot(ax_pol, ax_top, sim, frames[i], phi_idx, levels)
        ax_pol.set_title(f"{label}, t={times[i]:.2f}")

    path = out_path(sim, f"{label}_movie.gif")
    animation.FuncAnimation(fig, draw, frames=len(frames)).save(path, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved {path}")


def plot_energies(sim, keys=("en_U", "en_B", "en_thermal", "en_tot")):
    """Energy scalars saved during the run, and the relative drift of en_tot (should stay ~flat)."""
    with h5py.File(os.path.join(sim.env.path_out, "data", "data_proc0.hdf5"), "r") as f:
        t = f["time/value"][:]
        en = {k: f["scalar/" + k][:] for k in set(keys) | {"en_tot"}}

    fig, (ax, ax_drift) = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for k in keys:
        ax.plot(t, en[k], label=k, **(dict(color="k", lw=2) if k == "en_tot" else {}))
    ax.set(xlabel="t", ylabel="energy", title="Perturbed energy components")
    ax.legend()
    ax_drift.plot(t, (en["en_tot"] - en["en_tot"][0]) / en["en_tot"][0])
    ax_drift.set(xlabel="t", ylabel="relative drift in en_tot", title="Should stay ~flat")
    save(fig, sim, "energies.png")


def plot_modes(sim, t_index, fields=(("v", "r"), ("b", "r")), m_list=(9, 10, 11, 12), n_list=(-1,)):
    """Radial profiles of the Fourier modes (m, n) at one time, one panel per field component.

    n is the logical toroidal mode number (physical n = n * tor_period).
    """
    t = sim.t_grid[t_index]
    r = minor_radius(sim)
    tor_period = sim.domain.params["tor_period"]
    fig, axs = plt.subplots(1, len(fields), figsize=(6 * len(fields), 5), squeeze=False, layout="constrained")
    for ax, (field, comp) in zip(axs[0], fields):
        arr, label = get_field(sim, field, comp, t)
        # 2D FFT over (theta, phi); the last grid point equals the first (periodic), so it is dropped
        amps = np.abs(np.fft.fft2(arr[:, :-1, :-1], axes=(1, 2)))
        norm = max(amps[:, m, n].max() for m in m_list for n in n_list) or 1.0
        for i, (m, n) in enumerate((m, n) for m in m_list for n in n_list):
            ax.plot(r, amps[:, m, n] / norm, ls=("-", "--", "-.", ":")[i % 4], label=f"m={m}, n={n * tor_period}")
        ax.set(xlabel="minor radius r", ylabel="normalised amplitude", title=f"{label}, t={t:.2f}")
        ax.legend(fontsize=8)
    save(fig, sim, f"modes/modes_t{t:.2f}.png")


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

    sim = load_sim(sim_folder, run_pproc)

    if do_field_vs_r:
        plot_field_vs_r(sim, field, comp, t_indices_vs_r, phi_idx=phi_idx)
    if do_snapshot:
        plot_snapshot(sim, field, comp, t_index_snapshot, phi_idx)
    if do_energies:
        plot_energies(sim)
    if do_modes:
        for n in t_indices_modes:
            plot_modes(sim, n)
    if do_movie:
        make_movie(sim, movie_field, movie_comp, phi_idx, every=movie_every)


if __name__ == "__main__":
    main()
