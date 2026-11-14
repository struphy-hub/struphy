"""Post-process the GuidingCenter tokamak_equilibrium example: the long-time equilibrium diagnostics and pictures of
the distribution function.

Run as ``python pproc_tokamak_equilibrium.py [sim_folder] [variant flags of params_tokamak_equilibrium.py]
[--no-video]``. The variant flags must be those of the run; the run parameters are taken from
``params_tokamak_equilibrium.py``, which reads them. Written into ``<sim_folder>/results``, where the profiling
packaging picks them up and uploads them with the run (``submit_tokamak_equilibrium.py`` calls this script after
each simulation):

* the diagnostics of ``struphy.post_processing.particle_diagnostics`` (binned distributions at the standard
  snapshots, their relative L2 deviation from t=0, the energy drift and the lost markers);
* ``distribution.png``: the binned distribution in the poloidal plane (R, Z), in (r, v_par) and in (v_par, mu), at
  the first and last saved time and the change f(t) - f(0), with the trapped-passing boundaries;
* ``frames/distribution.mp4`` (needs ffmpeg, skipped with ``--no-video``): the distribution in the poloidal plane
  and the change f(t) - f(0) in the poloidal plane and in (r, v_par) over time, with fixed color scales.

The change f(t) - f(0) is computed from the same markers (fixed weights), so it does not contain the Monte-Carlo
noise of the initial loading, only that of the marker motion.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from struphy import Output
from struphy.post_processing.particle_diagnostics import save_equilibrium_diagnostics

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"
SPECIES = "kinetic_ions"

# The variant flags and the profiles of the run (parsed from the same command line).
sys.path.insert(0, str(SCRIPT_DIR))
from params_tokamak_equilibrium import args as run_args  # noqa: E402
from params_tokamak_equilibrium import parser, vth_of_s  # noqa: E402

# radii (eta1 = r/a) of the trapped-passing cones in the (v_par, mu) plane
CONE_RADII = (0.3, 0.6, 0.9)
# color scales: percentile of the values (single noisy bins near the axis, where the binning divides by the
# small Jacobian, would set the scale otherwise)
PERCENTILE = 99.5


def _scale(values):
    """Color scale of the values: the PERCENTILE of their absolute values (1 if zero)."""
    return float(np.percentile(np.abs(values), PERCENTILE)) or 1.0


def _edges(centers):
    """Bin edges of uniform bins with the given centers."""
    d = centers[1] - centers[0]
    return np.concatenate([centers - d / 2, [centers[-1] + d / 2]])


def _binned(out, slice_name):
    """Binned distribution f (time first), the saved times and the bin centers of a slice, or None."""
    # the slice names carry the binned quantity (output_quantity of BinningPlot)
    key = f"{SPECIES}/{slice_name}_density/f"
    if key not in out.distribution_catalog:
        return None
    f_arr = out.distribution_catalog[key]
    dims = [dim for dim in f_arr.dims if dim != "t"]
    return f_arr.values, f_arr.coords["t"].values, [f_arr.coords[dim].values for dim in dims]


def _field_strength_range(out, eta1):
    """Minimum and maximum of |B0| on the flux surfaces eta1 (over the poloidal angle, at eta3 = 0)."""
    eta2 = np.linspace(0.0, 1.0, 145)
    absB = np.asarray(out.equil.absB0(np.asarray(eta1, dtype=float), eta2, np.array([0.0]), squeeze_out=False))
    absB = absB.reshape(len(eta1), len(eta2))
    return absB.min(axis=1), absB.max(axis=1)


def _poloidal_mesh(out, e1_centers, e2_centers):
    """Physical coordinates (R, Z) of the bin edges of the (eta1, eta2) binning in the poloidal plane eta3 = 0."""
    e1 = np.clip(_edges(e1_centers), 0.0, 1.0)
    e2 = _edges(e2_centers)
    x, y, z = (np.asarray(c).reshape(len(e1), len(e2)) for c in out.domain(e1, e2, np.array([0.0]), squeeze_out=False))
    return np.sqrt(x**2 + y**2), z


class _Panels:
    """Data and geometry of the three views (poloidal plane, (r, v_par), (v_par, mu)) of a run."""

    def __init__(self, out):
        self.pol = _binned(out, "e1_e2")
        self.rv = _binned(out, "e1_v1")
        self.vmu = _binned(out, "v1_v2")
        self.time = next(b[1] for b in (self.pol, self.rv, self.vmu) if b is not None)
        if self.pol is not None:
            self.R, self.Z = _poloidal_mesh(out, *self.pol[2])
        if self.rv is not None:
            r = self.rv[2][0]
            b_min, b_max = _field_strength_range(out, r)
            # trapped below |v_par| = sqrt(2 mu (B_max - B_min)) at the outboard midplane, for mu B_min = vth^2
            self.rv_boundary = (r, vth_of_s(r**2) * np.sqrt(2.0 * (b_max / b_min - 1.0)))
        b_min, b_max = _field_strength_range(out, np.array(CONE_RADII))
        self.cones = list(zip(CONE_RADII, 2.0 * (b_max - b_min)))

    def draw(self, ax, view, values, cmap, vmin, vmax, title):
        """Draw one view of the values (a time slice or a difference) into ax; return the image for a colorbar."""
        from matplotlib import pyplot as plt

        if view == "pol":
            im = ax.pcolormesh(self.R, self.Z, values, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
            ax.set_aspect("equal")
            ax.set_xlabel("R")
            ax.set_ylabel("Z")
        elif view == "rv":
            r, v = self.rv[2]
            im = ax.pcolormesh(r, v, values.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
            r_b, v_b = self.rv_boundary
            for sign in (1, -1):
                ax.plot(r_b, sign * v_b, "k--", lw=1.0, label="trapped-passing" if sign == 1 else None)
            ax.set_xlabel("r / a")
            ax.set_ylabel(r"$v_\parallel$")
            ax.set_ylim(v[0], v[-1])
        else:
            v, mu = self.vmu[2]
            im = ax.pcolormesh(v, mu, values.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
            colors = plt.cm.Greys(np.linspace(0.45, 0.9, len(self.cones)))
            for (radius, slope), color in zip(self.cones, colors):
                ax.plot(v, v**2 / slope, "--", color=color, lw=1.0, label=f"trapped above, r/a = {radius}")
            ax.set_xlabel(r"$v_\parallel$")
            ax.set_ylabel(r"$\mu$")
            ax.set_ylim(mu[0], mu[-1])
        ax.set_title(title, fontsize=10)
        return im


def _views(panels):
    return [(view, data) for view, data in (("pol", panels.pol), ("rv", panels.rv), ("vmu", panels.vmu)) if data]


def _row_label(view):
    return {"pol": "poloidal plane", "rv": r"$(r, v_\parallel)$", "vmu": r"$(v_\parallel, \mu)$"}[view]


def plot_distribution(out, results_dir, title):
    """Write ``distribution.png``: per view the distribution at the first and last saved time and the change."""
    from matplotlib import pyplot as plt

    panels = _Panels(out)
    views = _views(panels)
    fig, axs = plt.subplots(len(views), 3, figsize=(14, 4.2 * len(views)), squeeze=False)
    for row, (view, (f, time, _)) in zip(axs, views):
        scale = _scale(f[0])
        df = (f[-1] - f[0]) / scale
        dlim = _scale(df)
        for ax, values, cmap, vmin, vmax, panel_title in (
            (row[0], f[0] / scale, "viridis", 0.0, 1.0, f"f, t = {time[0]:.4g}"),
            (row[1], f[-1] / scale, "viridis", 0.0, 1.0, f"f, t = {time[-1]:.4g}"),
            (row[2], df, "RdBu_r", -dlim, dlim, "f(t) - f(0)"),
        ):
            im = panels.draw(ax, view, values, cmap, vmin, vmax, panel_title)
            fig.colorbar(im, ax=ax, shrink=0.85, extend="max" if vmin == 0.0 else "both")
        row[0].annotate(
            _row_label(view), xy=(-0.32, 0.5), xycoords="axes fraction", rotation=90, va="center", fontsize=11
        )
        if view != "pol":
            row[2].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"{title}\n(normalized to the {PERCENTILE}th percentile of f at t = 0; color scales clipped there)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(Path(results_dir) / "distribution.png", dpi=110)
    plt.close(fig)


def write_distribution_video(out, results_dir, title, max_frames=200):
    """Write the frames of the distribution over time into ``frames/distribution/`` and make the video."""
    from matplotlib import pyplot as plt

    from struphy.post_processing.frames import frames_to_video

    panels = _Panels(out)
    shown = [("pol", panels.pol, False), ("pol", panels.pol, True), ("rv", panels.rv, True)]
    shown = [(view, data, diff) for view, data, diff in shown if data is not None]
    if not shown:
        return None
    folder = Path(results_dir) / "frames" / "distribution"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)

    steps = np.unique(np.linspace(0, len(panels.time) - 1, min(max_frames, len(panels.time))).astype(int))
    # fixed color scales over the video
    limits = [_scale((data[0] - data[0][0]) / _scale(data[0][0])) for _, data, _ in shown]
    for k, n in enumerate(steps):
        fig, axs = plt.subplots(1, len(shown), figsize=(5.2 * len(shown), 4.6), squeeze=False)
        for ax, (view, (f, _, _), diff), dlim in zip(axs[0], shown, limits):
            scale = _scale(f[0])
            if diff:
                values, cmap, vmin, vmax, panel_title = (f[n] - f[0]) / scale, "RdBu_r", -dlim, dlim, "f(t) - f(0)"
            else:
                values, cmap, vmin, vmax, panel_title = f[n] / scale, "viridis", 0.0, 1.0, "f"
            im = panels.draw(ax, view, values, cmap, vmin, vmax, panel_title)
            fig.colorbar(im, ax=ax, shrink=0.85, extend="both" if diff else "max")
        fig.suptitle(f"{title}, t = {panels.time[n]:.4g}", fontsize=11)
        fig.tight_layout()
        # even image size for yuv420p
        fig.savefig(folder / f"frame_{k:04d}.png", dpi=100)
        plt.close(fig)
    np.savetxt(folder / "frames.txt", np.column_stack([steps, panels.time[steps]]), header="step time")
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found: frames written, no video.")
        return None
    return frames_to_video(folder, fps=10)


def main(path_out=DEFAULT_OUTPUT, video=True):
    import matplotlib

    matplotlib.use("Agg")

    out = Output(path_out)
    save_equilibrium_diagnostics(out, SPECIES, scalars=("en_tot", "n_lost_particles"))

    results_dir = Path(out.path_out) / "results"
    title = f"{run_args.init} Maxwellian" + (f" (shift={run_args.shift})" if run_args.init == "canonical" else "")
    title += rf", {run_args.algo}, $\epsilon$ = {run_args.epsilon}"
    plot_distribution(out, results_dir, title)
    if video:
        write_distribution_video(out, results_dir, title)


if __name__ == "__main__":
    # the variant flags are parsed by params_tokamak_equilibrium.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: sim_1 beside this script)",
    )
    pproc_parser.add_argument("--no-video", action="store_true", help="Skip the frames and the video.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, video=not pproc_args.no_video)
