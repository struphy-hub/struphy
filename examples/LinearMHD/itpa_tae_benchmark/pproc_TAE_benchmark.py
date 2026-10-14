"""Post-process and plot the ITPA TAE benchmark run (LinearMHD).

Run as ``python pproc_TAE_benchmark.py [sim_folder] [flags of params_TAE_benchmark.py] [--show]``. The standard
plots of the examples (slices in the physical planes and 3D views of the fields) are made at the first, middle and
last saved time, together with the energy budget and the (m, n) harmonics of the radial components. Figures are
written into ``<sim_folder>/results``. The energy and harmonics plots use plasma-plots
(``pip install "struphy[pproc]"``).
"""

import argparse
import os
import sys
from pathlib import Path

from matplotlib import pyplot as plt

from struphy import Output
from struphy.post_processing.plot_3d import plot_field_3d
from struphy.post_processing.slices import plot_physical_slices

SCRIPT_DIR = Path(__file__).resolve().parent

# The flags of the run (parsed from the same command line), for the default output folder.
sys.path.insert(0, str(SCRIPT_DIR))
from params_TAE_benchmark import args, parser  # noqa: E402

DEFAULT_OUTPUT = Path(args.out_folder) / args.sim_folder

# standard plots: (key, label, species, variable, kind) per field; kind is None for a scalar field and the
# push-forward of a vector field, which is shown by its magnitude
FIELDS = [
    ("u", "|U|", "mhd", "velocity", "2"),
    ("b", "|B|", "em_fields", "b_field", "2"),
    ("n", "n", "mhd", "density", None),
    ("p", "p", "mhd", "pressure", None),
]
# (m, n) harmonics of the radial 2-form components (radial direction eta1)
HARMONICS = [("u_r", "mhd/velocity"), ("b_r", "em_fields/b_field")]
ENERGIES = ("en_U", "en_B", "en_thermal")  # energy scalars shown next to en_tot


def minor_radius(run):
    """The minor radius of the HollowTorus as a function of eta1."""
    a1, a2 = run.domain.params["a1"], run.domain.params["a2"]
    return lambda eta1: a1 + (a2 - a1) * eta1


def finish(figs, results_dir, save_figs):
    """Save the figures to ``results_dir`` and close them (``save_figs``), or keep them open for ``plt.show()``."""
    if save_figs:
        for fig_name, fig in figs.items():
            fig.savefig(os.path.join(results_dir, f"{fig_name}.png"))
            plt.close(fig)


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    run = Output(path_out)
    snapshots = run.standard_snapshots()
    # VTK files (for the 3D views) of the standard snapshots only
    run.pproc(create_vtk=True, physical=True, vtk_steps=snapshots)
    times = [float(run.time[n]) for n in snapshots]

    # The profiling packaging picks these files up from here and uploads them with the run.
    results_dir = os.path.join(run.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    finish({"energies": run.plot.energies(parts=ENERGIES).fig}, results_dir, save_figs)
    for n, t in zip(snapshots, times):
        figs = {}
        fields = [
            (f"{key}_t{t:.2f}", f"{label}, t = {t:.3g},", f"{species}/{variable}", n, None, kind)
            for key, label, species, variable, kind in FIELDS
        ]
        figs.update(plot_physical_slices(run, run.domain, run.grid.num_elements, fields))

        for key, name in HARMONICS:
            data = run.evaluate(name, representation="2", t=n, component=0)
            data.name = key
            result = data.plasma.plot.mode_profiles(
                scale=(1, run.domain.params["tor_period"]),
                x_of=minor_radius(run),
                xlabel="minor radius r",
                top=4,
            )
            figs[f"{key}_harmonics_t{t:.2f}"] = result.fig
        # saved per snapshot, so that only the figures of one snapshot are open at a time
        finish(figs, results_dir, save_figs)

        # 3D views of the fields, saved as <field>_3d_t<time>.png
        for key, _, species, variable, _ in FIELDS:
            filename = os.path.join(results_dir, f"{key}_3d_t{t:.2f}.png")
            plot_field_3d(run.path_out, variable, species=species, snapshot=n, filename=filename)

    if not save_figs:
        plt.show()


if __name__ == "__main__":
    # the flags of the run are parsed by params_TAE_benchmark.py, the rest here
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process and plot a saved simulation run.")
    pproc_parser.add_argument(
        "path_out",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help="Simulation output folder (default: the output folder of the run flags, sim_data beside this script)",
    )
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
