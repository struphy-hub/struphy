"""Post-process and plot the diocotron instability.

Run as ``python pproc_diocotron.py [sim_1 sim_2 ...]`` to compare several runs; with
more than one folder only the growth-rate comparison is shown.
"""

import os
import sys

from struphy import open_run
from struphy.diagnostics.plotting import (
    GrowthFit,
    InteractiveSliceViewer,
    View,
    plot_marker_trajectories,
    plot_timeseries,
    plot_equilibrium_profile,
)

FIT_QUANTITY = "en_phi"
FIT_WINDOW = (0.0, 42.0)

SHOW_EQUIL_PROFILE = True

# binned densities to sweep, as (bin name, quantity, physical plane)
DENSITY_PLOTS = [
    ("e1_e2_density", "f_binned", "XY"),
    ("e1_e2_density", "delta_f_binned", "XY"),
]

# fields to sweep, as (species, field, component, physical plane)
FIELD_PLOTS = [
    ("em_fields", "phi_phy", 0, "XY"),
]


def main(paths):
    runs = {os.path.basename(p): open_run(p).process(physical=True) for p in paths}

    # growth rate of the electrostatic energy, one curve per run
    series = []
    for name, run in runs.items():
        energy = run.scalars[FIT_QUANTITY].copy()
        energy.attrs["label"] = name if len(runs) > 1 else FIT_QUANTITY
        series.append(energy)

    plot = plot_timeseries(
        series,
        fit=GrowthFit(FIT_WINDOW),
        run_label=next(iter(runs.values())).label,
        title=f"Evolution of {FIT_QUANTITY}",
    ).show()

    for name, result in zip(runs, plot.fit_results):
        print(f"{name}: growth rate = {None if result is None else result.rate}")

    if len(runs) > 1:
        return

    path_out, run = paths[0], next(iter(runs.values()))

    if SHOW_EQUIL_PROFILE:
        plot_equilibrium_profile(path_out)

    for bin_name, quantity, plane in DENSITY_PLOTS:
        data = getattr(getattr(run.distributions.kinetic_ions, bin_name), quantity)
        InteractiveSliceViewer(data, view=View(x="e1", y="e2", coordinates="physical", plane=plane),
                               run_label=run.label).show()

    for species, field, component, plane in FIELD_PLOTS:
        data = getattr(getattr(run.fields, species), field).isel(component=component)
        InteractiveSliceViewer(data, view=View(x="e1", y="e2", coordinates="physical", plane=plane),
                               run_label=run.label).show()

    plot_marker_trajectories(run.orbits.kinetic_ions, max_markers=1000).show()


if __name__ == "__main__":
    sim_names = sys.argv[1:] or ["sim_1"]
    main([os.path.join(os.getcwd(), name) for name in sim_names])
