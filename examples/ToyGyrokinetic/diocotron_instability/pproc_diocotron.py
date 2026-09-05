"""Post-process and plot the diocotron instability.

Run as ``python pproc_diocotron.py [sim_1 sim_2 ...]`` to compare several runs; with
more than one folder only the growth-rate comparison is shown.
"""

import os
import sys

from struphy import PlottingData, PostProcessor
from struphy.diagnostics.plotting import (
    MarkerTrajectoryPlot,
    SliderPlot,
    TimeSeriesPlot,
    field_slice_grids,
    physical_grids,
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


def load(path_out):
    PostProcessor(path_out=path_out).process(physical=True, force=False)
    pdata = PlottingData(path_out=path_out)
    pdata.load()
    return pdata


def main(paths):
    runs = {os.path.basename(p): load(p) for p in paths}

    # growth rate of the electrostatic energy, one curve per run
    series = []
    for name, pdata in runs.items():
        energy = pdata.scalars[FIT_QUANTITY]
        energy.label = name if len(runs) > 1 else FIT_QUANTITY
        series.append(energy)

    plot = TimeSeriesPlot(
        series,
        fit=True,
        fit_window=FIT_WINDOW,
        params=next(iter(runs.values())).params,
        title=f"Evolution of {FIT_QUANTITY}",
    ).show()

    for name, (gamma, _, _) in zip(runs, plot.fit_results):
        print(f"{name}: growth rate = {gamma}")

    if len(runs) > 1:
        return

    path_out, pdata = paths[0], next(iter(runs.values()))

    if SHOW_EQUIL_PROFILE:
        plot_equilibrium_profile(path_out)

    for bin_name, quantity, plane in DENSITY_PLOTS:
        data = pdata.f.kinetic_ions[bin_name][quantity]
        SliderPlot(
            data,
            grids=physical_grids(data.isel(t=0), pdata.domain, axes=plane),
            params=pdata.params,
            title=f"{quantity} ({plane})",
        ).show()

    for species, field, component, plane in FIELD_PLOTS:
        data = pdata.spline_values[species][field].array.isel(comp=component)
        SliderPlot(
            data,
            # the cut plane moves with the slider, so the grids follow it
            grids=lambda index, plane=plane: field_slice_grids(
                pdata.grids_phy, fixed_dim="e3", index=index, plane=plane
            ),
            slice_dim="e3",
            params=pdata.params,
            title=f"{species}.{field} ({plane})",
        ).show()

    MarkerTrajectoryPlot(pdata.orbits.kinetic_ions, max_markers=1000).show()


if __name__ == "__main__":
    sim_names = sys.argv[1:] or ["sim_1"]
    main([os.path.join(os.getcwd(), name) for name in sim_names])
