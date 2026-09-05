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

# quantity whose exponential growth rate is fitted
FIT_QUANTITY = "phi_integral"
FIT_WINDOW = (0.0, None)

SHOW_EQUIL_PROFILE = False

# binned densities to sweep, as (bin name, quantity, physical plane)
DENSITY_PLOTS = [
    ("e1_e2_density", "f_binned", "XY"),
    ("e1_e2_density", "delta_f_binned", "XY"),
]

# fields to sweep, as (species, field, component, physical plane)
FIELD_PLOTS = [
    ("em_fields", "phi_phy", 0, "XY"),
    ("diagnostics", "rho_phy", 0, "XY"),
]


def main(path_out):
    PostProcessor(path_out=path_out).process(physical=True, force=False)

    pdata = PlottingData(path_out=path_out)
    pdata.load()

    # growth rate of the electrostatic potential
    TimeSeriesPlot(
        pdata.scalars[FIT_QUANTITY],
        fit=True,
        fit_window=FIT_WINDOW,
        fit_of_sqrt=True,
        params=pdata.params,
        title=f"Evolution of {FIT_QUANTITY}",
    ).show()

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
    sim_name = sys.argv[1] if len(sys.argv) > 1 else "sim_1"
    main(os.path.join(os.getcwd(), sim_name))
