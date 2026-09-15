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
    run = open_run(path_out).process(physical=True)

    # growth rate of the electrostatic potential
    plot_timeseries(
        run.scalars[FIT_QUANTITY],
        fit=GrowthFit(FIT_WINDOW, amplitude_from_quadratic=True),
        run_label=run.label,
        title=f"Evolution of {FIT_QUANTITY}",
    ).show()

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
    sim_name = sys.argv[1] if len(sys.argv) > 1 else "sim_1"
    main(os.path.join(os.getcwd(), sim_name))
