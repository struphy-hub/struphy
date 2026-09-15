import os
import sys

from struphy import open_run

# quantity whose exponential growth rate is fitted
FIT_QUANTITY = "phi_integral"
FIT_WINDOW = (0.0, None)

SHOW_EQUIL_PROFILE = False

# products to sweep interactively, as (name, displayed component or None, physical plane)
SWEEPS = [
    ("kinetic_ions/e1_e2_density/delta_f_binned", None, "RZ"),
    ("em_fields/phi_phy", None, "RZ"),
    ("diagnostics/rho_phy", None, "RZ"),
    ("diagnostics/rho_phy", None, "XY"),
]


def main(path_out):
    run = open_run(path_out).process(physical=True)

    # growth rate of the electrostatic potential
    run.plot.timeseries(
        FIT_QUANTITY,
        fit=FIT_WINDOW,
        fit_amplitude=True,
        title=f"Evolution of {FIT_QUANTITY}",
    ).show()

    if SHOW_EQUIL_PROFILE:
        run.plot.equilibrium()

    for name, component, plane in SWEEPS:
        isel = None if component is None else {"component": component}
        run.plot.viewer(name, x="e1", y="e2", isel=isel, coords="physical", plane=plane).show()

    run.plot.orbits("kinetic_ions", max_markers=1000).show()


if __name__ == "__main__":
    sim_name = sys.argv[1] if len(sys.argv) > 1 else "sim_1"
    main(os.path.join(os.getcwd(), sim_name))
