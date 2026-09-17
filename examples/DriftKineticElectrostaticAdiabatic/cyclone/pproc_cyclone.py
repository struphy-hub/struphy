import sys
from pathlib import Path

from struphy import Output

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "sim_1"

# quantity whose exponential growth rate is fitted
FIT_QUANTITY = "phi_integral"
FIT_WINDOW = (0.0, None)

SHOW_EQUIL_PROFILE = False

# products to sweep interactively, as (name, displayed component or None, physical plane)
SWEEPS = [
    ("kinetic_ions/e1_e2_density/delta_f", None, "RZ"),
    ("em_fields/phi_xyz", None, "RZ"),
    ("diagnostics/rho_xyz", None, "RZ"),
    ("diagnostics/rho_xyz", None, "XY"),
]


def main(path_out=DEFAULT_OUTPUT):
    run = Output(path_out).process(physical=True)

    # growth rate of the electrostatic potential
    run[FIT_QUANTITY].struphy.plot.timeseries(
        fit=FIT_WINDOW,
        fit_amplitude=True,
        title=f"Evolution of {FIT_QUANTITY}",
    ).show()

    if SHOW_EQUIL_PROFILE:
        run.plot.equilibrium()

    for name, component, plane in SWEEPS:
        selection = {} if component is None else {"component": component}
        run[name].struphy.plot.viewer(x="e1", y="e2", coords="physical", plane=plane, **selection).show()

    run.kinetic_ions.orbits.struphy.plot.trajectories(max_markers=1000).show()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT)
