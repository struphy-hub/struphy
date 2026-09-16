"""Post-process and plot the diocotron instability.

Run as ``python pproc_diocotron.py [sim_1 sim_2 ...]`` to compare several runs; with
more than one folder only the growth-rate comparison is shown.
"""

import os
import sys

from struphy import open_output

FIT_QUANTITY = "en_phi"
FIT_WINDOW = (0.0, 42.0)

SHOW_EQUIL_PROFILE = True

# products to sweep interactively in the physical XY plane
SWEEPS = [
    "kinetic_ions/e1_e2_density/f",
    "kinetic_ions/e1_e2_density/delta_f",
    "em_fields/phi_xyz",
]


def main(paths):
    runs = [open_output(path).process(physical=True) for path in paths]
    run = runs[0]

    # growth rate of the electrostatic energy, one curve per run
    first, *rest = (each[FIT_QUANTITY] for each in runs)
    plot = first.struphy.plot.timeseries(
        *rest,
        fit=FIT_WINDOW,
        title=f"Evolution of {FIT_QUANTITY}",
    ).show()

    for each, result in zip(runs, plot.fit_results):
        print(f"{each.path_out.name}: growth rate = {None if result is None else result.rate}")

    if len(runs) > 1:
        return

    if SHOW_EQUIL_PROFILE:
        run.plot.equilibrium()

    for name in SWEEPS:
        run[name].struphy.plot.viewer(x="e1", y="e2", coords="physical", plane="XY").show()

    run.kinetic_ions.orbits.struphy.plot.trajectories(max_markers=1000).show()


if __name__ == "__main__":
    sim_names = sys.argv[1:] or ["sim_1"]
    main([os.path.join(os.getcwd(), name) for name in sim_names])
