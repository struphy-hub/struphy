"""Energy conservation of RK4 guiding-centre orbits near the axis against the time step, for polar splines of
regularity r = 0, ..., 4 (the orbit study of the polar FEEC talk): runs params_vacuum_near_axis.py for every (r, dt),
post-processes the runs and plots the maximum (over the eight orbits) of max_t |E(t) - E(0)| / (mu |B|) against dt,
with a dotted O(dt^4) reference.

Run as ``python convergence_vacuum_near_axis.py [--regularities 0 1 2 3 4] [--dts 0.04 0.02 0.01 0.005 0.0025]
[--periods 20]``. The default (20 transits, dt down to 0.0025) is a long study.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def run(r, dt, periods, mpi):
    folder = f"conv_r{r}_dt{dt:g}"
    flags = ["--regularity", str(r), "--dt", str(dt), "--periods", str(periods), "--sim-folder", folder]
    flags += ["--save-step", str(max(1, int(round(0.04 / dt))))]
    command = [sys.executable, str(SCRIPT_DIR / "params_vacuum_near_axis.py"), *flags]
    if mpi > 1:
        command = ["mpirun", "-n", str(mpi), *command]
    subprocess.run(command, check=True)
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "pproc_vacuum_near_axis.py"), str(SCRIPT_DIR / folder), *flags], check=True
    )
    with open(SCRIPT_DIR / folder / "results" / "energy_errors.json") as f:
        return max(json.load(f)["errors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regularities", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dts", type=float, nargs="+", default=[0.04, 0.02, 0.01, 0.005, 0.0025])
    parser.add_argument("--periods", type=float, default=20)
    parser.add_argument("--mpi", type=int, default=1)
    args = parser.parse_args()

    dts = np.array(args.dts)
    table = {r: np.array([run(r, dt, args.periods, args.mpi) for dt in dts]) for r in args.regularities}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (r, errors) in enumerate(table.items()):
        print(f"r = {r}: {errors}")
        ax.loglog(dts, errors, "o-", color=f"C{i}", label=rf"$\mathfrak{{r}} = {r}$")
    reference = table[max(table)]
    ax.loglog(dts, reference[0] * (dts / dts[0]) ** 4, "k:", label=r"$O(\Delta t^4)$")
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$\max_t |E(t) - E(0)| / (\mu |B|)$")
    ax.set_title("RK4 guiding-centre orbits near the axis")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "convergence_vacuum_near_axis.png", dpi=150)
    print("wrote", SCRIPT_DIR / "convergence_vacuum_near_axis.png")


if __name__ == "__main__":
    main()
