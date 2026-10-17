"""h-convergence of the Poisson polar_rounded_square example (struphy counterpart of the Poisson panel of
PolarFEEC.jl example 05): runs params_polar_rounded_square.py for n radial cells and degrees p (regularity r = p),
post-processes every run and plots the L2 error against h = 1/n with dotted h^(p + 1) reference slopes.

Run as ``python convergence_polar_rounded_square.py [--degrees 2 3 4] [--cells 4 6 8 12] [--mpi N]``.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def run(p, n, mpi):
    folder = f"conv_p{p}_n{n}"
    flags = ["--degree", str(p), "--cells", str(n), "--sim-folder", folder]
    command = [sys.executable, str(SCRIPT_DIR / "params_polar_rounded_square.py"), *flags]
    if mpi > 1:
        command = ["mpirun", "-n", str(mpi), *command]
    subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "pproc_polar_rounded_square.py"), str(SCRIPT_DIR / folder),
                    *flags], check=True)
    with open(SCRIPT_DIR / folder / "results" / "errors.json") as f:
        return json.load(f)["l2_error"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--cells", type=int, nargs="+", default=[4, 6, 8, 12])
    parser.add_argument("--mpi", type=int, default=1, help="Number of MPI processes per run.")
    args = parser.parse_args()

    hs = 1 / np.array(args.cells)
    table = {p: np.array([run(p, n, args.mpi) for n in args.cells]) for p in args.degrees}
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (p, errors) in enumerate(table.items()):
        rates = np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])
        print(f"p = {p}: errors {errors}, observed rates {rates}")
        ax.loglog(hs, errors, "o-", color=f"C{i}", label=f"$p = r = {p}$")
        ax.loglog(hs[-3:], 0.3 * errors[-1] * (hs[-3:] / hs[-1]) ** (p + 1), ":", color=f"C{i}")
    ax.set_xticks(hs, [f"1/{n}" for n in args.cells])
    ax.set_xlabel("$h = 1/n$")
    ax.set_ylabel(r"$\Vert \phi - \phi_h \Vert_{L^2}$")
    ax.set_title("Poisson on a rounded square, homotopy polar splines\n(dotted: $h^{p+1}$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SCRIPT_DIR / "convergence_polar_rounded_square.png", dpi=150)
    print("wrote", SCRIPT_DIR / "convergence_polar_rounded_square.png")


if __name__ == "__main__":
    main()
