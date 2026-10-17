"""Post-process the GuidingCenter vacuum_near_axis example: orbits projected onto the cross-section (overview and
close-up) and the relative energy error max_t |E(t) - E(0)| / (mu |B|) of each orbit, with E = v_par^2 / 2 + mu |B|_h
and |B|_h the projected field strength used by the pusher. Run as
``python pproc_vacuum_near_axis.py [sim_folder] [variant flags]``."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from struphy import Output

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "sim_1"

sys.path.insert(0, str(SCRIPT_DIR))
from params_vacuum_near_axis import LZ, A, args, parser, sim  # noqa: E402


def projected_field_strength():
    """|B|_h: the commuting projection of |B| into the polar 0-forms (as in the GuidingCenter model)."""
    from struphy.feec.psydac_derham import Derham

    derham = Derham(sim.grid, sim.derham_opts, domain=sim.domain)
    sim.equil.domain = sim.domain
    absB = derham.create_spline_function("absB", "H1")
    absB.vector = derham.P0(sim.equil.absB0)
    return absB


def logical(x, y, z):
    """Logical coordinates of points in the cylinder HollowCylinder(a1=0, a2=A, Lz=LZ)."""
    return np.hypot(x, y) / A, (np.arctan2(y, x) / (2 * np.pi)) % 1.0, (z / LZ) % 1.0


def energy_errors(orbits, absB):
    x, y, z = (np.asarray(orbits[c]) for c in ("x", "y", "z"))
    v_par, mu = np.asarray(orbits["v_par"]), np.asarray(orbits["mu"])
    e1, e2, e3 = logical(x, y, z)
    markers = np.column_stack([e1.ravel(), e2.ravel(), e3.ravel()])
    B = np.asarray(absB(markers)).reshape(x.shape)
    energy = v_par**2 / 2 + mu * B
    return np.max(np.abs(energy - energy[0]), axis=0) / (mu[0] * B[0])


def main(path_out=DEFAULT_OUTPUT, save_figs=True):
    out = Output(path_out)
    orbits = out.evaluate("kinetic_ions/orbits")
    results_dir = os.path.join(out.path_out, "results")
    os.makedirs(results_dir, exist_ok=True)

    errors = energy_errors(orbits, projected_field_strength())
    print("relative energy errors max_t |E(t) - E(0)| / (mu |B|):", np.array2string(errors, precision=2))
    with open(os.path.join(results_dir, "energy_errors.json"), "w") as f:
        json.dump(dict(errors=errors.tolist(), dt=args.dt, regularity=args.regularity), f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    x, y = np.asarray(orbits["x"]), np.asarray(orbits["y"])
    for ax, extent in ((axes[0], A), (axes[1], 0.05)):
        for i in range(x.shape[1]):
            ax.plot(x[:, i], y[:, i], ".", ms=1, color=f"C{i % 10}")
        ax.add_patch(plt.Circle((0, 0), A, color="k", fill=False))
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    axes[0].set_title("orbits projected onto a cross-section")
    axes[1].set_title("close-up")
    fig.suptitle(rf"guiding centres near the axis, $\mathfrak{{r}} = {args.regularity}$, $\Delta t = {args.dt}$")
    fig.tight_layout()

    if save_figs:
        fig.savefig(os.path.join(results_dir, "orbits.png"))
        plt.close("all")
    else:
        plt.show()
    return errors


if __name__ == "__main__":
    _, other_flags = parser.parse_known_args()
    pproc_parser = argparse.ArgumentParser(description="Post-process a saved simulation run.")
    pproc_parser.add_argument("path_out", nargs="?", default=DEFAULT_OUTPUT, help="Simulation output folder.")
    pproc_parser.add_argument("--show", action="store_true", help="Show the figures instead of saving them.")
    pproc_args = pproc_parser.parse_args(other_flags)
    main(pproc_args.path_out, save_figs=not pproc_args.show)
