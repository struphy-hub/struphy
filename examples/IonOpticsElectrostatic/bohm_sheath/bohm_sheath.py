"""Plasma extraction building block: the planar Bohm sheath (Poisson–Boltzmann).

Run from the repository root with::

    python examples/IonOpticsElectrostatic/bohm_sheath/bohm_sheath.py

Positive ions enter from a plasma at potential 0 with velocity v0 >= v_B. Thermal electrons
follow a Boltzmann density (``BoltzmannElectrons``), and a wall at x = L is held at phi_W.
In natural units (length lambda_D, potential kT_e/e, density n_0) the steady state
satisfies (Kalvas 2013, eq. 2.23)

    phi'' = -( v0 / sqrt(v0² - 2 phi) - exp(phi) ),   phi(0) = 0, phi(L) = phi_W.

Struphy solves it with ``SteadyStateIteration``: ions are ray-traced, and the potential
minimizes the convex Poisson–Boltzmann energy, using Newton with a line search on the
energy. ``solve_bvp`` gives the reference.

Writes ``bohm_sheath.png`` next to this file.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_bvp

HERE = Path(__file__).resolve().parent

from struphy import domains  # noqa: E402
from struphy.initial.perturbations import PiecewiseLinearPotential  # noqa: E402
from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.models.ion_optics_steady_state import SteadyStateOptions, build_steady_state_simulation  # noqa: E402
from struphy.physics.ion_optics_units import IonOpticsUnits  # noqa: E402
from struphy.physics.plasma_models import BoltzmannElectrons  # noqa: E402
from struphy.pic.ion_beams import LossTag, PlaneSource  # noqa: E402

LENGTH = 20.0  # Debye lengths
PHI_WALL = -10.0  # kT_e / e
# Natural units: the length unit is lambda_D and the voltage unit is kT_e/e. The SI
# values below (0.1 mm, 5 eV) only label the axes; the solution is universal.
UNITS = IonOpticsUnits(length=1e-4, voltage=5.0)
PLASMA = BoltzmannElectrons(density=1.0, temperature=1.0, plasma_potential=0.0)


def sheath(output_dir, v0=1.0, num_elements=80, n_rays=200, dt=0.02, tol=1e-6, max_rounds=80):
    # ion flux n0 * v0 per unit area: quasi-neutral with the electrons at phi = 0
    source = PlaneSource(rate=1.0, current=PLASMA.density * v0, velocity=(v0, 0.0, 0.0))
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_faces=((True, True), (False, False), (False, False)),
        plasma=PLASMA,
        steady_state=SteadyStateOptions(
            source=source,
            n_rays=n_rays,
            dt=dt,
            loss_tags=(LossTag("wall", axis=0, side=1), LossTag("plasma", axis=0, side=0)),
            exit_tag="wall",
            criterion="potential",
            tol=tol,
            max_rounds=max_rounds,
        ),
    )
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential((0.0, LENGTH), (0.0, PHI_WALL), coordinate=0))
    sim = build_steady_state_simulation(
        output_dir,
        f"sheath_v{v0:g}_n{num_elements}",
        model,
        n_rays,
        domains.Cuboid(r1=LENGTH),
        (num_elements, 1, 1),
        (3, 1, 1),
        ("remove", "periodic", "periodic"),
    )
    sim.run()
    iteration = model.steady_state_iteration
    x = np.linspace(0.0, LENGTH, 401)
    phi = np.asarray(model.em_fields.phi.spline(x / LENGTH, np.array([0.5]), np.array([0.5]))).ravel()
    return iteration, x, phi


def reference(v0, x):
    def rhs(_, y):
        return np.vstack([y[1], -(v0 / np.sqrt(v0**2 - 2.0 * np.minimum(y[0], 0.0)) - np.exp(y[0]))])

    mesh = np.linspace(0.0, LENGTH, 2001)
    guess = np.vstack([PHI_WALL * (mesh / LENGTH) ** 4, 4 * PHI_WALL * mesh**3 / LENGTH**4])
    sol = solve_bvp(rhs, lambda a, b: np.array([a[0], b[0] - PHI_WALL]), mesh, guess, tol=1e-9, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol(x)[0]


def plot_results(runs, output):
    fig, (ax_phi, ax_err, ax_conv) = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    fig.suptitle("Planar Bohm sheath: ray-traced ions + Boltzmann electrons (Poisson–Boltzmann Newton)")
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for colour, (v0, (iteration, x, phi)) in zip(colours, runs.items()):
        ref = reference(v0, x)
        ax_phi.plot(x, phi, color=colour, label=f"v₀ = {v0:g} v_B")
        ax_phi.plot(x[::20], ref[::20], "o", color=colour, fillstyle="none")
        ax_err.plot(x, phi - ref, color=colour, label=f"v₀ = {v0:g} v_B")
        changes = [r.potential_change for r in iteration.history[:-1]]
        ax_conv.semilogy(
            changes, "o-", color=colour, markersize=3, label=f"v₀ = {v0:g} v_B ({len(iteration.history)} rounds)"
        )
    ax_phi.set(xlabel="x / λ_D", ylabel="eφ / kT_e", title="Potential (circles: steady-state ODE)")
    ax_phi.legend(fontsize=8)
    ax_err.set(xlabel="x / λ_D", ylabel="φ − φ_ref (kT_e/e)", title="Error against the ODE (80 elements, p = 3)")
    ax_err.legend(fontsize=8)
    ax_conv.axhline(1e-6, color="k", linestyle=":", linewidth=0.8)
    ax_conv.set(xlabel="Iteration round", ylabel="‖φ_k − φ_{k−1}‖ / ‖φ_k‖", title="Outer iteration, α = 1")
    ax_conv.legend(fontsize=8)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    runs = {}
    for v0 in (1.0, 1.3):
        runs[v0] = sheath(HERE / "output", v0)
        iteration, x, phi = runs[v0]
        print(
            f"v0 = {v0:g} v_B: converged={iteration.converged} in {len(iteration.history)} rounds, "
            f"max |phi - phi_ref| = {np.max(np.abs(phi - reference(v0, x))):.2e} kT_e/e"
        )
    plot_results(runs, HERE / "bohm_sheath.png")


if __name__ == "__main__":
    main()
