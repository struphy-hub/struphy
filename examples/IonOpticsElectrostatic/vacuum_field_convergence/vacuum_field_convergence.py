"""Convergence of the electrode Laplace solve against analytic potentials.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/vacuum_field_convergence/vacuum_field_convergence.py

Two cases, each under h-refinement for several spline degrees p:

* **Coaxial capacitor** on the curved ``HollowCylinder`` mapping: electrodes on
  the inner (V = 1) and outer (V = 0) radial faces, periodic in angle.
* **Slit immersion lens** (``../slit_immersion_lens/analytic.py``): the exact
  potential is imposed on all four faces of a Cartesian window around the gap.
  The linear-gap plate trace has kinks at the gap edges, so convergence there is
  limited by the solution's corner regularity.

The potential error is measured in the maximum norm and the field error from
the model's ``e_field`` (= -grad phi); both are written to
``vacuum_field_convergence.png`` next to this file.
"""

import sys
from pathlib import Path

import numpy as np
from feectools.ddm.mpi import mpi as MPI
from matplotlib import pyplot as plt

from struphy import (
    BoundaryParameters,
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    SavingParameters,
    Simulation,
    Time,
    WeightsParameters,
    domains,
    grids,
    maxwellians,
)
from struphy.initial.base import Perturbation
from struphy.models import IonOpticsElectrostatic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slit_immersion_lens"))
from analytic import SlitImmersionLens  # noqa: E402

A1, A2 = 0.5, 1.0
LENS = SlitImmersionLens(h=5.0, g=2.0, xc=0.0, v1=0.0, v2=-10.0)
LENS_WINDOW = (-15.0, 15.0)


class FunctionPerturbation(Perturbation):
    """Wrap a physical-space function ``f(x, y, z)`` as a scalar initial condition."""

    def __init__(self, function):
        self.params = {}
        self.function = function
        self.given_in_basis = "physical"
        self.comp = 0

    def __call__(self, x, y, z):
        return self.function(x, y, z)


def coaxial_potential(x, y, z):
    return np.log(np.sqrt(x**2 + y**2) / A2) / np.log(A1 / A2)


def lens_potential(x, y, z):
    return LENS.phi(x, np.clip(y, -LENS.h, LENS.h))


def solve(output_dir, name, domain, num_elements, degree, bcs, electrode_faces, potential):
    """Allocate a model (which runs the vacuum solve) and return it."""
    model = IonOpticsElectrostatic(electrode_faces=electrode_faces)
    model.em_fields.phi.add_perturbation(FunctionPerturbation(potential))
    model.ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.ions.set_markers(
        loading_params=LoadingParameters(Np=1, specific_markers=((0.5, 0.5, 0.5, 0.0, 0.0, 0.0),)),
        weights_params=WeightsParameters(control_variate=False),
        boundary_params=BoundaryParameters(bc=("remove", "remove", "periodic")),
        saving_params=SavingParameters(n_markers=1),
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(output_dir), sim_folder=name),
        time_opts=Time(dt=0.1, Tend=0.1),
        domain=domain,
        grid=grids.TensorProductGrid(num_elements=(*num_elements, 1)),
        derham_opts=DerhamOptions(degree=(degree, degree, 1), bcs=bcs),
    )
    sim.allocate()
    return model


def coaxial_errors(output_dir, n, degree):
    model = solve(
        output_dir,
        f"coax_{n}_{degree}",
        domains.HollowCylinder(a1=A1, a2=A2, Lz=1.0),
        (n, 4 * n),
        degree,
        (("free", "free"), None, ("free", "free")),
        ((True, True), (False, False), (False, False)),
        coaxial_potential,
    )
    eta1, eta2, eta3 = np.linspace(0.0, 1.0, 101), np.linspace(0.0, 1.0, 9)[:-1], np.array([0.5])
    r = (A1 + (A2 - A1) * eta1)[:, None]
    phi = np.asarray(model.em_fields.phi.spline(eta1, eta2, eta3))[:, :, 0]
    # Covariant radial component e_1 = E_r * dr/deta1.
    e_r = np.asarray(model.em_fields.e_field.spline(eta1, eta2, eta3)[0])[:, :, 0] / (A2 - A1)
    exact_phi = np.log(r / A2) / np.log(A1 / A2)
    exact_e_r = -1.0 / (r * np.log(A1 / A2))
    return np.max(np.abs(phi - exact_phi)), np.max(np.abs(e_r - exact_e_r))


def lens_errors(output_dir, n, degree):
    (left, right), h = LENS_WINDOW, LENS.h
    model = solve(
        output_dir,
        f"lens_{n}_{degree}",
        domains.Cuboid(l1=left, r1=right, l2=-h, r2=h),
        (3 * n, n),
        degree,
        (("free", "free"), ("free", "free"), ("free", "free")),
        ((True, True), (True, True), (False, False)),
        lens_potential,
    )
    # Beam region |y| <= h/2, away from the singular gap corners on the plates.
    eta1, eta2, eta3 = np.linspace(0.0, 1.0, 121), np.linspace(0.25, 0.75, 21), np.array([0.5])
    x = (left + (right - left) * eta1)[:, None]
    y = (-h + 2 * h * eta2)[None, :]
    phi = np.asarray(model.em_fields.phi.spline(eta1, eta2, eta3))[:, :, 0]
    e1, e2, _ = model.em_fields.e_field.spline(eta1, eta2, eta3)
    ex, ey = np.asarray(e1)[:, :, 0] / (right - left), np.asarray(e2)[:, :, 0] / (2 * h)
    exact_ex, exact_ey = LENS.efield(x, y)
    return np.max(np.abs(phi - LENS.phi(x, y))), np.max(np.hypot(ex - exact_ex, ey - exact_ey))


CASES = {
    "Coaxial capacitor (HollowCylinder)": (coaxial_errors, (4, 8, 16, 32), (2, 3, 4)),
    "Slit lens, beam region |y| ≤ h/2": (lens_errors, (5, 10, 20, 40), (2, 3, 4)),
}


def run_study(output_dir):
    results = {}
    for title, (errors, resolutions, degrees) in CASES.items():
        for degree in degrees:
            values = np.array([errors(output_dir, n, degree) for n in resolutions])
            results[title, degree] = (np.array(resolutions), values)
            rates = np.log2(values[:-1] / values[1:])
            print(f"{title}, p = {degree}")
            for n, (e_phi, e_field), rate in zip(resolutions, values, [(np.nan, np.nan), *rates]):
                print(
                    f"  n = {n:3d}: |phi err| = {e_phi:.3e} (rate {rate[0]:4.2f}), |E err| = {e_field:.3e} (rate {rate[1]:4.2f})"
                )
    return results


def plot_study(results, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    fig.suptitle("Vacuum Laplace solve with electrode Dirichlet faces: h-convergence")
    colours = {2: "#4e79a7", 3: "#f28e2b", 4: "#59a14f"}
    for ax, title in zip(axes, CASES):
        for (case, degree), (resolutions, values) in results.items():
            if case != title:
                continue
            ax.loglog(resolutions, values[:, 0], "o-", color=colours[degree], label=f"φ, p = {degree}")
            ax.loglog(resolutions, values[:, 1], "s--", color=colours[degree], label=f"E, p = {degree}")
            reference = values[0, 0] * (resolutions / resolutions[0]) ** -(degree + 1.0)
            ax.loglog(resolutions, reference, ":", color=colours[degree], alpha=0.6)
        ax.set(
            xlabel="Elements across radius / slit width",
            ylabel="Max. error (normalized)",
            title=f"{title}\n(dotted: order p + 1)",
        )
        ax.set_xticks(resolutions, [str(n) for n in resolutions])
        ax.minorticks_off()
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("Run this convergence study on one MPI rank.")
    results = run_study(Path(__file__).parent / "output")
    plot_study(results, Path(__file__).with_name("vacuum_field_convergence.png"))


if __name__ == "__main__":
    main()
