"""Cylindrically symmetric ion optics: the two-tube immersion lens in an (r, z) wedge.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/axisymmetric_lens/axisymmetric_lens.py

The axisymmetric problem is solved on a thin wedge of revolution
(``AxisymmetricElectrodeChannel``, a ``PoloidalSplineTorus`` whose poloidal plane is the
meridional (r, z) plane). The FEEC volume element r dr dz dθ makes the weak Poisson problem
exactly axisymmetric. The natural boundary on the wedge faces and on a thin axis cylinder
gives the symmetry conditions, and markers reflect on those surfaces.

Checks, against the Bessel-series potential of the two-tube lens (``two_tube_analytic.py``):

1. the vacuum potential converges under mesh refinement;
2. a parallel 5 keV proton beam focuses like rays integrated in the exact field.

Writes ``axisymmetric_lens.png`` next to this file.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from two_tube_analytic import TwoTubeLens  # noqa: E402

from struphy.diagnostics.beam_diagnostics import plane_crossings  # noqa: E402
from struphy.geometry.axisymmetric import AxisymmetricElectrodeChannel, meridional  # noqa: E402
from struphy.initial.perturbations import PiecewiseLinearPotential  # noqa: E402
from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.models.ion_optics_steady_state import RayBundle, SteadyStateOptions, build_steady_state_simulation  # noqa: E402
from struphy.physics.ion_optics_units import IonOpticsUnits  # noqa: E402
from struphy.pic.ion_beams import LossTag  # noqa: E402

UNITS = IonOpticsUnits(length=1e-3, voltage=1e3)  # mm, kV, protons
RADIUS, LENGTH, GAP_CENTER, GAP = 5.0, 60.0, 30.0, 2.0
V1, V2 = 0.0, -30.0  # kV (focus at 45–47 mm, inside the 60 mm domain)
ENERGY_EV = 5e3
LENS = TwoTubeLens(RADIUS, GAP, GAP_CENTER, V1, V2, n_terms=120, n_quad=24)  # converged to 1e-9 in the beam region


def build(output_dir, num_elements=(120, 20), n_rays=21, name="axisymmetric_lens", steady_state=None):
    domain = AxisymmetricElectrodeChannel(
        LENGTH, ((0.0, LENGTH), (RADIUS, RADIUS)), axis_radius=0.01, num_elements=num_elements
    )
    model = IonOpticsElectrostatic(
        electrode_faces=((False, False), (False, True), (False, False)), steady_state=steady_state
    )
    model.em_fields.phi.add_perturbation(
        PiecewiseLinearPotential((GAP_CENTER - GAP / 2, GAP_CENTER + GAP / 2), (V1, V2), coordinate=2)
    )
    sim = build_steady_state_simulation(
        output_dir,
        name,
        model,
        n_rays,
        domain,
        (*num_elements, 1),
        (3, 3, 1),
        ("remove", ("reflect", "remove"), "reflect"),
    )
    return sim, domain


def potential_error(output_dir, num_elements):
    sim, domain = build(output_dir, num_elements, name=f"axi_{num_elements[0]}")
    sim.allocate()
    e1, e2 = np.linspace(0, 1, 121), np.linspace(0, 0.8, 17)
    phi = np.asarray(sim.model.em_fields.phi.spline(e1, e2, np.array([0.5])))[:, :, 0]
    E1, E2 = np.meshgrid(e1, e2, indexing="ij")
    z, r = meridional(domain, np.stack([E1, E2, 0.5 + 0 * E1], -1))
    return np.max(np.abs(phi - LENS.phi(r, z)))


def trace(output_dir, radii, num_elements=(120, 20), dt=0.04):
    """Zero-current rays launched parallel at z = 1 mm, radii ``radii``; one round of ray tracing."""
    domain = AxisymmetricElectrodeChannel(
        LENGTH, ((0.0, LENGTH), (RADIUS, RADIUS)), axis_radius=0.01, num_elements=num_elements
    )
    v0 = float(UNITS.speed(ENERGY_EV))
    eta = np.column_stack(
        [
            np.full(len(radii), 1.0 / LENGTH),
            (np.asarray(radii) - domain.axis_radius) / (RADIUS - domain.axis_radius),
            np.full(len(radii), 0.5),
        ]
    )
    v = np.column_stack([np.zeros(len(radii)), np.zeros(len(radii)), np.full(len(radii), v0)])
    sim, _ = build(
        output_dir,
        num_elements,
        n_rays=len(radii),
        steady_state=SteadyStateOptions(
            rays=RayBundle(eta=eta, v=v, current=1e-12),
            dt=dt,
            loss_tags=(
                LossTag("outlet", axis=0, side=1),
                LossTag("tube", axis=1, side=1),
                LossTag("inlet", axis=0, side=0),
            ),
            exit_tag="outlet",
            max_rounds=1,
            n_tracked=len(radii),
        ),
    )
    sim.run()
    iteration = sim.model.steady_state_iteration
    z, r = meridional(domain, iteration.trajectories)
    return z, r, iteration


def reference(radii, t_end=40.0):
    """Meridional rays in the exact field: y'' = E_r(|y|) sgn(y), z'' = E_z."""
    v0 = float(UNITS.speed(ENERGY_EV))

    def rhs(_, s):
        y, z, vy, vz = s.reshape(4, -1)
        er, ez = LENS.efield(np.abs(y), z)
        return np.concatenate([vy, vz, er * np.sign(y), ez])

    s0 = np.concatenate([radii, np.full(len(radii), 1.0), np.zeros(len(radii)), np.full(len(radii), v0)])
    sol = solve_ivp(rhs, (0, t_end), s0, method="DOP853", rtol=1e-8, atol=1e-9, dense_output=True)
    t = np.linspace(0, t_end, 4001)
    y, z = sol.sol(t).reshape(4, len(radii), -1)[:2]
    return z.T, y.T


def incoming_axis_crossing(z, r, window=(0.02, 0.15)):
    """Axial position where a ray reaches the axis, from a line fit to its incoming segment.

    Rays reflect on the thin axis cylinder (radius 0.01 mm) instead of passing through the axis. For
    near-paraxial rays the minimum of r(z) is therefore poorly localized, so the incoming segment with
    ``window[0] < r < window[1]`` (mm) before the first minimum is extrapolated to r = 0.
    """
    valid = np.isfinite(r)
    z, r = z[valid], r[valid]
    first_min = int(np.argmin(r))
    zi, ri = z[:first_min], r[:first_min]
    sel = (ri > window[0]) & (ri < window[1])
    if np.count_nonzero(sel) < 2:
        return z[first_min]
    slope, intercept = np.polyfit(zi[sel], ri[sel], 1)
    return -intercept / slope


def axis_crossing(z, r_signed):
    states = np.stack([z, r_signed], axis=-1)
    return plane_crossings(states, 0.0, axis=1)[:, 0]


def main():
    output_dir = HERE / "output"
    resolutions = [(60, 10), (120, 20)]
    errors = [potential_error(output_dir, ne) for ne in resolutions]
    for ne, e in zip(resolutions, errors):
        print(f"{ne}: max |phi - phi_exact| = {e:.2e} kV")

    radii = np.linspace(0.5, 3.0, 6)
    z, r, iteration = trace(output_dir, radii)
    z_ref, y_ref = reference(radii)
    crossings = np.array([incoming_axis_crossing(z[:, j], r[:, j]) for j in range(len(radii))])
    reference_crossings = axis_crossing(z_ref, y_ref)
    print("focus (axis crossing) struphy vs exact:", np.round(crossings, 2), np.round(reference_crossings, 2))

    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.3, 1))
    ax = fig.add_subplot(grid[0, :])
    ax_conv, ax_focus, ax_rays = (fig.add_subplot(grid[1, i]) for i in range(3))
    fig.suptitle(
        f"Axisymmetric two-tube lens in Struphy (r–z wedge): {ENERGY_EV / 1e3:g} keV protons, "
        f"tubes at {V1:g} / {V2:g} kV, R = {RADIUS:g} mm"
    )
    zz, rr = np.meshgrid(np.linspace(0, LENGTH, 301), np.linspace(0, RADIUS, 51), indexing="ij")
    phi = LENS.phi(rr, zz)
    for sign in (1, -1):
        cont = ax.contourf(zz, sign * rr, phi, levels=30, cmap="viridis")
    fig.colorbar(cont, ax=ax, label="Potential (kV)", pad=0.01)
    for sign in (1, -1):
        ax.plot((0, GAP_CENTER - GAP / 2), (sign * RADIUS,) * 2, color="0.6", linewidth=6, solid_capstyle="butt")
        ax.plot(
            (GAP_CENTER + GAP / 2, LENGTH), (sign * RADIUS,) * 2, color="#f28e2b", linewidth=6, solid_capstyle="butt"
        )
    for j in range(len(radii)):
        for sign in (1, -1):
            ax.plot(z[:, j], sign * r[:, j], color="#e15759", linewidth=0.8)
    ax.set(
        xlabel="z (mm)",
        ylabel="r (mm)",
        title="Rays traced in the Struphy wedge, mirrored for display",
        xlim=(0, LENGTH),
    )

    ax_conv.loglog([n[1] for n in resolutions], errors, "o-")
    ax_conv.set(xlabel="Radial elements", ylabel="max |φ − φ_exact| (kV)", title="Vacuum potential vs Bessel series")

    ax_focus.plot(radii, crossings, "o", label="Struphy")
    ax_focus.plot(radii, reference_crossings, "k+", markersize=9, label="Exact-field rays")
    ax_focus.set(xlabel="Initial radius (mm)", ylabel="Axis crossing z (mm)", title="Focus and spherical aberration")
    ax_focus.legend()

    for j in (0, len(radii) // 2, len(radii) - 1):
        ax_rays.plot(z[:, j], r[:, j], label=f"Struphy r₀ = {radii[j]:.2f}")
        ax_rays.plot(z_ref[:, j], np.abs(y_ref[:, j]), "k:", linewidth=1)
    ax_rays.set(
        xlabel="z (mm)", ylabel="r (mm)", title="Ray radius (dotted: exact field)", xlim=(20, LENGTH), ylim=(0, 3.5)
    )
    ax_rays.legend(fontsize=7)
    fig.savefig(HERE / "axisymmetric_lens.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {HERE / 'axisymmetric_lens.png'}")


if __name__ == "__main__":
    main()
