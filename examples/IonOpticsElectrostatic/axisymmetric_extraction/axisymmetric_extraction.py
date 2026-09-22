"""IBSimu-style cylindrically symmetric positive-ion extraction (round apertures) in Struphy.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/axisymmetric_extraction/axisymmetric_extraction.py

The generic extraction of ``../plasma_extraction`` (same wall profile, voltages and plasma),
here revolved about the beam axis. The apertures are round holes, solved on a thin (r, z)
wedge (``AxisymmetricElectrodeChannel``). Ions leave a disk at the back of the plasma chamber
with the Bohm speed, the quasi-neutral current density e n_0 v_B, and an isotropic transverse
temperature (``RayBundle.axisymmetric_disk``). The steady state comes from ray tracing plus the
Poisson–Boltzmann Newton solve.

Writes ``axisymmetric_extraction.png`` next to this file.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plasma_extraction"))
from plasma_extraction import DEFAULT, UNITS, plasma_model  # noqa: E402

from struphy.geometry.axisymmetric import AxisymmetricElectrodeChannel, meridional  # noqa: E402
from struphy.geometry.domains import ElectrodeSegment  # noqa: E402
from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.models.ion_optics_steady_state import RayBundle, SteadyStateOptions, build_steady_state_simulation  # noqa: E402
from struphy.pic.ion_beams import LossTag  # noqa: E402


def build_domain(case=DEFAULT, num_elements=(128, 48), axis_radius=0.01, tor_period=360):
    z, r = case.profile()
    gap0, gap1 = case.gap
    v_puller = float(UNITS.potential(-case.extraction_voltage))
    segments = (
        ElectrodeSegment("upper", 0.0, gap0, 0.0, "plasma electrode"),
        ElectrodeSegment("upper", gap1, case.length, v_puller, "puller"),
    )
    return AxisymmetricElectrodeChannel(
        case.length,
        (z, r),
        segments=segments,
        axis_radius=axis_radius,
        tor_period=tor_period,
        num_elements=num_elements,
    )


def run(
    output_dir,
    case=DEFAULT,
    num_elements=(48, 18),
    n_rays=300,
    dt=0.1,
    alpha=0.2,
    max_rounds=60,
    n_tracked=None,
    relaxation="adaptive",
    criterion="residual",
    tol=1e-3,
    verbose=True,
):
    domain = build_domain(case, num_elements)
    plasma = plasma_model(case)
    v_bohm = plasma.bohm_speed()
    rays = RayBundle.axisymmetric_disk(
        domain,
        n_rays,
        z0=0.0,
        radius=0.98 * case.chamber_half_height,
        current_density=plasma.density * v_bohm,
        axial_speed=v_bohm,
        transverse_spread=float(np.sqrt(case.transverse_temperature_eV / UNITS.voltage)),
    )
    x_gap = 0.5 * sum(case.gap)
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_segments=domain.segments,
        electrode_length=domain.length,
        plasma=plasma,
        steady_state=SteadyStateOptions(
            rays=rays,
            dt=dt,
            alpha=alpha,
            relaxation=relaxation,
            criterion=criterion,
            tol=tol,
            loss_tags=(
                LossTag("plasma electrode", axis=1, side=1, coordinate=2, interval=(-np.inf, x_gap)),
                LossTag("puller", axis=1, side=1, coordinate=2, interval=(x_gap, np.inf)),
                LossTag("outlet", axis=0, side=1),
                LossTag("plasma", axis=0, side=0),
            ),
            exit_tag="outlet",
            max_rounds=max_rounds,
            n_tracked=n_rays if n_tracked is None else n_tracked,
            verbose=verbose,
        ),
    )
    sim = build_steady_state_simulation(
        output_dir,
        f"axisymmetric_extraction_{case.extraction_voltage:g}",
        model,
        n_rays,
        domain,
        (*num_elements, 1),
        (3, 3, 1),
        (
            "remove",
            ("reflect", "remove"),
            "reflect",
        ),  # reflect on the axis cylinder and wedge faces, remove at the wall
    )
    sim.run()
    return model.steady_state_iteration


def exit_phase_space(iteration):
    """Radial phase space (r, r') of the rays at the outlet."""
    exits = iteration.history[-1].exit_records
    x, y, vx, vy, vz = exits[:, 0], exits[:, 1], exits[:, 3], exits[:, 4], exits[:, 5]
    r = np.hypot(x, y)
    vr = (x * vx + y * vy) / np.maximum(r, 1e-12)
    return r, vr / vz, exits[:, 6]


def rms_radial_emittance(r, rp, weights):
    """RMS emittance in one transverse plane of an axisymmetric beam: 4ε_x = ... use ε_x = sqrt(<x²><x'²>-<xx'>²) with x = r cos θ."""
    # for an axisymmetric beam <x²> = <r²>/2, <x'²> = <r'²>/2, <x x'> = <r r'>/2 (no rotation)
    w = weights / weights.sum()
    xx, xpxp, xxp = (np.sum(w * q) / 2 for q in (r * r, rp * rp, r * rp))
    return float(np.sqrt(max(xx * xpxp - xxp**2, 0.0)))


def plot_results(iteration, output, case=DEFAULT):
    model = iteration.model
    domain = iteration.sim.domain
    plasma = model.plasma
    record = iteration.history[-1]

    e1, e2 = np.linspace(0.0, 1.0, 321), np.linspace(0.0, 1.0, 97)
    phi = np.asarray(model.em_fields.phi.spline(e1, e2, np.array([0.5])))[:, :, 0]
    E1, E2 = np.meshgrid(e1, e2, indexing="ij")
    Z, R = meridional(domain, np.stack([E1, E2, 0.5 + 0 * E1], -1))
    volts = UNITS.volts(phi)

    fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.6, 1))
    ax = fig.add_subplot(grid[0, :])
    ax_ne, ax_conv, ax_phase = (fig.add_subplot(grid[1, i]) for i in range(3))
    fig.suptitle(
        f"Struphy axisymmetric extraction (round apertures, IBSimu-style): H⁺, T_e = {case.electron_temperature_eV:g} eV, "
        f"n₀ = {case.electron_density:.2g} m⁻³, φ_P = {case.plasma_potential_V:g} V, puller at −{case.extraction_voltage / 1e3:g} kV"
    )

    zw, rw = case.profile()
    zs = np.linspace(0.0, case.length, 400)
    wall = np.interp(zs, zw, rw)
    gap0, gap1 = case.gap
    for sign in (1, -1):
        for (z0, z1), colour, label in (
            ((0.0, gap0), "#8c8c8c", "Plasma electrode (0 V)"),
            ((gap1, case.length), "#f28e2b", f"Puller (−{case.extraction_voltage / 1e3:g} kV)"),
        ):
            sel = (zs >= z0) & (zs <= z1)
            ax.fill_between(
                zs[sel],
                sign * wall[sel],
                sign * (case.chamber_half_height + 0.4),
                color=colour,
                label=label if sign > 0 else None,
            )
        ax.contour(
            Z, sign * R, volts, levels=np.linspace(-case.extraction_voltage, 0, 16), colors="0.35", linewidths=0.5
        )
        ax.contour(
            Z,
            sign * R,
            volts,
            levels=[case.plasma_potential_V - 2 * case.electron_temperature_eV],
            colors="#59a14f",
            linewidths=1.5,
        )
    if iteration.trajectories is not None:
        zt, rt = meridional(domain, np.nan_to_num(iteration.trajectories, nan=0.0))
        alive = np.all(np.isfinite(iteration.trajectories), axis=-1)
        zt[~alive], rt[~alive] = np.nan, np.nan
        last = np.array([zt[alive[:, j], j][-1] if alive[:, j].any() else 0.0 for j in range(zt.shape[1])])
        transmitted = np.nonzero(last > case.length - 0.2)[0]
        lost = np.random.default_rng(0).permutation(np.nonzero(last <= case.length - 0.2)[0])[:60]
        for sign in (1, -1):
            for j in lost:
                ax.plot(zt[:, j], sign * rt[:, j], color="0.55", linewidth=0.3, alpha=0.7)
            for j in transmitted[:150]:
                ax.plot(zt[:, j], sign * rt[:, j], color="#e15759", linewidth=0.35)
        ax.plot([], [], color="#e15759", label="Extracted rays (mirrored)")
        ax.plot([], [], color="0.55", label="Lost rays (sample)")
    ax.plot([], [], color="#59a14f", label="Meniscus (φ = φ_P − 2T_e)")
    ax.set(
        xlabel="z (mm)",
        ylabel="r (mm)",
        xlim=(0, case.length),
        ylim=(-case.chamber_half_height - 0.4, case.chamber_half_height + 0.4),
        aspect="equal",
    )
    ax.set_title(
        f"{'Converged in' if iteration.converged else 'Stopped (not converged) after'} {len(iteration.history)} rounds: "
        f"transmission {100 * record.exit_current:.1f} %, plasma-electrode loss {100 * record.lost_current['plasma electrode']:.1f} %, "
        f"puller loss {100 * record.lost_current['puller']:.1f} %",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=5)

    ne = -plasma.charge_density(phi) / plasma.density
    for sign in (1, -1):
        image = ax_ne.pcolormesh(Z, sign * R, ne, shading="auto", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax_ne, label="n_e / n₀")
    ax_ne.set(xlabel="z (mm)", ylabel="r (mm)", title="Plasma electrons (Boltzmann)", xlim=(0, 6), aspect="equal")

    changes = [r.potential_change for r in iteration.history[:-1]]
    ax_conv.semilogy(changes, "o-", markersize=3, label="‖Δφ‖/‖φ‖")
    ax_conv.set(xlabel="Iteration round", title=f"Convergence (α = {iteration.alpha:g})")
    ax_conv.legend(fontsize=8)

    r, rp, w = exit_phase_space(iteration)
    emittance = rms_radial_emittance(r, rp, w)
    ax_phase.scatter(r, 1e3 * rp, s=3, alpha=0.6)
    ax_phase.set(
        xlabel="r (mm)", ylabel="r' (mrad)", title=f"Exit phase space: ε_x,rms = {1e3 * emittance:.2f} mm·mrad"
    )
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Wrote {output}")
    return emittance


def main():
    iteration = run(HERE / "output")
    record = iteration.history[-1]
    emittance = plot_results(iteration, HERE / "axisymmetric_extraction.png")
    print(
        f"converged={iteration.converged} after {len(iteration.history)} rounds; transmission "
        f"{100 * record.exit_current:.1f} %, exit emittance (x) {1e3 * emittance:.3f} mm·mrad"
    )


if __name__ == "__main__":
    main()
