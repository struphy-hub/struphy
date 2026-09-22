"""IBSimu-style 2D positive-ion slit extraction in Struphy.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/plasma_extraction/plasma_extraction.py

A generic slit extraction, not a specific device:

* Plasma chamber on the left, bounded by the plasma electrode (0 V) with a slit aperture.
* Gap, then a puller electrode at ``-V_ext`` with its own slit, followed by a drift region.
  The electrode surfaces are the walls of a ``SegmentedElectrodeChannel``, a conforming mapping.
* Plasma: Boltzmann electrons (T_e, plasma potential phi_P, density n_0). Ions start at
  the back of the chamber with the Bohm velocity and a transverse temperature T_t.
* Steady state: ``SteadyStateIteration`` (ray tracing plus Poisson–Boltzmann Newton on the
  convex energy). The plasma meniscus is not prescribed; it forms where the ion space
  charge and the electrons balance the extraction field.

Writes ``plasma_extraction.png`` next to this file.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent

from struphy.geometry.domains import ElectrodeSegment, SegmentedElectrodeChannel  # noqa: E402
from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.models.ion_optics_steady_state import SteadyStateOptions, build_steady_state_simulation  # noqa: E402
from struphy.physics.ion_optics_units import IonOpticsUnits  # noqa: E402
from struphy.physics.plasma_models import BoltzmannElectrons  # noqa: E402
from struphy.pic.ion_beams import LossTag, PlaneSource  # noqa: E402

UNITS = IonOpticsUnits(length=1e-3, voltage=1e3)  # mm, kV, protons


@dataclass(frozen=True)
class Extraction:
    """Geometry (mm), voltages (V) and plasma (SI) of a generic slit extraction."""

    length: float = 16.0
    chamber_half_height: float = 3.0
    plasma_aperture: float = 1.0  # half-width of the plasma-electrode slit
    plasma_lip: tuple = (2.5, 3.0)  # x range of the plasma-electrode lip
    puller_aperture: float = 1.5
    puller_lip: tuple = (6.5, 7.5)
    transition: float = 0.5  # length of the wall ramps into and out of each lip
    extraction_voltage: float = 1.5e3  # puller at -V_ext
    electron_temperature_eV: float = 5.0
    plasma_potential_V: float = 17.0  # ~ (Te/2)(1 + ln(m_i / 2 pi m_e)) for hydrogen, Kalvas eq. 2.28
    electron_density: float = 2.76e16  # m^-3 -> lambda_D = 0.1 mm
    transverse_temperature_eV: float = 0.5
    wall_nodes: tuple = field(default=None, repr=False)

    def profile(self):
        """Upper wall (x, y) nodes; the lower wall is its mirror image."""
        r, t = self.chamber_half_height, self.transition
        (p0, p1), (q0, q1) = self.plasma_lip, self.puller_lip
        x = (0.0, p0 - t, p0, p1, p1 + t, q0 - t, q0, q1, q1 + t, self.length)
        y = (r, r, self.plasma_aperture, self.plasma_aperture, r, r, self.puller_aperture, self.puller_aperture, r, r)
        return np.array(x), np.array(y)

    @property
    def gap(self):
        """Uncovered wall between the plasma electrode and the puller (natural boundary)."""
        return self.plasma_lip[1] + self.transition, self.puller_lip[0] - self.transition


DEFAULT = Extraction()


def build_domain(case=DEFAULT, num_elements=(160, 60)):
    x, y = case.profile()
    gap0, gap1 = case.gap
    v_puller = float(UNITS.potential(-case.extraction_voltage))
    segments = []
    for side in ("lower", "upper"):
        segments += [
            ElectrodeSegment(side, 0.0, gap0, 0.0, "plasma electrode"),
            ElectrodeSegment(side, gap1, case.length, v_puller, "puller"),
        ]
    return SegmentedElectrodeChannel(
        length=case.length,
        width=1.0,
        lower_profile=(tuple(x), tuple(-y)),
        upper_profile=(tuple(x), tuple(y)),
        segments=tuple(segments),
        num_elements=num_elements,
        degree=(3, 3),
    )


def plasma_model(case=DEFAULT):
    return BoltzmannElectrons.from_si(
        UNITS,
        electron_density=case.electron_density,
        electron_temperature_eV=case.electron_temperature_eV,
        plasma_potential_V=case.plasma_potential_V,
    )


def run(
    output_dir,
    case=DEFAULT,
    num_elements=(48, 18),
    n_rays=900,
    dt=0.1,
    alpha=0.2,
    max_rounds=60,
    n_tracked=None,
    relaxation="adaptive",
    criterion="residual",
    tol=1e-3,
    step_control=None,
):
    domain = build_domain(case, num_elements)
    plasma = plasma_model(case)
    # ions leave the plasma with the Bohm speed and the quasi-neutral flux n_0 v_B, over the chamber height
    v_bohm = plasma.bohm_speed()
    height = 2.0 * case.chamber_half_height
    source = PlaneSource(
        rate=1.0,
        current=plasma.density * v_bohm * height * 1.0,  # z-thickness 1 mm
        axis=0,
        eta_plane=0.0,
        eta_ranges=((0.0, 1.0), (0.0, 1.0)),
        velocity=(v_bohm, 0.0, 0.0),
        velocity_spread=(0.0, float(np.sqrt(case.transverse_temperature_eV / UNITS.voltage)), 0.0),
    )
    x_gap = 0.5 * sum(case.gap)
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_segments=domain.segments,
        electrode_length=domain.length,
        plasma=plasma,
        steady_state=SteadyStateOptions(
            source=source,
            n_rays=n_rays,
            dt=dt,
            criterion=criterion,
            tol=tol,
            alpha=alpha,
            relaxation=relaxation,
            step_control=step_control,
            loss_tags=(
                LossTag("plasma electrode", axis=1, coordinate=0, interval=(-np.inf, x_gap)),
                LossTag("puller", axis=1, coordinate=0, interval=(x_gap, np.inf)),
                LossTag("outlet", axis=0, side=1),
                LossTag("plasma", axis=0, side=0),
            ),
            exit_tag="outlet",
            max_rounds=max_rounds,
            n_tracked=n_rays if n_tracked is None else n_tracked,
            verbose=True,
        ),
    )
    sim = build_steady_state_simulation(
        output_dir,
        f"extraction_{case.extraction_voltage:g}",
        model,
        n_rays,
        domain,
        (*num_elements, 1),
        (3, 3, 1),
        ("remove", "remove", "periodic"),
    )
    sim.run(profiling_activated=True)
    return model.steady_state_iteration


def plot_results(iteration, output, case=DEFAULT):
    model = iteration.model
    model_domain = iteration.sim.domain
    plasma = model.plasma
    record = iteration.history[-1]

    # potential and electron density on the mapped grid
    e1, e2 = np.linspace(0.0, 1.0, 321), np.linspace(0.0, 1.0, 121)
    phi = np.asarray(model.em_fields.phi.spline(e1, e2, np.array([0.5])))[:, :, 0]
    xy = np.asarray(model_domain(e1, e2, np.array([0.5])))
    X, Y = xy[0][:, :, 0], xy[1][:, :, 0]
    volts = UNITS.volts(phi)

    fig = plt.figure(figsize=(15, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.6, 1))
    ax = fig.add_subplot(grid[0, :])
    ax_ne, ax_conv, ax_phase = (fig.add_subplot(grid[1, i]) for i in range(3))
    fig.suptitle(
        f"Struphy slit extraction (IBSimu-style): H⁺, T_e = {case.electron_temperature_eV:g} eV, "
        f"n₀ = {case.electron_density:.2g} m⁻³, φ_P = {case.plasma_potential_V:g} V, "
        f"puller at −{case.extraction_voltage / 1e3:g} kV"
    )

    # electrodes: fill outside the walls
    xw, yw = case.profile()
    xs = np.linspace(0.0, case.length, 400)
    top = np.interp(xs, xw, yw)
    gap0, gap1 = case.gap
    for sign in (1, -1):
        for (x0, x1), colour, label in (
            ((0.0, gap0), "#8c8c8c", "Plasma electrode (0 V)"),
            ((gap1, case.length), "#f28e2b", f"Puller (−{case.extraction_voltage / 1e3:g} kV)"),
        ):
            sel = (xs >= x0) & (xs <= x1)
            ax.fill_between(
                xs[sel],
                sign * top[sel],
                sign * (case.chamber_half_height + 0.4),
                color=colour,
                label=label if sign > 0 else None,
            )
    ax.contour(X, Y, volts, levels=np.linspace(-case.extraction_voltage, 0, 16), colors="0.35", linewidths=0.5)
    ax.contour(
        X,
        Y,
        volts,
        levels=[case.plasma_potential_V - 2 * case.electron_temperature_eV],
        colors="#59a14f",
        linewidths=1.5,
    )
    length = case.length
    eta = iteration.trajectories
    if eta is not None:
        xy_rays = _map_points(model_domain, eta)
        # transmitted rays in red, a random sample of the lost ones in grey (a strided subset of
        # Sobol points would be correlated in y)
        last = np.array([xy_rays[np.isfinite(xy_rays[:, j, 0]), j, 0][-1] for j in range(xy_rays.shape[1])])
        transmitted = np.nonzero(last > case.length - 0.2)[0]
        lost = np.random.default_rng(0).permutation(np.nonzero(last <= case.length - 0.2)[0])[:80]
        for j in lost:
            ax.plot(xy_rays[:, j, 0], xy_rays[:, j, 1], color="0.55", linewidth=0.3, alpha=0.7)
        for j in transmitted:
            ax.plot(xy_rays[:, j, 0], xy_rays[:, j, 1], color="#e15759", linewidth=0.35)
        ax.plot([], [], color="#e15759", label=f"Extracted rays ({len(transmitted)})")
        ax.plot([], [], color="0.55", label="Lost rays (sample)")
    ax.plot([], [], color="#59a14f", label="Meniscus (φ = φ_P − 2T_e)")
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        xlim=(0, length),
        ylim=(-case.chamber_half_height - 0.4, case.chamber_half_height + 0.4),
        aspect="equal",
    )

    def percent(key):
        """Last-round value if converged, else mean ± std of the last 10 rounds (both as % of the emitted current)."""
        if iteration.converged:
            value = record.exit_current if key == "exit_current" else record.lost_current[key]
            return f"{100 * value:.1f} %"
        mean, std = iteration.averaged(10)["exit_current" if key == "exit_current" else f"lost_current[{key}]"]
        return f"{100 * mean:.1f} ± {100 * std:.1f} %"

    status = (
        f"Converged in {len(iteration.history)} rounds (residual {record.residual:.0e})"
        if iteration.converged
        else f"NOT converged after {len(iteration.history)} rounds: mean ± std of the last 10 rounds"
    )
    ax.set_title(
        f"{status}: transmission {percent('exit_current')}, plasma-electrode loss {percent('plasma electrode')}, "
        f"puller loss {percent('puller')}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8, ncol=5)

    ne = -plasma.charge_density(phi) / plasma.density
    image = ax_ne.pcolormesh(X, Y, ne, shading="auto", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax_ne, label="n_e / n₀")
    ax_ne.set(xlabel="x (mm)", ylabel="y (mm)", title="Plasma electrons (Boltzmann)", xlim=(0, 6), aspect="equal")

    residual = [r.residual for r in iteration.history]
    ax_conv.semilogy(residual, "o-", markersize=3, label="Fixed-point residual ‖ρ* − ρ‖/‖ρ*‖")
    ax_conv.axhline(iteration.tol, color="k", linestyle=":", linewidth=0.8, label=f"tol = {iteration.tol:g}")
    ax_conv.set(xlabel="Iteration round", title=f"Convergence ({iteration.relaxation} damping)")
    ax_alpha = ax_conv.twinx()
    ax_alpha.plot([r.alpha for r in iteration.history], color="#59a14f", linewidth=1.2, label="damping α")
    ax_alpha.set(ylabel="damping α", ylim=(0, 1.05))
    handles = ax_conv.get_legend_handles_labels()[0] + ax_alpha.get_legend_handles_labels()[0]
    ax_conv.legend(handles=handles, fontsize=7, loc="lower left")

    exits = record.exit_records
    ax_phase.scatter(exits[:, 1], 1e3 * exits[:, 4] / exits[:, 3], s=3, alpha=0.6)
    ax_phase.set(
        xlabel="y (mm)",
        ylabel="y' (mrad)",
        title=f"Exit phase space: ε_rms = {1e3 * record.exit_emittance:.2f} mm·mrad",
    )
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Wrote {output}")


def _map_points(domain, eta):
    """Physical (x, y) of logical trajectory points ``eta`` (n_steps, n_rays, 3); NaN stays NaN."""
    flat = eta.reshape(-1, 3)
    out = np.full((len(flat), 2), np.nan)
    valid = np.all(np.isfinite(flat), axis=1)
    if np.any(valid):
        xyz = np.asarray(domain(np.clip(flat[valid], 0.0, 1.0), change_out_order=True, remove_outside=False)).reshape(
            -1, 3
        )
        out[valid] = xyz[:, :2]
    return out.reshape(*eta.shape[:2], 2)


def main():
    iteration = run(HERE / "output")
    record = iteration.history[-1]
    print(
        f"converged={iteration.converged} after {len(iteration.history)} rounds; transmission "
        f"{100 * record.exit_current:.1f} %, exit emittance {1e3 * record.exit_emittance:.3f} mm·mrad"
    )
    plot_results(iteration, HERE / "plasma_extraction.png")


if __name__ == "__main__":
    main()
