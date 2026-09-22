"""Steady-state Vlasov–Poisson iteration (IBSimu cycle) vs time-dependent PIC.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/steady_state_iteration/steady_state_iteration.py

``SteadyStateIteration`` traces a fixed set of quasi-random rays through the frozen
field, deposits each ray's charge I·dt along its trajectory, under-relaxes the charge,
solves Poisson, and repeats until the exit emittance satisfies the stopping rule of
Kalvas (2013, §5.8.2). Two cases:

1. **Planar diode** (``../child_langmuir_diode``). Below the Child–Langmuir limit, the
   iteration converges to the steady-state ODE solution. Above the limit, it flips between a
   passing and a fully reflected beam and never converges, as the thesis predicts.
   Time-dependent PIC handles that case.
2. **Slit lens at 400 µA per mm of slit** (``../slit_lens_space_charge``). The converged
   ray-traced beam is compared with the time-dependent run, whose steady window is
   detected with the same emittance criterion.

Writes ``steady_state_iteration.png`` next to this file.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
for sibling in ("child_langmuir_diode", "slit_immersion_lens", "slit_lens_injection", "slit_lens_space_charge"):
    sys.path.insert(0, str(HERE.parent / sibling))

import child_langmuir_diode as diode  # noqa: E402
import slit_lens_space_charge as lens_sc  # noqa: E402
from slit_immersion_lens import DESIGN, UNITS, analytic_lens  # noqa: E402
from slit_lens_injection import loss_tags, source  # noqa: E402

from struphy import domains  # noqa: E402
from struphy.diagnostics.beam_diagnostics import plane_crossings  # noqa: E402
from struphy.initial.perturbations import PiecewiseLinearPotential  # noqa: E402
from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.models.ion_optics_steady_state import SteadyStateOptions, build_steady_state_simulation  # noqa: E402
from struphy.pic.ion_beams import LossTag, PlaneSource  # noqa: E402

LENS_CURRENT_UA_PER_MM = 400.0


def diode_iteration(output_dir, fraction, n_rays=400, dt=0.005, alpha=1.0, max_rounds=40, step_control=None):
    """Steady-state iteration of the planar diode at ``fraction * J_CL``."""
    src = PlaneSource(rate=1.0, current=fraction * diode.J_CL, velocity=(diode.V0, 0.0, 0.0))
    model = IonOpticsElectrostatic(
        base_units=diode.UNITS.base_units(),
        electrode_faces=((True, True), (False, False), (False, False)),
        steady_state=SteadyStateOptions(
            source=src,
            n_rays=n_rays,
            dt=dt,
            alpha=alpha,
            loss_tags=(LossTag("collector", axis=0, side=1), LossTag("emitter", axis=0, side=0)),
            exit_tag="collector",
            max_rounds=max_rounds,
            step_control=step_control,
        ),
    )
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential((0.0, 1.0), (1.0, 0.0), coordinate=0))
    sim = build_steady_state_simulation(
        output_dir,
        f"diode_ss_{fraction:g}",
        model,
        n_rays,
        domains.Cuboid(),
        (64, 1, 1),
        (3, 1, 1),
        ("remove", "periodic", "periodic"),
    )
    sim.run()
    iteration = model.steady_state_iteration
    x = np.linspace(0.0, 1.0, 201)
    phi = np.asarray(model.em_fields.phi.spline(x, np.array([0.5]), np.array([0.5]))).ravel()
    return iteration, x, phi


def lens_iteration(
    output_dir, ua_per_mm=LENS_CURRENT_UA_PER_MM, n_rays=2000, dt=0.04, alpha=1.0, num_elements=(160, 20)
):
    """Steady-state iteration of the slit lens with the laminar beam of ``slit_lens_space_charge``."""
    lens = analytic_lens()
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    src = source(lens_sc.LAMINAR_BEAM, current=lens_sc.normalized_current(ua_per_mm))
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        charge_number=DESIGN.charge_number,
        mass_number=DESIGN.mass_number,
        electrode_faces=((False, False), (True, True), (False, False)),
        steady_state=SteadyStateOptions(
            source=src,
            n_rays=n_rays,
            dt=dt,
            alpha=alpha,
            loss_tags=loss_tags(),
            exit_tag="outlet",
            max_rounds=40,
            n_tracked=n_rays,
        ),
    )
    model.em_fields.phi.add_perturbation(PiecewiseLinearPotential(lens.plate_nodes, (lens.v1, lens.v2), coordinate=0))
    sim = build_steady_state_simulation(
        output_dir,
        f"lens_ss_{ua_per_mm:g}",
        model,
        n_rays,
        domains.Cuboid(l1=0.0, r1=length, l2=-h, r2=h),
        (*num_elements, 1),
        (3, 3, 1),
        ("remove", "remove", "periodic"),
    )
    sim.run()
    return model.steady_state_iteration


def ray_envelope(iteration, planes):
    """RMS size of the traced rays at planes x = const (tracked trajectories of the final round)."""
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    eta = iteration.trajectories
    states = np.stack([eta[..., 0] * length, -h + 2 * h * eta[..., 1]], axis=-1)
    weights = iteration.rays.current
    sizes = []
    for x in planes:
        y = plane_crossings(states, x)[:, 1]
        valid = np.isfinite(y)
        sizes.append(
            np.sqrt(np.average((y[valid] - np.average(y[valid], weights=weights[valid])) ** 2, weights=weights[valid]))
        )
    return np.array(sizes), states


def time_dependent_reference(output_dir, ua_per_mm=LENS_CURRENT_UA_PER_MM, **kwargs):
    """Time-dependent run; its steady window is detected with the emittance criterion (``steady_state_start``)."""
    sim = lens_sc.run(output_dir, ua_per_mm, **kwargs)
    result = lens_sc.steady_beam(sim)
    return {"sim": sim, "result": result, "t_steady": result["t_steady"], "moments": result["outlet"]}


def plot_results(diode_runs, lens_run, reference, output):
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    lens = analytic_lens()
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1.25, 1))
    fig.suptitle("Steady-state Vlasov–Poisson iteration (IBSimu cycle, Kalvas 2013) in Struphy")

    # --- diode ---
    ax_phi, ax_conv, ax_osc = (fig.add_subplot(grid[0, i]) for i in range(3))
    x_ref = np.linspace(0.0, 1.0, 201)
    ax_phi.plot(x_ref, 1.0 - x_ref, "k:", label="Vacuum")
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for colour, (fraction, (iteration, x, phi)) in zip(colours, diode_runs.items()):
        if fraction > 1.0:
            continue
        reference_phi = diode.reference_potential(fraction, x)
        error = np.max(np.abs(phi - reference_phi))
        ax_phi.plot(
            x, phi, color=colour, label=f"{fraction:g} J_CL: {len(iteration.history)} rounds, error {error:.0e}"
        )
        ax_phi.plot(x[::10], reference_phi[::10], "o", color=colour, fillstyle="none")
        # 1D diode: the exit emittance is ~0, so the transit steps serve as the convergence observable
        steps = np.array([r.steps for r in iteration.history], dtype=float)
        ax_conv.semilogy(np.abs(steps - steps[-1]) / steps[-1] + 1e-6, "o-", color=colour, label=f"{fraction:g} J_CL")
    ax_phi.set(xlabel="x / d", ylabel="φ / U", title="Diode: converged potential (circles: steady-state ODE)")
    ax_phi.legend(fontsize=7)
    ax_conv.set(
        xlabel="Iteration round", ylabel="|transit steps − final| / final (+1e-6)", title="Diode: convergence, α = 1"
    )
    ax_conv.legend(fontsize=8)
    iteration = diode_runs[2.0][0]
    ax_osc.plot([r.exit_current for r in iteration.history], "o-", label="Collector")
    ax_osc.plot([r.lost_current["emitter"] for r in iteration.history], "s--", label="Reflected to emitter")
    ax_osc.set(
        xlabel="Iteration round",
        ylabel="Fraction of emitted current",
        title=f"Diode at 2 J_CL: no convergence (converged = {iteration.converged})",
    )
    ax_osc.legend(fontsize=8)

    # --- lens: trajectories ---
    ax_rays = fig.add_subplot(grid[1, :])
    x_grid, y_grid = np.linspace(0, length, 321), np.linspace(-h, h, 81)
    model = lens_run.model
    phi = np.asarray(model.em_fields.phi.spline(x_grid / length, (y_grid + h) / (2 * h), np.array([0.5])))[:, :, 0]
    contours = ax_rays.contourf(x_grid, y_grid, UNITS.volts(phi.T) / 1e3, levels=30, cmap="viridis")
    fig.colorbar(contours, ax=ax_rays, label="Potential (kV)", pad=0.01)
    planes = np.linspace(2.0, length - 0.5, 160)
    sizes, states = ray_envelope(lens_run, planes)
    for j in range(0, states.shape[1], 25):
        ax_rays.plot(states[:, j, 0], states[:, j, 1], color="#e15759", linewidth=0.5)
    for xs, colour in (((0.0, lens.plate_nodes[0]), "#9c9c9c"), ((lens.plate_nodes[1], length), "#f28e2b")):
        for y_plate in (-h, h):
            ax_rays.plot(xs, (y_plate, y_plate), color=colour, linewidth=6, solid_capstyle="butt")
    record = lens_run.history[-1]
    ax_rays.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        xlim=(0, length),
        ylim=(-h * 1.08, h * 1.08),
        title=(
            f"Slit lens, {LENS_CURRENT_UA_PER_MM:g} µA per mm of slit: converged ray tracing "
            f"({len(lens_run.rays)} rays, every 25th drawn; {len(lens_run.history)} rounds). "
            f"The potential includes the beam space charge."
        ),
    )

    # --- lens: comparisons ---
    ax_lconv, ax_env, ax_phase = (fig.add_subplot(grid[2, i]) for i in range(3))
    eps = np.array([r.exit_emittance for r in lens_run.history])
    ax_lconv.semilogy(np.abs(eps - eps[-1]) / eps[-1] + 1e-6, "o-", label="Ray iteration: |ε − ε_final| / ε_final")
    ax_lconv.set(
        xlabel="Iteration round", ylabel="Relative change (+1e-6)", title="Lens: exit-emittance convergence, α = 1"
    )
    ax_lconv.legend(fontsize=8)

    result = reference["result"]
    ax_env.plot(planes, sizes, label="Ray iteration")
    ax_env.plot(result["centers"], result["rms"], alpha=0.7, label="Time-dependent PIC (live markers)")
    ax_env.set(xlabel="x (mm)", ylabel="RMS beam size (mm)", title="Lens: envelope")
    ax_env.legend(fontsize=8)

    exits = record.exit_records
    steady = reference["sim"].model.ledger.records["outlet"]
    steady = steady[steady[:, 7] > reference["t_steady"]]
    ax_phase.scatter(
        steady[:, 1],
        1e3 * steady[:, 4] / steady[:, 3],
        s=2,
        alpha=0.3,
        label=f"PIC, ε = {1e3 * reference['moments']['emittance']:.2f}",
    )
    ax_phase.scatter(
        exits[:, 1],
        1e3 * exits[:, 4] / exits[:, 3],
        s=2,
        alpha=0.5,
        label=f"Ray iteration, ε = {1e3 * record.exit_emittance:.2f}",
    )
    ax_phase.set(xlabel="y (mm)", ylabel="y' (mrad)", title="Lens: outlet phase space (ε in mm·mrad)")
    ax_phase.legend(fontsize=8, markerscale=4)

    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    output_dir = HERE / "output"
    diode_runs = {}
    for fraction in (0.25, 0.5, 0.9, 2.0):
        diode_runs[fraction] = diode_iteration(output_dir, fraction)
        iteration, x, phi = diode_runs[fraction]
        line = f"Diode {fraction:4.2f} J_CL: converged={iteration.converged} after {len(iteration.history)} rounds"
        if fraction <= 1.0:
            line += f", max |phi - phi_ref| = {np.max(np.abs(phi - diode.reference_potential(fraction, x))):.2e}"
        print(line)

    lens_run = lens_iteration(output_dir)
    record = lens_run.history[-1]
    sizes, _ = ray_envelope(lens_run, np.linspace(40.0, 79.5, 400))
    planes = np.linspace(40.0, 79.5, 400)
    print(
        f"Lens ray iteration: converged={lens_run.converged} after {len(lens_run.history)} rounds; "
        f"exit emittance {1e3 * record.exit_emittance:.3f} mm·mrad, exit rms {record.exit_size:.3f} mm, "
        f"waist x = {planes[np.argmin(sizes)]:.1f} mm, transmission {100 * record.exit_current:.1f} %"
    )
    reference = time_dependent_reference(output_dir, beam=replace(lens_sc.LAMINAR_BEAM))
    x_waist, _ = lens_sc.waist(reference["result"])
    print(
        f"Lens time-dependent PIC: steady after t = {reference['t_steady']:.0f}; exit emittance "
        f"{1e3 * reference['moments']['emittance']:.3f} mm·mrad, exit rms {reference['moments']['size']:.3f} mm, "
        f"waist x = {x_waist:.1f} mm"
    )
    plot_results(diode_runs, lens_run, reference, HERE / "steady_state_iteration.png")


if __name__ == "__main__":
    main()
