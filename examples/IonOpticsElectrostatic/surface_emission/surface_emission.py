"""Surface-emitted ions from the aperture lips, traced together with the plasma-volume ions.

Run from the repository root with::

    python examples/IonOpticsElectrostatic/surface_emission/surface_emission.py

This is the first step towards surface-produced negative ions (Kalvas 2013, §2.3): ions leave the plasma
electrode surface with a fixed energy and a surface current density. The geometry, plasma and volume source are
those of ``plasma_extraction``; the surface source (``RayBundle.from_wall_surface``) covers the sloped and the flat
parts of both aperture lips. As a start the surface ions are the same species as the volume ions (protons), so this
is not yet H⁻ extraction: surface ions on the plasma-facing slope are pushed back to the wall by the sheath, and
those on the aperture side are pulled out by the puller field.

Writes ``surface_emission.png`` next to this file.
"""

import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "plasma_extraction"))

from plasma_extraction import DEFAULT, UNITS, _map_points, build_domain, plasma_model  # noqa: E402

from struphy.models import IonOpticsElectrostatic  # noqa: E402
from struphy.models.ion_optics_steady_state import RayBundle, SteadyStateOptions, build_steady_state_simulation  # noqa: E402
from struphy.pic.ion_beams import LossTag, PlaneSource  # noqa: E402


def surface_rays(domain, case=DEFAULT, n_rays=600, current_density=20.0, energy_eV=1.0, seed=1):
    """Rays from both aperture lips (slope plus flat part): ``current_density`` in A/m^2, ``energy_eV`` per ion."""
    x_range = (case.plasma_lip[0] - case.transition, case.plasma_lip[1])
    j = current_density / UNITS.current_density
    speed = UNITS.speed(energy_eV)
    n = n_rays // 2
    lower = RayBundle.from_wall_surface(domain, "lower", x_range, n, j, speed, seed=seed)
    upper = RayBundle.from_wall_surface(domain, "upper", x_range, n_rays - n, j, speed, seed=seed + 1)
    return lower + upper


def run(
    output_dir,
    case=DEFAULT,
    num_elements=(48, 18),
    n_volume=1800,
    n_surface=600,
    surface_current_density=20.0,
    surface_energy_eV=1.0,
    dt=0.1,
    max_rounds=80,
    n_tracked=None,
    tol=1e-3,
):
    domain = build_domain(case, num_elements)
    plasma = plasma_model(case)
    v_bohm = plasma.bohm_speed()
    volume_source = PlaneSource(
        rate=1.0,
        current=plasma.density * v_bohm * 2.0 * case.chamber_half_height,
        axis=0,
        eta_plane=0.0,
        velocity=(v_bohm, 0.0, 0.0),
        velocity_spread=(0.0, float(np.sqrt(case.transverse_temperature_eV / UNITS.voltage)), 0.0),
    )
    rays = RayBundle.from_plane_source(volume_source, n_volume) + surface_rays(
        domain, case, n_surface, surface_current_density, surface_energy_eV
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
            criterion="residual",
            tol=tol,
            alpha=0.2,
            relaxation="adaptive",
            loss_tags=(
                LossTag("plasma electrode", axis=1, coordinate=0, interval=(-np.inf, x_gap)),
                LossTag("puller", axis=1, coordinate=0, interval=(x_gap, np.inf)),
                LossTag("outlet", axis=0, side=1),
                LossTag("plasma", axis=0, side=0),
            ),
            exit_tag="outlet",
            max_rounds=max_rounds,
            n_tracked=len(rays) if n_tracked is None else n_tracked,
            verbose=True,
        ),
    )
    sim = build_steady_state_simulation(
        output_dir,
        "surface_emission",
        model,
        len(rays),
        domain,
        (*num_elements, 1),
        (3, 3, 1),
        ("remove", "remove", "periodic"),
    )
    sim.run()
    iteration = model.steady_state_iteration
    iteration.n_volume = n_volume  # ray ids below n_volume are plasma-volume rays
    return iteration


def outlet_currents(iteration):
    """Outlet current from volume and from surface rays, as fractions of the total emitted current."""
    exits = iteration.history[-1].exit_records
    total = iteration.rays.current.sum()
    from_surface = exits[:, 8] >= iteration.n_volume
    return exits[~from_surface, 6].sum() / total, exits[from_surface, 6].sum() / total


def plot_results(iteration, output, case=DEFAULT):
    model = iteration.model
    domain = iteration.sim.domain
    record = iteration.history[-1]
    n_volume = iteration.n_volume
    e1, e2 = np.linspace(0.0, 1.0, 321), np.linspace(0.0, 1.0, 121)
    phi = np.asarray(model.em_fields.phi.spline(e1, e2, np.array([0.5])))[:, :, 0]
    xy = np.asarray(domain(e1, e2, np.array([0.5])))
    X, Y = xy[0][:, :, 0], xy[1][:, :, 0]
    volts = UNITS.volts(phi)

    fig, (ax, ax_phase) = plt.subplots(1, 2, figsize=(15, 5.2), width_ratios=(2.6, 1), constrained_layout=True)
    xw, yw = case.profile()
    xs = np.linspace(0.0, case.length, 400)
    top = np.interp(xs, xw, yw)
    gap0, gap1 = case.gap
    for sign in (1, -1):
        for (x0, x1), colour in (((0.0, gap0), "#8c8c8c"), ((gap1, case.length), "#f28e2b")):
            sel = (xs >= x0) & (xs <= x1)
            ax.fill_between(xs[sel], sign * top[sel], sign * (case.chamber_half_height + 0.4), color=colour)
    ax.contour(X, Y, volts, levels=np.linspace(-case.extraction_voltage, 0, 16), colors="0.35", linewidths=0.5)
    ax.contour(X, Y, volts, levels=[case.plasma_potential_V - 2 * case.electron_temperature_eV], colors="#59a14f")
    eta = iteration.trajectories
    if eta is not None:
        xy_rays = _map_points(domain, eta)
        rng = np.random.default_rng(0)
        for j in rng.permutation(n_volume)[:120]:
            ax.plot(xy_rays[:, j, 0], xy_rays[:, j, 1], color="0.6", linewidth=0.3, alpha=0.6)
        for j in range(n_volume, xy_rays.shape[1]):
            ax.plot(xy_rays[:, j, 0], xy_rays[:, j, 1], color="#4e79a7", linewidth=0.35, alpha=0.8)
        ax.plot([], [], color="0.6", label="Plasma-volume rays (sample)")
        ax.plot([], [], color="#4e79a7", label="Surface-emitted rays")
    volume_out, surface_out = outlet_currents(iteration)
    if iteration.converged:
        status = f"converged in {len(iteration.history)} rounds"
    else:
        mean, std = iteration.averaged(10)["exit_current"]
        status = f"NOT converged after {len(iteration.history)} rounds, outlet current {100 * mean:.1f} ± {100 * std:.1f} %"
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        xlim=(0, case.length),
        ylim=(-case.chamber_half_height - 0.4, case.chamber_half_height + 0.4),
        aspect="equal",
    )
    ax.set_title(
        f"Surface emission from the aperture lips ({status})\nlast round: outlet current {100 * volume_out:.1f} % "
        f"from the plasma volume, {100 * surface_out:.1f} % from the surface (of the total emitted current)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    exits = record.exit_records
    surface = exits[:, 8] >= n_volume
    for mask, colour, label in ((~surface, "0.5", "volume"), (surface, "#4e79a7", "surface")):
        ax_phase.scatter(exits[mask, 1], 1e3 * exits[mask, 4] / exits[mask, 3], s=3, alpha=0.6, color=colour, label=label)
    ax_phase.set(xlabel="y (mm)", ylabel="y' (mrad)", title="Exit phase space by origin")
    ax_phase.legend(fontsize=8)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Wrote {output}")


def _draw_electrodes(ax, case):
    xw, yw = case.profile()
    xs = np.linspace(0.0, case.length, 800)
    top = np.interp(xs, xw, yw)
    gap0, gap1 = case.gap
    for sign in (1, -1):
        for (x0, x1), colour in (((0.0, gap0), "#8c8c8c"), ((gap1, case.length), "#f28e2b")):
            sel = (xs >= x0) & (xs <= x1)
            ax.fill_between(xs[sel], sign * top[sel], sign * (case.chamber_half_height + 0.4), color=colour, zorder=3)


def ray_fates(iteration, xy_rays, case=DEFAULT):
    """Fate of every tracked ray from its last stored point: 'outlet', 'puller', 'plasma electrode' or 'plasma'."""
    x_gap = 0.5 * sum(case.gap)
    fates = []
    for j in range(xy_rays.shape[1]):
        finite = np.isfinite(xy_rays[:, j, 0])
        if not np.any(finite):
            fates.append("plasma electrode")  # lost within the first step, next to the wall it started on
            continue
        x = xy_rays[finite, j, 0][-1]
        if x > case.length - 0.2:
            fates.append("outlet")
        elif x < 0.2:
            fates.append("plasma")
        elif x < x_gap:
            fates.append("plasma electrode")
        else:
            fates.append("puller")
    return np.array(fates)


def plot_beam(iteration, output, case=DEFAULT):
    """Beam figure: the whole channel, and a zoom on the aperture with every surface-produced trajectory by fate."""
    model = iteration.model
    domain = iteration.sim.domain
    n_volume = iteration.n_volume
    e1, e2 = np.linspace(0.0, 1.0, 641), np.linspace(0.0, 1.0, 241)
    phi = np.asarray(model.em_fields.phi.spline(e1, e2, np.array([0.5])))[:, :, 0]
    xy = np.asarray(domain(e1, e2, np.array([0.5])))
    X, Y = xy[0][:, :, 0], xy[1][:, :, 0]
    volts = UNITS.volts(phi)
    xy_rays = _map_points(domain, iteration.trajectories)
    fates = ray_fates(iteration, xy_rays, case)
    colours = {"outlet": "#e15759", "puller": "#b07aa1", "plasma electrode": "#4e79a7", "plasma": "#76b7b2"}
    labels = {
        "outlet": "SP ions: extracted",
        "puller": "SP ions: lost on the puller",
        "plasma electrode": "SP ions: back on the plasma electrode",
        "plasma": "SP ions: into the plasma",
    }
    surface = np.arange(xy_rays.shape[1]) >= n_volume
    current = iteration.rays.current
    total_surface = current[surface].sum()

    fig, (ax, ax_zoom) = plt.subplots(2, 1, figsize=(15, 11), height_ratios=(1, 1.25), constrained_layout=True)
    for panel, zoom in ((ax, False), (ax_zoom, True)):
        _draw_electrodes(panel, case)
        levels = np.arange(-case.extraction_voltage, 1, 20.0 if zoom else 100.0)
        panel.contour(X, Y, volts, levels=levels, colors="0.4", linewidths=0.4, zorder=1)
        panel.contour(
            X, Y, volts, levels=[case.plasma_potential_V - 2 * case.electron_temperature_eV], colors="#59a14f", zorder=2
        )
        volume_ids = np.random.default_rng(0).permutation(n_volume)[: (60 if zoom else 150)]
        for j in volume_ids:
            panel.plot(xy_rays[:, j, 0], xy_rays[:, j, 1], color="0.7", linewidth=0.3, alpha=0.7, zorder=1)
        for j in np.nonzero(surface)[0]:
            panel.plot(
                xy_rays[:, j, 0],
                xy_rays[:, j, 1],
                color=colours[fates[j]],
                linewidth=0.5 if zoom else 0.35,
                alpha=0.85,
                zorder=2,
            )
        panel.set(xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    ax.set(xlim=(0, case.length), ylim=(-case.chamber_half_height - 0.4, case.chamber_half_height + 0.4))
    # launch points of the SP ions (the ions turned back by the sheath travel only micrometres)
    launch = np.asarray(domain(iteration.rays.eta[surface], change_out_order=True, remove_outside=False)).reshape(-1, 3)
    ax_zoom.scatter(
        launch[:, 0], launch[:, 1], s=6, c=[colours[f] for f in fates[surface]], zorder=4, linewidths=0, alpha=0.9
    )
    ax_zoom.set(xlim=(case.plasma_lip[0] - 1.0, case.puller_lip[0] + 0.5), ylim=(-2.0, 2.0))
    ax.plot([], [], color="0.7", label="plasma-volume ions (sample)")
    ax.plot([], [], color="#59a14f", label="meniscus (φ = φ_P − 2T_e)")
    for fate in ("outlet", "puller", "plasma electrode", "plasma"):
        share = current[surface][fates[surface] == fate].sum() / total_surface
        ax.plot([], [], color=colours[fate], label=f"{labels[fate]} ({100 * share:.0f} % of the SP current)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    status = f"converged in {len(iteration.history)} rounds" if iteration.converged else "not converged"
    ax.set_title(
        f"Surface-produced (SP) ions from the aperture lips, 1 eV, cosine law, {status}; "
        f"equipotentials every 100 V (top) and 20 V (zoom)",
        fontsize=10,
    )
    ax_zoom.set_title(
        "Aperture region: every SP trajectory and launch point (dots), coloured by where the ion ends", fontsize=10
    )
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    iteration = run(HERE / "output")
    volume_out, surface_out = outlet_currents(iteration)
    print(
        f"converged={iteration.converged} after {len(iteration.history)} rounds; outlet current: "
        f"{100 * volume_out:.1f} % from the volume, {100 * surface_out:.1f} % from the surface"
    )
    plot_results(iteration, HERE / "surface_emission.png")
    plot_beam(iteration, HERE / "surface_emission_beam.png")


if __name__ == "__main__":
    main()
