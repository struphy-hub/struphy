"""Slit immersion lens at increasing beam current (self-consistent space charge).

Run from the repository root with::

    python examples/IonOpticsElectrostatic/slit_lens_space_charge/slit_lens_space_charge.py

The lens, units and injection setup are those of ``../slit_lens_injection``,
with a narrower, almost laminar beam. For each beam current (per mm of slit
length), the model deposits the ion charge and re-solves Poisson's equation, with the
electrode voltages, before every time step. The steady-state beam is then
compared across currents: the space-charge defocusing moves the waist
downstream and enlarges it, and the outlet emittance and transmission change.

Writes ``slit_lens_space_charge.png`` next to this file.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "slit_lens_injection"))
sys.path.insert(0, str(HERE.parent / "slit_immersion_lens"))
from slit_immersion_lens import DESIGN, UNITS, analytic_lens  # noqa: E402
from slit_lens_injection import BEAM, build_simulation  # noqa: E402

from struphy.diagnostics.beam_diagnostics import rms_moments, steady_state_start  # noqa: E402

LAMINAR_BEAM = replace(BEAM, half_width=3e-3, angular_spread=3e-3, markers_per_ns=80.0)
CURRENTS_UA_PER_MM = (0.0, 200.0, 400.0)  # beam current per mm of slit length


def normalized_current(ua_per_mm):
    """µA per mm of slit length -> normalized current (the domain is one length unit thick in z)."""
    return ua_per_mm * 1e-6 / UNITS.current


def run(output_dir, ua_per_mm, **kwargs):
    space_charge = ua_per_mm > 0.0
    sim = build_simulation(
        output_dir,
        beam=kwargs.pop("beam", LAMINAR_BEAM),
        current=normalized_current(ua_per_mm) if space_charge else 1.0,
        space_charge=space_charge,
        name=f"space_charge_{ua_per_mm:g}",
        **kwargs,
    )
    sim.run()
    return sim


def steady_beam(sim):
    """Live marker positions, rms envelope along x, and outlet phase-space moments."""
    model = sim.model
    model.update_ledger()
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    particles = model.ions.var.particles
    live = particles.markers[particles.valid_mks]
    x, y = live[:, 0] * length, -h + 2 * h * live[:, 1]
    bins = np.linspace(2.0, length - 1.0, 156)
    index = np.digitize(x, bins) - 1
    centers = 0.5 * (bins[1:] + bins[:-1])
    rms = np.array([np.std(y[index == i]) if np.count_nonzero(index == i) > 5 else np.nan for i in range(len(centers))])
    exits = model.ledger.records["outlet"]
    # steady window: from where the windowed outlet emittance has converged (Kalvas 2013, §5.8.2)
    t_steady = steady_state_start(exits, window=2.0)
    if t_steady is None:
        raise RuntimeError("No steady outlet emittance reached; increase end_time.")
    exits = exits[exits[:, 7] > t_steady]
    outlet = rms_moments(exits[:, 1], exits[:, 4] / exits[:, 3], exits[:, 6])
    lost = model.ledger.lost_charge
    injected = model.ledger.injected_charge
    transmission = lost["outlet"] / max(injected - model.ions.var.particles.weights.sum(), 1e-300)
    return {
        "x": x,
        "y": y,
        "centers": centers,
        "rms": rms,
        "outlet": outlet,
        "exits": exits,
        "transmission": transmission,
        "t_steady": t_steady,
    }


def waist(result):
    """Position and size of the smallest rms beam size downstream of the gap."""
    downstream = result["centers"] > analytic_lens().xc + 5.0
    i = np.nanargmin(np.where(downstream, result["rms"], np.nan))
    return result["centers"][i], result["rms"][i]


def plot_results(results, output):
    length = DESIGN.length / UNITS.length
    h = DESIGN.half_gap / UNITS.length
    lens = analytic_lens()
    n = len(results)
    fig = plt.figure(figsize=(15, 3.0 * n + 4.2), constrained_layout=True)
    grid = fig.add_gridspec(n + 1, 3, height_ratios=(*([1.0] * n), 1.5))
    fig.suptitle(
        f"Slit immersion lens with space charge: {DESIGN.beam_energy_eV / 1e3:g} keV protons, "
        f"V₂ = {DESIGN.v2 / 1e3:g} kV, ±{LAMINAR_BEAM.half_width * 1e3:g} mm, "
        f"{LAMINAR_BEAM.angular_spread * 1e3:g} mrad rms"
    )
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for row, (current, result) in enumerate(results.items()):
        ax = fig.add_subplot(grid[row, :])
        counts, xe, ye = np.histogram2d(result["x"], result["y"], bins=(160, 50), range=((0, length), (-h, h)))
        ax.pcolormesh(xe, ye, counts.T, cmap="inferno")
        for xs, colour in (((0.0, lens.plate_nodes[0]), "#9c9c9c"), ((lens.plate_nodes[1], length), "#f28e2b")):
            for y_plate in (-h, h):
                ax.plot(xs, (y_plate, y_plate), color=colour, linewidth=5, solid_capstyle="butt")
        x_waist, _ = waist(result)
        ax.axvline(x_waist, color="white", linestyle=":", linewidth=1)
        label = "zero current" if current == 0 else f"{current:g} µA per mm of slit"
        ax.set(ylabel="y (mm)", xlim=(0, length), ylim=(-h * 1.1, h * 1.1))
        ax.set_title(
            f"{label}: waist at x = {x_waist:.1f} mm, transmission {100 * result['transmission']:.1f} %",
            loc="left",
            fontsize=10,
        )
        if row == n - 1:
            ax.set_xlabel("x (mm)")

    ax_env = fig.add_subplot(grid[n, 0])
    ax_waist = fig.add_subplot(grid[n, 1])
    ax_phase = fig.add_subplot(grid[n, 2])
    currents = np.array(list(results))
    for colour, (current, result) in zip(colours, results.items()):
        ax_env.plot(result["centers"], result["rms"], color=colour, label=f"{current:g} µA/mm")
        exits = result["exits"]
        ax_phase.scatter(exits[:, 1], 1e3 * exits[:, 4] / exits[:, 3], s=2, color=colour, alpha=0.4)
    ax_env.axvline(lens.xc, color="0.6", linestyle="--", linewidth=0.8)
    ax_env.set(xlabel="x (mm)", ylabel="RMS beam size (mm)", title="Envelope (dashed: gap centre)")
    ax_env.legend(fontsize=8)

    waists = np.array([waist(r) for r in results.values()])
    emittance = np.array([1e3 * r["outlet"]["emittance"] for r in results.values()])
    ax_waist.plot(currents, waists[:, 0], "o-", label="Waist position (mm)")
    ax_waist.set(
        xlabel="Beam current (µA per mm of slit)", ylabel="Waist position x (mm)", title="Waist and outlet emittance"
    )
    ax_emit = ax_waist.twinx()
    ax_emit.plot(currents, emittance, "s--", color=colours[3])
    ax_emit.set_ylabel("Outlet rms emittance (mm·mrad), squares", color=colours[3])

    ax_phase.set(xlabel="y (mm)", ylabel="y' (mrad)", title="Outlet phase space (colours as envelope)")
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"Wrote {output}")


def main():
    results = {}
    for current in CURRENTS_UA_PER_MM:
        sim = run(HERE / "output", current)
        results[current] = steady_beam(sim)
        x_waist, size = waist(results[current])
        print(
            f"{current:6.1f} µA/mm: waist x = {x_waist:.1f} mm (rms {size:.3f} mm), "
            f"outlet emittance {1e3 * results[current]['outlet']['emittance']:.2f} mm·mrad, "
            f"transmission {100 * results[current]['transmission']:.1f} %"
        )
    plot_results(results, HERE / "slit_lens_space_charge.png")


if __name__ == "__main__":
    main()
