"""Axisymmetric space-charge benchmark: a cold uniform round beam in a drift tube.

The beam is represented by equal-current Sobol rays in a thin wedge of
revolution.  For a uniform round beam, Gauss' law gives the edge-envelope
equation

    a''(z) = I / (2 pi v_z**3 a),

in the normalized ion-optics units used here.  The grounded cylindrical wall
does not change the field inside a centred uniform beam.  Comparing the rms
radius with ``a/sqrt(2)`` therefore exercises the cylindrical volume element,
trajectory deposition, Poisson solve, and particle push together.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from struphy.geometry.axisymmetric import AxisymmetricElectrodeChannel, meridional
from struphy.geometry.domains import ElectrodeSegment
from struphy.models import IonOpticsElectrostatic
from struphy.models.ion_optics_steady_state import RayBundle, SteadyStateOptions, build_steady_state_simulation
from struphy.physics.ion_optics_units import IonOpticsUnits
from struphy.pic.ion_beams import LossTag

UNITS = IonOpticsUnits(length=1e-3, voltage=1e3)
LENGTH = 100.0
PIPE_RADIUS = 7.0
BEAM_RADIUS = 2.0
START = 5.0
ENERGY_EV = 5e3
CURRENT_A = 1e-4


def reference_envelope(z, radius=BEAM_RADIUS, current_a=CURRENT_A, energy_ev=ENERGY_EV, start=START):
    """Paraxial cold-fluid envelope of a uniform round beam."""
    z = np.asarray(z, dtype=float)
    speed = float(UNITS.speed(energy_ev))
    current = current_a / UNITS.current
    perveance = current / (2.0 * np.pi * speed**3)

    def rhs(_, state):
        a, slope = state
        return slope, perveance / a

    solution = solve_ivp(rhs, (start, float(np.max(z))), (radius, 0.0), rtol=1e-11, atol=1e-13, dense_output=True)
    return solution.sol(z)[0]


def build_domain(num_elements=(80, 20)):
    segments = (ElectrodeSegment("upper", 0.0, LENGTH, 0.0, "drift tube"),)
    return AxisymmetricElectrodeChannel(
        LENGTH,
        ((0.0, LENGTH), (PIPE_RADIUS, PIPE_RADIUS)),
        segments=segments,
        axis_radius=0.02,
        num_elements=num_elements,
    )


def run(
    output_dir,
    num_elements=(80, 20),
    n_rays=512,
    dt=0.06,
    max_rounds=14,
    planes=np.linspace(0.15, 0.85, 15),
    n_tracked=0,
):
    """Run the steady Vlasov--Poisson round-beam benchmark."""
    domain = build_domain(num_elements)
    speed = float(UNITS.speed(ENERGY_EV))
    current = CURRENT_A / UNITS.current
    rays = RayBundle.axisymmetric_disk(
        domain,
        n_rays,
        z0=START,
        radius=BEAM_RADIUS,
        current_density=current / (np.pi * BEAM_RADIUS**2),
        axial_speed=speed,
        seed=17,
    )
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_segments=domain.segments,
        electrode_length=domain.length,
        steady_state=SteadyStateOptions(
            rays=rays,
            dt=dt,
            alpha=1.0,
            criterion="potential",
            tol=2e-4,
            max_rounds=max_rounds,
            n_tracked=n_tracked,
            planes=(0, planes),
            loss_tags=(
                LossTag("outlet", axis=0, side=1),
                LossTag("wall", axis=1, side=1),
                LossTag("inlet", axis=0, side=0),
            ),
            exit_tag="outlet",
        ),
    )
    simulation = build_steady_state_simulation(
        output_dir,
        "axisymmetric_beam_expansion",
        model,
        n_rays,
        domain,
        (*num_elements, 1),
        (3, 3, 1),
        ("remove", ("reflect", "remove"), "reflect"),
    )
    simulation.run()
    return model.steady_state_iteration


def measured_envelope(iteration):
    """Return physical plane locations and equivalent uniform-beam radii."""
    crossings = iteration.history[-1].plane_crossings
    z_values, radii = [], []
    for states in crossings:
        valid = np.all(np.isfinite(states[:, :3]), axis=1)
        z, r = meridional(iteration.sim.domain, states[valid, :3])
        z_values.append(float(np.mean(z)))
        radii.append(float(np.sqrt(2.0 * np.mean(r**2))))
    return np.asarray(z_values), np.asarray(radii)


def main():
    iteration = run(Path(__file__).parent / "output")
    z, measured = measured_envelope(iteration)
    expected = reference_envelope(z)
    relative = measured / expected - 1.0
    print(f"converged={iteration.converged} after {len(iteration.history)} rounds")
    print(f"maximum round-beam envelope error: {100 * np.max(np.abs(relative)):.2f} %")


if __name__ == "__main__":
    main()
