"""Verification helpers for the curved segmented-electrode mapping.

Struphy trajectories are compared with a high-order SciPy integration of the
same finite-element field.  SciPy advances the coupled logical-coordinate and
physical-velocity ODE directly, independently of Struphy's split particle
pusher and boundary handling.
"""

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from double_aperture_accelerator import UNITS, build_domain
from struphy.models import IonOpticsElectrostatic
from struphy.models.ion_optics_steady_state import RayBundle, SteadyStateOptions, build_steady_state_simulation
from struphy.pic.ion_beams import LossTag


def trace(output_dir, y0=(-1.5, 0.0, 1.5), num_elements=(64, 16), dt=0.03):
    """Trace zero-current rays through the two curved apertures."""
    domain = build_domain(num_elements=num_elements)
    y0 = np.asarray(y0, dtype=float)
    x0 = 1.0
    eta = np.column_stack(
        [
            np.full(len(y0), x0 / domain.length),
            (y0 + 5.0) / 10.0,
            np.full(len(y0), 0.5),
        ]
    )
    speed = float(UNITS.speed(2e3))
    velocity = np.column_stack([np.full(len(y0), speed), np.zeros(len(y0)), np.zeros(len(y0))])
    model = IonOpticsElectrostatic(
        base_units=UNITS.base_units(),
        electrode_segments=domain.segments,
        electrode_length=domain.length,
        steady_state=SteadyStateOptions(
            rays=RayBundle(eta, velocity, current=1e-12),
            dt=dt,
            max_rounds=1,
            n_tracked=len(y0),
            loss_tags=(
                LossTag("outlet", axis=0, side=1),
                LossTag("wall", axis=1),
                LossTag("inlet", axis=0, side=0),
            ),
            exit_tag="outlet",
        ),
    )
    simulation = build_steady_state_simulation(
        output_dir,
        "segmented_verification",
        model,
        len(y0),
        domain,
        (*num_elements, 1),
        (3, 3, 1),
        ("remove", "remove", "periodic"),
    )
    simulation.run()
    return model.steady_state_iteration


def physical_paths(iteration):
    """Physical Cartesian positions of the Struphy paths."""
    logical = iteration.trajectories
    xyz = np.asarray(
        iteration.sim.domain(
            np.nan_to_num(logical, nan=0.0).reshape(-1, 3),
            change_out_order=True,
            remove_outside=False,
        )
    ).reshape(logical.shape)
    xyz[~np.all(np.isfinite(logical), axis=-1)] = np.nan
    return xyz


def scipy_reference(iteration, ray, t_end=60.0):
    """DOP853 reference for one ray, returning a dense physical path."""
    domain = iteration.sim.domain
    field = iteration.model.em_fields.e_field.spline
    initial = np.concatenate([iteration.rays.eta[ray], iteration.rays.v[ray]])

    def rhs(_, state):
        eta, velocity = state[:3], state[3:]
        inverse = np.asarray(domain.jacobian_inv(*eta, squeeze_out=True, change_out_order=True, remove_outside=False))
        logical_field = np.array([float(component) for component in field(*eta, squeeze_out=True)])
        return np.concatenate([inverse @ velocity, inverse.T @ logical_field])

    def outlet(_, state):
        return state[0] - 1.0

    outlet.terminal = True
    outlet.direction = 1
    solution = solve_ivp(
        rhs, (0.0, t_end), initial, method="DOP853", rtol=2e-10, atol=2e-12, events=outlet, dense_output=True
    )
    stop = solution.t_events[0][0] if len(solution.t_events[0]) else solution.t[-1]
    state = solution.sol(np.linspace(0.0, stop, 2001)).T
    xyz = np.asarray(domain(state[:, :3], change_out_order=True, remove_outside=False))
    return xyz, state[:, 3:]


def transverse_at_planes(path, planes):
    """Interpolate physical y along a monotone physical-x trajectory."""
    valid = np.all(np.isfinite(path[:, :2]), axis=1)
    return np.interp(planes, path[valid, 0], path[valid, 1])


def main():
    iteration = trace(Path(__file__).parent / "output")
    paths = physical_paths(iteration)
    planes = np.linspace(5.0, 75.0, 15)
    for ray in range(len(iteration.rays)):
        reference, _ = scipy_reference(iteration, ray)
        error = np.max(np.abs(transverse_at_planes(paths[:, ray], planes) - transverse_at_planes(reference, planes)))
        print(f"ray {ray}: maximum transverse deviation {1e3 * error:.2f} um")


if __name__ == "__main__":
    main()
