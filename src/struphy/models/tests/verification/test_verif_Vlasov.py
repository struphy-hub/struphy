"""Verification of Vlasov (test particles in a static magnetic field): in a uniform B0 = e_z (Cartesian domain)
every marker moves on a helix. The Lorentz-force step is an exact rotation (velocities exact, kinetic energy of each
marker conserved), the Strang splitting with the streaming step gives second-order convergence of the positions in dt.
Without magnetic field, in a curvilinear (Colella) domain, the markers move on straight lines in physical space."""

import logging
import os

import numpy as np
import pytest
from maybempi import MPI

from struphy import EnvironmentOptions, LoadingParameters, Simulation, Time, domains, equils, grids, maxwellians
from struphy.models import Vlasov

logger = logging.getLogger("struphy")

# periodic box with lengths L; Colella is periodic with the same lengths
L = np.array([8.0, 8.0, 8.0])
DOMAINS = {
    "Cuboid": lambda: domains.Cuboid(r1=L[0], r2=L[1], r3=L[2]),
    "Colella": lambda: domains.Colella(Lx=L[0], Ly=L[1], alpha=0.05, Lz=L[2]),
}


def _helix(x0, v0, t, B0z):
    """Exact orbit for dv/dt = v x B0z e_z (normalized cyclotron frequency B0z; straight lines for B0z = 0)."""
    vx, vy, vz = v0
    if B0z == 0.0:
        return x0 + v0 * t, v0
    w = B0z
    c, s = np.cos(w * t), np.sin(w * t)
    v = np.array([vx * c + vy * s, vy * c - vx * s, vz])
    x = x0 + np.array([vx * s + vy * (1 - c), vy * s - vx * (1 - c), vz * w * t]) / w
    return x, v


def _errors(out_folder: str, domain_name: str, B0z: float, n_steps: int, T: float):
    """Run to T with n_steps; return the max errors of positions and velocities and of |v| (all ranks)."""
    model = Vlasov()
    model.kinetic_ions.set_markers(loading_params=LoadingParameters(Np=200, seed=1234), bufsize=1.0)
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    domain = DOMAINS[domain_name]()
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=f"{domain_name}_{n_steps}"),
        time_opts=Time(dt=T / n_steps, Tend=T, split_algo="Strang"),
        domain=domain,
        equil=equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=B0z),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 4)),
    )

    # initial state of all markers (gathered: the markers move across ranks)
    sim.allocate()
    p = model.kinetic_ions.var.particles
    m = p.markers[:, p.valid_mks]
    init = np.vstack([m[-1], domain(m[0:3], remove_outside=False), m[3:6]])
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        init = np.hstack(comm.allgather(init))
    init = {int(i): (init[1:4, k], init[4:7, k]) for k, i in enumerate(init[0])}

    sim.run()
    t = float(sim.time_state["value"][0])
    p = model.kinetic_ions.var.particles  # run() allocates again (same seeded markers)
    m = p.markers[:, p.valid_mks]
    x = domain(m[0:3], remove_outside=False)
    err_x = err_v = err_speed = 0.0
    for k, i in enumerate(m[-1]):
        x0, v0 = init[int(i)]
        x_ex, v_ex = _helix(x0, v0, t, B0z)
        dx = x[:, k] - x_ex
        dx -= L * np.round(dx / L)
        err_x = max(err_x, np.max(np.abs(dx)))
        err_v = max(err_v, np.max(np.abs(m[3:6, k] - v_ex)))
        err_speed = max(err_speed, abs(np.linalg.norm(m[3:6, k]) - np.linalg.norm(v0)))
    if comm.Get_size() > 1:
        err_x, err_v, err_speed = (comm.allreduce(e, op=MPI.MAX) for e in (err_x, err_v, err_speed))
    return err_x, err_v, err_speed


@pytest.mark.parametrize("domain_name, B0z", [("Cuboid", 1.0), ("Colella", 0.0)])
def test_helix_orbits(mpi_out_folder: str, domain_name: str, B0z: float):
    """Markers against the exact orbits at T = 1: helices in a uniform B0 = e_z (T not a multiple of the cyclotron
    period, at which the splitting error vanishes), straight lines without field in the curvilinear Colella domain.
    Exact velocities and speeds, second order in dt for the positions (or better)."""
    errs = []
    for n_steps in (10, 20, 40):
        err_x, err_v, err_speed = _errors(mpi_out_folder, domain_name, B0z, n_steps, T=1.0)
        logger.info(f"{domain_name}, {B0z = }, {n_steps = }: {err_x = :.3e}, {err_v = :.3e}, {err_speed = :.3e}")
        assert err_v < 1e-12, "the Lorentz-force step is an exact rotation"
        assert err_speed < 1e-12, "the kinetic energy of each marker is conserved"
        errs.append(err_x)
    # measured (Cuboid): 1.3e-3, 3.3e-4, 8.3e-5
    assert errs[-1] < 2e-4
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5, f"not second order: {errs}"


if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "struphy_verification_tests")
    test_helix_orbits(out, "Cuboid", 1.0)
    test_helix_orbits(out, "Colella", 0.0)
