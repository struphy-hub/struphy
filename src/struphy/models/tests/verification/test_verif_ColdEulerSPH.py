"""Verification of ColdEulerSPH (pressureless Euler equations, SPH markers): a cold fluid at rest in the harmonic
potential phi = omega^2 (x - L/2)^2 / 2 oscillates as x(t) = L/2 + (x0 - L/2) cos(Omega t), Omega = omega / sqrt(eps).
The quadratic potential is represented exactly by clamped splines of degree 2, so the error is the one of the time
splitting: second order in dt. Mass (the sum of the marker weights) is conserved."""

import logging
import os

import numpy as np
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    LoadingParameters,
    Simulation,
    SortingParameters,
    Time,
    domains,
    equils,
    grids,
)
from struphy.models import ColdEulerSPH

logger = logging.getLogger("struphy")

L, OMEGA, EPS = 4.0, 1.5, 1.0


def _phi(e1, e2, e3):
    """Harmonic potential in logical coordinates of the Cuboid (x = L eta1)."""
    return 0.5 * OMEGA**2 * (L * e1 - L / 2) ** 2


def _errors(out_folder: str, n_steps: int, T: float = 1.0):
    model = ColdEulerSPH(epsilon=EPS)
    model.fluid.set_markers(loading_params=LoadingParameters(Np=200, seed=1234), sorting_params=SortingParameters())
    model.fluid.var.add_background(equils.ConstantVelocity(ux=0.0, uy=0.0))
    model.propagators.push_v.potential = _phi
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=f"n{n_steps}"),
        time_opts=Time(dt=T / n_steps, Tend=T, split_algo="Strang"),
        domain=domains.Cuboid(r1=L),
        equil=equils.ConstantVelocity(ux=0.0),
        grid=grids.TensorProductGrid(num_elements=(8, 1, 1)),
        # clamped splines of degree 2 contain the quadratic potential
        derham_opts=DerhamOptions(degree=(2, 1, 1), bcs=(("free", "free"), None, None)),
    )

    comm = MPI.COMM_WORLD
    parallel = comm.Get_size() > 1

    # initial positions and total mass of all markers (gathered: the markers move across ranks)
    sim.allocate()
    p = model.fluid.var.particles
    m = p.markers[:, p.valid_mks]
    init = np.vstack([m[-1], L * m[0]])
    mass0 = float(np.sum(m[p.index["weights"]]))
    if parallel:
        init = np.hstack(comm.allgather(init))
        mass0 = comm.allreduce(mass0, op=MPI.SUM)
    x0 = dict(zip(init[0].astype(int), init[1]))

    sim.run()
    t = float(sim.time_state["value"][0])
    p = model.fluid.var.particles  # run() allocates again (same seeded markers)
    m = p.markers[:, p.valid_mks]
    Om = OMEGA / np.sqrt(EPS)
    missing = sum(int(i) not in x0 for i in m[-1])
    x0_m = np.array([x0.get(int(i), np.nan) for i in m[-1]])
    err_x = float(np.max(np.abs(L * m[0] - (L / 2 + (x0_m - L / 2) * np.cos(Om * t))), initial=0.0))
    err_u = float(np.max(np.abs(m[3] - (-(x0_m - L / 2) * Om * np.sin(Om * t))), initial=0.0))
    err_uyz = float(np.max(np.abs(m[4:6]), initial=0.0))
    mass = float(np.sum(m[p.index["weights"]]))
    n_markers = m.shape[1]
    if parallel:
        err_x, err_u, err_uyz = (comm.allreduce(e, op=MPI.MAX) for e in (err_x, err_u, err_uyz))
        mass = comm.allreduce(mass, op=MPI.SUM)
        missing, n_markers = (comm.allreduce(c, op=MPI.SUM) for c in (missing, n_markers))
    # checked on all ranks after the reductions (an assertion on one rank only would hang the others)
    assert missing == 0, f"{missing} markers of the run are not in the initial state"
    assert n_markers == len(x0), f"markers lost: {n_markers} of {len(x0)}"
    return err_x, err_u, err_uyz, abs(mass - mass0) / mass0


def test_harmonic_potential(mpi_out_folder: str):
    errs_x, errs_u = [], []
    for n_steps in (10, 20, 40):
        err_x, err_u, err_uyz, mass_change = _errors(mpi_out_folder, n_steps)
        logger.info(f"{n_steps = }: {err_x = :.3e}, {err_u = :.3e}, {err_uyz = :.1e}, {mass_change = :.1e}")
        assert err_uyz < 1e-14, "the force acts along x only"
        assert mass_change < 1e-12
        errs_x.append(err_x)
        errs_u.append(err_u)
    # measured: x 2.8e-3, 7.0e-4, 1.8e-4; u 8.7e-3, 2.2e-3, 5.4e-4
    assert errs_x[-1] < 3e-4 and errs_u[-1] < 1e-3
    for errs in (errs_x, errs_u):
        assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5, f"not second order: {errs}"


if __name__ == "__main__":
    test_harmonic_potential(os.path.join(os.getcwd(), "struphy_verification_tests"))
