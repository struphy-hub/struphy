"""Verification of ColdPlasma (Maxwell's equations with a cold-electron current): the O-mode dispersion relation
omega^2 = k^2 + alpha^2 n0 / eps^2 with second-order convergence in dt, and exact energy conservation of the splitting
(each step conserves the energy) with the j x B0 term active."""

import logging
import os
from dataclasses import fields

import numpy as np

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, perturbations
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import ColdPlasma
from struphy.propagators.base import Propagator

logger = logging.getLogger("struphy")

ALPHA, EPS, N0 = 2.0, 1.0, 1.0
LX = 2 * np.pi


def _simulation(out_folder: str, name: str, dt: float, comp: int, ls=(1,), amps=(1e-3,)) -> Simulation:
    """Uniform B0 = e_z, density n0, a perturbation of the component ``comp`` of E with wave vector along x."""
    model = ColdPlasma(alpha=ALPHA, epsilon=EPS)
    model.em_fields.e_field.add_perturbation(
        perturbations.ModesCos(ls=ls, amps=amps, comp=comp, given_in_basis="physical", Lx=LX)
    )
    return Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=name),
        time_opts=Time(dt=dt, Tend=dt, split_algo="Strang"),
        domain=domains.Cuboid(r1=LX),
        equil=equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, n0=N0),
        grid=grids.TensorProductGrid(num_elements=(32, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )


def _o_mode_frequency(out_folder: str, dt: float, T: float = 10.0) -> float:
    """Frequency of the standing O-mode E_z = cos(k x) cos(omega t), from the zero crossings of the projection of E
    on its initial value."""
    sim = _simulation(out_folder, f"o_mode_{dt}", dt, comp=2)
    sim.allocate()
    model = sim.model
    M1 = Propagator.mass_ops.M1
    e0 = model.em_fields.e_field.spline.vector.copy()
    norm0 = M1.dot_inner(e0, e0)
    ts, a = [0.0], [1.0]
    for k in range(int(round(T / dt))):
        model.integrate(dt, split_algo="Strang")
        ts.append((k + 1) * dt)
        a.append(M1.dot_inner(model.em_fields.e_field.spline.vector, e0) / norm0)
    ts, a = np.array(ts), np.array(a)
    # zero crossings t_j = (j + 1/2) pi / omega (linear interpolation), least-squares fit of omega
    idx = np.nonzero(np.sign(a[:-1]) != np.sign(a[1:]))[0]
    tz = ts[idx] - a[idx] * (ts[idx + 1] - ts[idx]) / (a[idx + 1] - a[idx])
    j = np.arange(len(tz))
    assert len(tz) >= 5
    return float(np.pi * np.sum((j + 0.5) * tz) / np.sum(tz**2))


def test_o_mode_dispersion(mpi_out_folder: str):
    """E along B0, k perpendicular: j x B0 = 0 and omega^2 = k^2 + omega_p^2 (omega_p = alpha sqrt(n0) / eps). The
    implicit (Crank-Nicolson) steps are second order: the frequency error decreases by 4 when dt is halved."""
    k = 2 * np.pi / LX
    exact = np.sqrt(k**2 + ALPHA**2 * N0 / EPS**2)
    errs = []
    for dt in (0.1, 0.05, 0.025):
        omega = _o_mode_frequency(mpi_out_folder, dt)
        errs.append(abs(omega / exact - 1))
        logger.info(f"{dt = }: {omega = :.6f}, {exact = :.6f}, relative error {errs[-1]:.2e}")
    # measured: 1.05e-3, 2.6e-4, 6.5e-5
    assert errs[-1] < 2e-4
    assert errs[0] / errs[1] > 3.5 and errs[1] / errs[2] > 3.5, f"not second order: {errs}"


def test_energy_conservation(mpi_out_folder: str):
    """E perpendicular to B0 (X-mode polarization, two wave numbers): all three steps (Maxwell, Ohm, j x B0) conserve
    en_tot = en_E + en_B + en_J exactly, up to the tolerance of their linear solvers (here 1e-13)."""
    dt = 0.05
    sim = _simulation(mpi_out_folder, "energy", dt, comp=1, ls=(1, 2), amps=(1e-3, 5e-4))
    model = sim.model
    for prop in vars(model.propagators).values():
        names = {f.name for f in fields(prop.Options)}
        if "solver_params" in names:
            prop.options = prop.Options(solver_params=SolverParameters(tol=1e-13, maxiter=10000))
    sim.allocate()

    model.update_scalar_quantities()
    en0 = float(model.scalars.host_value("en_tot")[0])
    drift = 0.0
    en_J = []
    for _ in range(100):
        model.integrate(dt, split_algo="Strang")
        model.update_scalar_quantities()
        drift = max(drift, abs(float(model.scalars.host_value("en_tot")[0]) - en0))
        en_J.append(float(model.scalars.host_value("en_J")[0]))
    logger.info(f"relative energy drift {drift / en0:.2e}, en_J in [{min(en_J):.2e}, {max(en_J):.2e}]")
    # energy is exchanged with the current (the cold-electron term is active) ...
    assert max(en_J) > 0.5 * en0
    # ... and the total is conserved (measured 3e-14; with the default solver tolerance 1e-8: 2e-5)
    assert drift < 1e-10 * en0


if __name__ == "__main__":
    out = os.path.join(os.getcwd(), "struphy_verification_tests")
    test_o_mode_dispersion(out)
    test_energy_conservation(out)
