"""Verification of PerturbativeEulerWithQ (compressible Euler equations for the perturbations, variable q = sqrt(p)):
a nonlinear simple sound wave against the exact solution (method of characteristics, before the shock), with the
convergence order in dt, conservation of mass, and the small-amplitude limit (LinearEulerWithQ)."""

import logging

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import (
    BaseUnits,
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
    set_logging_level,
)
from struphy.initial.base import GenericPerturbation
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models import LinearEulerWithQ, PerturbativeEulerWithQ

set_logging_level()
logger = logging.getLogger("struphy")

GAMMA = 5 / 3
N0 = 1.0
P0 = 0.6  # c0 = sqrt(gamma p0 / rho0) = 1
C0 = np.sqrt(GAMMA * P0 / N0)
AMP = 0.05  # velocity amplitude of the simple wave; shock at t = 1 / ((gamma + 1) / 2 * 2 pi AMP) = 2.39
TOL = 1e-13


def _global_max(value: float) -> float:
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        value = comm.allreduce(value, op=MPI.MAX)
    return value


def _simple_wave_state(x):
    """Right-going simple wave with u = AMP sin(2 pi x): Riemann invariant u - 2 c / (gamma - 1) constant,
    isentropic: rho = rho0 (c / c0)^(2 / (gamma - 1)), p = p0 (rho / rho0)^gamma."""
    u = AMP * np.sin(2 * np.pi * x)
    c = C0 + (GAMMA - 1) / 2 * u
    rho = N0 * (c / C0) ** (2 / (GAMMA - 1))
    q = np.sqrt(P0 * (rho / N0) ** GAMMA)
    return u, rho, q


def _exact_velocity(x, t):
    """u(x, t) = u0(xi) with x = xi + (c0 + (gamma + 1) / 2 u0(xi)) t (Newton iteration for xi)."""
    xi = x - C0 * t
    for _ in range(50):
        u0 = AMP * np.sin(2 * np.pi * xi)
        f = xi + (C0 + (GAMMA + 1) / 2 * u0) * t - x
        df = 1 + (GAMMA + 1) / 2 * AMP * 2 * np.pi * np.cos(2 * np.pi * xi) * t
        xi -= f / df
    return AMP * np.sin(2 * np.pi * xi)


def _model(model_class, weights="midpoint"):
    model = model_class(base_units=BaseUnits(kBT=1.0))
    props = model.propagators
    props.sound.options = props.sound.Options(solver_params=SolverParameters(tol=TOL))
    if model_class is PerturbativeEulerWithQ:
        nonlin = NonlinearSolverParameters(tol=1e-12)
        props.nonlin_sound.options = props.nonlin_sound.Options(
            weights=weights, solver_params=SolverParameters(tol=TOL), nonlin_solver=nonlin
        )
        props.mom_adv.options = props.mom_adv.Options(
            weights=weights, solver_params=SolverParameters(tol=TOL), nonlin_solver=nonlin
        )
    return model


def _simple_wave_run(model_class, dt: float, Tend: float, num_elements: int = 128, weights="midpoint"):
    """Max error of the velocity (relative to AMP) at Tend against the exact simple wave, and the mass change."""
    model = _model(model_class, weights)
    u0 = lambda x, y, z: _simple_wave_state(x)[0]
    model.mhd.velocity.add_perturbation(GenericPerturbation(u0, given_in_basis="physical", comp=0))
    if model_class is PerturbativeEulerWithQ:
        rho = lambda x, y, z: _simple_wave_state(x)[1] - N0
        model.mhd.density.add_perturbation(GenericPerturbation(rho, given_in_basis="physical"))
        q = lambda x, y, z: _simple_wave_state(x)[2] - np.sqrt(P0)
    else:
        # linearized simple wave
        q = lambda x, y, z: np.sqrt(P0) * GAMMA / 2 * _simple_wave_state(x)[0] / C0
    model.mhd.sqrt_p.add_perturbation(GenericPerturbation(q, given_in_basis="physical"))

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders="verification_tests", sim_folder="PerturbativeEulerWithQ"),
        time_opts=Time(dt=dt, Tend=Tend, split_algo="Strang"),
        domain=domains.Cuboid(),
        equil=equils.HomogenSlab(B0z=1.0, beta=2 * P0, n0=N0),
        grid=grids.TensorProductGrid(num_elements=(num_elements, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )
    sim.allocate()

    n = model.mhd.density.spline.vector
    ones = n.space.zeros()
    ones[:] = 1.0
    mass0 = n.inner(ones)
    steps = int(round(Tend / dt))
    for _ in range(steps):
        model.integrate(dt, split_algo="Strang")

    eta = (np.arange(200) + 0.5) / 200
    u = model.mhd.velocity.spline(eta, np.array([0.5]), np.array([0.5]), squeeze_out=True)[0]
    error = float(np.max(np.abs(u - _exact_velocity(eta, steps * dt)))) / AMP
    return error, abs(n.inner(ones) - mass0)


def test_simple_wave():
    """Nonlinear simple sound wave before the shock: PerturbativeEulerWithQ follows the exact (steepening) solution,
    with errors much smaller than those of the linear model, converging with second order in dt (midpoint weights,
    Strang splitting); mass is conserved."""
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("evaluation of the spline on the full grid: serial only")
    Tend = 1.0
    errors = []
    for dt in (0.04, 0.02, 0.01):
        error, mass_change = _simple_wave_run(PerturbativeEulerWithQ, dt, Tend)
        errors.append(error)
        logger.info(f"{dt = }: max error of u / A {error:.2e}, mass change {mass_change:.1e}")
        assert mass_change < 1e-13
    error_lin, _ = _simple_wave_run(LinearEulerWithQ, 0.01, Tend)
    rates = [np.log2(e0 / e1) for e0, e1 in zip(errors[:-1], errors[1:])]
    logger.info(f"{errors = }, {rates = }, linear model: {error_lin:.2e}")
    assert errors[-1] < 0.02 * error_lin
    assert errors[-1] < 5e-3
    assert all(rate > 1.8 for rate in rates)


def test_small_amplitude_limit():
    """The difference between PerturbativeEulerWithQ and LinearEulerWithQ after a few steps is quadratic in the amplitude
    of the perturbation (Colella domain, non-uniform density, constant pressure)."""
    equil = equils.GenericCartesianFluidEquilibrium(
        n_xyz=lambda x, y, z: 1.0 + 0.3 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
        p_xyz=lambda x, y, z: P0 * np.ones_like(x),
    )

    def run(model_class, amp):
        model = _model(model_class)
        model.mhd.velocity.add_perturbation(
            perturbations.ModesSin(ls=(1,), ms=(1,), amps=(amp,), comp=0, given_in_basis="physical")
        )
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=(1,), amps=(amp,), comp=1, given_in_basis="physical")
        )
        model.mhd.sqrt_p.add_perturbation(
            perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(amp,), given_in_basis="physical")
        )
        sim = Simulation(
            model=model,
            env=EnvironmentOptions(out_folders="verification_tests", sim_folder="PerturbativeEulerWithQ_small"),
            time_opts=Time(dt=0.05, Tend=0.2),
            domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
            equil=equil,
            grid=grids.TensorProductGrid(num_elements=(8, 8, 1)),
            derham_opts=DerhamOptions(degree=(2, 2, 1)),
        )
        sim.allocate()
        for _ in range(4):
            model.integrate(0.05)
        return model.mhd.velocity.spline.vector.toarray().copy()

    diffs = []
    for amp in (1e-2, 5e-3):
        diff = _global_max(float(xp.max(xp.abs(run(PerturbativeEulerWithQ, amp) - run(LinearEulerWithQ, amp)))))
        size = _global_max(float(xp.max(xp.abs(run(LinearEulerWithQ, amp)))))
        diffs.append(diff / size)
    logger.info(f"relative differences to the linear model {diffs}")
    assert diffs[0] < 0.1
    assert 1.8 < diffs[0] / diffs[1] < 2.2


if __name__ == "__main__":
    test_simple_wave()
    test_small_amplitude_limit()
