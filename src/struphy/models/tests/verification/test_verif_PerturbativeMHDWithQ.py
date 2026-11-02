"""Verification of PerturbativeMHDWithQ (ideal MHD equations for the perturbations, variable q = sqrt(p)): the
large-amplitude circularly polarized Alfvén wave (an exact nonlinear solution, propagating undistorted), with the
weights of the nonlinear terms at t^n (no numerical damping or growth) and at t^{n+1/2}, and the small-amplitude limit
(LinearMHDWithQ)."""

import logging

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import (
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
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models import LinearMHDWithQ, PerturbativeMHDWithQ

set_logging_level()
logger = logging.getLogger("struphy")

TOL = 1e-13


def _global_max(value: float) -> float:
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        value = comm.allreduce(value, op=MPI.MAX)
    return value


def _set_options(model, weights="frozen"):
    props = model.propagators
    tol = SolverParameters(tol=TOL)
    props.shear_alf.options = props.shear_alf.Options(solver_params=tol)
    props.bg_current.options = props.bg_current.Options(solver_params=tol)
    props.sound.options = props.sound.Options(solver_params=tol)
    if isinstance(model, PerturbativeMHDWithQ):
        nonlin = NonlinearSolverParameters(tol=1e-12)
        for name in ("nonlin_alf", "nonlin_sound", "mom_adv"):
            prop = getattr(props, name)
            prop.options = prop.Options(weights=weights, solver_params=tol, nonlin_solver=nonlin)


@pytest.mark.parametrize("weights", ["frozen", "midpoint"])
def test_circularly_polarized_alfven_wave(weights: str):
    """B = B0 e_x + A (0, cos 2 pi (x - t), sin 2 pi (x - t)), u = -(B - B0 e_x) / sqrt(rho0) with |B| constant is an
    exact solution of ideal MHD for any amplitude A (v_A = 1): after two periods with A = 0.5 B0 the magnetic field
    agrees with it, and the energy of the perturbation is conserved (no numerical damping or growth with either
    weight)."""
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("evaluation of the spline on the full grid: serial only")
    amp = 0.5
    model = PerturbativeMHDWithQ()
    _set_options(model, weights)
    b, u = model.em_fields.b_field, model.mhd.velocity
    b.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(amp,), comp=1, given_in_basis="physical"))
    b.add_perturbation(perturbations.ModesSin(ls=(1,), amps=(amp,), comp=2, given_in_basis="physical"))
    u.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(-amp,), comp=1, given_in_basis="physical"))
    u.add_perturbation(perturbations.ModesSin(ls=(1,), amps=(-amp,), comp=2, given_in_basis="physical"))

    dt, Tend = 0.02, 2.0
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders="verification_tests", sim_folder=f"PerturbativeMHDWithQ_alfven_{weights}"),
        time_opts=Time(dt=dt, Tend=Tend, split_algo="Strang"),
        domain=domains.Cuboid(),
        equil=equils.HomogenSlab(B0x=1.0, B0y=0.0, B0z=0.0, beta=0.5, n0=1.0),
        grid=grids.TensorProductGrid(num_elements=(32, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )
    sim.allocate()

    model.update_scalar_quantities()
    en0 = float(model.scalars.host_value("en_tot")[0])
    steps = int(round(Tend / dt))
    for _ in range(steps):
        model.integrate(dt, split_algo="Strang")
    model.update_scalar_quantities()
    en1 = float(model.scalars.host_value("en_tot")[0])

    eta = (np.arange(100) + 0.5) / 100
    values = b.spline(eta, np.array([0.5]), np.array([0.5]), squeeze_out=True)
    t = steps * dt
    error = max(
        np.max(np.abs(values[1] - amp * np.cos(2 * np.pi * (eta - t)))),
        np.max(np.abs(values[2] - amp * np.sin(2 * np.pi * (eta - t)))),
    )
    amplitude = np.sqrt(np.mean(values[1] ** 2 + values[2] ** 2))
    logger.info(
        f"{weights = }: max error of B / A {error / amp:.2e}, amplitude / A {amplitude / amp:.6f}, "
        f"relative change of en_tot {abs(en1 - en0) / en0:.2e}"
    )
    assert error / amp < 0.02
    assert abs(amplitude / amp - 1) < 1e-3
    assert abs(en1 - en0) / en0 < 1e-6


def test_small_amplitude_limit():
    """The difference between PerturbativeMHDWithQ and LinearMHDWithQ after a few steps is quadratic in the amplitude
    of the perturbation (homogeneous slab in the Colella domain, all variables perturbed)."""

    def run(model_class, amp):
        model = model_class()
        _set_options(model)
        model.mhd.velocity.add_perturbation(
            perturbations.ModesSin(ls=(1,), ms=(1,), amps=(amp,), comp=0, given_in_basis="physical")
        )
        model.mhd.sqrt_p.add_perturbation(
            perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(amp,), given_in_basis="physical")
        )
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesCos(ls=(1,), amps=(amp,), comp=1, given_in_basis="physical")
        )
        model.mhd.density.add_perturbation(perturbations.ModesCos(ms=(1,), amps=(amp,), given_in_basis="physical"))
        sim = Simulation(
            model=model,
            env=EnvironmentOptions(out_folders="verification_tests", sim_folder="PerturbativeMHDWithQ_small"),
            time_opts=Time(dt=0.05, Tend=0.2),
            domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
            equil=equils.HomogenSlab(B0x=0.3, B0z=1.0, beta=1.0, n0=1.0),
            grid=grids.TensorProductGrid(num_elements=(8, 8, 1)),
            derham_opts=DerhamOptions(degree=(2, 2, 1)),
        )
        sim.allocate()
        for _ in range(4):
            model.integrate(0.05)
        return xp.concatenate(
            [model.mhd.velocity.spline.vector.toarray(), model.em_fields.b_field.spline.vector.toarray()]
        )

    diffs = []
    for amp in (1e-2, 5e-3):
        linear = run(LinearMHDWithQ, amp)
        diff = _global_max(float(xp.max(xp.abs(run(PerturbativeMHDWithQ, amp) - linear))))
        diffs.append(diff / _global_max(float(xp.max(xp.abs(linear)))))
    logger.info(f"relative differences to the linear model {diffs}")
    assert diffs[0] < 0.1
    assert 1.8 < diffs[0] / diffs[1] < 2.2


if __name__ == "__main__":
    test_circularly_polarized_alfven_wave("frozen")
    test_circularly_polarized_alfven_wave("midpoint")
    test_small_amplitude_limit()
