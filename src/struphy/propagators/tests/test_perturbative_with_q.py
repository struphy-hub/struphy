"""Tests of the steps of the perturbative models with the variable q = sqrt(p): SoundWaveNonlinearWithQ,
ShearAlfvenNonlinear, MomentumAdvection, and the linear steps with the inertia of the total density (rho_field)."""

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import EnvironmentOptions, Simulation, Time, domains, equils, perturbations
from struphy.feec.density_weighted import HdivDensityOperators, VelocityMass
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models import LinearMHDWithQ, PerturbativeEulerWithQ, PerturbativeMHDWithQ
from struphy.propagators.base import Propagator
from struphy.propagators.momentum_advection import MomentumAdvection
from struphy.propagators.shear_alfven_nonlinear import ShearAlfvenNonlinear
from struphy.propagators.sound_wave_nonlinear_with_q import SoundWaveNonlinearWithQ
from struphy.topology import grids

GAMMA = 5 / 3
TOL = 1e-14
NONLIN = NonlinearSolverParameters(tol=1e-13, maxiter=200)


def allocate_model(tmp_path, name, model_class=PerturbativeMHDWithQ, weights="frozen", amp=0.2):
    """A model with smooth finite-amplitude perturbations of all variables in the Colella domain (non-uniform density),
    allocated, with tight solver tolerances."""
    model = model_class()
    m = model.mhd
    for c in range(3):
        m.velocity.add_perturbation(
            perturbations.ModesSin(ls=(1,), ms=(c % 2,), amps=(amp / 2,), comp=c, given_in_basis="physical")
        )
    m.sqrt_p.add_perturbation(perturbations.ModesCos(ls=(1,), ms=(1,), amps=(amp,), given_in_basis="physical"))
    m.density.add_perturbation(perturbations.ModesCos(ls=(2,), amps=(amp,), given_in_basis="physical"))
    if hasattr(model, "em_fields"):
        b = model.em_fields.b_field
        b.add_perturbation(perturbations.ModesCos(ms=(1,), amps=(amp,), comp=0, given_in_basis="physical"))
        b.add_perturbation(perturbations.ModesCos(ls=(1,), amps=(amp,), comp=2, given_in_basis="physical"))

    props = model.propagators
    tol = SolverParameters(tol=TOL, maxiter=5000)
    props.sound.options = props.sound.Options(solver_params=tol)
    if hasattr(props, "shear_alf"):
        props.shear_alf.options = props.shear_alf.Options(solver_params=tol)
        props.bg_current.options = props.bg_current.Options(solver_params=tol)
    if hasattr(props, "nonlin_sound"):
        for name_ in ("nonlin_sound", "mom_adv") + (("nonlin_alf",) if hasattr(props, "nonlin_alf") else ()):
            prop = getattr(props, name_)
            prop.options = prop.Options(weights=weights, solver_params=tol, nonlin_solver=NONLIN)

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=name),
        time_opts=Time(dt=0.05, Tend=0.05),
        domain=domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0),
        equil=equils.HomogenSlab(B0x=0.2, B0z=1.0, beta=1.0, n0=1.0)
        if model_class is not PerturbativeEulerWithQ
        else equils.GenericCartesianFluidEquilibrium(
            n_xyz=lambda x, y, z: 1.0 + 0.3 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y),
            p_xyz=lambda x, y, z: 0.6 * np.ones_like(x),
        ),
        grid=grids.TensorProductGrid(num_elements=(8, 8, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()
    return model


def kinetic_energy(model):
    """1/2 u^T M2_rho u of the total density."""
    mass = VelocityMass(
        Propagator.derham, Propagator.domain, Propagator.mass_ops, Propagator.projected_equil, model.mhd.density
    )
    u = model.mhd.velocity.spline.vector
    return u.inner(mass.M.dot(u)) / 2


def q_energy(model, coef):
    q = model.mhd.sqrt_p.spline.vector
    return coef * q.inner(Propagator.mass_ops.M3.dot(q))


def b_energy(model):
    b = model.em_fields.b_field.spline.vector
    return b.inner(Propagator.mass_ops.M2.dot(b)) / 2


@pytest.mark.mpi_skip
def test_options():
    for prop_class in (SoundWaveNonlinearWithQ, ShearAlfvenNonlinear, MomentumAdvection):
        assert prop_class.Options().weights == "frozen"
        with pytest.raises(AssertionError):
            prop_class.Options(weights="explicit")
        with pytest.raises(AssertionError):
            prop_class.Options(nonlin_solver=NonlinearSolverParameters(type="Newton"))
    with pytest.raises(AssertionError):
        SoundWaveNonlinearWithQ(rho_field=LinearMHDWithQ().mhd.velocity)
    model = PerturbativeMHDWithQ()
    assert model.propagators.sound.rho_field is model.mhd.density
    assert model.propagators.shear_alf.rho_field is model.mhd.density
    assert model.propagators.bg_current.rho_field is model.mhd.density
    assert LinearMHDWithQ().propagators.sound.rho_field is None


@pytest.mark.parametrize("weights", ["frozen", "midpoint"])
def test_energy_conservation_nonlinear_steps(tmp_path, weights):
    """The nonlinear sound wave and shear-Alfven steps conserve their energies exactly (to the solver tolerances), with
    the inertia of the total density, while energy is exchanged; the advection step conserves the kinetic energy and
    the mass."""
    model = allocate_model(tmp_path, f"nonlin_{weights}", weights=weights)
    props = model.propagators
    cases = (
        ("nonlin_sound", lambda: kinetic_energy(model) + q_energy(model, 1 / (GAMMA - 1))),
        ("nonlin_alf", lambda: kinetic_energy(model) + b_energy(model)),
        ("mom_adv", lambda: kinetic_energy(model)),
    )
    n = model.mhd.density.spline.vector
    ones = n.space.zeros()
    ones[:] = 1.0
    for name, energy in cases:
        e0, k0, mass0 = energy(), kinetic_energy(model), n.inner(ones)
        for _ in range(5):
            getattr(props, name)(0.05)
        change = abs(energy() - e0) / e0
        exchange = abs(kinetic_energy(model) - k0) / e0
        print(f"{name}, {weights}: relative energy change {change:.2e}, exchange {exchange:.2e}")
        assert change < 1e-10
        assert abs(n.inner(ones) - mass0) < 1e-14
        if name != "mom_adv":
            assert exchange > 1e-3


def test_energy_conservation_linear_steps_with_density(tmp_path):
    """The linear sound wave and shear-Alfven steps with the inertia of the total density conserve their energies
    exactly and do not change the density."""
    model = allocate_model(tmp_path, "linear_rho")
    props = model.propagators
    n0 = model.mhd.density.spline.vector.toarray().copy()
    for name, energy in (
        ("sound", lambda: kinetic_energy(model) + q_energy(model, 2 / GAMMA)),
        ("shear_alf", lambda: kinetic_energy(model) + b_energy(model)),
    ):
        e0 = energy()
        for _ in range(5):
            getattr(props, name)(0.1)
        change = abs(energy() - e0) / e0
        print(f"{name}: relative energy change {change:.2e}")
        assert change < 1e-10
    assert xp.max(xp.abs(model.mhd.density.spline.vector.toarray() - n0)) == 0.0


def test_density_sign_check(tmp_path, caplog):
    """HdivDensityOperators.update_density reports the minimum of the total density at the quadrature points and warns
    (once) when it is non-positive."""
    import logging

    allocate_model(tmp_path, "sign")
    ops = HdivDensityOperators(Propagator.derham, Propagator.domain, Propagator.mass_ops)
    rho = Propagator.projected_equil.n3.copy()
    ops.update_density(rho)
    assert ops.min_density > 0.0
    rho *= -1.0
    with caplog.at_level(logging.WARNING, logger="struphy"):
        ops.update_density(rho)
        ops.update_density(rho)
    assert ops.min_density < 0.0
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert caplog.text.count("non-positive") == 1


def test_density_weighted_operators(tmp_path):
    """HdivDensityOperators: the kinetic linear form gives rho^T k(u) = u^T M_rho u exactly, the vorticity matrix is
    antisymmetric, and for a uniform density on the Cuboid M_rho equals M2n."""
    model = allocate_model(tmp_path, "operators")
    ops = HdivDensityOperators(Propagator.derham, Propagator.domain, Propagator.mass_ops)
    u = model.mhd.velocity.spline.vector
    rho = Propagator.projected_equil.n3.copy()
    rho += model.mhd.density.spline.vector
    ops.update_density(rho)
    k = ops.kinetic_form(u, out=rho.space.zeros())
    uMu = u.inner(ops.M.dot(u))
    assert abs(rho.inner(k) - uMu) < 1e-13 * uMu
    omega = model.em_fields.b_field.spline.vector  # any 2-form for the test: use its weak curl
    from feectools.linalg.solvers import inverse

    from struphy.feec import preconditioner

    M1 = Propagator.mass_ops.M1
    M1_inv = inverse(M1, "pcg", pc=preconditioner.MassMatrixPreconditioner(M1), tol=1e-14)
    omega1 = M1_inv.dot(Propagator.derham.curl.T.dot(Propagator.mass_ops.M2.dot(omega)))
    ops.update_vorticity(rho, omega1)
    v = Propagator.mass_ops.M2.dot(u)
    Wu, Wv = ops.W.dot(u), ops.W.dot(v)
    scale = xp.sqrt(Wu.inner(Wu) * v.inner(v))
    assert abs(u.inner(Wu)) < 1e-13 * scale
    assert abs(v.inner(Wu) + u.inner(Wv)) < 1e-13 * scale
    assert abs(v.inner(Wu)) > 1e-6 * scale


def _burgers(tmp_path, weights, dt, T=0.25, num_elements=64, amp=0.1):
    """Pressureless 1D Burgers u_t + u u_x = 0 with u0 = amp sin(2 pi x), rho0 = 1 (mom_adv only): max errors of u (relative to
    amp) and of the density perturbation against the solution by characteristics (shock at t = 1.6)."""
    model = PerturbativeEulerWithQ()
    model.mhd.velocity.add_perturbation(perturbations.ModesSin(ls=(1,), amps=(amp,), comp=0, given_in_basis="physical"))
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=f"burgers_{weights}_{dt}"),
        time_opts=Time(dt=dt, Tend=T),
        domain=domains.Cuboid(),
        equil=equils.HomogenSlab(B0z=1.0, beta=1.0, n0=1.0),
        grid=grids.TensorProductGrid(num_elements=(num_elements, 1, 1)),
        derham_opts=DerhamOptions(degree=(3, 1, 1)),
    )
    sim.allocate()
    adv = model.propagators.mom_adv
    adv.options = MomentumAdvection.Options(weights=weights, nonlin_solver=NONLIN)
    adv.allocate()
    steps = int(round(T / dt))
    for _ in range(steps):
        adv(dt)
    t = steps * dt
    xi = (np.arange(200) + 0.5) / 200
    x = np.mod(xi + t * amp * np.sin(2 * np.pi * xi), 1.0)
    order = np.argsort(x)
    eta = x[order]
    u_ex = amp * np.sin(2 * np.pi * xi[order])
    rho_ex = 1 / (1 + t * 2 * np.pi * amp * np.cos(2 * np.pi * xi[order])) - 1
    u = model.mhd.velocity.spline(eta, np.array([0.5]), np.array([0.5]), squeeze_out=True)[0]
    n = model.mhd.density.spline(eta, np.array([0.5]), np.array([0.5]), squeeze_out=True)
    return np.max(np.abs(u - u_ex)) / amp, np.max(np.abs(n - rho_ex))


@pytest.mark.parametrize("weights, order", [("frozen", 1), ("midpoint", 2)])
def test_momentum_advection_burgers(tmp_path, weights, order):
    """MomentumAdvection reproduces pressureless 1D Burgers flow (exact solution by characteristics), converging in dt
    with first (frozen weights) or second (midpoint weights) order."""
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("evaluation of the spline on the full grid: serial only")
    errors = [_burgers(tmp_path, weights, dt)[0] for dt in (0.05, 0.025, 0.0125)]
    rates = [np.log2(e0 / e1) for e0, e1 in zip(errors[:-1], errors[1:])]
    print(f"{weights}: {errors = }, {rates = }")
    assert all(abs(rate - order) < 0.2 for rate in rates)
    assert errors[-1] < (2e-3 if order == 1 else 1e-4)
