"""Verification of the ideal perturbative MHD model in q = sqrt(p),
:class:`~struphy.models.visco_resistive_perturbative_mhd_with_q.ViscoResistivePerturbativeMHD_with_q` without viscosity
and resistivity, against conservation laws, analytic wave frequencies and the full-f and linear variational models:

1. the energy with the auxiliary variables q^ and B^ is conserved (non-uniform equilibrium, nonlinear amplitude);
2. a zero perturbation of a non-uniform equilibrium stays exactly zero, while the full-f model
   (ViscoResistiveMHD_with_q) is driven by the discrete equilibrium force;
3. around a homogeneous equilibrium, where the discrete equilibrium force vanishes, the perturbative and full-f
   models give the same solution (the transport operators are linear in their coefficient), and the p-based
   perturbative model (ViscoResistivePerturbativeMHD) agrees up to a time discretization error of first order;
4. in the linear limit the model converges to ViscoResistiveLinearMHD_with_q, and both give the analytic frequencies
   of the shear Alfvén and the fast magnetosonic wave.
"""

import logging

import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    FieldsBackground,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
    set_logging_level,
)
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models import (
    ViscoResistiveLinearMHD_with_q,
    ViscoResistiveMHD_with_q,
    ViscoResistivePerturbativeMHD,
    ViscoResistivePerturbativeMHD_with_q,
)
from struphy.propagators.base import Propagator

set_logging_level()
logger = logging.getLogger("struphy")

GAMMA = 5.0 / 3.0

# default solver tolerances (the nonlinear iterations stop at a residual of 1e-8); with much smaller tolerances the
# Jacobian solves break down once the iteration has converged
SOLVER = SolverParameters()
NONLIN = NonlinearSolverParameters()

# variational propagator option `model` for each model class
VARIANTS = {
    ViscoResistivePerturbativeMHD_with_q: "perturbative_q",
    ViscoResistiveMHD_with_q: "full_q",
    ViscoResistiveLinearMHD_with_q: "linear_q",
    ViscoResistivePerturbativeMHD: "perturbative",
}


def _sheared_slab():
    """Non-uniform equilibrium, periodic in x: B = B0 (e_z + a / (q(x) R0) e_y) with q(x) = q0 + q1 sin(2 pi x / a)
    and the pressure in force balance."""
    equil = equils.ShearedSlab(a=1.0, R0=3.0, B0=1.0, q0=1.05, q1=0.5, beta=1.0, q_kind=1)
    domain = domains.Cuboid(r1=1.0, r2=2 * xp.pi, r3=2 * xp.pi * 3.0)
    return equil, domain, (16, 1, 1), (3, 1, 1)


def _homogeneous_slab(B0=(0.0, 1.0, 1.0), Lz=10.0, num_elements=16):
    """Homogeneous equilibrium, waves along z."""
    equil = equils.HomogenSlab(B0x=B0[0], B0y=B0[1], B0z=B0[2], beta=1.0, n0=0.7)
    domain = domains.Cuboid(r3=Lz)
    return equil, domain, (1, 1, num_elements), (1, 1, 3)


def _allocate(tmp_path, name, model_class, setup, dt, perturb=None):
    """Allocate an ideal model of `model_class` for the `setup` (equil, domain, num_elements, degree).

    `perturb(model)` adds the perturbations of the perturbative variables. The full-f model gets the equilibrium
    as background in addition."""
    equil, domain, num_elements, degree = setup
    model = model_class(with_viscosity=False, with_resistivity=False)
    variant = VARIANTS[model_class]

    model.propagators.variat_dens.options = model.propagators.variat_dens.Options(
        model=variant, solver_params=SOLVER, nonlin_solver=NONLIN
    )
    if hasattr(model.propagators, "variat_mom"):
        model.propagators.variat_mom.options = model.propagators.variat_mom.Options(
            solver_params=SOLVER, nonlin_solver=NONLIN
        )
    if variant == "perturbative":
        model.propagators.variat_pb.options = model.propagators.variat_pb.Options(
            model=variant, solver_params=SOLVER, nonlin_solver=NONLIN
        )
    else:
        model.propagators.variat_qb.options = model.propagators.variat_qb.Options(
            model=variant, solver_params=SOLVER, nonlin_solver=NONLIN
        )

    if variant == "full_q":
        model.mhd.density.add_background(FieldsBackground(type="FluidEquilibrium", variable="n3"))
        model.mhd.sqrt_p.add_background(FieldsBackground(type="FluidEquilibrium", variable="q3"))
        model.em_fields.b_field.add_background(FieldsBackground(type="FluidEquilibrium", variable="b2"))

    if perturb is not None:
        perturb(model)

    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder=name),
        time_opts=Time(dt=dt, Tend=dt),
        domain=domain,
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=degree),
    )
    sim.allocate()
    return model


def _state(model):
    """Local coefficient arrays of (rho, u, q or p, B), comparable between different simulations."""
    thermo = model.mhd.sqrt_p if hasattr(model.mhd, "sqrt_p") else model.mhd.pressure
    return [
        var.spline.vector.toarray() for var in (model.mhd.density, model.mhd.velocity, thermo, model.em_fields.b_field)
    ]


def _norm(arr):
    """Euclidean norm of a coefficient array distributed over the MPI ranks."""
    return float(xp.sqrt(MPI.COMM_WORLD.allreduce(float(xp.sum(arr**2)), op=MPI.SUM)))


def _perturb_velocity(amp, direction="x", L=1.0):
    """Smooth perturbation of all (physical) velocity components, varying along x or z (length L of the domain)."""
    modes = "ls" if direction == "x" else "ns"
    lengths = {"Lx": L} if direction == "x" else {"Lz": L}

    def perturb(model):
        for comp, (mode, phase) in enumerate(((1, 0.0), (1, 1.0), (2, 2.0))):
            model.mhd.velocity.add_perturbation(
                perturbations.ModesSin(
                    **{modes: (mode,)}, amps=(amp,), theta=(phase,), comp=comp, given_in_basis="physical", **lengths
                )
            )

    return perturb


def _perturb_all_along_z(amp, Lz):
    """Perturbation of u, q and B along z (B~_x(z), B~_y(z) keep div B = 0)."""

    def perturb(model):
        _perturb_velocity(amp, direction="z", L=Lz)(model)
        model.mhd.sqrt_p.add_perturbation(perturbations.ModesCos(ns=(2,), amps=(amp,), given_in_basis="3"))
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesSin(ns=(1,), amps=(amp,), comp=0, given_in_basis="2")
        )
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesCos(ns=(2,), amps=(amp,), comp=1, given_in_basis="2")
        )

    return perturb


def test_energy_conservation(tmp_path):
    """The energy H = int rho |u|^2/2 + (q~^2 + 2 q0 q^)/(gamma-1) + |B~|^2/2 + B0.B^ is conserved, around a
    non-uniform equilibrium and for a nonlinear amplitude."""
    model = _allocate(
        tmp_path,
        "energy",
        ViscoResistivePerturbativeMHD_with_q,
        _sheared_slab(),
        dt=0.05,
        perturb=_perturb_velocity(0.05),
    )
    energies = []
    for _ in range(10):
        model.update_scalar_quantities()
        energies.append(model.scalars.host_value("en_tot")[0])
        model.integrate(0.05)
    model.update_scalar_quantities()
    energies.append(model.scalars.host_value("en_tot")[0])

    en_U = model.scalars.host_value("en_U")[0]
    assert en_U > 1e-4, "the perturbation must not have died out"
    # the energy is exchanged between the terms ...
    assert abs(model.scalars.host_value("en_thermo_2")[0]) > 1e-6
    assert abs(model.scalars.host_value("en_mag_2")[0]) > 1e-6
    # ... while the total is conserved up to the solver tolerance
    drift = max(abs(en - energies[0]) for en in energies)
    logger.info(f"relative energy drift {drift / abs(energies[0]):.2e}, en_U {en_U:.3e}")
    assert drift < 1e-8 * abs(energies[0]), energies


def test_equilibrium_preservation(tmp_path):
    """A zero perturbation of a non-uniform equilibrium stays exactly zero; the full-f model, initialized with the
    projected equilibrium, is driven by the discrete equilibrium force."""
    setup = _sheared_slab()
    dt, n_steps = 0.05, 5

    pert = _allocate(tmp_path, "pert", ViscoResistivePerturbativeMHD_with_q, setup, dt)
    for _ in range(n_steps):
        pert.integrate(dt)
    for vec in _state(pert):
        assert _norm(vec) == 0.0

    full = _allocate(tmp_path, "full", ViscoResistiveMHD_with_q, setup, dt)
    for _ in range(n_steps):
        full.integrate(dt)
    u_full = _norm(_state(full)[1])
    logger.info(f"|u| of the full-f model driven by the discrete equilibrium force: {u_full:.2e}")
    assert u_full > 1e-8


def test_same_as_full_f(tmp_path):
    """Around a homogeneous equilibrium the perturbative and the full-f model give the same discrete solution, also
    for a nonlinear amplitude."""
    Lz = 10.0
    setup = _homogeneous_slab(Lz=Lz)
    dt, n_steps, amp = 0.05, 10, 0.1

    # the simulations run one after the other, since they share the class attributes of Propagator (derham, ...)
    def run(model_class, perturb):
        """Initial and final coefficient arrays."""
        model = _allocate(tmp_path, model_class.__name__, model_class, setup, dt, perturb)
        initial = _state(model)
        for _ in range(n_steps):
            model.integrate(dt)
        return initial, _state(model)

    pert_0, pert_n = run(ViscoResistivePerturbativeMHD_with_q, _perturb_all_along_z(amp, Lz))
    full_0, full_n = run(ViscoResistiveMHD_with_q, _perturb_all_along_z(amp, Lz))

    for name, p0, p, f0, f in zip(("rho", "u", "q", "B"), pert_0, pert_n, full_0, full_n):
        # f0 - p0 are the background coefficients of the full-f model
        diff = _norm(f - (f0 - p0) - p)
        scale = max(_norm(p), 1e-12)
        logger.info(f"{name}: |full - background - pert| / |pert| = {diff / scale:.2e}")
        assert diff < 1e-10 * scale + 1e-14, f"{name}: |full - background - pert| = {diff}, |pert| = {scale}"


def test_same_as_p_model(tmp_path):
    """The q-based and the p-based perturbative model (ViscoResistivePerturbativeMHD) solve the same equations with
    different time discretizations: for the same initial velocity (zero q~ and p~), u and B agree up to a difference
    that is of first order in the time step (it does not change under grid refinement)."""
    Lz, amp, t_end = 10.0, 0.1, 0.5
    setup = _homogeneous_slab(Lz=Lz)

    def final_state(model_class, dt):
        model = _allocate(
            tmp_path, f"{model_class.__name__}_{dt}", model_class, setup, dt, _perturb_velocity(amp, "z", Lz)
        )
        for _ in range(int(round(t_end / dt))):
            model.integrate(dt)
        return _state(model)

    rel_diff = {}
    for dt in (0.05, 0.025):
        q_n = final_state(ViscoResistivePerturbativeMHD_with_q, dt)
        p_n = final_state(ViscoResistivePerturbativeMHD, dt)
        rel_diff[dt] = [_norm(q_n[i] - p_n[i]) / _norm(q_n[i]) for i in (1, 3)]

    logger.info(f"relative difference q-model / p-model (u, B) per dt: {rel_diff}")
    for i, name in enumerate(("u", "B")):
        assert rel_diff[0.05][i] < 3e-2, (name, rel_diff)
        assert rel_diff[0.025][i] < 0.6 * rel_diff[0.05][i], (name, rel_diff)


@pytest.mark.parametrize("wave", ["alfven", "fast"])
def test_linear_limit_and_waves(tmp_path, wave):
    """For small amplitudes the perturbative model converges to the linear model (difference O(amp^2)), and both give
    the analytic frequency of a standing wave with k = 2 pi / Lz along z:
    shear Alfvén (B0 along z, u_x) with omega = k |B0| / sqrt(n0) and fast magnetosonic (B0 along y, u_z) with
    omega = k sqrt((gamma p0 + |B0|^2) / n0)."""
    Lz, n0 = 10.0, 0.7
    B0 = (0.0, 0.0, 1.0) if wave == "alfven" else (0.0, 1.0, 0.0)
    comp = 0 if wave == "alfven" else 2
    setup = _homogeneous_slab(B0=B0, Lz=Lz, num_elements=32)
    k = 2 * xp.pi / Lz
    p0 = 1.0 * 1.0 / 2  # beta |B0|^2 / 2
    omega_exact = k / xp.sqrt(n0) if wave == "alfven" else k * xp.sqrt((GAMMA * p0 + 1.0) / n0)
    period = 2 * xp.pi / omega_exact
    n_steps = 40
    dt = period / 2 / n_steps

    def perturb(amp):
        def f(model):
            model.mhd.velocity.add_perturbation(
                perturbations.ModesSin(ns=(1,), amps=(amp,), comp=comp, Lz=Lz, given_in_basis="physical")
            )

        return f

    def run(model_class, amp):
        """Velocity coefficients over half a period, and the frequency from the first zero of the projection of u(t)
        on u(0)."""
        model = _allocate(tmp_path, f"{model_class.__name__}_{amp}", model_class, setup, dt, perturb(amp))
        Mv = Propagator.mass_ops.Mv
        u0 = model.mhd.velocity.spline.vector.copy()
        norm0 = Mv.dot_inner(u0, u0)
        proj = [1.0]
        for _ in range(n_steps):
            model.integrate(dt)
            proj.append(Mv.dot_inner(model.mhd.velocity.spline.vector, u0) / norm0)
        proj = xp.array(proj)
        i = int(xp.argmax(proj < 0.0))
        assert i > 0, proj
        t_zero = dt * (i - 1 + proj[i - 1] / (proj[i - 1] - proj[i]))
        return model.mhd.velocity.spline.vector.toarray(), xp.pi / 2 / t_zero

    u_lin, omega_lin = run(ViscoResistiveLinearMHD_with_q, 1.0)
    rel_diff = []
    for amp in (1e-2, 1e-3):
        u_pert, omega_pert = run(ViscoResistivePerturbativeMHD_with_q, amp)
        rel_diff.append(_norm(u_pert / amp - u_lin) / _norm(u_lin))
        logger.info(f"{wave}: amp {amp}, omega {omega_pert:.5f}")
        assert abs(omega_pert - omega_exact) < 1e-2 * omega_exact, (omega_pert, omega_exact)

    logger.info(
        f"{wave}: omega exact {omega_exact:.5f}, linear {omega_lin:.5f}, relative difference to linear {rel_diff}"
    )
    assert abs(omega_lin - omega_exact) < 1e-2 * omega_exact, (omega_lin, omega_exact)
    assert rel_diff[0] < 5e-2, rel_diff
    # the nonlinear correction decreases (at least linearly) with the amplitude
    assert rel_diff[1] < rel_diff[0] / 5 + 1e-8, rel_diff
