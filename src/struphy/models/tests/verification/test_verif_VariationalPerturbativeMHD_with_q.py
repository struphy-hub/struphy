"""Verification of the ideal perturbative MHD model in q = sqrt(p),
:class:`~struphy.models.variational_perturbative_mhd_with_q.VariationalPerturbativeMHD_with_q` without viscosity and
resistivity, and of its linear version (``linear=True``), against conservation laws, analytic wave frequencies and
the full-f and p-based variational models.

Perturbative model:

1. the energy with the auxiliary variables q^ and B^ is conserved (non-uniform equilibrium, nonlinear amplitude);
2. a zero perturbation of a non-uniform equilibrium stays exactly zero, while the full-f model
   (VariationalMHD_with_q) is driven by the discrete equilibrium force;
3. around a homogeneous equilibrium, where the discrete equilibrium force vanishes, the perturbative and full-f
   models give the same solution (the transport operators are linear in their coefficient), and the p-based
   perturbative model (VariationalPerturbativeMHD) agrees up to a time discretization error of first order;
4. in the linear limit the model converges to the linear model, and both give the analytic frequencies of the shear
   Alfvén and the fast magnetosonic wave.

Linear model:

5. the energy is conserved around a non-uniform (sheared slab) equilibrium, where the terms with grad q0 and curl B0
   are active;
6. the discrete model is linear in the initial perturbation;
7. it is the linear limit of the perturbative model, also around the sheared slab;
8. it gives the analytic frequencies of the shear Alfvén, the fast magnetosonic and the sound wave in a homogeneous
   slab, with second-order convergence in the time step;
9. the p-based linear model agrees up to its first-order time discretization error.
"""

import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level
from struphy.models import (
    VariationalMHD_with_q,
    VariationalPerturbativeMHD,
    VariationalPerturbativeMHD_with_q,
)
from struphy.models.tests.verification.variational_mhd_utils import (
    GAMMA,
    allocate_ideal,
    energy_history,
    homogeneous_slab,
    linear,
    norm,
    perturb_all_along_z,
    perturb_velocity,
    sheared_slab,
    standing_wave,
    state,
)

set_logging_level()
logger = logging.getLogger("struphy")

LINEAR = linear(VariationalPerturbativeMHD_with_q)


def test_energy_conservation(tmp_path):
    """The energy H = int rho |u|^2/2 + (q~^2 + 2 q0 q^)/(gamma-1) + |B~|^2/2 + B0.B^ is conserved, around a
    non-uniform equilibrium and for a nonlinear amplitude."""
    model = allocate_ideal(
        tmp_path,
        "energy",
        VariationalPerturbativeMHD_with_q,
        sheared_slab(),
        dt=0.05,
        perturb=perturb_velocity(0.05),
    )
    energies = energy_history(model, dt=0.05, n_steps=10)

    en_U = model.scalars.host_value("en_U")[0]
    assert en_U > 1e-4, "the perturbation must not have died out"
    # the energy is exchanged between the terms ...
    assert abs(model.scalars.host_value("en_thermo_2")[0]) > 1e-6
    assert abs(model.scalars.host_value("en_mag_2")[0]) > 1e-6
    # ... while the total is conserved up to the solver tolerance
    drift = max(abs(en - energies[0]) for en in energies)
    logger.info(f"relative energy drift {drift / abs(energies[0]):.2e}, en_U {en_U:.3e}")
    assert drift < 2e-8 * abs(energies[0]), energies


def test_equilibrium_preservation(tmp_path):
    """A zero perturbation of a non-uniform equilibrium stays exactly zero; the full-f model, initialized with the
    projected equilibrium, is driven by the discrete equilibrium force."""
    setup = sheared_slab()
    dt, n_steps = 0.05, 5

    pert = allocate_ideal(tmp_path, "pert", VariationalPerturbativeMHD_with_q, setup, dt)
    for _ in range(n_steps):
        pert.integrate(dt)
    for vec in state(pert):
        assert norm(vec) == 0.0

    full = allocate_ideal(tmp_path, "full", VariationalMHD_with_q, setup, dt)
    for _ in range(n_steps):
        full.integrate(dt)
    u_full = norm(state(full)[1])
    logger.info(f"|u| of the full-f model driven by the discrete equilibrium force: {u_full:.2e}")
    assert u_full > 1e-8


def test_same_as_full_f(tmp_path):
    """Around a homogeneous equilibrium the perturbative and the full-f model give the same discrete solution, also
    for a nonlinear amplitude.

    The discrete force of the equilibrium vanishes here, and the backgrounds of the full-f model (projections of the
    equilibrium n3, q3, b2) coincide with the projected equilibrium used inside the perturbative propagators, since
    the equilibrium is constant and the map (Cuboid) is affine. For a non-uniform equilibrium the two models differ
    by the discrete equilibrium force (see test_equilibrium_preservation)."""
    Lz = 10.0
    setup = homogeneous_slab(Lz=Lz)
    dt, n_steps, amp = 0.05, 10, 0.1

    # the simulations run one after the other, since they share the class attributes of Propagator (derham, ...)
    def run(model_class, perturb):
        """Initial and final coefficient arrays (both models with the transport operators at the midpoint)."""
        model = allocate_ideal(tmp_path, model_class.__name__, model_class, setup, dt, perturb, implicit_weights=True)
        initial = state(model)
        for _ in range(n_steps):
            model.integrate(dt)
        return initial, state(model)

    pert_0, pert_n = run(VariationalPerturbativeMHD_with_q, perturb_all_along_z(amp, Lz))
    full_0, full_n = run(VariationalMHD_with_q, perturb_all_along_z(amp, Lz))

    for name, p0, p, f0, f in zip(("rho", "u", "q", "B"), pert_0, pert_n, full_0, full_n):
        # f0 - p0 are the background coefficients of the full-f model
        diff = norm(f - (f0 - p0) - p)
        scale = max(norm(p), 1e-12)
        logger.info(f"{name}: |full - background - pert| / |pert| = {diff / scale:.2e}")
        assert diff < 1e-10 * scale + 1e-14, f"{name}: |full - background - pert| = {diff}, |pert| = {scale}"


def test_same_as_p_model(tmp_path):
    """The q-based and the p-based perturbative model (VariationalPerturbativeMHD) solve the same equations with
    different time discretizations: for the same initial velocity (zero q~ and p~), u and B agree up to a difference
    that is of first order in the time step."""
    Lz, amp, t_end = 10.0, 0.1, 0.5
    setup = homogeneous_slab(Lz=Lz)

    def finalstate(model_class, dt):
        model = allocate_ideal(
            tmp_path, f"{model_class.__name__}_{dt}", model_class, setup, dt, perturb_velocity(amp, "z", Lz)
        )
        for _ in range(int(round(t_end / dt))):
            model.integrate(dt)
        return state(model)

    rel_diff = {}
    for dt in (0.05, 0.025):
        q_n = finalstate(VariationalPerturbativeMHD_with_q, dt)
        p_n = finalstate(VariationalPerturbativeMHD, dt)
        rel_diff[dt] = [norm(q_n[i] - p_n[i]) / norm(q_n[i]) for i in (1, 3)]

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
    setup = homogeneous_slab(B0=B0, Lz=Lz, num_elements=32)
    k = 2 * xp.pi / Lz
    p0 = 1.0 * 1.0 / 2  # beta |B0|^2 / 2
    omega_exact = k / xp.sqrt(n0) if wave == "alfven" else k * xp.sqrt((GAMMA * p0 + 1.0) / n0)
    period = 2 * xp.pi / omega_exact
    n_steps = 40
    dt = period / 2 / n_steps

    def run(model_class, amp):
        return standing_wave(tmp_path, model_class, setup, Lz, comp, amp, dt, n_steps)

    u_lin, omega_lin = run(LINEAR, 1.0)
    rel_diff = []
    for amp in (1e-2, 1e-3):
        u_pert, omega_pert = run(VariationalPerturbativeMHD_with_q, amp)
        rel_diff.append(norm(u_pert / amp - u_lin) / norm(u_lin))
        logger.info(f"{wave}: amp {amp}, omega {omega_pert:.5f}")
        assert abs(omega_pert - omega_exact) < 2e-3 * omega_exact, (omega_pert, omega_exact)

    logger.info(
        f"{wave}: omega exact {omega_exact:.5f}, linear {omega_lin:.5f}, relative difference to linear {rel_diff}"
    )
    assert abs(omega_lin - omega_exact) < 2e-3 * omega_exact, (omega_lin, omega_exact)
    assert rel_diff[0] < 5e-2, rel_diff
    # the relative nonlinear correction decreases linearly with the amplitude (factor 10 here; checked with margin)
    assert rel_diff[1] < rel_diff[0] / 5 + 1e-8, rel_diff


# linear model


def _final_state(tmp_path, model_class, setup, amp, dt, n_steps):
    """Coefficient arrays after n_steps, started from a velocity perturbation of amplitude amp."""
    model = allocate_ideal(
        tmp_path, f"{model_class.__name__}_{amp}_{dt}", model_class, setup, dt, perturb_velocity(amp)
    )
    for _ in range(n_steps):
        model.integrate(dt)
    return state(model)


def test_linear_energy_conservation(tmp_path):
    """H = int rho0 |u|^2/2 + (q~^2 + 2 q0 q^)/(gamma-1) + |B~|^2/2 + B0.B^ is conserved around the sheared slab."""
    model = allocate_ideal(tmp_path, "energy", LINEAR, sheared_slab(), dt=0.05, perturb=perturb_velocity(0.05))
    energies = energy_history(model, dt=0.05, n_steps=10)

    # the second-order energies are active ...
    assert abs(model.scalars.host_value("en_thermo_2")[0]) > 1e-5
    assert abs(model.scalars.host_value("en_mag_2")[0]) > 1e-5
    # ... and the total is conserved up to the solver tolerance
    drift = max(abs(en - energies[0]) for en in energies)
    logger.info(f"relative energy drift {drift / abs(energies[0]):.2e}")
    assert drift < 2e-8 * abs(energies[0]), energies


def test_linearity(tmp_path):
    """Doubling the initial perturbation doubles the solution (up to the solver tolerance)."""
    setup = sheared_slab()
    single = _final_state(tmp_path, LINEAR, setup, 0.05, 0.05, 10)
    double = _final_state(tmp_path, LINEAR, setup, 0.1, 0.05, 10)
    for i, name in ((1, "u"), (2, "q"), (3, "B")):
        rel = norm(double[i] - 2 * single[i]) / norm(double[i])
        logger.info(f"linearity {name}: {rel:.2e}")
        assert rel < 1e-6, (name, rel)


def test_limit_of_perturbative_model(tmp_path):
    """Around the sheared slab, the perturbative model divided by the amplitude converges to the linear model, with a
    relative difference proportional to the amplitude."""
    setup = sheared_slab()
    u_lin = _final_state(tmp_path, LINEAR, setup, 1.0, 0.05, 10)[1]
    rel_diff = []
    for amp in (1e-2, 1e-3):
        u_pert = _final_state(tmp_path, VariationalPerturbativeMHD_with_q, setup, amp, 0.05, 10)[1]
        rel_diff.append(norm(u_pert / amp - u_lin) / norm(u_lin))
    logger.info(f"relative difference perturbative / linear for amplitudes 1e-2, 1e-3: {rel_diff}")
    assert rel_diff[0] < 5e-2, rel_diff
    # proportional to the amplitude (factor 10 here; checked with margin)
    assert rel_diff[1] < rel_diff[0] / 5, rel_diff


@pytest.mark.parametrize("wave", ["alfven", "fast", "sound"])
def test_linear_waves(tmp_path, wave):
    """Standing waves with k = 2 pi / Lz along z in a homogeneous slab (beta = 1, n0 = 0.7): shear Alfvén (B0 along
    z, u_x) with omega = k |B0| / sqrt(n0), fast magnetosonic (B0 along y, u_z) with omega = k sqrt((gamma p0 + |B0|^2)
    / n0) and sound (B0 along z, u_z) with omega = k sqrt(gamma p0 / n0). The frequency error is that of the implicit
    midpoint rule, of second order in the time step."""
    Lz, n0, p0 = 10.0, 0.7, 0.5
    k = 2 * xp.pi / Lz
    B0, comp, omega_exact = {
        "alfven": ((0.0, 0.0, 1.0), 0, k / xp.sqrt(n0)),
        "fast": ((0.0, 1.0, 0.0), 2, k * xp.sqrt((GAMMA * p0 + 1.0) / n0)),
        "sound": ((0.0, 0.0, 1.0), 2, k * xp.sqrt(GAMMA * p0 / n0)),
    }[wave]
    setup = homogeneous_slab(B0=B0, Lz=Lz, num_elements=32)
    half_period = xp.pi / omega_exact

    errors = []
    for n_steps in (20, 40):
        _, omega = standing_wave(tmp_path, LINEAR, setup, Lz, comp, 1.0, half_period / n_steps, n_steps)
        errors.append(abs(omega - omega_exact) / omega_exact)
    logger.info(f"{wave}: relative frequency errors for 20 and 40 steps per half period: {errors}")
    assert errors[1] < 1e-3, errors
    assert errors[0] / errors[1] > 3.0, errors


def test_linear_same_as_p_model(tmp_path):
    """The p-based linear model (VariationalPerturbativeMHD with linear=True) solves the same equations; its time
    step is of first order (operator weights at t^n), so around the sheared slab the two models agree up to a
    difference of first order in the time step."""
    setup = sheared_slab()
    t_end = 0.5
    rel_diff = {}
    for dt in (0.05, 0.025):
        n_steps = int(round(t_end / dt))
        q_n = _final_state(tmp_path, LINEAR, setup, 0.05, dt, n_steps)
        p_n = _final_state(tmp_path, linear(VariationalPerturbativeMHD), setup, 0.05, dt, n_steps)
        rel_diff[dt] = [norm(q_n[i] - p_n[i]) / norm(q_n[i]) for i in (1, 3)]
    logger.info(f"relative difference q-model / p-model (u, B) per dt: {rel_diff}")
    for i, name in enumerate(("u", "B")):
        assert rel_diff[0.05][i] < 0.6, (name, rel_diff)
        assert rel_diff[0.025][i] < 0.6 * rel_diff[0.05][i], (name, rel_diff)
