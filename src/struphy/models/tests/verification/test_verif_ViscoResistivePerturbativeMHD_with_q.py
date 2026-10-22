"""Verification of the ideal perturbative MHD model in q = sqrt(p),
:class:`~struphy.models.visco_resistive_perturbative_mhd_with_q.ViscoResistivePerturbativeMHD_with_q` without viscosity
and resistivity, against conservation laws, analytic wave frequencies and the full-f and linear variational models:

1. the energy with the auxiliary variables q^ and B^ is conserved (non-uniform equilibrium, nonlinear amplitude);
2. a zero perturbation of a non-uniform equilibrium stays exactly zero, while the full-f model
   (ViscoResistiveMHD_with_q) is driven by the discrete equilibrium force;
3. around a homogeneous equilibrium, where the discrete equilibrium force vanishes, the perturbative and full-f
   models give the same solution (the transport operators are linear in their coefficient), and the p-based
   perturbative model (ViscoResistivePerturbativeMHD) agrees up to a time discretization error of first order;
4. in the linear limit the model converges to ViscoResistiveLinearPerturbativeMHD_with_q, and both give the analytic
   frequencies of the shear Alfvén and the fast magnetosonic wave.
"""

import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level
from struphy.models import (
    ViscoResistiveLinearPerturbativeMHD_with_q,
    ViscoResistiveMHD_with_q,
    ViscoResistivePerturbativeMHD,
    ViscoResistivePerturbativeMHD_with_q,
)
from struphy.models.tests.verification.variational_mhd_utils import (
    GAMMA,
    allocate_ideal,
    energy_history,
    homogeneous_slab,
    norm,
    perturb_all_along_z,
    perturb_velocity,
    sheared_slab,
    standing_wave,
    state,
)

set_logging_level()
logger = logging.getLogger("struphy")


def test_energy_conservation(tmp_path):
    """The energy H = int rho |u|^2/2 + (q~^2 + 2 q0 q^)/(gamma-1) + |B~|^2/2 + B0.B^ is conserved, around a
    non-uniform equilibrium and for a nonlinear amplitude."""
    model = allocate_ideal(
        tmp_path,
        "energy",
        ViscoResistivePerturbativeMHD_with_q,
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

    pert = allocate_ideal(tmp_path, "pert", ViscoResistivePerturbativeMHD_with_q, setup, dt)
    for _ in range(n_steps):
        pert.integrate(dt)
    for vec in state(pert):
        assert norm(vec) == 0.0

    full = allocate_ideal(tmp_path, "full", ViscoResistiveMHD_with_q, setup, dt)
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

    pert_0, pert_n = run(ViscoResistivePerturbativeMHD_with_q, perturb_all_along_z(amp, Lz))
    full_0, full_n = run(ViscoResistiveMHD_with_q, perturb_all_along_z(amp, Lz))

    for name, p0, p, f0, f in zip(("rho", "u", "q", "B"), pert_0, pert_n, full_0, full_n):
        # f0 - p0 are the background coefficients of the full-f model
        diff = norm(f - (f0 - p0) - p)
        scale = max(norm(p), 1e-12)
        logger.info(f"{name}: |full - background - pert| / |pert| = {diff / scale:.2e}")
        assert diff < 1e-10 * scale + 1e-14, f"{name}: |full - background - pert| = {diff}, |pert| = {scale}"


def test_same_as_p_model(tmp_path):
    """The q-based and the p-based perturbative model (ViscoResistivePerturbativeMHD) solve the same equations with
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
        q_n = finalstate(ViscoResistivePerturbativeMHD_with_q, dt)
        p_n = finalstate(ViscoResistivePerturbativeMHD, dt)
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

    u_lin, omega_lin = run(ViscoResistiveLinearPerturbativeMHD_with_q, 1.0)
    rel_diff = []
    for amp in (1e-2, 1e-3):
        u_pert, omega_pert = run(ViscoResistivePerturbativeMHD_with_q, amp)
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
