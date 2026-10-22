"""Verification of the ideal linear perturbative MHD model in q = sqrt(p),
:class:`~struphy.models.visco_resistive_linear_perturbative_mhd_with_q.ViscoResistiveLinearPerturbativeMHD_with_q`
without viscosity and resistivity (ideal linear MHD around a zero-flow equilibrium, written in q):

1. the energy with the auxiliary variables q^ and B^ is conserved around a non-uniform (sheared slab) equilibrium,
   where the terms with grad q0 and curl B0 are active;
2. the discrete model is linear in the initial perturbation;
3. it is the linear limit of the perturbative model ViscoResistivePerturbativeMHD_with_q, also around the
   sheared slab;
4. it gives the analytic frequencies of the shear Alfvén, the fast magnetosonic and the sound wave in a homogeneous
   slab, with second-order convergence in the time step;
5. the p-based linear model ViscoResistiveLinearPerturbativeMHD agrees up to its first-order time discretization
   error.
"""

import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level
from struphy.models import (
    ViscoResistiveLinearPerturbativeMHD,
    ViscoResistiveLinearPerturbativeMHD_with_q,
    ViscoResistivePerturbativeMHD_with_q,
)
from struphy.models.tests.verification.variational_mhd_utils import (
    GAMMA,
    allocate_ideal,
    energy_history,
    homogeneous_slab,
    norm,
    perturb_velocity,
    sheared_slab,
    standing_wave,
    state,
)

set_logging_level()
logger = logging.getLogger("struphy")

LINEAR = ViscoResistiveLinearPerturbativeMHD_with_q


def _final_state(tmp_path, model_class, setup, amp, dt, n_steps):
    """Coefficient arrays after n_steps, started from a velocity perturbation of amplitude amp."""
    model = allocate_ideal(
        tmp_path, f"{model_class.__name__}_{amp}_{dt}", model_class, setup, dt, perturb_velocity(amp)
    )
    for _ in range(n_steps):
        model.integrate(dt)
    return state(model)


def test_energy_conservation(tmp_path):
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
        u_pert = _final_state(tmp_path, ViscoResistivePerturbativeMHD_with_q, setup, amp, 0.05, 10)[1]
        rel_diff.append(norm(u_pert / amp - u_lin) / norm(u_lin))
    logger.info(f"relative difference perturbative / linear for amplitudes 1e-2, 1e-3: {rel_diff}")
    assert rel_diff[0] < 5e-2, rel_diff
    # proportional to the amplitude (factor 10 here; checked with margin)
    assert rel_diff[1] < rel_diff[0] / 5, rel_diff


@pytest.mark.parametrize("wave", ["alfven", "fast", "sound"])
def test_waves(tmp_path, wave):
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


def test_same_as_p_model(tmp_path):
    """The p-based linear model (ViscoResistiveLinearPerturbativeMHD) solves the same equations; its time step is of
    first order (operator weights at t^n), so around the sheared slab the two models agree up to a difference of
    first order in the time step."""
    setup = sheared_slab()
    t_end = 0.5
    rel_diff = {}
    for dt in (0.05, 0.025):
        n_steps = int(round(t_end / dt))
        q_n = _final_state(tmp_path, LINEAR, setup, 0.05, dt, n_steps)
        p_n = _final_state(tmp_path, ViscoResistiveLinearPerturbativeMHD, setup, 0.05, dt, n_steps)
        rel_diff[dt] = [norm(q_n[i] - p_n[i]) / norm(q_n[i]) for i in (1, 3)]
    logger.info(f"relative difference q-model / p-model (u, B) per dt: {rel_diff}")
    for i, name in enumerate(("u", "B")):
        assert rel_diff[0.05][i] < 0.6, (name, rel_diff)
        assert rel_diff[0.025][i] < 0.6 * rel_diff[0.05][i], (name, rel_diff)
