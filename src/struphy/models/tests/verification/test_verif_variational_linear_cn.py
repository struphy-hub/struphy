"""Verification of the linear variational MHD model with one linear solve per substep (``linear=True,
crank_nicolson=True``), i.e. of
:class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve` and
:class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve`, in the model
:class:`~struphy.models.variational_perturbative_mhd.VariationalPerturbativeMHD`:

1. the Alfvén step is the step of VariationalPBEvolve (linear) without pressure perturbation, around the sheared
   slab, where the term with curl B0 is active;
2. the sound step is the step of VariationalPBEvolve (linear) for a sound wave along B0, where the magnetic field is
   not perturbed;
3. the energy with the auxiliary variables is conserved around the sheared slab;
4. the analytic frequencies of the shear Alfvén, the fast magnetosonic and the sound wave in a homogeneous slab;
5. the model agrees with the linear model with Picard iteration up to a time discretization error of first order.
"""

import logging

import cunumpy as xp
import pytest

from struphy import perturbations, set_logging_level
from struphy.models import VariationalPerturbativeMHD
from struphy.models.tests.verification.variational_mhd_utils import (
    GAMMA,
    allocate_ideal,
    energy_history,
    homogeneous_slab,
    linear,
    linear_cn,
    norm,
    perturb_velocity,
    sheared_slab,
    standing_wave,
    state,
)

set_logging_level()
logger = logging.getLogger("struphy")


def _perturb_u_and_b(amp):
    """Velocity perturbation and a divergence-free magnetic perturbation B~_y(x) (sheared slab, periodic in x)."""

    def perturb(model):
        perturb_velocity(amp)(model)
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesSin(ls=(1,), amps=(amp,), comp=1, given_in_basis="2")
        )

    return perturb


def _perturb_sound(amp, Lz):
    """Sound wave along z (u_z(z) and the pressure), for B0 along z."""

    def perturb(model):
        model.mhd.velocity.add_perturbation(
            perturbations.ModesSin(ns=(1,), amps=(amp,), comp=2, Lz=Lz, given_in_basis="physical")
        )
        model.mhd.pressure.add_perturbation(perturbations.ModesCos(ns=(1,), amps=(amp,), Lz=Lz, given_in_basis="3"))

    return perturb


def test_alfven_step_same_as_pb(tmp_path):
    """One step of VariationalLinearAlfvenEvolve gives the velocity, the magnetic field and B^ of one step of
    VariationalPBEvolve (linear), whose pressure force vanishes for a zero pressure perturbation."""
    setup, dt, perturb = sheared_slab(), 0.05, _perturb_u_and_b(0.05)

    model = allocate_ideal(tmp_path, "cn", linear_cn(VariationalPerturbativeMHD), setup, dt, perturb)
    model.propagators.variat_lin_alfven(dt)
    cn = state(model) + [model.diagnostics.bt2.spline.vector.toarray()]

    model = allocate_ideal(tmp_path, "pb", linear(VariationalPerturbativeMHD), setup, dt, perturb)
    model.propagators.variat_pb(dt)
    pb = state(model) + [model.diagnostics.bt2.spline.vector.toarray()]

    assert norm(cn[4]) > 0.0
    for i, name in ((1, "u"), (3, "B"), (4, "B^")):
        rel = norm(cn[i] - pb[i]) / norm(pb[i])
        logger.info(f"Alfvén step vs. VariationalPBEvolve, {name}: {rel:.2e}")
        assert rel < 1e-6, (name, rel)


def test_sound_step_same_as_picard(tmp_path):
    """For a sound wave along B0 the magnetic field stays zero, and the linear model with one linear solve per
    substep gives the solution of VariationalPBEvolve (linear), which solves the same equations."""
    Lz, dt, n_steps = 10.0, 0.2, 3
    setup = homogeneous_slab(B0=(0.0, 0.0, 1.0), Lz=Lz)
    perturb = _perturb_sound(0.05, Lz)

    model = allocate_ideal(tmp_path, "cn", linear_cn(VariationalPerturbativeMHD), setup, dt, perturb)
    for _ in range(n_steps):
        model.integrate(dt)
    cn = state(model)

    model = allocate_ideal(
        tmp_path, "picard", linear(VariationalPerturbativeMHD), setup, dt, perturb, implicit_weights=False
    )
    for _ in range(n_steps):
        model.integrate(dt)
    picard = state(model)

    assert norm(cn[3]) == 0.0
    for i, name in ((1, "u"), (2, "p")):
        rel = norm(cn[i] - picard[i]) / norm(picard[i])
        logger.info(f"sound vs. Picard propagator, {name}: {rel:.2e}")
        assert rel < 1e-6, (name, rel)


def test_energy_conservation(tmp_path):
    """The energy with the auxiliary variables (B^ and p^) is conserved around the sheared slab, where the
    second-order terms (curl B0 and grad p0) are active."""
    model = allocate_ideal(
        tmp_path,
        "energy",
        linear_cn(VariationalPerturbativeMHD),
        sheared_slab(),
        dt=0.05,
        perturb=_perturb_u_and_b(0.05),
    )
    energies = energy_history(model, dt=0.05, n_steps=10)

    assert abs(model.scalars.host_value("en_mag_2")[0]) > 1e-5
    drift = max(abs(en - energies[0]) for en in energies)
    logger.info(f"relative energy drift {drift / abs(energies[0]):.2e}")
    assert drift < 2e-8 * abs(energies[0]), energies


@pytest.mark.parametrize("wave", ["alfven", "fast", "sound"])
def test_linear_waves(tmp_path, wave):
    """Standing waves with k = 2 pi / Lz along z in a homogeneous slab (beta = 1, n0 = 0.7): shear Alfvén
    (B0 along z, u_x) with omega = k |B0| / sqrt(n0), fast magnetosonic (B0 along y, u_z) with
    omega = k sqrt((gamma p0 + |B0|^2) / n0) and sound (B0 along z, u_z) with omega = k sqrt(gamma p0 / n0). The scheme
    is of first order in time (the terms with the weight at t^n, the splitting of the Alfvén and the sound step, and
    the explicit pressure force); the shear Alfvén wave in the homogeneous slab is of second order (Crank-Nicolson)."""
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
    for n_steps in (40, 80):
        _, omega = standing_wave(
            tmp_path, linear_cn(VariationalPerturbativeMHD), setup, Lz, comp, 1.0, half_period / n_steps, n_steps
        )
        errors.append(abs(omega - omega_exact) / omega_exact)
    logger.info(f"{wave}: relative frequency errors for 40 and 80 steps per half period: {errors}")
    assert errors[1] < 1e-2, errors
    assert errors[1] < 0.75 * errors[0], errors


def test_same_as_picard_model(tmp_path):
    """Around the sheared slab the model agrees with the linear model with Picard iteration up to a difference of
    first order in the time step."""
    setup = sheared_slab()
    t_end = 0.5
    rel_diff = {}
    for dt in (0.05, 0.025):
        n_steps = int(round(t_end / dt))
        runs = []
        for model_class in (linear_cn(VariationalPerturbativeMHD), linear(VariationalPerturbativeMHD)):
            model = allocate_ideal(
                tmp_path, f"{model_class.__name__}_{dt}", model_class, setup, dt, perturb_velocity(0.05)
            )
            for _ in range(n_steps):
                model.integrate(dt)
            runs.append(state(model))
        rel_diff[dt] = [norm(runs[0][i] - runs[1][i]) / norm(runs[1][i]) for i in (1, 3)]
    logger.info(f"relative difference to the Picard model (u, B) per dt: {rel_diff}")
    for i, name in enumerate(("u", "B")):
        assert rel_diff[0.05][i] < 0.6, (name, rel_diff)
        assert rel_diff[0.025][i] < 0.7 * rel_diff[0.05][i], (name, rel_diff)
