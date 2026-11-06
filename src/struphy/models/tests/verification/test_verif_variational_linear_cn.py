"""Verification of the linear variational MHD model
:class:`~struphy.models.variational_linear_mhd.VariationalLinearMHD` (one linear solve per substep), i.e. of
:class:`~struphy.propagators.variational_linear_alfven_evolve.VariationalLinearAlfvenEvolve` and
:class:`~struphy.propagators.variational_linear_sound_evolve.VariationalLinearSoundEvolve`. The reference "Picard"
step is :class:`~struphy.propagators.variational_pb_evolve.VariationalPBEvolve` (``model="linear"``), attached to the
same variables:

1. with the weights at t^n (``weights="frozen"``), the Alfvén step agrees with the step of VariationalPBEvolve
   (linear) without pressure perturbation, around the sheared slab, where the term with curl B0 is active (both use
   the force T[b]^T M2 b0 of the second-order term, the adjoint of its transport);
2. with the weights at t^n, the sound step is the step of VariationalPBEvolve (linear) for a sound wave along B0,
   where the magnetic field is not perturbed;
3. the energy with the auxiliary variables is conserved around the sheared slab;
4. the analytic frequencies of the shear Alfvén, the fast magnetosonic and the sound wave in a homogeneous slab;
5. the model (weights at t^n) agrees with the density step plus VariationalPBEvolve (linear, Picard iteration) up
   to a time discretization error of first order;
6. no spurious instability at a wall where the equilibrium current is tangential (screw pinch in a hollow cylinder).
"""

import logging

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import perturbations, set_logging_level
from struphy.models import VariationalLinearMHD
from struphy.models.tests.verification.variational_mhd_utils import (
    GAMMA,
    NONLIN,
    SOLVER,
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


def _picard_step(model):
    """VariationalPBEvolve (model="linear", Picard iteration) on the variables of the allocated linear model."""
    from struphy.propagators.variational_pb_evolve import VariationalPBEvolve

    diag = model.diagnostics
    pb = VariationalPBEvolve(div_u=diag.div_u, u2=diag.u2, pt3=diag.pt3, bt2=diag.bt2)
    pb.variables.u = model.mhd.velocity
    pb.variables.p = model.mhd.pressure
    pb.variables.b = model.em_fields.b_field
    pb.options = pb.Options(model="linear", solver_params=SOLVER, nonlin_solver=NONLIN)
    pb.allocate()
    return pb


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
    r"""One step of VariationalLinearAlfvenEvolve gives the velocity, the magnetic field and B^ of one step of
    VariationalPBEvolve (linear), whose pressure force vanishes for a zero pressure perturbation: both have the force
    :math:`\mathbb T[\mathbf b]^\top \mathbb M^2 \mathbf b_0` of the second-order term and advance
    :math:`\hat{\mathbf B}` with the same transport operator."""
    setup, dt, perturb = sheared_slab(), 0.05, _perturb_u_and_b(0.05)

    model = allocate_ideal(tmp_path, "cn", VariationalLinearMHD, setup, dt, perturb, weights="frozen")
    model.propagators.variat_lin_alfven(dt)
    cn = state(model) + [model.diagnostics.bt2.spline.vector.toarray()]

    model = allocate_ideal(tmp_path, "pb", VariationalLinearMHD, setup, dt, perturb)
    _picard_step(model)(dt)
    pb = state(model) + [model.diagnostics.bt2.spline.vector.toarray()]

    assert norm(cn[4]) > 0.0
    for i, name, tol in ((1, "u", 1e-4), (3, "B", 1e-4), (4, "B^", 1e-2)):
        rel = norm(cn[i] - pb[i]) / norm(pb[i])
        logger.info(f"Alfvén step vs. VariationalPBEvolve, {name}: {rel:.2e}")
        assert rel < tol, (name, rel)


def test_sound_step_same_as_picard(tmp_path):
    """For a sound wave along B0 the magnetic field stays zero, and the linear model with one linear solve per
    substep (weights at t^n) gives the solution of VariationalPBEvolve (linear), which solves the same equations."""
    Lz, dt, n_steps = 10.0, 0.2, 3
    setup = homogeneous_slab(B0=(0.0, 0.0, 1.0), Lz=Lz)
    perturb = _perturb_sound(0.05, Lz)

    model = allocate_ideal(tmp_path, "cn", VariationalLinearMHD, setup, dt, perturb, weights="frozen")
    for _ in range(n_steps):
        model.integrate(dt)
    cn = state(model)

    model = allocate_ideal(tmp_path, "picard", VariationalLinearMHD, setup, dt, perturb)
    pb = _picard_step(model)
    for _ in range(n_steps):
        model.propagators.variat_dens(dt)
        pb(dt)
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
        VariationalLinearMHD,
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
    the explicit pressure force); the shear Alfvén wave in the homogeneous slab is of second order (Crank-Nicolson).
    (With midpoint weights, the default, the substeps are of second order, the Lie-Trotter splitting of first order.)"""
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
        _, omega = standing_wave(tmp_path, VariationalLinearMHD, setup, Lz, comp, 1.0, half_period / n_steps, n_steps)
        errors.append(abs(omega - omega_exact) / omega_exact)
    logger.info(f"{wave}: relative frequency errors for 40 and 80 steps per half period: {errors}")
    assert errors[1] < 1e-2, errors
    assert errors[1] < 0.75 * errors[0], errors


def test_same_as_picard_model(tmp_path):
    """Around the sheared slab the model agrees with the density step plus the linear Picard step
    (VariationalPBEvolve) up to a difference of first order in the time step."""
    setup = sheared_slab()
    t_end = 0.5
    rel_diff = {}
    for dt in (0.05, 0.025):
        n_steps = int(round(t_end / dt))
        runs = []
        for picard in (False, True):
            model = allocate_ideal(
                tmp_path,
                f"{'picard' if picard else 'cn'}_{dt}",
                VariationalLinearMHD,
                setup,
                dt,
                perturb_velocity(0.05),
                weights="frozen",
            )
            pb = _picard_step(model) if picard else None
            for _ in range(n_steps):
                if picard:
                    model.propagators.variat_dens(dt)
                    pb(dt)
                else:
                    model.integrate(dt)
            runs.append(state(model))
        rel_diff[dt] = [norm(runs[0][i] - runs[1][i]) / norm(runs[1][i]) for i in (1, 3)]
    logger.info(f"relative difference to the Picard model (u, B) per dt: {rel_diff}")
    for i, name in enumerate(("u", "B")):
        assert rel_diff[0.05][i] < 0.6, (name, rel_diff)
        assert rel_diff[0.025][i] < 0.7 * rel_diff[0.05][i], (name, rel_diff)


def _reduced_one_step_map(model, dt: float, m: int, n: int) -> tuple[np.ndarray, float]:
    """The one-step map of (u, b, p) restricted to the Fourier mode (m, n) of the periodic directions eta2, eta3
    (cos and sin patterns of the cell indices): the equilibrium and the grid are invariant in eta2 and eta3, so the
    map does not mix Fourier modes. Returns the real matrix (two columns per radial coefficient) and the largest
    relative part of a column outside the mode (solver error)."""
    from feectools.linalg.utilities import array_to_psydac

    vecs = [model.mhd.velocity.spline.vector, model.em_fields.b_field.spline.vector, model.mhd.pressure.spline.vector]
    spaces = [s for v in vecs for s in (v.space.spaces if hasattr(v.space, "spaces") else [v.space])]
    shapes = [tuple(s.npts) for s in spaces]
    coff = np.cumsum([0] + [int(np.prod(s)) for s in shapes])
    voff = np.cumsum([0] + [len(v.toarray()) for v in vecs])
    n2, n3 = shapes[0][1], shapes[0][2]
    j2, j3 = np.meshgrid(np.arange(n2), np.arange(n3), indexing="ij")
    phase = 2 * np.pi * (m * j2 / n2 + n * j3 / n3)
    patterns = (np.cos(phase), np.sin(phase))
    basis = [(c, i, k) for c, sh in enumerate(shapes) for i in range(sh[0]) for k in range(2)]
    A = np.zeros((len(basis), len(basis)))
    leak = 0.0
    for col, (c, i, k) in enumerate(basis):
        x = np.zeros(voff[-1])
        x[coff[c] : coff[c + 1]].reshape(shapes[c])[i] = patterns[k]
        for v, lo, hi in zip(vecs, voff[:-1], voff[1:]):
            array_to_psydac(x[lo:hi], v.space).copy(out=v)
            v.update_ghost_regions()
        model.integrate(dt)
        y = np.concatenate([v.toarray() for v in vecs])
        rest = y.copy()
        row = 0
        for cc, sh in enumerate(shapes):
            Y = y[coff[cc] : coff[cc + 1]].reshape(sh)
            for pat in patterns:
                amp = 2 / (n2 * n3) * np.einsum("ijk,jk->i", Y, pat)
                rest[coff[cc] : coff[cc + 1]].reshape(sh)[:] -= amp[:, None, None] * pat
            amps = [2 / (n2 * n3) * np.einsum("ijk,jk->i", Y, pat) for pat in patterns]
            A[row : row + 2 * sh[0] : 2, col] = amps[0]
            A[row + 1 : row + 2 * sh[0] : 2, col] = amps[1]
            row += 2 * sh[0]
        leak = max(leak, np.linalg.norm(rest) / max(np.linalg.norm(y), 1e-300))
    return A, leak


def test_no_wall_mode(tmp_path):
    """Screw pinch (R0 = 20, Suydam-stable) in a hollow cylinder with the inner wall at r = 0.2, where the equilibrium
    current is tangential: the one-step map of the Fourier mode (m, n) = (3, -2) has no growing eigenvalue. With the
    discrete curl of B0 as the current of the second-order force (tangential components zero at the wall), the same
    map had a purely growing mode at the inner wall with the rate 2.6e-3."""
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("one-step map assembled from the full coefficient arrays: serial only")
    from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids
    from struphy.linear_algebra.solver import SolverParameters

    R0, dt = 20.0, 0.5
    model = VariationalLinearMHD(u_space="Hdiv", with_viscosity=False, with_resistivity=False)
    props = model.propagators
    lin = SolverParameters(tol=1e-12, maxiter=5000)
    props.variat_dens.options = props.variat_dens.Options(model="linear", gamma=GAMMA)
    props.variat_lin_alfven.options = props.variat_lin_alfven.Options(weights="midpoint", solver_params=lin)
    props.variat_lin_sound.options = props.variat_lin_sound.Options(
        model="linear_p", gamma=GAMMA, weights="midpoint", solver_params=lin
    )
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=str(tmp_path), sim_folder="wall"),
        time_opts=Time(dt=dt, Tend=dt),
        domain=domains.HollowCylinder(a1=0.2, a2=1.0, Lz=2 * np.pi * R0),
        equil=equils.ScrewPinch(a=1.0, R0=R0, B0=1.0, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0, beta=0.1, p0=0.1),
        grid=grids.TensorProductGrid(num_elements=(8, 8, 6)),
        derham_opts=DerhamOptions(degree=(2, 2, 2), bcs=(("dirichlet", "dirichlet"), None, None)),
    )
    sim.allocate()

    A, leak = _reduced_one_step_map(model, dt, m=3, n=-2)
    lam = np.linalg.eigvals(A)
    rate = np.log(np.abs(lam).max()) / dt
    logger.info(f"wall mode test: largest growth rate {rate:.2e}, leak out of the Fourier mode {leak:.1e}")
    assert leak < 1e-8, leak
    # the eigenvalues lie on the unit circle up to the solver tolerance
    assert rate < 1e-8, rate
