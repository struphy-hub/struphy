"""Verification of DriftKineticMaxwell: the shear (inertial) Alfvén wave of a cold guiding-center plasma in a slab
with uniform B0 = e_z, against the roots of the dispersion relation

.. math::

    \\omega^2 (\\lambda^2 \\mathbf E + \\chi \\mathbf E_\\perp) = c_1 (k^2 \\mathbf E - \\mathbf k\\, \\mathbf k \\cdot \\mathbf E)
    + P E_z \\mathbf e_z - \\mathrm i \\omega H\\, \\mathbf E \\times \\mathbf e_z\\,,

with :math:`P = c_2 c_3 n_0` (parallel response) and :math:`H = c_2 n_0` (E x B current, zero if switched off), in the
thermal normalization: :math:`\\chi = (\\varepsilon\\kappa_\\textrm{s})^2`, :math:`c_2 = \\varepsilon\\kappa_\\textrm{s}^2`,
:math:`c_3 = 1/\\varepsilon`, and :math:`\\lambda` the option ``lam`` (coefficient of the displacement current).
For :math:`\\lambda = 0` and :math:`H = 0` the shear branch is the inertial Alfvén wave
:math:`\\omega^2 = (c_1/\\chi)\\, k_z^2 / (1 + k_x^2 c_1/P)`. The energy is conserved exactly by the splitting
(cold plasma, uniform B0).

Run it on several processes with::

    mpirun -n 4 pytest -p no:xdist --with-mpi src/struphy/models/tests/verification/test_verif_DriftKineticMaxwell.py
"""

import logging

import numpy as np
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# species parameters (in the thermal normalization of the model): chi = (EPSILON KAPPA)^2 = 9, c2 = 18, c3 = 2
EPSILON, LAM, KAPPA = 0.5, 2.0, 6.0
LX, LZ = 1.0, 4.0
VTH = 1e-6  # cold


def _dispersion_roots(lam, exb_current):
    """Positive real roots omega of the determinant of the cold-slab dispersion matrix (polynomial of degree 6,
    interpolated exactly from 7 values)."""
    c1, c2, c3 = 1.0, KAPPA**2 * EPSILON, 1.0 / EPSILON
    chi, P, H = (EPSILON * KAPPA) ** 2, c2 * c3, c2 if exb_current else 0.0
    k = np.array([2 * np.pi / LX, 0.0, 2 * np.pi / LZ])
    proj_perp = np.diag([1.0, 1.0, 0.0])
    cross_z = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # E -> E x e_z

    def det(w):
        mat = (
            w**2 * (lam**2 * np.eye(3) + chi * proj_perp)
            - c1 * (k @ k * np.eye(3) - np.outer(k, k))
            - P * np.diag([0.0, 0.0, 1.0])
            + 1j * w * H * cross_z
        )
        return np.linalg.det(mat)

    nodes = np.linspace(-1.0, 1.0, 7)
    coeffs = np.polyfit(nodes, [det(w).real for w in nodes], 6)
    roots = np.roots(coeffs)
    return np.sort(roots[(abs(roots.imag) < 1e-8) & (roots.real > 1e-8)].real)


def _alfven_signal(lam, exb_current, dt, n_steps):
    """Time series s(t) = b(t)^T M2 b(0) / |b(0)|^2 of the shear Alfvén wave, excited by a B_y perturbation, and the
    relative change of the energy."""
    from struphy import (
        DerhamOptions,
        LoadingParameters,
        Simulation,
        domains,
        equils,
        grids,
        maxwellians,
        models,
        perturbations,
    )
    from struphy.linear_algebra.solver import SolverParameters

    model = models.DriftKineticMaxwell(epsilon=EPSILON, lam=LAM, kappa=KAPPA)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=50, seed=1234, moments=(0.0, 0.0, VTH, VTH), B0=1.0)
    )
    cold = dict(n=(1.0, None), vth_para=(VTH, None), vth_perp=(VTH, None), B0=1.0)
    model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(**cold))
    model.kinetic_ions.var.add_initial_condition(maxwellians.GyroMaxwellian2D(**cold))
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesSin(ls=(1,), ns=(1,), given_in_basis="2", comp=1, amps=(1e-2,))
    )
    prop = model.propagators.coupling
    solver = {} if exb_current else {"solver": "pcg"}
    prop.options = prop.Options(
        lam=lam, exb_current=exb_current, solver_params=SolverParameters(tol=1e-12, maxiter=3000), **solver
    )
    Simulation(
        model=model,
        domain=domains.Cuboid(r1=LX, r2=1.0, r3=LZ),
        equil=equils.HomogenSlab(B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(8, 1, 8)),
        derham_opts=DerhamOptions(degree=(3, 1, 3)),
    ).allocate()

    props = model.propagators
    prop.prepare_time_step(dt)  # the initial projection for lam = 0
    energy_0 = prop.energy()
    b = model.em_fields.b_field.spline.vector
    b_init = b.copy()
    m2_b_init = prop._M2.dot(b_init)
    signal = [b.inner(m2_b_init)]
    for _ in range(n_steps):
        props.coupling(dt)
        props.push_parallel(dt)
        props.push_bxe(dt)
        signal.append(b.inner(m2_b_init))
    return dt * np.arange(n_steps + 1), np.array(signal) / signal[0], prop.energy() / energy_0 - 1.0


def _fit(t, s, omegas):
    """Least-squares fit of s(t) by a constant plus one harmonic per initial guess in omegas."""
    from scipy.optimize import curve_fit

    def model(t, c, *p):
        out = c + 0.0 * t
        for i in range(len(p) // 3):
            out = out + p[3 * i] * np.cos(p[3 * i + 2] * t) + p[3 * i + 1] * np.sin(p[3 * i + 2] * t)
        return out

    p0 = [0.0]
    for w in omegas:
        p0 += [0.5, 0.0, w]
    popt, _ = curve_fit(model, t, s, p0=p0)
    return np.abs(popt[3::3])


@pytest.mark.parametrize(
    "lam, exb_current, dt, n_steps",
    [(0.0, False, 0.5, 160), (0.0, True, 0.5, 160), (1.0, False, 0.1, 400)],
)
def test_shear_alfven_wave(lam, exb_current, dt, n_steps):
    roots = _dispersion_roots(lam, exb_current)
    t, s, energy_change = _alfven_signal(lam, exb_current, dt, n_steps)
    if lam > 0.0:
        # the B_y perturbation also excites the fast (light-wave like) upper branch of the shear mode
        omegas = _fit(t, s, (roots[0], roots[-1]))
    else:
        omegas = _fit(t, s, (roots[0],))
    assert abs(omegas[0] / roots[0] - 1.0) < 0.02, (omegas, roots)
    assert abs(energy_change) < 1e-7
