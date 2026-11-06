"""VariationalNonlinearPBEvolve (the nonlinear terms of the perturbative variational MHD model): the substep alone
conserves the kinetic plus the first-order magnetic energy (implicit midpoint rule, Picard iteration), with the
velocity in H1vec and in H(div), and it does nothing without magnetic and pressure perturbations."""

import logging

import numpy as np
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _setup(u_space: str, amp_b: float = 2e-3, tol: float = 1e-13):
    """The ideal perturbative model on a Colella mapping with a sheared slab equilibrium, random perturbations of
    the velocity, the magnetic field (divergence free: curl of a random 1-form) and the pressure."""
    from struphy import DerhamOptions, Simulation, domains, equils, grids, perturbations
    from struphy.linear_algebra.solver import NonlinearSolverParameters
    from struphy.models import VariationalPerturbativeMHD
    from struphy.propagators.base import Propagator

    model = VariationalPerturbativeMHD(u_space=u_space, with_viscosity=False, with_resistivity=False)
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model="perturbative")
    props.variat_lin_sound.options = props.variat_lin_sound.Options(model="linear_p")
    props.variat_nonlinear.options = props.variat_nonlinear.Options(
        nonlin_solver=NonlinearSolverParameters(tol=tol, maxiter=200)
    )
    for comp in range(3):
        model.mhd.velocity.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=comp, seed=10 + comp)
        )
    model.mhd.pressure.add_perturbation(perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=None, seed=30))
    Simulation(
        model=model,
        domain=domains.Colella(alpha=0.1, Lx=1.0, Ly=1.0, Lz=1.0),
        equil=equils.ShearedSlab(a=1.0, R0=1.0, n1=0.4, n2=0.0, na=0.5, beta=0.6),
        grid=grids.TensorProductGrid(num_elements=(6, 4, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    ).allocate()

    # divergence-free magnetic perturbation
    der = Propagator.derham
    a = der.V1pol.zeros()
    rng = np.random.default_rng(1)
    for blk in a.blocks:
        blk._data[:] = amp_b * rng.standard_normal(blk._data.shape)
    a.update_ghost_regions()
    b = der.curl.dot(a)
    b.copy(out=model.em_fields.b_field.spline.vector)
    model.em_fields.b_field.spline.vector.update_ghost_regions()
    return model


def _energy(model):
    model.scalars.update()
    return float(np.squeeze(model.scalars.host_value("en_U"))) + float(np.squeeze(model.scalars.host_value("en_mag_1")))


@pytest.mark.parametrize("u_space", ["H1vec", "Hdiv"])
def test_nonlinear_substep_energy(u_space: str):
    """The nonlinear substep conserves 1/2 u^T M_rho u + 1/2 b^T M2 b and changes the state."""
    model = _setup(u_space)
    prop = model.propagators.variat_nonlinear
    u0 = model.mhd.velocity.spline.vector.copy()
    e0 = _energy(model)
    for _ in range(3):
        prop(0.05)
    e1 = _energy(model)
    du = model.mhd.velocity.spline.vector - u0
    assert np.sqrt(du.inner(du)) > 1e-8 * np.sqrt(u0.inner(u0))
    assert abs(e1 - e0) < 1e-10 * e0, (e0, e1)


@pytest.mark.parametrize("u_space", ["H1vec", "Hdiv"])
def test_nonlinear_substep_without_perturbations(u_space: str):
    """Without magnetic and pressure perturbations the nonlinear terms vanish: the state does not change."""
    model = _setup(u_space, amp_b=0.0)
    model.mhd.pressure.spline.vector *= 0.0
    model.mhd.pressure.spline.vector.update_ghost_regions()
    x0 = [v.spline.vector.copy() for v in (model.mhd.velocity, model.em_fields.b_field, model.mhd.pressure)]
    model.propagators.variat_nonlinear(0.05)
    for x, v in zip(x0, (model.mhd.velocity, model.em_fields.b_field, model.mhd.pressure)):
        d = v.spline.vector - x
        assert np.sqrt(d.inner(d)) < 1e-14 * max(np.sqrt(x.inner(x)), 1.0)
