"""VariationalLinearSoundEvolve with total_weight=True (the sound step of the perturbative variational MHD model, with
the transport weight p0 + p~): the step is symmetric in time and conserves the energy, and the coupled steps of the
model depend only on the total pressure p0 + p~, not on how it is split into equilibrium and perturbation (a constant
offset p0 -> p0 + pc, p~ -> p~ - pc), also at large time steps, where a separate (explicit) transport of p~ with the
weight p~ is unstable."""

import logging

import numpy as np
import pytest
from maybempi import MPI

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _setup(beta: float, pc: float, u_space: str = "Hdiv"):
    """The ideal perturbative model on a periodic slab (Cuboid, B0 along z, uniform pressure beta/2 + pc), with a sound
    wave along B0, a smooth perturbation across it and grid-scale noise in the pressure; the pressure perturbation is
    shifted by -pc (same total pressure for every pc)."""
    from struphy import DerhamOptions, Simulation, domains, equils, grids
    from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
    from struphy.models import VariationalPerturbativeMHD
    from struphy.propagators.base import Propagator

    model = VariationalPerturbativeMHD(u_space=u_space, with_viscosity=False, with_resistivity=False)
    props = model.propagators
    lin = SolverParameters(tol=1e-13, maxiter=3000)
    nl = NonlinearSolverParameters(tol=1e-12, maxiter=100)
    props.variat_dens.options = props.variat_dens.Options(model="perturbative", solver_params=lin, nonlin_solver=nl)
    props.variat_lin_alfven.options = props.variat_lin_alfven.Options(solver_params=lin)
    props.variat_lin_sound.options = props.variat_lin_sound.Options(
        model="linear_p", solver_params=lin, nonlin_solver=nl
    )
    props.variat_nonlinear.options = props.variat_nonlinear.Options(solver_params=lin, nonlin_solver=nl)
    Simulation(
        model=model,
        domain=domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0),
        equil=equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=beta + 2.0 * pc, n0=1.0),
        grid=grids.TensorProductGrid(num_elements=(4, 1, 16)),
        derham_opts=DerhamOptions(degree=(2, 1, 3)),
    ).allocate()

    der = Propagator.derham
    k = 2.0 * np.pi
    zero = lambda e1, e2, e3: 0.0 * e1
    u = der.projectors["2" if u_space == "Hdiv" else "v"](
        [lambda e1, e2, e3: 1e-2 * np.sin(k * e1) * np.cos(k * e3), zero, lambda e1, e2, e3: 2e-2 * np.sin(k * e3)]
    )
    der.boundary_ops["2" if u_space == "Hdiv" else "v"].dot(u).copy(out=model.mhd.velocity.spline.vector)
    model.mhd.velocity.spline.vector.update_ghost_regions()
    p = der.projectors["3"](lambda e1, e2, e3: -pc + 2e-3 * np.cos(k * e3) + 1e-3 * np.sin(k * e1))
    # grid-scale noise (the same on every rank layout: a function of the coordinates)
    p += der.projectors["3"](lambda e1, e2, e3: 1e-6 * np.sin(97.0 * e1 + 31.0 * e3) * np.cos(53.0 * e3))
    p.copy(out=model.mhd.pressure.spline.vector)
    model.mhd.pressure.spline.vector.update_ghost_regions()
    return model


def _group(model, dt: float):
    """The Strang group of the model: linear Alfven, sound (total weight), nonlinear substep, in symmetric order."""
    props = model.propagators
    props.variat_lin_alfven(dt / 2)
    props.variat_lin_sound(dt / 2)
    props.variat_nonlinear(dt)
    props.variat_lin_sound(dt / 2)
    props.variat_lin_alfven(dt / 2)


def _en_tot(model):
    model.scalars.update()
    return float(np.squeeze(model.scalars.host_value("en_tot")))


@pytest.mark.parametrize("u_space", ["Hdiv", "H1vec"])
def test_sound_total_weight_symmetric_energy(u_space: str):
    """The sound step with the total pressure as weight is symmetric in time (P(-h) P(h) = I, second order) and
    conserves the energy, with a finite pressure perturbation (a third of the equilibrium pressure)."""
    model = _setup(beta=0.1, pc=0.0167, u_space=u_space)
    prop = model.propagators.variat_lin_sound
    vecs = [model.mhd.velocity.spline.vector, model.mhd.pressure.spline.vector]
    x0 = [v.copy() for v in vecs]
    e0 = _en_tot(model)
    prop(0.2)
    e1 = _en_tot(model)
    assert abs(e1 - e0) < 1e-10 * abs(e0), (e0, e1)
    d = vecs[0] - x0[0]
    assert np.sqrt(d.inner(d)) > 1e-6 * np.sqrt(x0[0].inner(x0[0]))
    prop(-0.2)
    for v, x in zip(vecs, x0):
        d = v - x
        assert np.sqrt(d.inner(d)) < 1e-9 * np.sqrt(x.inner(x)), np.sqrt(d.inner(d) / x.inner(x))


def test_sound_total_weight_split_invariance():
    """The coupled steps (Strang group) at a large time step (omega dt >> 1 for the grid-scale sound waves) give the
    same velocity for p0 + pc, p~ - pc as for pc = 0, with exact energy conservation: the result depends only on the
    total pressure, and the pressure offset does not make the grid-scale sound waves grow."""
    dt, nsteps = 0.25, 12
    u = {}
    for pc in (0.0, 0.5):
        model = _setup(beta=0.1, pc=pc)
        e0 = _en_tot(model)
        for _ in range(nsteps):
            _group(model, dt)
        e1 = _en_tot(model)
        assert abs(e1 - e0) < 1e-9 * abs(e0), (pc, e0, e1)
        # the two models have different (equal) vector spaces: compare the local coefficients, summed over the ranks
        u[pc] = model.mhd.velocity.spline.vector.toarray()
    loc = np.array([np.sum((u[0.5] - u[0.0]) ** 2), np.sum(u[0.0] ** 2)])
    glob = np.zeros(2)
    MPI.COMM_WORLD.Allreduce(loc, glob, op=MPI.SUM)
    rel = np.sqrt(glob[0] / glob[1])
    assert rel < 1e-7, rel
