"""Verification of LinearMHDWithQ (linear MHD with the variable q = sqrt(p)): the dispersion of the three MHD waves in
a homogeneous slab (1D), and the spectrum of the one-step map in a screw pinch with a pressure gradient against
LinearMHD (same unstable modes and growth rate, no spurious unstable or overstable modes)."""

import logging
import os
import shutil

import cunumpy as xp
import numpy as np
import pytest
from feectools.linalg.utilities import array_to_psydac
from maybempi import MPI

from struphy import (
    DerhamOptions,
    EnvironmentOptions,
    Simulation,
    Time,
    domains,
    equils,
    grids,
    perturbations,
    set_logging_level,
)
from struphy.linear_algebra import schur_solver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models import LinearMHD, LinearMHDWithQ
from struphy.models.tests.verification.spectrum_utils import fit_branches, power_spectrum

set_logging_level()
logger = logging.getLogger("struphy")


@pytest.mark.parametrize("b_space, projected", [("Hdiv", False), ("Hcurl", True)])
def test_slab_waves_1d(b_space: str, projected: bool):
    """Noise in the velocity along z in a homogeneous slab: the space-time spectra of u_x and of q have the branches of
    the shear-Alfven wave and of the slow and fast magnetosonic waves."""
    model = LinearMHDWithQ(b_space=b_space, projected=projected)

    test_folder = os.path.join(os.getcwd(), "verification_tests")
    out_folders = os.path.join(test_folder, f"LinearMHDWithQ_{b_space}{'_projected' if projected else ''}")
    env = EnvironmentOptions(out_folders=out_folders, sim_folder="slab_waves_1d")

    B0x, B0y, B0z = 0.0, 1.0, 1.0
    beta = 3.0
    n0 = 0.7
    equil = equils.HomogenSlab(B0x=B0x, B0y=B0y, B0z=B0z, beta=beta, n0=n0)

    for comp in range(3):
        model.mhd.velocity.add_perturbation(perturbations.Noise(amp=0.1, comp=comp, seed=123))

    sim = Simulation(
        model=model,
        env=env,
        time_opts=Time(dt=0.15, Tend=90.0),
        domain=domains.Cuboid(r3=60.0),
        grid=grids.TensorProductGrid(num_elements=(1, 1, 64)),
        derham_opts=DerhamOptions(degree=(1, 1, 3)),
        equil=equil,
    )
    run = sim.run().with_time_units("normalized")
    run.pproc()

    if MPI.COMM_WORLD.Get_rank() == 0:
        Bsquare = B0x**2 + B0y**2 + B0z**2
        p0 = beta * Bsquare / 2

        omega, k, power = power_spectrum(run.fields.mhd.velocity, component=0)
        fits = fit_branches(omega, k, power, n_branches=1, noise_level=0.5)
        vA = xp.sqrt(Bsquare / n0)
        v_alfven = vA * B0z / xp.sqrt(Bsquare)
        logger.info(f"{v_alfven =}, measured {fits[0][0]}")
        assert xp.abs(fits[0][0] - v_alfven) < 0.07

        omega, k, power = power_spectrum(run.fields.mhd.sqrt_p)
        fits = fit_branches(omega, k, power, n_branches=2, noise_level=0.4)
        cS = xp.sqrt(5 / 3 * p0 / n0)
        delta = (4 * B0z**2 * cS**2 * vA**2) / ((cS**2 + vA**2) ** 2 * Bsquare)
        v_slow = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 - xp.sqrt(1 - delta)))
        v_fast = xp.sqrt(1 / 2 * (cS**2 + vA**2) * (1 + xp.sqrt(1 - delta)))
        logger.info(f"{v_slow =}, measured {fits[0][0]}")
        logger.info(f"{v_fast =}, measured {fits[1][0]}")
        # the fits underestimate the speeds by a few percent (numerical dispersion at the larger k of the fit), as
        # for LinearMHD
        assert xp.abs(fits[0][0] - v_slow) < 0.075
        assert xp.abs(fits[1][0] - v_fast) < 0.19

        shutil.rmtree(test_folder)


def _schur_call_relative(self, xn, Byn, dt, out=None):
    """SchurSolver.__call__ with the solve for the normalized right-hand side (relative tolerance), such that the
    columns of the one-step map (unit vectors and their images) are all solved to the same relative accuracy."""
    self._schur *= 0.0
    self._schur += self._BC
    self._schur *= -(dt**2)
    self._schur += self._A
    self._rhs_m *= 0.0
    self._rhs_m += self._BC
    self._rhs_m *= dt**2
    self._rhs_m += self._A
    self._solver.linop = self._schur
    rhs = self._rhs_m.dot(xn, out=self._rhs)
    rhs /= 2 * dt
    rhs -= Byn
    rhs *= 2 * dt
    norm = np.sqrt(rhs.inner(rhs))
    out = self._A.domain.zeros() if out is None else out
    if not norm > 0.0:
        out *= 0.0
        return out, {"niter": 0}
    rhs *= 1.0 / norm
    x = self._solver.dot(rhs, out=out)
    x *= norm
    rhs *= norm
    return x, self._solver._info


def _one_step_spectrum(model_class, dt: float):
    """Eigenvalues of the one-step map (model.integrate) of (u, b, p or q) in the screw pinch R0 = 20 (exact
    equilibrium with a pressure gradient balanced by the current), 4 x 4 x 4 elements, degree 3."""
    model = model_class()
    thermo = model.mhd.sqrt_p if model_class is LinearMHDWithQ else model.mhd.pressure
    props = model.propagators
    solver_params = SolverParameters(tol=1e-12, maxiter=5000)
    props.shear_alf.options = props.shear_alf.Options(precond="StiffnessPreconditioner", solver_params=solver_params)
    props.sound.options = props.sound.Options(
        solver_params=solver_params, solver_params_M3=SolverParameters(tol=1e-15, maxiter=5000)
    )
    props.bg_current.options = props.bg_current.Options(solver_params=SolverParameters(tol=1e-15, maxiter=5000))

    R0 = 20.0
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(
            out_folders=os.path.join(os.getcwd(), "verification_tests", "LinearMHDWithQ_spectrum"),
            sim_folder=model_class.__name__,
        ),
        time_opts=Time(dt=dt, Tend=dt),
        domain=domains.HollowCylinder(a1=0.1, a2=1.0, Lz=2 * np.pi * R0),
        equil=equils.ScrewPinch(a=1.0, R0=R0, B0=1.0, q0=1.25, q1=2.25, n1=0.0, n2=0.0, na=1.0, beta=0.1),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 4)),
        derham_opts=DerhamOptions(degree=(3, 3, 3), bcs=(("dirichlet", "dirichlet"), None, None)),
    )
    sim.allocate()

    vecs = [model.mhd.velocity.spline.vector, model.em_fields.b_field.spline.vector, thermo.spline.vector]
    offsets = np.cumsum([0] + [len(v.toarray()) for v in vecs])
    n = offsets[-1]
    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        for v, a, b in zip(vecs, offsets[:-1], offsets[1:]):
            array_to_psydac(e[a:b], v.space).copy(out=v)
            v.update_ghost_regions()
        model.integrate(dt)
        A[:, j] = np.concatenate([v.toarray() for v in vecs])
    return np.linalg.eigvals(A)


def test_screw_pinch_spectrum_agrees_with_linear_mhd(monkeypatch):
    """One-step map in a screw pinch with a pressure gradient (R0 = 20, interchange-type unstable): LinearMHDWithQ has
    the unstable modes of LinearMHD, with the same growth rate, and no spurious unstable or overstable modes (unlike the
    variational q model, whose pressure-gradient drive is split between two energies). The background q step makes the
    drive -grad(2 q0 q) consistent with -grad p of LinearMHD up to the projection errors."""
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("dense one-step map: serial only")
    monkeypatch.setattr(schur_solver.SchurSolver, "__call__", _schur_call_relative)
    dt = 0.5

    stats = {}
    for model_class in (LinearMHD, LinearMHDWithQ):
        lam = _one_step_spectrum(model_class, dt)
        mod = np.abs(lam)
        # growth above the solver tolerance level (marginal pairs near omega = 0 have |lambda| - 1 ~ 1e-8)
        growing = mod > 1.0 + 1e-6
        overstable = growing & (np.abs(np.angle(lam)) > 1e-6)
        stats[model_class.__name__] = (np.sum(growing), np.log(mod.max()) / dt, np.sum(overstable))
        logger.info(f"{model_class.__name__}: (#growing, max growth rate, #overstable) = {stats[model_class.__name__]}")

    n_lin, rate_lin, _ = stats["LinearMHD"]
    n_q, rate_q, overstable_q = stats["LinearMHDWithQ"]
    assert n_lin > 0 and rate_lin > 1e-4
    assert n_q == n_lin
    assert abs(rate_q - rate_lin) < 0.01 * rate_lin
    assert overstable_q == 0

    MPI.COMM_WORLD.Barrier()
    shutil.rmtree(os.path.join(os.getcwd(), "verification_tests", "LinearMHDWithQ_spectrum"), ignore_errors=True)


if __name__ == "__main__":
    test_slab_waves_1d("Hdiv", False)
