import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _model(projected: bool, nonlinear: bool, n_hot: float, b_amp: float = 1e-2, **options):
    """LinearMHDDriftkineticCC with the magnetic field in H(curl) in a slab with an oblique background field, a velocity
    and a magnetic field perturbation and energetic ions of density ``n_hot``, allocated."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, equils, grids, maxwellians, perturbations
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import LinearMHDDriftkineticCC
    from struphy.propagators.shear_alfven_curl_curl_current_coupling_5d import ShearAlfvenCurlCurlCurrentCoupling5D

    model = LinearMHDDriftkineticCC(b_space="Hcurl", projected=projected)
    prop = model.propagators.shear_alf
    assert isinstance(prop, ShearAlfvenCurlCurlCurrentCoupling5D)
    model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=200, seed=1234))
    model.energetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(n_hot, None)))
    model.mhd.velocity.add_perturbation(
        perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
    )
    if b_amp > 0.0:
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesSin(ls=(1,), ns=(1,), given_in_basis="1", comp=1, amps=(b_amp,))
        )
    prop.options = prop.Options(nonlinear=nonlinear, solver_params=SolverParameters(tol=1e-14, maxiter=3000), **options)
    sim = Simulation(
        model=model,
        equil=equils.HomogenSlab(B0x=0.2, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(8, 8, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()
    return model


def _energy(model):
    """Perturbed fluid energy plus the energy of the magnetic moments in the parallel perturbation, a^T PB1form b."""
    from struphy.propagators.base import Propagator

    prop = model.propagators.shear_alf
    u = model.mhd.velocity.spline.vector
    b = model.em_fields.b_field.spline.vector
    mass_ops = Propagator.mass_ops
    prop._ACC(prop._ep_scale)
    en_fB = prop._ACC.vectors[0].inner(Propagator.basis_ops.PB1form.dot(b))
    return 0.5 * u.inner(mass_ops.M2n.dot(u)) + 0.5 * b.inner(mass_ops.M1.dot(b)) + en_fB, en_fB


@pytest.mark.parametrize("nonlinear", [False, True])
def test_energy_conservation(nonlinear: bool):
    """With the projection, the implicit step conserves the fluid energy plus the parallel magnetic-moment energy up to
    the solver tolerances, with an O(1) exchange between the two."""
    model = _model(projected=True, nonlinear=nonlinear, n_hot=1e-1, precond="MassMatrixPreconditioner")
    en0, en_fB0 = _energy(model)
    for _ in range(3):
        model.propagators.shear_alf(0.5)
    en1, en_fB1 = _energy(model)
    assert abs(en_fB1 - en_fB0) > 1e-6 * abs(en0)
    assert abs(en1 - en0) < 1e-10 * abs(en0)


def test_nonlinear_without_b_tilde():
    """Without magnetic field perturbation, the nonlinear step (mass matrices of the total field) equals the linear
    one."""
    u = []
    for nonlinear in (False, True):
        model = _model(projected=False, nonlinear=nonlinear, n_hot=1e-1, b_amp=0.0)
        model.propagators.shear_alf(0.1)
        u += [model.mhd.velocity.spline.vector.toarray()]
    assert xp.linalg.norm(u[1] - u[0]) < 1e-10 * xp.linalg.norm(u[0])


@pytest.mark.parametrize("projected", [False, True])
def test_negligible_ions(projected: bool):
    """With negligible energetic ions, the coupled step is the step of ShearAlfvenCurlCurlPropagator."""
    from struphy.propagators.shear_alfven_curl_curl_propagator import ShearAlfvenCurlCurlPropagator

    results = []
    for coupled in (True, False):
        model = _model(projected=projected, nonlinear=False, n_hot=1e-14)
        prop = model.propagators.shear_alf
        if coupled:
            prop(0.2)
        else:
            ShearAlfvenCurlCurlPropagator.__call__(prop, 0.2)
        results += [(model.mhd.velocity.spline.vector.toarray(), model.em_fields.b_field.spline.vector.toarray())]
    for a, b in zip(*results):
        assert xp.linalg.norm(a - b) < 1e-8 * xp.linalg.norm(b)
