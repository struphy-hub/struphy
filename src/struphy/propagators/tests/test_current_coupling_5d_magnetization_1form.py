import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _model(split: bool, n_hot: float = 1e-1, b_amp: float = 1e-2, mag_options: dict = None):
    """LinearMHDDriftkineticCC with the magnetic field in H(curl), with the magnetization split off or not, in a slab
    with an oblique background field, velocity and magnetic field perturbations and energetic ions, allocated."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, equils, grids, maxwellians, perturbations
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import LinearMHDDriftkineticCC

    model = LinearMHDDriftkineticCC(b_space="Hcurl", projected=True, split_magnetization=split)
    model.energetic_ions.set_markers(loading_params=LoadingParameters(Np=200, seed=1234))
    model.energetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(n_hot, None)))
    model.mhd.velocity.add_perturbation(
        perturbations.ModesCos(ls=(1,), ms=(1,), given_in_basis="2", comp=0, amps=(1e-2,))
    )
    if b_amp > 0.0:
        model.em_fields.b_field.add_perturbation(
            perturbations.ModesSin(ls=(1,), ns=(1,), given_in_basis="1", comp=1, amps=(b_amp,))
        )
    props = model.propagators
    tight = SolverParameters(tol=1e-14, maxiter=3000)
    if split:
        props.shear_alf.options = props.shear_alf.Options(precond="MassMatrixPreconditioner", solver_params=tight)
        props.cc5d_mag.options = props.cc5d_mag.Options(**(mag_options or {}))
    else:
        props.shear_alf.options = props.shear_alf.Options(
            precond="MassMatrixPreconditioner", nonlinear=False, solver_params=tight
        )
    sim = Simulation(
        model=model,
        equil=equils.HomogenSlab(B0x=0.2, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(8, 8, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.allocate()
    return model


def _state(model):
    return model.mhd.velocity.spline.vector.toarray(), model.em_fields.b_field.spline.vector.toarray()


def test_split_converges_to_coupled():
    """One step of shear_alf + cc5d_mag (Lie splitting) against one step of the coupled curl-curl step: the local
    splitting error is O(dt^2) (in b; for dt below the inverse frequency of the grid-scale Alfven waves excited by the
    marker noise)."""
    errors = []
    for dt in (0.02, 0.01, 0.005):
        coupled = _model(split=False)
        coupled.propagators.shear_alf(dt)
        split = _model(split=True)
        split.propagators.shear_alf(dt)
        split.propagators.cc5d_mag(dt)
        (u_c, b_c), (u_s, b_s) = _state(coupled), _state(split)
        u0 = _model(split=False).mhd.velocity.spline.vector.toarray()
        assert xp.linalg.norm(u_s - u0) > 0.0
        errors += [xp.linalg.norm(u_s - u_c) + xp.linalg.norm(b_s - b_c)]
    for e_coarse, e_fine in zip(errors[:-1], errors[1:]):
        assert e_fine < e_coarse / 3, errors


def test_negligible_ions():
    """Without energetic ions the magnetization step does not change the velocity."""
    model = _model(split=True, n_hot=1e-14)
    u0, _ = _state(model)
    model.propagators.cc5d_mag(0.5)
    u1, _ = _state(model)
    assert xp.linalg.norm(u1 - u0) < 1e-10 * xp.linalg.norm(u0)


@pytest.mark.parametrize("b_amp", [0.0, 1e-2])
def test_nonlinear(b_amp: float):
    """The nonlinear step (Lorentz force of the total field) equals the linear one without magnetic field perturbation
    and differs from it with one."""
    du = []
    for nonlinear in (False, True):
        model = _model(split=True, b_amp=b_amp, mag_options={"nonlinear": nonlinear})
        u0, _ = _state(model)
        model.propagators.cc5d_mag(0.5)
        u1, _ = _state(model)
        du += [u1 - u0]
    diff = xp.linalg.norm(du[1] - du[0]) / xp.linalg.norm(du[0])
    if b_amp == 0.0:
        assert diff < 1e-10
    else:
        assert diff > 1e-6
