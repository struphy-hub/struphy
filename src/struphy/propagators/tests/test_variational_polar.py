"""Variational MHD with polar splines, end to end: the projected equilibrium in the polar spaces (the FEEC space of the
Derham) and the evaluation of polar spline functions on the quadrature grid (SplineFunction.eval_tp_fixed_loc with the
polar coefficients, issue #921)."""

import logging

import numpy as np
import pytest
from feectools.linalg.block import BlockVector

from struphy import DerhamOptions, EnvironmentOptions, Simulation, Time, domains, equils, grids, models
from struphy.feec.variational_utilities import L2PressureTransportOperator, L2TransportOperator, NormalizedInverse
from struphy.initial import perturbations
from struphy.models import VariationalLinearMHD, VariationalMHD_with_q
from struphy.propagators.variational_linear_sound_evolve import VariationalLinearSoundEvolve

# relative change of the energy over the steps below: about 1e-8 with the polar coefficients in the evaluation of the
# spline functions, about 2e-7 without (issue #921); measured with the default solver and Newton tolerances of the
# propagators of VariationalMHD_with_q, which bound the energy error and must stay well below this tolerance
ENERGY_TOL = 5e-8


def test_variational_mhd_with_q_polar(mpi_out_folder):
    """The nonlinear variational MHD model (with q = sqrt(p)) on a polar cylinder (C1 polar splines, no hole) with a
    screw pinch: the velocity changes and the energy is conserved."""
    model = VariationalMHD_with_q()
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=mpi_out_folder, sim_folder="variational_polar"),
        time_opts=Time(dt=0.02, Tend=0.02),
        domain=domains.IGAPolarCylinder(num_elements=(4, 6), degree=(2, 2), a=1.0, Lz=4.0),
        equil=equils.ScrewPinch(a=1.0, R0=4.0 / (2 * np.pi), B0=1.0, q0=1.05, q1=1.8, n1=0.0, n2=0.0, na=1.0, beta=0.1),
        grid=grids.TensorProductGrid(num_elements=(4, 6, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
    )
    for comp, (ls, ms) in enumerate((((1,), (1,)), ((1,), (2,)), ((2,), (1,)))):
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=ls, ms=ms, amps=(1e-3,), comp=comp, given_in_basis="physical")
        )
    sim.allocate()

    def energy():
        model.scalars.update()
        return float(np.squeeze(model.scalars.host_value("en_tot")))

    en0 = energy()
    u0 = model.mhd.velocity.spline.vector.copy()
    for _ in range(3):
        for prop in vars(model.propagators).values():
            prop(0.02)
    en1 = energy()
    du = model.mhd.velocity.spline.vector - u0
    assert du.inner(du) > 0.0
    assert abs(en1 - en0) / en0 < ENERGY_TOL, f"relative energy change {abs(en1 - en0) / en0:.2e}"


# The linear and the perturbative variational MHD models with the pressure p: the assembled adjoint matrices of the
# second-order forces (ProjectionAdjointMatrix: the matrix F of the Alfven step, A of the sound step) with polar splines.


def _variational_p_polar(
    model, out_folder: str, sim_folder: str, degree=(2, 2, 1), transport_projection: str = "commuting"
) -> Simulation:
    """Allocated simulation of ``model`` (VariationalLinearMHD or VariationalPerturbativeMHD, ideal) on a polar cylinder
    with a screw pinch, with the propagator options of the model and a velocity perturbation; ``transport_projection``
    of the linear Alfven and sound steps."""
    for name, variant in model.propagator_variants.items():
        prop = getattr(model.propagators, name, None)
        if prop is not None:
            kw = dict(model=variant)
            if name == "variat_lin_sound":
                kw["transport_projection"] = transport_projection
            prop.options = prop.Options(**kw)
    alfven = getattr(model.propagators, "variat_lin_alfven", None)
    if alfven is not None:
        alfven.options = alfven.Options(transport_projection=transport_projection)
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=out_folder, sim_folder=sim_folder),
        time_opts=Time(dt=0.02, Tend=0.02),
        domain=domains.IGAPolarCylinder(num_elements=(4, 6), degree=(2, 2), a=1.0, Lz=4.0),
        equil=equils.ScrewPinch(a=1.0, R0=4.0 / (2 * np.pi), B0=1.0, q0=1.05, q1=1.8, n1=0.0, n2=0.0, na=1.0, beta=0.1),
        grid=grids.TensorProductGrid(num_elements=(4, 6, 2)),
        derham_opts=DerhamOptions(degree=degree, bcs=(("free", "dirichlet"), None, None), polar_splines=True),
    )
    for comp, (ls, ms) in enumerate((((1,), (1,)), ((1,), (2,)), ((2,), (1,)))):
        model.mhd.velocity.add_perturbation(
            perturbations.ModesCos(ls=ls, ms=ms, amps=(1e-3,), comp=comp, given_in_basis="physical")
        )
    sim.allocate()
    return sim


def _random_vector(derham, form: str, rng):
    """A random vector of the (polar, boundary-reduced) space of ``form``: extraction and boundary operators applied to
    random tensor-product coefficients."""
    v = derham.extraction_ops[form].domain.zeros()
    for blk in v.blocks if isinstance(v, BlockVector) else [v]:
        blk._data[:] = rng.standard_normal(blk._data.shape)
    v.update_ghost_regions()
    return derham.boundary_ops[form].dot(derham.extraction_ops[form].dot(v))


@pytest.mark.parametrize(
    "u_space, transport_projection", [("H1vec", "commuting"), ("Hdiv", "commuting"), ("Hdiv", "L2")]
)
def test_projection_adjoint_matrices_polar(mpi_out_folder, u_space, transport_projection):
    """With polar splines, the assembled matrices of the second-order forces are the adjoints of the transport
    operators: :math:`(\\mathbb F \\mathbf b) \\cdot \\mathbf u = \\mathbf l_0 \\cdot \\mathbb T[\\mathbf b] \\mathbf u`
    (Alfven step) and :math:`(\\mathbb A \\mathbf p) \\cdot \\mathbf u = \\mathbf l_0 \\cdot \\mathbb S[\\mathbf p]
    \\mathbf u` (sound step), and :math:`\\mathbb T_0^\\top` is the transpose of :math:`\\mathbb T_0`, with the
    commuting and the :math:`L^2` projections in the transport."""
    model = VariationalLinearMHD(u_space=u_space, with_viscosity=False, with_resistivity=False)
    _variational_p_polar(
        model, mpi_out_folder, f"adjoint_{u_space}_{transport_projection}", transport_projection=transport_projection
    )
    alfven = model.propagators.variat_lin_alfven
    sound = model.propagators.variat_lin_sound
    derham = alfven.derham
    rng = np.random.default_rng(0)
    u = _random_vector(derham, "v" if u_space == "H1vec" else "2", rng)
    b = _random_vector(derham, "2", rng)
    y = _random_vector(derham, "2", rng)
    p = _random_vector(derham, "3", rng)

    def assert_close(a, b_, what):
        assert abs(a - b_) <= 1e-12 * abs(b_), f"{what}: {a:.15e} != {b_:.15e}"

    assert_close(alfven._T0T.dot(y).inner(u), y.inner(alfven._T0.dot(u)), "T0^T")
    alfven._Tw.update_coeffs(b)
    assert_close(alfven._F.dot(b).inner(u), alfven._lf0.inner(alfven._Tw.dot(u)), "F")
    sound._Ss.update_coeffs(p)
    assert_close(sound._A.dot(p).inner(u), sound._lf0.inner(sound._Ss.dot(u)), "A")


# relative change of the energy over three steps; measured: about 2e-9 (VariationalLinearMHD, H1vec), 6e-10 (Hdiv),
# 5e-11 (VariationalPerturbativeMHD, H1vec) and 5e-9 (Hdiv; 1e-8 for the same case on a hollow cylinder with
# tensor-product splines), bounded by the default solver and Newton tolerances of the propagators
ENERGY_TOL_P = 5e-8


@pytest.mark.parametrize(
    "model_class, u_space",
    [
        ("VariationalLinearMHD", "H1vec"),
        ("VariationalLinearMHD", "Hdiv"),
        ("VariationalPerturbativeMHD", "H1vec"),
        ("VariationalPerturbativeMHD", "Hdiv"),
    ],
)
def test_variational_p_models_polar(mpi_out_folder, model_class, u_space):
    """The ideal linear and perturbative variational MHD models (pressure p) with polar splines: the velocity changes
    and the energy ``en_tot`` is conserved."""
    model = getattr(models, model_class)(u_space=u_space, with_viscosity=False, with_resistivity=False)
    # the bracket of the nonlinear model with a velocity in H(div) needs degree >= 2 in all directions
    degree = (2, 2, 2) if (model_class, u_space) == ("VariationalPerturbativeMHD", "Hdiv") else (2, 2, 1)
    _variational_p_polar(model, mpi_out_folder, f"{model_class}_{u_space}", degree=degree)

    def energy():
        model.scalars.update()
        return float(np.squeeze(model.scalars.host_value("en_tot")))

    en0 = energy()
    u0 = model.mhd.velocity.spline.vector.copy()
    for _ in range(3):
        for prop in vars(model.propagators).values():
            prop(0.02)
    en1 = energy()
    du = model.mhd.velocity.spline.vector - u0
    assert du.inner(du) > 0.0
    assert abs(en1 - en0) / en0 < ENERGY_TOL_P, f"relative energy change {abs(en1 - en0) / en0:.2e}"


def test_l2_transport_polar(mpi_out_folder):
    """The linear variational MHD model with a velocity in H(div) and the :math:`L^2` projections in the transport
    (``transport_projection="L2"`` in the Alfven and sound steps) on a polar cylinder: the velocity changes and the
    energy ``en_tot`` (with the second-order fields) is conserved (measured: 3e-10)."""
    model = VariationalLinearMHD(u_space="Hdiv", with_viscosity=False, with_resistivity=False)
    _variational_p_polar(model, mpi_out_folder, "l2_transport", transport_projection="L2")
    assert isinstance(model.propagators.variat_lin_alfven._T0, L2TransportOperator)
    assert isinstance(model.propagators.variat_lin_sound._S0, L2PressureTransportOperator)

    def energy():
        model.scalars.update()
        return float(np.squeeze(model.scalars.host_value("en_tot")))

    en0 = energy()
    u0 = model.mhd.velocity.spline.vector.copy()
    for _ in range(3):
        for prop in vars(model.propagators).values():
            prop(0.02)
    en1 = energy()
    du = model.mhd.velocity.spline.vector - u0
    assert du.inner(du) > 0.0
    assert abs(en1 - en0) / en0 < ENERGY_TOL_P, f"relative energy change {abs(en1 - en0) / en0:.2e}"


@pytest.mark.mpi_skip
def test_l2_transport_options():
    """The :math:`L^2` transport of the sound step is only available for the pressure model."""
    with pytest.raises(AssertionError):
        VariationalLinearSoundEvolve.Options(model="linear_q", transport_projection="L2")
    VariationalLinearSoundEvolve.Options(model="linear_p", transport_projection="L2")


@pytest.mark.parametrize("transport_projection", ["commuting", "L2"])
def test_sound_schur_preconditioner_polar(mpi_out_folder, transport_projection):
    """The Kronecker preconditioner of the Schur complement of the sound step (``schur_precond="StiffnessPreconditioner"``,
    linear p model, velocity in H(div), polar splines): the same velocity and pressure as with the mass-matrix
    preconditioner, with fewer iterations."""
    from struphy.models import VariationalLinearMHD

    results = {}
    for schur_precond in ("MassMatrixPreconditioner", "StiffnessPreconditioner"):
        model = VariationalLinearMHD(u_space="Hdiv", with_viscosity=False, with_resistivity=False)
        _variational_p_polar(
            model,
            mpi_out_folder,
            f"sound_pc_{transport_projection}_{schur_precond}",
            transport_projection=transport_projection,
        )
        sound = model.propagators.variat_lin_sound
        sound.options = sound.Options(
            model="linear_p", transport_projection=transport_projection, schur_precond=schur_precond
        )
        sound.allocate()
        assert (sound._schur_pc is not None) == (schur_precond == "StiffnessPreconditioner")
        niter = 0
        for _ in range(3):
            sound(0.5)
            niter += sound._inv_J._solver._info["niter"]
        # arrays: the two simulations have different Derham objects (and spaces)
        results[schur_precond] = (
            model.mhd.velocity.spline.vector.toarray(allreduce=True),
            model.mhd.pressure.spline.vector.toarray(allreduce=True),
            niter,
        )

    (u_m, p_m, it_m), (u_s, p_s, it_s) = results["MassMatrixPreconditioner"], results["StiffnessPreconditioner"]
    assert np.linalg.norm(u_s - u_m) < 1e-6 * np.linalg.norm(u_m)
    assert np.linalg.norm(p_s - p_m) < 1e-6 * np.linalg.norm(p_m)
    assert it_s < it_m, f"{it_s} iterations with the Kronecker preconditioner, {it_m} with the mass matrix"


def test_sound_schur_preconditioner_fallback(mpi_out_folder):
    """``schur_precond="StiffnessPreconditioner"`` with a velocity in H1vec falls back to the mass-matrix
    preconditioner."""
    from struphy.models import VariationalLinearMHD

    model = VariationalLinearMHD(u_space="H1vec", with_viscosity=False, with_resistivity=False)
    _variational_p_polar(model, mpi_out_folder, "sound_pc_fallback")
    sound = model.propagators.variat_lin_sound
    sound.options = sound.Options(model="linear_p", schur_precond="StiffnessPreconditioner")
    sound.allocate()
    assert sound._schur_pc is None


def test_normalized_inverse_warning(mpi_out_folder, caplog):
    """The inner mass solves of the :math:`L^2` transport (NormalizedInverse, polar splines) converge with the default
    tolerance and iteration limit, and a solve that does not converge is logged as a warning."""
    model = VariationalLinearMHD(u_space="Hdiv", with_viscosity=False, with_resistivity=False)
    _variational_p_polar(model, mpi_out_folder, "normalized_inverse", transport_projection="L2")
    alfven = model.propagators.variat_lin_alfven
    b = _random_vector(alfven.derham, "1", np.random.default_rng(0))
    # a mass matrix with a weight that is not separable, so that its Kronecker preconditioner is not exact (that of M1
    # nearly is on the cylinder)
    M = alfven.mass_ops.create_weighted_mass(
        "Hcurl",
        "Hcurl",
        name="M1w",
        weights=("Ginv", "sqrt_g", lambda e1, e2, e3: 1.0 + 0.9 * e1 * np.sin(2 * np.pi * e2) * np.cos(2 * np.pi * e3)),
        assemble=True,
    )
    with caplog.at_level(logging.WARNING, logger="struphy"):
        x = NormalizedInverse(M).dot(b)
        r = M.dot(x) - b
        assert np.sqrt(r.inner(r)) < 1e-12 * np.sqrt(b.inner(b))
        assert "did not converge" not in caplog.text
        NormalizedInverse(M, maxiter=3).dot(b)
        # the struphy logger writes on rank 0 only
        if alfven.derham.comm is None or alfven.derham.comm.Get_rank() == 0:
            assert "NormalizedInverse(M1w): pcg did not converge in 3 iterations" in caplog.text
