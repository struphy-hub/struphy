"""Linear variational MHD with the velocity in H(div) (``u_space="Hdiv"``): exact energy conservation (up to the
solver tolerance) of the propagators with one linear solve per substep (``crank_nicolson=True``) of the ideal models,
the only ones with this option."""

import logging

import numpy as np
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# tolerance of the linear solvers, and the accepted relative energy error after the steps
SOLVER_TOL = 1e-13
ENERGY_TOL = 1e-10


def _setup(thermo: str, schur_precond: str = "MassMatrixPreconditioner"):
    """Ideal linear model with u in Hdiv on a Colella mapping (non-constant metric), a sheared slab equilibrium
    (non-uniform B0, p0, n0) and a random initial perturbation."""
    from struphy import DerhamOptions, FieldsBackground, Simulation, domains, equils, grids, perturbations
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import VariationalPerturbativeMHD, VariationalPerturbativeMHD_with_q

    model_class = VariationalPerturbativeMHD_with_q if thermo == "q" else VariationalPerturbativeMHD
    model = model_class(linear=True, crank_nicolson=True, u_space="Hdiv", with_viscosity=False, with_resistivity=False)
    thermo_var = model.mhd.sqrt_p if thermo == "q" else model.mhd.pressure

    sim = Simulation(
        model=model,
        domain=domains.Colella(alpha=0.1, Lx=1.0, Ly=1.0, Lz=1.0),
        equil=equils.ShearedSlab(a=1.0, R0=1.0, n1=0.4, n2=0.0, na=0.5, beta=0.6),
        grid=grids.TensorProductGrid(num_elements=(6, 4, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )

    lin = SolverParameters(tol=SOLVER_TOL, maxiter=3000)
    variants = model.propagator_variants
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model=variants["variat_dens"])
    props.variat_lin_alfven.options = props.variat_lin_alfven.Options(solver_params=lin, schur_precond=schur_precond)
    props.variat_lin_sound.options = props.variat_lin_sound.Options(
        model=variants["variat_lin_sound"], solver_params=lin
    )

    model.mhd.density.add_background(FieldsBackground())
    thermo_var.add_background(FieldsBackground())
    for comp in range(3):
        model.mhd.velocity.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=comp, seed=10 + comp)
        )
        model.em_fields.b_field.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=comp, seed=20 + comp)
        )
    thermo_var.add_perturbation(perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=None, seed=30))

    sim.allocate()
    return model


def _energy(model):
    model.scalars.update()
    return float(np.squeeze(model.scalars.host_value("en_tot")))


def _assert_conserved(model, props, dt=0.05, n_steps=3):
    en0 = _energy(model)
    u0 = model.mhd.velocity.spline.vector.copy()
    for _ in range(n_steps):
        for prop in props:
            prop(dt)
    en1 = _energy(model)
    du = model.mhd.velocity.spline.vector - u0
    # the step does something
    assert du.inner(du) > 0.0
    assert abs(en1 - en0) / abs(en0) < ENERGY_TOL, f"relative energy error {abs(en1 - en0) / abs(en0):.3e}"


@pytest.mark.parametrize("thermo", ["q", "p"])
@pytest.mark.parametrize("names", [("variat_lin_alfven",), ("variat_lin_sound",)])
def test_substep_energy(thermo: str, names: tuple):
    """VariationalLinearAlfvenEvolve and VariationalLinearSoundEvolve alone conserve the energy."""
    model = _setup(thermo)
    _assert_conserved(model, [getattr(model.propagators, n) for n in names])


@pytest.mark.parametrize("schur_precond", ["MassMatrixPreconditioner", "StiffnessPreconditioner"])
@pytest.mark.parametrize("thermo", ["q", "p"])
def test_full_step_energy(thermo: str, schur_precond: str):
    """All propagators of the ideal linear model in splitting order (including the density) conserve the energy,
    with the mass-matrix and with the Kronecker preconditioner of the shear-Alfven Schur system."""
    model = _setup(thermo, schur_precond=schur_precond)
    _assert_conserved(model, list(vars(model.propagators).values()))


@pytest.mark.mpi_skip
@pytest.mark.parametrize("cls_name", ["VariationalPerturbativeMHD", "VariationalPerturbativeMHD_with_q"])
def test_hdiv_needs_ideal_linear_cn(cls_name: str):
    """u_space="Hdiv" is rejected for the nonlinear (perturbative) models, without crank_nicolson=True and for the
    dissipative models."""
    from struphy import models

    model_class = getattr(models, cls_name)
    ideal = {"with_viscosity": False, "with_resistivity": False}
    with pytest.raises(AssertionError):
        model_class(u_space="Hdiv", **ideal)
    with pytest.raises(AssertionError):
        model_class(linear=True, u_space="Hdiv", **ideal)
    with pytest.raises(AssertionError):
        model_class(linear=True, crank_nicolson=True, u_space="Hdiv")
    with pytest.raises(AssertionError):
        model_class(linear=True, crank_nicolson=True, u_space="Hdiv", with_viscosity=False)
    with pytest.raises(AssertionError):
        model_class(linear=True, crank_nicolson=True, u_space="Hcurl", **ideal)
    model_class(linear=True, crank_nicolson=True, u_space="Hdiv", **ideal)


if __name__ == "__main__":
    test_full_step_energy("q", "StiffnessPreconditioner")
