"""Visco-resistive linear variational MHD with the pressure (``VariationalPerturbativeMHD(linear=True)``,
``crank_nicolson=True``): the viscous and resistive heating goes into the auxiliary pressure p-hat, so that the total
energy is conserved (up to the solver tolerance), with the velocity in H1vec and in H(div)."""

import logging

import numpy as np
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# tolerance of the linear solvers, and the accepted relative energy error after the steps
SOLVER_TOL = 1e-13
ENERGY_TOL = 1e-10


def _setup(u_space: str, dirichlet: bool = False, mu: float = 1e-2, eta: float = 1e-2):
    """Linear p model on a Colella mapping (non-constant metric) with a sheared slab equilibrium (non-uniform B0, p0,
    n0) and a random initial perturbation."""
    from struphy import DerhamOptions, FieldsBackground, Simulation, domains, equils, grids, perturbations
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import VariationalPerturbativeMHD

    model = VariationalPerturbativeMHD(linear=True, crank_nicolson=True, u_space=u_space)
    sim = Simulation(
        model=model,
        domain=domains.Colella(alpha=0.1, Lx=1.0, Ly=1.0, Lz=1.0),
        equil=equils.ShearedSlab(a=1.0, R0=1.0, n1=0.4, n2=0.0, na=0.5, beta=0.6),
        grid=grids.TensorProductGrid(num_elements=(6, 4, 2)),
        derham_opts=DerhamOptions(
            degree=(2, 2, 1), bcs=(("dirichlet", "dirichlet"), None, None) if dirichlet else (None, None, None)
        ),
    )

    lin = SolverParameters(tol=SOLVER_TOL, maxiter=3000)
    variants = model.propagator_variants
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model=variants["variat_dens"])
    props.variat_lin_alfven.options = props.variat_lin_alfven.Options(solver_params=lin)
    props.variat_lin_sound.options = props.variat_lin_sound.Options(
        model=variants["variat_lin_sound"], solver_params=lin
    )
    props.variat_viscous.options = props.variat_viscous.Options(
        model=variants["variat_viscous"], mu=mu, solver_params=lin
    )
    props.variat_resist.options = props.variat_resist.Options(
        model=variants["variat_resist"], eta=eta, solver_params=lin
    )

    model.mhd.density.add_background(FieldsBackground())
    model.mhd.pressure.add_background(FieldsBackground())
    for comp in range(3):
        model.mhd.velocity.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=comp, seed=10 + comp)
        )
        model.em_fields.b_field.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=comp, seed=20 + comp)
        )
    model.mhd.pressure.add_perturbation(perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=None, seed=30))

    sim.allocate()
    return model


def _scalar(model, name):
    return float(np.squeeze(model.scalars.host_value(name)))


@pytest.mark.parametrize("dirichlet", [False, True])
@pytest.mark.parametrize("u_space", ["H1vec", "Hdiv"])
@pytest.mark.parametrize("name", ["variat_viscous", "variat_resist"])
def test_dissipation_heats_p_hat(name: str, u_space: str, dirichlet: bool):
    """VariationalViscosity and VariationalResistivity alone: the kinetic (magnetic) energy decreases, the thermal
    energy of p-hat increases by the same amount."""
    model = _setup(u_space, dirichlet=dirichlet)
    prop = getattr(model.propagators, name)
    model.scalars.update()
    en0 = {k: _scalar(model, k) for k in ("en_tot", "en_U", "en_mag_1", "en_thermo")}
    for _ in range(3):
        prop(0.05)
    model.scalars.update()
    en1 = {k: _scalar(model, k) for k in ("en_tot", "en_U", "en_mag_1", "en_thermo")}

    key = "en_U" if name == "variat_viscous" else "en_mag_1"
    lost = en0[key] - en1[key]
    heat = en1["en_thermo"] - en0["en_thermo"]
    # a noticeable part of the kinetic (magnetic) energy is dissipated
    assert lost > 1e-2 * en0[key], f"dissipated energy {lost:.3e} of {en0[key]:.3e}"
    assert abs(heat - lost) / lost < 1e-8, f"heat {heat:.6e}, dissipated {lost:.6e}"
    err = abs(en1["en_tot"] - en0["en_tot"]) / abs(en0["en_tot"])
    assert err < ENERGY_TOL, f"relative energy error {err:.3e}"


@pytest.mark.parametrize("u_space", ["H1vec", "Hdiv"])
def test_full_step_energy(u_space: str):
    """All propagators of the visco-resistive linear model in splitting order conserve the energy."""
    model = _setup(u_space)
    model.scalars.update()
    en0 = _scalar(model, "en_tot")
    for _ in range(3):
        for prop in vars(model.propagators).values():
            prop(0.05)
    model.scalars.update()
    err = abs(_scalar(model, "en_tot") - en0) / abs(en0)
    assert err < ENERGY_TOL, f"relative energy error {err:.3e}"


if __name__ == "__main__":
    test_dissipation_heats_p_hat("variat_viscous", "Hdiv", False)
