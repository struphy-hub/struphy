"""Linear variational MHD with one linear solve per substep (``crank_nicolson=True``) and the second-order terms in
weak form, with midpoint (``weights="midpoint"``) or frozen weights: exact energy conservation (up to the solver
tolerance) of every substep and of the full step, and the stability of the sound waves of the p model, which grow with
the frozen weight."""

import logging

import numpy as np
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# tolerance of the linear solvers, and the accepted relative energy error after the steps
NUM_EL = (6, 4, 2)
SOLVER_TOL = 1e-13
ENERGY_TOL = 1e-10


def _setup(
    thermo: str,
    u_space: str,
    weights: str = None,
    homogeneous: bool = False,
    noise_b: bool = True,
    dirichlet: bool = False,
):
    """Ideal linear model on a Colella mapping (non-constant metric) with a sheared slab equilibrium (non-uniform B0,
    p0, n0), or on a cuboid with a homogeneous equilibrium, and a random initial perturbation. ``weights=None``: the
    defaults of the propagators (midpoint, frozen in the sound step of the q model)."""
    from struphy import DerhamOptions, FieldsBackground, Simulation, domains, equils, grids, perturbations
    from struphy.linear_algebra.solver import SolverParameters
    from struphy.models import VariationalPerturbativeMHD, VariationalPerturbativeMHD_with_q

    model_class = VariationalPerturbativeMHD_with_q if thermo == "q" else VariationalPerturbativeMHD
    model = model_class(linear=True, crank_nicolson=True, u_space=u_space, with_viscosity=False, with_resistivity=False)
    thermo_var = model.mhd.sqrt_p if thermo == "q" else model.mhd.pressure

    if homogeneous:
        domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
        equil = equils.HomogenSlab(B0x=0.0, B0y=0.0, B0z=1.0, beta=1.0, n0=1.0)
    else:
        domain = domains.Colella(alpha=0.1, Lx=1.0, Ly=1.0, Lz=1.0)
        equil = equils.ShearedSlab(a=1.0, R0=1.0, n1=0.4, n2=0.0, na=0.5, beta=0.6)
    sim = Simulation(
        model=model,
        domain=domain,
        equil=equil,
        grid=grids.TensorProductGrid(num_elements=NUM_EL),
        derham_opts=DerhamOptions(
            degree=(2, 2, 1), bcs=(("dirichlet", "dirichlet"), None, None) if dirichlet else (None, None, None)
        ),
    )

    lin = SolverParameters(tol=SOLVER_TOL, maxiter=3000)
    variants = model.propagator_variants
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model=variants["variat_dens"])
    props.variat_lin_alfven.options = props.variat_lin_alfven.Options(weights=weights or "midpoint", solver_params=lin)
    sound_weights = "frozen" if thermo == "q" else weights
    props.variat_lin_sound.options = props.variat_lin_sound.Options(
        model=variants["variat_lin_sound"], weights=sound_weights, solver_params=lin
    )

    model.mhd.density.add_background(FieldsBackground())
    thermo_var.add_background(FieldsBackground())
    for comp in range(3):
        model.mhd.velocity.add_perturbation(
            perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=comp, seed=10 + comp)
        )
        if noise_b:
            model.em_fields.b_field.add_perturbation(
                perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=comp, seed=20 + comp)
            )
    thermo_var.add_perturbation(perturbations.Noise(direction="e1e2e3", amp=1e-2, comp=None, seed=30))

    sim.allocate()
    return model


def _energy(model):
    model.scalars.update()
    return float(np.squeeze(model.scalars.host_value("en_tot")))


@pytest.mark.parametrize("dirichlet", [False, True])
@pytest.mark.parametrize("weights", ["midpoint", "frozen"])
@pytest.mark.parametrize("u_space", ["H1vec", "Hdiv"])
@pytest.mark.parametrize("prop_name", ["variat_lin_alfven", "variat_lin_sound"])
def test_substep_energy(prop_name: str, u_space: str, weights: str, dirichlet: bool):
    """Each substep of the p model alone conserves the energy, with its second-order term in weak form (the work of
    the force equals the change of the auxiliary energy), periodic and with Dirichlet boundary conditions in eta1."""
    model = _setup("p", u_space, weights=weights, dirichlet=dirichlet)
    prop = getattr(model.propagators, prop_name)
    model.scalars.update()
    aux = "en_mag_2" if prop_name == "variat_lin_alfven" else "en_thermo"
    en0 = _energy(model)
    aux0 = float(np.squeeze(model.scalars.host_value(aux)))
    for _ in range(3):
        prop(0.05)
    en1 = _energy(model)
    aux1 = float(np.squeeze(model.scalars.host_value(aux)))
    # the auxiliary energy takes part in the exchange
    assert abs(aux1 - aux0) > 1e-6 * abs(en0)
    assert abs(en1 - en0) / abs(en0) < ENERGY_TOL, f"relative energy error {abs(en1 - en0) / abs(en0):.3e}"


@pytest.mark.parametrize("u_space", ["H1vec", "Hdiv"])
@pytest.mark.parametrize("thermo", ["q", "p"])
def test_midpoint_energy(thermo: str, u_space: str):
    """All propagators of the ideal linear model with midpoint weights conserve the energy."""
    model = _setup(thermo, u_space)
    en0 = _energy(model)
    u0 = model.mhd.velocity.spline.vector.copy()
    for _ in range(3):
        for prop in vars(model.propagators).values():
            prop(0.05)
    en1 = _energy(model)
    du = model.mhd.velocity.spline.vector - u0
    assert du.inner(du) > 0.0
    assert abs(en1 - en0) / abs(en0) < ENERGY_TOL, f"relative energy error {abs(en1 - en0) / abs(en0):.3e}"


def test_p_sound_stable():
    """Sound waves of the p model (u in Hdiv, homogeneous equilibrium, large time step): the velocity and pressure stay
    bounded with midpoint weights and grow with the frozen weight (in both cases the energy with the auxiliary
    variable is conserved)."""

    def amplitude(weights):
        model = _setup("p", "Hdiv", weights=weights, homogeneous=True, noise_b=False)
        sound = model.propagators.variat_lin_sound
        u, p = model.mhd.velocity.spline.vector, model.mhd.pressure.spline.vector
        norm0 = u.inner(u) + p.inner(p)
        for _ in range(40):
            sound(0.2)
        return (u.inner(u) + p.inner(p)) / norm0

    # |u|^2 + |p|^2 is not the conserved energy: it changes by a bounded factor with midpoint weights
    assert amplitude("midpoint") < 5.0
    assert amplitude("frozen") > 1e3


@pytest.mark.mpi_skip
@pytest.mark.parametrize("prop_name", ["VariationalLinearAlfvenEvolve", "VariationalLinearSoundEvolve"])
def test_midpoint_options(prop_name: str):
    """weights="midpoint" (the default, except in the sound step of the q model) chooses the non-symmetric solver and
    rejects the symmetric ones; the frozen weight the symmetric one."""
    from struphy import propagators

    Options = getattr(propagators, prop_name).Options
    kwargs = {}
    if prop_name == "VariationalLinearSoundEvolve":
        assert Options().weights == "frozen" and Options().solver == "pcg"
        with pytest.raises(AssertionError):
            Options(model="linear_q", weights="midpoint")
        kwargs = {"model": "linear_p"}
    assert Options(**kwargs).weights == "midpoint" and Options(**kwargs).solver == "rpbicgstab"
    assert Options(weights="frozen", **kwargs).solver == "pcg"
    with pytest.raises(AssertionError):
        Options(weights="midpoint", solver="pcg", **kwargs)
    with pytest.raises(AssertionError):
        Options(weights="implicit", **kwargs)


if __name__ == "__main__":
    test_p_sound_stable()
