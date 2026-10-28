"""Linear variational MHD with midpoint weights of the second-order terms (``weights="midpoint"``) in the propagators
with one linear solve per substep (``crank_nicolson=True``): the assembled maps of the weights, exact energy
conservation (up to the solver tolerance) and the stability of the sound waves of the p model, which grow with the
frozen weight."""

import logging

import numpy as np
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)

# tolerance of the linear solvers, and the accepted relative energy error after the steps
SOLVER_TOL = 1e-13
ENERGY_TOL = 1e-10


def _setup(thermo: str, u_space: str, weights: str = "midpoint", homogeneous: bool = False, noise_b: bool = True):
    """Ideal linear model on a Colella mapping (non-constant metric) with a sheared slab equilibrium (non-uniform B0,
    p0, n0), or on a cuboid with a homogeneous equilibrium, and a random initial perturbation."""
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
        grid=grids.TensorProductGrid(num_elements=(6, 4, 2)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )

    lin = SolverParameters(tol=SOLVER_TOL, maxiter=3000)
    variants = model.propagator_variants
    props = model.propagators
    props.variat_dens.options = props.variat_dens.Options(model=variants["variat_dens"])
    props.variat_lin_alfven.options = props.variat_lin_alfven.Options(weights=weights, solver_params=lin)
    props.variat_lin_sound.options = props.variat_lin_sound.Options(
        model=variants["variat_lin_sound"], weights=weights, solver_params=lin
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


def _random(V, rng):
    v = V.zeros()
    for blk in v.blocks if hasattr(v, "blocks") else [v]:
        blk._data[:] = rng.standard_normal(blk._data.shape)
    v.update_ghost_regions()
    return v


@pytest.mark.parametrize("bcs", [(None, None, None), (("dirichlet", "dirichlet"), None, None)])
def test_assemble_by_probing(bcs):
    """A random stencil matrix, assembled from its action: periodic directions whose number of points is a multiple
    of 2p + 1 (10, p = 2), or not (13, p = 2: extra colors for the remainder), and a non-periodic direction."""
    from feectools.linalg.stencil import StencilMatrix
    from maybempi import MPI

    from struphy.feec.psydac_derham import Derham
    from struphy.feec.variational_utilities import assemble_by_probing
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    derham = Derham(
        TensorProductGrid(num_elements=[10, 13, 4]), DerhamOptions(degree=[2, 2, 1], bcs=bcs), comm=MPI.COMM_WORLD
    )
    V = derham.fem_spaces["H1"].coeff_space
    rng = np.random.default_rng(MPI.COMM_WORLD.Get_rank())
    M = StencilMatrix(V, V)
    # random entries in the rows of the process only (not in the ghost rows)
    rows = tuple(slice(p, p + e - s + 1) for p, s, e in zip(V.pads, V.starts, V.ends))
    M._data[rows] = rng.standard_normal(M._data[rows].shape)
    # no columns outside of the domain in the non-periodic direction
    M.remove_spurious_entries()
    A = assemble_by_probing(M.dot, V, V)
    for _ in range(2):
        x = _random(V, rng)
        d = A.dot(x) - M.dot(x)
        y = M.dot(x)
        assert np.sqrt(d.inner(d) / y.inner(y)) < 1e-13


@pytest.mark.parametrize("u_space", ["H1vec", "Hdiv"])
@pytest.mark.parametrize("thermo", ["q", "p"])
def test_weight_map(u_space: str, thermo: str):
    """The assembled maps b -> T[b]^T l and s -> S[s]^T l (Hdiv0_transport_operator.weight_map,
    Pressure_transport_operator.weight_map) agree with the operators with updated weights."""
    model = _setup(thermo, u_space, weights="frozen")
    A, S = model.propagators.variat_lin_alfven, model.propagators.variat_lin_sound
    rng = np.random.default_rng(0)
    for op, lin_form in ((A._Tb.T, A._M2b0), (S._Ss.T, S._lf0)):
        R = op.weight_map(lin_form)
        for _ in range(2):
            x = _random(op.domain, rng)
            y = R.dot(x)
            op.update_coeffs(x)
            d = y - op.dot(lin_form)
            assert np.sqrt(d.inner(d) / y.inner(y)) < 1e-12


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
    """weights="midpoint" (the default) chooses the non-symmetric solver and rejects the symmetric ones; the frozen
    weight the symmetric one."""
    from struphy import propagators

    Options = getattr(propagators, prop_name).Options
    assert Options().weights == "midpoint" and Options().solver == "pbicgstab"
    assert Options(weights="frozen").solver == "pcg"
    with pytest.raises(AssertionError):
        Options(weights="midpoint", solver="pcg")
    with pytest.raises(AssertionError):
        Options(weights="implicit")


if __name__ == "__main__":
    test_p_sound_stable()
