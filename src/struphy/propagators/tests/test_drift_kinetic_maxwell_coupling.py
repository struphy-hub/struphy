import logging

import cunumpy as xp
import pytest
from maybempi import MPI

from struphy import domains, equils, set_logging_level

set_logging_level(logging.WARNING)


class _NonUniformSlab(equils.HomogenSlab):
    """HomogenSlab with a density varying in the perpendicular directions, n0 = 1 + 0.5 sin(2 pi x) + 0.2 cos(2 pi y)."""

    def n_xyz(self, x, y, z):
        return 1.0 + 0.5 * xp.sin(2 * xp.pi * x) + 0.2 * xp.cos(2 * xp.pi * y)


class _NonUniformSlabZ(equils.HomogenSlab):
    """HomogenSlab with n0 = 1 + 0.5 sin(2 pi z) (independent of the perpendicular directions)."""

    def n_xyz(self, x, y, z):
        return 1.0 + 0.5 * xp.sin(2 * xp.pi * z)


class _MisalignedSlab(equils.HomogenSlab):
    """HomogenSlab with a perpendicular field component b0_eta1 = 0.3 sin(2 pi eta1), which vanishes at the
    coarser sample points eta1 = 0, 0.5, 1 of the aligned-field check."""

    def unit_bv_1(self, e1, e2, e3):
        return 0.3 * xp.sin(2 * xp.pi * e1)


def _model(control_variate=False, domain=None, polar=False, equil=None, **options):
    """Allocated DriftKineticMaxwell model (slab with a tilted homogeneous B0, full-f guiding centers with epsilon=0.5,
    lam=2, kappa=2, a parallel flow and a B perturbation) with the given options of its coupling propagator; a
    different equilibrium can be passed with ``equil``. With ``polar=True`` in a cylinder with polar splines (enough
    elements in eta_1 for the polar rings on the pole rank, also on 4 MPI ranks)."""
    from struphy import (
        DerhamOptions,
        LoadingParameters,
        Simulation,
        WeightsParameters,
        domains,
        equils,
        grids,
        maxwellians,
        models,
        perturbations,
    )
    from struphy.linear_algebra.solver import SolverParameters

    model = models.DriftKineticMaxwell(epsilon=0.5, lam=2.0, kappa=2.0)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=20, seed=1234),
        weights_params=WeightsParameters(control_variate=control_variate),
    )
    model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(1.0, None), B0=1.0))
    model.kinetic_ions.var.add_initial_condition(
        maxwellians.GyroMaxwellian2D(
            n=(1.0, None), u_para=(0.0, perturbations.ModesCos(ls=(1,), ns=(1,), amps=(1e-1,))), B0=1.0
        )
    )
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesSin(ls=(1,), ns=(1,), given_in_basis="2", comp=1, amps=(1e-1,))
    )

    prop = model.propagators.coupling
    prop.options = prop.Options(solver_params=SolverParameters(tol=1e-13, maxiter=3000), **options)

    if polar:
        geometry = {
            "domain": domains.IGAPolarCylinder(num_elements=(12, 6), degree=(2, 2), a=1.0, Lz=4.0),
            "grid": grids.TensorProductGrid(num_elements=(12, 6, 2)),
            "derham_opts": DerhamOptions(degree=(2, 2, 1), bcs=(("free", "dirichlet"), None, None), polar_splines=True),
        }
    else:
        geometry = {
            "domain": domains.Cuboid(r1=4.0, r2=2.0, r3=4.0) if domain is None else domain,
            "grid": grids.TensorProductGrid(num_elements=(6, 2, 6)),
            "derham_opts": DerhamOptions(degree=(2, 1, 2)),
        }
    sim = Simulation(
        model=model,
        equil=equils.HomogenSlab(B0x=0.3, B0z=1.0) if equil is None else equil,
        **geometry,
    )
    sim.allocate()
    assert sim.derham.polar_splines == polar
    return model


def _state(model):
    e = model.em_fields.e_field.spline.vector.copy()
    b = model.em_fields.b_field.spline.vector.copy()
    v = model.kinetic_ions.var.particles.markers[3, model.kinetic_ions.var.particles.valid_mks].copy()
    return e, b, v


def _norm(x):
    return x.inner(x) ** 0.5


def _random_v1(derham, seed):
    """A random V1 coefficient vector (different on each rank, which does not matter for the tests)."""
    rng = xp.random.default_rng(seed)
    x = derham.V1.zeros()
    for block in x.blocks:
        block._data[:] = rng.standard_normal(block._data.shape)
    x.update_ghost_regions()
    return x


@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("exb_current", [True, False])
@pytest.mark.parametrize("lam", [1.0, 0.0])
def test_energy_conservation(lam, exb_current, matrix_free):
    """The coupling step conserves its energy H exactly (up to the solver tolerance), also for a large time step and
    in the quasi-neutral limit lam=0 (without the initial projection, which changes H once)."""
    solver = {} if exb_current else {"solver": "pcg"}
    model = _model(lam=lam, exb_current=exb_current, matrix_free=matrix_free, initial_projection=False, **solver)
    prop = model.propagators.coupling
    b = model.em_fields.b_field.spline.vector
    energy_0 = prop.energy()
    energy_b_0 = 0.5 * b.inner(prop._M2.dot(b))
    for _ in range(3):
        prop(0.5)
        assert prop._solver._info["success"]
    # H is dominated by the thermal energy: compare its change with the energy exchanged with the B field
    exchanged = abs(0.5 * b.inner(prop._M2.dot(b)) - energy_b_0)
    assert exchanged > 0.1 * energy_b_0
    assert abs(prop.energy() - energy_0) < 1e-10 * exchanged


def test_energy_conservation_mapped():
    """Also in a non-trivial mapping (metric in the parallel response and in the E x B current)."""
    from struphy import domains

    model = _model(domain=domains.Colella(Lx=4.0, Ly=2.0, alpha=0.05, Lz=4.0), initial_projection=False)
    prop = model.propagators.coupling
    energy_0 = prop.energy()
    for _ in range(3):
        prop(0.5)
        assert prop._solver._info["success"]
    assert abs(prop.energy() - energy_0) < 1e-11 * energy_0


def _darwin_operator(case="cuboid", num_elements=(6, 2, 6), degree=(2, 1, 2), **options):
    """DarwinLongitudinalOperator on a periodic unit cube with B0 = e_z (standalone, no particles). The cases cover a
    uniform Cuboid, a density varying in the perpendicular directions or only in eta3, and the Colella mapping."""
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.propagators.drift_kinetic_maxwell_coupling import DarwinLongitudinalOperator
    from struphy.topology.grids import TensorProductGrid

    if case == "cuboid":
        domain, equil = domains.Cuboid(), equils.HomogenSlab(B0z=1.0)
    elif case == "cuboid-nz":
        domain, equil = domains.Cuboid(), _NonUniformSlabZ(B0z=1.0)
    elif case == "cuboid-nxy":
        domain, equil = domains.Cuboid(), _NonUniformSlab(B0z=1.0)
    elif case == "colella":
        domain, equil = domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0), equils.HomogenSlab(B0z=1.0)
    elif case == "colella-nxy":
        domain, equil = domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0), _NonUniformSlab(B0z=1.0)
    elif case == "misaligned":
        domain, equil = domains.Cuboid(), _MisalignedSlab(B0z=1.0)
    else:
        raise ValueError(f"Unknown case {case!r}.")
    equil.domain = domain
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree), comm=MPI.COMM_WORLD)
    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil)
    return derham, mass_ops, DarwinLongitudinalOperator(derham, mass_ops, **options)


@pytest.mark.parametrize("case", ["cuboid", "cuboid-nz", "cuboid-nxy", "colella", "colella-nxy"])
@pytest.mark.parametrize("perp_polarization", [False, True])
def test_darwin_polarization_operator(case, perp_polarization):
    """The Darwin longitudinal polarization operator is symmetric and positive semi-definite (several random pairs),
    equals P = M1gyro on gradients off the kernel of G^T P G, annihilates gradients parallel to b0, and retains the
    polarization of the P-orthogonal complement of the gradient part in the k_perp = 0 modes."""
    derham, mass_ops, op = _darwin_operator(case=case, perp_polarization=perp_polarization)
    P = mass_ops.M1gyro

    # symmetry and positive semi-definiteness on several random pairs
    for seed in (1, 2, 3):
        x = _random_v1(derham, seed)
        y = _random_v1(derham, 100 + seed)
        assert float(x.inner(op.dot(x))) > 0.0
        xPy = float(x.inner(op.dot(y)))
        assert abs(xPy - float(y.inner(op.dot(x)))) < 1e-10 * abs(xPy)

    # dot is alias-safe (out is v)
    x = _random_v1(derham, 7)
    expected = op.dot(x)
    alias = x.copy()
    op.dot(alias, out=alias)
    err = alias - expected
    assert float(err.inner(err)) ** 0.5 < 1e-12 * float(expected.inner(expected)) ** 0.5

    # P_L G = P G on gradients off the kernel (the potential has no part constant in eta1, eta2)
    phi = derham.P0(lambda x, y, z: xp.cos(2 * xp.pi * x) + 0.3 * xp.cos(2 * xp.pi * y))
    op.remove_average(phi)
    Gphi = derham.grad.dot(phi)
    PG = P.dot(Gphi)
    PLG = op.dot(Gphi)
    assert float((PLG - PG).inner(PLG - PG)) ** 0.5 < 1e-10 * float(PG.inner(PG)) ** 0.5

    # kernel: phi constant in eta1, eta2, so G phi is parallel to b0 and carries no polarization
    phi_k = derham.P0(lambda x, y, z: xp.sin(2 * xp.pi * z))
    Gk = derham.grad.dot(phi_k)
    PLGk = op.dot(Gk)
    assert float(PLGk.inner(PLGk)) ** 0.5 < 1e-12 * float(Gk.inner(Gk)) ** 0.5

    # for a weight and metric independent of eta1, eta2 the transverse field E_y(x) has no Darwin polarization
    perpendicular_uniform = case in ("cuboid", "cuboid-nz")
    if perpendicular_uniform:
        v = derham.P1([lambda x, y, z: 0 * x, lambda x, y, z: xp.sin(2 * xp.pi * x), lambda x, y, z: 0 * x])
        PLv = op.dot(v)
        assert float(PLv.inner(PLv)) ** 0.5 < 1e-12 * float(v.inner(v)) ** 0.5

    # k_perp = 0 field E_x(z): full polarization retained on the P-complement of the gradient part, or dropped
    u = derham.P1([lambda x, y, z: xp.sin(2 * xp.pi * z), lambda x, y, z: 0 * x, lambda x, y, z: 0 * x])
    Pu = P.dot(u)
    PLu = op.dot(u)
    if perpendicular_uniform:
        if perp_polarization:
            assert float((PLu - Pu).inner(PLu - Pu)) ** 0.5 < 1e-10 * float(Pu.inner(Pu)) ** 0.5
        else:
            assert float(PLu.inner(PLu)) ** 0.5 < 1e-12 * float(Pu.inner(Pu)) ** 0.5


@pytest.mark.parametrize("case", ["cuboid", "cuboid-nz", "cuboid-nxy", "colella", "colella-nxy"])
def test_darwin_sigma_independence(case):
    """For B0 along eta3 the regularized gradient part (and therefore P_L) does not depend on the kernel
    regularization sigma, also when n0 or the metric varies in the perpendicular directions."""
    from struphy.propagators.drift_kinetic_maxwell_coupling import DarwinLongitudinalOperator

    derham, mass_ops, op = _darwin_operator(case=case)
    op_sigma = DarwinLongitudinalOperator(derham, mass_ops, stab_eps=100.0)
    for seed in (1, 2, 3):
        y = _random_v1(derham, seed)
        ref = op.dot(y)
        diff = ref - op_sigma.dot(y)
        assert float(diff.inner(diff)) ** 0.5 < 1e-9 * float(ref.inner(ref)) ** 0.5


@pytest.mark.parametrize("matrix_free", [False, True])
def test_darwin_energy_conservation(matrix_free):
    """The Darwin coupling step conserves its energy H exactly (up to the solver tolerance), also with the
    matrix-free particle matrices (uniform aligned slab with thermal guiding centers)."""
    from struphy.propagators.drift_kinetic_maxwell_coupling import DarwinLongitudinalOperator

    model = _model(
        equil=equils.HomogenSlab(B0z=1.0),
        lam=0.0,
        polarization="darwin",
        exb_current=False,
        matrix_free=matrix_free,
        initial_projection=False,
        solver="pcg",
    )
    prop = model.propagators.coupling
    assert isinstance(prop._PL, DarwinLongitudinalOperator)
    assert prop._ME is prop._PL or prop._ME.operator is prop._PL
    b = model.em_fields.b_field.spline.vector
    energy_0 = prop.energy()
    energy_b_0 = 0.5 * b.inner(prop._M2.dot(b))
    for _ in range(3):
        prop(0.5)
        assert prop._solver._info["success"]
    exchanged = abs(0.5 * b.inner(prop._M2.dot(b)) - energy_b_0)
    assert exchanged > 0.1 * energy_b_0
    assert abs(prop.energy() - energy_0) < 1e-10 * exchanged


@pytest.mark.parametrize("perp_polarization", [False, True])
def test_darwin_energy_conservation_mapped(perp_polarization):
    """Also in the Colella mapping (metric varying in the perpendicular directions) with an aligned background
    field; the corrected P_L is symmetric, so the mid-point step conserves the energy. The paired run without the
    retained term isolates the gradient part."""
    model = _model(
        domain=domains.Colella(Lx=4.0, Ly=2.0, alpha=0.05, Lz=4.0),
        equil=equils.HomogenSlab(B0z=1.0),
        lam=0.0,
        polarization="darwin",
        exb_current=False,
        initial_projection=False,
        solver="pcg",
        perp_polarization=perp_polarization,
    )
    prop = model.propagators.coupling
    b = model.em_fields.b_field.spline.vector
    energy_0 = prop.energy()
    energy_b_0 = 0.5 * b.inner(prop._M2.dot(b))
    for _ in range(3):
        prop(0.5)
        assert prop._solver._info["success"]
    exchanged = abs(0.5 * b.inner(prop._M2.dot(b)) - energy_b_0)
    assert exchanged > 0.1 * energy_b_0
    assert abs(prop.energy() - energy_0) < 1e-11 * exchanged


def _field_model(
    polarization,
    *,
    field=None,
    comp=0,
    ls=(1,),
    ns=(1,),
    num_elements=(8, 4, 8),
    amplitude=1e-2,
    div_free_b=False,
):
    """DriftKineticMaxwell in a periodic unit cube (B0 = e_z) with a single-mode seed in the electric (``field="e"``,
    given in basis 1) or magnetic (``field="b"``, basis 2) field, or with the divergence-free magnetic seed
    ``B = curl(A_y e_y)``, ``A_y = amplitude/(2 pi) cos(2 pi eta_1) cos(2 pi eta_3)`` (``div_free_b=True``; both
    ``B_x`` and ``B_z`` are nonzero, ``k_y = 0`` and ``div B = 0``). The marker weights are set to zero, so that the
    test isolates the field step: the parallel current and the parallel response would otherwise drive a large
    parallel electric field, which couples to E_y through the curl-curl term (the particle coupling is tested
    separately)."""
    from struphy import (
        DerhamOptions,
        LoadingParameters,
        Simulation,
        WeightsParameters,
        domains,
        equils,
        grids,
        maxwellians,
        models,
        perturbations,
    )
    from struphy.linear_algebra.solver import SolverParameters

    model = models.DriftKineticMaxwell(epsilon=0.5, lam=2.0, kappa=2.0)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=2, seed=1234),
        weights_params=WeightsParameters(control_variate=False),
    )
    model.kinetic_ions.var.add_background(maxwellians.GyroMaxwellian2D(n=(1.0, None), B0=1.0))
    model.kinetic_ions.var.add_initial_condition(maxwellians.GyroMaxwellian2D(n=(1.0, None), B0=1.0))

    if field is not None:
        perturbation = perturbations.ModesSin(
            ls=ls,
            ms=(0,),
            ns=ns,
            amps=(amplitude,),
            given_in_basis="1" if field == "e" else "2",
            comp=comp,
        )
        getattr(model.em_fields, "e_field" if field == "e" else "b_field").add_perturbation(perturbation)
    assert field is not None or div_free_b

    prop = model.propagators.coupling
    prop.options = prop.Options(
        lam=0.0,
        polarization=polarization,
        exb_current=False,
        initial_projection=True,
        solver="pcg",
        solver_params=SolverParameters(tol=1e-13, maxiter=3000),
    )
    sim = Simulation(
        model=model,
        equil=equils.HomogenSlab(B0z=1.0),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=num_elements),
        derham_opts=DerhamOptions(degree=(2, 1, 2)),
    )
    sim.allocate()

    if div_free_b:
        # B = curl(A_y e_y) is divergence-free by the de Rham complex
        derham = prop.derham
        a = amplitude / (2 * xp.pi)
        vector_potential = derham.P1(
            [
                lambda x, y, z: 0 * x,
                lambda x, y, z: a * xp.cos(2 * xp.pi * x) * xp.cos(2 * xp.pi * z),
                lambda x, y, z: 0 * x,
            ]
        )
        b_seed = derham.curl.dot(vector_potential)
        b_seed.copy(out=model.em_fields.b_field.spline.vector)
        model.em_fields.b_field.spline.vector.update_ghost_regions()

    particles = model.kinetic_ions.var.particles
    particles.markers[5, particles.valid_mks] = 0.0
    return model, prop


def test_darwin_compressional_mode():
    """With the divergence-free compressional seed (B_x and B_z, k_y = 0, div B = 0) the Darwin field does not
    oscillate: B is constant after the initial projection (which sets it on the Darwin constraint). The
    full-polarization model oscillates at the compressional frequency (magnetic-field variation of order half its
    amplitude)."""
    dt, n_steps = 0.05, 10
    variation = {}
    for polarization in ("darwin", "full"):
        model, prop = _field_model(polarization, div_free_b=True)
        b = model.em_fields.b_field.spline.vector
        b_norm_0 = float(b.inner(b)) ** 0.5
        div_b = prop.derham.div.dot(b)
        assert float(div_b.inner(div_b)) ** 0.5 < 1e-12 * b_norm_0

        norms = []
        for _ in range(n_steps):
            prop(dt)
            b = model.em_fields.b_field.spline.vector
            norms.append(float(b.inner(b)) ** 0.5)
        # b updates are curls, so the divergence stays zero also through the initial projection
        div_b = prop.derham.div.dot(model.em_fields.b_field.spline.vector)
        assert float(div_b.inner(div_b)) ** 0.5 < 1e-12 * b_norm_0
        # the first step includes the initial projection; afterwards Darwin keeps B, the full model oscillates
        variation[polarization] = max(abs(n - norms[1]) for n in norms[1:]) / b_norm_0

    assert variation["darwin"] < 1e-10
    assert variation["full"] > 0.1


def test_darwin_shear_branch_agrees():
    """For k_perp = 0 the field subsystem of the shear branch (E_x(z), B_y(z)) is identical to the
    full-polarization model: the retained polarization of the kernel modes of G^T P G gives P_L = P on E_x(z).
    With zero marker weights the kinetic parallel response is absent, which isolates the polarization operator
    (the kinetic inertial-Alfvén dispersion is covered by test_verif_DriftKineticMaxwell)."""
    n_steps = 6
    states = {}
    for polarization in ("darwin", "full"):
        model, prop = _field_model(polarization, field="e", comp=0, ls=(0,), ns=(1,), num_elements=(4, 2, 8))
        e = model.em_fields.e_field.spline.vector
        b = model.em_fields.b_field.spline.vector
        steps = []
        for _ in range(n_steps):
            prop(0.05)
            steps.append([block.toarray().copy() for var in (e, b) for block in var.blocks])
        states[polarization] = steps

    scale = max(xp.max(xp.abs(a)) for steps in states.values() for arrays in steps for a in arrays)
    diff = max(
        xp.max(xp.abs(darwin - full))
        for darwin_step, full_step in zip(states["darwin"], states["full"])
        for darwin, full in zip(darwin_step, full_step)
    )
    # a rank without local data has scale = diff = 0 (collective quantities are compared elsewhere)
    assert diff <= 1e-10 * scale + 1e-300


@pytest.mark.xfail(
    strict=True,
    reason="known limitation: the transverse e in ker P_L has no inertia and flips sign every step; it is stored "
    "and passed to the E x B push",
)
def test_darwin_transverse_e_is_not_stored():
    """Desired behaviour: the part of e in the kernel of P_L (transverse to the longitudinal and retained modes)
    has no inertia and should not survive as an alternating field; currently the update e^{n+1} = 2 e_mid - e^n
    with e_mid = 0 flips its sign. This test is expected to fail until the kernel component is replaced by
    e_mid's; it is marked strict so that a fix makes it XPASS and forces an update."""
    model, prop = _field_model("darwin", field="e", comp=1, ls=(1,), ns=(0,), num_elements=(8, 4, 4))
    e0 = model.em_fields.e_field.spline.vector.copy()
    e0_norm = float(e0.inner(e0))
    for _ in range(5):
        prop(0.05)
        en = model.em_fields.e_field.spline.vector
        assert abs(float(en.inner(e0))) / e0_norm < 1e-10


@pytest.mark.parametrize("lam", [1.0, 0.0])
@pytest.mark.parametrize("precond", ["StiffnessPreconditioner", "MassMatrixPreconditioner"])
def test_polar_energy_conservation(lam, precond):
    """With polar splines: the coupling step conserves its energy (with the E x B current; the mass matrix
    preconditioner falls back to the polar one), and the matrix-free particle matrices give the same step as the
    assembled ones."""
    from struphy.feec.preconditioner import PolarMassMatrixPreconditioner

    states = []
    for matrix_free in (False, True):
        model = _model(lam=lam, precond=precond, matrix_free=matrix_free, initial_projection=False, polar=True)
        prop = model.propagators.coupling
        if precond == "MassMatrixPreconditioner":
            assert isinstance(prop._mass_pc, PolarMassMatrixPreconditioner)
        b = model.em_fields.b_field.spline.vector
        energy_0 = prop.energy()
        energy_b_0 = 0.5 * b.inner(prop._M2.dot(b))
        # the largest energy exchanged with the B field
        exchanged = 0.0
        for _ in range(2):
            prop(0.5)
            assert prop._solver._info["success"]
            exchanged = max(exchanged, abs(0.5 * b.inner(prop._M2.dot(b)) - energy_b_0))
            assert abs(prop.energy() - energy_0) < 1e-9 * max(exchanged, 1e-3 * energy_b_0)
        assert exchanged > 0.1 * energy_b_0
        e, b, v = _state(model)
        states.append((e.toarray(), b.toarray(), v))
    for x0, x1 in zip(*states):
        assert xp.max(xp.abs(x1 - x0)) < 1e-8 * xp.max(xp.abs(x0))


@pytest.mark.parametrize("lam", [1.0, 0.0])
def test_matrix_free_agrees(lam):
    """The matrix-free particle matrices give the same step as the assembled ones (with the E x B current)."""
    states = []
    for matrix_free in (False, True):
        model = _model(lam=lam, matrix_free=matrix_free, initial_projection=False)
        model.propagators.coupling(0.5)
        # the two simulations have different Derham objects: compare the arrays
        e, b, v = _state(model)
        states.append((e.toarray(), b.toarray(), v))
    for x0, x1 in zip(*states):
        assert xp.max(xp.abs(x1 - x0)) < 1e-9 * xp.max(xp.abs(x0))


def test_particle_matrices():
    """The parallel response A_par is symmetric positive semi-definite, the E x B operator S antisymmetric."""
    model = _model(matrix_free=True, initial_projection=False)
    prop = model.propagators.coupling
    prop._accumulate()
    x = _random_v1(prop.derham, 1)
    y = _random_v1(prop.derham, 2)
    A, S = prop._A_par, prop._S
    scale = x.inner(A.dot(x)) + y.inner(A.dot(y))
    assert x.inner(A.dot(x)) >= 0.0
    assert abs(x.inner(A.dot(y)) - y.inner(A.dot(x))) < 1e-12 * scale
    scale_s = abs(x.inner(S.dot(y)))
    assert scale_s > 0.0
    assert abs(x.inner(S.dot(x))) < 1e-12 * scale_s
    assert abs(x.inner(S.dot(y)) + y.inner(S.dot(x))) < 1e-12 * scale_s


@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("lam", [1.0, 0.0])
def test_discrete_equations(lam, matrix_free):
    """Without the E x B current, the new state satisfies the Crank-Nicolson equations: Ampère's law
    M_E (e^{n+1} - e^n) / dt = c1 C^T M2 b_mid - c2 j(V_mid) (j linear in v_par at fixed positions and fixed
    a_p = B*/B*_par, which depends on v_par only through the curvature, zero here) and Faraday's law."""
    dt = 0.5
    model = _model(lam=lam, matrix_free=matrix_free, exb_current=False, solver="pcg", initial_projection=False)
    prop = model.propagators.coupling
    e0, b0, _ = _state(model)
    prop._accumulate()
    j0 = prop._accum.vectors[0].copy()
    prop(dt)
    e1, b1, _ = _state(model)
    prop._accumulate()
    j_mid = (j0 + prop._accum.vectors[0]) * 0.5
    b_mid = (b0 + b1) * 0.5
    e_mid = (e0 + e1) * 0.5

    curl_b = prop._curl.T.dot(prop._M2.dot(b_mid)) * prop._c1
    current = j_mid * prop._c2
    residual = curl_b - current - prop._ME.dot(e1 - e0) * (1 / dt)
    assert _norm(residual) < 1e-8 * (_norm(curl_b) + _norm(current))

    faraday = (b1 - b0) * (1 / dt) + prop._curl.dot(e_mid)
    assert _norm(faraday) < 1e-10 * _norm(b1 - b0) / dt


def test_initial_projection():
    """For lam=0, without the initial projection the magnetic energy oscillates with period 2 (the constraint
    residual changes sign in every step); the projection removes this."""

    def en_b_series(projection):
        model = _model(lam=0.0, initial_projection=projection)
        prop = model.propagators.coupling
        b = model.em_fields.b_field.spline.vector
        series = []
        for _ in range(6):
            prop(0.5)
            series.append(0.5 * b.inner(prop._M2.dot(b)))
        return xp.array(series)

    def zigzag(series):
        # second differences relative to the mean: large for a period-2 oscillation
        return float(xp.max(xp.abs(series[2:] - 2 * series[1:-1] + series[:-2])) / xp.mean(series))

    assert zigzag(en_b_series(False)) > 3 * zigzag(en_b_series(True))


def test_dt_change():
    """The Schur solver and its preconditioner are rebuilt only when dt changes; the energy stays conserved."""
    model = _model(n_ref=1.0)
    prop = model.propagators.coupling
    energy_0 = prop.energy()
    prop(0.5)
    solver = prop._solver
    prop(0.5)
    assert prop._solver is solver
    prop(1.0)
    assert prop._solver is not solver
    assert prop._schur_dt == 1.0
    assert abs(prop.energy() - energy_0) < 1e-11 * energy_0


@pytest.mark.mpi_skip
def test_invalid_options():
    from struphy.propagators.drift_kinetic_maxwell_coupling import DriftKineticMaxwellCoupling

    Options = DriftKineticMaxwellCoupling.Options
    with pytest.raises(AssertionError):
        Options(lam=-1.0)
    with pytest.raises(AssertionError):
        Options(chi=-1.0)
    with pytest.raises(AssertionError):
        # no displacement and no polarization current: the perpendicular field is undetermined
        Options(lam=0.0, chi=0.0)
    with pytest.raises(AssertionError):
        Options(c1=0.0)
    with pytest.raises(AssertionError):
        Options(c2=1.0, c3=-1.0)
    with pytest.raises(AssertionError):
        # the E x B current makes the system non-symmetric
        Options(solver="pcg")
    with pytest.raises(AssertionError):
        # the Darwin polarization is quasi-neutral: no displacement current
        Options(lam=1.0, polarization="darwin")
    with pytest.raises(AssertionError):
        Options(polarization="full_perp")
    Options(solver="pcg", exb_current=False)


def test_control_variate_needs_lam():
    """lam=0 needs full-f weights (the particle matrix is indefinite with the control variate)."""
    with pytest.raises(AssertionError):
        _model(lam=0.0, control_variate=True)


def _cold_model(push_para_e_field=False):
    """DriftKineticMaxwell with cold guiding centers (mu ~ 0), uniform B0 and no B perturbation (no mirror force), and
    a large electric field with a parallel component. Optionally pushes with the electric field also in push_parallel
    (forbidden in the model)."""
    from struphy import DerhamOptions, LoadingParameters, Simulation, domains, equils, grids, maxwellians, models
    from struphy.propagators.push_guiding_center_parallel import PushGuidingCenterParallel

    vth = 1e-6
    model = models.DriftKineticMaxwell(epsilon=0.5, lam=2.0, kappa=2.0)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=5, seed=1, moments=(0.0, 0.0, 1.0, vth), B0=1.0)
    )
    model.kinetic_ions.var.add_background(
        maxwellians.GyroMaxwellian2D(n=(1.0, None), vth_para=(1.0, None), vth_perp=(vth, None), B0=1.0)
    )
    if push_para_e_field:
        model.propagators.push_parallel = PushGuidingCenterParallel(
            b_tilde=model.em_fields.b_field, e_field=model.em_fields.e_field
        )
        model.propagators.push_parallel.variables.ions = model.kinetic_ions.var
    Simulation(
        model=model,
        domain=domains.Cuboid(r1=2.0, r2=2.0, r3=2.0),
        equil=equils.HomogenSlab(B0x=0.3, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(4, 4, 4)),
        derham_opts=DerhamOptions(degree=(2, 2, 2)),
    ).allocate()
    e = model.em_fields.e_field.spline.vector
    for blk in e.blocks:
        blk._data[:] = 1.0
    e.update_ghost_regions()
    return model


def _all_vpar(particles):
    """Sorted parallel velocities of all valid markers on all processes (markers may change their process during a
    push, so they are compared as a set)."""
    from maybempi import MPI

    local = particles.markers[3, particles.valid_mks].copy()
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        local = xp.concatenate(comm.allgather(local))
    return xp.sort(local)


@pytest.mark.parametrize("evaluate_e_field", [False, True])
@pytest.mark.parametrize("algo", ["discrete_gradient_1st_order", "explicit"])
def test_parallel_force_applied_once(algo, evaluate_e_field):
    """The parallel electric force b* . E / epsilon is applied only by the coupling step: push_parallel of the model
    leaves v_par unchanged when E != 0 (cold markers, no mirror force), for every algorithm. An electric field passed
    to push_parallel in the model is rejected."""
    model = _cold_model()
    prop = model.propagators.push_parallel
    prop.options = prop.Options(algo=algo, evaluate_e_field=evaluate_e_field)
    # re-allocate with the new options: drop the kernels registered by the first allocation
    prop._init_kernels = ()
    prop._eval_kernels = ()
    prop.allocate()
    particles = model.kinetic_ions.var.particles
    v_0 = _all_vpar(particles)
    prop(0.1)
    assert xp.max(xp.abs(_all_vpar(particles) - v_0)) == 0.0

    # the same push with the electric field changes v_par (by about dt / epsilon * b* . E)
    model.propagators.push_parallel.e_field = model.em_fields.e_field
    prop.options = prop.Options(algo="explicit")
    # re-allocate with the new options: drop the kernels registered by the first allocation
    prop._init_kernels = ()
    prop._eval_kernels = ()
    prop.allocate()
    prop(0.1)
    assert xp.max(xp.abs(_all_vpar(particles) - v_0)) > 1e-2


def test_parallel_force_in_push_para_rejected():
    with pytest.raises(AssertionError):
        _cold_model(push_para_e_field=True)


def test_darwin_requires_aligned_b0():
    """The kernel average and the retained polarization assume B0 along eta3: a tilted homogeneous background field
    is rejected in the allocator."""
    with pytest.raises(AssertionError, match="background field along eta3"):
        _model(lam=0.0, polarization="darwin", exb_current=False, solver="pcg")


def test_darwin_rejects_spatially_varying_misaligned_b0():
    """A perpendicular field component that vanishes at the coarse sample points eta1 = 0, 0.5, 1 is still caught
    by the aligned-field check on its five-point grid."""
    with pytest.raises(AssertionError, match="background field along eta3"):
        _darwin_operator(case="misaligned")
