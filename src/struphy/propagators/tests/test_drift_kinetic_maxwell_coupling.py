import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _model(control_variate=False, domain=None, **options):
    """Allocated DriftKineticMaxwell model (slab with a tilted homogeneous B0, full-f guiding centers with alpha=1,
    epsilon=0.5, a parallel flow and a B perturbation) with the given options of its coupling propagator."""
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

    model = models.DriftKineticMaxwell(alpha=1.0, epsilon=0.5)
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

    Simulation(
        model=model,
        domain=domains.Cuboid(r1=4.0, r2=2.0, r3=4.0) if domain is None else domain,
        equil=equils.HomogenSlab(B0x=0.3, B0z=1.0),
        grid=grids.TensorProductGrid(num_elements=(6, 2, 6)),
        derham_opts=DerhamOptions(degree=(2, 1, 2)),
    ).allocate()
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
@pytest.mark.parametrize("kappa", [1.0, 0.0])
def test_energy_conservation(kappa, exb_current, matrix_free):
    """The coupling step conserves its energy H exactly (up to the solver tolerance), also for a large time step and
    in the quasi-neutral limit kappa=0 (without the initial projection, which changes H once)."""
    solver = {} if exb_current else {"solver": "pcg"}
    model = _model(kappa=kappa, exb_current=exb_current, matrix_free=matrix_free, initial_projection=False, **solver)
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


@pytest.mark.parametrize("kappa", [1.0, 0.0])
def test_matrix_free_agrees(kappa):
    """The matrix-free particle matrices give the same step as the assembled ones (with the E x B current)."""
    states = []
    for matrix_free in (False, True):
        model = _model(kappa=kappa, matrix_free=matrix_free, initial_projection=False)
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
@pytest.mark.parametrize("kappa", [1.0, 0.0])
def test_discrete_equations(kappa, matrix_free):
    """Without the E x B current, the new state satisfies the Crank-Nicolson equations: Ampère's law
    M_E (e^{n+1} - e^n) / dt = c1 C^T M2 b_mid - c2 j(V_mid) (j linear in v_par at fixed positions and fixed
    a_p = B*/B*_par, which depends on v_par only through the curvature, zero here) and Faraday's law."""
    dt = 0.5
    model = _model(kappa=kappa, matrix_free=matrix_free, exb_current=False, solver="pcg", initial_projection=False)
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
    """For kappa=0, without the initial projection the magnetic energy oscillates with period 2 (the constraint
    residual changes sign in every step); the projection removes this."""

    def en_b_series(projection):
        model = _model(kappa=0.0, initial_projection=projection)
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
        Options(kappa=-1.0)
    with pytest.raises(AssertionError):
        Options(chi=-1.0)
    with pytest.raises(AssertionError):
        # no displacement and no polarization current: the perpendicular field is undetermined
        Options(kappa=0.0, chi=0.0)
    with pytest.raises(AssertionError):
        Options(c1=0.0)
    with pytest.raises(AssertionError):
        Options(c2=1.0, c3=-1.0)
    with pytest.raises(AssertionError):
        # the E x B current makes the system non-symmetric
        Options(solver="pcg")
    Options(solver="pcg", exb_current=False)


def test_control_variate_needs_kappa():
    """kappa=0 needs full-f weights (the particle matrix is indefinite with the control variate)."""
    with pytest.raises(AssertionError):
        _model(kappa=0.0, control_variate=True)
