import logging

import cunumpy as xp
import pytest

from struphy import set_logging_level

set_logging_level(logging.WARNING)


def _model(model_name="VlasovMaxwellQuasiNeutral", control_variate=False, **options):
    """Allocated model (1D-ish slab, full-f electrons with alpha=1, epsilon=-1, a density and a B perturbation) and
    its coupling propagator(s) with the given options."""
    from struphy import (
        DerhamOptions,
        LoadingParameters,
        Simulation,
        WeightsParameters,
        domains,
        grids,
        maxwellians,
        models,
        perturbations,
    )
    from struphy.linear_algebra.solver import SolverParameters

    model = getattr(models, model_name)(alpha=1.0, epsilon=-1.0)
    model.kinetic_ions.set_markers(
        loading_params=LoadingParameters(ppc=40, seed=1234),
        weights_params=WeightsParameters(control_variate=control_variate),
    )
    model.kinetic_ions.var.add_background(maxwellians.Maxwellian3D(n=(1.0, None)))
    model.kinetic_ions.var.add_initial_condition(
        maxwellians.Maxwellian3D(n=(1.0, perturbations.ModesCos(ls=(1,), amps=(1e-1,))))
    )
    model.em_fields.b_field.add_perturbation(
        perturbations.ModesSin(ls=(1,), ms=(1,), given_in_basis="2", comp=2, amps=(1e-1,))
    )

    solver_params = SolverParameters(tol=1e-13, maxiter=3000)
    if model_name == "VlasovMaxwellQuasiNeutral":
        prop = model.propagators.coupling
        prop.options = prop.Options(solver_params=solver_params, **options)
    else:
        for prop in (model.propagators.maxwell, model.propagators.coupling_va):
            prop.options = prop.Options(solver_params=solver_params)

    Simulation(
        model=model,
        domain=domains.Cuboid(r1=4.0, r2=2.0),
        grid=grids.TensorProductGrid(num_elements=(8, 4, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    ).allocate()
    return model


def _state(model):
    e = model.em_fields.e_field.spline.vector.copy()
    b = model.em_fields.b_field.spline.vector.copy()
    v = model.kinetic_ions.var.particles.velocities.copy()
    return e, b, v


def _norm(x):
    return x.inner(x) ** 0.5


def _vnorm(v):
    """Global 2-norm of the local marker velocities (a collective operation)."""
    from maybempi import MPI

    from struphy.propagators.base import Propagator

    val = xp.array([float(xp.sum(v**2))])
    comm = Propagator.derham.comm
    if comm is not None and comm.Get_size() > 1:
        comm.Allreduce(MPI.IN_PLACE, val, op=MPI.SUM)
    return float(val[0]) ** 0.5


# explicit coefficients with c1 != |c2| != |c3| (the species ones are c1 = 1, c2 = c3 = -1 here)
COEFFS = {"species": {}, "explicit": {"c1": 2.0, "c2": -0.5, "c3": -3.0}}


@pytest.mark.parametrize("coeffs", list(COEFFS))
@pytest.mark.parametrize("kappa", [1.0, 0.0])
@pytest.mark.parametrize("precond", ["StiffnessPreconditioner", "MassMatrixPreconditioner"])
def test_energy_conservation(kappa, precond, coeffs):
    """The coupling step conserves its energy H exactly (up to the solver tolerance), also for dt >> 1/omega_p and
    in the quasi-neutral limit kappa=0 (without the initial projection, after which the kappa=0 step at frozen
    positions exchanges no energy)."""
    model = _model(kappa=kappa, precond=precond, initial_projection=False, **COEFFS[coeffs])
    prop = model.propagators.coupling
    b = model.em_fields.b_field.spline.vector
    energy_0 = prop.energy()
    energy_b_0 = 0.5 * b.inner(prop._M2.dot(b))
    for _ in range(3):
        prop(2.0)
        assert prop._solver._info["success"]
    # H is dominated by the thermal energy: compare its change also with the energy exchanged with the B field
    exchanged = abs(0.5 * b.inner(prop._M2.dot(b)) - energy_b_0)
    assert exchanged > energy_b_0
    assert abs(prop.energy() - energy_0) < 1e-9 * exchanged


@pytest.mark.parametrize("matrix_free", [False, True])
@pytest.mark.parametrize("kappa", [1.0, 0.0])
def test_discrete_equations(kappa, matrix_free):
    """The new state satisfies the Crank-Nicolson equations: Ampère's law
    kappa M1 (e^{n+1} - e^n) / dt = c1 C^T M2 b_mid - c2 j(V_mid) (with V^{n+1} from the pusher, j linear in V at
    fixed positions) and Faraday's law (b^{n+1} - b^n) / dt = -C e_mid, with e_mid = e^{n+1} for kappa = 0 (without
    the initial projection, so that the step is not trivial)."""
    dt = 0.5
    model = _model(kappa=kappa, matrix_free=matrix_free, initial_projection=False, **COEFFS["explicit"])
    prop = model.propagators.coupling
    e0, b0, _ = _state(model)
    prop._accum(*prop._args_accum)
    j0 = prop._accum.vectors[0].copy()
    prop(dt)
    e1, b1, _ = _state(model)
    prop._accum(*prop._args_accum)
    j_mid = (j0 + prop._accum.vectors[0]) * 0.5
    b_mid = (b0 + b1) * 0.5
    e_mid = (e0 + e1) * 0.5 if kappa > 0.0 else e1

    curl_b = prop._curl.T.dot(prop._M2.dot(b_mid)) * prop._c1
    current = j_mid * prop._c2
    residual = curl_b - current - prop._M1.dot(e1 - e0) * (kappa / dt)
    assert _norm(residual) < 1e-8 * (_norm(curl_b) + _norm(current))

    faraday = (b1 - b0) * (1 / dt) + prop._curl.dot(e_mid)
    assert _norm(faraday) < 1e-10 * _norm(b1 - b0) / dt

    if kappa == 0.0:
        # weakly divergence-free mid-point current
        div_j = prop.derham.grad.T.dot(j_mid)
        assert _norm(div_j) < 1e-8 * _norm(j_mid)


def test_dt_change():
    """The Schur solver and its preconditioner are rebuilt only when dt changes (here with a given n_ref, so the
    Schur operator is built before any accumulation); the energy stays conserved across the change."""
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
    assert abs(prop.energy() - energy_0) < 1e-12 * energy_0


def test_agrees_with_split_propagators():
    """For kappa=1 and the species coefficients, one step agrees with MaxwellWeakAmpere followed by
    VlasovAmpereCoupling up to the splitting error O(dt^2)."""
    diffs = []
    for dt in (0.1, 0.05):
        coupled = _model()
        coupled.propagators.coupling(dt)
        split = _model("VlasovMaxwellOneSpecies")
        split.propagators.maxwell(dt)
        split.propagators.coupling_va(dt)
        # the two simulations have different Derham objects: compare the arrays (global 2-norms)
        diff = 0.0
        for x_c, x_s in zip(_state(coupled), _state(split)):
            x_c = x_c if isinstance(x_c, xp.ndarray) else x_c.toarray()
            x_s = x_s if isinstance(x_s, xp.ndarray) else x_s.toarray()
            diff += _vnorm(x_c - x_s) / _vnorm(x_s)
        diffs.append(diff)
    # local splitting error O(dt^2): halving dt divides the difference by about 4
    assert 3.0 < diffs[0] / diffs[1] < 5.0, diffs


def test_quasi_neutral_limit():
    """b and V converge like kappa to the kappa=0 solution (of the step without the initial projection); the
    iteration counts stay bounded as kappa -> 0."""
    dt = 0.5
    results = {}
    niter = {}
    for kappa in (1e-2, 1e-4, 0.0):
        model = _model(kappa=kappa, initial_projection=False)
        prop = model.propagators.coupling
        prop(dt)
        niter[kappa] = prop._solver._info["niter"]
        _, b1, v1 = _state(model)
        results[kappa] = (b1.toarray(), v1)

    b_ref, v_ref = results[0.0]
    errs = [_vnorm(results[k][0] - b_ref) + _vnorm(results[k][1] - v_ref) for k in (1e-2, 1e-4)]
    assert errs[1] < 3e-2 * errs[0], errs
    assert max(niter.values()) <= 2 * min(niter.values()), niter


def _constraint_residual(prop):
    """r = c1 C^T M2 b - c2 j(V) at the current markers."""
    b = prop.variables.b.spline.vector
    prop._accum(*prop._args_accum)
    return prop._curl.T.dot(prop._M2.dot(b)) * prop._c1 - prop._accum.vectors[0] * prop._c2


@pytest.mark.parametrize("matrix_free", [False, True])
def test_initial_projection(matrix_free):
    """kappa=0: without the initial projection, the constraint residual r = c1 C^T M2 b - c2 j(V) changes sign in
    every step (period-2 oscillation). The projection in the setup makes r vanish, does not depend on dt, and the
    steps then leave b and V unchanged (frozen positions)."""
    # without projection: r^{n+1} = -r^n
    model = _model(kappa=0.0, initial_projection=False, matrix_free=matrix_free)
    prop = model.propagators.coupling
    r0 = _constraint_residual(prop)
    prop(0.5)
    r1 = _constraint_residual(prop)
    assert _norm(r0) > 0.0
    assert _norm(r1 + r0) < 1e-10 * _norm(r0)

    # with projection (default), for two dt
    projected = []
    for dt in (0.5, 2.0):
        model = _model(kappa=0.0, matrix_free=matrix_free)
        prop = model.propagators.coupling
        j_scale = abs(prop._c2) * _norm(_constraint_residual(prop) * 0.0 + prop._accum.vectors[0])
        prop.prepare_time_step(dt)
        assert _norm(_constraint_residual(prop)) < 1e-10 * j_scale
        e, b, v = _state(model)
        projected.append((b.toarray(), v))
        for _ in range(2):
            prop(dt)
        _, b2, v2 = _state(model)
        assert _norm(b2 - b) < 1e-10 * _norm(b)
        assert _vnorm(v2 - v) < 1e-10 * _vnorm(v)
        assert _norm(_constraint_residual(prop)) < 1e-10 * j_scale
    (b_a, v_a), (b_b, v_b) = projected
    assert _vnorm(b_a - b_b) < 1e-10 * _vnorm(b_a)
    assert _vnorm(v_a - v_b) < 1e-10 * _vnorm(v_a)


def test_control_variate_quasi_neutral_rejected():
    """The quasi-neutral limit needs full-f weights."""
    with pytest.raises(AssertionError, match="quasi-neutral"):
        _model(control_variate=True, kappa=0.0)


@pytest.mark.mpi_skip
def test_options():
    """Invalid coefficients are rejected; solver="cg" takes no preconditioner."""
    from struphy.propagators.vlasov_maxwell_coupling import VlasovMaxwellCoupling

    Options = VlasovMaxwellCoupling.Options
    with pytest.raises(AssertionError):
        Options(kappa=-1.0)
    with pytest.raises(AssertionError):
        Options(c1=0.0)
    with pytest.raises(AssertionError):
        Options(c2=1.0, c3=-1.0)
    with pytest.raises(AssertionError):
        Options(precond="HiptmairXu")
    with pytest.raises(AssertionError):
        Options(matrix_free=1)
    assert Options().matrix_free is False
    with pytest.raises(AssertionError):
        Options(initial_projection=None)
    assert Options().initial_projection is True
    assert Options().precond == "StiffnessPreconditioner"
    assert Options(solver="cg").precond is None
