"""Reuse polar factors without retaining obsolete matrices or confusing solver configurations."""

import gc
import subprocess
import sys
from weakref import WeakValueDictionary

import cunumpy as xp
import numpy as np
import pytest
import scipy.sparse as sparse

from struphy.geometry.tests.test_domain import _cupy_installed, serial_child_env
from struphy.polar import kron_2dx1d as kron
from struphy.polar.tests.test_polar_cupy import DenseDeviceSparse, fill, make_polar_derham, to_host


@pytest.mark.parametrize("kind", ("schur", "lu", "ilu"))
def test_poloidal_factors_match_contents_and_settings(kind):
    cache = WeakValueDictionary()
    A = sparse.csr_matrix([[4.0, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]])
    pole = np.array([True, False, False])
    first = kron.make_poloidal_solver(A, kind, pole=pole, _cache=cache)
    same = kron.make_poloidal_solver(A.tocsc(), kind, pole=pole.copy(), _cache=cache)
    assert first is same
    unsorted = sparse.coo_matrix(
        ([4.0, 0.5, 0.5, 2.0, 5.0, 1.0, 1.0, 3.0, 0.0], ([0, 0, 0, 1, 1, 1, 2, 2, 0], [0, 1, 1, 0, 1, 2, 1, 2, 2])),
        shape=(3, 3),
    )
    assert kron.make_poloidal_solver(unsorted, kind, pole=pole, _cache=cache) is first
    # Mutating the original matrix must never retrieve stale factors.
    A.data *= 2.0
    changed = kron.make_poloidal_solver(A, kind, pole=pole, _cache=cache)
    assert changed is not first
    other_kind = "ilu" if kind != "ilu" else "lu"
    assert kron.make_poloidal_solver(A, other_kind, pole=pole, _cache=cache) is not changed
    rhs = np.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(changed.solve(rhs), first.solve(rhs) / 2, rtol=1e-12, atol=1e-12)
    transposed = kron.make_poloidal_solver(A.T, kind, pole=pole, _cache=cache)
    assert transposed is not changed
    np.testing.assert_allclose(A.T @ transposed.solve(rhs), rhs, atol=1e-12)
    if kind == "ilu":
        different_options = kron.make_poloidal_solver(A, kind, drop_tol=1e-2, pole=pole, _cache=cache)
        assert different_options is not changed
    elif kind == "schur":
        other_pole = np.array([False, True, False])
        different_options = kron.make_poloidal_solver(A, kind, pole=other_pole, _cache=cache)
        assert different_options is not changed
        approximate = kron.make_poloidal_solver(A, kind, pole=pole, schur_outer="ilu", _cache=cache)
        assert approximate is not changed and not approximate.exact
        outer = [(np.eye(1), A.toarray()[1:, 1:])]
        separable = kron.make_poloidal_solver(A, kind, pole=pole, outer_kronecker=outer, _cache=cache)
        assert separable is not changed
        np.testing.assert_allclose(A @ separable.solve(rhs), rhs, atol=1e-12)


def test_1d_factors_reuse_exact_values_and_release_old_entries(monkeypatch):
    cache = WeakValueDictionary()
    calls = []
    original = kron.splu

    def factorize(matrix):
        calls.append(matrix.shape)
        return original(matrix)

    monkeypatch.setattr(kron, "splu", factorize)
    A = np.array([[3.0, 1.0], [0.0, 2.0]])
    first = kron._factorized_1d(A, cache)
    same = kron._factorized_1d(A.copy(), cache)
    assert same is first and len(calls) == 1
    A[0, 0] += 1e-14  # no tolerance-based matching of distinct operators
    changed = kron._factorized_1d(A, cache)
    assert changed is not first and len(calls) == 2
    single_precision = kron._factorized_1d(A.astype(np.float32), cache)
    assert single_precision is not changed and len(calls) == 3
    assert len(cache) == 3
    del first, same, changed, single_precision
    gc.collect()
    assert len(cache) == 0


def test_preconditioners_share_factors_across_spaces_and_solver_instances(monkeypatch):
    from struphy.feec.preconditioner import PolarMassMatrixPreconditioner

    _, derham, mass = make_polar_derham("IGAPolarCylinder")
    solvers = [PolarMassMatrixPreconditioner(getattr(mass, "M" + form)).solver for form in ("0", "1", "2", "3")]
    assert all(s.domain._factorization_cache is solvers[0].domain._factorization_cache for s in solvers)
    seen = {}
    reused = 0
    for solver in solvers:
        for group, factor in zip(solver.groups, solver._lu_3):
            key = kron._matrix_key(group.A_3)
            if key in seen:
                assert factor is seen[key]
                reused += 1
            seen[key] = factor
    assert reused > 0

    def unexpected_factorization(*args, **kwargs):
        pytest.fail("an identical solver repeated the factorization")

    monkeypatch.setattr(kron, "splu", unexpected_factorization)
    monkeypatch.setattr(kron, "lu_factor", unexpected_factorization)
    duplicate = kron.PolarKron2Dx1DSolver(solvers[1].domain, solvers[1].groups)
    assert duplicate._lu_3[0] is solvers[1]._lu_3[0]
    assert duplicate._pol_solvers[0] is solvers[1]._pol_solvers[0]
    _, other_derham, _ = make_polar_derham("IGAPolarCylinder", with_mass=False)
    assert other_derham.V0pol._factorization_cache is not solvers[0].domain._factorization_cache


def test_mass_weight_update_reuses_1d_factors_and_rebuilds_poloidal_factors():
    from struphy.feec.preconditioner import PolarMassMatrixPreconditioner

    _, _, mass = make_polar_derham("IGAPolarCylinder")
    pc = PolarMassMatrixPreconditioner(mass.M0)
    previous = pc.solver
    updated = mass.create_weighted_mass(
        "H1", "H1", weights=[[lambda e1, e2, e3: 1 + e1 + 0 * e2 + 0 * e3]], assemble=True
    )
    pc.update_mass_operator(updated)
    assert pc.solver._lu_3[0] is previous._lu_3[0]
    assert pc.solver._pol_solvers[0] is not previous._pol_solvers[0]
    x = updated._V_boundary_op.dot(fill(updated.domain.zeros(), seed=96))
    actual = pc.dot(updated.dot(x))
    for a, b in zip(to_host(actual, interior=True), to_host(x, interior=True)):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)
    same_weights = pc.solver
    pc.update_mass_operator(updated)
    assert pc.solver._pol_solvers[0] is same_weights._pol_solvers[0]


def check_device_factor_reuse(fake=False):
    from contextlib import nullcontext

    from struphy.pic.tests.cuda_emulation import emulated_launches
    from struphy.polar.linear_operators import set_device_sparse_module

    previous = set_device_sparse_module(DenseDeviceSparse) if fake else None
    cache = WeakValueDictionary()
    matrix = np.array([[3.0, 1.0], [0.0, 2.0]])
    with xp.use_backend("numpy"):
        host_factor = kron._factorized_1d(matrix, cache)
    try:
        with xp.use_backend("cupy"), emulated_launches() if fake else nullcontext():
            device_factor = kron._factorized_1d(matrix, cache)
            assert device_factor is not host_factor
            assert kron._factorized_1d(matrix.copy(), cache) is device_factor
            np.testing.assert_allclose(xp.to_numpy(device_factor.solve(xp.asarray(matrix))), np.eye(2), atol=1e-12)
            from struphy.feec.preconditioner import PolarMassMatrixPreconditioner

            _, derham, mass = make_polar_derham("IGAPolarCylinder")
            first = PolarMassMatrixPreconditioner(mass.M1).solver
            x = fill(first.domain.zeros(), seed=91)
            expected = first.dot(x)
            duplicate = kron.PolarKron2Dx1DSolver(first.domain, first.groups)
            assert duplicate._lu_3[0] is first._lu_3[0]
            assert duplicate._pol_solvers[0] is first._pol_solvers[0]
            with xp.profiling.count_transfers() as transfers:
                actual = duplicate.dot(x)
            assert transfers.total == 0, transfers.report()
            for a, b in zip(to_host(actual), to_host(expected)):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    finally:
        if fake:
            set_device_sparse_module(previous)


@pytest.mark.skipif(_cupy_installed(), reason="the fake cannot replace installed CuPy")
def test_device_factors_are_shared_fake_cupy():
    code = "from struphy.polar.tests.test_factorization_cache import check_device_factor_reuse; check_device_factor_reuse(True)"
    result = subprocess.run(
        [sys.executable, "-c", code], env=serial_child_env(CUNUMPY_FAKE_CUPY="1"), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.skipif(not xp.cupy_available(), reason="requires a GPU")
def test_device_factors_are_shared_on_cupy():
    check_device_factor_reuse()
