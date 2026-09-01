"""Regression test for SchurSolverFull's solver="petsc" support.

Like SchurSolver (see test_petsc_schur_solver.py), SchurSolverFull/SchurSolverFull3 always
imported feectools' own `inverse` directly, which does not recognize "petsc" -- so
solver="petsc" would raise for the variational MHD propagators built on them
(VariationalPBEvolve, VariationalEntropyEvolve, VariationalMagFieldEvolve, VariationalQBEvolve).

Unlike SchurSolver, there is no in-place-mutation caching hazard here: `self._S` is built once
in __init__ and never mutated afterwards -- callers (e.g. VariationalQBEvolve) rebuild the whole
SchurSolverFull3 object fresh every Newton iteration rather than reusing one across calls (see
those propagators' "local version to avoid creating new version of LinearOperator every time"
comment, which refers to the *Jacobian's blocks*, not to reusing the Schur solver object itself).
So the fix here is the dispatch alone: use struphy.linear_algebra.solver.inverse (which knows
"petsc" and safely strips petsc-only kwargs for every other solver) instead of feectools' own.

This test exercises that dispatch directly on a small synthetic block system (matching
test_petsc_solver.py's style), not through a real variational MHD model: those models' Jacobian
blocks involve operator types (BasisProjectionOperator-derived, nonlinear-model-specific) that
have not been checked against _assemble_petsc_matrix's supported set, and building one from
scratch without an existing example/test to adapt was judged too failure-prone to do blind. If a
real variational-MHD case is wired up to use solver="petsc" later, verify it separately.
"""

import cunumpy as xp
import pytest

pytest.importorskip("petsc4py")

from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace

from struphy.linear_algebra.schur_solver import SchurSolverFull, SchurSolverFull3


def _make_space(n, p):
    domain_decomposition = DomainDecomposition([n - p], [False], comm=MPI.COMM_WORLD)
    cart = CartDecomposition(domain_decomposition, [n], [xp.array([0])], [xp.array([n - 1])], [p], [1])
    return StencilVectorSpace(cart)


def _spd_tridiagonal(V, p, scale):
    """Banded, symmetric positive-definite StencilMatrix with 2p+1 diagonals on space `V`."""
    A = StencilMatrix(V, V)
    A[:, -p:0] = -scale
    A[:, 0:1] = 2 * p * scale
    A[:, 1 : p + 1] = -scale
    A.remove_spurious_entries()
    return A


def test_schur_solver_full_petsc_matches_pcg():
    """SchurSolverFull(solver_name="petsc") must solve [[A B],[C Id]] to the same accuracy as
    solver_name="pcg", for a small synthetic system [[A B],[C Id]] x = v.
    """
    n, p = 12, 1
    xp.random.seed(0)

    V = _make_space(n, p)
    A = _spd_tridiagonal(V, p, scale=1.0)
    B = _spd_tridiagonal(V, p, scale=0.01)
    C = _spd_tridiagonal(V, p, scale=0.01)

    domain = BlockVectorSpace(V, V)
    M = BlockLinearOperator(domain, domain)
    M[0, 0] = A
    M[0, 1] = B
    M[1, 0] = C
    M[1, 1] = IdentityOperator(V)

    s = V.starts[0]
    e = V.ends[0]
    bx = StencilVector(V)
    bx[s : e + 1] = xp.random.random(e + 1 - s)
    by = StencilVector(V)
    by[s : e + 1] = xp.random.random(e + 1 - s)

    v = BlockVector(domain)
    v[0] = bx
    v[1] = by

    solver_kwargs = {"pc": None, "tol": 1e-13, "maxiter": 2000, "verbose": False, "recycle": False}
    solver_pcg = SchurSolverFull(M, "pcg", **solver_kwargs)
    solver_petsc = SchurSolverFull(M, "petsc", **solver_kwargs)

    x_pcg = solver_pcg.dot(v)
    x_petsc = solver_petsc.dot(v)

    err_x = xp.linalg.norm((x_pcg[0] - x_petsc[0]).toarray())
    err_y = xp.linalg.norm((x_pcg[1] - x_petsc[1]).toarray())
    assert err_x < 1e-8, f"x-block mismatch: {err_x:.2e}"
    assert err_y < 1e-8, f"y-block mismatch: {err_y:.2e}"


def test_schur_solver_full3_petsc_matches_pcg():
    """SchurSolverFull3(solver_name="petsc") must solve [[A B D],[C Id 0],[E 0 Id]] to the same
    accuracy as solver_name="pcg", for a small synthetic system.
    """
    n, p = 12, 1
    xp.random.seed(1)

    V = _make_space(n, p)
    A = _spd_tridiagonal(V, p, scale=1.0)
    B = _spd_tridiagonal(V, p, scale=0.01)
    C = _spd_tridiagonal(V, p, scale=0.01)
    D = _spd_tridiagonal(V, p, scale=0.01)
    E = _spd_tridiagonal(V, p, scale=0.01)

    domain = BlockVectorSpace(V, V, V)
    M = BlockLinearOperator(domain, domain)
    M[0, 0] = A
    M[0, 1] = B
    M[1, 0] = C
    M[1, 1] = IdentityOperator(V)
    M[0, 2] = D
    M[2, 0] = E
    M[2, 2] = IdentityOperator(V)

    s = V.starts[0]
    e = V.ends[0]
    bx = StencilVector(V)
    bx[s : e + 1] = xp.random.random(e + 1 - s)
    by = StencilVector(V)
    by[s : e + 1] = xp.random.random(e + 1 - s)
    bz = StencilVector(V)
    bz[s : e + 1] = xp.random.random(e + 1 - s)

    v = BlockVector(domain)
    v[0] = bx
    v[1] = by
    v[2] = bz

    solver_kwargs = {"pc": None, "tol": 1e-13, "maxiter": 2000, "verbose": False, "recycle": False}
    solver_pcg = SchurSolverFull3(M, "pcg", **solver_kwargs)
    solver_petsc = SchurSolverFull3(M, "petsc", **solver_kwargs)

    x_pcg = solver_pcg.dot(v)
    x_petsc = solver_petsc.dot(v)

    err_x = xp.linalg.norm((x_pcg[0] - x_petsc[0]).toarray())
    err_y = xp.linalg.norm((x_pcg[1] - x_petsc[1]).toarray())
    err_z = xp.linalg.norm((x_pcg[2] - x_petsc[2]).toarray())
    assert err_x < 1e-8, f"x-block mismatch: {err_x:.2e}"
    assert err_y < 1e-8, f"y-block mismatch: {err_y:.2e}"
    assert err_z < 1e-8, f"z-block mismatch: {err_z:.2e}"


if __name__ == "__main__":
    test_schur_solver_full_petsc_matches_pcg()
    test_schur_solver_full3_petsc_matches_pcg()
