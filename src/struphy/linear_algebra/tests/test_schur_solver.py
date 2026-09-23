from time import perf_counter

import pytest
from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.linalg.stencil import StencilMatrix, StencilVectorSpace

from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters


def make_diagonal_problem(n=5):
    domain = DomainDecomposition([n - 1], [False])
    cart = CartDecomposition(domain, [n], [[0]], [[n - 1]], [1], [1])
    space = StencilVectorSpace(cart)

    def diagonal(value):
        matrix = StencilMatrix(space, space)
        matrix[:, 0] = value
        return matrix

    return space, diagonal


def test_schur_solver_reuses_fixed_step_and_rebuilds_after_changes(monkeypatch):
    space, diagonal = make_diagonal_problem()
    solver = SchurSolver(diagonal(4.0), diagonal(-1.0), "pcg", solver_params=SolverParameters(tol=1e-12))
    xn = space.zeros()
    xn[:] = 1.0
    byn = space.zeros()
    schur_matrix = solver._solver.linop
    rebuilds = 0
    original_imul = StencilMatrix.__imul__

    def count_rebuilds(matrix, value):
        nonlocal rebuilds
        if matrix is schur_matrix:
            rebuilds += 1
        return original_imul(matrix, value)

    monkeypatch.setattr(StencilMatrix, "__imul__", count_rebuilds)

    def solve(dt, a, bc):
        result, info = solver(xn, byn, dt)
        assert info["success"]
        assert result.toarray() == pytest.approx((a + dt**2 * bc) / (a - dt**2 * bc))
        assert solver._solver.linop is schur_matrix

    solve(0.5, 4.0, -1.0)
    first_rebuilds = rebuilds
    solve(0.5, 4.0, -1.0)
    assert rebuilds == first_rebuilds
    solve(1.0, 4.0, -1.0)
    assert rebuilds > first_rebuilds

    solver.BC = diagonal(-2.0)
    solve(1.0, 4.0, -2.0)

    solver.A = diagonal(6.0)
    solve(1.0, 6.0, -2.0)


def test_schur_solver_cached_step_benchmark():
    """Report setup and full-solve savings without a flaky timing assertion."""
    space, diagonal = make_diagonal_problem(n=1024)
    solver = SchurSolver(diagonal(4.0), diagonal(-1.0), "pcg", solver_params=SolverParameters(tol=1e-12))
    xn = space.zeros()
    byn = space.zeros()
    out = space.zeros()
    steps = 200

    def time_calls(call, dts):
        start = perf_counter()
        for dt in dts:
            call(dt)
        return perf_counter() - start

    fixed_dts = [0.5] * steps
    varying_dts = [0.5, 0.6] * (steps // 2)
    solver._update_operators(0.5)
    cached_setup = time_calls(solver._update_operators, fixed_dts)
    rebuilt_setup = time_calls(solver._update_operators, varying_dts)

    def solve(dt):
        _, info = solver(xn, byn, dt, out=out)
        assert info["success"]

    # The full solve also includes matrix-vector products and PCG work.
    solve(0.5)
    cached_solve = time_calls(solve, fixed_dts)
    rebuilt_solve = time_calls(solve, varying_dts)

    print(
        f"Schur, {steps} steps: setup cached={cached_setup / steps * 1e6:.2f}us/step, "
        f"rebuilt={rebuilt_setup / steps * 1e6:.2f}us/step; "
        f"full solve cached={cached_solve / steps * 1e3:.2f}ms/step, "
        f"rebuilt={rebuilt_solve / steps * 1e3:.2f}ms/step"
    )

if __name__ == "__main__":
    test_schur_solver_cached_step_benchmark()
