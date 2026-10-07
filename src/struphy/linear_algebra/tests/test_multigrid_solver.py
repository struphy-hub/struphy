import numpy as np
import pytest
from feectools.linalg.basic import LinearOperator
from feectools.linalg.solvers import inverse
from maybempi import MPI

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.feec.utilities import create_equal_random_arrays
from struphy.geometry.domains import Cuboid
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.multigrid.preconditioner import (
    MultiGridOptions,
    MultiGridPreconditioner,
    _assemble_dense,
)
from struphy.linear_algebra.multigrid.smoothers import (
    ChebyshevSmoother,
    DiagonalComputer,
    JacobiSmoother,
    KrylovSmoother,
    inverse_diagonal,
)
from struphy.topology.grids import TensorProductGrid

DIRICHLET = (("dirichlet", "dirichlet"), ("dirichlet", "dirichlet"), None)
PERIODIC = (None, None, None)


def _poisson(n, p, bcs, sigma=0.0):
    domain = Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=2.0, l3=0.0, r3=1.0)
    derham = Derham(
        TensorProductGrid(num_elements=(n, n, 1)),
        DerhamOptions(degree=(p, p, 1), bcs=bcs),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    mass_ops = WeightedMassOperators(derham, domain)
    A = derham.grad.T @ mass_ops.M1 @ derham.grad
    if sigma != 0.0:
        A = sigma * mass_ops.M0 + A
    return derham, domain, mass_ops, A


class _SmootherAsOperator(LinearOperator):
    """x = S b (one smoother call from zero initial guess)."""

    def __init__(self, S):
        self._S = S

    domain = property(lambda self: self._S.A.domain)
    codomain = property(lambda self: self._S.A.domain)
    dtype = property(lambda self: float)

    def transpose(self, conjugate=False):
        return self

    def dot(self, b, out=None):
        x = self.domain.zeros()
        self._S.smooth(b, x)
        if out is None:
            return x
        x.copy(out=out)
        return out


@pytest.mark.mpi_skip
@pytest.mark.parametrize("bcs", [DIRICHLET, PERIODIC])
def test_diagonal(bcs):
    """Probed diagonal equals the diagonal of the assembled operator."""
    derham, _, mass_ops, A = _poisson(8, 3, bcs, sigma=0.3)
    d = DiagonalComputer(derham.degree)(A)
    assert np.allclose(d.toarray(), np.diag(_assemble_dense(A)), atol=1e-14)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("kind", ["chebyshev_jacobi", "chebyshev_mass", "jacobi"])
def test_smoother_symmetric(kind):
    """The linear smoothers are symmetric (as matrices from right-hand side to iterate)."""
    from struphy.feec.preconditioner import MassMatrixPreconditioner

    derham, _, mass_ops, A = _poisson(8, 2, PERIODIC, sigma=0.5)
    D_inv = inverse_diagonal(DiagonalComputer(derham.degree)(A))
    if kind == "chebyshev_jacobi":
        S = ChebyshevSmoother(A, D_inv, degree=3)
    elif kind == "chebyshev_mass":
        S = ChebyshevSmoother(A, MassMatrixPreconditioner(mass_ops.M0), degree=3)
    else:
        S = JacobiSmoother(A, D_inv, sweeps=3)
    Sd = _assemble_dense(_SmootherAsOperator(S))
    assert np.abs(Sd - Sd.T).max() < 1e-12 * np.abs(Sd).max()


@pytest.mark.mpi_skip
@pytest.mark.parametrize("iterations", [1, 2, 4])
def test_krylov_smoother(iterations):
    """KrylovSmoother performs exactly ``iterations`` CG steps and is safe for a zero residual."""
    derham, _, mass_ops, A = _poisson(8, 2, PERIODIC, sigma=0.5)
    _, b = create_equal_random_arrays(derham.fem_spaces["0"], seed=3)

    # zero right-hand side with zero initial guess: x stays zero (no NaN)
    x = A.domain.zeros()
    KrylovSmoother(A, iterations=iterations).smooth(A.domain.zeros(), x)
    assert np.all(x.toarray() == 0.0)

    # the k-th CG iterate minimizes the A-norm error over the k-th Krylov space, so the error
    # decreases strictly with the number of iterations; compare with one call of k - 1 iterations
    x_ref = A.domain.zeros()
    if iterations > 1:
        KrylovSmoother(A, iterations=iterations - 1).smooth(b, x_ref)
    x = A.domain.zeros()
    KrylovSmoother(A, iterations=iterations).smooth(b, x)

    Ad = _assemble_dense(A)
    x_ex = np.linalg.solve(Ad, b.toarray())

    def err(y):
        e = y.toarray() - x_ex
        return e @ Ad @ e

    assert err(x) < err(x_ref)

    # one step from zero equals the steepest descent step alpha * b
    if iterations == 1:
        alpha = b.inner(b) / b.inner(A.dot(b))
        assert np.allclose(x.toarray(), alpha * b.toarray(), rtol=1e-12, atol=1e-14)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("bcs, nullspace", [(DIRICHLET, None), (PERIODIC, "constants")])
@pytest.mark.parametrize("smoother_precond", ["mass", "jacobi"])
def test_vcycle_spd(bcs, nullspace, smoother_precond):
    """The V-cycle is symmetric positive definite (on the complement of the null space) and contracts."""
    derham, domain, mass_ops, A = _poisson(16, 2, bcs)
    pc = MultiGridPreconditioner(
        A, derham, domain, MultiGridOptions(smoother_precond=smoother_precond, nullspace=nullspace), mass_ops=mass_ops
    )
    B = _assemble_dense(pc)
    Ad = _assemble_dense(A)
    assert np.abs(B - B.T).max() < 1e-12 * np.abs(B).max()

    # restrict to the dofs/modes that matter: interior dofs (Dirichlet) or zero-mean vectors (periodic)
    N = Ad.shape[0]
    if nullspace == "constants":
        Q = np.linalg.qr(np.eye(N) - np.ones((N, N)) / N)[0][:, : N - 1]
    else:
        Q = np.eye(N)[:, np.flatnonzero(np.diag(Ad) != 0.0)]
    assert np.linalg.eigvalsh(Q.T @ B @ Q).min() > 0.0
    E = Q.T @ (np.eye(N) - B @ Ad) @ Q
    assert np.abs(np.linalg.eigvals(E)).max() < 0.5


@pytest.mark.parametrize("bcs, nullspace", [(DIRICHLET, None), (PERIODIC, "constants")])
@pytest.mark.parametrize("p", [2, 3])
@pytest.mark.parametrize(
    "smoother, smoother_precond",
    [("chebyshev", "mass"), ("chebyshev", "jacobi"), ("jacobi", "identity"), ("cg", "mass")],
)
def test_poisson_h_independent(bcs, nullspace, p, smoother, smoother_precond):
    """MG-preconditioned CG converges in a small, mesh-independent number of iterations."""
    niter = []
    for n in (16, 32):
        derham, domain, mass_ops, A = _poisson(n, p, bcs)
        _, u = create_equal_random_arrays(derham.fem_spaces["0"], seed=3)
        b = A.dot(u)
        opts = MultiGridOptions(smoother=smoother, smoother_precond=smoother_precond, nullspace=nullspace)
        pc = MultiGridPreconditioner(A, derham, domain, opts, mass_ops=mass_ops)
        tol = 1e-8 * np.sqrt(b.inner(b))
        solver = inverse(A, "pcg", pc=pc, tol=tol, maxiter=100, recycle=False)
        x = solver.dot(b)
        r = b - A.dot(x)
        assert solver.get_info()["success"]
        assert np.sqrt(r.inner(r)) <= tol
        niter.append(solver.get_info()["niter"])
    assert max(niter) <= 15
    assert niter[1] <= niter[0] + 2


def test_update():
    """Changing a scalar of the operator re-uses the coarse operators and still converges."""
    derham, domain, mass_ops, A = _poisson(16, 2, DIRICHLET, sigma=2.0)
    pc = MultiGridPreconditioner(A, derham, domain, mass_ops=mass_ops)
    coarse_M0 = pc.operators[1].addends[0].operator

    A2 = 100.0 * mass_ops.M0 + derham.grad.T @ mass_ops.M1 @ derham.grad
    pc.update(A2)
    assert pc.operators[1].addends[0].operator is coarse_M0

    _, u = create_equal_random_arrays(derham.fem_spaces["0"], seed=4)
    b = A2.dot(u)
    tol = 1e-10 * np.sqrt(b.inner(b))
    solver = inverse(A2, "pcg", pc=pc, tol=tol, maxiter=100, recycle=False)
    x = solver.dot(b)
    r = b - A2.dot(x)
    assert np.sqrt(r.inner(r)) <= tol
    assert solver.get_info()["niter"] <= 15


@pytest.mark.mpi_skip
def test_options():
    opts = MultiGridOptions(smoother="jacobi", max_levels=3)
    assert MultiGridOptions.from_dict(opts.to_dict()) == opts
    with pytest.raises(AssertionError):
        MultiGridOptions(smoother="gauss-seidel")
    with pytest.raises(AssertionError):
        MultiGridOptions(n_pre=0, n_post=0)
