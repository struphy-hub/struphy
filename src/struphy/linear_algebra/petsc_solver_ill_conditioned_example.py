"""Example: PETSc beats feectools' native solver on an ill-conditioned SPD system.

Mass-matrix solves (see ``petsc_solver_example.py``) are *not* where PETSc helps:
they are already well-conditioned and feectools' diagonal-preconditioned CG converges
in a couple of iterations, so per-solve overhead dominates and makes PETSc slower there.

Where PETSc *does* win is on badly-conditioned elliptic systems, where its algebraic
multigrid preconditioner (``pc_type="gamg"``) keeps the iteration count roughly constant
while plain (or diagonally-preconditioned) CG needs an iteration count that grows like
``sqrt(condition number)``.

This example builds the standard 1D discrete Laplacian (tridiagonal, -1/2/-1), whose
condition number scales like ``O(n^2)`` in the number of unknowns ``n``, and compares:

- feectools' plain, unpreconditioned CG
- :class:`~struphy.linear_algebra.petsc_solver.PETScSolver` with ``ksp_type="cg", pc_type="gamg"``

For this to show a genuine win (not just fewer iterations but less wall time), the
per-solve vector conversion (:func:`feectools.linalg.topetsc.vec_topetsc` /
:func:`~feectools.linalg.utilities.petsc_to_psydac`) needs to be vectorized rather than
looping in pure Python over every DOF -- that conversion cost otherwise swamps any
iteration-count savings. There's a rough sweet spot: below a few thousand unknowns
GAMG's one-time setup cost dominates and plain CG wins; above it, PETSc's roughly
constant iteration count pulls ahead, increasingly so as ``n`` grows (and, as a bonus,
plain unpreconditioned CG's accuracy degrades from floating-point error accumulation
once it needs tens of thousands of iterations, while PETSc stays accurate).

Run with:

.. code-block:: bash

    python3 -m struphy.linear_algebra.petsc_solver_ill_conditioned_example
"""

import time

import cunumpy as xp
from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace

from struphy.linear_algebra.petsc_solver import PETScSolver


def build_1d_laplacian(n, comm):
    """Standard 1D discrete Laplacian (tridiagonal, -1, 2, -1); condition number ~ O(n^2)."""
    p = 1
    dd = DomainDecomposition([n - p], [False], comm=comm)
    cart = CartDecomposition(dd, [n], [xp.array([0])], [xp.array([n - 1])], [p], [1])
    V = StencilVectorSpace(cart)
    s = V.starts[0]
    e = V.ends[0]

    A = StencilMatrix(V, V)
    A[:, -1:0] = -1.0
    A[:, 0:1] = 2.0
    A[:, 1:2] = -1.0
    A.remove_spurious_entries()

    xe = StencilVector(V)
    xe[s : e + 1] = xp.random.random(e + 1 - s)

    return V, A, xe


def main(n=50_000, tol=1e-8, maxiter=200_000):
    comm = MPI.COMM_WORLD

    _, A, xe = build_1d_laplacian(n, comm)
    b = A @ xe

    t0 = time.perf_counter()
    cg_solver = inverse(A, "cg", tol=tol, maxiter=maxiter, verbose=False, recycle=False)
    x_cg = cg_solver.solve(b)
    t_cg = time.perf_counter() - t0

    t0 = time.perf_counter()
    petsc_solver = PETScSolver(A, tol=tol, maxiter=maxiter, ksp_type="cg", pc_type="gamg")
    x_petsc = petsc_solver.solve(b)
    t_petsc = time.perf_counter() - t0

    if comm.Get_rank() == 0:
        print(f"n={n} dofs (1D Laplacian, condition number ~ O(n^2))")
        print(f"  cg (unpreconditioned) : {t_cg * 1e3:9.2f} ms, niter={cg_solver.get_info()['niter']}")
        print(f"  petsc (cg + gamg)     : {t_petsc * 1e3:9.2f} ms, niter={petsc_solver.get_info()['niter']}")
        print(f"  speedup: {t_cg / t_petsc:.2f}x")
        print(f"  ||x_cg - x_exact||    = {xp.linalg.norm((x_cg - xe).toarray()):.3e}")
        print(f"  ||x_petsc - x_exact|| = {xp.linalg.norm((x_petsc - xe).toarray()):.3e}")


if __name__ == "__main__":
    main()
