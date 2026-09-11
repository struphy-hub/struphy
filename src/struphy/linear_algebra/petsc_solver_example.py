"""Example: solve a struphy mass-matrix system with :class:`~struphy.linear_algebra.petsc_solver.PETScSolver`.

Builds the ``H1`` mass matrix ``M0`` of a small 3D Derham complex on a cuboid domain,
manufactures a right-hand side from a known exact solution, and solves ``M0 x = b``
with a PETSc KSP (CG + Jacobi preconditioner), comparing against feectools' native
preconditioned CG solver.

Run with:

.. code-block:: bash

    python3 -m struphy.linear_algebra.petsc_solver_example
"""

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.fields_background.equils import HomogenSlab
from struphy.geometry.domains import Cuboid
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.petsc_solver import PETScSolver
from struphy.topology.grids import TensorProductGrid


def main():
    comm = MPI.COMM_WORLD

    # domain, equilibrium and Derham complex
    domain = Cuboid()
    equil = HomogenSlab(n0=2.0)
    equil.domain = domain

    grid = TensorProductGrid(num_elements=[8, 8, 8])
    derham_opts = DerhamOptions(degree=[2, 2, 2])
    derham = Derham(grid, derham_opts, comm=comm, domain=domain)

    # weighted mass operators -- M0.matrix is the assembled StencilMatrix on the H1 space
    mass_ops = WeightedMassOperators(derham, domain, eq_mhd=equil)
    M0 = mass_ops.M0.matrix

    # manufacture a right-hand side from a known exact solution
    xe = M0.domain.zeros()
    xe[:] = xp.random.random(xe[:].shape)
    xe.update_ghost_regions()
    b = M0.dot(xe)

    # reference solve with feectools' preconditioned CG
    pc = M0.diagonal(inverse=True)
    cg_solver = inverse(M0, "pcg", pc=pc, tol=1e-12, maxiter=2000, verbose=False, recycle=False)
    x_cg = cg_solver.solve(b)

    # solve the same system with PETSc's CG + Jacobi preconditioner
    petsc_solver = PETScSolver(M0, tol=1e-12, maxiter=2000, ksp_type="cg", pc_type="jacobi")
    x_petsc = petsc_solver.solve(b)

    error_vs_exact = xp.linalg.norm((x_petsc - xe).toarray())
    error_vs_cg = xp.linalg.norm((x_petsc - x_cg).toarray())

    if comm.Get_rank() == 0:
        print(f"PETSc KSP info: {petsc_solver.get_info()}")
        print(f"||x_petsc - x_exact|| = {error_vs_exact:.3e}")
        print(f"||x_petsc - x_cg||    = {error_vs_cg:.3e}")


if __name__ == "__main__":
    main()
