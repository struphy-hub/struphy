from psydac.linalg.basic import Vector, LinearOperator, IdentityOperator
from psydac.linalg.block import BlockVector, BlockLinearOperator
from psydac.linalg.solvers import inverse
from psydac.linalg.utilities import petsc_to_psydac
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
from math                  import sqrt
from psydac.linalg.stencil import StencilVectorSpace, StencilVector
from psydac.linalg.block   import BlockVectorSpace

from struphy.profiling.profiling import (
        ProfileRegion,
    )
def print_solver_info(ksp):
    #print(f'x_petsc = {x_petsc.getArray()}')
    if PETSc.COMM_WORLD.rank == 0:
        #print('#' + '-'*78 + '#')
        
        # Get and print the number of iterations
        iter_count = ksp.getIterationNumber()
        print(f'Number of iterations: {iter_count}')

        # Get and print the solver method used
        solver_type = ksp.getType()
        print(f'Solver type: {solver_type}')

        # Optional: get solver options and parameters
        ksp_options = ksp.getOptionsPrefix()
        print(f'Solver options prefix: {ksp_options}')
        
        # Print the final residual norm
        residual = ksp.getResidualNorm()
        print(f'Residual norm: {residual}')

        # Retrieve the solution norm
        # solution_norm = x_petsc.norm()
        # print(f'Solution norm: {solution_norm}')
        print('#' + '-'*78 + '#')

def compare_petsc_vecs(vec1, vec2, show_vecs = False):
    # Ensure both vectors have the same size
    assert vec1.getSize() == vec2.getSize(), "Vectors must have the same size."
    
    # Create a new vector to store the difference
    diff_vec = vec1.duplicate()
    
    # Call the custom waxpy method: diff_vec = vec2 - vec1
    diff_vec.waxpy(-1.0, vec1, vec2)

    # Compute the norm of the difference vector
    norm_diff = diff_vec.norm()
    
    # Check if the norm is below the tolerance
    if show_vecs:

        print('vec1:')
        print(vec1.view())

        print('vec2:')
        print(vec2.view())

        print('diff_vec:')
        print(diff_vec.view())

    print(f"{norm_diff = }")

class SchurSolver:
    '''Solves for :math:`x^{n+1}` in the block system

    .. math::

        \left( \matrix{
            A & \Delta t B \cr
            \Delta t C & \\text{Id}
        } \\right)
        \left( \matrix{
            x^{n+1} \cr y^{n+1}
        } \\right)
        =
        \left( \matrix{
            A & - \Delta t B \cr
            - \Delta t C & \\text{Id}
        } \\right)
        \left( \matrix{
            x^n \cr y^n
        } \\right)

    using the Schur complement :math:`S = A - \Delta t^2 BC`, where Id is the identity matrix
    and :math:`(x^n, y^n)` is given. The solution is given by

    .. math::

        x^{n+1} = S^{-1} \left[ (A + \Delta t^2 BC) \, x^n - 2 \Delta t B \, y^n \\right] \,.

    Parameters
    ----------
    A : LinearOperator
        Upper left block from [[A B], [C Id]].

    BC : LinearOperator
        Product from [[A B], [C Id]].

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    '''

    def __init__(self, 
                 A: LinearOperator, 
                 BC: LinearOperator,
                 solver_name: str,
                 petsc: bool,
                 **solver_params):

        assert isinstance(A, LinearOperator)
        assert isinstance(BC, LinearOperator)

        assert A.domain == BC.domain
        assert A.codomain == BC.codomain

        # linear operators
        self._A = A
        self._BC = BC

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params['pc'] is None:
            solver_params.pop('pc')
        
        print(f"{solver_params = }")
        self.petsc = petsc
        

        self._solver = inverse(A, solver_name, **solver_params)

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = A.codomain.zeros()


        if self.petsc:
            with ProfileRegion("petsc_matrix_setup"):
                # -------------------------------------------------------------------#
                # PETSc setup
                # print('self.A.matrix.shape()',self.A.matrix.shape)
                print("Setting up petsc matrix")
                #A_petsc = self.A.matrix.topetsc()
                # ---------------------------------#
                # Converting via numpy array
                # ---------------------------------#
                # numpy_matrix = np.array([
                #     [18.,         0.,           0.,          0.,          0.,          0.,         0.,          0.        ],
                #     [ 0.,         18.,          0.,          0.,          0.,          0.,         0.,          0.        ],
                #     [ 0.,          0.,          0.66666667,  0.33333333,  0.,          0.,         0.,          0.        ],
                #     [ 0.,          0.,          0.33333333,  1.33333333,  0.33333333,  0.,         0.,          0.        ],
                #     [ 0.,          0.,          0.,          0.33333333,  0.66666667,  0.,         0.,          0.        ],
                #     [ 0.,          0.,          0.,          0.,          0.,          0.16666667, 0.08333333,  0.        ],
                #     [ 0.,          0.,          0.,          0.,          0.,          0.08333333, 0.33333333,  0.08333333],
                #     [ 0.,          0.,          0.,          0.,          0.,          0.,         0.08333333,  0.16666667]])        
                
                # numpy_matrix = self.A.matrix.toarray()
                # # Get the dimensions of the NumPy matrix
                # rows, cols = numpy_matrix.shape
                # # Create a PETSc matrix with the same dimensions
                # petsc_mat = PETSc.Mat().create(MPI.COMM_WORLD)
                # petsc_mat.setSizes([rows, cols])
                # petsc_mat.setFromOptions()
                # petsc_mat.setUp()
                # # Set the values of the PETSc matrix from the NumPy matrix
                # for i in range(rows):
                #     for j in range(cols):
                #         petsc_mat.setValue(i, j, numpy_matrix[i, j])
                # # Assemble the PETSc matrix
                # petsc_mat.assemble()
                # # print('full mat to PETSc:')
                # # print(petsc_mat.view())
                # ---------------------------------#

                
                # ---------------------------------#
                # Convert via sparse matrix
                # ---------------------------------#
                sparse_matrix = self.A.matrix.tosparse()
                # print(sparse_matrix)
                
                petsc_mat_sparse = PETSc.Mat().create(comm=MPI.COMM_WORLD)
                petsc_mat_sparse.setSizes(sparse_matrix.shape)
                petsc_mat_sparse.setType('aij')
                petsc_mat_sparse.setFromOptions()
                petsc_mat_sparse.setUp()

                # Extract data from scipy sparse matrix
                # rows = sparse_matrix.row
                # cols = sparse_matrix.col
                # data = sparse_matrix.data

                # Set values to PETSc matrix
                for row,col,dat in zip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data):
                    petsc_mat_sparse.setValue(row, col, dat, PETSc.InsertMode.ADD_VALUES)
                    # petsc_mat_sparse.setValue(rows[i], cols[i], data[i], PETSc.InsertMode.ADD_VALUES)
                petsc_mat_sparse.assemble()
                # print('sparse to PETSc:')
                # print(petsc_mat_sparse.view())
                # ---------------------------------#


                # ---------------------------------#
                # Setup solver
                # ---------------------------------#
                # Initialize ksp solver.
                self.ksp = PETSc.KSP().create(comm=MPI.COMM_WORLD)
                self.ksp.setOperators(petsc_mat_sparse)
                self.ksp.setInitialGuessNonzero(True)
                # self.ksp.setType(PETSc.KSP.Type.CG)
                self.ksp.setTolerances(rtol=1e-7, atol=1e-7, divtol=None, max_it=3000)        
                # self.ksp.setFromOptions()
                self.ksp.setUp()


                # Clean up PETSc objects when done
                petsc_mat_sparse.destroy()
                # ---------------------------------#
    @property
    def A(self):
        """ Upper left block from [[A B], [C Id]].
        """
        return self._A

    @property
    def BC(self):
        """ Product from [[A B], [C Id]].
        """
        return self._BC

    @A.setter
    def A(self, a):
        """ Upper left block from [[A B], [C Id]].
        """
        self._A = a

    @BC.setter
    def BC(self, bc):
        """ Product from [[A B], [C Id]].
        """
        self._BC = bc

    def __call__(self, xn, Byn, dt, out=None):
        """
        Solves the 2x2 block matrix linear system.

        Parameters
        ----------
        xn : psydac.linalg.basic.Vector
            Solution from previous time step.

        Byn : psydac.linalg.basic.Vector
            The product B*yn.

        dt : float
            Time step size.

        out : psydac.linalg.basic.Vector, optional
            If given, the converged solution will be written into this vector (in-place).

        Returns
        -------
        out : psydac.linalg.basic.Vector
            Converged solution.

        info : dict
            Convergence information.
        """

        assert isinstance(xn, Vector)
        assert isinstance(Byn, Vector)
        assert xn.space == self._A.domain
        assert Byn.space == self._A.codomain
        use_both_solvers = False

        # left- and right-hand side operators
        schur = self._A - dt**2 * self._BC
        rhs_m = self._A + dt**2 * self._BC
        
        
        # right-hand side vector rhs = 2*dt*[ rhs_m/(2*dt) @ xn - Byn ] (in-place!)
        rhs = rhs_m.dot(xn, out=self._rhs)
        rhs /= 2*dt
        rhs -= Byn
        rhs *= 2*dt




        if self.petsc or use_both_solvers:
            #print('Solving with PETSc')
            # ---------------------------------#
            # Solve with PETSc
            with ProfileRegion("psydac2petsc"):
                x_petsc = xn.topetsc()
                rhs_petsc = rhs.topetsc()
            with ProfileRegion("petsc_solver"):
                self.ksp.solve(rhs_petsc, x_petsc)
            
            # print_solver_info(self.ksp)
            with ProfileRegion("petsc2psydac"):
                x_petsc2psydac = petsc_to_psydac(x_petsc, xn.space)

        if (not self.petsc) or use_both_solvers:
            # ---------------------------------#
            # Solve with psydac
            # solve linear system (in-place if out is not None)
            with ProfileRegion("psydac_solver"):
                self._solver.linop = schur
                x = self._solver.dot(rhs, out=out)
            # if PETSc.COMM_WORLD.rank == 0:
            #     print(f'x_pdydac = {x.toarray()}')
            #     print(self._solver._info)
            # ---------------------------------#
        
        if use_both_solvers:
            compare_petsc_vecs(x_petsc, x.topetsc(), show_vecs = True)
            exit()
        
        if self.petsc:
            return x_petsc2psydac, self._solver._info
        else:
            return x, self._solver._info

class SchurSolverFull:
    '''Solves the block system

    .. math::

        \left( \matrix{
            A & B \cr
            C & \\text{Id}
        } \\right)
        \left( \matrix{
            x \cr y
        } \\right)
        =
        \left( \matrix{
            b_x \cr b_y
        } \\right)

    using the Schur complement :math:`S = A - BC`, where Id is the identity matrix
    and :math:`(b_x, b_y)^T` is given. The solution is given by

    .. math::

        x &= S^{-1} \, (b_x - B b_y ) \,,

        y &= b_y - C x \,.

    Parameters
    ----------
    M : BlockLinearOperator
        Matrix [[A B], [C Id]].

    solver_name : str
        See [psydac.linalg.solvers](https://github.com/pyccel/psydac/blob/535717c6f5ea328aacbbbbcc2d582a92b31c9377/psydac/linalg/solvers.py#L47) for possible names.

    **solver_params : 
        Must correspond to the chosen solver.
    '''

    def __init__(self, M, solver_name, **solver_params):

        assert isinstance(M, BlockLinearOperator)
        assert M.domain == M.codomain  # solve square system

        # initialize solver with dummy matrix A
        self._solver_name = solver_name

        if solver_params['pc'] is None:
            solver_params.pop('pc')

        self._M = M

        self._A = M[0, 0]
        self._B = M[0, 1]
        self._C = M[1, 0]
        assert isinstance(M[1, 1], IdentityOperator)

        self._S = self._A - self._B @ self._C

        self._solver = inverse(self._S, solver_name, **solver_params)

        # right-hand side vector (avoids temporary memory allocation!)
        self._rhs = self._A.codomain.zeros()

    def dot(self, v, out=None):
        """
        Solves the 2x2 block matrix linear system.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            Left hand side of the system.

        out : psydac.linalg.basic.Vector, optional
            If given, the converged solution will be written into this vector (in-place).

        Returns
        -------
        out : psydac.linalg.block.BLockVector
            Converged solution.

        info : dict
            Convergence information.
        """

        assert isinstance(v, BlockVector)
        assert v.space == self._M.domain

        if out is None:
            out = self._M.codomain.zeros()
        else:
            assert out.space == self._M.codomain

        bx = v[0]
        by = v[1]

        # right-hand side vector rhs bx - B by
        rhs = self._B.dot(by, out=self._rhs)
        rhs *= -1
        rhs += bx

        # solve linear system (in-place if out is not None)
        x = self._solver.dot(rhs, out=out[0])
        y = self._C.dot(x, out=out[1])
        y *= -1
        y += by

        return out
