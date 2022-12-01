import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockMatrix
from psydac.linalg.kron import KroneckerStencilMatrix
from psydac.feec.global_projectors import GlobalProjector
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api.linear_operators import CompositeLinearOperator, BoundaryOperator
from struphy.psydac_api.preconditioner import ProjectorPreconditioner
from struphy.polar.linear_operators import PolarExtractionOperator
from struphy.linear_algebra.iterative_solvers import pbicgstab


class Projector:
    """
    Wrapper class for a commuting projector in 3d de Rham diagram.
    
    Parameters
    ----------
        projector_tensor : GlobalProjector
            The pure tensor product projector.
            
        dofs_extraction_op : PolarExtractionOperator
            The degree of freedom extraction operator mapping tensor product dofs to polar dofs.
        
        base_extraction_op : PolarExtractionOperator
            The basis extraction operator mapping tensor product basis functions to polar basis functions.
            
        boundary_op : BoundaryOperator
            The boundary operator setting homogeneous Dirichlet boundary conditions.
    """
    
    def __init__(self, projector_tensor, dofs_extraction_op=None, base_extraction_op=None, boundary_op=None):
        
        assert isinstance(projector_tensor, GlobalProjector)
        
        self._projector_tensor = projector_tensor
        
        if dofs_extraction_op is not None:
            assert isinstance(dofs_extraction_op, PolarExtractionOperator)
            
        if base_extraction_op is not None:
            assert isinstance(base_extraction_op, PolarExtractionOperator)
            
        self._dofs_extraction_op = dofs_extraction_op
        self._base_extraction_op = base_extraction_op
        
        if boundary_op is not None:
            assert isinstance(boundary_op, BoundaryOperator)
        
        self._boundary_op = boundary_op
        
        # set symbolic name of continuous space
        if hasattr(projector_tensor.space.symbolic_space, 'name'):
            self._space_symbolic_name = projector_tensor.space.symbolic_space.name
        else:
            self._space_symbolic_name = 'H1vec'
            
        # build inter-/histopolation matrix on full tensor-product space (only needed in polar case, is set to None otherwise)
        if self.dofs_extraction_op is not None:
            
            if isinstance(projector_tensor.imat_kronecker, KroneckerStencilMatrix):
                self._imat_stencil = projector_tensor.imat_kronecker.tostencil()
            else:
                
                blocks = [[projector_tensor.imat_kronecker.blocks[0][0].tostencil(), None, None],
                          [None, projector_tensor.imat_kronecker.blocks[1][1].tostencil(), None],
                          [None, None, projector_tensor.imat_kronecker.blocks[2][2].tostencil()]]
                
                self._imat_stencil = BlockMatrix(self.space.vector_space, self.space.vector_space, blocks)

            # set backend
            self._imat_stencil.set_backend(PSYDAC_BACKEND_GPYCCEL)

            # build inter-histopolation matrix P * I * E^T as CompositeLinearOperator
            self._I = CompositeLinearOperator(self._dofs_extraction_op, self._imat_stencil, self._base_extraction_op.transpose())
            
            # transposed inter-histopolation matrix E * I^T * P^T as CompositeLinearOperator
            self._IT = self._I.transpose()
            
            # preconditioner P * I^(-1) * E^T, resp. P * I^(-T) * E^T for iterative polar projections
            self._pc  = ProjectorPreconditioner(self, transposed=False)
            self._pcT = ProjectorPreconditioner(self, transposed=True)
            
        else:
            self._imat_stencil = None
            
            self._I  = None
            self._IT = None
        
            self._pc  = None
            self._pcT = None
            
    @property
    def projector_tensor(self):
        """ Tensor product projector.
        """
        return self._projector_tensor
    
    @property
    def space(self):
        """ Tensor product FEM space corresponding to projector.
        """
        return self._projector_tensor.space

    @property 
    def dofs_extraction_op(self):
        """ Degrees of freedom extraction operator (tensor product DOFs --> polar DOFs).
        """
        return self._dofs_extraction_op

    @property
    def base_extraction_op(self):
        """ Basis functions extraction operator (tensor product basis functions --> polar basis functions).
        """
        return self._base_extraction_op
    
    @property
    def boundary_op(self):
        """ Boundary operator seeting essential boundary conditions to array.
        """
        return self._boundary_op

    @property
    def I(self):
        """ Inter-/histopolation matrix P * I * E^T as CompositeLinearOperator.
        """
        return self._I

    @property
    def IT(self):
        """ Transposed inter-/histopolation matrix E * I^T * P^T as CompositeLinearOperator.
        """
        return self._IT
    
    @property
    def pc(self):
        """ Preconditioner P * I^(-1) * E^T for iterative polar projections.
        """
        return self._pc
    
    @property
    def pcT(self):
        """ Transposed preconditioner P * I^(-T) * E^T for iterative polar projections.
        """
        return self._pcT

    def solve(self, rhs, transposed=False, apply_bc=False, tol=1e-14, maxiter=1000, verbose=False):
        """
        Solves the linear system I * x = rhs, resp. I^T * x = rhs for x, where I is the inter-/histopolation matrix.
        
        Parameters
        ----------
            rhs : StencilVector | BlockVector | PolarVector
                The right-hand side of the linear system.
                
            transposed : bool
                Whether to invert the transposed inter-/histopolation matrix.
                
            apply_bc : bool
                Whether to apply boundary operator to input (rhs) and output (x).
                
            tol : float
                Stop tolerance in iterative solve (only used in polar case).
                
            maxiter : int
                Maximum number of iterations in iterative solve (only used in polar case).
                
            verbose : bool
                Whether to print some information in each iteration in iterative solve (only used in polar case).
                
        Returns
        -------
            x : StencilVector | BlockVector | PolarVector
                Output vector (result of linear system).
        """
        
        # apply boundary operator
        if apply_bc and self.boundary_op is not None:
            rhs = self.boundary_op.dot(rhs)
            
        if transposed:
            
            # polar case (iterative solve with pbicgstab)
            if self.dofs_extraction_op is not None:
                x = pbicgstab(self.IT, rhs, self.pcT, tol=tol, maxiter=maxiter, verbose=verbose)[0]

            # standard (tensor product) case (Kronecker solver)
            else:
                x = self.projector_tensor.solver.solve(rhs, transposed=True)
            
        else:

            # polar case (iterative solve with pbicgstab)
            if self.dofs_extraction_op is not None:
                x = pbicgstab(self.I, rhs, self.pc, tol=tol, maxiter=maxiter, verbose=verbose)[0]

            # standard (tensor product) case (Kronecker solver)
            else:
                x = self.projector_tensor.solver.solve(rhs, transposed=False)

        # apply boundary operator
        if apply_bc and self.boundary_op is not None:
            x = self.boundary_op.dot(x)

        return x

    def get_dofs(self, fun, apply_bc=False):
        """
        Computes the geometric degrees of freedom associated to given callable(s).
        
        Parameters
        ----------
            fun : callable | list
                The function for which the geometric degrees of freedom shall be computed. List of callables for vector-valued functions.
            apply_bc : bool
                Whether to apply homogeneous Dirichlet boundary conditions to degrees of freedom.
                
        Returns
        -------
            dofs : StencilVector | BlockVector | PolarVector
                The geometric degrees of freedom associated to given callable(s).
        """

        # get dofs on tensor-product grid
        dofs = self.projector_tensor(fun, dofs_only=True)

        # apply DOF extraction operator
        if self.dofs_extraction_op is not None:
            dofs = self.dofs_extraction_op.dot(dofs)

        # apply boundary operator
        if apply_bc and self.boundary_op is not None:
            dofs = self.boundary_op.dot(dofs)

        return dofs

    def __call__(self, fun, apply_bc=False, tol=1e-14, maxiter=1000, verbose=False):
        """
        Applies projector to given callable(s).
        
        Parameters
        ----------
            fun : callable | list
                The function for which the geometric degrees of freedom shall be computed. List of callables for vector-valued functions.
            apply_bc : bool
                Whether to apply homogeneous Dirichlet boundary conditions to degrees of freedom.
                
        Returns
        -------
            coeffs : StencilVector | BlockVector | PolarVector
                The spline coefficients of the FEM field after projection.
        """
        return self.solve(self.get_dofs(fun, apply_bc), transposed=False, apply_bc=apply_bc, tol=tol, maxiter=maxiter, verbose=verbose)
    
    
def evaluate_fun_weights_1d(pts, wts, fun):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version of the function loops over all elements and is fixed to dimension 1.

    Parameters
    ----------
        pts : 1-tuple of 2d float arrays
            Quadrature points in each dimension in format (element, quadrature point).

        wts : 1-tuple of 2d float arrays
            Quadrature weights in each dimension in format (element, quadrature point).

        fun : callable
            The function which shall be evaluated at eta1.
    
    Returns
    -------
        values : ndarray[float]
            A 2d array (1 cell grid dimension, 1 quadrature point dimension) which contains all the pre-evaluated values.
    """

    # will not be pyccelized, due to dependence on func (or could we call back to Python?)
    values = np.zeros((pts[0].shape[0], pts[0].shape[1]), dtype=float)
                      
    for i in range(pts[0].shape[0]): # element index
        for iq in range(pts[0].shape[1]): # quadrature point index
            values[i, iq] = fun(pts[0][i, iq]) *  wts[0][i, iq]
            
    return values


def evaluate_fun_weights_2d(pts, wts, fun):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version of the function loops over all elements and is fixed to dimension 2.

    Parameters
    ----------
        pts : 2-tuple of 2d float arrays
            Quadrature points in each dimension in format (element, quadrature point).

        wts : 2-tuple of 2d float arrays
            Quadrature weights in each dimension in format (element, quadrature point).

        fun : callable
            The function which shall be evaluated at eta1, eta2.
    
    Returns
    -------
        values : ndarray[float]
            A 4d array (2 cell grid dimensions, 2 quadrature point dimensions) which contains all the pre-evaluated values.
    """

    # will not be pyccelized, due to dependence on func (or could we call back to Python?)
    values = np.zeros((pts[0].shape[0], pts[1].shape[0], 
                       pts[0].shape[1], pts[1].shape[1]), dtype=float)
    
    for i in range(pts[0].shape[0]): # element index
        for j in range(pts[1].shape[0]):
            for iq in range(pts[0].shape[1]): # quadrature point index
                for jq in range(pts[1].shape[1]):
                        funval = fun(pts[0][i, iq], pts[1][j, jq])
                        weightval = wts[0][i, iq] * wts[1][j, jq]
                        values[i, j, iq, jq] = weightval * funval
                            
    return values


def evaluate_fun_weights_3d(pts, wts, fun):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version of the function loops over all elements and is fixed to dimension 3.

    Parameters
    ----------
        pts : 3-tuple of 2d float arrays
            Quadrature points in each dimension in format (element, quadrature point).

        wts : 3-tuple of 2d float arrays
            Quadrature weights in each dimension in format (element, quadrature point).

        fun : callable
            The function which shall be evaluated at eta1, eta2, eta3.
    
    Returns
    -------
        values : ndarray[float]
            A 6d array (3 cell grid dimensions, 3 quadrature point dimensions) which contains all the pre-evaluated values.
    """

    # will not be pyccelized, due to dependence on func (or could we call back to Python?)
    values = np.zeros((pts[0].shape[0], pts[1].shape[0], pts[2].shape[0], 
                       pts[0].shape[1], pts[1].shape[1], pts[2].shape[1]), dtype=float)
    
    for i in range(pts[0].shape[0]): # element index
        for j in range(pts[1].shape[0]):
            for k in range(pts[2].shape[0]):
                for iq in range(pts[0].shape[1]): # quadrature point index
                    for jq in range(pts[1].shape[1]):
                        for kq in range(pts[2].shape[1]):
                            funval = fun(pts[0][i, iq], pts[1][j, jq], pts[2][k, kq])
                            weightval = wts[0][i, iq] * wts[1][j, jq] * wts[2][k, kq]
                            values[i, j, k, iq, jq, kq] = weightval * funval
                            
    return values


def assemble_funccache_numpy(u, w, func):
    """
    Pre-evaluates the given function at the quadrature points,
    and multiplies the result with the quadrature weights of this point.
    Quadrature weights and coordinates are given in a tensor-product format.

    This version tries to use numpy where possible, and is usable in arbitrary dimensions.

    Parameters
    ----------
        u : three-tuple of two-dimensional numpy arrays
            The quadrature points in each dimension.

        w : three-tuple of two-dimensional numpy arrays
            The quadrature weights in each dimension for the respective points.

        func : callable, with three parameters
            The function which shall be evaluated.
    
    Returns
    -------
        values : ndarray[float]
            A 6d array (3 cell grid dimensions, 3 quadrature point dimensions) which contains all the pre-evaluated values.
    """

    import numpy as np

    funcvec = np.vectorize(func)
    grid = np.meshgrid(*u, sparse=True, indexing='ij')
    funceval = funcvec(*grid)

    for wg in np.meshgrid(*w, sparse=True, indexing='ij'):
        funceval *= wg
    
    funceval.shape = tuple(uxx for ux in u for uxx in ux.shape)

    n = len(u)
    return funceval.transpose([2*i for i in range(n)] + [2*i+1 for i in range(n)])
