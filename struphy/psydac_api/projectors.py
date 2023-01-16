import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockMatrix
from psydac.linalg.kron import KroneckerStencilMatrix
from psydac.feec.global_projectors import GlobalProjector
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api.linear_operators import CompositeLinearOperator, BoundaryOperator, IdentityOperator
from struphy.psydac_api.preconditioner import ProjectorPreconditioner
from struphy.polar.linear_operators import PolarExtractionOperator
from struphy.linear_algebra.iterative_solvers import pbicgstab


class Projector:
    """
    A commuting projector in a 3d (polar) de Rham diagram. The general structure of the inter-/histopolation problem reads
    
         (B * P * I * E^T * B^T) * coeffs = B * P * dofs,
    
    with the following linear operators:
    
        * B : boundary operator,
        * P : polar degrees of freedom extraction operator,
        * I : tensor product inter-/histopolation matrix,
        * E : polar basis extraction operator.
        
    P and E (and B in case of no boundary conditions) can be identity operators.
    
    Parameters
    ----------
        projector_tensor : GlobalProjector
            The pure tensor product projector.
            
        dofs_extraction_op : PolarExtractionOperator | IdentityOperator
            The degrees of freedom extraction operator mapping tensor product DOFs to polar DOFs.
        
        base_extraction_op : PolarExtractionOperator | IdentityOperator
            The basis extraction operator mapping tensor product basis functions to polar basis functions.
            
        boundary_op : BoundaryOperator | IdentityOperator
            The boundary operator applying essential boundary conditions to a vector.
    """
    
    def __init__(self, projector_tensor, dofs_extraction_op=None, base_extraction_op=None, boundary_op=None):
        
        assert isinstance(projector_tensor, GlobalProjector)
        
        self._projector_tensor = projector_tensor
        
        if dofs_extraction_op is not None:
            assert isinstance(dofs_extraction_op, (PolarExtractionOperator, IdentityOperator))
            self._dofs_extraction_op = dofs_extraction_op
        else:
            self._dofs_extraction_op = IdentityOperator(self.space.vector_space)
            
        if base_extraction_op is not None:
            assert isinstance(base_extraction_op, (PolarExtractionOperator, IdentityOperator))
            self._base_extraction_op = base_extraction_op
        else:
            self._base_extraction_op = IdentityOperator(self.space.vector_space)
        
        if boundary_op is not None:
            assert isinstance(boundary_op, (BoundaryOperator, IdentityOperator))
            self._boundary_op = boundary_op
        else:
            self._boundary_op = IdentityOperator(self.space.vector_space)
        
        # convert Kronecker inter-/histopolation matrix to Stencil-/BlockMatrix (only needed in polar case)
        if isinstance(self.dofs_extraction_op, PolarExtractionOperator):
            
            self._is_polar = True
            
            if isinstance(projector_tensor.imat_kronecker, KroneckerStencilMatrix):
                self._imat = projector_tensor.imat_kronecker.tostencil()
            else:
                
                blocks = [[projector_tensor.imat_kronecker.blocks[0][0].tostencil(), None, None],
                          [None, projector_tensor.imat_kronecker.blocks[1][1].tostencil(), None],
                          [None, None, projector_tensor.imat_kronecker.blocks[2][2].tostencil()]]
                
                self._imat = BlockMatrix(self.space.vector_space, self.space.vector_space, blocks)

            # set backend for acceleration
            self._imat.set_backend(PSYDAC_BACKEND_GPYCCEL)
            
        else:
            
            self._is_polar = False
            
            self._imat = projector_tensor.imat_kronecker
            
        # transposed
        self._imatT = self._imat.T
        
        # some shortcuts
        P = self._dofs_extraction_op
        E = self._base_extraction_op
        
        B = self._boundary_op

        # build inter-/histopolation matrix I = P * I * E^T and I0 = B * P * I * E^T * B^T as CompositeLinearOperator
        self._I  = CompositeLinearOperator(P, self._imat, E.transpose())
        self._I0 = CompositeLinearOperator(B, P, self._imat, E.transpose(), B.transpose())

        # transposed
        self._IT  = CompositeLinearOperator(E, self._imatT, P.transpose())
        self._I0T = CompositeLinearOperator(B, E, self._imatT, P.transpose(), B.transpose())

        # preconditioner P * I^(-1) * E^T and B * P * I^(-1) * E^T * B^T for iterative polar projections
        self._pc  = ProjectorPreconditioner(self, transposed=False, apply_bc=False)
        self._pc0 = ProjectorPreconditioner(self, transposed=False, apply_bc=True)
        
        # transposed
        self._pcT  = ProjectorPreconditioner(self, transposed=True, apply_bc=False)
        self._pc0T = ProjectorPreconditioner(self, transposed=True, apply_bc=True)
        
            
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
    def is_polar(self):
        """ Whether the projector maps to polar splines (True) or pure tensor product splines.
        """
        return self._is_polar

    @property
    def I(self):
        """ Inter-/histopolation matrix P * I * E^T as CompositeLinearOperator.
        """
        return self._I
    
    @property
    def I0(self):
        """ Inter-/histopolation matrix B * P * I * E^T * B^T as CompositeLinearOperator.
        """
        return self._I0

    @property
    def IT(self):
        """ Transposed inter-/histopolation matrix E * I^T * P^T as CompositeLinearOperator.
        """
        return self._IT
    
    @property
    def I0T(self):
        """ Transposed inter-/histopolation matrix B * E * I^T * P^T * B^T as CompositeLinearOperator.
        """
        return self._I0T
    
    @property
    def pc(self):
        """ Preconditioner P * I^(-1) * E^T for iterative polar projections.
        """
        return self._pc
    
    @property
    def pc0(self):
        """ Preconditioner B * P * I^(-1) * E^T * B^T for iterative polar projections.
        """
        return self._pc0
    
    @property
    def pcT(self):
        """ Transposed preconditioner P * I^(-T) * E^T for iterative polar projections.
        """
        return self._pcT
    
    @property
    def pc0T(self):
        """ Transposed preconditioner B * P * I^(-T) * E^T * B^T for iterative polar projections.
        """
        return self._pc0T

    def solve(self, rhs, transposed=False, apply_bc=False, tol=1e-14, maxiter=1000, verbose=False):
        """
        Solves the linear system I * x = rhs, resp. I^T * x = rhs for x, where I is the composite inter-/histopolation matrix.
        
        Parameters
        ----------
            rhs : StencilVector | BlockVector | PolarVector
                The right-hand side of the linear system.
                
            transposed : bool
                Whether to invert the transposed inter-/histopolation matrix.
                
            apply_bc : bool
                Whether to apply essential boundary conditions to degrees of freedom and coefficients.
                
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
            
        if transposed:
            
            # polar case (iterative solve with pbicgstab)
            if self.is_polar:
                if apply_bc:
                    x = pbicgstab(self.I0T, self.I0T.operators[0].dot(rhs), self.pc0T, tol=tol, maxiter=maxiter, verbose=verbose)[0]
                else:
                    x = pbicgstab(self.IT, rhs, self.pcT, tol=tol, maxiter=maxiter, verbose=verbose)[0]
                    
            # standard (tensor product) case (Kronecker solver)
            else:
                if apply_bc:
                    x = self.pc0T.solve(rhs)
                else:
                    x = self.pcT.solve(rhs)
            
        else:

            # polar case (iterative solve with pbicgstab)
            if self.is_polar:
                if apply_bc:
                    x = pbicgstab(self.I0, self.I0.operators[0].dot(rhs), self.pc0, tol=tol, maxiter=maxiter, verbose=verbose)[0]
                else:
                    x = pbicgstab(self.I, rhs, self.pc, tol=tol, maxiter=maxiter, verbose=verbose)[0]
                    
            # standard (tensor product) case (Kronecker solver)
            else:
                if apply_bc:
                    x = self.pc0.solve(rhs)
                else:
                    x = self.pc.solve(rhs)

        return x

    def get_dofs(self, fun, apply_bc=False):
        """
        Computes the geometric degrees of freedom associated to given callable(s).
        
        Parameters
        ----------
            fun : callable | list
                The function for which the geometric degrees of freedom shall be computed. List of callables for vector-valued functions.
                
            apply_bc : bool
                Whether to apply essential boundary conditions to degrees of freedom.
                
        Returns
        -------
            dofs : StencilVector | BlockVector | PolarVector
                The geometric degrees of freedom associated to given callable(s) "fun".
        """

        # get dofs on tensor-product grid + apply polar DOF extraction operator
        dofs = self.dofs_extraction_op.dot(self.projector_tensor(fun, dofs_only=True))

        # apply boundary operator
        if apply_bc:
            dofs = self.boundary_op.dot(dofs)

        return dofs

    def __call__(self, fun, apply_bc=False, tol=1e-14, maxiter=1000, verbose=False):
        """
        Applies projector to given callable(s).
        
        Parameters
        ----------
            fun : callable | list
                The function to be projected. List of three callables for vector-valued functions.
                
            apply_bc : bool
                Whether to apply essential boundary conditions to degrees of freedom and coefficients.
                
            tol : float
                Stop tolerance in iterative solve (only used in polar case).
                
            maxiter : int
                Maximum number of iterations in iterative solve (only used in polar case).
                
            verbose : bool
                Whether to print some information in each iteration in iterative solve (only used in polar case).
                
        Returns
        -------
            coeffs : StencilVector | BlockVector | PolarVector
                The FEM spline coefficients after projection.
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
