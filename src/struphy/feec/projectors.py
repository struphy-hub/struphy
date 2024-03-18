from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector, BlockLinearOperator
from psydac.linalg.kron import KroneckerStencilMatrix
from psydac.linalg.basic import Vector, IdentityOperator
from psydac.linalg.solvers import inverse
from psydac.fem.tensor import TensorFemSpace
from psydac.feec.global_projectors import GlobalProjector
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.polar.linear_operators import PolarExtractionOperator
from struphy.feec.mass import WeightedMassOperators
from struphy.feec import preconditioner
from struphy.feec.preconditioner import ProjectorPreconditioner
from struphy.feec import mass_kernels
from struphy.fields_background.mhd_equil.equils import set_defaults

import numpy as np


class CommutingProjector:
    """
    A commuting projector in a 3d de Rham diagram (can be polar). 

    The general structure of the inter-/histopolation problem reads

         (B * P * Imat * E^T * B^T) * coeffs = B * P * dofs,

    with the following linear operators:

        * B    : boundary operator,
        * P    : polar degrees of freedom extraction operator,
        * Imat : tensor product inter-/histopolation matrix,
        * E    : polar basis extraction operator.

    P and E (and B in case of no boundary conditions) can be identity operators, 
    which gives the pure tensor-product Psydac projector.

    Parameters
    ----------
    projector_tensor : psydac.feec.global_projectors.GlobalProjector
        The pure tensor product projector.

    dofs_extraction_op : struphy.polar.linear_operators.PolarExtractionOperator, optional
        The degrees of freedom extraction operator mapping tensor product DOFs to polar DOFs. If not given, is set to identity.

    base_extraction_op : struphy.polar.linear_operators.PolarExtractionOperator, optional
        The basis extraction operator mapping tensor product basis functions to polar basis functions. If not given, is set to identity.

    boundary_op : struphy.feec.linear_operators.BoundaryOperator.
        The boundary operator applying essential boundary conditions to a vector. If not given, is set to identity.
    """

    def __init__(self, projector_tensor: GlobalProjector, dofs_extraction_op=None, base_extraction_op=None, boundary_op=None):

        self._projector_tensor = projector_tensor

        if dofs_extraction_op is not None:
            self._dofs_extraction_op = dofs_extraction_op
        else:
            self._dofs_extraction_op = IdentityOperator(
                self.space.vector_space)

        if base_extraction_op is not None:
            self._base_extraction_op = base_extraction_op
        else:
            self._base_extraction_op = IdentityOperator(
                self.space.vector_space)

        if boundary_op is not None:
            self._boundary_op = boundary_op
        else:
            self._boundary_op = IdentityOperator(self.space.vector_space)

        # convert Kronecker inter-/histopolation matrix to Stencil-/BlockLinearOperator (only needed in polar case)
        if isinstance(self.dofs_extraction_op, PolarExtractionOperator):

            self._is_polar = True

            if isinstance(projector_tensor.imat_kronecker, KroneckerStencilMatrix):
                self._imat = projector_tensor.imat_kronecker.tostencil()
                self._imat.set_backend(
                    PSYDAC_BACKEND_GPYCCEL, precompiled=True)
            else:

                b11 = projector_tensor.imat_kronecker.blocks[0][0].tostencil()
                b11.set_backend(PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                b22 = projector_tensor.imat_kronecker.blocks[1][1].tostencil()
                b22.set_backend(PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                b33 = projector_tensor.imat_kronecker.blocks[2][2].tostencil()
                b33.set_backend(PSYDAC_BACKEND_GPYCCEL, precompiled=True)

                blocks = [[b11, None, None],
                          [None, b22, None],
                          [None, None, b33]]

                self._imat = BlockLinearOperator(
                    self.space.vector_space, self.space.vector_space, blocks)

        else:

            self._is_polar = False

            self._imat = projector_tensor.imat_kronecker

        # transposed
        self._imatT = self._imat.T

        # some shortcuts
        P = self._dofs_extraction_op
        E = self._base_extraction_op

        B = self._boundary_op

        # build inter-/histopolation matrix I = ID * P * I * E^T * ID^T and I0 = B * P * I * E^T * B^T as ComposedLinearOperator
        self._I = P @ self._imat @ E.T
        self._I0 = B @ self._I @ B.T

        # transposed
        self._IT = E @ self._imatT @ P.T
        self._I0T = B @ self._IT @ B.T

        # preconditioner ID * P * I^(-1) * E^T * ID^T and B * P * I^(-1) * E^T * B^T for iterative polar projections
        self._pc = ProjectorPreconditioner(
            self, transposed=False, apply_bc=False)
        self._pc0 = ProjectorPreconditioner(
            self, transposed=False, apply_bc=True)

        # transposed
        self._pcT = ProjectorPreconditioner(
            self, transposed=True, apply_bc=False)
        self._pc0T = ProjectorPreconditioner(
            self, transposed=True, apply_bc=True)

        # linear solver used for polar projections
        if self._is_polar:
            self._polar_solver = inverse(
                self._I, 'pbicgstab', pc=self._pc, tol=1e-14, maxiter=1000, verbose=False)
            self._polar_solver0 = inverse(
                self._I0, 'pbicgstab', pc=self._pc0, tol=1e-14, maxiter=1000, verbose=False)
            self._polar_solverT = inverse(
                self._IT, 'pbicgstab', pc=self._pcT, tol=1e-14, maxiter=1000, verbose=False)
            self._polar_solver0T = inverse(
                self._I0T, 'pbicgstab', pc=self._pc0T, tol=1e-14, maxiter=1000, verbose=False)
        else:
            self._polar_solver = None

        self._polar_info = None

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
        """ Boundary operator setting essential boundary conditions to Stencil-/BlockVector.
        """
        return self._boundary_op

    @property
    def is_polar(self):
        """ Whether the projector maps to polar splines (True) or pure tensor product splines.
        """
        return self._is_polar

    @property
    def I(self):
        """ Inter-/histopolation matrix ID * P * I * E^T * ID^T as ComposedLinearOperator (ID = IdentityOperator).
        """
        return self._I

    @property
    def I0(self):
        """ Inter-/histopolation matrix B * P * I * E^T * B^T as ComposedLinearOperator.
        """
        return self._I0

    @property
    def IT(self):
        """ Transposed inter-/histopolation matrix ID * E * I^T * P^T * ID^T as ComposedLinearOperator (ID = IdentityOperator).
        """
        return self._IT

    @property
    def I0T(self):
        """ Transposed inter-/histopolation matrix B * E * I^T * P^T * B^T as ComposedLinearOperator.
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

    def solve(self, rhs, transposed=False, apply_bc=False, out=None):
        """
        Solves the linear system I * x = rhs, resp. I^T * x = rhs for x, where I is the composite inter-/histopolation matrix.

        Parameters
        ----------
        rhs : psydac.linalg.basic.vector
            The right-hand side of the linear system.

        transposed : bool, optional
            Whether to invert the transposed inter-/histopolation matrix.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom and coefficients.

        out : psydac.linalg.basic.vector, optional
            If given, the result will be written into this vector in-place.

        Returns
        -------
        x : psydac.linalg.basic.vector
            Output vector (result of linear system).
        """

        assert isinstance(rhs, Vector)
        assert rhs.space == self._I.domain

        if transposed:
            # polar case (iterative solve with PBiConjugateGradientStab)
            if self.is_polar:
                if apply_bc:
                    x = self._polar_solver0T.solve(
                        self._boundary_op.T.dot(rhs), out=out)
                else:
                    x = self._polar_solverT.solve(
                        self._boundary_op.T.dot(rhs), out=out)
            # standard (tensor product) case (Kronecker solver)
            else:
                if apply_bc:
                    x = self.pc0T.solve(rhs, out=out)
                else:
                    x = self.pcT.solve(rhs, out=out)
        else:
            # polar case (iterative solve with PBiConjugateGradientStab)
            if self.is_polar:
                if apply_bc:
                    x = self._polar_solver0.solve(
                        self._boundary_op.T.dot(rhs), out=out)
                else:
                    x = self._polar_solver.solve(
                        self._boundary_op.T.dot(rhs), out=out)
            # standard (tensor product) case (Kronecker solver)
            else:
                if apply_bc:
                    x = self.pc0.solve(rhs, out=out)
                else:
                    x = self.pc.solve(rhs, out=out)

        return x

    def get_dofs(self, fun, dofs=None, apply_bc=False):
        """
        Computes the geometric degrees of freedom associated to given callable(s).

        Parameters
        ----------
        fun : callable | list
            The function for which the geometric degrees of freedom shall be computed. List of callables for vector-valued functions.

        dofs : psydac.linalg.basic.vector, optional
            If given, the dofs will be written into this vector in-place.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom.

        Returns
        -------
        dofs : psydac.linalg.basic.vector
            The geometric degrees of freedom associated to given callable(s) "fun".
        """
        # get dofs on tensor-product grid + apply polar DOF extraction operator
        if dofs is None:
            dofs = self.dofs_extraction_op.dot(
                self.projector_tensor(fun, dofs_only=True))
        else:
            self.dofs_extraction_op.dot(
                self.projector_tensor(fun, dofs_only=True), out=dofs)

        # apply boundary operator
        if apply_bc:
            dofs = self.boundary_op.dot(dofs)

        return dofs

    def __call__(self, fun, out=None, dofs=None, apply_bc=False):
        """
        Applies projector to given callable(s).

        Parameters
        ----------
        fun : callable | list
            The function to be projected. List of three callables for vector-valued functions.

        out : psydac.linalg.basic.vector, optional
            If given, the result will be written into this vector in-place.

        dofs : psydac.linalg.basic.vector, optional
            If given, the dofs will be written into this vector in-place.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom and coefficients.

        Returns
        -------
        coeffs : psydac.linalg.basic.vector
            The FEM spline coefficients after projection.
        """
        return self.solve(self.get_dofs(fun, dofs=dofs, apply_bc=apply_bc), transposed=False,
                          apply_bc=apply_bc, out=out)


class L2Projector:
    r"""
    A projector into the discrete Derham spaces based on the L2-scalar product in logical coordinates.

    It solves the following system for the FE-coefficients

    .. math::

        \mathbb M^p_{ijk, lmn} f_{lmn} = (f^p(\boldsymbol \eta), \Lambda^p_{ijk})_{L^2}\,,

    where :math:`M^p` denotes the mass matrix of space :math:`p` and :math:`f^p` is a :math:`p`-form proxy function. 

    Parameters:
    -----------
    space_id : str
        One of "H1", "Hcurl", "Hdiv", "L2" or "H1vec".

    mass_ops : struphy.mass.WeighteMassOperators
        Mass operators object, see :ref:`mass_ops`.

    params : dict
        Keyword arguments for the solver parameters.
    """

    def __init__(self, space_id, mass_ops, **params):


        assert space_id in ('H1', 'Hcurl', 'Hdiv', 'L2', 'H1vec')

        params_default = {'type': ('pcg', 'MassMatrixPreconditioner'),
                          'tol': 1.e-14,
                          'maxiter': 500,
                          'info': False,
                          'verbose': False, }

        set_defaults(params, params_default)

        self._space_id = space_id
        self._mass_ops = mass_ops
        self._params = params
        self._space_key = mass_ops.derham.space_to_form[self.space_id]
        self._space = mass_ops.derham.Vh_fem[self.space_key]

        # mass matrix
        self._Mmat = getattr(self.mass_ops, 'M' + self.space_key)

        # quadrature grid
        self._quad_grid_pts = self.mass_ops.derham.quad_grid_pts[self.space_key]

        if space_id in ('H1', 'L2'):
            self._quad_grid_mesh = np.meshgrid(
                *[pt.flatten() for pt in self.quad_grid_pts], indexing='ij')
            self._geom_weights = self.Mmat.weights[0][0](*self.quad_grid_mesh)
        else:
            self._quad_grid_mesh = []
            self._tmp = []  # tmp for matrix-vector product of geom_weights with fun
            for pts in self.quad_grid_pts:
                self._quad_grid_mesh += [np.meshgrid(*[pt.flatten() for pt in pts], indexing='ij')]
                self._tmp += [np.zeros_like(self.quad_grid_mesh[-1][0])]
            # geometric weights evaluated at quadrature grid
            self._geom_weights = []
            # loop over rows (different meshes)
            for mesh, row_weights in zip(self.quad_grid_mesh, self.Mmat.weights):
                self._geom_weights += [[]]
                # loop over columns (differnt geometric coeffs)
                for weight in row_weights:
                    if weight is not None:
                        self._geom_weights[-1] += [weight(*mesh)]
                    else: 
                        self._geom_weights[-1] += [np.zeros_like(mesh[0])]

        # other quad grid info
        if isinstance(self.space, TensorFemSpace):
            self._tensor_fem_spaces = [self.space]
            self._wts_l = [self.mass_ops.derham.quad_grid_wts[self.space_key]]
            self._spans_l = [
                self.mass_ops.derham.quad_grid_spans[self.space_key]]
            self._bases_l = [
                self.mass_ops.derham.quad_grid_bases[self.space_key]]
        else:
            self._tensor_fem_spaces = self.space.spaces
            self._wts_l = self.mass_ops.derham.quad_grid_wts[self.space_key]
            self._spans_l = self.mass_ops.derham.quad_grid_spans[self.space_key]
            self._bases_l = self.mass_ops.derham.quad_grid_bases[self.space_key]

        # Preconditioner
        if self.params['type'][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.params['type'][1])
            pc = pc_class(self.Mmat)

        # solver
        self._solver = inverse(self.Mmat,
                               self.params['type'][0],
                               pc=pc,
                               tol=self.params['tol'],
                               maxiter=self.params['maxiter'],
                               verbose=self.params['verbose'])

    @property
    def mass_ops(self):
        '''Struphy mass operators object, see :ref:`mass_ops`..'''
        return self._mass_ops

    @property
    def space_id(self):
        """ The ID of the space (H1, Hcurl, Hdiv, L2 or H1vec)."""
        return self._space_id

    @property
    def space_key(self):
        """ The key of the space (0, 1, 2, 3 or v)."""
        return self._space_key

    @property
    def space(self):
        '''The Derham finite element space (from Derham.Vh_fem).'''
        return self._space

    @property
    def params(self):
        '''Parameters for the iterative solver.'''
        return self._params

    @property
    def Mmat(self):
        '''The mass matrix of space.'''
        return self._Mmat

    @property
    def quad_grid_pts(self):
        '''List of quadrature points in each direction for integration over grid cells in format (ni, nq) = (cell, quadrature point).'''
        return self._quad_grid_pts

    @property
    def quad_grid_mesh(self):
        '''Mesh grids of quad_grid_pts.'''
        return self._quad_grid_mesh

    @property
    def geom_weights(self):
        '''Geometric coefficients (e.g. Jacobians) evaluated at quad_grid_mesh, stored as list[list] either 1x1 or 3x3.'''
        return self._geom_weights

    def solve(self, rhs, out=None):
        """
        Solves the linear system M * x = rhs, where M is the mass matrix.

        Parameters
        ----------
        rhs : psydac.linalg.basic.vector
            The right-hand side of the linear system.

        out : psydac.linalg.basic.vector, optional
            If given, the result will be written into this vector in-place.

        Returns
        -------
        out : psydac.linalg.basic.vector
            Output vector (result of linear system).
        """

        assert isinstance(rhs, Vector)

        if out is None:
            out = self._solver.dot(rhs)
        else:
            self._solver.dot(rhs, out=out)

        return out

    def get_dofs(self, fun, dofs=None, apply_bc=False, clear=True):
        r"""
        Assembles (in 3d) the Stencil-/BlockVector
        
        .. math::
        
            V_{ijk} = \int f * w_\textrm{geom} * \Lambda^\alpha_{ijk}\,\textrm d \boldsymbol \eta = \left( f\,, \Lambda^\alpha_{ijk}\right)_{L^2}\,, 
            
        where :math:`\Lambda^\alpha_{ijk}` are the basis functions of :math:`V_h^\alpha`,
        :math:`f` is an :math:`\alpha`-form proxy function and :math:`w_\textrm{geom}` stand for metric coefficients.

        Note that any geometric terms (e.g. Jacobians) in the L2 scalar product are automatically assembled 
        into :math:`w_\textrm{geom}`, depending on the space of :math:`\alpha`-forms.

        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.

        Parameters
        ----------
        fun : callable | list
            Weight function(s) (callables or np.ndarrays) in a 1d list of shape corresponding to number of components.

        dofs : StencilVector | BlockVector, optional
            The vector for the output.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom.

        clear : bool
            Whether to first set all data to zero before assembly. If False, the new contributions are added to existing ones in vec.
        """

        # evaluate fun at quad_grid or check array size
        if callable(fun):
            fun_weights = fun(*self._quad_grid_mesh)
        elif isinstance(fun, np.ndarray):
            assert fun.shape == self._quad_grid_mesh[
                0].shape, f'Expected shape {self._quad_grid_mesh[0].shape}, got {fun.shape = } instead.'
            fun_weights = fun
        else:
            assert len(
                fun) == 3, f'List input only for vector-valued spaces of size 3, but {len(fun) = }.'
            fun_weights = []
            # loop over rows (different meshes)
            for mesh in self._quad_grid_mesh:
                fun_weights += [[]]
                # loop over columns (different functions)
                for f in fun:
                    if callable(f):
                        fun_weights[-1] += [f(*mesh)]
                    elif isinstance(f, np.ndarray):
                        assert f.shape == mesh[
                            0].shape, f'Expected shape {mesh[0].shape}, got {f.shape = } instead.'
                        fun_weights[-1] += [f]
                    else:
                        raise ValueError(
                            f'Expected callable or numpy array, got {type(f) = } instead.')

        # check output vector
        if dofs is None:
            dofs = self.space.vector_space.zeros()
        else:
            assert isinstance(dofs, (StencilVector, BlockVector))
            assert dofs.space == self.Mmat.codomain

        # compute matrix data for kernel, i.e. fun * geom_weight
        tot_weights = []
        if isinstance(fun_weights, np.ndarray):
            tot_weights += [fun_weights * self.geom_weights]
        else:
            # loop over rows (differnt meshes)
            for row_fun, row_geom, tmp in zip(fun_weights, self.geom_weights, self._tmp):
                tmp *= 0.
                # loop over columns (different functions)
                for fun_weight, geom_weight in zip(row_fun, row_geom):
                    # matrix-vector product
                    tmp += fun_weight * geom_weight
                tot_weights += [tmp]
                
        # clear data
        if clear:
            if isinstance(dofs, StencilVector):
                dofs._data[:] = 0.
            else:
                for block in dofs.blocks:
                    block._data[:] = 0.

        # loop over components (just one for scalar spaces)
        for a, (fem_space, spans, wts, basis, mat_w) in enumerate(zip(self._tensor_fem_spaces,
                                                                      self._spans_l,
                                                                      self._wts_l,
                                                                      self._bases_l,
                                                                      tot_weights)):
            # indices
            starts = [int(start) for start in fem_space.vector_space.starts]
            pads = fem_space.vector_space.pads

            if isinstance(dofs, StencilVector):
                mass_kernels.kernel_3d_vec(*spans, *fem_space.degree, *starts, *pads,
                                           *wts, *basis, mat_w, dofs._data)
            else:
                mass_kernels.kernel_3d_vec(*spans, *fem_space.degree, *starts, *pads,
                                           *wts, *basis, mat_w, dofs[a]._data)

        # exchange assembly data (accumulate ghost regions) and update ghost regions
        dofs.exchange_assembly_data()
        dofs.update_ghost_regions()

        # apply boundary operator
        if apply_bc:
            dofs = self.mass_ops.derham.boundary_ops[self.space_key].dot(dofs)

        return dofs

    def __call__(self, fun, out=None, dofs=None, apply_bc=False):
        """
        Applies projector to given callable(s).

        Parameters
        ----------
        fun : callable | list
            The function to be projected. List of three callables for vector-valued functions.

        out : psydac.linalg.basic.vector, optional
            If given, the result will be written into this vector in-place.

        dofs : psydac.linalg.basic.vector, optional
            If given, the dofs will be written into this vector in-place.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom and coefficients.

        Returns
        -------
        coeffs : psydac.linalg.basic.vector
            The FEM spline coefficients after projection.
        """
        return self.solve(self.get_dofs(fun, dofs=dofs, apply_bc=apply_bc), out=out)
