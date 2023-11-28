from abc import ABCMeta, abstractmethod

from psydac.linalg.block import BlockLinearOperator
from psydac.linalg.kron import KroneckerStencilMatrix
from psydac.linalg.basic import Vector, IdentityOperator
from psydac.linalg.solvers import inverse
from psydac.feec.global_projectors import GlobalProjector
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.feec.preconditioner import ProjectorPreconditioner
from struphy.polar.linear_operators import PolarExtractionOperator

import numpy as np


# class Projector(metaclass=ABCMeta):
#     """Projector into one of the discrete de Rham spaces.

#     The __call__ method takes callable functions as an input and returns a Stencil-/BlockVector.
#     """

#     def __init__(self, derham, space_id, solver=('tensor_product', None), tol=1e-14, maxiter=1000, verbose=False):
#         """
#         Parameters
#         ----------
#         derham : struphy.feec.psydac_derham.Derham
#             Struphy de Rham sequence.

#         space_id : str
#             Space to project into ("H1", "Hcurl", "Hdiv", "L2" or "H1vec").

#         solver : tuple
#             Tuple of the form (solver_name, preconditioner), describing the solver to be used in the projection problem.
#             Solvers from :ref:`struphy.linear_algebra.direct_solvers` or from `struphy.linear_algebra.iterative_solvers`.
#             Preconditioners from :ref:`struphy.feec.preconditioner`.

#         tol : float, optional
#             Stop tolerance in iterative solve (only used in polar case).

#         maxiter : int, optional
#             Maximum number of iterations in iterative solve (only used in polar case).

#         verbose : bool, optional
#             Whether to print some information in each iteration in iterative solve (only used in polar case).
#         """

#         self._derham = derham
#         self._space_id = space_id
#         self._form = derham.space_to_form[space_id]

#         if 'tensor_product' in solver[0]:

#             if solver[1] is None:
#                 self._solver = derham.P[self.form].solver

#             elif solver[1] == 'imat_inverse':
#                 self._solver = PBiConjugateGradientStab(derham.Vh[self.form])

#             else:
#                 raise ValueError(f'solver[1] must be None or "imat_inverse", but is {solver[1] = }')

#     @property
#     def derham(self):
#         """:class:`struphy.feec.psydac_derham.Derham`"""
#         return self._derham

#     @property
#     def space_id(self):
#         """Space to project from ("H1", "Hcurl", "Hdiv", "L2" or "H1vec")."""
#         return self._space_id

#     @property
#     def form(self):
#         """p-form proxy ("0", "1", "2", "3" or "v")."""
#         return self._form

#     @property
#     def solver(self):
#         """The solver to be used in the projection problem. """
#         return self._solver

#     @abstractmethod
#     def get_dofs(self, fun):
#         """Computes the right-hand side of the projection problem.

#         Parameters
#         ----------
#         fun : callable | list
#             The function to be projected. List of three callables for vector-valued functions.

#         Returns
#         -------
#             A Stencil-/BlockVector corresponding to self.space_id.
#         """
#         pass

#     def __call__(self, fun):
#         """
#         Applies projector to given callable(s).

#         Parameters
#         ----------
#         fun : callable | list
#             The function to be projected. List of three callables for vector-valued functions.

#         Returns
#         -------
#         coeffs : psydac.linalg.basic.vector
#             The FEM spline coefficients after projection.
#         """
#         rhs = self.get_dofs(fun)
#         self.solver.solve(rhs, out=self._out)

#         return self._out


# class CommutingProjector(Projector):
#     """
#     A commuting projector in a 3d tensor-product de Rham diagram.

#     The general structure of the inter-/histopolation problem reads

#          (B * I * B^T) * coeffs = B * dofs,

#     with the following linear operators:

#         * B : Dirichlet boundary operator,
#         * I : tensor product inter-/histopolation matrix,

#     B can be the identity operator, which gives the pure tensor-product Psydac projector.

#     Parameters
#     ----------
#     derham : struphy.feec.psydac_derham.Derham
#         Struphy de Rham sequence.

#     space_id : str
#         Space to project into ("H1", "Hcurl", "Hdiv", "L2" or "H1vec").
#     """

#     def __init__(self, derham, space_id, apply_bc=False):

#         if apply_bc:
#             solver = ('tensor_product', 'imat_inverse')
#         else:
#             solver = ('tensor_product', None)

#         super().__init__(derham, space_id, solver=solver)

#         # ------ the following is only relevant when apply_bc=True ------
#         self._projector_tensor = derham.P[self.form]

#         print(f'{self.projector_tensor = }')
#         print(f'{self.solver = }')

#         imat = derham.P[self.form].imat_kronecker
#         imatT = self._imat.T

#         print(f'{imat = }')

#         B = derham.boundary_ops[self.form]

#         # build inter-/histopolation matrix I = B * I * B^T as CompositLinearOperator
#         self._I = CompositeLinearOperator(B, imat, B.T)
#         self._IT = CompositeLinearOperator(B, imatT, B.T)

#         # preconditioner B * I^(-1) * B^T
#         self._pc = ProjectorPreconditioner(self, transposed=False)
#         self._pcT = ProjectorPreconditioner(self, transposed=True)

#     @property
#     def projector_tensor(self):
#         """ Tensor product projector.
#         """
#         return self._projector_tensor

#     @property
#     def I(self):
#         """ Inter-/histopolation matrix B * I * B^T as CompositLinearOperator.
#         """
#         return self._I

#     @property
#     def IT(self):
#         """ Transpose of I.
#         """
#         return self._IT

#     @property
#     def pc(self):
#         """ Preconditioner B * I^(-1) * B^T.
#         """
#         return self._pc

#     @property
#     def pcT(self):
#         """ Transpose of pc.
#         """
#         return self._pcT

#     def solve(self, rhs, transposed=False, apply_bc=False, tol=1e-14, maxiter=1000, verbose=False, out=None):
#         """
#         Solves the linear system I * x = rhs, resp. I^T * x = rhs for x, where I is the composite inter-/histopolation matrix.

#         Parameters
#         ----------
#         rhs : psydac.linalg.basic.vector
#             The right-hand side of the linear system.

#         transposed : bool, optional
#             Whether to invert the transposed inter-/histopolation matrix.

#         apply_bc : bool, optional
#             Whether to apply essential boundary conditions to degrees of freedom and coefficients.

#         tol : float, optional
#             Stop tolerance in iterative solve (only used in polar case).

#         maxiter : int, optional
#             Maximum number of iterations in iterative solve (only used in polar case).

#         verbose : bool, optional
#             Whether to print some information in each iteration in iterative solve (only used in polar case).

#         out : psydac.linalg.basic.vector, optional
#             If given, the result will be written into this vector in-place.

#         Returns
#         -------
#         x : psydac.linalg.basic.vector
#             Output vector (result of linear system).
#         """

#         assert isinstance(rhs, Vector)
#         assert rhs.space == self._I.domain

#         if transposed:

#             # polar case (iterative solve with PBiConjugateGradientStab)
#             if self.is_polar:
#                 if apply_bc:
#                     x, self._polar_info = self._polar_solver.solve(
#                         self.I0T, self.I0T.multiplicants[0].dot(rhs),
#                         self.pc0T, tol=tol, maxiter=maxiter,
#                         verbose=verbose, out=out)
#                 else:
#                     x, self._polar_info = self._polar_solver.solve(
#                         self.IT, self.IT.multiplicants[0].dot(rhs),
#                         self.pcT, tol=tol, maxiter=maxiter,
#                         verbose=verbose, out=out)

#             # standard (tensor product) case (Kronecker solver)
#             else:
#                 if apply_bc:
#                     x = self.pc0T.solve(rhs, out=out)
#                 else:
#                     x = self.pcT.solve(rhs, out=out)

#         else:

#             # polar case (iterative solve with PBiConjugateGradientStab)
#             if self.is_polar:
#                 if apply_bc:
#                     x, self._polar_info = self._polar_solver.solve(
#                         self.I0, self.I0.multiplicants[0].dot(rhs),
#                         self.pc0, tol=tol, maxiter=maxiter,
#                         verbose=verbose, out=out)
#                 else:
#                     x, self._polar_info = self._polar_solver.solve(
#                         self.I, self.I.multiplicants[0].dot(rhs),
#                         self.pc, tol=tol, maxiter=maxiter,
#                         verbose=verbose, out=out)

#             # standard (tensor product) case (Kronecker solver)
#             else:
#                 if apply_bc:
#                     x = self.pc0.solve(rhs, out=out)
#                 else:
#                     x = self.pc.solve(rhs, out=out)

#         return x

#     def get_dofs(self, fun, apply_bc=False):
#         """
#         Computes the geometric degrees of freedom associated to given callable(s).

#         Parameters
#         ----------
#         fun : callable | list
#             The function for which the geometric degrees of freedom shall be computed. List of callables for vector-valued functions.

#         apply_bc : bool, optional
#             Whether to apply essential boundary conditions to degrees of freedom.

#         Returns
#         -------
#         dofs : psydac.linalg.basic.vector
#             The geometric degrees of freedom associated to given callable(s) "fun".
#         """

#         # get dofs on tensor-product grid + apply polar DOF extraction operator
#         dofs = self.dofs_extraction_op.dot(
#             self.projector_tensor(fun, dofs_only=True))

#         # apply boundary operator
#         if apply_bc:
#             dofs = self.boundary_op.dot(dofs)

#         return dofs

#     def __call__(self, fun, apply_bc=False, tol=1e-14, maxiter=1000, verbose=False):
#         """
#         Applies projector to given callable(s).

#         Parameters
#         ----------
#         fun : callable | list
#             The function to be projected. List of three callables for vector-valued functions.

#         apply_bc : bool, optional
#             Whether to apply essential boundary conditions to degrees of freedom and coefficients.

#         tol : float, optional
#             Stop tolerance in iterative solve (only used in polar case).

#         maxiter : int, optional
#             Maximum number of iterations in iterative solve (only used in polar case).

#         verbose : bool, optional
#             Whether to print some information in each iteration in iterative solve (only used in polar case).

#         Returns
#         -------
#         coeffs : psydac.linalg.basic.vector
#             The FEM spline coefficients after projection.
#         """
#         return self.solve(self.get_dofs(fun, apply_bc), transposed=False,
#                           apply_bc=apply_bc, tol=tol,
#                           maxiter=maxiter, verbose=verbose)


class PolarCommutingProjector():
    """
    A commuting projector in a 3d polar de Rham diagram. 

    The general structure of the inter-/histopolation problem reads

         (B * P * I * E^T * B^T) * coeffs = B * P * dofs,

    with the following linear operators:

        * B : boundary operator,
        * P : polar degrees of freedom extraction operator,
        * I : tensor product inter-/histopolation matrix,
        * E : polar basis extraction operator.

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

    def __init__(self, projector_tensor, dofs_extraction_op=None, base_extraction_op=None, boundary_op=None):

        assert isinstance(projector_tensor, GlobalProjector)

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

        tol : float, optional
            Stop tolerance in iterative solve (only used in polar case).

        maxiter : int, optional
            Maximum number of iterations in iterative solve (only used in polar case).

        verbose : bool, optional
            Whether to print some information in each iteration in iterative solve (only used in polar case).

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

    def get_dofs(self, fun, apply_bc=False):
        """
        Computes the geometric degrees of freedom associated to given callable(s).

        Parameters
        ----------
        fun : callable | list
            The function for which the geometric degrees of freedom shall be computed. List of callables for vector-valued functions.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom.

        Returns
        -------
        dofs : psydac.linalg.basic.vector
            The geometric degrees of freedom associated to given callable(s) "fun".
        """

        # get dofs on tensor-product grid + apply polar DOF extraction operator
        dofs = self.dofs_extraction_op.dot(
            self.projector_tensor(fun, dofs_only=True))

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

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom and coefficients.

        tol : float, optional
            Stop tolerance in iterative solve (only used in polar case).

        maxiter : int, optional
            Maximum number of iterations in iterative solve (only used in polar case).

        verbose : bool, optional
            Whether to print some information in each iteration in iterative solve (only used in polar case).

        Returns
        -------
        coeffs : psydac.linalg.basic.vector
            The FEM spline coefficients after projection.
        """
        return self.solve(self.get_dofs(fun, apply_bc), transposed=False,
                          apply_bc=apply_bc)


# class L2Projector:
#     r"""
#     A projector into the discrete de Rham spaces based on the L2-scalar product.

#     Computes the L2 scalar product with basis functions via Gauss-Legendre quadrature:

#     .. math::

#         <f, \Lambda^\alpha_i> = \int_{[0,1]^3} f(\eta) \Lambda^\alpha_i(\eta) \, \text{d} \eta \approx \sum_k w_k f(\eta_k) \Lambda^\alpha_i(\eta_k)

#     TODO (for now only a=0, i.e. space=H1)
#     and solves the following system for the FE-coefficients

#     .. math::

#         M^\alpha_{ij} f_j = <f, \Lambda^\alpha_i>

#     Parameters:
#     -----------
#     """

#     def __init__(
#         self, mass_mat, derham: Derham,
#         **params
#     ):
#         params_default = {
#             'space': 'H1',
#             'solver_params': {
#                 'type': ('PConjugateGradient', 'MassMatrixPreconditioner'),
#                 'tol': 1.e-14,
#                 'maxiter': 3000,
#                 'info': False,
#                 'verbose': False,
#             }
#         }

#         set_defaults(params, params_default)

#         self._params = params
#         self._solver_params = params['solver_params']

#         # TODO generalize
#         assert params['space'] == 'H1'
#         self._space_id = params['space']

#         assert isinstance(derham, Derham)
#         self._derham = derham
#         self._lhs_mat = mass_mat

#         self._space_key = self.derham.space_to_form[self.space_id]
#         self._space = derham.Vh_fem[self.space_key]

#         if self.space_id in ("H1", "L2"):
#             self._lhs = StencilVector(
#                 derham.Vh_fem[self.space_key].vector_space)
#             self._rhs = StencilVector(
#                 derham.Vh_fem[self.space_key].vector_space)

#         elif self.space_id in ("Hcurl", "Hdiv", "H1vec"):
#             self._lhs = BlockVector(derham.Vh_fem[self.space_key].vector_space)
#             self._rhs = BlockVector(derham.Vh_fem[self.space_key].vector_space)

#         # Preconditioner
#         if self.solver_params['type'][1] is None:
#             self._pc = None
#         else:
#             pc_class = getattr(preconditioner, self.solver_params['type'][1])
#             self._pc = pc_class(mass_mat)

#         self._solver = getattr(
#             it_solvers, self.solver_params['type'][0])(mass_mat.domain)

#         self._kernel = getattr(util_kernels, 'l2_projection_V0')

#     @property
#     def derham(self):
#         return self._derham

#     @property
#     def space_id(self):
#         """ The ID of the space, one of [H1, Hcurl, Hdiv, L2, H1vec] """
#         return self._space_id

#     @property
#     def space_key(self):
#         """ The key of the space, one of [0, 1, 2, 3, 4] """
#         return self._space_key

#     @property
#     def space(self):
#         return self._space

#     @property
#     def params(self):
#         return self._params

#     @property
#     def params(self):
#         return self._params

#     @property
#     def solver_params(self):
#         return self._solver_params

#     @property
#     def lhs_mat(self):
#         return self._lhs_mat

#     def __call__(self, fun):
#         """
#         Performs the L2-projection of fun.

#         Parameters
#         ----------
#         fun : callable or np.ndarray
#             TODO
#         """
#         if callable(fun):
#             locs, scaled_wts = self.generate_quad_points()
#             etas = np.meshgrid(*locs, indexing='ij')

#             fun_vals = fun(*etas)

#             self._kernel(np.array(self.derham.p),
#                          self.derham.Vh_fem['0'].knots[0],
#                          self.derham.Vh_fem['0'].knots[1],
#                          self.derham.Vh_fem['0'].knots[2],
#                          np.array(self.derham.Vh['0'].starts),
#                          self._rhs._data,
#                          locs[0], locs[1], locs[2],
#                          scaled_wts[0], scaled_wts[1], scaled_wts[2],
#                          fun_vals)

#             self._rhs.exchange_assembly_data()
#             self._rhs.update_ghost_regions()

#         elif isinstance(fun, np.ndarray):
#             raise NotImplementedError('This option is not supported yet!')
#         else:
#             raise ValueError(
#                 f'fun must be either a callable function or a numpy array but is of type {type(fun)}')

#         self._solver.solve(
#             self.lhs_mat, self._rhs, pc=self._pc,
#             tol=self.solver_params['tol'], maxiter=self.solver_params['maxiter'],
#             verbose=self.solver_params['verbose'], out=self._lhs
#         )

#         return self._lhs

#     def generate_quad_points(self):
#         """ Generate the quadrature points and weights for all cells.

#         Returns
#         -------
#         locs : list of np.ndarrays
#             array with eta-directions in first axis and flattened quad points along the second axis.

#         scaled_weights : list of np.ndarrays
#             array with eta-directions in first axis and flattened quad weights multiplied by cell length
#             along the second axis.
#         """
#         pts = [None]*3
#         wts = [None]*3

#         for k in range(3):
#             pts[k], wts[k] = np.polynomial.legendre.leggauss(
#                 self.derham.nquads[k])

#         locs = [None]*3
#         scaled_weights = [None]*3

#         for k in range(3):
#             amb = (
#                 self.derham.breaks_loc[k][1:] - self.derham.breaks_loc[k][:-1]) / 2
#             apb = (
#                 self.derham.breaks_loc[k][1:] + self.derham.breaks_loc[k][:-1]) / 2

#             locs[k] = (np.multiply.outer(amb, pts[k]) + apb[:, None]).ravel()
#             scaled_weights[k] = np.multiply.outer(amb, wts[k]).ravel()

#         return locs, scaled_weights


# def evaluate_fun_weights_1d(pts, wts, fun):
#     """
#     Pre-evaluates the given function at the quadrature points,
#     and multiplies the result with the quadrature weights of this point.
#     Quadrature weights and coordinates are given in a tensor-product format.

#     This version of the function loops over all elements and is fixed to dimension 1.

#     Parameters
#     ----------
#     pts : 1-tuple of 2d float arrays
#         Quadrature points in each dimension in format (element, quadrature point).

#     wts : 1-tuple of 2d float arrays
#         Quadrature weights in each dimension in format (element, quadrature point).

#     fun : callable
#         The function which shall be evaluated at eta1.

#     Returns
#     -------
#     values : ndarray[float]
#         A 2d array (1 cell grid dimension, 1 quadrature point dimension) which contains all the pre-evaluated values.
#     """

#     # will not be pyccelized, due to dependence on func (or could we call back to Python?)
#     values = np.zeros((pts[0].shape[0], pts[0].shape[1]), dtype=float)

#     for i in range(pts[0].shape[0]):  # element index
#         for iq in range(pts[0].shape[1]):  # quadrature point index
#             values[i, iq] = fun(pts[0][i, iq]) * wts[0][i, iq]

#     return values


# def evaluate_fun_weights_2d(pts, wts, fun):
#     """
#     Pre-evaluates the given function at the quadrature points,
#     and multiplies the result with the quadrature weights of this point.
#     Quadrature weights and coordinates are given in a tensor-product format.

#     This version of the function loops over all elements and is fixed to dimension 2.

#     Parameters
#     ----------
#     pts : 2-tuple of 2d float arrays
#         Quadrature points in each dimension in format (element, quadrature point).

#     wts : 2-tuple of 2d float arrays
#         Quadrature weights in each dimension in format (element, quadrature point).

#     fun : callable
#         The function which shall be evaluated at eta1, eta2.

#     Returns
#     -------
#     values : ndarray[float]
#         A 4d array (2 cell grid dimensions, 2 quadrature point dimensions) which contains all the pre-evaluated values.
#     """

#     # will not be pyccelized, due to dependence on func (or could we call back to Python?)
#     values = np.zeros((pts[0].shape[0], pts[1].shape[0],
#                        pts[0].shape[1], pts[1].shape[1]), dtype=float)

#     for i in range(pts[0].shape[0]):  # element index
#         for j in range(pts[1].shape[0]):
#             for iq in range(pts[0].shape[1]):  # quadrature point index
#                 for jq in range(pts[1].shape[1]):
#                     funval = fun(pts[0][i, iq], pts[1][j, jq])
#                     weightval = wts[0][i, iq] * wts[1][j, jq]
#                     values[i, j, iq, jq] = weightval * funval

#     return values


# def evaluate_fun_weights_3d(pts, wts, fun):
#     """
#     Pre-evaluates the given function at the quadrature points,
#     and multiplies the result with the quadrature weights of this point.
#     Quadrature weights and coordinates are given in a tensor-product format.

#     This version of the function loops over all elements and is fixed to dimension 3.

#     Parameters
#     ----------
#     pts : 3-tuple of 2d float arrays
#         Quadrature points in each dimension in format (element, quadrature point).

#     wts : 3-tuple of 2d float arrays
#         Quadrature weights in each dimension in format (element, quadrature point).

#     fun : callable
#         The function which shall be evaluated at eta1, eta2, eta3.

#     Returns
#     -------
#     values : ndarray[float]
#         A 6d array (3 cell grid dimensions, 3 quadrature point dimensions) which contains all the pre-evaluated values.
#     """

#     # will not be pyccelized, due to dependence on func (or could we call back to Python?)
#     values = np.zeros((pts[0].shape[0], pts[1].shape[0], pts[2].shape[0],
#                        pts[0].shape[1], pts[1].shape[1], pts[2].shape[1]), dtype=float)

#     for i in range(pts[0].shape[0]):  # element index
#         for j in range(pts[1].shape[0]):
#             for k in range(pts[2].shape[0]):
#                 for iq in range(pts[0].shape[1]):  # quadrature point index
#                     for jq in range(pts[1].shape[1]):
#                         for kq in range(pts[2].shape[1]):
#                             funval = fun(pts[0][i, iq], pts[1]
#                                          [j, jq], pts[2][k, kq])
#                             weightval = wts[0][i, iq] * \
#                                 wts[1][j, jq] * wts[2][k, kq]
#                             values[i, j, k, iq, jq, kq] = weightval * funval

#     return values


# def assemble_funccache_numpy(u, w, func):
#     """
#     Pre-evaluates the given function at the quadrature points,
#     and multiplies the result with the quadrature weights of this point.
#     Quadrature weights and coordinates are given in a tensor-product format.

#     This version tries to use numpy where possible, and is usable in arbitrary dimensions.

#     Parameters
#     ----------
#     u : three-tuple of two-dimensional numpy arrays
#         The quadrature points in each dimension.

#     w : three-tuple of two-dimensional numpy arrays
#         The quadrature weights in each dimension for the respective points.

#     func : callable, with three parameters
#         The function which shall be evaluated.

#     Returns
#     -------
#     values : ndarray[float]
#         A 6d array (3 cell grid dimensions, 3 quadrature point dimensions) which contains all the pre-evaluated values.
#     """

#     import numpy as np

#     funcvec = np.vectorize(func)
#     grid = np.meshgrid(*u, sparse=True, indexing='ij')
#     funceval = funcvec(*grid)

#     for wg in np.meshgrid(*w, sparse=True, indexing='ij'):
#         funceval *= wg

#     funceval.shape = tuple(uxx for ux in u for uxx in ux.shape)

#     n = len(u)
#     return funceval.transpose([2*i for i in range(n)] + [2*i+1 for i in range(n)])
