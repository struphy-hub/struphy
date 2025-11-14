import cunumpy as xp
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.ddm.mpi import mpi as MPI
from psydac.feec.global_geometric_projectors import GlobalGeometricProjector
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.basic import ComposedLinearOperator, IdentityOperator, LinearOperator, Vector
from psydac.linalg.block import BlockLinearOperator, BlockVector
from psydac.linalg.kron import KroneckerStencilMatrix
from psydac.linalg.solvers import inverse
from psydac.linalg.stencil import StencilMatrix, StencilVector

from struphy.feec import mass_kernels
from struphy.feec.local_projectors_kernels import (
    compute_shifts,
    get_dofs_local_1_form_ec_component,
    get_dofs_local_1_form_ec_component_weighted,
    get_dofs_local_2_form_ec_component,
    get_dofs_local_2_form_ec_component_weighted,
    get_dofs_local_3_form,
    get_dofs_local_3_form_weighted,
    get_local_problem_size,
    solve_local_main_loop,
    solve_local_main_loop_weighted,
)
from struphy.feec.utilities_local_projectors import (
    build_translation_list_for_non_zero_spline_indices,
    determine_non_zero_rows_for_each_spline,
    evaluate_relevant_splines_at_relevant_points,
    get_non_zero_B_spline_indices,
    get_non_zero_D_spline_indices,
    get_splines_that_are_relevant_for_at_least_one_block,
    is_spline_zero_at_quadrature_points,
    split_points,
)
from struphy.fields_background.equils import set_defaults
from struphy.kernel_arguments.local_projectors_args_kernels import LocalProjectorsArguments
from struphy.polar.basic import PolarVector
from struphy.polar.linear_operators import PolarExtractionOperator


class CommutingProjector:
    r"""
    A commuting projector of the 3d :class:`~struphy.feec.psydac_derham.Derham` diagram (can be polar).

    The general structure of the inter-/histopolation problem reads:
    given a function :math:`f \in V^\alpha` in one of the (continuous) de Rham spaces :math:`\alpha \in \{0,1,2,3,v\}`,
    find its projection :math:`f_h \in V_h^\alpha \subset V^\alpha` determined by
    the spline coefficients :math:`\mathbf f \in \mathbb R^{N_\alpha}` such that

    .. math::

         \mathbb B \mathbb P\, \mathcal I\, \mathbb E^T \mathbb B^T \mathbf f = \mathbb B \mathbb P\, \mathbf d(f)\,,

    where :math:`\mathbf d(f) \in \mathbb R^{N_\alpha}` are the degrees of freedom corresponding to :math:`f`,
    and with the following linear operators:

    * :math:`\mathbb B`: :class:`~struphy.feec.linear_operators.BoundaryOperator`,
    * :math:`\mathbb P`: :class:`~struphy.polar.linear_operators.PolarExtractionOperator` for degrees of freedom,
    * :math:`\mathcal I`: Kronecker product inter-/histopolation matrix, from :class:`~psydac.feec.global_geometric_projectors.GlobalGeometricProjector`
    * :math:`\mathbb E`: :class:`~struphy.polar.linear_operators.PolarExtractionOperator` for FE coefficients.

    :math:`\mathbb P` and :math:`\mathbb E` (and :math:`\mathbb B` in case of no boundary conditions) can be identity operators,
    which gives the pure tensor-product Psydac :class:`~psydac.feec.global_geometric_projectors.GlobalGeometricProjector`.

    Parameters
    ----------
    projector_tensor : GlobalGeometricProjector
        The pure tensor product projector.

    dofs_extraction_op : PolarExtractionOperator, optional
        The degrees of freedom extraction operator mapping tensor product DOFs to polar DOFs. If not given, is set to identity.

    base_extraction_op : PolarExtractionOperator, optional
        The basis extraction operator mapping tensor product basis functions to polar basis functions. If not given, is set to identity.

    boundary_op : BoundaryOperator
        The boundary operator applying essential boundary conditions to a vector. If not given, is set to identity.
    """

    def __init__(
        self,
        projector_tensor: GlobalGeometricProjector,
        dofs_extraction_op=None,
        base_extraction_op=None,
        boundary_op=None,
    ):
        self._projector_tensor = projector_tensor

        if dofs_extraction_op is not None:
            self._dofs_extraction_op = dofs_extraction_op
        else:
            self._dofs_extraction_op = IdentityOperator(
                self.space.coeff_space,
            )

        if base_extraction_op is not None:
            self._base_extraction_op = base_extraction_op
        else:
            self._base_extraction_op = IdentityOperator(
                self.space.coeff_space,
            )

        if boundary_op is not None:
            self._boundary_op = boundary_op
        else:
            self._boundary_op = IdentityOperator(self.space.coeff_space)

        # convert Kronecker inter-/histopolation matrix to Stencil-/BlockLinearOperator (only needed in polar case)
        if isinstance(self.dofs_extraction_op, PolarExtractionOperator):
            self._is_polar = True

            if isinstance(projector_tensor.imat_kronecker, KroneckerStencilMatrix):
                self._imat = projector_tensor.imat_kronecker.tostencil()
                self._imat.set_backend(
                    PSYDAC_BACKEND_GPYCCEL,
                    precompiled=True,
                )
            else:
                b11 = projector_tensor.imat_kronecker.blocks[0][0].tostencil()
                b11.set_backend(PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                b22 = projector_tensor.imat_kronecker.blocks[1][1].tostencil()
                b22.set_backend(PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                b33 = projector_tensor.imat_kronecker.blocks[2][2].tostencil()
                b33.set_backend(PSYDAC_BACKEND_GPYCCEL, precompiled=True)

                blocks = [
                    [b11, None, None],
                    [None, b22, None],
                    [None, None, b33],
                ]

                self._imat = BlockLinearOperator(
                    self.space.coeff_space,
                    self.space.coeff_space,
                    blocks,
                )

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
            self,
            transposed=False,
            apply_bc=False,
        )
        self._pc0 = ProjectorPreconditioner(
            self,
            transposed=False,
            apply_bc=True,
        )

        # transposed
        self._pcT = ProjectorPreconditioner(
            self,
            transposed=True,
            apply_bc=False,
        )
        self._pc0T = ProjectorPreconditioner(
            self,
            transposed=True,
            apply_bc=True,
        )

        # linear solver used for polar projections
        if self._is_polar:
            self._polar_solver = inverse(
                self._I,
                "pbicgstab",
                pc=self._pc,
                tol=1e-14,
                maxiter=1000,
                verbose=False,
            )
            self._polar_solver0 = inverse(
                self._I0,
                "pbicgstab",
                pc=self._pc0,
                tol=1e-14,
                maxiter=1000,
                verbose=False,
            )
            self._polar_solverT = inverse(
                self._IT,
                "pbicgstab",
                pc=self._pcT,
                tol=1e-14,
                maxiter=1000,
                verbose=False,
            )
            self._polar_solver0T = inverse(
                self._I0T,
                "pbicgstab",
                pc=self._pc0T,
                tol=1e-14,
                maxiter=1000,
                verbose=False,
            )
        else:
            self._polar_solver = None

        self._polar_info = None

    @property
    def projector_tensor(self):
        """Tensor product projector."""
        return self._projector_tensor

    @property
    def space(self):
        """Tensor product FEM space corresponding to projector."""
        return self._projector_tensor.space

    @property
    def dofs_extraction_op(self):
        """Degrees of freedom extraction operator (tensor product DOFs --> polar DOFs)."""
        return self._dofs_extraction_op

    @property
    def base_extraction_op(self):
        """Basis functions extraction operator (tensor product basis functions --> polar basis functions)."""
        return self._base_extraction_op

    @property
    def boundary_op(self):
        """Boundary operator setting essential boundary conditions to Stencil-/BlockVector."""
        return self._boundary_op

    @property
    def is_polar(self):
        """Whether the projector maps to polar splines (True) or pure tensor product splines."""
        return self._is_polar

    @property
    def I(self):
        """Inter-/histopolation matrix ID * P * I * E^T * ID^T as ComposedLinearOperator (ID = IdentityOperator)."""
        return self._I

    @property
    def I0(self):
        """Inter-/histopolation matrix B * P * I * E^T * B^T as ComposedLinearOperator."""
        return self._I0

    @property
    def IT(self):
        """Transposed inter-/histopolation matrix ID * E * I^T * P^T * ID^T as ComposedLinearOperator (ID = IdentityOperator)."""
        return self._IT

    @property
    def I0T(self):
        """Transposed inter-/histopolation matrix B * E * I^T * P^T * B^T as ComposedLinearOperator."""
        return self._I0T

    @property
    def pc(self):
        """Preconditioner P * I^(-1) * E^T for iterative polar projections."""
        return self._pc

    @property
    def pc0(self):
        """Preconditioner B * P * I^(-1) * E^T * B^T for iterative polar projections."""
        return self._pc0

    @property
    def pcT(self):
        """Transposed preconditioner P * I^(-T) * E^T for iterative polar projections."""
        return self._pcT

    @property
    def pc0T(self):
        """Transposed preconditioner B * P * I^(-T) * E^T * B^T for iterative polar projections."""
        return self._pc0T

    def solve(self, rhs, transposed=False, apply_bc=False, out=None, x0=None):
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
                    self._polar_solver0T.set_options(x0=x0)
                    x = self._polar_solver0T.solve(
                        self._boundary_op.T.dot(rhs),
                        out=out,
                    )
                else:
                    self._polar_solverT.set_options(x0=x0)
                    x = self._polar_solverT.solve(
                        self._boundary_op.T.dot(rhs),
                        out=out,
                    )
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
                    self._polar_solver0.set_options(x0=x0)
                    x = self._polar_solver0.solve(
                        self._boundary_op.T.dot(rhs),
                        out=out,
                    )
                else:
                    self._polar_solver.set_options(x0=x0)
                    x = self._polar_solver.solve(
                        self._boundary_op.T.dot(rhs),
                        out=out,
                    )
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
                self.projector_tensor(fun, dofs_only=True),
            )
        else:
            self.dofs_extraction_op.dot(
                self.projector_tensor(fun, dofs_only=True),
                out=dofs,
            )

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
        return self.solve(
            self.get_dofs(fun, dofs=dofs, apply_bc=apply_bc),
            transposed=False,
            apply_bc=apply_bc,
            out=out,
        )


class CommutingProjectorLocal:
    r"""
    A commuting projector of the 3d :class:`~struphy.feec.psydac_derham.Derham` diagram,
    based on local quasi-inter/histopolation.

    We shall describe the algortihm by means of 1d inter- and histopolation, 
    which is then combined to give the 3d projections.

    For interpolation, given a knot vector :math:`\hat{T} = \{ \eta_i \}_{0 \leq i \leq n+2p}`, 
    we perform the following steps to obtain the i-th coefficient :math:`\lambda_i(f)` 
    of the quasi-interpolant 

    .. math::

        I^p f := \sum_{i=0}^{\hat{n}_N -1} \lambda_i(f) N_i^p\,.

    1. For :math:`i` fixed, choose :math:`\nu - \mu +p` equidistant interpolation points :math:`\{ x^i_j \}_{0 \leq j < 2p -1}` in the sub-interval :math:`Q = [\eta_\mu , \eta_\nu]` given by:

       * Clamped: 

       .. math:: 

            Q = \left\{\begin{array}{lr}
            [\eta_p = 0, \eta_{p+1}], & i = 0 \,,\\
            {[\eta_p = 0, \eta_{p+i}]}, & 0 < i < p-1\,,\\
            {[\eta_{i+1}, \eta_{i+p}]}, & p-1 \leq i \leq \hat{n}_N - p\,,\\
            {[\eta_{i+1}, \eta_{\hat{n}_N} = 1]}, &  \hat{n}_N - p < i < \hat{n}_N -1\,,\\
            {[\eta_{\hat{n}_N -1}, \eta_{\hat{n}_N} = 1]}, & i = \hat{n}_N -1 \,.
            \end{array} \; \right .

       * Periodic: 

       .. math::

            Q = [\eta_{i + 1}, \eta_{i + p}] \:\:\:\:\: \forall \:\: i.

       * In the periodic case the point set :math:`\{ x^i_j \}_{0 \leq j < 2p -1}` is then the union of the :math:`p` knots in :math:`Q` plus their :math:`p-1` mid-points.

    2. Determine the "local coefficients" :math:`(f_k)_{k=\mu-p}^{\nu-1} \in \mathbb R^{\nu - \mu +p}` by solving
    the local interpolation problem

    .. math::

        \sum_{k = \mu - p}^{\nu -1} f_k N^p_k(x^i_j) = f(x^i_j),\qquad \forall j \in \{0, ..., \nu - \mu +p -1\} .


    3. Set :math:`\lambda_i(f) = f_i`.

    Solving the local interpolation problem in step 2 means that :math:`\lambda_i(f)` can be written as a 
    linear combination of :math:`f(x^i_j)`

    .. math::

        \lambda_i(f) = \sum_{j=0}^{\nu - \mu +p-1}\omega^i_j f(x^i_j)\,,

    where :math:`\omega^i` is the :math:`i`-th line of the inverse collocation matrix 
    :math:`\omega = C^{-1} \in \mathbb R^{(\nu - \mu +p)\times (\nu - \mu +p)}` with :math:`C_{jk} = N^p_k(x^i_j)`. 

    On the other hand, the histopolation operator is defined by

    .. math::

        H^{p-1}f := \sum_{i=0}^{\hat{n}_N -1} \tilde{\lambda}_i(f) D^{p-1}_i\,.

    For the periodic case the FEEC coefficients :math:`\tilde{\lambda}_i(f)` are computed with

    .. math::

        \tilde{\lambda}_i(f) = \sum_{j=0}^{2p-1}\tilde{\omega}^i_j \int_{x^i_j}^{x^i_{j+1}}f(t)dt,
        
    with the weights given by

    .. math::

        \tilde{\omega}^i_j = \left\{\begin{array}{lr}
        \omega^i_0, & j=0 \,, \\
        \omega^i_0 + \omega^i_1, & j = 1 \,, \\
        \sum_{q=0}^{j}\omega^i_q - \sum_{q=0}^{j-2}\omega^{i+1}_q, & j = 2,...,2p-2 \,, \\
        \sum_{q=0}^{2p-2}\omega^i_q - \sum_{q=0}^{2p-3}\omega^{i+1}_q, &  j= 2p-1 \,.
        \end{array} \; \right . 

    For the clamped case the FEEC coefficients :math:`\tilde{\lambda}_i(f)` are computed with

    .. math::

        \tilde{\lambda}_i(f) = \sum_{j=0}^{4p-5}\tilde{\omega}^i_j \int_{x^i_j}^{x^i_{j+1}}f(t)dt,
    
    if :math:`i=0, \hat{n}_N -2`, the weights are given by

    .. math::

        \tilde{\omega}^i_j = \left\{\begin{array}{lr}
        \sum_{q=0}^j (\omega^i_q - \omega^{i+1}_q), & j=0,...,p-1 \,, \\
        0, &  j= p, ..., 4p-5 \,.
        \end{array} \; \right . 
        
    For :math:`0<i<p-1`, they are obtained from
    
    .. math::
        \tilde{\omega}^i_j = \left\{\begin{array}{lr}
        -\sum_{k=j+1}^{p+i-1} \omega^i_k, & j\leq p+i-2 \,, \\
        0, & j = p+i-1 \,, \\
        \sum_{k=j-p-i+1}^{p+i}\omega^{i+1}_k, &  p+i \leq j \leq 2p+2i-1 \,, \\
        0, & 2p+2i-1<j \leq 4p-5 \,.
        \end{array} \; \right .
        
    For :math:`p-1 \leq i < \hat{n}_N-p`, we use
    
    .. math::
        \tilde{\omega}^i_j = \left\{\begin{array}{lr}
        \omega^i_0, & j=0 \,, \\
        \omega^i_0 + \omega^i_1, & j = 1 \,, \\
        \sum_{q=0}^{j}\omega^i_q - \sum_{q=0}^{j-2}\omega^{i+1}_q, & j = 2,...,2p-2 \,, \\
        \sum_{q=0}^{2p-2}\omega^i_q - \sum_{q=0}^{2p-3}\omega^{i+1}_q, &  j= 2p-1 \,, \\
        0, & j= 2p, ..., 4p-5 \,.
        \end{array} \; \right .
        
    Finally, for :math:`\hat{n}_N-p\leq i < \hat{n}_N-2`, we have
    
    .. math::
        \tilde{\omega}^i_j = \left\{\begin{array}{lr}
        \sum_{k=0}^{j}\omega^i_k, & j\leq \hat{n}_N+p-i-3 \,, \\
        0, & j = \hat{n}_N+p-i-2 \,, \\
        -\sum_{k=0}^{j-\hat{n}_N-p+i+1}\omega^{i+1}_k, &  \hat{n}_N+p-i-1 \leq j \leq 2\hat{n}_N+2p-2i-5 \,, \\
        0, & j= 2\hat{n}_N+2p-2i-4, ..., 4p-5 \,.
        \end{array} \; \right .

    Furthermore, in the particular case :math:`p=1`, the weights are given by

    .. math::

        \tilde{\omega}^i_j = \omega^i_0 , \:\:\:\: j=0,1.

    The integration points :math:`\{x^i_j\}` are the same as the quasi-interpolation points. 
    Except in the case :math:`p=1, n=1` with periodic boundary conditions, in which they are :math:`\{0,0.5,1\}`. 


    Parameters:
    -----------
    space_id : str
        One of "H1", "Hcurl", "Hdiv", "L2" or "H1vec".

    space_key : str
        One of "0", "1", "2", "3" or "v".

    fem_space : FemSpace
        FEEC space into which the functions shall be projected.

    pts : list of xp.array
        3-list (or nested 3-list[3-list] for BlockVectors) of 2D arrays with the quasi-interpolation points 
        (or Gauss-Legendre quadrature points for histopolation). 
        In format [spatial direction](B-spline index, point) for StencilVector spaces 
        or [vector component][spatial direction](B-spline index, point) for BlockVector spaces.

    wts : list of xp.array
        3D (4D for BlockVectors) list of 2D array with the Gauss-Legendre quadrature weights 
        (full of ones for interpolation). 
        In format [spatial direction](B-spline index, point) for StencilVector spaces 
        or [vector component][spatial direction](B-spline index, point) for BlockVector spaces.

    wij : list of xp.array
        List of 2D arrays for the coefficients :math:`\omega_j^i` obtained by inverting the local collocation matrix. 
        Use for obtaining the FE coefficients of a function via interpolation. 
        In format [spatial direction](B-spline index, point).

    whij : list of xp.array
        List of 2D arrays for the coefficients :math:`\hat{\omega}_j^i` obtained from the :math:`\omega_j^i`. 
        Use for obtaining the FE coefficients of a function via histopolation. 
        In format [spatial direction](D-spline index, point).

    fem_space_B : TensorFemSpace
        FEEC space for the zero forms. 

    fem_space_D : TensorFemSpace
        FEEC space for the three forms.
    """

    def __init__(
        self,
        space_id: str,
        space_key: str,
        fem_space: FemSpace,
        pts: list,
        wts: list,
        wij: list,
        whij: list,
        fem_space_B: TensorFemSpace,
        fem_space_D: TensorFemSpace,
    ):
        assert space_id in ("H1", "Hcurl", "Hdiv", "L2", "H1vec")

        self._space_id = space_id
        self._space_key = space_key

        # I need to transform the space_key into an int so I can pass it to pyccel kernels as an input, since it does not support strings.
        if space_key == "v":
            self._space_key_int = 4
        else:
            self._space_key_int = int(space_key)

        self._fem_space = fem_space
        # codomain
        self._coeff_space = fem_space.coeff_space

        self._pts = pts
        self._wts = wts
        self._wij = wij
        self._whij = whij

        # FE space of zero forms. That means that we have B-splines in all three spatial directions.
        Bspaces_1d = [fem_space_B.spaces]
        self._B_nbasis = xp.array([space.nbasis for space in Bspaces_1d[0]])

        # Degree of the B-spline space, not to be confused with the degrees given by fem_space.spaces.degree since depending on the situation it will give the D-spline degree instead
        self._p = xp.zeros(3, dtype=int)
        for i, space in enumerate(fem_space_B.spaces):
            self._p[i] = space.degree

        # FE space of three forms. That means that we have D-splines in all three spatial directions.
        Dspaces_1d = [fem_space_D.spaces]
        D_nbasis = xp.array([space.nbasis for space in Dspaces_1d[0]])

        self._periodic = []

        for space in fem_space.spaces:
            self._periodic.append(space.periodic)
        self._periodic = xp.array(self._periodic)

        if isinstance(fem_space, TensorFemSpace):
            # The comm, rank and size are only necessary for debugging. In particular, for printing stuff
            self._comm = self._coeff_space.cart.comm
            if self._comm is None:
                self._rank = 0
                self._size = 1
            else:
                self._rank = self._comm.Get_rank()
                self._size = self._comm.Get_size()

            # We get the start and endpoint for each sublist in out
            self._starts = xp.array(self.coeff_space.starts)
            self._ends = xp.array(self.coeff_space.ends)

            # We compute the number of FE coefficients the current MPI rank is responsible for
            self._loc_num_coeff = xp.array([self._ends[i] + 1 - self._starts[i] for i in range(3)], dtype=int)

            # We get the pads
            self._pds = xp.array(self.coeff_space.pads)
            # We get the number of spaces we have
            self._nsp = 1

            self._localpts = []
            self._index_translation = []
            self._inv_index_translation = []
            self._original_pts_size = xp.zeros((3), dtype=int)

        elif isinstance(fem_space, VectorFemSpace):
            # The comm, rank and size are only necessary for debugging. In particular, for printing stuff
            self._comm = self._coeff_space.spaces[0].cart.comm
            if self._comm is None:
                self._rank = 0
                self._size = 1
            else:
                self._rank = self._comm.Get_rank()
                self._size = self._comm.Get_size()

            # we collect all starts and ends in two big lists
            self._starts = xp.array([vi.starts for vi in self.coeff_space.spaces])
            self._ends = xp.array([vi.ends for vi in self.coeff_space.spaces])

            # We compute the number of FE coefficients the current MPI rank is responsible for
            self._loc_num_coeff = xp.array(
                [[self._ends[h][i] + 1 - self._starts[h][i] for i in range(3)] for h in range(3)],
                dtype=int,
            )

            # We collect the pads
            self._pds = xp.array([vi.pads for vi in self.coeff_space.spaces])
            # We get the number of space we have
            self._nsp = len(self.coeff_space.spaces)

            # We define a list in which we shall append the index_translation for each block direction
            self._index_translation = [[], [], []]
            # We define a list in which we shall append the inv_index_translation for each block direction
            self._inv_index_translation = [[], [], []]

            # We define a list in which we shall append the meshgrid for each block direction
            self._meshgrid = []

            # We define a list in which we shall append the local_pts for each block direction
            self._localpts = [[], [], []]

            # Here we will store the global number of points for each block entry and for each spatial direction.
            self._original_pts_size = [xp.zeros((3), dtype=int), xp.zeros((3), dtype=int), xp.zeros((3), dtype=int)]

            # This will be a list of three elements (the first one for the first block element, the second one for the second block element, ...), each one being a list with three arrays,
            # each array will contain the B-spline indices of the corresponding spatial direction for which this MPI rank has to store at least one non-zero FE coefficient for the storage of the
            # BasisProjectionOperator
            self._Basis_functions_indices_block_B = []
            # Same as above but for D-splines
            self._Basis_functions_indices_block_D = []
            # Each list contains three dictionaries (one per block entry), look for translation_indices_B_or_D_splines_0 to know what each one is
            self._translation_indices_block_B_or_D_splines = [[], [], []]
            # Each list contains three dictionaries (one per block entry), look for values_B_or_D_splines_0 to know what each one is.
            self._values_block_B_or_D_splines = [[], [], []]
            # Each list contains three dictionaries (one per block entry), look for rows_B_or_D_splines_0 to know what each one is.
            self._rows_block_B_or_D_splines = [[], [], []]
            # Each list contains three dictionaries (one per block entry), look for rowe_B_or_D_splines_0 to know what each one is.
            self._rowe_block_B_or_D_splines = [[], [], []]
            # Each list contains three dictionaries (one per block entry), look for are_zero_B_or_D_splines_0 to know what each one is.
            self._are_zero_block_B_or_D_splines = [[], [], []]

            # self._Basis_function_indices_agreggated_B[i][j] = -1 if the jth B-spline is not necessary for any of the three block entries in the ith spatial direction, otherwise it is 0
            self._Basis_function_indices_agreggated_B = [-1 * xp.ones(nbasis, dtype=int) for nbasis in self._B_nbasis]
            self._Basis_function_indices_agreggated_D = [-1 * xp.ones(nbasis, dtype=int) for nbasis in D_nbasis]

            # List that will contain the LocalProjectorsArguments for each value of h = 0,1,2.
            self._solve_args = []
        else:
            raise TypeError(f"{fem_space =} is not of type FemSpace.")

        if isinstance(fem_space, TensorFemSpace):
            if space_id == "H1":
                # List of list that tell us for each spatial direction whether we have Interpolation or Histopolation.
                IoH_for_indices = ["I", "I", "I"]
                # Same list as before but with bools instead of chars
                self._IoH = xp.array([False, False, False], dtype=bool)
                # We make a list with the interpolation/histopolation weights we need for each block and each direction.
                self._geo_weights = [self._wij[0], self._wij[1], self._wij[2]]

            elif space_id == "L2":
                IoH_for_indices = ["H", "H", "H"]
                self._IoH = xp.array([True, True, True], dtype=bool)
                self._geo_weights = [self._whij[0], self._whij[1], self._whij[2]]

            lenj1, lenj2, lenj3 = get_local_problem_size(self._periodic, self._p, self._IoH)

            lenj = [lenj1, lenj2, lenj3]

            self._shift = xp.array([0, 0, 0], dtype=int)
            compute_shifts(self._IoH, self._p, self._B_nbasis, self._shift)

            split_points(
                IoH_for_indices,
                lenj,
                self._shift,
                self._pts,
                self._starts,
                self._ends,
                self._p,
                self._B_nbasis,
                self._periodic,
                self._wij,
                self._whij,
                self._localpts,
                self._original_pts_size,
                self._index_translation,
                self._inv_index_translation,
            )

            # We want to build the meshgrid for the evaluation of the degrees of freedom so it only contains the evaluation points that each specific MPI rank is actually going to use.
            self._meshgrid = xp.meshgrid(
                *[pt for pt in self._localpts],
                indexing="ij",
            )

            # We intialize the arguments for the solve method
            self._solve_args = LocalProjectorsArguments(
                self._space_key_int,
                self._IoH,
                self._shift,
                self._original_pts_size,
                self._index_translation[0],
                self._index_translation[1],
                self._index_translation[2],
                self._starts,
                self._ends,
                self._pds,
                self._B_nbasis,
                self._periodic,
                self._p,
                self._geo_weights[0],
                self._geo_weights[1],
                self._geo_weights[2],
                self._wts[0],
                self._wts[1],
                self._wts[2],
                self._inv_index_translation[0],
                self._inv_index_translation[1],
                self._inv_index_translation[2],
            )

            #####
            # The following is only necesary for building BasisProjectionOperators
            #####

            # To facilitate the construction of BasisProjectionOperators we want to evaluate all B-splines and D-splines that are not zero at these points so we can use this information at the time of assembling the
            # BasisProjectionOperator.
            # The FE coefficients that belong to our current MPI rank tell us which rows of the StencilMatrix representing the BasisProjectionOperator will belong to this MPI rank. On top of that due to the local support
            # of B and D splines only a handful of columns will be nonzero; that is, only a handful of basis functions must be evaluated for each row. In the following code lines we write down the indices of the basis
            # functions this MPI rank will need to assemble the BasisprojectionOperator.

            # This will be a list with three arrays, each array will contain the B-spline indices of the corresponding spatial direction for which this MPI rank has to store at least one non-zero FE coefficient for the storage of the
            # BasisProjectionOperator
            self._Basis_functions_indices_B = []

            get_non_zero_B_spline_indices(
                self._periodic,
                IoH_for_indices,
                self._p,
                self._B_nbasis,
                self._starts,
                self._ends,
                self._Basis_functions_indices_B,
            )

            # Now let us get the D-spline indices for which an MPI rank has non-zeros

            self._Basis_functions_indices_D = []

            get_non_zero_D_spline_indices(
                self._periodic,
                IoH_for_indices,
                self._p,
                D_nbasis,
                self._starts,
                self._ends,
                self._Basis_functions_indices_D,
            )

            # We also need an index translation list that given the Basis function index tell us at which entry of self._Basis_functions_indices_B(or D) that index is found.
            (
                self._translation_indices_B_or_D_splines_0,
                self._translation_indices_B_or_D_splines_1,
                self._translation_indices_B_or_D_splines_2,
            ) = build_translation_list_for_non_zero_spline_indices(
                self._B_nbasis,
                D_nbasis,
                self._Basis_functions_indices_B,
                self._Basis_functions_indices_D,
                space_id,
            )

            # Now that we know which B and D-splines will be needed for the MPI rank to assemble its portion of the BasisProjectionOperator we must evaluate each one of them over all our evaluation
            # points. Luckily we already have all the evaluation points this MPI rank needs stored in self._localpts. Let us build three dictionaries with two lists in which we store the values of each
            # relevant B and D-spline at each relevant evaluation point.
            self._values_B_or_D_splines_0, self._values_B_or_D_splines_1, self._values_B_or_D_splines_2 = (
                evaluate_relevant_splines_at_relevant_points(
                    self._localpts,
                    Bspaces_1d,
                    Dspaces_1d,
                    self._Basis_functions_indices_B,
                    self._Basis_functions_indices_D,
                )
            )

            # We want to know exaclty for which rows (between starts and ends) each basis function found in self._Basis_functions_indices_B(D) are going to produce a non-zero entry in the BasisProjectionOperatorLocal
            # so we can save a significant amount of computations in the solve method.
            # In the following dictionaries we have two entries, one for B and one for D-splines, each entry is a list of arrays. For instance in self._rows_B_or_D_splines_0["B"]
            # the i-th array represents the start indices of rows for which the B-spline with index self._Basis_functions_indices_B[0][i] produces non-zero entries in the BasisProjectionOperatorLocal,
            # while in self._rowe_B_or_D_splines_0["B"] the ith array represents the end indices of rows.
            (
                self._rows_B_or_D_splines_0,
                self._rows_B_or_D_splines_1,
                self._rows_B_or_D_splines_2,
                self._rowe_B_or_D_splines_0,
                self._rowe_B_or_D_splines_1,
                self._rowe_B_or_D_splines_2,
            ) = determine_non_zero_rows_for_each_spline(
                self._Basis_functions_indices_B,
                self._Basis_functions_indices_D,
                self._starts,
                self._ends,
                self._p,
                self._B_nbasis,
                D_nbasis,
                self._periodic,
                self._IoH,
            )

            if space_id == "L2":
                # Finally we want a list where each entry shall tell us if a given B or D spline evaluated at all Gauss-Legandre quadrature points that definfe one histopolation interval
                # are zero. In this way we can skip them during the dofs evaluation.
                self._are_zero_B_or_D_splines_0, self._are_zero_B_or_D_splines_1, self._are_zero_B_or_D_splines_2 = (
                    is_spline_zero_at_quadrature_points(
                        self._Basis_functions_indices_B,
                        self._Basis_functions_indices_D,
                        self._localpts,
                        self._p,
                        [self._values_B_or_D_splines_0, self._values_B_or_D_splines_1, self._values_B_or_D_splines_2],
                        [
                            self._translation_indices_B_or_D_splines_0,
                            self._translation_indices_B_or_D_splines_1,
                            self._translation_indices_B_or_D_splines_2,
                        ],
                        self._IoH,
                    )
                )

        elif isinstance(fem_space, VectorFemSpace):
            self._shift = [xp.array([0, 0, 0], dtype=int) for _ in range(3)]
            if space_id == "H1vec":
                # List of list that tell us for each block entry and for each spatial direction whether we have Interpolation or Histopolation.
                IoH_for_indices = [["I", "I", "I"], ["I", "I", "I"], ["I", "I", "I"]]
                # Same list as before but with bools instead of chars
                self._IoH = [
                    xp.array([False, False, False], dtype=bool),
                    xp.array(
                        [False, False, False],
                        dtype=bool,
                    ),
                    xp.array([False, False, False], dtype=bool),
                ]
                # We make a list with the interpolation/histopolation weights we need for each block and each direction.
                self._geo_weights = [[self._wij[0], self._wij[1], self._wij[2]] for _ in range(3)]

            elif space_id == "Hcurl":
                IoH_for_indices = [["H", "I", "I"], ["I", "H", "I"], ["I", "I", "H"]]
                self._IoH = [
                    xp.array([True, False, False], dtype=bool),
                    xp.array(
                        [False, True, False],
                        dtype=bool,
                    ),
                    xp.array([False, False, True], dtype=bool),
                ]
                self._geo_weights = [
                    [self._whij[0], self._wij[1], self._wij[2]],
                    [
                        self._wij[0],
                        self._whij[1],
                        self._wij[2],
                    ],
                    [self._wij[0], self._wij[1], self._whij[2]],
                ]

            elif space_id == "Hdiv":
                IoH_for_indices = [["I", "H", "H"], ["H", "I", "H"], ["H", "H", "I"]]
                self._IoH = [
                    xp.array([False, True, True], dtype=bool),
                    xp.array(
                        [True, False, True],
                        dtype=bool,
                    ),
                    xp.array([True, True, False], dtype=bool),
                ]
                self._geo_weights = [
                    [self._wij[0], self._whij[1], self._whij[2]],
                    [
                        self._whij[0],
                        self._wij[1],
                        self._whij[2],
                    ],
                    [self._whij[0], self._whij[1], self._wij[2]],
                ]

            for h in range(self._nsp):
                lenj1, lenj2, lenj3 = get_local_problem_size(self._periodic[0], self._p, self._IoH[h])

                lenj = [lenj1, lenj2, lenj3]

                compute_shifts(self._IoH[h], self._p, self._B_nbasis, self._shift[h])

                split_points(
                    IoH_for_indices[h],
                    lenj,
                    self._shift[h],
                    self._pts[h],
                    self._starts[h],
                    self._ends[h],
                    self._p,
                    self._B_nbasis,
                    self._periodic[0],
                    self._wij,
                    self._whij,
                    self._localpts[h],
                    self._original_pts_size[h],
                    self._index_translation[h],
                    self._inv_index_translation[h],
                )

                # meshgrid for h component
                self._meshgrid.append(
                    xp.meshgrid(
                        *[pt for pt in self._localpts[h]],
                        indexing="ij",
                    ),
                )

                # We intialize the arguments for the solve method
                self._solve_args.append(
                    LocalProjectorsArguments(
                        self._space_key_int,
                        self._IoH[h],
                        self._shift[h],
                        self._original_pts_size[h],
                        self._index_translation[h][0],
                        self._index_translation[h][1],
                        self._index_translation[h][2],
                        self._starts[h],
                        self._ends[h],
                        self._pds[h],
                        self._B_nbasis,
                        self._periodic[0],
                        self._p,
                        self._geo_weights[h][0],
                        self._geo_weights[h][1],
                        self._geo_weights[h][2],
                        self._wts[h][0],
                        self._wts[h][1],
                        self._wts[h][2],
                        self._inv_index_translation[h][0],
                        self._inv_index_translation[h][1],
                        self._inv_index_translation[h][2],
                    ),
                )

                #####
                # The following is only necesary for building BasisProjectionOperators
                #####

                # To facilitate the construction of BasisProjectionOperators we want to evaluate all B-splines and D-splines that produce non-zero entries in the BasisProjectionOperatorLocal matrix.
                # The FE coefficients that belong to our current MPI rank tell us which rows of the StencilMatrix representing the BasisProjectionOperator will belong to this MPI rank. On top of that
                # due to the local support of B and D splines only a handful of columns will be nonzero; that is, only a handful of basis functions must be evaluated for each row. In the following
                # code lines we write down the indices of the basis functions this MPI rank will need to assemble the BasisprojectionOperator.

                # This will be a list with three arrays, each array will contain the B-spline indices of the corresponding spatial direction for which this MPI rank has to store at least one non-zero
                # FE coefficient for the storage of the BasisProjectionOperator

                self._Basis_functions_indices_B = []
                get_non_zero_B_spline_indices(
                    self._periodic[h],
                    IoH_for_indices[h],
                    self._p,
                    self._B_nbasis,
                    self._starts[h],
                    self._ends[h],
                    self._Basis_functions_indices_B,
                )
                self._Basis_functions_indices_block_B.append(
                    self._Basis_functions_indices_B,
                )

                # Now let us get the D-spline indices for which an MPI rank has non-zeros

                self._Basis_functions_indices_D = []
                get_non_zero_D_spline_indices(
                    self._periodic[h],
                    IoH_for_indices[h],
                    self._p,
                    D_nbasis,
                    self._starts[h],
                    self._ends[h],
                    self._Basis_functions_indices_D,
                )
                self._Basis_functions_indices_block_D.append(
                    self._Basis_functions_indices_D,
                )

                # We also need an index translation list that given the Basis function index tell us at which entry of self._Basis_functions_indices_B(or D) that index is found.
                (
                    self._translation_indices_B_or_D_splines_0,
                    self._translation_indices_B_or_D_splines_1,
                    self._translation_indices_B_or_D_splines_2,
                ) = build_translation_list_for_non_zero_spline_indices(
                    self._B_nbasis,
                    D_nbasis,
                    self._Basis_functions_indices_B,
                    self._Basis_functions_indices_D,
                    space_id,
                    self._Basis_function_indices_agreggated_B,
                    self._Basis_function_indices_agreggated_D,
                )

                self._translation_indices_block_B_or_D_splines[0].append(
                    self._translation_indices_B_or_D_splines_0,
                )
                self._translation_indices_block_B_or_D_splines[1].append(
                    self._translation_indices_B_or_D_splines_1,
                )
                self._translation_indices_block_B_or_D_splines[2].append(
                    self._translation_indices_B_or_D_splines_2,
                )

                # Now that we know which B and D-splines will be needed for the MPI rank to assemble its portion of the BasisProjectionOperator we must evaluate each one of them over all our evaluation points.
                # Luckily we already have all the evaluation points this MPI rank needs store in self._localpts[0]. Let us build a dictionary with two lists in which we store the values of each relevant B and
                # D-spline at each relevant evaluation point.
                self._values_B_or_D_splines_0, self._values_B_or_D_splines_1, self._values_B_or_D_splines_2 = (
                    evaluate_relevant_splines_at_relevant_points(
                        self._localpts[h],
                        Bspaces_1d,
                        Dspaces_1d,
                        self._Basis_functions_indices_B,
                        self._Basis_functions_indices_D,
                    )
                )

                self._values_block_B_or_D_splines[0].append(
                    self._values_B_or_D_splines_0,
                )
                self._values_block_B_or_D_splines[1].append(
                    self._values_B_or_D_splines_1,
                )
                self._values_block_B_or_D_splines[2].append(
                    self._values_B_or_D_splines_2,
                )

                # We want to know exaclty for which rows (between starts and ends) each basis function found in self._Basis_functions_indices_B(D) are going to produce non-zero entry in the BasisProjectionOperatorLocal
                # so we can save a significant amount of computations in the solve method.
                # In the following dictionaries we have two entries, one for B and one for D-splines, each entry is a list of arrays. For instance in self._rows_B_or_D_splines_0["B"]
                # the i-th array represents the start indices of rows for which the B-spline with index self._Basis_functions_indices_B[0][i] produces non-zero entries in the BasisProjectionOperatorLocal,
                # while in self._rowe_B_or_D_splines_0["B"] the ith array represents the end indices of rows.
                (
                    self._rows_B_or_D_splines_0,
                    self._rows_B_or_D_splines_1,
                    self._rows_B_or_D_splines_2,
                    self._rowe_B_or_D_splines_0,
                    self._rowe_B_or_D_splines_1,
                    self._rowe_B_or_D_splines_2,
                ) = determine_non_zero_rows_for_each_spline(
                    self._Basis_functions_indices_B,
                    self._Basis_functions_indices_D,
                    self._starts[h],
                    self._ends[h],
                    self._p,
                    self._B_nbasis,
                    D_nbasis,
                    self._periodic[h],
                    self._IoH[h],
                )

                self._rows_block_B_or_D_splines[0].append(
                    self._rows_B_or_D_splines_0,
                )
                self._rows_block_B_or_D_splines[1].append(
                    self._rows_B_or_D_splines_1,
                )
                self._rows_block_B_or_D_splines[2].append(
                    self._rows_B_or_D_splines_2,
                )

                self._rowe_block_B_or_D_splines[0].append(
                    self._rowe_B_or_D_splines_0,
                )
                self._rowe_block_B_or_D_splines[1].append(
                    self._rowe_B_or_D_splines_1,
                )
                self._rowe_block_B_or_D_splines[2].append(
                    self._rowe_B_or_D_splines_2,
                )

            # Similar to self._Basis_function_indices_agreggated_B but instead of marking the presence of a B-spline index with -1 or 0, we simply put on the list the present B-splines, and
            # skip those who are not present
            self._Basis_function_indices_mark_B, self._Basis_function_indices_mark_D = (
                get_splines_that_are_relevant_for_at_least_one_block(
                    self._Basis_function_indices_agreggated_B,
                    self._Basis_function_indices_agreggated_D,
                )
            )

            if space_id == "Hcurl" or space_id == "Hdiv":
                for h in range(self._nsp):
                    # Finally we want a list where each entry shall tell us if a given B or D spline evaluated at all Gauss-Legandre quadrature points that definfe one histopolation interval
                    # are zero. In this way we can skip them during the dofs evaluation.
                    # For this case we do not need to check if the quadrature points in the direction e1, e2 are all zero, since in this direction we have interpolation, and just reading
                    # the value of the basis function will suffice. Still we create the dictionary and append it to self._are_zero_block_B_or_D_splines[1] for the sake of uniformity.
                    (
                        self._are_zero_B_or_D_splines_0,
                        self._are_zero_B_or_D_splines_1,
                        self._are_zero_B_or_D_splines_2,
                    ) = is_spline_zero_at_quadrature_points(
                        self._Basis_functions_indices_block_B[h],
                        self._Basis_functions_indices_block_D[h],
                        self._localpts[h],
                        self._p,
                        [
                            self._values_block_B_or_D_splines[0][h],
                            self._values_block_B_or_D_splines[1][h],
                            self._values_block_B_or_D_splines[2][h],
                        ],
                        [
                            self._translation_indices_block_B_or_D_splines[0][h],
                            self._translation_indices_block_B_or_D_splines[1][h],
                            self._translation_indices_block_B_or_D_splines[2][h],
                        ],
                        self._IoH[h],
                    )

                    self._are_zero_block_B_or_D_splines[0].append(
                        self._are_zero_B_or_D_splines_0,
                    )
                    self._are_zero_block_B_or_D_splines[1].append(
                        self._are_zero_B_or_D_splines_1,
                    )
                    self._are_zero_block_B_or_D_splines[2].append(
                        self._are_zero_B_or_D_splines_2,
                    )

    @property
    def space_id(self):
        """The ID of the space (H1, Hcurl, Hdiv, L2 or H1vec)."""
        return self._space_id

    @property
    def space_key(self):
        """The key of the space (0, 1, 2, 3 or v)."""
        return self._space_key

    @property
    def fem_space(self):
        """The Finite Elements spline space"""
        return self._fem_space

    @property
    def coeff_space(self):
        """The vector space underlying the FEM space."""
        return self._coeff_space

    @property
    def pts(self):
        """3D (4D for BlockVectors) list of 2D array with the quasi-interpolation points (or Gauss-Legendre quadrature points for histopolation). In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces or (nv,ns, nb, np) = (vector entry,spatial direction, B-spline index, point) for BlockVector spaces."""
        return self._pts

    @property
    def wts(self):
        """3D (4D for BlockVectors) list of 2D array with the Gauss-Legendre quadrature points (full of ones for interpolation). In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces or (nv,ns, nb, np) = (vector entry,spatial direction, B-spline index, point) for BlockVector spaces."""
        return self._wts

    @property
    def wij(self):
        r"""List of 2D arrays for the coefficients :math:`\omega_j^i` obtained by inverting the local collocation matrix. Use for obtaining the FE coefficients of a function via interpolation. In format (ns, nb, np) = (spatial direction, B-spline index, point)."""
        return self._wij

    @property
    def whij(self):
        r"""List of 2D arrays for the coefficients :math:`\hat{\omega}_j^i` obtained from the :math:`\omega_j^i`. Use for obtaining the FE coefficients of a function via histopolation. In format (ns, nb, np) = (spatial direction, D-spline index, point)."""
        return self._whij

    def solve(self, rhs, out=None):
        """
        Solves

        Parameters
        ----------
        rhs : numpy array
            The right-hand side of the linear system.

        out : psydac.linalg.basic.vector, optional
            If given, the result will be written into this vector in-place.

        Returns
        -------
        out : psydac.linalg.basic.vector
            Output vector (result of linear system).
        """
        if isinstance(self._fem_space, TensorFemSpace):
            if out is None:
                out = self.coeff_space.zeros()
            else:
                assert isinstance(out, StencilVector)

            solve_local_main_loop(self._solve_args, rhs, out._data)

            # Finally we update the ghost regions
            out.update_ghost_regions()

        elif isinstance(self._fem_space, VectorFemSpace):
            if out is None:
                out = self.coeff_space.zeros()
            else:
                assert isinstance(out, BlockVector)

            for h in range(3):
                solve_local_main_loop(self._solve_args[h], rhs[h], out[h]._data)

            # Finally we update the ghost regions
            for h in range(self._nsp):
                out[h].update_ghost_regions()

        return out

    def solve_weighted(self, rhs, out=None):
        """
        Solves

        Parameters
        ----------
        rhs : numpy array
            The right-hand side of the linear system.

        out : numpy array, optional
            If given, the result will be written into this array in-place.

        Returns
        -------
        out : numpy array
            3d numpy array containing the output vector (result of linear system), only the portion that corresponds to the current MPI rank.
        """

        if isinstance(self._fem_space, TensorFemSpace):
            if out is None:
                out = xp.zeros((self._loc_num_coeff[0], self._loc_num_coeff[1], self._loc_num_coeff[2]), dtype=float)
            else:
                assert xp.shape(out) == (self._loc_num_coeff[0], self._loc_num_coeff[1], self._loc_num_coeff[2])

            solve_local_main_loop_weighted(
                self._solve_args,
                rhs,
                self.get_rowstarts(0),
                self.get_rowstarts(1),
                self.get_rowstarts(2),
                self.get_rowends(0),
                self.get_rowends(1),
                self.get_rowends(2),
                out,
                self.get_values(0),
                self.get_values(1),
                self.get_values(2),
            )

        elif isinstance(self._fem_space, VectorFemSpace):
            if out is None:
                out = []
                for h in range(3):
                    out.append(
                        xp.zeros(
                            (
                                self._loc_num_coeff[h][0],
                                self._loc_num_coeff[h][1],
                                self._loc_num_coeff[h][2],
                            ),
                            dtype=float,
                        ),
                    )

            else:
                assert len(out) == 3
                for h in range(3):
                    assert xp.shape(out[h]) == (
                        self._loc_num_coeff[h][0],
                        self._loc_num_coeff[h][1],
                        self._loc_num_coeff[h][2],
                    )

                # In this case for the solve_local_main_loop_weighted to function properly we must make sure before hand to set to zero
                # the out block for which do_nothing tell us before hand they shall be zero.
                for h in range(3):
                    if self._do_nothing[h] == 1:
                        out[h] = xp.zeros(
                            (
                                self._loc_num_coeff[h][0],
                                self._loc_num_coeff[h][1],
                                self._loc_num_coeff[h][2],
                            ),
                            dtype=float,
                        )

            for h in range(3):
                if self._do_nothing[h] == 0:
                    solve_local_main_loop_weighted(
                        self._solve_args[h],
                        rhs[h],
                        self.get_rowstarts(
                            0,
                            h,
                        ),
                        self.get_rowstarts(1, h),
                        self.get_rowstarts(2, h),
                        self.get_rowends(0, h),
                        self.get_rowends(
                            1,
                            h,
                        ),
                        self.get_rowends(2, h),
                        out[h],
                        self.get_values(0, h),
                        self.get_values(1, h),
                        self.get_values(2, h),
                    )

        return out

    def get_dofs(self, fun, dofs=None):
        """
        Builds 3D numpy array with the evaluation of the right-hand-side
        """
        if self._space_key == "0":
            f_eval = fun(*self._meshgrid)

        elif self._space_key == "1" or self._space_key == "2":
            f_eval = []
            for h in range(3):
                # Case in which fun is one function with three outputs.
                if callable(fun):
                    # Evaluation of the function to compute the h component
                    fh = fun(*self._meshgrid[h])[h]
                # Case in which fun is a list of three functions, each one with one output.
                else:
                    assert len(fun) == 3, f"List input only for vector-valued spaces of size 3, but {len(fun) =}."
                    # Evaluation of the function to compute the h component
                    fh = fun[h](*self._meshgrid[h])

                # Array into which we will write the Dofs.
                f_eval_aux = xp.zeros(tuple(xp.shape(dim)[0] for dim in self._localpts[h]))

                # For 1-forms
                if self._space_key == "1":
                    get_dofs_local_1_form_ec_component(self._solve_args[h], fh, f_eval_aux, h)
                # For 2-forms
                else:
                    get_dofs_local_2_form_ec_component(self._solve_args[h], fh, f_eval_aux, h)

                f_eval.append(f_eval_aux)

        elif self._space_key == "3":
            f_eval = xp.zeros(tuple(xp.shape(dim)[0] for dim in self._localpts))
            # Evaluation of the function at all Gauss-Legendre quadrature points
            faux = fun(*self._meshgrid)
            get_dofs_local_3_form(self._solve_args, faux, f_eval)

        elif self._space_key == "v":
            f_eval = []
            # Case in which fun is one function with three outputs.
            if callable(fun):
                for h in range(3):
                    f0, f1, f2 = fun(*self._meshgrid[h])
                    f_eval.append((f0, f1, f2)[h])
            # Case in which fun is a list of three functions, each one with one output.
            else:
                assert (
                    len(
                        fun,
                    )
                    == 3
                ), f"List input only for vector-valued spaces of size 3, but {len(fun) =}."
                for h in range(3):
                    f_eval.append(fun[h](*self._meshgrid[h]))

        else:
            raise Exception(
                "Uknown space. It must be either H1, Hcurl, Hdiv, L2 or H1vec.",
            )

        return f_eval

    def get_dofs_weighted(self, fun, dofs=None, first_go=True, pre_computed_dofs=None):
        """
        Builds 3D numpy array with the evaluation of the right-hand-side.
        """
        if self._space_key == "0":
            if first_go:
                pre_computed_dofs = [fun(*self._meshgrid)]

        elif self._space_key == "1" or self._space_key == "2":
            assert len(fun) == 3, f"List input only for vector-valued spaces of size 3, but {len(fun) =}."

            self._do_nothing = xp.zeros(3, dtype=int)
            f_eval = []

            # If this is the first time this rank has to evaluate the weights degrees of freedom we declare the list where to store them.
            if first_go:
                pre_computed_dofs = []

            for h in range(3):
                # Evaluation of the function to compute the h component
                if first_go:
                    pre_computed_dofs.append(fun[h](*self._meshgrid[h]))

                # Array into which we will write the Dofs.
                f_eval_aux = xp.zeros(tuple(xp.shape(dim)[0] for dim in self._localpts[h]))

                # We check if the current set of basis functions is not one of those we have to compute in the current MPI rank.
                if (
                    self.get_translation_b(0, h) == -1
                    or self.get_translation_b(1, h) == -1
                    or self.get_translation_b(2, h) == -1
                ):
                    # We should do nothing here
                    self._do_nothing[h] = 1
                elif self._space_key == "1":
                    get_dofs_local_1_form_ec_component_weighted(
                        self._solve_args[h],
                        pre_computed_dofs[h],
                        self.get_values(
                            0,
                            h,
                        ),
                        self.get_values(1, h),
                        self.get_values(2, h),
                        self.get_are_zero(h, h),
                        f_eval_aux,
                        h,
                    )
                else:
                    # ind1 and ind2 are the indices of the two directions with histopolation, ind1 must be smaller than ind2.
                    (ind1, ind2) = [(1, 2), (0, 2), (0, 1)][h]
                    get_dofs_local_2_form_ec_component_weighted(
                        self._solve_args[h],
                        pre_computed_dofs[h],
                        self.get_values(
                            0,
                            h,
                        ),
                        self.get_values(1, h),
                        self.get_values(2, h),
                        self.get_are_zero(ind1, h),
                        self.get_are_zero(ind2, h),
                        f_eval_aux,
                        h,
                    )

                f_eval.append(f_eval_aux)

        elif self._space_key == "3":
            f_eval = xp.zeros(tuple(xp.shape(dim)[0] for dim in self._localpts))
            # Evaluation of the function at all Gauss-Legendre quadrature points
            if first_go:
                pre_computed_dofs = [fun(*self._meshgrid)]

            get_dofs_local_3_form_weighted(
                self._solve_args,
                pre_computed_dofs[0],
                self.get_values(0),
                self.get_values(
                    1,
                ),
                self.get_values(2),
                self.get_are_zero(0),
                self.get_are_zero(1),
                self.get_are_zero(2),
                f_eval,
            )

        elif self._space_key == "v":
            assert len(fun) == 3, f"List input only for vector-valued spaces of size 3, but {len(fun) =}."

            self._do_nothing = xp.zeros(3, dtype=int)
            for h in range(3):
                # We check if the current set of basis functions is not one of those we have to compute in the current MPI rank.
                if (
                    self.get_translation_b(0, h) == -1
                    or self.get_translation_b(1, h) == -1
                    or self.get_translation_b(2, h) == -1
                ):
                    # We should do nothing here
                    self._do_nothing[h] = 1

            if first_go:
                f_eval = []
                for h in range(3):
                    f_eval.append(fun[h](*self._meshgrid[h]))

        else:
            raise Exception(
                "Uknown space. It must be either H1, Hcurl, Hdiv, L2 or H1vec.",
            )

        if first_go:
            if self._space_key == "0":
                return pre_computed_dofs[0], pre_computed_dofs
            elif self._space_key == "v":
                return f_eval, f_eval
            else:
                return f_eval, pre_computed_dofs
        else:
            if self._space_key == "0":
                return pre_computed_dofs[0]
            elif self._space_key == "v":
                return pre_computed_dofs
            else:
                return f_eval

    def __call__(
        self,
        fun,
        out=None,
        dofs=None,
        weighted=False,
        B_or_D=None,
        basis_indices=None,
        first_go=True,
        pre_computed_dofs=None,
    ):
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

        weighted : bool
            Determines whether the function to be projected should be multiplied by some B or D-splines.
            Should only be used when assembling BasisprojectionOperators.

        B_or_D : list
            List with three strings, each one can be either "B" or "D". They determine if we have B or D-splines in
            each spatial direction for the weighting of the function with the basis functions.

        basis_indices : list
            List with three ints. They determine the index of each basis function with which we are weighting the function.

        first_go : bool
            This parameter is only useful for assembling BasisProjectionOperatorsLocal, that is, only if weighted is set to True.
            If this parameter is true it means we are computing the projection with this particular weights for the first time, if
            set to false it means we computed it once already and we can reuse the dofs evaluation of the weights instead of
            recomputing them.

        pre_computed_dofs : list of xp.arrays
            If we have already computed the evaluation of the weights at the dofs we can pass the arrays with their values here, so
            we do not have to compute them again.

        Returns
        -------
        coeffs : psydac.linalg.basic.vector | xp.array 3D
            The FEM spline coefficients after projection.
        """
        if not weighted:
            return self.solve(self.get_dofs(fun, dofs=dofs), out=out)
        else:
            # We set B_or_D and basis_indices as attributes of the projectors so we can easily access them in the get_rowstarts, get_rowends and get_values functions, where they are needed.
            self._B_or_D = B_or_D
            self._basis_indices = basis_indices

            if first_go:
                # rhs contains the evaluation over the degrees of freedom of the weights multiplied by the basis function
                # rhs_weights contains the evaluation over the degrees of freedom of only the weights
                rhs, rhs_weights = self.get_dofs_weighted(
                    fun,
                    dofs=dofs,
                    first_go=first_go,
                )
                return self.solve_weighted(rhs, out=out), rhs_weights
            else:
                return self.solve_weighted(
                    self.get_dofs_weighted(fun, dofs=dofs, first_go=False, pre_computed_dofs=pre_computed_dofs),
                    out=out,
                )

    def get_translation_b(self, i, h):
        """
        Selects the correct translation index value. The only real functionality of this function is to make the code easier to read by hiding from the user the
        intricate (but necessary) way of accessing this data.

        Parameters
        ----------
        i : int
            Determines for which of three spatial directions we want to get the translation index

        h : int
            Only for BlockVector spaces, determines the blockvector index.

        Returns
        -------
        self._translation_indices_block_B_or_D_splines[i][h][self._B_or_D[i]][self._basis_indices[i]] : int
            index of self._Basis_functions_indices_B(or D) where the B(or D) spline with the label self._basis_indices[i] is stored. This applies for the first i-th spatial direction
            and the h blockvector entry.

        """
        return self._translation_indices_block_B_or_D_splines[i][h][self._B_or_D[i]][self._basis_indices[i]]

    def get_rowstarts(self, i, h=None):
        """
        Selects the correct rowstarts array. The only real functionality of this function is to make the code easier to read by hiding from the user the
        intricate (but necessary) way of accessing this data.

        Parameters
        ----------
        i : int
            Determines for which of three spatial directions we want to get the rowstarts

        h : int
            Only for BlockVector spaces, determines the blockvector index.

        Returns
        -------
        self._rows_B_or_D_splines_i[self._B_or_D[i]][self._translation_indices_B_or_D_splines_i[self._B_or_D[i]][self._basis_indices[i]]] : 1d int array
            Array that tell us for which rows the basis function in the i-th direction produces non-zero entries in the BasisProjectionOperatorLocal matrix.
            This array contains the start indices of said regions.
        """
        if h == None:
            rows_splines = getattr(self, f"_rows_B_or_D_splines_{i}")
            translation_indices = getattr(self, f"_translation_indices_B_or_D_splines_{i}")
            return rows_splines[self._B_or_D[i]][translation_indices[self._B_or_D[i]][self._basis_indices[i]]]
        else:
            return self._rows_block_B_or_D_splines[i][h][self._B_or_D[i]][
                self._translation_indices_block_B_or_D_splines[i][h][self._B_or_D[i]][self._basis_indices[i]]
            ]

    def get_rowends(self, i, h=None):
        """
        Selects the correct rowends array. The only real functionality of this function is to make the code easier to read by hiding from the user the
        intricate (but necessary) way of accessing this data.

        Parameters
        ----------
        i : int
            intiger that determines for which of three spatial directions we want to get the rowends

        h : int
            Only for BlockVector spaces, determines the blockvector index.

        Returns
        -------
        self._rowe_B_or_D_splines_i[self._B_or_D[i]][self._translation_indices_B_or_D_splines_i[self._B_or_D[i]][self._basis_indices[i]]] : 1d int array
            Array that tell us for which rows the basis function in the i-th direction produces non-zero entries in the BasisProjectionOperatorLocal matrix.
            This array contains the end indices of said regions.
        """
        if h == None:
            rowe_splines = getattr(self, f"_rowe_B_or_D_splines_{i}")
            translation_indices = getattr(self, f"_translation_indices_B_or_D_splines_{i}")
            return rowe_splines[self._B_or_D[i]][translation_indices[self._B_or_D[i]][self._basis_indices[i]]]
        else:
            return self._rowe_block_B_or_D_splines[i][h][self._B_or_D[i]][
                self._translation_indices_block_B_or_D_splines[i][h][self._B_or_D[i]][self._basis_indices[i]]
            ]

    def get_values(self, i, h=None):
        """
        Returns array with the evaluated basis function for the i-th direction. The only real functionality of this function is to make the code easier to read by hiding from the user
        the intricate (but necessary) way of accessing this data.

        Parameters
        ----------
        i : int
            intiger that determines for which of three spatial directions we want to get the values.

        h : int
            Only for BlockVector spaces, determines the blockvector index.

        Returns
        -------
        self._values_B_or_D_splines_i[self._B_or_D[i]][self._translation_indices_B_or_D_splines_i[self._B_or_D[i]][self._basis_indices[i]]] : 1d float array
            Array with the evaluated basis function for the i-th direction.
        """
        if h == None:
            values_splines = getattr(self, f"_values_B_or_D_splines_{i}")
            translation_indices = getattr(self, f"_translation_indices_B_or_D_splines_{i}")
            return values_splines[self._B_or_D[i]][translation_indices[self._B_or_D[i]][self._basis_indices[i]]]
        else:
            return self._values_block_B_or_D_splines[i][h][self._B_or_D[i]][
                self._translation_indices_block_B_or_D_splines[i][h][self._B_or_D[i]][self._basis_indices[i]]
            ]

    def get_are_zero(self, i, h=None):
        """
        Selects the correct are_zero array. The only real functionality of this function is to make the code easier to read by hiding from the user the
        intricate (but necessary) way of accessing this data.

        Parameters
        ----------
        i : int
            intiger that determines for which of three spatial directions we want to get the arezero

        h : int
            Only for BlockVector spaces, determines the blockvector index.

        Returns
        -------
        self._are_zero_B_or_D_splines_i[self._B_or_D[i]][self._translation_indices_B_or_D_splines_i[self._B_or_D[i]][self._basis_indices[i]]] : 1d int array
            Array of zeros or ones. A one at index j means that for the set of quadrature points found in self._localpts[i][j] the basis function is not zero
            for at least one of them.
        """
        if h == None:
            are_zero_splines = getattr(self, f"_are_zero_B_or_D_splines_{i}")
            translation_indices = getattr(self, f"_translation_indices_B_or_D_splines_{i}")
            return are_zero_splines[self._B_or_D[i]][translation_indices[self._B_or_D[i]][self._basis_indices[i]]]
        else:
            return self._are_zero_block_B_or_D_splines[i][h][self._B_or_D[i]][
                self._translation_indices_block_B_or_D_splines[i][h][self._B_or_D[i]][self._basis_indices[i]]
            ]


class L2Projector:
    r"""
    An orthogonal projection into a discrete :class:`~struphy.feec.psydac_derham.Derham` space
    based on the L2-scalar product.

    It solves the following system for the FE-coefficients :math:`\mathbf f = (f_{lmn}) \in \mathbb R^{N_\alpha}`:

    .. math::

        \mathbb M^\alpha_{ijk, lmn} f_{lmn} = (f^\alpha, \Lambda^\alpha_{ijk})_{L^2}\,,

    where :math:`\mathbb M^\alpha` denotes the :ref:`mass matrix <weighted_mass>` of space :math:`\alpha \in \{0,1,2,3,v\}` and :math:`f^\alpha` is a :math:`\alpha`-form proxy function.

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
        from struphy.feec import preconditioner

        assert space_id in ("H1", "Hcurl", "Hdiv", "L2", "H1vec")

        params_default = {
            "type": ("pcg", "MassMatrixPreconditioner"),
            "tol": 1.0e-14,
            "maxiter": 500,
            "info": False,
            "verbose": False,
        }

        set_defaults(params, params_default)

        self._space_id = space_id
        self._mass_ops = mass_ops
        self._params = params
        self._space_key = mass_ops.derham.space_to_form[self.space_id]
        self._space = mass_ops.derham.Vh_fem[self.space_key]

        # mass matrix
        self._Mmat = getattr(self.mass_ops, "M" + self.space_key)

        # quadrature grid
        self._quad_grid_pts = self.mass_ops.derham.quad_grid_pts[self.space_key]

        if space_id in ("H1", "L2"):
            self._quad_grid_mesh = xp.meshgrid(
                *[pt.flatten() for pt in self.quad_grid_pts],
                indexing="ij",
            )
            self._geom_weights = self.Mmat.weights[0][0](*self.quad_grid_mesh)
        else:
            self._quad_grid_mesh = []
            self._tmp = []  # tmp for matrix-vector product of geom_weights with fun
            for pts in self.quad_grid_pts:
                self._quad_grid_mesh += [
                    xp.meshgrid(
                        *[pt.flatten() for pt in pts],
                        indexing="ij",
                    ),
                ]
                self._tmp += [xp.zeros_like(self.quad_grid_mesh[-1][0])]
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
                        self._geom_weights[-1] += [xp.zeros_like(mesh[0])]

        # other quad grid info
        if isinstance(self.space, TensorFemSpace):
            self._tensor_fem_spaces = [self.space]
            self._wts_l = [self.mass_ops.derham.quad_grid_wts[self.space_key]]
            self._spans_l = [
                self.mass_ops.derham.quad_grid_spans[self.space_key],
            ]
            self._bases_l = [
                self.mass_ops.derham.quad_grid_bases[self.space_key],
            ]
        else:
            self._tensor_fem_spaces = self.space.spaces
            self._wts_l = self.mass_ops.derham.quad_grid_wts[self.space_key]
            self._spans_l = self.mass_ops.derham.quad_grid_spans[self.space_key]
            self._bases_l = self.mass_ops.derham.quad_grid_bases[self.space_key]

        # Preconditioner
        if self.params["type"][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.params["type"][1])
            pc = pc_class(self.Mmat)

        # solver
        self._solver = inverse(
            self.Mmat,
            self.params["type"][0],
            pc=pc,
            tol=self.params["tol"],
            maxiter=self.params["maxiter"],
            verbose=self.params["verbose"],
        )

    @property
    def mass_ops(self):
        """Struphy mass operators object, see :ref:`mass_ops`.."""
        return self._mass_ops

    @property
    def space_id(self):
        """The ID of the space (H1, Hcurl, Hdiv, L2 or H1vec)."""
        return self._space_id

    @property
    def space_key(self):
        """The key of the space (0, 1, 2, 3 or v)."""
        return self._space_key

    @property
    def space(self):
        """The Derham finite element space (from ``Derham.Vh_fem``)."""
        return self._space

    @property
    def params(self):
        """Parameters for the iterative solver."""
        return self._params

    @property
    def Mmat(self):
        """The mass matrix of space."""
        return self._Mmat

    @property
    def quad_grid_pts(self):
        """List of quadrature points in each direction for integration over grid cells in format (ni, nq) = (cell, quadrature point)."""
        return self._quad_grid_pts

    @property
    def quad_grid_mesh(self):
        """Mesh grids of quad_grid_pts."""
        return self._quad_grid_mesh

    @property
    def geom_weights(self):
        """Geometric coefficients (e.g. Jacobians) evaluated at quad_grid_mesh, stored as list[list] either 1x1 or 3x3."""
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
            Weight function(s) (callables or xp.ndarrays) in a 1d list of shape corresponding to number of components.

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
        elif isinstance(fun, xp.ndarray):
            assert fun.shape == self._quad_grid_mesh[0].shape, (
                f"Expected shape {self._quad_grid_mesh[0].shape}, got {fun.shape =} instead."
            )
            fun_weights = fun
        else:
            assert (
                len(
                    fun,
                )
                == 3
            ), f"List input only for vector-valued spaces of size 3, but {len(fun) =}."
            fun_weights = []
            # loop over rows (different meshes)
            for mesh in self._quad_grid_mesh:
                fun_weights += [[]]
                # loop over columns (different functions)
                for f in fun:
                    if callable(f):
                        fun_weights[-1] += [f(*mesh)]
                    elif isinstance(f, xp.ndarray):
                        assert f.shape == mesh[0].shape, f"Expected shape {mesh[0].shape}, got {f.shape =} instead."
                        fun_weights[-1] += [f]
                    else:
                        raise ValueError(
                            f"Expected callable or numpy array, got {type(f) =} instead.",
                        )

        # check output vector
        if dofs is None:
            dofs = self.space.coeff_space.zeros()
        else:
            assert isinstance(dofs, (StencilVector, BlockVector, PolarVector))
            assert dofs.space == self.Mmat.codomain

        # compute matrix data for kernel, i.e. fun * geom_weight
        tot_weights = []
        if isinstance(fun_weights, xp.ndarray):
            tot_weights += [fun_weights * self.geom_weights]
        else:
            # loop over rows (differnt meshes)
            for row_fun, row_geom, tmp in zip(fun_weights, self.geom_weights, self._tmp):
                tmp *= 0.0
                # loop over columns (different functions)
                for fun_weight, geom_weight in zip(row_fun, row_geom):
                    # matrix-vector product
                    tmp += fun_weight * geom_weight
                tot_weights += [tmp]

        # clear data
        if clear:
            if isinstance(dofs, StencilVector):
                dofs._data[:] = 0.0
            elif isinstance(dofs, PolarVector):
                dofs.tp._data[:] = 0.0
            else:
                for block in dofs.blocks:
                    block._data[:] = 0.0

        # loop over components (just one for scalar spaces)
        for a, (fem_space, spans, wts, basis, mat_w) in enumerate(
            zip(
                self._tensor_fem_spaces,
                self._spans_l,
                self._wts_l,
                self._bases_l,
                tot_weights,
            ),
        ):
            # indices
            starts = [int(start) for start in fem_space.coeff_space.starts]
            pads = fem_space.coeff_space.pads

            if isinstance(dofs, StencilVector):
                mass_kernels.kernel_3d_vec(
                    *spans,
                    *fem_space.degree,
                    *starts,
                    *pads,
                    *wts,
                    *basis,
                    mat_w,
                    dofs._data,
                )
            elif isinstance(dofs, PolarVector):
                mass_kernels.kernel_3d_vec(
                    *spans,
                    *fem_space.degree,
                    *starts,
                    *pads,
                    *wts,
                    *basis,
                    mat_w,
                    dofs.tp._data,
                )
            else:
                mass_kernels.kernel_3d_vec(
                    *spans,
                    *fem_space.degree,
                    *starts,
                    *pads,
                    *wts,
                    *basis,
                    mat_w,
                    dofs[a]._data,
                )

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


class ProjectorPreconditioner(LinearOperator):
    r"""
    Preconditioner for approximately inverting a (polar) 3d inter-/histopolation matrix via

    .. math::

        (B * P * I * E^T * B^T)^{-1} \approx B * P * I^{-1} * E^T * B^T.

    In case that $P$ and $E$ are identity operators, the solution is exact (pure tensor product case).

    Parameters
    ----------
    projector : CommutingProjector
        The global commuting projector for which the inter-/histopolation matrix shall be inverted.

    transposed : bool, optional
        Whether to invert the transposed inter-/histopolation matrix.

    apply_bc : bool, optional
        Whether to include the boundary operators.
    """

    def __init__(self, projector, transposed=False, apply_bc=False):
        # vector space in tensor product case/polar case
        self._space = projector.I.domain

        self._codomain = projector.I.codomain

        self._dtype = projector.I.dtype

        self._projector = projector

        self._apply_bc = apply_bc

        # save Kronecker solver (needed in solve method)
        self._solver = projector.projector_tensor.solver
        if transposed:
            self._solver = self.solver.transpose()

        self._transposed = transposed

        # save inter-/histopolation matrix to be inverted
        if transposed:
            self._I = projector.IT
        else:
            self._I = projector.I

        self._is_composed = isinstance(self._I, ComposedLinearOperator)

        # temporary vectors for dot product
        if self._is_composed:
            tmp_vectors = []
            for op in self._I.multiplicants[1:]:
                tmp_vectors.append(op.codomain.zeros())

            self._tmp_vectors = tuple(tmp_vectors)
        else:
            self._tmp_vector = self._I.codomain.zeros()

    @property
    def space(self):
        """Stencil-/BlockVectorSpace or PolarDerhamSpace."""
        return self._space

    @property
    def solver(self):
        """KroneckerLinearSolver for exactly inverting tensor product inter-histopolation matrix."""
        return self._solver

    @property
    def transposed(self):
        """Whether to invert the transposed inter-/histopolation matrix."""
        return self._transposed

    @property
    def domain(self):
        """The domain of the linear operator - an element of Vectorspace"""
        return self._space

    @property
    def codomain(self):
        """The codomain of the linear operator - an element of Vectorspace"""
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    def tosparse(self):
        raise NotImplementedError()

    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return ProjectorPreconditioner(self._projector, True, self._apply_bc)

    def solve(self, rhs, out=None):
        """
        Computes (B * P * I^(-1) * E^T * B^T) * rhs, resp. (B * P * I^(-T) * E^T * B^T) * rhs (transposed=True) as an approximation for an inverse inter-/histopolation matrix.

        Parameters
        ----------
        rhs : psydac.linalg.basic.Vector
            The right-hand side vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output vector will be written into this vector in-place.

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The result of (B * E * M^(-1) * E^T * B^T) * rhs, resp. (B * P * I^(-T) * E^T * B^T) * rhs (transposed=True).
        """

        assert isinstance(rhs, Vector)
        assert rhs.space == self._space

        # successive dot products with all but last operator
        if self._is_composed:
            x = rhs
            for i in range(len(self._tmp_vectors)):
                y = self._tmp_vectors[-1 - i]
                A = self._I.multiplicants[-1 - i]
                if isinstance(A, (StencilMatrix, KroneckerStencilMatrix, BlockLinearOperator)):
                    self.solver.dot(x, out=y)
                else:
                    A.dot(x, out=y)
                x = y

            # last operator
            A = self._I.multiplicants[0]
            if out is None:
                out = A.dot(x)
            else:
                assert isinstance(out, Vector)
                assert out.space == self._space
                A.dot(x, out=out)

        else:
            if out is None:
                out = self.solver.dot(rhs)
            self.solver.dot(rhs, out=out)
        return out

    def dot(self, v, out=None):
        """Apply linear operator to Vector v. Result is written to Vector out, if provided."""

        assert isinstance(v, Vector)
        assert v.space == self.domain

        # newly created output vector
        if out is None:
            out = self.solve(v)

        # in-place dot-product (result is written to out)
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
            self.solve(v, out=out)

        return out
