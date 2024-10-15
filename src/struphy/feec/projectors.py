from psydac.linalg.stencil import StencilVector
from psydac.linalg.block import BlockVector, BlockLinearOperator
from psydac.linalg.kron import KroneckerStencilMatrix
from psydac.linalg.basic import Vector, IdentityOperator
from psydac.linalg.solvers import inverse
from psydac.fem.tensor import TensorFemSpace
from psydac.feec.global_projectors import GlobalProjector
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.polar.linear_operators import PolarExtractionOperator
from struphy.feec import preconditioner
from struphy.feec.preconditioner import ProjectorPreconditioner
from struphy.feec import mass_kernels
from struphy.fields_background.mhd_equil.equils import set_defaults
from struphy.feec.basis_projection_kernels import get_dofs_local_1_form_e1_component, get_dofs_local_1_form_e2_component, get_dofs_local_1_form_e3_component, solve_local_0_form, solve_local_1_form, get_dofs_local_2_form_e1_component, get_dofs_local_2_form_e2_component, get_dofs_local_2_form_e3_component, solve_local_2_form, solve_local_3_form, solve_local_0V_form, get_dofs_local_3_form
from struphy.polar.basic import PolarVector

import numpy as np
import time


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
    * :math:`\mathcal I`: Kronecker product inter-/histopolation matrix, from :class:`~psydac.feec.global_projectors.GlobalProjector`
    * :math:`\mathbb E`: :class:`~struphy.polar.linear_operators.PolarExtractionOperator` for FE coefficients.

    :math:`\mathbb P` and :math:`\mathbb E` (and :math:`\mathbb B` in case of no boundary conditions) can be identity operators, 
    which gives the pure tensor-product Psydac :class:`~psydac.feec.global_projectors.GlobalProjector`.

    Parameters
    ----------
    projector_tensor : GlobalProjector
        The pure tensor product projector.

    dofs_extraction_op : PolarExtractionOperator, optional
        The degrees of freedom extraction operator mapping tensor product DOFs to polar DOFs. If not given, is set to identity.

    base_extraction_op : PolarExtractionOperator, optional
        The basis extraction operator mapping tensor product basis functions to polar basis functions. If not given, is set to identity.

    boundary_op : BoundaryOperator
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

    1. For :math:`i` fixed, choose :math:`2p - 1` equidistant interpolation points :math:`\{ x^i_j \}_{0 \leq j < 2p -1}` in the sub-interval :math:`Q = [\eta_\mu , \eta_\nu]` given by:

       * Clamped: 

       .. math:: 

            Q = \left\{\begin{array}{lr}
            [\eta_p, \eta_{2p -1}], & i < p-1\\
            {[\eta_{i+1}, \eta_{i+p}]}, & p-1 \leq i \leq \hat{n}_N - p\\
            {[\eta_{\hat{n}_N - p +1}, \eta_{\hat{n}_N}]}, &  i > \hat{n}_N - p
            \end{array} \; \right .

       * Periodic: 

       .. math::

            Q = [\eta_{i + 1}, \eta_{i + p}] \:\:\:\:\: \forall \:\: i.

       * The point set :math:`\{ x^i_j \}_{0 \leq j < 2p -1}` is then the union of the :math:`p` knots in :math:`Q` plus their :math:`p-1` mid-points.

    2. Determine the "local coefficients" :math:`(f_k)_{k=\mu-p}^{\nu-1} \in \mathbb R^{2p-1}` by solving
    the local interpolation problem

    .. math::

        \sum_{k = \mu - p}^{\nu -1} f_k N^p_k(x^i_j) = f(x^i_j),\qquad \forall j \in \{0, ..., 2p -2\} .


    3. Set :math:`\lambda_i(f) = f_i`.

    Solving the local interpolation problem in step 2 means that :math:`\lambda_i(f)` can be written as a 
    linear combination of :math:`f(x^i_j)`

    .. math::

        \lambda_i(f) = \sum_{j=0}^{2p-2}\omega^i_j f(x^i_j)\,,

    where :math:`\omega^i` is the :math:`i`-th line of the inverse collocation matrix 
    :math:`\omega = C^{-1} \in \mathbb R^{(2p-1)\times (2p-1)}` with :math:`C_{jk} = N^p_k(x^i_j)`. 

    On the other hand, the histopolation operator is defined by

    .. math::

        H^{p-1}f := \sum_{i=0}^{\hat{n}_N -1} \tilde{\lambda}_i(f) D^{p-1}_i\,.

    The FEEC coefficients :math:`\tilde{\lambda}_i(f)` are computed with

    .. math::

        \tilde{\lambda}_i(f) = \sum_{j=0}^{2p-1}\tilde{\omega}^i_j \int_{x^i_j}^{x^i_{j+1}}f(t)dt,

    In the the clamped case, if :math:`i<p-1` 
    or :math:`i \geq \hat{n}_D -(p-1)`, the weights are given by

    .. math::

        \tilde{\omega}^i_j = \left\{\begin{array}{lr}
        \sum_{q=0}^{j}(\omega^i_q - \omega^{i+1}_q) , & j=0,...,2p-3\\
        0, &  j= 2p-2, 2p-1
        \end{array} \; \right . 

    In the periodic case, and in the clamped case with  :math:`p-1\leq i< \hat{n}_D -(p-1)`, 
    the weights given by

    .. math::

        \tilde{\omega}^i_j = \left\{\begin{array}{lr}
        \omega^i_0, & j=0\\
        \omega^i_0 + \omega^i_1, & j = 1\\
        \sum_{q=0}^{j}\omega^i_q - \sum_{q=0}^{j-2}\omega^{i+1}_q, & j = 2,...,2p-2\\
        \sum_{q=0}^{2p-2}\omega^i_q - \sum_{q=0}^{2p-3}\omega^{i+1}_q, &  j= 2p-1
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

    fem_space : VectorFemSpace
        FEEC space onto which the functions shall be projected

    pts : list of np.array
        3D (4D for BlockVectors) list of 2D array with the quasi-interpolation points 
        (or Gauss-Legendre quadrature points for histopolation). 
        In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces 
        or (nv,ns, nb, np) = (vector entry,spatial direction, B-spline index, point) for BlockVector spaces.

    wts : list
        3D (4D for BlockVectors) list of 2D array with the Gauss-Legendre quadrature points 
        (full of ones for interpolation). 
        In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces 
        or (nv,ns, nb, np) = (vector entry,spatial direction, B-spline index, point) for BlockVector spaces.

    wij : list of np.array
        List of 2D arrays for the coefficients :math:`\omega_j^i` obtained by inverting the local collocation matrix. 
        Use for obtaining the FE coefficients of a function via interpolation. 
        In format (ns, nb, np) = (spatial direction, B-spline index, point).

    whij : list of np.array
        List of 2D arrays for the coefficients :math:`\hat{\omega}_j^i` obtained from the :math:`\omega_j^i`. 
        Use for obtaining the FE coefficients of a function via histopolation. 
        In format (ns, nb, np) = (spatial direction, D-spline index, point).
    """

    def __init__(self, space_id, space_key, fem_space, pts, wts, wij, whij):

        assert space_id in ('H1', 'Hcurl', 'Hdiv', 'L2', 'H1vec')
        self._space_id = space_id
        self._space_key = space_key
        self._fem_space = fem_space
        self._pts = pts
        self._wts = wts
        self._wij = wij
        self._whij = whij

        self._domain = self._fem_space.vector_space

        if (self._space_key == "0" or self._space_key == "3"):
            comm = self._domain.cart.comm
            rank = comm.Get_rank()
            size = comm.Get_size()
            # We get the start and endpoint for each sublist in out
            self._starts = np.array(self._domain.starts)
            self._ends = np.array(self._domain.ends)
            # We get the dimensions of the StencilVector
            self._npts = np.array(self._domain.npts)
            # We get the pads
            self._pds = np.array(self._domain.pads)
            # We get the number of spaces we have
            self._nsp = 1
        elif (self._space_key == "1" or self._space_key == "2" or self._space_key == "v"):
            comm = self._domain.spaces[0].cart.comm
            rank = comm.Get_rank()
            size = comm.Get_size()
            # we collect all starts and ends in two big lists
            self._starts = np.array([vi.starts for vi in self._domain.spaces])
            self._ends = np.array([vi.ends for vi in self._domain.spaces])
            # We collect the pads
            self._pds = np.array([vi.pads for vi in self._domain.spaces])
            # We collect the dimension of the BlockVector
            self._npts = np.array([sp.npts for sp in self._domain.spaces])
            # We get the number of space we have
            self._nsp = len(self._domain.spaces)

        # Degree of the B-spline space, not to be confused with the degrees given by fem_space.spaces.degree since depending on the situation it will give the D-spline degree instead
        self._p = []
        self._periodic = []

        for space in fem_space.spaces:
            self._periodic.append(space.periodic)
        self._periodic = np.array(self._periodic)
        if space_id == 'H1':
            for space in fem_space.spaces:
                self._p.append(space.degree)
            # We want to build the meshgrid for the evaluation of the degrees of freedom so it only contains the evaluation points that each specific MPI rank is actually going to use.

            self._localpts = []
            self._index_translation = []
            self._original_pts_size = []

            lenj1 = 2*self._p[0]-1
            lenj2 = 2*self._p[1]-1
            lenj3 = 2*self._p[2]-1

            lenj = [lenj1, lenj2, lenj3]

            shift1 = - 2*self._npts[0]
            shift2 = - 2*self._npts[1]
            shift3 = - 2*self._npts[2]

            shift = [shift1, shift2, shift3]

            npts_split = [self._npts[0], self._npts[1], self._npts[2]]

            BoS = "S"
            IoH = ["I", "I", "I"]
            split_points(BoS, IoH, 0, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                         self._periodic, [], self._localpts, self._original_pts_size, self._index_translation)

            self._meshgrid = np.meshgrid(
                *[pt for pt in self._localpts], indexing='ij')

        elif space_id == 'H1vec':
            for n, space in enumerate(fem_space.spaces):
                if n == 0:
                    self._p = space.degree

            lenj1 = 2*self._p[0]-1
            lenj2 = 2*self._p[1]-1
            lenj3 = 2*self._p[2]-1

            lenj = [lenj1, lenj2, lenj3]

            shift1 = - 2*self._npts[0][0]
            shift2 = - 2*self._npts[0][1]
            shift3 = - 2*self._npts[0][2]

            shift = [shift1, shift2, shift3]

            npts_split = [self._npts[0][0], self._npts[0][1], self._npts[0][2]]

            BoS = "B"
            IoH = ["I", "I", "I"]

            for h in range(self._nsp):
                if (h == 0):

                    self._localptsx = []
                    self._index_translationx = []
                    self._original_pts_sizex = np.zeros((3), dtype=int)

                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split, self._periodic, [
                    ], self._localptsx, self._original_pts_sizex, self._index_translationx)
                    # meshgrid for x component
                    self._meshgridx = np.meshgrid(
                        *[pt for pt in self._localptsx], indexing='ij')

                elif (h == 1):
                    self._localptsy = []
                    self._index_translationy = []
                    self._original_pts_sizey = np.zeros((3), dtype=int)

                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split, self._periodic, [
                    ], self._localptsy, self._original_pts_sizey, self._index_translationy)

                    # meshgrid for y component
                    self._meshgridy = np.meshgrid(
                        *[pt for pt in self._localptsy], indexing='ij')

                elif (h == 2):
                    self._localptsz = []
                    self._index_translationz = []
                    self._original_pts_sizez = np.zeros((3), dtype=int)

                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split, self._periodic, [
                    ], self._localptsz, self._original_pts_sizez, self._index_translationz)

                    # meshgrid for z component
                    self._meshgridz = np.meshgrid(
                        *[pt for pt in self._localptsz], indexing='ij')

        elif space_id == 'Hcurl':
            for n, space in enumerate(fem_space.spaces):
                if n == 0:
                    self._p = space.degree
            # We need the degree of the B-Splines, since for the x direction Hcurl has a D-Spline on the x direction we need to add 1 to the degree we read from it.
            self._p[0] += 1
            BoS = "B"
            npts_split = [self._npts[1][0], self._npts[0][1], self._npts[0][2]]
            for h in range(self._nsp):
                if (h == 0):

                    self._localptsx = []
                    self._index_translationx = []
                    self._original_pts_sizex = np.zeros((3), dtype=int)

                    lenj1 = 2*self._p[0]
                    lenj2 = 2*self._p[1]-1
                    lenj3 = 2*self._p[2]-1

                    lenj = [lenj1, lenj2, lenj3]

                    # We compute the amout by which we must shift the indices to loop around the quasi-points.
                    if (self._p[0] == 1 and self._npts[1][0] != 1):
                        shift1 = - 2*self._npts[1][0] + 1
                    else:
                        shift1 = - 2*self._npts[1][0]

                    shift2 = - 2*self._npts[0][1]
                    shift3 = - 2*self._npts[0][2]

                    shift = [shift1, shift2, shift3]

                    IoH = ["H", "I", "I"]
                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                                 self._periodic, self._whij, self._localptsx, self._original_pts_sizex, self._index_translationx)

                    # meshgrid for x component
                    self._meshgridx = np.meshgrid(
                        *[pt for pt in self._localptsx], indexing='ij')

                elif (h == 1):
                    self._localptsy = []
                    self._index_translationy = []
                    self._original_pts_sizey = np.zeros((3), dtype=int)

                    lenj1 = 2*self._p[0]-1
                    lenj2 = 2*self._p[1]
                    lenj3 = 2*self._p[2]-1

                    lenj = [lenj1, lenj2, lenj3]

                    # We compute the amout by which we must shift the indices to loop around the quasi-points.
                    if (self._p[1] == 1 and self._npts[0][1] != 1):
                        shift2 = - 2*self._npts[0][1] + 1
                    else:
                        shift2 = - 2*self._npts[0][1]

                    shift1 = - 2 * self._npts[1][0]
                    shift3 = - 2*self._npts[0][2]

                    shift = [shift1, shift2, shift3]

                    IoH = ["I", "H", "I"]
                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                                 self._periodic, self._whij, self._localptsy, self._original_pts_sizey, self._index_translationy)

                    # meshgrid for y component
                    self._meshgridy = np.meshgrid(
                        *[pt for pt in self._localptsy], indexing='ij')

                elif (h == 2):
                    self._localptsz = []
                    self._index_translationz = []
                    self._original_pts_sizez = np.zeros((3), dtype=int)

                    lenj1 = 2*self._p[0]-1
                    lenj2 = 2*self._p[1]-1
                    lenj3 = 2*self._p[2]

                    lenj = [lenj1, lenj2, lenj3]

                    # We compute the amout by which we must shift the indices to loop around the quasi-points.
                    if (self._p[2] == 1 and self._npts[0][2] != 1):
                        shift3 = - 2*self._npts[0][2] + 1
                    else:
                        shift3 = - 2*self._npts[0][2]

                    shift1 = - 2 * self._npts[1][0]
                    shift2 = - 2*self._npts[0][1]

                    shift = [shift1, shift2, shift3]

                    IoH = ["I", "I", "H"]
                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                                 self._periodic, self._whij, self._localptsz, self._original_pts_sizez, self._index_translationz)

                    # meshgrid for z component
                    self._meshgridz = np.meshgrid(
                        *[pt for pt in self._localptsz], indexing='ij')

        elif space_id == 'Hdiv':
            for n, space in enumerate(fem_space.spaces):
                if n == 0:
                    self._p = space.degree
            # We need the degree of the B-Splines, since for the x direction Hdiv has a D-Spline for the y and z directions we need to add 1 to the degrees we read from it.
            self._p[1] += 1
            self._p[2] += 1
            BoS = "B"
            npts_split = [self._npts[0][0], self._npts[1][1], self._npts[2][2]]

            for h in range(self._nsp):
                if (h == 0):

                    self._localptsx = []
                    self._index_translationx = []
                    self._original_pts_sizex = np.zeros((3), dtype=int)

                    lenj1 = 2*self._p[0]-1
                    lenj2 = 2*self._p[1]
                    lenj3 = 2*self._p[2]

                    lenj = [lenj1, lenj2, lenj3]

                    # We compute the amout by which we must shift the indices to loop around the quasi-points.
                    shift1 = - 2*self._npts[0][0]
                    if (self._p[1] == 1 and self._npts[1][1] != 1):
                        shift2 = - 2*self._npts[1][1] + 1
                    else:
                        shift2 = - 2*self._npts[1][1]

                    if (self._p[2] == 1 and self._npts[2][2] != 1):
                        shift3 = - 2*self._npts[2][2] + 1
                    else:
                        shift3 = - 2*self._npts[2][2]

                    shift = [shift1, shift2, shift3]

                    IoH = ["I", "H", "H"]
                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                                 self._periodic, self._whij, self._localptsx, self._original_pts_sizex, self._index_translationx)

                    # meshgrid for x component
                    self._meshgridx = np.meshgrid(
                        *[pt for pt in self._localptsx], indexing='ij')

                elif (h == 1):
                    self._localptsy = []
                    self._index_translationy = []
                    self._original_pts_sizey = np.zeros((3), dtype=int)

                    lenj1 = 2*self._p[0]
                    lenj2 = 2*self._p[1]-1
                    lenj3 = 2*self._p[2]

                    lenj = [lenj1, lenj2, lenj3]

                    # We compute the amout by which we must shift the indices to loop around the quasi-points.

                    if (self._p[0] == 1 and self._npts[0][0] != 1):
                        shift1 = - 2*self._npts[0][0] + 1
                    else:
                        shift1 = - 2*self._npts[0][0]

                    shift2 = - 2*self._npts[1][1]

                    if (self._p[2] == 1 and self._npts[2][2] != 1):
                        shift3 = - 2*self._npts[2][2] + 1
                    else:
                        shift3 = - 2*self._npts[2][2]

                    shift = [shift1, shift2, shift3]

                    IoH = ["H", "I", "H"]
                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                                 self._periodic, self._whij, self._localptsy, self._original_pts_sizey, self._index_translationy)

                    # meshgrid for y component
                    self._meshgridy = np.meshgrid(
                        *[pt for pt in self._localptsy], indexing='ij')

                elif (h == 2):
                    self._localptsz = []
                    self._index_translationz = []
                    self._original_pts_sizez = np.zeros((3), dtype=int)

                    lenj1 = 2*self._p[0]
                    lenj2 = 2*self._p[1]
                    lenj3 = 2*self._p[2]-1

                    lenj = [lenj1, lenj2, lenj3]

                    # We compute the amout by which we must shift the indices to loop around the quasi-points.
                    if (self._p[0] == 1 and self._npts[0][0] != 1):
                        shift1 = - 2*self._npts[0][0] + 1
                    else:
                        shift1 = - 2*self._npts[0][0]

                    if (self._p[1] == 1 and self._npts[1][1] != 1):
                        shift2 = - 2*self._npts[1][1] + 1
                    else:
                        shift2 = - 2*self._npts[1][1]

                    shift3 = - 2*self._npts[2][2]

                    shift = [shift1, shift2, shift3]

                    IoH = ["H", "H", "I"]
                    split_points(BoS, IoH, h, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                                 self._periodic, self._whij, self._localptsz, self._original_pts_sizez, self._index_translationz)

                    # meshgrid for z component
                    self._meshgridz = np.meshgrid(
                        *[pt for pt in self._localptsz], indexing='ij')

            # Tensor product matrix of the Gauss-Legendre quadrature weigths to evaluate the x component of the vector function
            self._GLweightsx = np.tensordot(wts[0][1][0], wts[0][2][0], axes=0)
            # Tensor product matrix of the Gauss-Legendre quadrature weigths to evaluate the y component of the vector function
            self._GLweightsy = np.tensordot(wts[1][0][0], wts[1][2][0], axes=0)
            # Tensor product matrix of the Gauss-Legendre quadrature weigths to evaluate the z component of the vector function
            self._GLweightsz = np.tensordot(wts[2][0][0], wts[2][1][0], axes=0)

        elif space_id == 'L2':
            for space in fem_space.spaces:
                self._p.append(space.degree)
            # We need the degree of the B-Splines, since L2 has a D-Spline for the x, y and z directions we need to add 1 to the degrees we read from it.
            self._p[0] += 1
            self._p[1] += 1
            self._p[2] += 1

            self._localpts = []
            self._index_translation = []
            self._original_pts_size = []

            # We get the number of B-Splines
            if (self._periodic[0]):
                NB0 = self._npts[0]
            else:
                NB0 = self._npts[0]+1
            if (self._periodic[1]):
                NB1 = self._npts[1]
            else:
                NB1 = self._npts[1]+1
            if (self._periodic[2]):
                NB2 = self._npts[2]
            else:
                NB2 = self._npts[2]+1

            # We compute the amout by which we must shift the indices to loop around the quasi-points.
            if (self._p[0] == 1 and NB0 != 1):
                shift1 = - 2*NB0 + 1
            else:
                shift1 = - 2*NB0

            if (self._p[1] == 1 and NB1 != 1):
                shift2 = - 2*NB1 + 1
            else:
                shift2 = - 2*NB1

            if (self._p[2] == 1 and NB2 != 1):
                shift3 = - 2*NB2 + 1
            else:
                shift3 = - 2*NB2

            shift = [shift1, shift2, shift3]

            lenj = [2*self._p[0], 2*self._p[1], 2*self._p[2]]
            BoS = "S"
            IoH = ["H", "H", "H"]

            npts_split = [self._npts[0]+1, self._npts[1]+1, self._npts[2]+1]

            split_points(BoS, IoH, 0, lenj, shift, self._pts, self._starts, self._ends, self._p, npts_split,
                         self._periodic, self._whij, self._localpts, self._original_pts_size, self._index_translation)

            self._meshgrid = np.meshgrid(
                *[pt for pt in self._localpts], indexing='ij')

            # Tensor product matrix of the Gauss-Legendre quadrature weigths to evaluate the function
            self._GLweights = np.tensordot(np.tensordot(
                wts[0][0], wts[1][0], axes=0), wts[2][0], axes=0)

    @property
    def space_id(self):
        """ The ID of the space (H1, Hcurl, Hdiv, L2 or H1vec)."""
        return self._space_id

    @property
    def space_key(self):
        """ The key of the space (0, 1, 2, 3 or v)."""
        return self._space_key

    @property
    def fem_space(self):
        '''The Finite Elements spline space'''
        return self._fem_space

    @property
    def pts(self):
        '''3D (4D for BlockVectors) list of 2D array with the quasi-interpolation points (or Gauss-Legendre quadrature points for histopolation). In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces or (nv,ns, nb, np) = (vector entry,spatial direction, B-spline index, point) for BlockVector spaces.'''
        return self._pts

    @property
    def wts(self):
        '''3D (4D for BlockVectors) list of 2D array with the Gauss-Legendre quadrature points (full of ones for interpolation). In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces or (nv,ns, nb, np) = (vector entry,spatial direction, B-spline index, point) for BlockVector spaces.'''
        return self._wts

    @property
    def wij(self):
        '''List of 2D arrays for the coefficients :math:`\omega_j^i` obtained by inverting the local collocation matrix. Use for obtaining the FE coefficients of a function via interpolation. In format (ns, nb, np) = (spatial direction, B-spline index, point).'''
        return self._wij

    @property
    def whij(self):
        '''List of 2D arrays for the coefficients :math:`\hat{\omega}_j^i` obtained from the :math:`\omega_j^i`. Use for obtaining the FE coefficients of a function via histopolation. In format (ns, nb, np) = (spatial direction, D-spline index, point).'''
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
        # We extract the B-splines'degrees.
        p1 = self._p[0]
        p2 = self._p[1]
        p3 = self._p[2]
        if (self._space_key == "0"):
            if out is None:
                out = self._domain.zeros()
            else:
                assert isinstance(out, StencilVector)

            solve_local_0_form(self._original_pts_size[0], self._original_pts_size[1], self._original_pts_size[2], self._index_translation[0], self._index_translation[1], self._index_translation[2], self._starts, self._ends, self._pds, self._npts,
                               self._periodic, p1, p2, p3, self._wij[0], self._wij[1], self._wij[2], rhs, out._data)
        elif (self._space_key == "1"):
            if out is None:
                out = self._domain.zeros()
            else:
                assert isinstance(out, BlockVector)

            solve_local_1_form(self._original_pts_sizex, self._original_pts_sizey, self._original_pts_sizez, self._index_translationx[0], self._index_translationx[1], self._index_translationx[2], self._index_translationy[0], self._index_translationy[1], self._index_translationy[2], self._index_translationz[0], self._index_translationz[1], self._index_translationz[2], self._nsp, self._starts, self._ends, self._pds, self._npts, self._periodic, p1, p2, p3, self._wij[0],
                               self._wij[1], self._wij[2], self._whij[0], self._whij[1], self._whij[2], rhs[0], rhs[1], rhs[2], out[0]._data, out[1]._data, out[2]._data)

        elif (self._space_key == "2"):
            if out is None:
                out = self._domain.zeros()
            else:
                assert isinstance(out, BlockVector)

            solve_local_2_form(self._original_pts_sizex, self._original_pts_sizey, self._original_pts_sizez, self._index_translationx[0], self._index_translationx[1], self._index_translationx[2], self._index_translationy[0], self._index_translationy[1], self._index_translationy[2], self._index_translationz[0], self._index_translationz[1], self._index_translationz[2], self._nsp, self._starts, self._ends, self._pds, self._npts, self._periodic, p1, p2, p3, self._wij[0],
                               self._wij[1], self._wij[2], self._whij[0], self._whij[1], self._whij[2], rhs[0], rhs[1], rhs[2], out[0]._data, out[1]._data, out[2]._data)

        elif (self._space_key == "3"):
            if out is None:
                out = self._domain.zeros()
            else:
                assert isinstance(out, StencilVector)

            solve_local_3_form(self._original_pts_size[0], self._original_pts_size[1], self._original_pts_size[2], self._index_translation[0], self._index_translation[1], self._index_translation[2], self._starts, self._ends, self._pds, self._npts,
                               self._periodic, p1, p2, p3, self._whij[0], self._whij[1], self._whij[2], rhs, out._data)

        elif (self._space_key == "v"):

            if out is None:
                out = self._domain.zeros()
            else:
                assert isinstance(out, BlockVector)

            solve_local_0V_form(self._original_pts_sizex, self._original_pts_sizey, self._original_pts_sizez, self._index_translationx[0], self._index_translationx[1], self._index_translationx[2], self._index_translationy[0], self._index_translationy[1], self._index_translationy[2], self._index_translationz[0], self._index_translationz[1], self._index_translationz[2], self._nsp, self._starts, self._ends, self._pds, self._npts, self._periodic, p1, p2, p3,
                                self._wij[0], self._wij[1], self._wij[2], rhs[0], rhs[1], rhs[2], out[0]._data, out[1]._data, out[2]._data)

        else:
            raise Exception(
                "Uknown space. It must be either H1, Hcurl, Hdiv, L2 or H1vec.")

        # Finally we update the ghost regions
        if isinstance(out, StencilVector):
            out.update_ghost_regions()
        elif isinstance(out, BlockVector):
            for h in range(self._nsp):
                out[h].update_ghost_regions()

        return out

    def get_dofs(self, fun, dofs=None):
        """
        Builds 3D numpy array with the evaluation of the right-hand-side
        """
        # Erase. I have verified by hand that get_dofs works for "0"
        if (self._space_key == "0"):
            f_eval = fun(*self._meshgrid)

        # Erase. I have verified by hand the correct functioning of get_dofs for "1"
        elif (self._space_key == "1"):
            if callable(fun):
                f_eval = []
                ############################
                # Computing the x component
                ############################

                # Evaluation of the function to compute the x component
                f0, f1, f2 = fun(*self._meshgridx)

                f_eval_aux = np.zeros((np.shape(self._localptsx[0])[0], np.shape(
                    self._localptsx[1])[0], np.shape(self._localptsx[2])[0]))

                get_dofs_local_1_form_e1_component(
                    f0, self._p[0], self._wts[0][0][0], f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the y component
                ############################

                # Evaluation of the function to compute the y component
                f0, f1, f2 = fun(*self._meshgridy)

                f_eval_aux = np.zeros((np.shape(self._localptsy[0])[0], np.shape(
                    self._localptsy[1])[0], np.shape(self._localptsy[2])[0]))

                get_dofs_local_1_form_e2_component(
                    f1, self._p[1], self._wts[1][1][0], f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the z component
                ############################

                # Evaluation of the function to compute the z component
                f0, f1, f2 = fun(*self._meshgridz)

                f_eval_aux = np.zeros((np.shape(self._localptsz[0])[0], np.shape(
                    self._localptsz[1])[0], np.shape(self._localptsz[2])[0]))

                get_dofs_local_1_form_e3_component(
                    f2, self._p[2], self._wts[2][2][0], f_eval_aux)

                f_eval.append(f_eval_aux)
            else:
                assert len(
                    fun) == 3, f'List input only for vector-valued spaces of size 3, but {len(fun) = }.'

                f_eval = []
                ############################
                # Computing the x component
                ############################

                # Evaluation of the function to compute the x component
                f0 = fun[0](*self._meshgridx)

                f_eval_aux = np.zeros((np.shape(self._localptsx[0])[0], np.shape(
                    self._localptsx[1])[0], np.shape(self._localptsx[2])[0]))

                get_dofs_local_1_form_e1_component(
                    f0, self._p[0], self._wts[0][0][0], f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the y component
                ############################

                # Evaluation of the function to compute the y component
                f1 = fun[1](*self._meshgridy)

                f_eval_aux = np.zeros((np.shape(self._localptsy[0])[0], np.shape(
                    self._localptsy[1])[0], np.shape(self._localptsy[2])[0]))

                get_dofs_local_1_form_e2_component(
                    f1, self._p[1], self._wts[1][1][0], f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the z component
                ############################

                # Evaluation of the function to compute the z component
                f2 = fun[2](*self._meshgridz)
                f_eval_aux = np.zeros((np.shape(self._localptsz[0])[0], np.shape(
                    self._localptsz[1])[0], np.shape(self._localptsz[2])[0]))

                get_dofs_local_1_form_e3_component(
                    f2, self._p[2], self._wts[2][2][0], f_eval_aux)

                f_eval.append(f_eval_aux)

        # Erase. I have verified by hand the correct functioning of get_dofs for "2"
        elif (self._space_key == "2"):
            if callable(fun):
                f_eval = []
                ############################
                # Computing the x component
                ############################

                # Evaluation of the function to compute the x component
                f0, f1, f2 = fun(*self._meshgridx)

                f_eval_aux = np.zeros((np.shape(self._localptsx[0])[0], np.shape(
                    self._localptsx[1])[0], np.shape(self._localptsx[2])[0]))

                get_dofs_local_2_form_e1_component(
                    f0, self._p[1], self._p[2], self._GLweightsx, f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the y component
                ############################
                # Evaluation of the function to compute the y component
                f0, f1, f2 = fun(*self._meshgridy)

                f_eval_aux = np.zeros((np.shape(self._localptsy[0])[0], np.shape(
                    self._localptsy[1])[0], np.shape(self._localptsy[2])[0]))

                get_dofs_local_2_form_e2_component(
                    f1, self._p[0], self._p[2], self._GLweightsy, f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the z component
                ############################
                # Evaluation of the function to compute the z component
                f0, f1, f2 = fun(*self._meshgridz)

                f_eval_aux = np.zeros((np.shape(self._localptsz[0])[0], np.shape(
                    self._localptsz[1])[0], np.shape(self._localptsz[2])[0]))

                get_dofs_local_2_form_e3_component(
                    f2, self._p[0], self._p[1], self._GLweightsz, f_eval_aux)

                f_eval.append(f_eval_aux)
            else:
                assert len(
                    fun) == 3, f'List input only for vector-valued spaces of size 3, but {len(fun) = }.'

                f_eval = []
                ############################
                # Computing the x component
                ############################

                # Evaluation of the function to compute the x component
                f0 = fun[0](*self._meshgridx)
                f_eval_aux = np.zeros((np.shape(self._localptsx[0])[0], np.shape(
                    self._localptsx[1])[0], np.shape(self._localptsx[2])[0]))

                get_dofs_local_2_form_e1_component(
                    f0, self._p[1], self._p[2], self._GLweightsx, f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the y component
                ############################
                # Evaluation of the function to compute the y component
                f1 = fun[1](*self._meshgridy)

                f_eval_aux = np.zeros((np.shape(self._localptsy[0])[0], np.shape(
                    self._localptsy[1])[0], np.shape(self._localptsy[2])[0]))

                get_dofs_local_2_form_e2_component(
                    f1, self._p[0], self._p[2], self._GLweightsy, f_eval_aux)

                f_eval.append(f_eval_aux)

                ############################
                # Computing the z component
                ############################
                # Evaluation of the function to compute the z component
                f2 = fun[2](*self._meshgridz)

                f_eval_aux = np.zeros((np.shape(self._localptsz[0])[0], np.shape(
                    self._localptsz[1])[0], np.shape(self._localptsz[2])[0]))

                get_dofs_local_2_form_e3_component(
                    f2, self._p[0], self._p[1], self._GLweightsz, f_eval_aux)

                f_eval.append(f_eval_aux)

        # Erase. I have verified by hand the correct functioning of get_dofs for "3"
        elif (self._space_key == "3"):
            f_eval = np.zeros((np.shape(self._localpts[0])[0], np.shape(
                self._localpts[1])[0], np.shape(self._localpts[2])[0]))
            # Evaluation of the function at all Gauss-Legendre quadrature points
            faux = fun(*self._meshgrid)

            get_dofs_local_3_form(
                faux, self._p[0], self._p[1], self._p[2], self._GLweights, f_eval)

        elif (self._space_key == "v"):
            if callable(fun):
                f_eval = []
                f0, f1, f2 = fun(*self._meshgridx)
                f_eval.append(f0)
                f0, f1, f2 = fun(*self._meshgridy)
                f_eval.append(f1)
                f0, f1, f2 = fun(*self._meshgridz)
                f_eval.append(f2)
            else:
                assert len(
                    fun) == 3, f'List input only for vector-valued spaces of size 3, but {len(fun) = }.'

                f_eval = []
                f_eval.append(fun[0](*self._meshgridx))
                f_eval.append(fun[1](*self._meshgridy))
                f_eval.append(fun[2](*self._meshgridz))

        else:
            raise Exception(
                "Uknown space. It must be either H1, Hcurl, Hdiv, L2 or H1vec.")

        return f_eval

    def __call__(self, fun, out=None, dofs=None):
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

        Returns
        -------
        coeffs : psydac.linalg.basic.vector
            The FEM spline coefficients after projection.
        """
        return self.solve(self.get_dofs(fun, dofs=dofs), out=out)


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
                self._quad_grid_mesh += [np.meshgrid(*[pt.flatten()
                                                     for pt in pts], indexing='ij')]
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
        '''The Derham finite element space (from ``Derham.Vh_fem``).'''
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
            assert isinstance(dofs, (StencilVector, BlockVector, PolarVector))
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
            elif isinstance(dofs, PolarVector):
                dofs.tp._data[:] = 0.
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
            elif isinstance(dofs, PolarVector):
                mass_kernels.kernel_3d_vec(*spans, *fem_space.degree, *starts, *pads,
                                           *wts, *basis, mat_w, dofs.tp._data)
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


# We need a functions that tell us which of the quasi-interpolation points to take for a any given i
def select_quasi_points(i, p, Nbasis, periodic):
    '''Determines the start and end indices of the quasi-interpolation points that must be taken to get the ith FEEC coefficient.

    Parameters
    ----------
    i : int
        Index of the FEEC coefficient that must be computed.

    p : int
        B-spline degree.

    Nbasis: int
        Number of B-spline.

    periodic: bool
        Whether we have periodic boundary conditions.

    Returns
    -------
    offset : int
        Start index of the quasi-interpolation points that must be consider to obtain the ith FEEC coefficient.

    2*p-1+offset : int
        End index of the quasi-interpolation points that must be consider to obtain the ith FEEC coefficient.
    '''
    if periodic:
        return 2*i, int(2*p)-1+2*i
    else:
        # We need the number of elements n, to compute it we substract the B-spline degree from the number of B-splines.
        n = Nbasis-p
        if i >= 0 and i < p-1:
            offset = 0
        elif i >= p-1 and i <= n:
            offset = int(2*(i-p+1))
        elif i > n and i <= n+p-1:
            offset = int(2*(n-p+1))
        else:
            raise Exception("index i must be between 0 and n+p-1")

        return offset, int(2*p)-1+offset

# This function splits the interpolation points and quadrature points between the MPI ranks, in such a way that every rank only gets the points it will need to compute the FE coefficients assigned to it by the
# starts and ends splitting.


def split_points(BoS, IoH, h, lenj, shift, pts, starts, ends, p, npts, periodic, whij, localptsout, original_pts_size, index_translation):
    '''Splits the interpolaton points and quadrature points between the MPI ranks. Making sure that each rank only gets the points it needs to compute the FE coefficients assignes to it.

    Parameters
    ----------
    BoS : string
        Determines if we are working with BlockVectors (B) or StencilVectors (S).

    IoH : list of strings
        Determines if we have Interpolation (I) or Histopolation (H) for each one of the three spatial dimentions.

    h : int
        Only useful for BlockVectors, determines in which one of the three entries we are working on.

    lenj : list of int
        Determines the number of inner itterations we need to run over all values of j for each one of the three spatial directions.

    shifts : list of ints
        For each one of the three spatial directions it determines by which amount to shift the position index (pos) in case we have to loop over the evaluation points.

    pts : list of np.array
        3D (4D for BlockVectors) list of 2D array with the quasi-interpolation points 
        (or Gauss-Legendre quadrature points for histopolation). 
        In format (ns, nb, np) = (spatial direction, B-spline index, point) for StencilVector spaces 
        or (nv,ns, nb, np) = (vector entry,spatial direction, B-spline index, point) for BlockVector spaces.

    starts : 2d (or 1D) int array
        Array with the BlockVector start indices for each MPI rank. Or 1d array with the StencilVector start indices for each MPI rank.

    ends : 2d (or 1D) int array
        2d Array with the BlockVector end indices for each MPI rank. Or 1d array with the StencilVector end indices for each MPI rank.

    p : list of ints
        Contains the B-splines degrees for each one of the three spatial directions.

    npts : list of ints
        Contains the number of B-splines for each one of the three spatial directions.

    periodic : 1D bool np.array
        For each one of the three spatial directions contains the information of whether the B-splines are periodic or not.

    whij: 3d float array
        Array with the histopolation geometric weights for all three directions. In format (spatial directions, i index, j index)

    localptsout : empty list
        Here this function shall write the interpolation or quadrature points that are relevant for the MPI rank.

    original_pts_size : empty list
        Here this function shall write the total number of interpolation points or histopolation integrals for all three spatial diections.

    index_translation1 : empty list
        This function makes sure that this list translates for all three spatial direction from the global indices to the local indices to evaluate the right-hand-side. index_local = index_translation[spatial-direction][index_global]



    '''
    if (BoS == "S"):
        # For this case h is not necessary so we ignore it
        for n, pt in enumerate(pts):
            original_pts_size.append(np.shape(pt)[0])
            if IoH[n] == "I":
                localpts = np.full(
                    (np.shape(pt)[0]), fill_value=-1, dtype=float)
            elif IoH[n] == "H":
                localpts = np.full((np.shape(pt)), fill_value=-1, dtype=float)
            for i in range(starts[n], ends[n]+1):

                startj1, endj1 = select_quasi_points(
                    i, p[n], npts[n], periodic[n])

                for j1 in range(lenj[n]):
                    if (startj1+j1 < np.shape(pt)[0]):
                        pos = startj1+j1
                    else:
                        pos = int(startj1+j1 + shift[n])
                    if IoH[n] == "I":
                        localpts[pos] = pt[pos]
                    elif IoH[n] == "H":
                        if (whij[n][i][j1] != 0.0):
                            localpts[pos] = pt[pos]

            if IoH[n] == "I":
                localpos = np.where(localpts != -1)[0]
            elif IoH[n] == "H":
                localpos = np.where(localpts[:, 0] != -1)[0]
            localpts = localpts[localpos]
            localptsout.append(np.array(localpts))

            mini_indextranslation = np.full(
                (np.shape(pt)[0]), fill_value=-1, dtype=int)
            for i, j in enumerate(localpos):
                mini_indextranslation[j] = i

            index_translation.append(np.array(mini_indextranslation))

    elif (BoS == "B"):
        for n, pt in enumerate(pts[h]):
            original_pts_size[n] = np.shape(pt)[0]
            if IoH[n] == "I":
                localpts = np.full(
                    (np.shape(pt)[0]), fill_value=-1, dtype=float)
            elif IoH[n] == "H":
                localpts = np.full((np.shape(pt)), fill_value=-1, dtype=float)

            for i in range(starts[h][n], ends[h][n]+1):

                startj1, endj1 = select_quasi_points(
                    i, p[n], npts[n], periodic[0][n])

                for j1 in range(lenj[n]):
                    if (startj1+j1 < np.shape(pt)[0]):
                        pos = startj1+j1
                    else:
                        pos = int(startj1+j1 + shift[n])
                    if IoH[n] == "I":
                        localpts[pos] = pt[pos]
                    elif IoH[n] == "H":
                        if (whij[n][i][j1] != 0.0):
                            localpts[pos] = pt[pos]

            if IoH[n] == "I":
                localpos = np.where(localpts != -1)[0]
            elif IoH[n] == "H":
                localpos = np.where(localpts[:, 0] != -1)[0]

            localpts = localpts[localpos]
            localptsout.append(np.array(localpts))

            mini_indextranslation = np.full(
                (np.shape(pt)[0]), fill_value=-1, dtype=int)
            for i, j in enumerate(localpos):
                mini_indextranslation[j] = i

            index_translation.append(
                np.array(mini_indextranslation))
