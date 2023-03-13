import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockLinearOperator
from psydac.linalg.basic import Vector

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api import mass_kernels
from struphy.psydac_api.utilities import RotationMatrix
from struphy.psydac_api.linear_operators import LinOpWithTransp, BoundaryOperator, IdentityOperator
from struphy.psydac_api.linear_operators import CompositeLinearOperator as Compose

from struphy.polar.linear_operators import PolarExtractionOperator


class WeightedMassOperators:
    r"""
    Class for assembling weighted mass matrices in 3d.

    Weighted mass matrices :math:`\mathbb M^{\beta\alpha}: \mathbb R^{N_\alpha} \to \mathbb R^{N_\beta}` are of the general form

    .. math::

        \mathbb M^{\beta \alpha}_{(\mu,ijk),(\nu,mno)} = \int_{[0, 1]^3} \Lambda^\beta_{\mu,ijk} \, W_{\mu,\nu} \, \Lambda^\alpha_{\nu,mno} \, \textnormal d^3 \boldsymbol\eta\,,

    where the weight fuction :math:`W` can be rank 0, 1 or 2, depending on domain and co-domain of the operator, and :math:`\Lambda^\alpha_{\nu, mno}` is the B-spline basis function with tensor-product index :math:`mno` of the
    :math:`\nu`-th component in the space :math:`V^\alpha_h`. These matrices are sparse and stored in StencilMatrix format.

    Parameters
    ----------
    derham : struphy.psydac_api.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.

    domain : :ref:`avail_mappings`
        Mapping from logical unit cube to physical domain and corresponding metric coefficients.

    **weights : dict
        Objects to access callables that can serve as weight functions.

    Notes
    -----
    Possible choices for key-value pairs in ****weights** are, at the moment:

    - eq_mhd: :class:`struphy.fields_background.mhd_equil.base.MHDequilibrium`
    """

    def __init__(self, derham, domain, **weights):

        self._derham = derham
        self._domain = domain
        self._weights = weights

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

    @property
    def derham(self):
        """ Discrete de Rham sequence on the logical unit cube. 
        """
        return self._derham

    @property
    def domain(self):
        """ Mapping from the logical unit cube to the physical domain with corresponding metric coefficients.
        """
        return self._domain

    @property
    def weights(self):
        '''Dictionary of objects that provide access to callables that can serve as weight functions.'''
        return self._weights

    # Wrapper functions for evaluating metric coefficients in right order (3x3 entries are last two axes!!)
    def G(self, e1, e2, e3):
        '''Metric tensor callable.'''
        return self.domain.metric(e1, e2, e3, change_out_order=True, squeeze_out=False)

    def Ginv(self, e1, e2, e3):
        '''Inverse metric tensor callable.'''
        return self.domain.metric_inv(e1, e2, e3, change_out_order=True, squeeze_out=False)

    def sqrt_g(self, e1, e2, e3):
        '''Jacobian determinant callable.'''
        return abs(self.domain.jacobian_det(e1, e2, e3, squeeze_out=False))

    #######################################################################
    # Mass matrices related to L2-scalar products in all 3d derham spaces #
    #######################################################################
    @property
    def M0(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^0_{ijk, mno} = \int \Lambda^0_{ijk}\,  \Lambda^0_{mno} \sqrt g\,  \textnormal d \boldsymbol\eta.
        """

        if not hasattr(self, '_M0'):
            fun = [[lambda e1, e2, e3: self.sqrt_g(e1, e2, e3)]]
            self._M0 = self.assemble_weighted_mass(fun, 'H1', 'H1')

        return self._M0

    @property
    def M1(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^1_{(\mu,ijk), (\nu,mno)} = \int \Lambda^1_{\mu,ijk}\, G^{-1}_{\mu,\nu}\, \Lambda^1_{\nu, mno} \sqrt g\,  \textnormal d \boldsymbol\eta. 
        """

        if not hasattr(self, '_M1'):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.Ginv(e1, e2, e3)[:, :, :, m, n] * self.sqrt_g(e1, e2, e3)]

            self._M1 = self.assemble_weighted_mass(fun, 'Hcurl', 'Hcurl')

        return self._M1

    @property
    def M2(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^2_{(\mu,ijk), (\nu,mno)} = \int \Lambda^2_{\mu,ijk}\, G_{\mu,\nu}\, \Lambda^2_{\nu, mno} \frac{1}{\sqrt g}\,  \textnormal d \boldsymbol\eta. 
        """

        if not hasattr(self, '_M2'):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.G(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._M2 = self.assemble_weighted_mass(fun, 'Hdiv', 'Hdiv')

        return self._M2

    @property
    def M3(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^3_{ijk, mno} = \int \Lambda^3_{ijk}\,  \Lambda^3_{mno} \frac{1}{\sqrt g}\,  \textnormal d \boldsymbol\eta.
        """

        if not hasattr(self, '_M3'):
            fun = [[lambda e1, e2, e3: 1. / self.sqrt_g(e1, e2, e3)]]
            self._M3 = self.assemble_weighted_mass(fun, 'L2', 'L2')

        return self._M3

    @property
    def Mv(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^2_{(\mu,ijk), (\nu,mno)} = \int \Lambda^2_{\mu,ijk}\, G_{\mu,\nu}\, \Lambda^2_{\nu, mno} \sqrt g\,  \textnormal d \boldsymbol\eta. 
        """

        if not hasattr(self, '_Mv'):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.G(e1, e2, e3)[:, :, :, m, n] * self.sqrt_g(e1, e2, e3)]

            self._Mv = self.assemble_weighted_mass(fun, 'H1vec', 'H1vec')

        return self._Mv

    ######################################
    # Predefined weighted mass operators #
    ######################################
    @property
    def M1n(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{1,n}_{(\mu,ijk), (\nu,mno)} = \int n^0_{\textnormal{eq}}(\boldsymbol \eta) \Lambda^1_{\mu,ijk}\, G^{-1}_{\mu,\nu}\, \Lambda^1_{\nu, mno} \sqrt g\,  \textnormal d \boldsymbol\eta. 

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol \eta)` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, '_M1n'):
            assert 'eq_mhd' in self.weights
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.Ginv(e1, e2, e3)[:, :, :, m, n] * self.sqrt_g(
                        e1, e2, e3) * self.weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False)]

            self._M1n = self.assemble_weighted_mass(fun, 'Hcurl', 'Hcurl')

        return self._M1n

    @property
    def M2n(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{2,n}_{(\mu,ijk), (\nu,mno)} = \int n^0_{\textnormal{eq}}(\boldsymbol \eta) \Lambda^2_{\mu,ijk}\, G_{\mu,\nu}\, \Lambda^2_{\nu, mno} \frac{1}{\sqrt g}\,  \textnormal d \boldsymbol\eta. 

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol \eta)` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, '_M2n'):
            assert 'eq_mhd' in self.weights
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.G(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(
                        e1, e2, e3) * self.weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False)]

            self._M2n = self.assemble_weighted_mass(fun, 'Hdiv', 'Hdiv')

        return self._M2n

    @property
    def Mvn(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{v,n}_{(\mu,ijk), (\nu,mno)} = \int n^0_{\textnormal{eq}}(\boldsymbol \eta) \Lambda^v_{\mu,ijk}\, G_{\mu,\nu}\, \Lambda^v_{\nu, mno} \sqrt g\,  \textnormal d \boldsymbol\eta. 

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol \eta)` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, '_Mvn'):
            assert 'eq_mhd' in self.weights
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.G(e1, e2, e3)[:, :, :, m, n] * self.sqrt_g(
                        e1, e2, e3) * self.weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False)]

            self._Mvn = self.assemble_weighted_mass(fun, 'H1vec', 'H1vec')

        return self._Mvn

    @property
    def M1J(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{1,J}_{(\mu,ijk), (\nu,mno)} = \int \Lambda^1_{\mu,ijk}\, G^{-1}_{\mu,\alpha}\, \mathcal R^J_{\alpha, \nu}\, \Lambda^2_{\nu, mno} \,  \textnormal d \boldsymbol\eta. 

        with the rotation matrix

        .. math::

            \mathcal R^J_{\alpha, \nu} := \epsilon_{\alpha \beta \nu}\, J^2_{\textnormal{eq}, \beta}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec J^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium current density (2-form).
        """

        if not hasattr(self, '_M1J'):

            rot_J = RotationMatrix(
                self.weights['eq_mhd'].j2_1, self.weights['eq_mhd'].j2_2, self.weights['eq_mhd'].j2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: (self.Ginv(e1, e2, e3) @ rot_J(e1, e2, e3))[:, :, :, m, n]]

            self._M1J = self.assemble_weighted_mass(fun, 'Hdiv', 'Hcurl')

        return self._M1J

    @property
    def M2J(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{2,J}_{(\mu,ijk), (\nu,mno)} = \int \Lambda^2_{\mu,ijk}\, \mathcal R^J_{\alpha, \nu}\, \Lambda^2_{\nu, mno} \, \frac{1}{\sqrt g}\,  \textnormal d \boldsymbol\eta. 

        with the rotation matrix

        .. math::

            \mathcal R^J_{\alpha, \nu} := \epsilon_{\alpha \beta \nu}\, J^2_{\textnormal{eq}, \beta}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec J^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium current density (2-form).
        """

        if not hasattr(self, '_M2J'):

            rot_J = RotationMatrix(
                self.weights['eq_mhd'].j2_1, self.weights['eq_mhd'].j2_2, self.weights['eq_mhd'].j2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: rot_J(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._M2J = self.assemble_weighted_mass(fun, 'Hdiv', 'Hdiv')

        return self._M2J

    @property
    def MvJ(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{v,J}_{(\mu,ijk), (\nu,mno)} = \int \Lambda^v_{\mu,ijk}\, \mathcal R^J_{\alpha, \nu}\, \Lambda^2_{\nu, mno} \,  \textnormal d \boldsymbol\eta. 

        with the rotation matrix

        .. math::

            \mathcal R^J_{\alpha, \nu} := \epsilon_{\alpha \beta \nu}\, J^2_{\textnormal{eq}, \beta}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec J^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium current density (2-form).
        """

        if not hasattr(self, '_MvJ'):
            
            rot_J = RotationMatrix(
                self.weights['eq_mhd'].j2_1, self.weights['eq_mhd'].j2_2, self.weights['eq_mhd'].j2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: rot_J(e1, e2, e3)[:, :, :, m, n]]

            self._MvJ = self.assemble_weighted_mass(fun, 'Hdiv', 'H1vec')

        return self._MvJ

    #######################################
    # Wrapper around WeightedMassOperator #
    #######################################
    def assemble_weighted_mass(self, fun: list, V_id: str, W_id: str):
        r""" Weighted mass matrix :math:`V^\alpha_h \to V^\beta_h` with given (matrix-valued) weight function :math:`W(\boldsymbol \eta)`:

        .. math::

            \mathbb M_{(\mu, ijk), (\nu, mno)}(W) = \int \Lambda^\beta_{\mu, ijk}\, W_{\mu,\nu}(\boldsymbol \eta)\,  \Lambda^\alpha_{\nu, mno} \,  \textnormal d \boldsymbol\eta. 

        Here, :math:`\alpha \in \{0, 1, 2, 3, v\}` indicates the domain and :math:`\beta \in \{0, 1, 2, 3, v\}` indicates the co-domain 
        of the operator.

        Parameters
        ----------
        fun : list[list[callable | ndarray]]
            2d list of either all 3d arrays or all scalar functions of eta1, eta2, eta3 (must allow matrix evaluations). 
            3d arrays must have shape corresponding to the 1d quad_grids of V1-ProductFemSpace.

        V_id : str
            Specifier for the domain of the operator ('H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec').

        W_id : str
            Specifier for the co-domain of the operator ('H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec').

        Returns
        -------
        out : A WeightedMassOperator object.
        """

        assert isinstance(fun, list)

        if W_id in {'H1', 'L2'}:
            assert len(fun) == 1
        else:
            assert len(fun) == 3

        for row in fun:
            assert isinstance(row, list)
            if V_id in {'H1', 'L2'}:
                assert len(row) == 1
            else:
                assert len(row) == 3

        V_id = self.derham.spaces_dict[V_id]
        W_id = self.derham.spaces_dict[W_id]

        out = WeightedMassOperator(self.derham.Vh_fem[V_id], self.derham.Vh_fem[W_id],
                                   V_extraction_op=self.derham.E[V_id], W_extraction_op=self.derham.E[W_id],
                                   V_boundary_op=self.derham.B[V_id], W_boundary_op=self.derham.B[W_id],
                                   weights=fun, transposed=False)

        out.assemble()
        out.matrix.exchange_assembly_data()

        return out


class WeightedMassOperator(LinOpWithTransp):
    """
    Weighted mass matrix of the form B * E * M * E^T * B^T, with E and B being basis extraction and boundary operators, respectively.

    Parameters
    ----------
    V : TensorFemSpace | ProductFemSpace
        Tensor product spline space from psydac.fem.tensor (domain, input space).

    W : TensorFemSpace | ProductFemSpace
        Tensor product spline space from psydac.fem.tensor (codomain, output space).

    V_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of V.

    W_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of W.

    V_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions.

    W_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions.

    weights : list | NoneType
        Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain. If None, identity weights are assumed.

    symmetry : str
        Symmetry of block matrices ('symm', 'asym' or 'diag').

    transposed : bool
        Whether to assemble the transposed operator.
    """

    def __init__(self, V, W, V_extraction_op=None, W_extraction_op=None, V_boundary_op=None, W_boundary_op=None, weights=None, symmetry=None, transposed=False):

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        assert isinstance(V, (TensorFemSpace, ProductFemSpace))
        assert isinstance(W, (TensorFemSpace, ProductFemSpace))

        self._V = V
        self._W = W

        # set basis extraction operators
        if V_extraction_op is not None:
            assert isinstance(
                V_extraction_op, (PolarExtractionOperator, IdentityOperator))
            assert V_extraction_op.domain == V.vector_space
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.vector_space)

        if W_extraction_op is not None:
            assert isinstance(
                W_extraction_op, (PolarExtractionOperator, IdentityOperator))
            assert W_extraction_op.domain == W.vector_space
            self._W_extraction_op = W_extraction_op
        else:
            self._W_extraction_op = IdentityOperator(W.vector_space)

        # set boundary operators
        if V_boundary_op is not None:
            assert isinstance(
                V_boundary_op, (BoundaryOperator, IdentityOperator))
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(
                self._V_extraction_op.codomain)

        if W_boundary_op is not None:
            assert isinstance(
                W_boundary_op, (BoundaryOperator, IdentityOperator))
            self._W_boundary_op = W_boundary_op
        else:
            self._W_boundary_op = IdentityOperator(
                self._W_extraction_op.codomain)

        self._symmetry = symmetry
        self._transposed = transposed

        self._dtype = V.vector_space.dtype

        # set domain and codomain symbolic names
        if hasattr(V.symbolic_space, 'name'):
            V_name = V.symbolic_space.name
        else:
            if V.ldim == 3 or V.ldim == 2:
                V_name = 'H1vec'
            elif V.ldim == 1:
                if V.spaces[0].basis == 'B':
                    V_name = 'H1'
                else:
                    V_name = 'L2'

        if hasattr(W.symbolic_space, 'name'):
            W_name = W.symbolic_space.name
        else:
            if W.ldim == 3 or W.ldim == 2:
                W_name = 'H1vec'
            elif W.ldim == 1:
                if W.spaces[0].basis == 'B':
                    W_name = 'H1'
                else:
                    W_name = 'L2'

        if transposed:
            self._domain_femspace = W
            self._domain_symbolic_name = W_name

            self._codomain_femspace = V
            self._codomain_symbolic_name = V_name
        else:
            self._domain_femspace = V
            self._domain_symbolic_name = V_name

            self._codomain_femspace = W
            self._codomain_symbolic_name = W_name

        # ====== initialize Stencil-/BlockLinearOperator ====

        # collect TensorFemSpaces for each component in tuple
        if isinstance(V, TensorFemSpace):
            Vspaces = (V,)
        else:
            Vspaces = V.spaces

        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces

        # initialize blocks according to given symmetry
        if symmetry is not None:

            assert symmetry in {'symm', 'asym',
                                'diag', 'upper_tri', 'lower_tri'}
            assert V_name in {'Hcurl', 'Hdiv', 'H1vec'}
            assert V_name == W_name, 'only square matrices (V=W) allowed!'

            if symmetry == 'symm':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                           for Vs in V.spaces] for Ws in W.spaces]
            elif symmetry == 'asym':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                           if i != j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            elif symmetry == 'diag':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                           if i == j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            elif symmetry == 'upper_tri':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                           if i <= j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            elif symmetry == 'lower_tri':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                           if i <= j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]

            self._mat = BlockLinearOperator(
                V.vector_space, W.vector_space, blocks=blocks)

            # set default identity weights if not given
            self._weights = []
            for a, row in enumerate(blocks):
                self._weights += [[]]
                for b, col in enumerate(row):
                    if col is None:
                        self._weights[-1] += [None]
                    else:
                        if weights is None:
                            if symmetry == 'asym' and (a, b) in [(1, 0), (2, 0), (2, 1)]:
                                self._weights[-1] += [lambda *etas: -
                                                      np.ones(etas[0].shape, dtype=float)]
                            else:
                                self._weights[-1] += [
                                    lambda *etas: np.ones(etas[0].shape, dtype=float)]
                        else:
                            self._weights[-1] += [weights[a][b]]

        # initialize all blocks or according to given weights
        else:

            blocks = []
            self._weights = []

            # loop over codomain spaces (rows)
            for a, wspace in enumerate(Wspaces):
                blocks += [[]]
                self._weights += [[]]

                # loop over domain spaces (columns)
                for b, vspace in enumerate(Vspaces):

                    # set default identity weights if not given
                    if weights is None:
                        blocks[-1] += [StencilMatrix(
                            vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)]
                        self._weights[-1] += [lambda *etas:
                                              np.ones(etas[0].shape, dtype=float)]

                    else:

                        if weights[a][b] is None:
                            blocks[-1] += [None]
                            self._weights[-1] += [None]

                        else:

                            # test weight function at quadrature points to identify zero blocks
                            pts = [quad_grid.points.flatten()
                                   for quad_grid in wspace.quad_grids]

                            if callable(weights[a][b]):
                                PTS = np.meshgrid(*pts, indexing='ij')
                                mat_w = weights[a][b](*PTS).copy()
                            elif isinstance(weights[a][b], np.ndarray):
                                mat_w = weights[a][b]

                            assert mat_w.shape == tuple(
                                [pt.size for pt in pts])

                            if np.any(np.abs(mat_w) > 1e-14):
                                blocks[-1] += [StencilMatrix(
                                    vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)]
                                self._weights[-1] += [weights[a][b]]
                            else:
                                blocks[-1] += [None]
                                self._weights[-1] += [None]

            if len(blocks) == len(blocks[0]) == 1:
                self._mat = blocks[0][0]
            else:
                self._mat = BlockLinearOperator(
                    V.vector_space, W.vector_space, blocks=blocks)

        # transpose of matrix and weights
        if transposed:
            self._mat = self._mat.transpose()

            n_rows = len(self._weights)
            n_cols = len(self._weights[0])

            tmp_weights = []

            for m in range(n_cols):
                tmp_weights += [[]]
                for n in range(n_rows):
                    if self._weights[n][m] is not None:
                        tmp_weights[-1] += [self._weights[n][m]]
                    else:
                        tmp_weights[-1] += [None]

            self._weights = tmp_weights

        # ===============================================

        # some shortcuts
        BW = self._W_boundary_op
        BV = self._V_boundary_op

        EW = self._W_extraction_op
        EV = self._V_extraction_op

        # build composite linear operators BW * EW * M * EV^T * BV^T, resp. IDV * EV * M^T * EW^T * IDW^T
        if transposed:
            self._M = Compose(IdentityOperator(EV.codomain), EV,
                              self._mat, EW.T, IdentityOperator(EW.codomain).T)
            self._M0 = Compose(BV, EV, self._mat, EW.T, BW.T)
        else:
            self._M = Compose(IdentityOperator(EW.codomain), EW,
                              self._mat, EV.T, IdentityOperator(EV.codomain).T)
            self._M0 = Compose(BW, EW, self._mat, EV.T, BV.T)

        # set domain and codomain
        self._domain = self._M.domain
        self._codomain = self._M.codomain

        # load assembly kernel
        self._assembly_kernel = getattr(
            mass_kernels, 'kernel_' + str(self._V.ldim) + 'd_mat')

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def domain_femspace(self):
        return self._domain_femspace

    @property
    def codomain_femspace(self):
        return self._codomain_femspace

    @property
    def dtype(self):
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def M(self):
        return self._M

    @property
    def M0(self):
        return self._M0

    @property
    def matrix(self):
        return self._mat

    @property
    def weights(self):
        return self._weights

    def dot(self, v, out=None, apply_bc=True):
        """
        Dot product of the operator with a vector.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            The input (domain) vector.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        apply_bc : bool
            Whether to apply the boundary operators (True) or not (False).

        Returns
        -------
        out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self.domain

        # newly created output vector
        if out is None:
            if apply_bc:
                out = self._M0.dot(v)
            else:
                out = self._M.dot(v)

        # in-place dot-product (result is written to out)
        else:

            assert isinstance(out, Vector)
            assert out.space == self.codomain

            if apply_bc:
                self._M0.dot(v, out=out)
            else:
                self._M.dot(v, out=out)

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """

        # bring weights back in "right" (not transposed order)
        if self._transposed:
            n_rows = len(self._weights)
            n_cols = len(self._weights[0])

            weights = []

            for m in range(n_cols):
                weights += [[]]
                for n in range(n_rows):
                    if self._weights[n][m] is not None:
                        weights[-1] += [self._weights[n][m]]
                    else:
                        weights[-1] += [None]
        else:
            weights = self._weights

        M = WeightedMassOperator(self._V, self._W,
                                 self._V_extraction_op, self._W_extraction_op,
                                 self._V_boundary_op, self._W_boundary_op,
                                 weights, self._symmetry, not self._transposed)

        M.assemble(verbose=False)
        M._mat.exchange_assembly_data()

        return M

    def assemble(self, weights=None, verbose=True):
        """
        Assembles a weighted mass matrix (StencilMatrix/BlockLinearOperator) corresponding to given domain/codomain spline spaces.

        General form (in 3d) is mat_(ijk,mno) = integral[ Lambda_ijk * weight * Lambda_lmn ],
        where Lambda_ijk are the basis functions of the spline space and weight is some weight function.

        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.

        Parameters
        ----------
        weights : list | NoneType
            Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain. If weights=None, the weight is taken from the given weights in the instanziation of the object, else it will ne overriden.

        verbose : bool
            Whether to do some printing.
        """

        # identify rank for printing
        if self._domain_symbolic_name in {'H1', 'L2'}:
            if self._V.vector_space.cart.comm is not None:
                rank = self._V.vector_space.cart.comm.Get_rank()
            else:
                rank = 0
        else:
            if self._V.vector_space[0].cart.comm is not None:
                rank = self._V.vector_space[0].cart.comm.Get_rank()
            else:
                rank = 0

        if rank == 0 and verbose:
            print(
                f'Assembling matrix of WeightedMassOperator with V={self._domain_symbolic_name}, W={self._codomain_symbolic_name}.')

        # collect domain/codomain TensorFemSpaces for each component in tuple
        if self._transposed:
            if isinstance(self._W, TensorFemSpace):
                domain_spaces = (self._W,)
            else:
                domain_spaces = self._W.spaces

            if isinstance(self._V, TensorFemSpace):
                codomain_spaces = (self._V,)
            else:
                codomain_spaces = self._V.spaces
        else:
            if isinstance(self._V, TensorFemSpace):
                domain_spaces = (self._V,)
            else:
                domain_spaces = self._V.spaces

            if isinstance(self._W, TensorFemSpace):
                codomain_spaces = (self._W,)
            else:
                codomain_spaces = self._W.spaces

        # override weights
        if weights is not None:
            self._weights = weights

        # loop over codomain spaces (rows)
        for a, codomain_space in enumerate(codomain_spaces):

            # knot span indices of elements of local domain
            codomain_spans = [
                quad_grid.spans for quad_grid in codomain_space.quad_grids]

            # global start spline index on process
            codomain_starts = [int(start)
                               for start in codomain_space.vector_space.starts]

            # pads (ghost regions)
            codomain_pads = codomain_space.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid.points.flatten()
                   for quad_grid in codomain_space.quad_grids]
            wts = [quad_grid.weights for quad_grid in codomain_space.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            codomain_basis = [
                quad_grid.basis for quad_grid in codomain_space.quad_grids]

            # loop over domain spaces (columns)
            for b, domain_space in enumerate(domain_spaces):

                if self._weights[a][b] is not None:

                    if callable(self._weights[a][b]):
                        PTS = np.meshgrid(*pts, indexing='ij')
                        mat_w = self._weights[a][b](*PTS).copy()
                    elif isinstance(self._weights[a][b], np.ndarray):
                        mat_w = self._weights[a][b]

                # None weight blocks are identified as zeros
                else:
                    mat_w = np.zeros([pt.size for pt in pts], dtype=float)

                assert mat_w.shape == tuple([pt.size for pt in pts])

                # evaluated basis functions at quadrature points of domain space
                domain_basis = [
                    quad_grid.basis for quad_grid in domain_space.quad_grids]

                # assemble matrix (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)
                if np.any(np.abs(mat_w) > 1e-14):
                    if isinstance(self._mat, StencilMatrix):
                        self._assembly_kernel(*codomain_spans, *codomain_space.degree, *domain_space.degree, *
                                              codomain_starts, *codomain_pads, *wts, *codomain_basis, *domain_basis, mat_w, self._mat._data)
                    else:
                        self._assembly_kernel(*codomain_spans, *codomain_space.degree, *domain_space.degree, *codomain_starts,
                                              *codomain_pads, *wts, *codomain_basis, *domain_basis, mat_w, self._mat[a, b]._data)

        if rank == 0 and verbose:
            print('Done.')

    @staticmethod
    def assemble_vec(W, vec, weight=None):
        """
        Assembles (in 3d) vec_ijk = integral[ weight * Lambda_ijk ] into the Stencil-/BlockVector vec,
        where Lambda_ijk are the basis functions of the spline space and weight is some weight function.

        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.

        Parameters
        ----------
        W : TensorFemSpace | ProductFemSpace
            Tensor product spline space from psydac.fem.tensor.

        vec : StencilVector | BlockVector
            The vector to be filled.

        weight : list | NoneType
            Weight function(s) (callables or np.ndarrays) in a 1d list of shape corresponding to number of components.
        """

        assert isinstance(W, (TensorFemSpace, ProductFemSpace))
        assert isinstance(vec, (StencilVector, BlockVector))
        assert W.vector_space == vec.space

        # collect TensorFemSpaces for each component in tuple
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces

        # loag assembly kernel
        kernel = getattr(mass_kernels, 'kernel_' + str(W.ldim) + 'd_vec')

        # loop over components
        for a, wspace in enumerate(Wspaces):

            # knot span indices of elements of local domain
            spans = [quad_grid.spans for quad_grid in wspace.quad_grids]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid.points.flatten()
                   for quad_grid in wspace.quad_grids]
            wts = [quad_grid.weights for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            basis = [quad_grid.basis for quad_grid in wspace.quad_grids]

            if weight is not None:
                if weight[a] is not None:

                    if callable(weight[a]):
                        PTS = np.meshgrid(*pts, indexing='ij')
                        mat_w = weight[a](*PTS).copy()
                    elif isinstance(weight[a], np.ndarray):
                        mat_w = weight[a]

                else:
                    mat_w = np.zeros([pt.size for pt in pts], dtype=float)
            else:
                mat_w = np.ones([pt.size for pt in pts], dtype=float)

            assert mat_w.shape == tuple([pt.size for pt in pts])

            # assemble vector (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)
            if np.any(np.abs(mat_w) > 1e-14):
                if isinstance(vec, StencilVector):
                    kernel(*spans, *wspace.degree, *starts, *pads,
                           *wts, *basis, mat_w, vec._data)
                else:
                    kernel(*spans, *wspace.degree, *starts, *pads,
                           *wts, *basis, mat_w, vec[a]._data)

    @staticmethod
    def eval_quad(W, coeffs, out=None):
        """
        Evaluates a given FEM field defined by its coefficients at the L2 quadrature points.

        Parameters
        ----------
        W : TensorFemSpace | ProductFemSpace
            Tensor product spline space from psydac.fem.tensor.

        coeffs : StencilVector | BlockVector
            The coefficient vector corresponding to the FEM field. Ghost regions must be up-to-date!
            
        out : np.ndarray | list/tuple of np.ndarrays, optional
            If given, the result will be written into these arrays in-place. Number of outs must be compatible with number of components of FEM field.
            
        Returns
        -------
        out : np.ndarray | list/tuple of np.ndarrays
            The values of the FEM field at the quadrature points.
        """

        assert isinstance(W, (TensorFemSpace, ProductFemSpace))
        assert isinstance(coeffs, (StencilVector, BlockVector))
        assert W.vector_space == coeffs.space

        # collect TensorFemSpaces for each component in tuple
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces
            
        # prepare output
        if out is None:
            out = ()
            if isinstance(W, TensorFemSpace):
                out += (np.zeros([q_grid.points.size for q_grid in W.quad_grids], dtype=float),)
            else:
                for space in W.spaces:
                    out += (np.zeros([q_grid.points.size for q_grid in space.quad_grids], dtype=float),)
                    
        else:
            if isinstance(W, TensorFemSpace):
                assert isinstance(out, np.ndarray)
                out = (out,)
            else:
                assert isinstance(out, (list, tuple))

        # load assembly kernel
        kernel = getattr(mass_kernels, 'kernel_' + str(W.ldim) + 'd_eval')
        
        # loop over components
        for a, wspace in enumerate(Wspaces):

            # knot span indices of elements of local domain
            spans = [quad_grid.spans for quad_grid in wspace.quad_grids]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid.points.flatten()
                   for quad_grid in wspace.quad_grids]
            wts = [quad_grid.weights for quad_grid in wspace.quad_grids]

            # evaluated basis functions at quadrature points of codomain space
            basis = [quad_grid.basis for quad_grid in wspace.quad_grids]

            if isinstance(coeffs, StencilVector):
                kernel(*spans, *wspace.degree, *starts, *
                       pads, *basis, coeffs._data, out[a])
            else:
                kernel(*spans, *wspace.degree, *starts, *
                       pads, *basis, coeffs[a]._data, out[a])
        
        if len(out) == 1:
            return out[0]
        else:
            return out
