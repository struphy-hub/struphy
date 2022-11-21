import numpy as np

import psydac.core.bsplines as bsp

from psydac.linalg.basic import LinearOperator
from psydac.fem.basic import FemSpace
from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix
from psydac.feec.global_projectors import GlobalProjector

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api.linear_operators import LinOpWithTransp
from struphy.psydac_api.linear_operators import ApplyHomogeneousDirichletToOperator

from struphy.psydac_api.basis_projection_kernels import assemble_dofs_for_weighted_basisfuns_1d as assemble_1d
from struphy.psydac_api.basis_projection_kernels import assemble_dofs_for_weighted_basisfuns_2d as assemble_2d
from struphy.psydac_api.basis_projection_kernels import assemble_dofs_for_weighted_basisfuns_3d as assemble_3d

from struphy.psydac_api.prepare_projection import evaluate_fun_weights_1d, evaluate_fun_weights_2d, evaluate_fun_weights_3d


class BasisProjectionOperators:
    """
    Assembles some or all basis projection operators needed for various discretizations of linear MHD equations in 3d.

    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.

        eq_mhd : EquilibriumMHD
            MHD equilibrium from struphy.fields_background.mhd_equil (pullbacks must be enabled).

    Notes
    -----
    The `X0`, `X1`, `X2` operators are handled differently, because it outputs 3 scalar spaces instead of a pure scalar or vector space.
    In order not to modify the `MHDOperator` class, we give a set of three functions, each accessing each row of the input matrix-valued function.
    """

    def __init__(self, derham, domain, eq_mhd):

        assert np.all(np.array(
            derham.p) > 1), 'Spline degrees must be >1 to use basis projection operators (-> avoid interpolation of piece-wise constants).'

        self._derham = derham
        self._domain = domain

        # Psydac spline spaces
        self._V0 = derham.V0
        self._V1 = derham.V1
        self._V2 = derham.V2
        self._V3 = derham.V3
        self._V0vec = derham.V0vec

        # Psydac projectors
        self._P0 = derham.P0
        self._P1 = derham.P1
        self._P2 = derham.P2
        self._P3 = derham.P3
        self._P0vec = derham.P0vec

        # Wrapper functions for metric coefficients
        def DF(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, False, False, True, False)

        def DFT(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, False, False, True, True)

        def DFinv(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, False, False, True, False)

        def DFinvT(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, False, False, True, True)

        def G(e1, e2, e3):
            return domain.metric(e1, e2, e3, False, False, True)

        def Ginv(e1, e2, e3):
            return domain.metric_inv(e1, e2, e3, False, False, True)

        def sqrt_g(e1, e2, e3):
            return abs(domain.jacobian_det(e1, e2, e3, False, False))

        # Cross product matrices and evaluation of cross products
        cross_mask = [[1, -1,  1],
                      [1,  1, -1],
                      [-1,  1,  1]]

        def eval_cross(e1, e2, e3, fun_list):

            cross = np.array([[cross_mask[m][n] * fun(e1, e2, e3)
                             for n, fun in enumerate(row)] for m, row in enumerate(fun_list)])

            return np.transpose(cross, axes=(2, 3, 4, 0, 1))

        j2_cross = [[lambda e1, e2, e3: 0*e1, eq_mhd.j2_3, eq_mhd.j2_2],
                    [eq_mhd.j2_3, lambda e1, e2, e3: 0*e2, eq_mhd.j2_1],
                    [eq_mhd.j2_2, eq_mhd.j2_1, lambda e1, e2, e3: 0*e3]]

        b2_cross = [[lambda e1, e2, e3: 0*e1, eq_mhd.b2_3, eq_mhd.b2_2],
                    [eq_mhd.b2_3, lambda e1, e2, e3: 0*e2, eq_mhd.b2_1],
                    [eq_mhd.b2_2, eq_mhd.b2_1, lambda e1, e2, e3: 0*e3]]

        # Scalar functions
        fun_K0 = [[lambda e1, e2, e3: eq_mhd.p3(
            e1, e2, e3) / sqrt_g(e1, e2, e3)]]

        fun_K1 = [[lambda e1, e2, e3: eq_mhd.p3(
            e1, e2, e3) / sqrt_g(e1, e2, e3)]]
        fun_K10 = [[lambda e1, e2, e3: eq_mhd.p0(e1, e2, e3)]]

        fun_K2 = [[lambda e1, e2, e3: eq_mhd.p3(
            e1, e2, e3) / sqrt_g(e1, e2, e3)]]
        fun_Y20 = [[lambda e1, e2, e3: sqrt_g(e1, e2, e3)]]

        # 'Matrix' functions
        fun_Q0 = []
        fun_Tv = []
        fun_S0 = []
        fun_Uv = []
        fun_X0 = []

        fun_Q1 = []
        fun_W1 = []
        fun_U1 = []
        fun_R1 = []
        fun_S1 = []
        fun_T1 = []
        fun_X1 = []
        fun_S10 = []

        fun_Q2 = []
        fun_T2 = []
        fun_R2 = []
        fun_S2 = []
        fun_X2 = []
        fun_Z20 = []
        fun_S20 = []

        for m in range(3):
            fun_Q0 += [[]]
            fun_Tv += [[]]
            fun_S0 += [[]]
            fun_Uv += [[]]
            fun_X0 += [[]]

            fun_Q1 += [[]]
            fun_W1 += [[]]
            fun_U1 += [[]]
            fun_R1 += [[]]
            fun_S1 += [[]]
            fun_T1 += [[]]
            fun_X1 += [[]]
            fun_S10 += [[]]

            fun_Q2 += [[]]
            fun_T2 += [[]]
            fun_R2 += [[]]
            fun_S2 += [[]]
            fun_X2 += [[]]
            fun_Z20 += [[]]
            fun_S20 += [[]]

            for n in range(3):
                # See documentation in `struphy.feec.projectors.pro_global.mhd_operators_MF_for_tests.projectors_dot_x`.
                fun_Q0[-1] += [lambda e1, e2, e3, m=m,
                               n=n: eq_mhd.n3(e1, e2, e3) if m == n else 0*e1]
                fun_Tv[-1] += [lambda e1, e2, e3, m=m,
                               n=n: cross_mask[m][n] * b2_cross[m][n](e1, e2, e3)]
                fun_S0[-1] += [lambda e1, e2, e3, m=m,
                               n=n: eq_mhd.p3(e1, e2, e3) if m == n else 0*e1]
                fun_Uv[-1] += [lambda e1, e2, e3, m=m,
                               n=n: sqrt_g(e1, e2, e3) if m == n else 0*e1]
                fun_X0[-1] += [lambda e1, e2, e3, m=m,
                               n=n: DF(e1, e2, e3)[:, :, :, m, n]]

                fun_Q1[-1] += [lambda e1, e2, e3, m=m,
                               n=n: eq_mhd.n3(e1, e2, e3) * Ginv(e1, e2, e3)[:, :, :, m, n]]
                fun_W1[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.n3(
                    e1, e2, e3) / sqrt_g(e1, e2, e3) if m == n else 0*e1]
                fun_U1[-1] += [lambda e1, e2, e3, m=m,
                               n=n: sqrt_g(e1, e2, e3) * Ginv(e1, e2, e3)[:, :, :, m, n]]
                fun_R1[-1] += [lambda e1, e2, e3, m=m, n=n: cross_mask[m][n] *
                               j2_cross[m][n](e1, e2, e3) / sqrt_g(e1, e2, e3)]
                fun_S1[-1] += [lambda e1, e2, e3, m=m,
                               n=n: eq_mhd.p3(e1, e2, e3) * Ginv(e1, e2, e3)[:, :, :, m, n]]
                fun_T1[-1] += [lambda e1, e2, e3, m=m, n=n: (eval_cross(
                    e1, e2, e3, b2_cross) @ Ginv(e1, e2, e3))[:, :, :, m, n]]  # Matrix product!
                fun_X1[-1] += [lambda e1, e2, e3, m=m,
                               n=n: DFinvT(e1, e2, e3)[:, :, :, m, n]]
                fun_S10[-1] += [lambda e1, e2, e3, m=m,
                                n=n: eq_mhd.p0(e1, e2, e3) if m == n else 0*e1]

                fun_Q2[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.n3(
                    e1, e2, e3) / sqrt_g(e1, e2, e3) if m == n else 0*e1]
                fun_T2[-1] += [lambda e1, e2, e3, m=m, n=n: cross_mask[m][n] *
                               b2_cross[m][n](e1, e2, e3) / sqrt_g(e1, e2, e3)]
                fun_R2[-1] += [lambda e1, e2, e3, m=m, n=n: (Ginv(e1, e2, e3) @ eval_cross(
                    e1, e2, e3, j2_cross))[:, :, :, m, n]]  # Matrix product!
                fun_S2[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.p3(
                    e1, e2, e3) / sqrt_g(e1, e2, e3) if m == n else 0*e1]
                fun_X2[-1] += [lambda e1, e2, e3, m=m,
                               n=n: DF(e1, e2, e3)[:, :, :, m, n] / sqrt_g(e1, e2, e3)]
                fun_Z20[-1] += [lambda e1, e2, e3, m=m,
                                n=n: G(e1, e2, e3)[:, :, :, m, n] / sqrt_g(e1, e2, e3)]
                fun_S20[-1] += [lambda e1, e2, e3, m=m, n=n: eq_mhd.p0(
                    e1, e2, e3) * G(e1, e2, e3)[:, :, :, m, n] / sqrt_g(e1, e2, e3)]

        # Scalar functions
        self._fun_K0 = fun_K0

        self._fun_K1 = fun_K1
        self._fun_K10 = fun_K10

        self._fun_K2 = fun_K2
        self._fun_Y20 = fun_Y20

        # 'Matrix' functions
        self._fun_Q0 = fun_Q0
        self._fun_Tv = fun_Tv
        self._fun_S0 = fun_S0
        self._fun_Uv = fun_Uv
        self._fun_X0 = fun_X0

        self._fun_Q1 = fun_Q1
        self._fun_W1 = fun_W1
        self._fun_U1 = fun_U1
        self._fun_R1 = fun_R1
        self._fun_S1 = fun_S1
        self._fun_T1 = fun_T1
        self._fun_X1 = fun_X1
        self._fun_S10 = fun_S10

        self._fun_Q2 = fun_Q2
        self._fun_T2 = fun_T2
        self._fun_R2 = fun_R2
        self._fun_S2 = fun_S2
        self._fun_X2 = fun_X2
        self._fun_Z20 = fun_Z20
        self._fun_S20 = fun_S20

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

    @property
    def derham(self):
        return self._derham

    @property
    def domain(self):
        return self._domain

    ###################################################
    # Basis projection operators with velocity (up) in H1^3 (V0vec):
    ###################################################
    @property
    def K0(self):
        r'''Basis projection operator :math:`\mathcal{K}^0` with the velocity in :math:`(H^1)^3`, denoted :math:`\hat{\mathbf{U}}`,
         and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{K}^0 = \hat{\Pi}_3 \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\Lambda}^3 \right] \in \mathbb{R}^{N^3 \times N^3}, 
            \qquad \mathcal{K}^0_{(ijk),(mno)} := \hat{\Pi}_{3,(ijk)} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\Lambda}^3_{(mno)} \right] \,.

        '''
        if not hasattr(self, '_K0'):
            self._K0 = ApplyHomogeneousDirichletToOperator(
                'L2', 'L2', self.derham.bc, BasisProjectionOp(self._P3, self._V3, self._fun_K0))

        return self._K0

    @property
    def Q0(self):
        r'''Basis projection operator :math:`\mathcal{Q}^0` with the velocity in :math:`(H^1)^3`, denoted :math:`\hat{\mathbf{U}}`,
         and the pressure as 3-form :math:`\hat{p}^3`.

        .. math::

            \mathcal{Q}^0 = \hat{\Pi}_2 \left[ \hat{\rho}^3_{\text{eq}} \mathbf{\vec{\Lambda}}^0 \right] \in \mathbb{R}^{N^2 \times 3 \times N^0}, 
            \qquad \mathcal{Q}^0_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\hat{\rho}^3_{\text{eq}} \mathbf{\vec{\Lambda}}^0_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Q0'):
            self._Q0 = ApplyHomogeneousDirichletToOperator(
                'H1vec', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V0vec, self._fun_Q0))

        return self._Q0

    @property
    def Tv(self):
        r'''Basis projection operator :math:`\mathcal{T}^0` with the velocity in :math:`(H^1)^3`, denoted :math:`\hat{\mathbf{U}}`, 
        and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{T}^0 = \hat{\Pi}_1 \left[ \hat{B}^2_{\text{eq}} \times \mathbf{\vec{\Lambda}}^0 \right] \in \mathbb{R}^{N^1 \times 3 \times N^0}, 
            \qquad \mathcal{T}^0_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\epsilon_{\mu \alpha \nu} \hat{B}^2_{\text{eq},\alpha} \mathbf{\vec{\Lambda}}^0_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Tv'):
            self._Tv = ApplyHomogeneousDirichletToOperator(
                'H1vec', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V0vec, self._fun_Tv))

        return self._Tv

    @property
    def S0(self):
        r'''Basis projection operator :math:`\mathcal{S}^0` with the velocity in :math:`(H^1)^3`, denoted :math:`\hat{\mathbf{U}}`, 
        and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{S}^0 = \hat{\Pi}_2 \left[ \hat{p}^3_{\text{eq}} \mathbf{\vec{\Lambda}}^0 \right] \in \mathbb{R}^{N^2 \times 3 \times N^0}, 
            \qquad \mathcal{S}^0_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\hat{p}^3_{\text{eq}} \mathbf{\vec{\Lambda}}^0_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S0'):
            self._S0 = ApplyHomogeneousDirichletToOperator(
                'H1vec', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V0vec, self._fun_S0))

        return self._S0

    @property
    def Uv(self):
        r'''Basis projection operator :math:`\mathcal{J}^0` with the velocity in :math:`(H^1)^3`, denoted :math:`\hat{\mathbf{U}}`, 
        and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{J}^0 = \hat{\Pi}_2 \left[ \sqrt{g} \, \mathbf{\vec{\Lambda}}^0 \right] \in \mathbb{R}^{N^2 \times 3 \times N^0}, 
            \qquad \mathcal{J}^0_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\sqrt{g} \, \mathbf{\vec{\Lambda}}^0_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Uv'):
            self._Uv = ApplyHomogeneousDirichletToOperator(
                'H1vec', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V0vec, self._fun_Uv))

        return self._Uv

    @property
    def X0(self):
        r'''Basis projection operator :math:`\mathcal{X}^0` with the velocity in :math:`(H^1)^3`, denoted :math:`\hat{\mathbf{U}}`, 
        and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{X}^0 = \hat{\Pi}_0 \left[ DF\mathbf{\vec{\Lambda}}^0 \right] \in \mathbb{R}^{N^0 \times 3 \times N^0}, 
            \qquad \mathcal{X}^0_{\nu,(ijk),(mno)} := \hat{\Pi}_{0,(ijk)} \left[ DF\mathbf{\vec{\Lambda}}^0_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_X0'):
            self._X0 = ApplyHomogeneousDirichletToOperator(
                'H1vec', 'H1vec', self.derham.bc, BasisProjectionOp(self._P0vec, self._V0vec, self._fun_X0))

        return self._X0

    #############################################
    # Basis projection operators with velocity (up) as 1-form:
    #############################################
    @property
    def K1(self):
        r'''Basis projection operator :math:`\mathcal{K}^1` with the velocity in :math:`H(\textnormal(curl))`, denoted :math:`\hat{\mathbf{U}}^1`,
         and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{K}^1 = \hat{\Pi}_3 \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\Lambda}^3 \right] \in \mathbb{R}^{N^3 \times N^3}, 
            \qquad \mathcal{K}^1_{(ijk),(mno)} := \hat{\Pi}_{3,(ijk)} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\Lambda}^3_{(mno)} \right] \,.

        '''
        if not hasattr(self, '_K1'):
            self._K1 = ApplyHomogeneousDirichletToOperator(
                'L2', 'L2', self.derham.bc, BasisProjectionOp(self._P3, self._V3, self._fun_K1))

        return self._K1

    @property
    def Q1(self):
        r'''Basis projection operator :math:`\mathcal{Q}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{Q}^1 = \hat{\Pi}_2 \left[ \hat{\rho}^3_{\text{eq}} G^{-1} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^2 \times N^1}, 
            \qquad \mathcal{Q}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\hat{\rho}^3_{\text{eq}}G^{-1}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Q1'):
            self._Q1 = ApplyHomogeneousDirichletToOperator(
                'Hcurl', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V1, self._fun_Q1))

        return self._Q1

    @property
    def W1(self):
        r'''Basis projection operator :math:`\mathcal{W}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{W}^1 = \hat{\Pi}_1 \left[ \frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^1 \times N^1}, 
            \qquad \mathcal{W}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_W1'):
            self._W1 = ApplyHomogeneousDirichletToOperator(
                'Hcurl', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V1, self._fun_W1))

        return self._W1

    @property
    def U1(self):
        r'''Basis projection operator :math:`\mathcal{U}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{U}^1 = \hat{\Pi}_2 \left[ \sqrt{g} \, G^{-1} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^2 \times N^1}, 
            \qquad \mathcal{U}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[ \sqrt{g} \, G^{-1} \mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_U1'):
            self._U1 = ApplyHomogeneousDirichletToOperator(
                'Hcurl', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V1, self._fun_U1))

        return self._U1

    @property
    def R1(self):
        r'''Basis projection operator :math:`\mathcal{R}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{R}^1 = \hat{\Pi}_1 \left[ \frac{\hat{J}^2_{\text{eq}}}{\sqrt{g}} \times \mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{R}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\epsilon_{\mu \alpha \nu}\frac{\hat{J}^2_{\text{eq}, \alpha}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_R1'):
            self._R1 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V2, self._fun_R1))

        return self._R1

    @property
    def S1(self):
        r'''Basis projection operator :math:`\mathcal{S}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{S}^1 = \hat{\Pi}_2 \left[ \hat{p}^3_{\text{eq}} G^{-1} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^2 \times N^1}, 
            \qquad \mathcal{S}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\hat{p}^3_{\text{eq}}G^{-1}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S1'):
            self._S1 = ApplyHomogeneousDirichletToOperator(
                'Hcurl', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V1, self._fun_S1))

        return self._S1

    @property
    def T1(self):
        r'''Basis projection operator :math:`\mathcal{T}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{T}^1 = \hat{\Pi}_1 \left[ \hat{B}^2_{\text{eq}} \times (G^{-1} \mathbf{\vec{\Lambda}}^1) \right] \in \mathbb{R}^{N^1 \times N^1}, 
            \qquad \mathcal{T}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\epsilon_{\mu \alpha \beta} \hat{B}^2_{\text{eq},\alpha}G^{-1}_{\beta \nu}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_T1'):
            self._T1 = ApplyHomogeneousDirichletToOperator(
                'Hcurl', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V1, self._fun_T1))

        return self._T1

    @property
    def X1(self):
        r'''Basis projection operator :math:`\mathcal{X}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{X}^1 = \hat{\Pi}_0 \left[ DF^{-\top}\mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^0 \times 3 \times N^1}, 
            \qquad \mathcal{X}^1_{\nu,(ijk),(mno)} := \hat{\Pi}_{0,(ijk)} \left[ DF^{-\top}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_X1'):
            self._X1 = ApplyHomogeneousDirichletToOperator(
                'Hcurl', 'H1vec', self.derham.bc, BasisProjectionOp(self._P0vec, self._V1, self._fun_X1))

        return self._X1

    @property
    def K10(self):
        r'''Basis projection operator :math:`\mathcal{K}^{10}` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{K}^{10} = \hat{\Pi}_3 \left[ \hat{p}^0_{\text{eq}} \mathbf{\Lambda}^0 \right] \in \mathbb{R}^{N^0 \times N^0}, 
            \qquad \mathcal{K}^{10}_{(ijk),(mno)} := \hat{\Pi}_{3,(ijk)} \left[  \hat{p}^0_{\text{eq}} \mathbf{\Lambda}^0_{(mno)} \right] \,.

        '''
        if not hasattr(self, '_K10'):
            self._K10 = ApplyHomogeneousDirichletToOperator(
                'H1', 'H1', self.derham.bc, BasisProjectionOp(self._P0, self._V0, self._fun_K10))

        return self._K10

    @property
    def S10(self):
        r'''Basis projection operator :math:`\mathcal{S}^{10}` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{S}^{10} = \hat{\Pi}_1 \left[ \hat{p}^0_{\text{eq}} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^1 \times N^1}, 
            \qquad \mathcal{S}^{10}_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[ \hat{p}^0_{\text{eq}} \mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S10'):
            self._S10 = ApplyHomogeneousDirichletToOperator(
                'Hcurl', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V1, self._fun_S10))

        return self._S10

    #############################################
    # Basis projection operators with velocity (up) as 2-form:
    #############################################
    @property
    def K2(self):
        r'''Basis projection operator :math:`\mathcal{K}^2` with the velocity in :math:`H(\textnormal(div))`, denoted  :math:`\hat{\mathbf{U}}^2`,
         and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{K}^2 = \hat{\Pi}_3 \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\Lambda}^3 \right] \in \mathbb{R}^{N^3 \times N^3}, 
            \qquad \mathcal{K}^1_{(ijk),(mno)} := \hat{\Pi}_{3,(ijk)} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\Lambda}^3_{(mno)} \right] \,.

        '''
        if not hasattr(self, '_K2'):
            self._K2 = ApplyHomogeneousDirichletToOperator(
                'L2', 'L2', self.derham.bc, BasisProjectionOp(self._P3, self._V3, self._fun_K2))

        return self._K2

    @property
    def Q2(self):
        r'''Basis projection operator :math:`\mathcal{Q}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{Q}^2 = \hat{\Pi}_2 \left[ \frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^2 \times N^2}, 
            \qquad \mathcal{Q}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Q2'):
            self._Q2 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V2, self._fun_Q2))

        return self._Q2

    @property
    def T2(self):
        r'''Basis projection operator :math:`\mathcal{T}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{T}^2 = \hat{\Pi}_1 \left[ \frac{\hat{B}^2_{\text{eq}}}{\sqrt{g}} \times \mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{T}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\epsilon_{\mu \alpha \nu} \frac{\hat{B}^2_{\text{eq},\alpha}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_T2'):
            self._T2 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V2, self._fun_T2))

        return self._T2

    @property
    def R2(self):
        r'''Basis projection operator :math:`\mathcal{R}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{R}^2 = \hat{\Pi}_2 \left[\hat{J}^2_{\text{eq}} \times (G^{-1} \mathbf{\vec{\Lambda}}^2) \right] \in \mathbb{R}^{N^2 \times N^2}, 
            \qquad \mathcal{R}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\epsilon_{\mu \alpha \beta}\hat{J}^2_{\text{eq}, \alpha} G^{-1}_{\beta \nu} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_R2'):
            self._R2 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V2, self._fun_R2))

        return self._R2

    @property
    def S2(self):
        r'''Basis projection operator :math:`\mathcal{S}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{S}^2 = \hat{\Pi}_2 \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^2 \times N^2}, 
            \qquad \mathcal{S}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S2'):
            self._S2 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'Hdiv', self.derham.bc, BasisProjectionOp(self._P2, self._V2, self._fun_S2))

        return self._S2

    @property
    def X2(self):
        r'''Basis projection operator :math:`\mathcal{X}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{X}^2 = \hat{\Pi}_0 \left[ \frac{DF}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^0 \times 3 \times N^2}, 
            \qquad \mathcal{X}^2_{\nu,(ijk),(mno)} := \hat{\Pi}_{0,(ijk)} \left[ \frac{DF}{\sqrt{g}} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_X2'):
            self._X2 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'H1vec', self.derham.bc, BasisProjectionOp(self._P0vec, self._V2, self._fun_X2))

        return self._X2

    @property
    def Y20(self):
        r'''Basis projection operator :math:`\mathcal{Y}^{20}` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{Y}^{20} = \hat{\Pi}_3 \left[ \sqrt{g} \, \mathbf{\Lambda}^0 \right] \in \mathbb{R}^{N^3 \times N^0}, 
            \qquad \mathcal{Y}^{20}_{(ijk),(mno)} := \hat{\Pi}_{3,(ijk)} \left[\sqrt{g} \, \mathbf{\Lambda}^0_{(mno)} \right] \,.

        '''
        if not hasattr(self, '_Y20'):
            self._Y20 = ApplyHomogeneousDirichletToOperator(
                'H1', 'L2', self.derham.bc, BasisProjectionOp(self._P3, self._V0, self._fun_Y20))

        return self._Y20

    @property
    def Z20(self):
        r'''Basis projection operator :math:`\mathcal{Z}^{20}` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{Z}^{20} = \hat{\Pi}_1 \left[ \frac{G}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{Z}^{20}_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\frac{G}{\sqrt{g}} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Z20'):
            self._Z20 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V2, self._fun_Z20))

        return self._Z20

    @property
    def S20(self):
        r'''Basis projection operator :math:`\mathcal{S}^{20}` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{S}^{20} = \hat{\Pi}_1 \left[ \hat{p}_{\text{eq}} \frac{G}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{S}^{20}_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\hat{p}_{\text{eq}} \frac{G}{\sqrt{g}} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S20'):
            self._S20 = ApplyHomogeneousDirichletToOperator(
                'Hdiv', 'Hcurl', self.derham.bc, BasisProjectionOp(self._P1, self._V2, self._fun_S20))

        return self._S20


class BasisProjectionOp( LinOpWithTransp ):
    '''
    Class for "basis projection operators" PI_ijk(fun Lambda_mno).

    Parameters
    ----------
        P : GlobalProjector
            Psydac de Rham projector into space W = P.space (codomain of operator), henceforth called "output space".

        V : TensorFemSpace or ProductFemSpace
            Domain of the operator, henceforth called "input space".

        fun : list
            List of functions of (eta1, eta2, eta3) that multiply the basis functions of the input space V.
            3x3 matrix-valued (nested list [[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]]) if V is ProductFemSPace, 
            scalar-valued [f] list otherwise.

        with_transposed : boolean
            False: map V -> W or True: map W -> V.
    '''

    def __init__(self, P, V, fun, transposed=False):

        assert isinstance(P, GlobalProjector)
        assert isinstance(V, FemSpace)

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        self._P = P
        self._V = V
        self._fun = fun
        self._transposed = transposed

        if transposed:
            self._domain = P.space.vector_space
            self._codomain = V.vector_space
        else:
            self._domain = V.vector_space
            self._codomain = P.space.vector_space

        # Retrieve solver
        self._solver = P.solver

        # Input space: Stencil vector spaces and 1d spaces
        if hasattr(V.symbolic_space, 'name'):
            if V.symbolic_space.name in {'H1', 'L2'}:
                _Vspaces = [V.vector_space]
                _V1ds = [V.spaces]
            else:
                _Vspaces = V.vector_space
                _V1ds = [comp.spaces for comp in V.spaces]
            #print(f'From {V.symbolic_space.name} ...')
        else:
            _Vspaces = V.vector_space
            _V1ds = [comp.spaces for comp in V.spaces]
            #print(f'From H1vec ...')

        # Output space: Stencil vector spaces and 1d spaces
        if hasattr(P.space.symbolic_space, 'name'):
            if P.space.symbolic_space.name in {'H1', 'L2'}:
                _Wspaces = [P.space.vector_space]
                _W1ds = [P.space.spaces]
            else:
                _Wspaces = P.space.vector_space
                _W1ds = [comp.spaces for comp in P.space.spaces]
            #print(f'... to {P.space.symbolic_space.name}.')
        else:
            _Wspaces = P.space.vector_space
            _W1ds = [comp.spaces for comp in P.space.spaces]
            #print(f'... to H1vec.')

        # Retrieve number of quadrature points
        _nqs = []

        for d in range(P.dim):
            if hasattr(P.space.symbolic_space, 'name'):
                if P.space.symbolic_space.name == 'Hcurl':
                    _nqs += [P.grid_x[d][d].shape[1]]

                elif P.space.symbolic_space.name == 'Hdiv':

                    if P.dim == 2:
                        if d == 0:
                            _nqs += [P.grid_x[1][0].shape[1]]
                        elif d == 1:
                            _nqs += [P.grid_x[0][1].shape[1]]
                    else:
                        if d == 0:
                            _nqs += [P.grid_x[2][0].shape[1]]
                        elif d == 1:
                            _nqs += [P.grid_x[0][1].shape[1]]
                        elif d == 2:
                            _nqs += [P.grid_x[1][2].shape[1]]

                elif P.space.symbolic_space.name == 'L2':
                    _nqs += [P.grid_x[0][d].shape[1]]
                else:
                    _nqs += [1]
            else:
                _nqs += [1]

        # Block matrix for dofs
        _blocks = []
        # Ouptut vector space (codomain), row of block
        for Wspace, W1d, fun_line in zip(_Wspaces, _W1ds, fun):
            _blocks += [[]]
            # Input vector space (domain), column of block
            for Vspace, V1d, f in zip(_Vspaces, _V1ds, fun_line):

                # Initiate cell of block matrix
                _dofs_mat = StencilMatrix(
                    Vspace, Wspace, backend=PSYDAC_BACKEND_GPYCCEL)

                _starts_in = _dofs_mat.domain.starts
                _ends_in = _dofs_mat.domain.ends
                _pads_in = _dofs_mat.domain.pads
                _starts_out = _dofs_mat.codomain.starts
                _ends_out = _dofs_mat.codomain.ends
                _pads_out = _dofs_mat.codomain.pads

                _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                    V1d, W1d, _starts_out, _ends_out, _nqs)

                # Evaluate weight function times quadrature weights
                if V.ldim == 1:
                    #_fun_w = evaluate_fun_weights_1d(_ptsG, _wtsG, f)
                    pts1, = np.meshgrid(_ptsG[0].flatten(), indexing='ij')
                    _fun_q = f(pts1).copy().reshape(
                        _ptsG[0].shape[0], _ptsG[0].shape[1])

                elif V.ldim == 2:
                    #_fun_w = evaluate_fun_weights_2d(_ptsG, _wtsG, f)
                    pts1, pts2 = np.meshgrid(
                        _ptsG[0].flatten(), _ptsG[1].flatten(), indexing='ij')
                    _fun_q = f(pts1, pts2).copy().reshape(
                        _ptsG[0].shape[0], _ptsG[0].shape[1], _ptsG[1].shape[0], _ptsG[1].shape[1])

                elif V.ldim == 3:
                    #_fun_w = evaluate_fun_weights_3d(_ptsG, _wtsG, f)
                    pts1, pts2, pts3 = np.meshgrid(
                        _ptsG[0].flatten(), _ptsG[1].flatten(), _ptsG[2].flatten(), indexing='ij')
                    _fun_q = f(pts1, pts2, pts3).copy().reshape(
                        _ptsG[0].shape[0], _ptsG[0].shape[1], _ptsG[1].shape[0], _ptsG[1].shape[1], _ptsG[2].shape[0], _ptsG[2].shape[1])

                # Call the kernel if weight function is not zero
                if np.any(_fun_q):
                    if V.ldim == 1:

                        assemble_1d(_dofs_mat._data,
                                    np.array(_starts_in), np.array(
                                        _ends_in), np.array(_pads_in),
                                    np.array(_starts_out), np.array(
                                        _ends_out), np.array(_pads_out),
                                    _fun_q, _wtsG[0],
                                    _spans[0],
                                    _bases[0],
                                    _subs[0],
                                    V1d[0].nbasis,
                                    W1d[0].degree)

                    elif V.ldim == 2:

                        assemble_2d(_dofs_mat._data,
                                    np.array(_starts_in), np.array(
                                        _ends_in), np.array(_pads_in),
                                    np.array(_starts_out), np.array(
                                        _ends_out), np.array(_pads_out),
                                    _fun_q, _wtsG[0], _wtsG[1],
                                    _spans[0], _spans[1],
                                    _bases[0], _bases[1],
                                    _subs[0], _subs[1],
                                    V1d[0].nbasis, V1d[1].nbasis,
                                    W1d[0].degree, W1d[1].degree)

                    elif V.ldim == 3:

                        assemble_3d(_dofs_mat._data,
                                    np.array(_starts_in), np.array(
                                        _ends_in), np.array(_pads_in),
                                    np.array(_starts_out), np.array(
                                        _ends_out), np.array(_pads_out),
                                    _fun_q, _wtsG[0], _wtsG[1], _wtsG[2],
                                    _spans[0], _spans[1], _spans[2],
                                    _bases[0], _bases[1], _bases[2],
                                    _subs[0], _subs[1], _subs[2],
                                    V1d[0].nbasis, V1d[1].nbasis, V1d[2].nbasis,
                                    W1d[0].degree, W1d[1].degree, W1d[2].degree)

                    _blocks[-1] += [_dofs_mat]

                else:
                    _blocks[-1] += [None]

        _len = sum([len(li) for li in _blocks])

        if _len > 1:
            self._dofs_mat = BlockMatrix(
                V.vector_space, P.space.vector_space, _blocks)
        else:
            if _blocks[0][0] is not None:
                self._dofs_mat = _blocks[0][0]
            else:
                self._dofs_mat = _dofs_mat

        if transposed:
            self._dofs_mat = self._dofs_mat.transpose()

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._V.vector_space.dtype

    @property
    def transposed(self):
        return self._transposed

    def transpose(self):
        return BasisProjectionOp(self._P, self._V, self._fun, True)

    def dot(self, v, out=None):
        '''Applies the basis projection operator to the FE coefficients v belonging to V.

        Parameters
        ----------
            v : StencilVector or BlockVector
                Input FE coefficients from V.vector_space.

        Returns
        -------
            A StencilVector or BlockVector from W.vector_space.'''

        if self.transposed:

            assert v.space == self.domain

            tmp = self._solver.solve(v, transposed=True)
            # tmp.update_ghost_regions()

            assert tmp.space == self.domain

            tmp2 = self._dofs_mat.dot(tmp)
            # tmp2.update_ghost_regions()

            assert tmp2.space == self.codomain

            return tmp2

        else:

            assert v.space == self.domain

            dofs = self._dofs_mat.dot(v)
            # dofs.update_ghost_regions()

            assert dofs.space == self.codomain

            coeffs = self._solver.solve(dofs)
            # coeffs.update_ghost_regions()

            assert coeffs.space == self.codomain

            return coeffs


def prepare_projection_of_basis(V1d, W1d, starts_out, ends_out, n_quad=None):
    '''Obtain knot span indices and basis functions evaluated at projection point sets of a given space.

    Parameters
    ----------
        V1d : 3-list
            Three SplineSpace objects from Psydac from the input space (to be projected).

        W1d : 3-list
            Three SplineSpace objects from Psydac from the output space (projected onto).

        starts_out : 3-list
            Global starting indices of process. 

        ends_out : 3-list
            Global ending indices of process.

        n_quad : 3_list
            Number of quadrature points per histpolation interval. If not given, is set to V1d.degree + 1.

    Returns
    -------
        ptsG : 3-tuple of 2d float arrays
            Quadrature points (or Greville points for interpolation) in each dimension in format (interval, quadrature point).

        wtsG : 3-tuple of 2d float arrays
            Quadrature weights (or ones for interpolation) in each dimension in format (interval, quadrature point).

        spans : 3-tuple of 2d int arrays
            Knot span indices in each direction in format (n, nq).

        bases : 3-tuple of 3d float arrays
            Values of p + 1 non-zero eta basis functions at quadrature points in format (n, nq, basis).'''

    import psydac.core.bsplines as bsp

    x_grid, subs, pts, wts, spans, bases = [], [], [], [], [], []

    # Loop over direction, prepare point sets and evaluate basis functions
    direction = 0
    for space_in, space_out, s, e in zip(V1d, W1d, starts_out, ends_out):

        greville_loc = space_out.greville[s: e + 1]
        histopol_loc = space_out.histopolation_grid[s: e + 2]

        # make sure that greville points used for interpolation are in [0, 1]
        assert np.all(np.logical_and(greville_loc >= 0., greville_loc <= 1.))

        # k += 1
        # print(f'\nrank: {self._mpi_comm.Get_rank()} | Direction {k}, space_out attributes:')
        # # # print('--------------------------------')
        # print(f'rank: {self._mpi_comm.Get_rank()} | breaks       : {space_out.breaks}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | degree       : {space_out.degree}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | kind         : {space_out.basis}')
        # # print(f'rank: {self._mpi_comm.Get_rank()} | greville        : {space_out.greville}')
        # # print(f'rank: {self._mpi_comm.Get_rank()} | greville[s:e+1] : {greville_loc}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | ext_greville : {space_out.ext_greville}')
        # print(f'rank: {self._mpi_comm.Get_rank()} | histopol_grid : {space_out.histopolation_grid}')
        # print(f'rank: {self._mpi_comm.Get_rank()} | histopol_loc : {histopol_loc}')
        # print(f'rank: {self._mpi_comm.Get_rank()} | dim W: {space_out.nbasis}')
        # # # print(f'rank: {self._mpi_comm.Get_rank()} | project' + V1d[0].basis + V1d[1].basis + V1d[2].basis + ' to ' + W1d[0].basis + W1d[1].basis + W1d[2].basis)

        # interpolation
        if space_out.basis == 'B':
            x_grid += [greville_loc]
            pts += [greville_loc[:, None]]
            wts += [np.ones(pts[-1].shape, dtype=float)]

            # sub-interval index is always 0 for interpolation.
            subs += [np.zeros(pts[-1].shape[0], dtype=int)]

        # histopolation
        elif space_out.basis == 'M':

            if space_out.degree % 2 == 0:
                union_breaks = space_out.breaks
            else:
                union_breaks = space_out.breaks[:-1]

            # Make union of Greville and break points
            tmp = set(np.round_(space_out.histopolation_grid, decimals=14)).union(
                np.round_(union_breaks, decimals=14))

            tmp = list(tmp)
            tmp.sort()
            tmp_a = np.array(tmp)

            x_grid += [tmp_a[np.logical_and(tmp_a >= np.min(
                histopol_loc) - 1e-14, tmp_a <= np.max(histopol_loc) + 1e-14)]]

            # determine subinterval index (= 0 or 1):
            subs += [np.zeros(x_grid[-1][:-1].size, dtype=int)]
            for n, x_h in enumerate(x_grid[-1][:-1]):
                add = 1
                for x_g in histopol_loc:
                    if abs(x_h - x_g) < 1e-14:
                        add = 0
                subs[-1][n] += add

            # Gauss - Legendre quadrature points and weights
            if n_quad is None:
                # products of basis functions are integrated exactly
                nq = space_in.degree + 1
            else:
                nq = n_quad[direction]

            pts_loc, wts_loc = np.polynomial.legendre.leggauss(nq)

            x, w = bsp.quadrature_grid(x_grid[-1], pts_loc, wts_loc)

            pts += [x % 1.]
            wts += [w]

        #print(f'rank: {self._mpi_comm.Get_rank()} | Direction {k}, x_grid       : {x_grid[-1]}')

        # Knot span indices and V-basis functions evaluated at W-point sets
        s, b = get_span_and_basis(pts[-1], space_in)

        spans += [s]
        bases += [b]

        direction += 1

    return tuple(pts), tuple(wts), tuple(spans), tuple(bases), tuple(subs)


def get_span_and_basis(pts, space):
    '''Compute the knot span index and the values of p + 1 basis function at each point in pts.

    Parameters
    ----------
        pts : np.array
            2d array of points (interval, quadrature point).

        space : SplineSpace
            Psydac object, the 1d spline space to be projected.

    Returns
    -------
        span : np.array
            2d array indexed by (n, nq), where n is the interval and nq is the quadrature point in the interval.

        basis : np.array
            3d array of values of basis functions indexed by (n, nq, basis function). 
    '''

    import psydac.core.bsplines as bsp

    # Extract knot vectors, degree and kind of basis
    T = space.knots
    p = space.degree

    span = np.zeros(pts.shape, dtype=int)
    basis = np.zeros((*pts.shape, p + 1), dtype=float)

    for n in range(pts.shape[0]):
        for nq in range(pts.shape[1]):
            # avoid 1. --> 0. for clamped interpolation
            x = pts[n, nq] % (1. + 1e-14)
            span_tmp = bsp.find_span(T, p, x)
            basis[n, nq, :] = bsp.basis_funs_all_ders(
                T, p, x, span_tmp, 0, normalization=space.basis)
            span[n, nq] = span_tmp  # % space.nbasis

    return span, basis
