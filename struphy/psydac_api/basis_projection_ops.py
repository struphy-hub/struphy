import numpy as np

from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.feec.global_projectors import GlobalProjector
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api.projectors import Projector
from struphy.psydac_api.linear_operators import LinOpWithTransp, CompositeLinearOperator, IdentityOperator, BoundaryOperator
from struphy.psydac_api import basis_projection_kernels

from struphy.polar.linear_operators import PolarExtractionOperator


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

        # Wrapper functions for evaluating metric coefficients in right order (3x3 entries are last two axes!!)
        flat_eval = False
        squeeze_output = False
        change_out_order = True
        
        def DF(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, flat_eval, squeeze_output, change_out_order, transposed=False)

        def DFT(e1, e2, e3):
            return domain.jacobian(e1, e2, e3, flat_eval, squeeze_output, change_out_order, transposed=True)
            
        def DFinv(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, flat_eval, squeeze_output, change_out_order, transposed=False)

        def DFinvT(e1, e2, e3):
            return domain.jacobian_inv(e1, e2, e3, flat_eval, squeeze_output, change_out_order, transposed=True)

        def G(e1, e2, e3):
            return domain.metric(e1, e2, e3, flat_eval, squeeze_output, change_out_order)
            
        def Ginv(e1, e2, e3):
            return domain.metric_inv(e1, e2, e3, flat_eval, squeeze_output, change_out_order)
            
        def sqrt_g(e1, e2, e3):
            return abs(domain.jacobian_det(e1, e2, e3, flat_eval, squeeze_output))
        
        # Cross product matrices and evaluation of cross products
        cross_mask = [[ 1, -1,  1], 
                      [ 1,  1, -1], 
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
        fun_Q0, fun_Tv, fun_S0, fun_Uv, fun_X0 = [], [], [], [], []

        fun_Q1, fun_W1, fun_U1, fun_R1, fun_S1, fun_T1, fun_X1, fun_S10 = [], [], [], [], [], [], [], []
        
        fun_Q2, fun_T2, fun_R2, fun_S2, fun_X2, fun_Z20, fun_S20 = [], [], [], [], [], [], []

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
            self._K0 = BasisProjectionOperator(self.derham.P['3'], self.derham.Vh_fem['3'], self._fun_K0, 
                                               self.derham.E['3'], self.derham.B['3'],
                                               transposed=False, polar_shift=self.domain.pole)

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
            self._Q0 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['v'], self._fun_Q0,
                                               self.derham.E['v'], self.derham.B['v'],
                                               transposed=False, polar_shift=self.domain.pole)

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
            self._Tv = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['v'], self._fun_Tv, 
                                               self.derham.E['v'], self.derham.B['v'],
                                               transposed=False, polar_shift=self.domain.pole)

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
            self._S0 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['v'], self._fun_S0,
                                               self.derham.E['v'], self.derham.B['v'],
                                               transposed=False, polar_shift=self.domain.pole)

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
            self._Uv = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['v'], self._fun_Uv,
                                               self.derham.E['v'], self.derham.B['v'],
                                               transposed=False, polar_shift=self.domain.pole)
            
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
            self._X0 = BasisProjectionOperator(self.derham.P['v'], self.derham.Vh_fem['v'], self._fun_X0,
                                               self.derham.E['v'], self.derham.B['v'],
                                               transposed=False, polar_shift=self.domain.pole)

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
            self._K1 = BasisProjectionOperator(self.derham.P['3'], self.derham.Vh_fem['3'], self._fun_K1,
                                               self.derham.E['3'], self.derham.B['3'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._K1

    @property
    def Q1(self):
        r'''Basis projection operator :math:`\mathcal{Q}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{Q}^1 = \hat{\Pi}_2 \left[ \hat{\rho}^3_{\text{eq}} G^{-1} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^2 \times N^1}, 
            \qquad \mathcal{Q}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\hat{\rho}^3_{\text{eq}}G^{-1}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Q1'):
            self._Q1 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['1'], self._fun_Q1,
                                               self.derham.E['1'], self.derham.B['1'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._Q1

    @property
    def W1(self):
        r'''Basis projection operator :math:`\mathcal{W}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{W}^1 = \hat{\Pi}_1 \left[ \frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^1 \times N^1}, 
            \qquad \mathcal{W}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_W1'):
            self._W1 = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['1'], self._fun_W1,
                                               self.derham.E['1'], self.derham.B['1'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._W1

    @property
    def U1(self):
        r'''Basis projection operator :math:`\mathcal{U}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{U}^1 = \hat{\Pi}_2 \left[ \sqrt{g} \, G^{-1} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^2 \times N^1}, 
            \qquad \mathcal{U}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[ \sqrt{g} \, G^{-1} \mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_U1'):
            self._U1 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['1'], self._fun_U1,
                                               self.derham.E['1'], self.derham.B['1'],
                                               transposed=False, polar_shift=self.domain.pole)
            
        return self._U1

    @property
    def R1(self):
        r'''Basis projection operator :math:`\mathcal{R}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{R}^1 = \hat{\Pi}_1 \left[ \frac{\hat{J}^2_{\text{eq}}}{\sqrt{g}} \times \mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{R}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\epsilon_{\mu \alpha \nu}\frac{\hat{J}^2_{\text{eq}, \alpha}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_R1'):
            self._R1 = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['2'], self._fun_R1,
                                               self.derham.E['2'], self.derham.B['2'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._R1

    @property
    def S1(self):
        r'''Basis projection operator :math:`\mathcal{S}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{S}^1 = \hat{\Pi}_2 \left[ \hat{p}^3_{\text{eq}} G^{-1} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^2 \times N^1}, 
            \qquad \mathcal{S}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\hat{p}^3_{\text{eq}}G^{-1}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S1'):
            self._S1 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['1'], self._fun_S1,
                                               self.derham.E['1'], self.derham.B['1'],
                                               transposed=False, polar_shift=self.domain.pole)
            
        return self._S1

    @property
    def T1(self):
        r'''Basis projection operator :math:`\mathcal{T}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{T}^1 = \hat{\Pi}_1 \left[ \hat{B}^2_{\text{eq}} \times (G^{-1} \mathbf{\vec{\Lambda}}^1) \right] \in \mathbb{R}^{N^1 \times N^1}, 
            \qquad \mathcal{T}^1_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\epsilon_{\mu \alpha \beta} \hat{B}^2_{\text{eq},\alpha}G^{-1}_{\beta \nu}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_T1'):
            self._T1 = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['1'], self._fun_T1,
                                               self.derham.E['1'], self.derham.B['1'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._T1

    @property
    def X1(self):
        r'''Basis projection operator :math:`\mathcal{X}^1` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{X}^1 = \hat{\Pi}_0 \left[ DF^{-\top}\mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^0 \times 3 \times N^1}, 
            \qquad \mathcal{X}^1_{\nu,(ijk),(mno)} := \hat{\Pi}_{0,(ijk)} \left[ DF^{-\top}\mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_X1'):
            self._X1 = BasisProjectionOperator(self.derham.P['v'], self.derham.Vh_fem['1'], self._fun_X1,
                                               self.derham.E['1'], self.derham.B['1'],
                                               transposed=False, polar_shift=self.domain.pole)
            
        return self._X1

    @property
    def K10(self):
        r'''Basis projection operator :math:`\mathcal{K}^{10}` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{K}^{10} = \hat{\Pi}_3 \left[ \hat{p}^0_{\text{eq}} \mathbf{\Lambda}^0 \right] \in \mathbb{R}^{N^0 \times N^0}, 
            \qquad \mathcal{K}^{10}_{(ijk),(mno)} := \hat{\Pi}_{3,(ijk)} \left[  \hat{p}^0_{\text{eq}} \mathbf{\Lambda}^0_{(mno)} \right] \,.

        '''
        if not hasattr(self, '_K10'):
            self._K10 = BasisProjectionOperator(self.derham.P['0'], self.derham.Vh_fem['0'], self._fun_K10,
                                                self.derham.E['0'], self.derham.B['0'],
                                                transposed=False, polar_shift=self.domain.pole)

        return self._K10

    @property
    def S10(self):
        r'''Basis projection operator :math:`\mathcal{S}^{10}` with the velocity as 1-form :math:`\hat{\mathbf{U}}^1` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{S}^{10} = \hat{\Pi}_1 \left[ \hat{p}^0_{\text{eq}} \mathbf{\vec{\Lambda}}^1 \right] \in \mathbb{R}^{N^1 \times N^1}, 
            \qquad \mathcal{S}^{10}_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[ \hat{p}^0_{\text{eq}} \mathbf{\vec{\Lambda}}^1_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S10'):
            self._S10 = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['1'], self._fun_S10,
                                                self.derham.E['1'], self.derham.B['1'],
                                                transposed=False, polar_shift=self.domain.pole)

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
            self._K2 = BasisProjectionOperator(self.derham.P['3'], self.derham.Vh_fem['3'], self._fun_K2,
                                               self.derham.E['3'], self.derham.B['3'],
                                               transposed=False, polar_shift=self.domain.pole)
            
        return self._K2

    @property
    def Q2(self):
        r'''Basis projection operator :math:`\mathcal{Q}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{Q}^2 = \hat{\Pi}_2 \left[ \frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^2 \times N^2}, 
            \qquad \mathcal{Q}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Q2'):
            self._Q2 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['2'], self._fun_Q2,
                                               self.derham.E['2'], self.derham.B['2'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._Q2

    @property
    def T2(self):
        r'''Basis projection operator :math:`\mathcal{T}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{T}^2 = \hat{\Pi}_1 \left[ \frac{\hat{B}^2_{\text{eq}}}{\sqrt{g}} \times \mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{T}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\epsilon_{\mu \alpha \nu} \frac{\hat{B}^2_{\text{eq},\alpha}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_T2'):
            self._T2 = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['2'], self._fun_T2,
                                               self.derham.E['2'], self.derham.B['2'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._T2

    @property
    def R2(self):
        r'''Basis projection operator :math:`\mathcal{R}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{R}^2 = \hat{\Pi}_2 \left[\hat{J}^2_{\text{eq}} \times (G^{-1} \mathbf{\vec{\Lambda}}^2) \right] \in \mathbb{R}^{N^2 \times N^2}, 
            \qquad \mathcal{R}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\epsilon_{\mu \alpha \beta}\hat{J}^2_{\text{eq}, \alpha} G^{-1}_{\beta \nu} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_R2'):
            self._R2 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['2'], self._fun_R2,
                                               self.derham.E['2'], self.derham.B['2'],
                                               transposed=False, polar_shift=self.domain.pole)
            
        return self._R2

    @property
    def S2(self):
        r'''Basis projection operator :math:`\mathcal{S}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{S}^2 = \hat{\Pi}_2 \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^2 \times N^2}, 
            \qquad \mathcal{S}^2_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{2,\mu,(ijk)} \left[\frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S2'):
            self._S2 = BasisProjectionOperator(self.derham.P['2'], self.derham.Vh_fem['2'], self._fun_S2,
                                               self.derham.E['2'], self.derham.B['2'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._S2

    @property
    def X2(self):
        r'''Basis projection operator :math:`\mathcal{X}^2` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 3-form  :math:`\hat{p}^3`.

        .. math::

            \mathcal{X}^2 = \hat{\Pi}_0 \left[ \frac{DF}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^0 \times 3 \times N^2}, 
            \qquad \mathcal{X}^2_{\nu,(ijk),(mno)} := \hat{\Pi}_{0,(ijk)} \left[ \frac{DF}{\sqrt{g}} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_X2'):
            self._X2 = BasisProjectionOperator(self.derham.P['v'], self.derham.Vh_fem['2'], self._fun_X2,
                                               self.derham.E['2'], self.derham.B['2'],
                                               transposed=False, polar_shift=self.domain.pole)

        return self._X2

    @property
    def Y20(self):
        r'''Basis projection operator :math:`\mathcal{Y}^{20}` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{Y}^{20} = \hat{\Pi}_3 \left[ \sqrt{g} \, \mathbf{\Lambda}^0 \right] \in \mathbb{R}^{N^3 \times N^0}, 
            \qquad \mathcal{Y}^{20}_{(ijk),(mno)} := \hat{\Pi}_{3,(ijk)} \left[\sqrt{g} \, \mathbf{\Lambda}^0_{(mno)} \right] \,.

        '''
        if not hasattr(self, '_Y20'):
            self._Y20 = BasisProjectionOperator(self.derham.P['3'], self.derham.Vh_fem['0'], self._fun_Y20,
                                                self.derham.E['0'], self.derham.B['0'],
                                                transposed=False, polar_shift=self.domain.pole)

        return self._Y20

    @property
    def Z20(self):
        r'''Basis projection operator :math:`\mathcal{Z}^{20}` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{Z}^{20} = \hat{\Pi}_1 \left[ \frac{G}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{Z}^{20}_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\frac{G}{\sqrt{g}} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_Z20'):
            self._Z20 = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['2'], self._fun_Z20,
                                                self.derham.E['2'], self.derham.B['2'],
                                                transposed=False, polar_shift=self.domain.pole)

        return self._Z20

    @property
    def S20(self):
        r'''Basis projection operator :math:`\mathcal{S}^{20}` with the velocity as 2-form :math:`\hat{\mathbf{U}}^2` and the pressure as 0-form  :math:`\hat{p}^0`.

        .. math::

            \mathcal{S}^{20} = \hat{\Pi}_1 \left[ \hat{p}_{\text{eq}} \frac{G}{\sqrt{g}}\mathbf{\vec{\Lambda}}^2 \right] \in \mathbb{R}^{N^1 \times N^2}, 
            \qquad \mathcal{S}^{20}_{\mu\nu,(ijk),(mno)} := \hat{\Pi}_{1,\mu,(ijk)} \left[\hat{p}_{\text{eq}} \frac{G}{\sqrt{g}} \mathbf{\vec{\Lambda}}^2_{\nu,(mno)} \right] \,.

        '''
        if not hasattr(self, '_S20'):
            self._S20 = BasisProjectionOperator(self.derham.P['1'], self.derham.Vh_fem['2'], self._fun_S20,
                                                self.derham.E['2'], self.derham.B['2'],
                                                transposed=False, polar_shift=self.domain.pole)

        return self._S20


class BasisProjectionOperator( LinOpWithTransp ):
    """
    Class for "basis projection operators" PI_ijk(fun Lambda_mno) in the general form BP * P * DOF * EV^T * BV^T.

    Parameters
    ----------
        P : Projector
            Global commuting projector mapping into TensorFemSpace/ProductFemSpace W = P.space (codomain of operator).

        V : TensorFemSpace | ProductFemSpace
            Tensor product spline space from psydac.fem.tensor (domain, input space).

        fun : list
            Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.
            
        V_extraction_op : PolarExtractionOperator | IdentityOperator
            Extraction operator to polar sub-space of V.
            
        V_boundary_op : BoundaryOperator | IdentityOperator
            Boundary operator that sets essential boundary conditions.

        transposed : bool
            Whether to assemble the transposed operator.
            
        polar_shift : bool
                Whether there are metric coefficients contained in "fun" which are singular at eta1=0. If True, interpolation points at eta1=0 are shifted away from the singularity by 1e-5.
    """

    def __init__(self, P, V, fun, V_extraction_op=None, V_boundary_op=None, transposed=False, polar_shift=False):

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
        
        assert isinstance(P, Projector)
        assert isinstance(V, FemSpace)

        self._P = P
        self._V = V
        
        # set extraction operators
        self._P_extraction_op = P.dofs_extraction_op
        
        if V_extraction_op is not None:
            assert isinstance(V_extraction_op, (PolarExtractionOperator, IdentityOperator))
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.vector_space)
        
        # set boundary operators
        self._P_boundary_op = P.boundary_op
        
        if V_boundary_op is not None:
            assert isinstance(V_boundary_op, (BoundaryOperator, IdentityOperator))
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(V.vector_space)
        
        self._fun = fun
        self._transposed = transposed
        self._polar_shift = polar_shift
        self._dtype = V.vector_space.dtype
        
        # set domain and codomain symbolic names
        if hasattr(P.space.symbolic_space, 'name'):
            P_name = P.space.symbolic_space.name
        else:
            P_name = 'H1vec'

        if hasattr(V.symbolic_space, 'name'):
            V_name = V.symbolic_space.name
        else:
            V_name = 'H1vec'

        if transposed:
            self._domain_symbolic_name = P_name
            self._codomain_symbolic_name = V_name
        else:
            self._domain_symbolic_name = V_name
            self._codomain_symbolic_name = P_name
            
        # ============= assemble tensor-product dof matrix =======
        dof_mat = BasisProjectionOperator.assemble_mat(P.projector_tensor, V, fun, polar_shift)
        # ========================================================
        
        # build composite linear operator BP * P * DOF * EV^T * BV^T
        self._dof_operator = CompositeLinearOperator(self._P_boundary_op, self._P_extraction_op, dof_mat, self._V_extraction_op.transpose(), self._V_boundary_op.transpose())
        
        if transposed:
            self._dof_operator = self._dof_operator.transpose()
            
        # set domain and codomain
        self._domain = self.dof_operator.domain
        self._codomain = self.dof_operator.codomain

    @property
    def domain(self):
        """ Domain vector space (input) of the operator.
        """
        return self._domain

    @property
    def codomain(self):
        """ Codomain vector space (input) of the operator.
        """
        return self._codomain

    @property
    def dtype(self):
        """ Datatype of the operator.
        """
        return self._dtype

    @property
    def transposed(self):
        """ If the transposed operator is in play.
        """
        return self._transposed
    
    @property
    def dof_operator(self):
        """ The degrees of freedom operator as composite linear operator containing polar extraction and boundary operators. 
        """
        return self._dof_operator

    def dot(self, v, out=None, tol=1e-14, maxiter=1000, verbose=False):
        """
        Applies the basis projection operator to the FE coefficients v.

        Parameters
        ----------
            v : StencilVector | BlockVector | PolarVector
                Vector the operator shall be applied to.
                
            tol : float
                Stop tolerance in iterative solve (only used in polar case).
                
            maxiter : int
                Maximum number of iterations in iterative solve (only used in polar case).
                
            verbose : bool
                Whether to print some information in each iteration in iterative solve (only used in polar case).

        Returns
        -------
            out : StencilVector | BlockVector | PolarVector
                The result of the dot product.
        """

        assert v.space == self.domain
        
        if self.transposed:
            # 1. apply inverse transposed inter-/histopolation matrix, 2. apply transposed dof operator
            out = self.dof_operator.dot(self._P.solve(v, True, apply_bc=True, tol=tol, maxiter=maxiter, verbose=verbose))
        else:
            # 1. apply dof operator, 2. apply inverse inter-/histopolation matrix
            out = self._P.solve(self.dof_operator.dot(v), False, apply_bc=True, tol=tol, maxiter=maxiter, verbose=verbose)

        assert out.space == self.codomain

        return out
        
    def transpose(self):
        """
        Returns the transposed operator.
        """
        return BasisProjectionOperator(self._P, self._V, self._fun, 
                                       self._V_extraction_op, self._V_boundary_op,
                                       not self.transposed, self._polar_shift)
    
    @staticmethod
    def assemble_mat(P, V, fun, polar_shift=False):
        """
        Assembles the tensor-product DOF matrix sigma_i(fun*Lambda_j), where i=(i1, i2, ...) and j=(j1, j2, ...) depending on the number of spatial dimensions (1d, 2d or 3d).
        
        Parameters
        ----------
            P : GlobalProjector
                The psydac global tensor product projector defining the space onto which the input shall be projected.
                
            V : TensorFemSpace | ProductFemSpace
                The spline space which shall be projected.
                
            fun : list
                Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.
                
            polar_shift : bool
                Whether there are metric coefficients contained in "fun" which are singular at eta1=0. If True, interpolation points at eta1=0 are shifted away from the singularity by 1e-5.
                
        Returns
        -------
            dof_mat : StencilMatrix | BlockMatrix
                Degrees of freedom matrix in the full tensor product setting.
        """
        
        # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component 
        if isinstance(V, TensorFemSpace):
            _Vspaces = [V.vector_space]
            _V1ds = [V.spaces]
        else:
            _Vspaces = V.vector_space
            _V1ds = [comp.spaces for comp in V.spaces]

        # output space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(P.space, TensorFemSpace):
            _Wspaces = [P.space.vector_space]
            _W1ds = [P.space.spaces]
        else:
            _Wspaces = P.space.vector_space
            _W1ds = [comp.spaces for comp in P.space.spaces]
            
        # retrieve number of quadrature points of each component (=1 for interpolation)
        _nqs = [[P.grid_x[comp][direction].shape[1] for direction in range(V.ldim)] for comp in range(len(_W1ds))]

        # blocks of dof matrix
        blocks = []
        
        # ouptut vector space (codomain), row of block
        for Wspace, W1d, nq, fun_line in zip(_Wspaces, _W1ds, _nqs, fun):
            blocks += [[]]
            _Wdegrees = [space.degree for space in W1d]
            
            # input vector space (domain), column of block
            for Vspace, V1d, f in zip(_Vspaces, _V1ds, fun_line):

                # instantiate cell of block matrix
                dofs_mat = StencilMatrix(Vspace, Wspace, backend=PSYDAC_BACKEND_GPYCCEL)

                _starts_in = np.array(dofs_mat.domain.starts)
                _ends_in = np.array(dofs_mat.domain.ends)
                _pads_in = np.array(dofs_mat.domain.pads)
                
                _starts_out = np.array(dofs_mat.codomain.starts)
                _ends_out = np.array(dofs_mat.codomain.ends)
                _pads_out = np.array(dofs_mat.codomain.pads)

                _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                    V1d, W1d, _starts_out, _ends_out, nq, polar_shift)
                
                _ptsG = [pts.flatten() for pts in _ptsG]
                
                _Vnbases = [space.nbasis for space in V1d]

                # Evaluate weight function at quadrature points
                pts = np.meshgrid(*_ptsG, indexing='ij')
                _fun_q = f(*pts).copy()

                # Call the kernel if weight function is not zero
                if np.any(np.abs(_fun_q) > 1e-14):
                    
                    kernel = getattr(basis_projection_kernels, 'assemble_dofs_for_weighted_basisfuns_' + str(V.ldim) + 'd')
                    
                    kernel(dofs_mat._data, _starts_in, _ends_in, _pads_in, _starts_out, _ends_out, _pads_out, _fun_q, *_wtsG, *_spans, *_bases, *_subs, *_Vnbases, *_Wdegrees)

                    blocks[-1] += [dofs_mat]

                else:
                    blocks[-1] += [None]
        
        # build BlockMatrix (if necessary) and return
        if len(blocks) == len(blocks[0]) == 1:
            if blocks[0][0] is not None:
                return blocks[0][0]
            else:
                return dofs_mat
        else:
            return BlockMatrix(V.vector_space, P.space.vector_space, blocks)


def prepare_projection_of_basis(V1d, W1d, starts_out, ends_out, n_quad=None, polar_shift=False):
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

        greville_loc = space_out.greville[s: e + 1].copy()
        histopol_loc = space_out.histopolation_grid[s: e + 2].copy()

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
            
            # !! shift away first interpolation point in eta_1 direction for polar domains !!
            if direction == 0 and pts[0][0] == 0. and polar_shift:
                pts[0][0] += 0.00001

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
