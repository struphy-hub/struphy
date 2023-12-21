import numpy as np

from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockLinearOperator, BlockVector
from psydac.linalg.basic import Vector, IdentityOperator, LinearOperator
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.feec.geom_projectors import PolarCommutingProjector
from struphy.feec.linear_operators import LinOpWithTransp, BoundaryOperator
from struphy.feec import basis_projection_kernels
from struphy.feec.utilities import RotationMatrix

from struphy.polar.linear_operators import PolarExtractionOperator


class BasisProjectionOperators:
    r"""
    Collection of pre-defined :class:`struphy.feec.basis_projection_ops.BasisProjectionOperator`.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.

    domain : :ref:`avail_mappings`
        Mapping from logical unit cube to physical domain and corresponding metric coefficients.

    **weights : dict
        Objects to access callables that can serve as weight functions.

    Note
    ----
    Possible choices for key-value pairs in ****weights** are, at the moment:

    - eq_mhd: :class:`struphy.fields_background.mhd_equil.base.MHDequilibrium`
    """

    def __init__(self, derham, domain, **weights):

        if np.any(np.array(derham.p) == 1):
            if derham.comm.Get_rank() == 0:
                print(
                    f'\nWARNING: Class "BasisProjectionOperators" called with p={derham.p} (interpolation of piece-wise constants should be avoided).\n')

        self._derham = derham
        self._domain = domain
        self._weights = weights

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
    def DF(self, e1, e2, e3):
        '''Jacobian callable.'''
        return self.domain.jacobian(e1, e2, e3, transposed=False, change_out_order=True, squeeze_out=False)

    def DFT(self, e1, e2, e3):
        '''Jacobain transpose callable.'''
        return self.domain.jacobian(e1, e2, e3, transposed=True, change_out_order=True, squeeze_out=False)

    def DFinv(self, e1, e2, e3):
        '''Jacobain inverse callable.'''
        return self.domain.jacobian_inv(e1, e2, e3, transposed=False, change_out_order=True, squeeze_out=False)

    def DFinvT(self, e1, e2, e3):
        '''Jacobian inverse transpose callable.'''
        return self.domain.jacobian_inv(e1, e2, e3, transposed=True, change_out_order=True, squeeze_out=False)

    def G(self, e1, e2, e3):
        '''Metric tensor callable.'''
        return self.domain.metric(e1, e2, e3, change_out_order=True, squeeze_out=False)

    def Ginv(self, e1, e2, e3):
        '''Inverse metric tensor callable.'''
        return self.domain.metric_inv(e1, e2, e3, change_out_order=True, squeeze_out=False)

    def sqrt_g(self, e1, e2, e3):
        '''Jacobian determinant callable.'''
        return abs(self.domain.jacobian_det(e1, e2, e3, squeeze_out=False))

    @property
    def K0(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{K}^{0}_{ijk,mno} := \hat{\Pi}^0_{ijk} \left[  \hat{p}^0_{\text{eq}} \mathbf{\Lambda}^0_{mno} \right] \,.
        '''
        if not hasattr(self, '_K0'):
            fun = [[lambda e1, e2, e3: self.weights['eq_mhd'].p0(e1, e2, e3)]]
            self._K0 = self.assemble_basis_projection_operator(
                fun, 'H1', 'H1', name='K0')

        return self._K0

    @property
    def K3(self):
        r'''Basis projection operator

        .. math::

            \mathcal{K}^3_{ijk,mno} := \hat{\Pi}^3_{ijk} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\Lambda^3_{mno} \right] \,.
        '''
        if not hasattr(self, '_K3'):
            fun = [[lambda e1, e2, e3: self.weights['eq_mhd'].p3(
                e1, e2, e3) / self.sqrt_g(e1, e2, e3)]]
            self._K3 = self.assemble_basis_projection_operator(
                fun, 'L2', 'L2', name='K3')

        return self._K3

    @property
    def Qv(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{Q}^v_{(\mu,ijk),(\nu,mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{\rho}^3_{\text{eq}} \Lambda^{0,\nu}_{mno} \right] \,.

        '''
        if not hasattr(self, '_Qv'):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.weights['eq_mhd'].n3(e1, e2, e3) if m == n else 0*e1]
            self._Qv = self.assemble_basis_projection_operator(
                fun, 'H1vec', 'Hdiv', name='Qv')

        return self._Qv

    @property
    def Q1(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{Q}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{\rho}^3_{\text{eq}}G^{-1}_{\mu,\nu}\Lambda^1_{(\nu, mno)} \right] \,.

        '''
        if not hasattr(self, '_Q1'):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.weights['eq_mhd'].n3(e1, e2, e3) * self.Ginv(e1, e2, e3)[:, :, :, m, n]]

            self._Q1 = self.assemble_basis_projection_operator(
                fun, 'Hcurl', 'Hdiv', name='Q1')

        return self._Q1

    @property
    def Q2(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{Q}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        '''
        if not hasattr(self, '_Q2'):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.weights['eq_mhd'].n3(
                        e1, e2, e3) / self.sqrt_g(e1, e2, e3) if m == n else 0*e1]

            self._Q2 = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'Hdiv', name='Q2')

        return self._Q2

    @property
    def Q3(self):
        r'''Basis projection operator

        .. math::

            \mathcal{Q}^3_{ijk,mno} := \hat{\Pi}^3_{ijk} \left[ \frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\Lambda^3_{mno} \right] \,.
        '''
        if not hasattr(self, '_Q3'):
            fun = [[lambda e1, e2, e3: self.weights['eq_mhd'].n3(
                e1, e2, e3) / self.sqrt_g(e1, e2, e3)]]
            self._Q3 = self.assemble_basis_projection_operator(
                fun, 'L2', 'L2', name='Q3')

        return self._Q3

    @property
    def Tv(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{T}^v_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\mathcal R^B_{\mu,\nu} \Lambda^0_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^B_{\mu, \nu} := \epsilon_{\mu \alpha \nu}\, B^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^B \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium magnetic field (2-form).
        '''
        if not hasattr(self, '_Tv'):

            rot_B = RotationMatrix(
                self.weights['eq_mhd'].b2_1, self.weights['eq_mhd'].b2_2, self.weights['eq_mhd'].b2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: rot_B(e1, e2, e3)[:, :, :, m, n]]

            self._Tv = self.assemble_basis_projection_operator(
                fun, 'H1vec', 'Hcurl', name='Tv')

        return self._Tv

    @property
    def T1(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{T}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\mathcal R^B_{\mu, \beta}G^{-1}_{\beta, \nu}\Lambda^1_{(\nu, mno)} \right] \,,

        with the rotation matrix

        .. math::

            \mathcal R^B_{\mu, \beta} := \epsilon_{\mu \alpha \beta}\, B^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^B \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \beta}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium magnetic field (2-form).

        '''
        if not hasattr(self, '_T1'):

            rot_B = RotationMatrix(
                self.weights['eq_mhd'].b2_1, self.weights['eq_mhd'].b2_2, self.weights['eq_mhd'].b2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: (rot_B(e1, e2, e3) @ self.Ginv(e1, e2, e3))[:, :, :, m, n]]

            self._T1 = self.assemble_basis_projection_operator(
                fun, 'Hcurl', 'Hcurl', name='T1')

        return self._T1

    @property
    def T2(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{T}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\frac{\mathcal R^B_{\mu, \nu}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^B_{\mu, \nu} := \epsilon_{\mu \alpha \nu}\, B^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^B \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium magnetic field (2-form).
        '''
        if not hasattr(self, '_T2'):

            rot_B = RotationMatrix(
                self.weights['eq_mhd'].b2_1, self.weights['eq_mhd'].b2_2, self.weights['eq_mhd'].b2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: rot_B(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._T2 = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'Hcurl', name='T2')

        return self._T2

    @property
    def Sv(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{S}^v_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{p}^3_{\text{eq}} \Lambda^{0,\nu}_{mno} \right] \,.
        '''
        if not hasattr(self, '_Sv'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.weights['eq_mhd'].p3(e1, e2, e3) if m == n else 0*e1]

            self._Sv = self.assemble_basis_projection_operator(
                fun, 'H1vec', 'Hdiv', name='Sv')

        return self._Sv

    @property
    def S1(self):
        r'''Basis projection operator

        .. math::

            \mathcal{S}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{p}^3_{\text{eq}}G^{-1}_{\mu,\nu}\Lambda^1_{(\nu, mno)} \right] \,.
        '''
        if not hasattr(self, '_S1'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.weights['eq_mhd'].p3(e1, e2, e3) * self.Ginv(e1, e2, e3)[:, :, :, m, n]]

            self._S1 = self.assemble_basis_projection_operator(
                fun, 'Hcurl', 'Hdiv', name='S1')

        return self._S1

    @property
    def S2(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{S}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        '''
        if not hasattr(self, '_S2'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.weights['eq_mhd'].p3(
                        e1, e2, e3) / self.sqrt_g(e1, e2, e3) if m == n else 0*e1]

            self._S2 = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'Hdiv', name='S2')

        return self._S2

    @property
    def S11(self):
        r'''Basis projection operator

        .. math::

            \mathcal{S}^{11}_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[ \hat{p}^0_{\text{eq}} \Lambda^1_{(\nu, mno)} \right] \,.
        '''
        if not hasattr(self, '_S11'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.weights['eq_mhd'].p0(e1, e2, e3) if m == n else 0*e1]

            self._S11 = self.assemble_basis_projection_operator(
                fun, 'Hcurl', 'Hcurl', name='S11')

        return self._S11

    @property
    def S21(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{S}^{21}_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\hat{p}_{\text{eq}} \frac{G_{\mu, \nu}}{\sqrt{g}} \Lambda^2_{(\nu, mno)} \right] \,.
        '''
        if not hasattr(self, '_S21'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.weights['eq_mhd'].p0(
                        e1, e2, e3) * self.G(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._S21 = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'Hcurl', name='S21')

        return self._S21

    @property
    def Uv(self):
        r'''Basis projection operator

        .. math::

            \mathcal{U}^v_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\sqrt{g} \, \Lambda^{0, \nu}_{mno} \right] \,.
        '''
        if not hasattr(self, '_Uv'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.sqrt_g(e1, e2, e3) if m == n else 0*e1]

            self._Uv = self.assemble_basis_projection_operator(
                fun, 'H1vec', 'Hdiv', name='Uv')

        return self._Uv

    @property
    def U1(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{U}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[ \sqrt{g} \, G^{-1}_{\mu, \nu} \Lambda^1_{(\nu, mno)} \right] \,.
        '''
        if not hasattr(self, '_U1'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.sqrt_g(e1, e2, e3) * self.Ginv(e1, e2, e3)[:, :, :, m, n]]

            self._U1 = self.assemble_basis_projection_operator(
                fun, 'Hcurl', 'Hdiv', name='U1')

        return self._U1

    @property
    def Xv(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{X}^v_{(\mu,ijk),(\nu,mno)} := \hat{\Pi}^{0, \mu}_{ijk} \left[ DF_{\mu, \nu}\Lambda^{0, \nu}_{mno} \right] \,.
        '''
        if not hasattr(self, '_Xv'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.DF(e1, e2, e3)[:, :, :, m, n]]

            self._Xv = self.assemble_basis_projection_operator(
                fun, 'H1vec', 'H1vec', name='Xv')

        return self._Xv

    @property
    def X1(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{X}^1_{(\mu, ijk),(\nu, mno)} := \hat{\Pi}^{0, \mu}_{ijk} \left[ DF^{-\top}_{\mu, \nu}\Lambda^1_{(\nu, mno)} \right] \,.
        '''
        if not hasattr(self, '_X1'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.DFinvT(e1, e2, e3)[:, :, :, m, n]]

            self._X1 = self.assemble_basis_projection_operator(
                fun, 'Hcurl', 'H1vec', name='X1')

        return self._X1

    @property
    def X2(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{X}^2_{(\mu, ijk),(\nu, mno)} := \hat{\Pi}^{0, \mu}_{ijk} \left[ \frac{DF_{\mu, \nu}}{\sqrt{g}} \Lambda^2_{(\nu, mno)} \right] \,.
        '''
        if not hasattr(self, '_X2'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: self.DF(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._X2 = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'H1vec', name='X2')

        return self._X2

    @property
    def W1(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{W}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\Lambda^1_{(\nu, mno)} \right] \,.
        '''
        if not hasattr(self, '_W1'):

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.weights['eq_mhd'].n3(
                        e1, e2, e3) / self.sqrt_g(e1, e2, e3) if m == n else 0*e1]

            self._W1 = self.assemble_basis_projection_operator(
                fun, 'Hcurl', 'Hcurl', name='W1')

        return self._W1

    @property
    def R1(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{R}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\frac{\mathcal R^J_{\mu, \nu}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^J_{\mu, \nu} := \epsilon_{\mu \alpha \nu}\, J^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec J^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium current density (2-form).
        '''

        if not hasattr(self, '_R1'):

            rot_J = RotationMatrix(
                self.weights['eq_mhd'].j2_1, self.weights['eq_mhd'].j2_2, self.weights['eq_mhd'].j2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: rot_J(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._R1 = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'Hcurl', name='R1')

        return self._R1

    @property
    def R2(self):
        r'''Basis projection operator 

        .. math::

            \mathcal{R}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\mathcal R^J_{\mu, \beta} G^{-1}_{\beta, \nu} \Lambda^2_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^J_{\mu, \beta} := \epsilon_{\mu \alpha \beta}\, J^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec J^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \beta}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium current density (2-form).
        '''
        if not hasattr(self, '_R2'):

            rot_J = RotationMatrix(
                self.weights['eq_mhd'].j2_1, self.weights['eq_mhd'].j2_2, self.weights['eq_mhd'].j2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: (self.Ginv(e1, e2, e3) @ rot_J(e1, e2, e3))[:, :, :, m, n]]

            self._R2 = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'Hdiv', name='R2')

        return self._R2

    @property
    def PB(self):
        r'''
        Basis projection operator

        .. math::

            \mathcal P^b_{ijk, (\mu, mno)} := \hat \Pi^0_{ijk} \left[\frac{1}{\sqrt g} \hat{b]^1_{\text{eq},\mu} \cdot \Lambda^2_{\mu, mno}\right]\,.
        '''
        if not hasattr(self, '_PB'):

            fun = [[]]
            for m in range(3):
                fun[-1] += [lambda e1, e2, e3,
                            m=m: self.weights['eq_mhd'].unit_b1(e1, e2, e3)[m] / self.sqrt_g(e1, e2, e3)]

            self._PB = self.assemble_basis_projection_operator(
                fun, 'Hdiv', 'H1', name='PB')

        return self._PB

    ##########################################
    # Wrapper around BasisProjectionOperator #
    ##########################################
    def assemble_basis_projection_operator(self, fun: list, V_id: str, W_id: str, verbose=True, name=None):
        r"""Basis projection operator :math:`V^\alpha_h \to V^\beta_h` with given (rank 0, 1 or 2) weight function :math:`A(\boldsymbol \eta)`:

        .. math::

            \mathcal P_{(\mu, ijk),(\nu, mno)} = \hat \Pi^\beta_{\mu, ijk} \left( A_{\mu,\nu}\,\Lambda^\alpha_{\nu, mno} \right)\,.

        Here, :math:`\alpha \in \{0, 1, 2, 3, v\}` indicates the domain and :math:`\beta \in \{0, 1, 2, 3, v\}` indicates the co-domain 
        of the operator.

        Parameters
        ----------
        fun : list[list[callable | ndarray]]
            2d list of either all 3d arrays or all scalar functions of eta1, eta2, eta3 (must allow matrix evaluations). 
            3d arrays must have shape corresponding to the 1d quad_grids of V1-VectorFemSpace.

        V_id : str
            Specifier for the domain of the operator ('H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec').

        W_id : str
            Specifier for the co-domain of the operator ('H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec').

        Returns
        -------
        out : A BasisProjectionOperator object.
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

        if self.derham.comm.Get_rank() == 0 and verbose:
            print(
                f'Assembling BasisProjectionOperator "{name}" with V={V_id}, W={W_id}.')

        V_id = self.derham.space_to_form[V_id]
        W_id = self.derham.space_to_form[W_id]

        out = BasisProjectionOperator(self.derham.P[W_id],
                                      self.derham.Vh_fem[V_id],
                                      fun,
                                      self.derham.extraction_ops[V_id],
                                      self.derham.boundary_ops[V_id],
                                      transposed=False,
                                      polar_shift=self.domain.pole)

        if self.derham.comm.Get_rank() == 0 and verbose:
            print('Done.')

        return out


class BasisProjectionOperator(LinOpWithTransp):
    r"""
    Class for assembling basis projection operators in 3d.

    A basis projection operator :math:`\mathcal P: \mathbb R^{N_\alpha} \to \mathbb R^{N_\beta}` is defined by the matrix

    .. math::

        \mathcal P_{(\mu, ijk),(\nu, mno)} = \hat \Pi^\beta_{\mu, ijk} \left( A_{\mu,\nu}\,\Lambda^\alpha_{\nu, mno} \right)\,,

    where the weight fuction :math:`A` is a tensor of rank 0, 1 or 2, depending on domain and co-domain of the operator, and
    :math:`\Lambda^\alpha_{\nu, mno}` is the B-spline basis function with tensor-product index :math:`mno` of the
    :math:`\nu`-th component in the space :math:`V^\alpha_h`. The operator :math:`\hat \Pi^\beta: V^\beta \to \mathbb R^{N_\beta}`
    is a commuting projector from the continuous space
    into the space of coefficients; it can be decomposed into computation of degrees of freedom (DOFs)
    :math:`\sigma^\beta: V^\beta \to \mathbb R^{N_\beta}` and inversion of the inter/-histopolation matrix
    :math:`\mathcal (I^\beta)^{-1}: \mathbb R^{N_\beta} \to \mathbb R^{N_\beta}`:

    .. math::

        \hat \Pi^\beta = (I^\beta)^{-1} \sigma^\beta\,.

    :math:`I^\beta` is usually a Kronecker product and thus fast to invert; this inversion is performed when calling the dot-product
    of the ``BasisProjectionOperator``. The DOFs are precomputed and stored in StencilVector
    format, because the local support of each :math:`\Lambda^\alpha_{\nu, mno}`.

    Finally, extraction and boundary operators can be applied to the DOFs, :math:`B_P * P * \sigma * E_V^T * B_V^T`.

    Parameters
    ----------
    P : struphy.feec.projectors.Projector
        Global commuting projector mapping into TensorFemSpace/VectorFemSpace W = P.space (codomain of operator).

    V : psydac.fem.basic.FemSpace
        Finite element spline space (domain, input space).

    weights : list
        Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain.

    V_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of V.

    V_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions.

    transposed : bool
        Whether to assemble the transposed operator.

    polar_shift : bool
        Whether there are metric coefficients contained in "weights" which are singular at eta1=0. If True, interpolation points at eta1=0 are shifted away from the singularity by 1e-5.

    use_cache : bool
        Whether to store some information computed in _assemble_mat for reuse. Set it to true if planned to update the weights later.
    """

    def __init__(self, P, V, weights, V_extraction_op=None, V_boundary_op=None, transposed=False, polar_shift=False, use_cache=False):

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        assert isinstance(P, PolarCommutingProjector)
        assert isinstance(V, FemSpace)

        self._P = P
        self._V = V

        # set extraction operators
        self._P_extraction_op = P.dofs_extraction_op

        if V_extraction_op is not None:
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.vector_space)

        # set boundary operators
        self._P_boundary_op = P.boundary_op

        if V_boundary_op is not None:
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(V.vector_space)

        self._weights = weights
        self._transposed = transposed
        self._polar_shift = polar_shift
        self._dtype = V.vector_space.dtype
        self._use_cache = use_cache

        # Create cache
        if use_cache:
            self._cache = {}

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

        # Are both space scalar spaces : useful to know if _dof_mat will be Stencil or Block Matrix
        self._is_scalar = True
        if not isinstance(V, TensorFemSpace):
            self._is_scalar = False

        if not isinstance(P.space, TensorFemSpace):
            self._is_scalar = False

        # ============= create and assemble tensor-product dof matrix =======
        if self._is_scalar:
            self._dof_mat = StencilMatrix(V.vector_space, P.space.vector_space)
        else:
            self._dof_mat = BlockLinearOperator(
                V.vector_space, P.space.vector_space)

        self._dof_mat = self._assemble_mat()
        # ========================================================

        # build composed linear operator BP * P * DOF * EV^T * BV^T or transposed
        if transposed:
            self._dof_mat_T = self._dof_mat.T
            self._dof_operator = self._V_boundary_op @ self._V_extraction_op @ self._dof_mat_T @ self._P_extraction_op.T @ self._P_boundary_op.T
        else:
            self._dof_operator = self._P_boundary_op @ self._P_extraction_op @ self._dof_mat @ self._V_extraction_op.T @ self._V_boundary_op.T

        # set domain and codomain
        self._domain = self.dof_operator.domain
        self._codomain = self.dof_operator.codomain

        # temporary vectors for dot product
        self._tmp_dom = self._dof_operator.domain.zeros()
        self._tmp_codom = self._dof_operator.codomain.zeros()

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

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
        v : psydac.linalg.basic.Vector
            Vector the operator shall be applied to.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        tol : float, optional
            Stop tolerance in iterative solve (only used in polar case).

        maxiter : int, optional
            Maximum number of iterations in iterative solve (only used in polar case).

        verbose : bool, optional
            Whether to print some information in each iteration in iterative solve (only used in polar case).

        Returns
        -------
         out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self.domain

        if out is None:

            if self.transposed:
                # 1. apply inverse transposed inter-/histopolation matrix, 2. apply transposed dof operator
                out = self.dof_operator.dot(self._P.solve(
                    v, True, apply_bc=True))
            else:
                # 1. apply dof operator, 2. apply inverse inter-/histopolation matrix
                out = self._P.solve(self.dof_operator.dot(
                    v), False, apply_bc=True)

        else:

            assert isinstance(out, Vector)
            assert out.space == self.codomain

            if self.transposed:
                # 1. apply inverse transposed inter-/histopolation matrix, 2. apply transposed dof operator
                self._P.solve(v, True, apply_bc=True, out=self._tmp_dom)
                self.dof_operator.dot(self._tmp_dom, out=out)
            else:
                # 1. apply dof operator, 2. apply inverse inter-/histopolation matrix
                self.dof_operator.dot(v, out=self._tmp_codom)
                self._P.solve(self._tmp_codom, False, apply_bc=True, out=out)

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return BasisProjectionOperator(self._P, self._V, self._weights,
                                       self._V_extraction_op, self._V_boundary_op,
                                       not self.transposed, self._polar_shift, self._use_cache)

    def update_weights(self, weights):
        '''Updates self.weights and computes new DOF matrix.

        Parameters
        ----------
        weights : list
            Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain.
        '''

        self._weights = weights

        # assemble tensor-product dof matrix
        self._dof_mat = self._assemble_mat()

        # only need to update the transposed in case where it's needed
        # (no need to recreate a new ComposedOperator)
        if self._transposed:
            self._dof_mat_T = self._dof_mat.transpose(out=self._dof_mat_T)

    def _assemble_mat(self):
        """
        Assembles the tensor-product DOF matrix sigma_i(weights[i,j]*Lambda_j), where i=(i1, i2, ...) 
        and j=(j1, j2, ...) depending on the number of spatial dimensions (1d, 2d or 3d). And 
        store it in self._dof_mat.
        """

        # get the needed data :
        V = self._V
        P = self._P.projector_tensor
        weights = self._weights
        polar_shift = self._polar_shift

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
        _nqs = [[P.grid_x[comp][direction].shape[1]
                 for direction in range(V.ldim)] for comp in range(len(_W1ds))]

        # ouptut vector space (codomain), row of block
        for i, (Wspace, W1d, nq, weight_line) in enumerate(zip(_Wspaces, _W1ds, _nqs, weights)):

            _Wdegrees = [space.degree for space in W1d]

            # input vector space (domain), column of block
            for j, (Vspace, V1d, loc_weight) in enumerate(zip(_Vspaces, _V1ds, weight_line)):

                _starts_in = np.array(Vspace.starts)
                _ends_in = np.array(Vspace.ends)
                _pads_in = np.array(Vspace.pads)

                _starts_out = np.array(Wspace.starts)
                _ends_out = np.array(Wspace.ends)
                _pads_out = np.array(Wspace.pads)

                # use cached information if asked
                if self._use_cache:
                    if (i, j) in self._cache:
                        _ptsG, _wtsG, _spans, _bases, _subs = self._cache[(
                            i, j)]
                    else:
                        _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                            V1d, W1d, _starts_out, _ends_out, nq, polar_shift)

                        self._cache[(i, j)] = (
                            _ptsG, _wtsG, _spans, _bases, _subs)
                else:
                    # no cache
                    _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                        V1d, W1d, _starts_out, _ends_out, nq, polar_shift)

                _ptsG = [pts.flatten() for pts in _ptsG]

                _Vnbases = [space.nbasis for space in V1d]

                # Evaluate weight function at quadrature points
                # evaluate weight at quadrature points
                if callable(loc_weight):
                    PTS = np.meshgrid(*_ptsG, indexing='ij')
                    mat_w = loc_weight(*PTS).copy()
                elif isinstance(loc_weight, np.ndarray):
                    mat_w = loc_weight
                elif loc_weight is not None:
                    raise TypeError("weights must be np.ndarray, callable or None")

                # Call the kernel if weight function is not zero or in the scalar case
                # to avoid calling _block of a StencilMatrix in the else
                if loc_weight is not None and np.any(np.abs(mat_w) > 1e-14) or self._is_scalar:

                    # get cell of block matrix (don't instantiate if all zeros)
                    if self._is_scalar:
                        dofs_mat = self._dof_mat
                    else:
                        dofs_mat = self._dof_mat[i, j]

                    if dofs_mat is None:
                        # Maybe in a previous iteration we had more zeros
                        self._dof_mat[i, j] = StencilMatrix(
                            Vspace, Wspace, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                        dofs_mat = self._dof_mat[i, j]

                    kernel = getattr(basis_projection_kernels,
                                     'assemble_dofs_for_weighted_basisfuns_' + str(V.ldim) + 'd')

                    kernel(dofs_mat._data, _starts_in, _ends_in, _pads_in, _starts_out, _ends_out,
                           _pads_out, mat_w, *_wtsG, *_spans, *_bases, *_subs, *_Vnbases, *_Wdegrees)

                    dofs_mat.set_backend(
                        backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)

                else:
                    self._dof_mat[i, j] = None

        return self._dof_mat


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
        Values of p + 1 non-zero eta basis functions at quadrature points in format (n, nq, basis).
    '''

    import psydac.core.bsplines as bsp

    x_grid, subs, pts, wts, spans, bases = [], [], [], [], [], []

    # Loop over direction, prepare point sets and evaluate basis functions
    direction = 0
    for space_in, space_out, s, e in zip(V1d, W1d, starts_out, ends_out):

        greville_loc = space_out.greville[s: e + 1].copy()
        histopol_loc = space_out.histopolation_grid[s: e + 2].copy()

        # make sure that greville points used for interpolation are in [0, 1]
        assert np.all(np.logical_and(greville_loc >= 0., greville_loc <= 1.))

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


class CoordinateProjector(LinearOperator):
    r"""
    Class of projectors on one component of a ProductFemSpace. Represent the projection on the i-th component :

    .. math::
        P_i : X = V_1 \times V_2 \times ... \times V_n \longrightarrow V_i \\
        \mathbf{x} = (x_1,...,x_n) \mapsto x_i


    Parameters
    ----------
    i : int
        The component on which to project

    V : psydac.fem.basic.(Product)FemSpace
        Finite element spline space (domain, input space).

    Vi : psydac.fem.basic.FemSpace
        Finite element spline space (codomain, out space), must be V i-th space
    """

    def __init__(self, i, V, Vi):
        assert isinstance(V, FemSpace)
        assert isinstance(i, int)
        assert V.spaces[i] == Vi

        self.full_space = V
        self.sub_space = Vi
        self.dir = i
        self._domain = V.vector_space
        self._codomain = Vi.vector_space
        self._dtype = V.vector_space.dtype

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return CoordinateInclusion(self.dir, self.full_space, self.sub_space)

    def dot(self, v, out=None):
        assert (v.space == self._domain)
        if out is not None:
            assert out.space == self._codomain
            out *= 0.
            out += v.blocks[self.dir]
        else:
            out = v.blocks[self.dir].copy()
        out.update_ghost_regions()
        return out

    def idot(self, v, out):
        assert (v.space == self._domain)
        assert (out.space == self._codomain)
        out += v.blocks[self.dir]


class CoordinateInclusion(LinearOperator):
    r"""
    Class of inclusion operator from one component of a ProductFemSpace. Represent the canonical inclusion on the i-th component :

    .. math::
        I_i : V_i \longrightarrow X = V_1 \times V_2 \times ... \times V_n \\
        x_i \mapsto \mathbf{x} = (0,...,x_i,...,0)


    Parameters
    ----------
    i : int
        The component on which to project

    V : psydac.fem.basic.(Product)FemSpace
        Finite element spline space (codomain, out space).

    Vi : psydac.fem.basic.FemSpace
        Finite element spline space (domain, in space), must be V i-th space
    """

    def __init__(self, i, V, Vi):
        assert isinstance(V, FemSpace)
        assert isinstance(i, int)
        assert V.spaces[i] == Vi

        self.full_space = V
        self.sub_space = Vi
        self.dir = i
        self._domain = Vi.vector_space
        self._codomain = V.vector_space
        self._dtype = V.vector_space.dtype

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
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return CoordinateProjector(self.dir, self.full_space, self.sub_space)

    def dot(self, v, out=None):
        assert (v.space == self._domain)
        if out is not None:
            assert out.space == self._codomain
            out *= 0.
            out._blocks[self.dir] += v

        else:
            blocks = [sspace.zeros() for sspace in self.codomain.spaces]
            blocks[self.dir] = v.copy()
            out = BlockVector(self._codomain, blocks)

        out.update_ghost_regions()
        return out

    def idot(self, v, out):
        assert (v.space == self._domain)
        assert (out.space == self._codomain)
        out._blocks[self.dir] += v
