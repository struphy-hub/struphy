import numpy as np

from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockLinearOperator
from psydac.linalg.basic import Vector, IdentityOperator

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.feec import mass_kernels
from struphy.feec.utilities import RotationMatrix
from struphy.feec.linear_operators import LinOpWithTransp


class WeightedMassOperators:
    r"""
    Collection of pre-defined :class:`struphy.feec.mass.WeightedMassOperator`.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
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
            self._M0 = self.assemble_weighted_mass(fun, 'H1', 'H1', name='M0')

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

            self._M1 = self.assemble_weighted_mass(
                fun, 'Hcurl', 'Hcurl', name='M1')

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

            self._M2 = self.assemble_weighted_mass(
                fun, 'Hdiv', 'Hdiv', name='M2')

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
            self._M3 = self.assemble_weighted_mass(fun, 'L2', 'L2', name='M3')

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

            self._Mv = self.assemble_weighted_mass(
                fun, 'H1vec', 'H1vec', name='Mv')

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

            self._M1n = self.assemble_weighted_mass(
                fun, 'Hcurl', 'Hcurl', name='M1n')

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

            self._M2n = self.assemble_weighted_mass(
                fun, 'Hdiv', 'Hdiv', name='M2n')

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

            self._Mvn = self.assemble_weighted_mass(
                fun, 'H1vec', 'H1vec', name='Mvn')

        return self._Mvn

    @property
    def M1ninv(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{1,\frac{1}{n}}_{(\mu,ijk), (\nu,mno)} = \int \frac{1}{n^0_{\textnormal{eq}}(\boldsymbol \eta)} \Lambda^1_{\mu,ijk}\, G^{-1}_{\mu,\nu}\, \Lambda^1_{\nu, mno} \sqrt g\,  \textnormal d \boldsymbol\eta. 

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol \eta)` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, '_M1ninv'):
            assert 'eq_mhd' in self.weights
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m, n=n: self.Ginv(e1, e2, e3)[:, :, :, m, n] * self.sqrt_g(
                        e1, e2, e3) / self.weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False)]

            self._M1ninv = self.assemble_weighted_mass(fun, 'Hcurl', 'Hcurl', name='M1ninv')

        return self._M1ninv

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

            self._M1J = self.assemble_weighted_mass(
                fun, 'Hdiv', 'Hcurl', name='M1J')

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
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: rot_J(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._M2J = self.assemble_weighted_mass(
                fun, 'Hdiv', 'Hdiv', name='M2J')

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
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: rot_J(e1, e2, e3)[:, :, :, m, n]]

            self._MvJ = self.assemble_weighted_mass(
                fun, 'Hdiv', 'H1vec', name='MvJ')

        return self._MvJ

    @property
    def M2B(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{2,B}_{(\mu,ijk), (\nu,mno)} = \int \Lambda^2_{\mu,ijk}\, \mathcal R^J_{\alpha, \nu}\, \Lambda^2_{\nu, mno} \, \frac{1}{\sqrt g}\,  \textnormal d \boldsymbol\eta. 

        with the rotation matrix

        .. math::

            \mathcal R^J_{\alpha, \nu} := \epsilon_{\alpha \beta \nu}\, B^2_{\textnormal{eq}, \beta}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium magnetic field (2-form).
        """

        if not hasattr(self, '_M2B'):

            a_eq = self.derham.P['1']([self.weights['eq_mhd'].a1_1,
                                       self.weights['eq_mhd'].a1_2,
                                       self.weights['eq_mhd'].a1_3])

            tmp_a2 = self.derham.curl.dot(a_eq)
            b02fun = self.derham.create_field('b02', 'Hdiv')
            b02fun.vector = tmp_a2

            def b02funx(x, y, z): return b02fun(
                x, y, z, squeeze_output=True, local=True)[0]

            def b02funy(x, y, z): return b02fun(
                x, y, z, squeeze_output=True, local=True)[1]
            def b02funz(x, y, z): return b02fun(
                x, y, z, squeeze_output=True, local=True)[2]
            # rot_B = RotationMatrix(
            # self.weights['eq_mhd'].b2_1, self.weights['eq_mhd'].b2_2, self.weights['eq_mhd'].b2_3)
            rot_B = RotationMatrix(
                b02funx, b02funy, b02funz)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: rot_B(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3)]

            self._M2B = self.assemble_weighted_mass(
                fun, 'Hdiv', 'Hdiv', name='M2B')

        return self._M2B

    @property
    def M2Bn(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{2,BN}_{(\mu,ijk), (\nu,mno)} = \int \Lambda^2_{\mu,ijk}\, \mathcal R^J_{\alpha, \nu}\, \Lambda^2_{\nu, mno} \, \frac{1}{n^0_{\textnormal{eq}}(\boldsymbol \eta)}\, \frac{1}{\sqrt g}\,  \textnormal d \boldsymbol\eta. 

        with the rotation matrix

        .. math::

            \mathcal R^J_{\alpha, \nu} := \epsilon_{\alpha \beta \nu}\, B^2_{\textnormal{eq}, \beta}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium magnetic field (2-form).
        """

        if not hasattr(self, '_M2BN'):

            a_eq = self.derham.P['1']([self.weights['eq_mhd'].a1_1,
                                       self.weights['eq_mhd'].a1_2,
                                       self.weights['eq_mhd'].a1_3])

            tmp_a2 = self.derham.Vh['2'].zeros()
            self.derham.curl.dot(a_eq, out=tmp_a2)
            b02fun = self.derham.create_field('b02', 'Hdiv')
            b02fun.vector = tmp_a2

            def b02funx(x, y, z): return b02fun(
                x, y, z, squeeze_output=True, local=True)[0]

            def b02funy(x, y, z): return b02fun(
                x, y, z, squeeze_output=True, local=True)[1]
            def b02funz(x, y, z): return b02fun(
                x, y, z, squeeze_output=True, local=True)[2]
            # rot_B = RotationMatrix(
            # self.weights['eq_mhd'].b2_1, self.weights['eq_mhd'].b2_2, self.weights['eq_mhd'].b2_3)
            rot_B = RotationMatrix(
                b02funx, b02funy, b02funz)
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: rot_B(e1, e2, e3)[:, :, :, m, n] / (self.sqrt_g(e1, e2, e3) * self.weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False))]

            self._M2BN = self.assemble_weighted_mass(
                fun, 'Hdiv', 'Hdiv', name='M2Bn')

        return self._M2BN

    @property
    def M1Bninv(self):
        r"""
        Mass matrix 

        .. math::

            \mathbb M^{1,B\frac{1}{n}}_{(\mu,ijk), (\nu,mno)} = \int \frac{1}{n^0_{\textnormal{eq}}(\boldsymbol \eta)}\, \Lambda^1_{\mu,ijk}\, G^{-1}_{\mu,\alpha}\, \mathcal R^J_{\alpha, \gamma}\, G^{-1}_{\gamma,\nu}\, \Lambda^1_{\nu, mno} \, \sqrt g\,  \textnormal d \boldsymbol\eta. 

        with the rotation matrix

        .. math::

            \mathcal R^J_{\alpha, \nu} := \epsilon_{\alpha \beta \nu}\, B^2_{\textnormal{eq}, \beta}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium magnetic field (2-form).
        """

        if not hasattr(self, '_M1Bninv'):

            rot_B = RotationMatrix(
                self.weights['eq_mhd'].b2_1, self.weights['eq_mhd'].b2_2, self.weights['eq_mhd'].b2_3)

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [lambda e1, e2, e3, m=m,
                                n=n: (self.Ginv(e1, e2, e3) @ rot_B(e1, e2, e3) @ self.Ginv(e1, e2, e3))[:, :, :, m, n] * (self.sqrt_g(e1, e2, e3) / self.weights['eq_mhd'].n0(e1, e2, e3, squeeze_out=False))]

            self._M1Bninv = self.assemble_weighted_mass(fun, 'Hcurl', 'Hcurl', name='M1Bninv')

        return self._M1Bninv

    #######################################
    # Wrapper around WeightedMassOperator #
    #######################################
    def assemble_weighted_mass(self, fun: list, V_id: str, W_id: str, name=None):
        r""" Weighted mass matrix :math:`V^\alpha_h \to V^\beta_h` with given (matrix-valued) weight function :math:`W(\boldsymbol \eta)`:

        .. math::

            \mathbb M_{(\mu, ijk), (\nu, mno)}(W) = \int \Lambda^\beta_{\mu, ijk}\, W_{\mu,\nu}(\boldsymbol \eta)\,  \Lambda^\alpha_{\nu, mno} \,  \textnormal d \boldsymbol\eta. 

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

        name: str
            Name of the operator.

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

        V_id = self.derham.space_to_form[V_id]
        W_id = self.derham.space_to_form[W_id]

        out = WeightedMassOperator(self.derham.Vh_fem[V_id], 
                                   self.derham.Vh_fem[W_id],
                                   V_extraction_op=self.derham.extraction_ops[V_id], 
                                   W_extraction_op=self.derham.extraction_ops[W_id],
                                   V_boundary_op=self.derham.boundary_ops[V_id], 
                                   W_boundary_op=self.derham.boundary_ops[W_id],
                                   weights_info=fun, 
                                   transposed=False)

        out.assemble(name=name)

        return out


class WeightedMassOperator(LinOpWithTransp):
    r"""
    Class for assembling weighted mass matrices in 3d.

    Weighted mass matrices :math:`\mathbb M^{\beta\alpha}: \mathbb R^{N_\alpha} \to \mathbb R^{N_\beta}` are of the general form

    .. math::

        \mathbb M^{\beta \alpha}_{(\mu,ijk),(\nu,mno)} = \int_{[0, 1]^3} \Lambda^\beta_{\mu,ijk} \, A_{\mu,\nu} \, \Lambda^\alpha_{\nu,mno} \, \textnormal d^3 \boldsymbol\eta\,,

    where the weight fuction :math:`A` is a tensor of rank 0, 1 or 2, depending on domain and co-domain of the operator, and :math:`\Lambda^\alpha_{\nu, mno}` is the B-spline basis function with tensor-product index :math:`mno` of the
    :math:`\nu`-th component in the space :math:`V^\alpha_h`. These matrices are sparse and stored in StencilMatrix format.

    Finally, :math:`\mathbb M^{\beta\alpha}` can be multiplied by extraction and boundary operators,  :math:`B * E * M^{\beta\alpha} * E^T * B^T`.

    Parameters
    ----------
    V : TensorFemSpace | VectorFemSpace
        Tensor product spline space from psydac.fem.tensor (domain, input space).

    W : TensorFemSpace | VectorFemSpace
        Tensor product spline space from psydac.fem.tensor (codomain, output space).

    V_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of V.

    W_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of W.

    V_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions.

    W_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions.

    weights_info : NoneType | str | list
        Information about the weights/block structure of the operator. Three cases are possible:
            * None : All blocks are allocated, disregarding zero-blocks or any symmetry.
            * str  : for square block matrices (V=W), a symmetry can be set in order to accelerate the assembly process.
                     Possible strings are "symm" (symmetric), "asym" (anti-symmetric) and "diag" (diagonal)
            * list : a 2d list with the same number of rows/columns as the number of components of the domain/codomain spaces.
                     The entries can then be either callables of np.ndarrays representing the weights at the quadrature points.
                     If a entry is zero or None, the corresponding block is set to None to accelerate the dot product.

    transposed : bool
        Whether to assemble the transposed operator.
    """

    def __init__(self, V, W, V_extraction_op=None, W_extraction_op=None, V_boundary_op=None, W_boundary_op=None, weights_info=None, transposed=False):

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

        assert isinstance(V, (TensorFemSpace, VectorFemSpace))
        assert isinstance(W, (TensorFemSpace, VectorFemSpace))

        self._V = V
        self._W = W

        # set basis extraction operators
        if V_extraction_op is not None:
            assert V_extraction_op.domain == V.vector_space
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.vector_space)

        if W_extraction_op is not None:
            assert W_extraction_op.domain == W.vector_space
            self._W_extraction_op = W_extraction_op
        else:
            self._W_extraction_op = IdentityOperator(W.vector_space)

        # set boundary operators
        if V_boundary_op is not None:
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(
                self._V_extraction_op.codomain)

        if W_boundary_op is not None:
            self._W_boundary_op = W_boundary_op
        else:
            self._W_boundary_op = IdentityOperator(
                self._W_extraction_op.codomain)

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

        # initialize blocks according to given symmetry and set zero default weights
        if isinstance(weights_info, str):

            self._symmetry = weights_info

            assert V_name == W_name, 'only square matrices (V=W) allowed!'
            assert len(
                V_name) > 2, 'only block matrices with domain/codomain spaces Hcurl, Hdiv and H1vec are allowed!'

            if weights_info == 'symm':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                           for Vs in V.spaces] for Ws in W.spaces]
            elif weights_info == 'asym':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                           if i != j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            elif weights_info == 'diag':
                blocks = [[StencilMatrix(Vs.vector_space, Ws.vector_space, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)
                           if i == j else None for j, Vs in enumerate(V.spaces)] for i, Ws in enumerate(W.spaces)]
            else:
                raise NotImplementedError(
                    f'given symmetry {weights_info} is not implemented!')

            self._mat = BlockLinearOperator(
                V.vector_space, W.vector_space, blocks=blocks)

            # set zero default weights with same block structure as block matrix
            self._weights = []
            for block_row in blocks:
                self._weights += [[]]
                for block in block_row:
                    if block is None:
                        self._weights[-1] += [None]
                    else:
                        self._weights[-1] += [lambda *etas: 0*etas[0]]

        # OR initialize blocks accoring to given weights by identifying zero blocks
        else:

            self._symmetry = None

            blocks = []
            self._weights = []

            # loop over codomain spaces (rows)
            for a, wspace in enumerate(Wspaces):
                blocks += [[]]
                self._weights += [[]]

                # loop over domain spaces (columns)
                for b, vspace in enumerate(Vspaces):

                    # set zero default weights if weights is None
                    if weights_info is None:
                        blocks[-1] += [StencilMatrix(
                            vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)]
                        self._weights[-1] += [lambda *etas: 0*etas[0]]

                    else:

                        if weights_info[a][b] is None:
                            blocks[-1] += [None]
                            self._weights[-1] += [None]

                        else:

                            # test weight function at quadrature points to identify zero blocks
                            pts = [quad_grid[nquad].points.flatten()
                                   for quad_grid, nquad in zip(wspace._quad_grids, wspace.nquads)]

                            if callable(weights_info[a][b]):
                                PTS = np.meshgrid(*pts, indexing='ij')
                                mat_w = weights_info[a][b](*PTS).copy()
                            elif isinstance(weights_info[a][b], np.ndarray):
                                mat_w = weights_info[a][b]

                            assert mat_w.shape == tuple(
                                [pt.size for pt in pts])

                            if np.any(np.abs(mat_w) > 1e-14):
                                blocks[-1] += [StencilMatrix(
                                    vspace.vector_space, wspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)]
                                self._weights[-1] += [weights_info[a][b]]
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
            self._M = EV @ self._mat @ EW.T
            self._M0 = BV @ self._M @ BW.T
        else:
            self._M = EW @ self._mat @ EV.T
            self._M0 = BW @ self._M @ BV.T

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
    def symmetry(self):
        return self._symmetry

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

        if self._symmetry is None:

            M = WeightedMassOperator(self._V, self._W,
                                     self._V_extraction_op, self._W_extraction_op,
                                     self._V_boundary_op, self._W_boundary_op,
                                     weights, not self._transposed)

            M.assemble(verbose=False)

        else:

            M = WeightedMassOperator(self._V, self._W,
                                     self._V_extraction_op, self._W_extraction_op,
                                     self._V_boundary_op, self._W_boundary_op,
                                     self._symmetry, not self._transposed)

            M.assemble(weights=weights, verbose=False)

        return M

    def assemble(self, weights=None, clear=True, verbose=True, name=None):
        """
        Assembles a weighted mass matrix (StencilMatrix/BlockLinearOperator) corresponding to given domain/codomain spline spaces.

        General form (in 3d) is mat_(ijk,mno) = integral[ Lambda_ijk * weight * Lambda_lmn ],
        where Lambda_ijk are the basis functions of the spline space and weight is some weight function.

        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.

        Parameters
        ----------
        weights : list | NoneType
            Weight function(s) (callables or np.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain. If weights=None, the weight is taken from the given weights in the instanziation of the object, else it will ne overriden.

        clear : bool
            Whether to first set all data to zero before assembly. If False, the new contributions are added to existing ones.

        verbose : bool
            Whether to do some printing.

        name : str
            Name of the operator.
        """

        # clear data
        if clear:
            if isinstance(self._mat, StencilMatrix):
                self._mat._data[:] = 0.
            else:
                for block_row in self._mat.blocks:
                    for block in block_row:
                        if block is not None:
                            block._data[:] = 0.

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
                f'Assembling matrix of WeightedMassOperator "{name}" with V={self._domain_symbolic_name}, W={self._codomain_symbolic_name}.')

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

        # set new weights and check for compatibility
        if weights is not None:
            assert isinstance(weights, list)

            for a, weights_row in enumerate(weights):
                for b, weight in enumerate(weights_row):
                    if weight is None:
                        assert isinstance(self._mat, BlockLinearOperator)
                        assert self._mat[a, b] is None
                    else:
                        assert callable(weight) or isinstance(
                            weight, np.ndarray)
                        if isinstance(self._mat, BlockLinearOperator):
                            assert self._mat[a, b] is not None

            self._weights = weights

        # loop over codomain spaces (rows)
        for a, codomain_space in enumerate(codomain_spaces):

            # knot span indices of elements of local domain
            codomain_spans = [
                quad_grid[nquad].spans for quad_grid, nquad in zip(codomain_space._quad_grids, codomain_space.nquads)]

            # global start spline index on process
            codomain_starts = [int(start)
                               for start in codomain_space.vector_space.starts]

            # pads (ghost regions)
            codomain_pads = codomain_space.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid[nquad].points.flatten()
                   for quad_grid, nquad in zip(codomain_space._quad_grids, codomain_space.nquads)]
            wts = [quad_grid[nquad].weights for quad_grid, nquad in zip(
                codomain_space._quad_grids, codomain_space.nquads)]

            # evaluated basis functions at quadrature points of codomain space
            codomain_basis = [
                quad_grid[nquad].basis for quad_grid, nquad in zip(codomain_space._quad_grids, codomain_space.nquads)]

            # loop over domain spaces (columns)
            for b, domain_space in enumerate(domain_spaces):

                # skip None and redundant blocks (lower half for symmetric and anti-symmetric)
                if isinstance(self._mat, BlockLinearOperator):
                    if self._mat[a, b] is None:
                        continue

                    if self._symmetry is not None and a > b:
                        continue

                if rank == 0 and verbose:
                    print(f'Assemble block {a, b}')

                # evaluate weight at quadrature points
                if callable(self._weights[a][b]):
                    PTS = np.meshgrid(*pts, indexing='ij')
                    mat_w = self._weights[a][b](*PTS).copy()
                elif isinstance(self._weights[a][b], np.ndarray):
                    mat_w = self._weights[a][b]

                assert mat_w.shape == tuple([pt.size for pt in pts])

                # evaluated basis functions at quadrature points of domain space
                domain_basis = [
                    quad_grid[nquad].basis for quad_grid, nquad in zip(domain_space._quad_grids, domain_space.nquads)]

                # assemble matrix (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)
                if np.any(np.abs(mat_w) > 1e-14):
                    if isinstance(self._mat, StencilMatrix):
                        self._assembly_kernel(*codomain_spans, *codomain_space.degree, *domain_space.degree, *
                                              codomain_starts, *codomain_pads, *wts, *codomain_basis, *domain_basis, mat_w, self._mat._data)
                    else:
                        self._assembly_kernel(*codomain_spans, *codomain_space.degree, *domain_space.degree, *codomain_starts,
                                              *codomain_pads, *wts, *codomain_basis, *domain_basis, mat_w, self._mat[a, b]._data)

        # exchange assembly data (accumulate ghost regions)
        self._mat.exchange_assembly_data()

        # copy data for symmetric/anti-symmetric block matrices
        if self.symmetry == 'symm':

            self._mat.update_ghost_regions()

            self._mat[1, 0]._data[:] = self._mat[0, 1].T._data
            self._mat[2, 0]._data[:] = self._mat[0, 2].T._data
            self._mat[2, 1]._data[:] = self._mat[1, 2].T._data

        elif self.symmetry == 'asym':

            self._mat.update_ghost_regions()

            self._mat[1, 0]._data[:] = -self._mat[0, 1].T._data
            self._mat[2, 0]._data[:] = -self._mat[0, 2].T._data
            self._mat[2, 1]._data[:] = -self._mat[1, 2].T._data

        if rank == 0 and verbose:
            print('Done.')

    @staticmethod
    def assemble_vec(W, vec, weight=None, clear=True):
        """
        Assembles (in 3d) vec_ijk = integral[ weight * Lambda_ijk ] into the Stencil-/BlockVector vec,
        where Lambda_ijk are the basis functions of the spline space and weight is some weight function.

        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.

        Parameters
        ----------
        W : TensorFemSpace | VectorFemSpace
            Tensor product spline space from psydac.fem.tensor.

        vec : StencilVector | BlockVector
            The vector to be filled.

        weight : list | NoneType
            Weight function(s) (callables or np.ndarrays) in a 1d list of shape corresponding to number of components.

        clear : bool
            Whether to first set all data to zero before assembly. If False, the new contributions are added to existing ones in vec.
        """

        assert isinstance(W, (TensorFemSpace, VectorFemSpace))
        assert isinstance(vec, (StencilVector, BlockVector))
        assert W.vector_space == vec.space

        # collect TensorFemSpaces for each component in tuple
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces

        # loag assembly kernel
        kernel = getattr(mass_kernels, 'kernel_' + str(W.ldim) + 'd_vec')

        # clear data
        if clear:
            if isinstance(vec, StencilVector):
                vec._data[:] = 0.
            else:
                for block in vec.blocks:
                    block._data[:] = 0.

        # loop over components
        for a, wspace in enumerate(Wspaces):

            # knot span indices of elements of local domain
            spans = [quad_grid[nquad].spans for quad_grid,
                     nquad in zip(wspace._quad_grids, wspace.nquads)]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid[nquad].points.flatten()
                   for quad_grid, nquad in zip(wspace._quad_grids, wspace.nquads)]
            wts = [quad_grid[nquad].weights for quad_grid,
                   nquad in zip(wspace._quad_grids, wspace.nquads)]

            # evaluated basis functions at quadrature points of codomain space
            basis = [quad_grid[nquad].basis for quad_grid,
                     nquad in zip(wspace._quad_grids, wspace.nquads)]

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

        # exchange assembly data (accumulate ghost regions) and update ghost regions
        vec.exchange_assembly_data()
        vec.update_ghost_regions()

    @staticmethod
    def eval_quad(W, coeffs, out=None):
        """
        Evaluates a given FEM field defined by its coefficients at the L2 quadrature points.

        Parameters
        ----------
        W : TensorFemSpace | VectorFemSpace
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

        assert isinstance(W, (TensorFemSpace, VectorFemSpace))
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
                out += (np.zeros([q_grid[nquad].points.size for q_grid,
                        nquad in zip(W._quad_grids, W.nquads)], dtype=float),)
            else:
                for space in W.spaces:
                    out += (np.zeros([q_grid[nquad].points.size for q_grid,
                            nquad in zip(space._quad_grids, space.nquad)], dtype=float),)

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
            spans = [quad_grid[nquad].spans for quad_grid,
                     nquad in zip(wspace._quad_grids, wspace.nquads)]

            # global start spline index on process
            starts = [int(start) for start in wspace.vector_space.starts]

            # pads (ghost regions)
            pads = wspace.vector_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [quad_grid[nquad].points.flatten()
                   for quad_grid, nquad in zip(wspace._quad_grids, wspace.nquads)]
            wts = [quad_grid[nquad].weights for quad_grid,
                   nquad in zip(wspace._quad_grids, wspace.nquads)]

            # evaluated basis functions at quadrature points of codomain space
            basis = [quad_grid[nquad].basis for quad_grid,
                     nquad in zip(wspace._quad_grids, wspace.nquads)]

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
