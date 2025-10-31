import cunumpy as xp
from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
from psydac.ddm.mpi import mpi as MPI
from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.linalg.basic import IdentityOperator, LinearOperator, Vector
from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.stencil import StencilMatrix, StencilVector, StencilVectorSpace

from struphy.feec import basis_projection_kernels
from struphy.feec.linear_operators import BoundaryOperator, LinOpWithTransp
from struphy.feec.local_projectors_kernels import assemble_basis_projection_operator_local
from struphy.feec.projectors import CommutingProjector, CommutingProjectorLocal
from struphy.feec.psydac_derham import get_pts_and_wts, get_span_and_basis
from struphy.feec.utilities import RotationMatrix
from struphy.polar.basic import PolarDerhamSpace, PolarVector
from struphy.polar.linear_operators import PolarExtractionOperator
from struphy.utils.pyccel import Pyccelkernel


class BasisProjectionOperators:
    r"""
    Collection of pre-defined :class:`struphy.feec.basis_projection_ops.BasisProjectionOperator`.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.

    domain : :ref:`avail_mappings`
        Mapping from logical unit cube to physical domain and corresponding metric coefficients.

    verbose : bool
        Show info on screen.

    **weights : dict
        Objects to access callables that can serve as weight functions.

    Note
    ----
    Possible choices for key-value pairs in ****weights** are, at the moment:

    - eq_mhd: :class:`struphy.fields_background.base.MHDequilibrium`
    """

    def __init__(self, derham, domain, verbose=True, **weights):
        self._derham = derham
        self._domain = domain
        self._weights = weights
        self._verbose = verbose

        self._rank = derham.comm.Get_rank() if derham.comm is not None else 0

        if xp.any(xp.array([p == 1 and Nel > 1 for p, Nel in zip(derham.p, derham.Nel)])):
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(
                    f'\nWARNING: Class "BasisProjectionOperators" called with p={derham.p} (interpolation of piece-wise constants should be avoided).',
                )

    @property
    def derham(self):
        """Discrete de Rham sequence on the logical unit cube."""
        return self._derham

    @property
    def domain(self):
        """Mapping from the logical unit cube to the physical domain with corresponding metric coefficients."""
        return self._domain

    @property
    def weights(self):
        """Dictionary of objects that provide access to callables that can serve as weight functions."""
        return self._weights

    @property
    def rank(self):
        """MPI rank, is 0 if no communicator."""
        return self._rank

    @property
    def verbose(self):
        """Bool: show info on screen."""
        return self._verbose

    # Wrapper functions for evaluating metric coefficients in right order (3x3 entries are last two axes!!)
    def DF(self, e1, e2, e3):
        """Jacobian callable."""
        return self.domain.jacobian(e1, e2, e3, transposed=False, change_out_order=True, squeeze_out=False)

    def DFT(self, e1, e2, e3):
        """Jacobain transpose callable."""
        return self.domain.jacobian(e1, e2, e3, transposed=True, change_out_order=True, squeeze_out=False)

    def DFinv(self, e1, e2, e3):
        """Jacobain inverse callable."""
        return self.domain.jacobian_inv(e1, e2, e3, transposed=False, change_out_order=True, squeeze_out=False)

    def DFinvT(self, e1, e2, e3):
        """Jacobian inverse transpose callable."""
        return self.domain.jacobian_inv(e1, e2, e3, transposed=True, change_out_order=True, squeeze_out=False)

    def G(self, e1, e2, e3):
        """Metric tensor callable."""
        return self.domain.metric(e1, e2, e3, change_out_order=True, squeeze_out=False)

    def Ginv(self, e1, e2, e3):
        """Inverse metric tensor callable."""
        return self.domain.metric_inv(e1, e2, e3, change_out_order=True, squeeze_out=False)

    def sqrt_g(self, e1, e2, e3):
        """Jacobian determinant callable."""
        return abs(self.domain.jacobian_det(e1, e2, e3, squeeze_out=False))

    @property
    def K0(self):
        r"""Basis projection operator

        .. math::

            \mathcal{K}^{0}_{ijk,mno} := \hat{\Pi}^0_{ijk} \left[  \hat{p}^0_{\text{eq}} \mathbf{\Lambda}^0_{mno} \right] \,.
        """
        if not hasattr(self, "_K0"):
            fun = [[lambda e1, e2, e3: self.weights["eq_mhd"].p0(e1, e2, e3)]]
            self._K0 = self.create_basis_op(
                fun,
                "H1",
                "H1",
                name="K0",
            )

        return self._K0

    @property
    def K3(self):
        r"""Basis projection operator

        .. math::

            \mathcal{K}^3_{ijk,mno} := \hat{\Pi}^3_{ijk} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\Lambda^3_{mno} \right] \,.
        """
        if not hasattr(self, "_K3"):
            fun = [
                [
                    lambda e1, e2, e3: self.weights["eq_mhd"].p3(
                        e1,
                        e2,
                        e3,
                    )
                    / self.sqrt_g(e1, e2, e3),
                ],
            ]
            self._K3 = self.create_basis_op(
                fun,
                "L2",
                "L2",
                name="K3",
            )

        return self._K3

    @property
    def Qv(self):
        r"""Basis projection operator

        .. math::

            \mathcal{Q}^v_{(\mu,ijk),(\nu,mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{\rho}^3_{\text{eq}} \Lambda^{0,\nu}_{mno} \right] \,.

        """
        if not hasattr(self, "_Qv"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].n3(e1, e2, e3) if m == n else 0 * e1,
                    ]
            self._Qv = self.create_basis_op(
                fun,
                "H1vec",
                "Hdiv",
                name="Qv",
            )

        return self._Qv

    @property
    def Q1(self):
        r"""Basis projection operator

        .. math::

            \mathcal{Q}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{\rho}^3_{\text{eq}}G^{-1}_{\mu,\nu}\Lambda^1_{(\nu, mno)} \right] \,.

        """
        if not hasattr(self, "_Q1"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].n3(e1, e2, e3)
                        * self.Ginv(e1, e2, e3)[:, :, :, m, n],
                    ]

            self._Q1 = self.create_basis_op(
                fun,
                "Hcurl",
                "Hdiv",
                name="Q1",
            )

        return self._Q1

    @property
    def Q2(self):
        r"""Basis projection operator

        .. math::

            \mathcal{Q}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        """
        if not hasattr(self, "_Q2"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].n3(
                            e1,
                            e2,
                            e3,
                        )
                        / self.sqrt_g(e1, e2, e3)
                        if m == n
                        else 0 * e1,
                    ]

            self._Q2 = self.create_basis_op(
                fun,
                "Hdiv",
                "Hdiv",
                name="Q2",
            )

        return self._Q2

    @property
    def Q3(self):
        r"""Basis projection operator

        .. math::

            \mathcal{Q}^3_{ijk,mno} := \hat{\Pi}^3_{ijk} \left[ \frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\Lambda^3_{mno} \right] \,.
        """
        if not hasattr(self, "_Q3"):
            fun = [
                [
                    lambda e1, e2, e3: self.weights["eq_mhd"].n3(
                        e1,
                        e2,
                        e3,
                    )
                    / self.sqrt_g(e1, e2, e3),
                ],
            ]
            self._Q3 = self.create_basis_op(
                fun,
                "L2",
                "L2",
                name="Q3",
            )

        return self._Q3

    @property
    def Tv(self):
        r"""Basis projection operator

        .. math::

            \mathcal{T}^v_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\mathcal R^B_{\mu,\nu} \Lambda^0_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^B_{\mu, \nu} := \epsilon_{\mu \alpha \nu}\, B^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^B \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium magnetic field (2-form).
        """
        if not hasattr(self, "_Tv"):
            rot_B = RotationMatrix(
                self.weights["eq_mhd"].b2_1,
                self.weights["eq_mhd"].b2_2,
                self.weights["eq_mhd"].b2_3,
            )

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: rot_B(e1, e2, e3)[:, :, :, m, n],
                    ]

            self._Tv = self.create_basis_op(
                fun,
                "H1vec",
                "Hcurl",
                name="Tv",
            )

        return self._Tv

    @property
    def T1(self):
        r"""Basis projection operator

        .. math::

            \mathcal{T}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\mathcal R^B_{\mu, \beta}G^{-1}_{\beta, \nu}\Lambda^1_{(\nu, mno)} \right] \,,

        with the rotation matrix

        .. math::

            \mathcal R^B_{\mu, \beta} := \epsilon_{\mu \alpha \beta}\, B^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^B \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \beta}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium magnetic field (2-form).

        """
        if not hasattr(self, "_T1"):
            rot_B = RotationMatrix(
                self.weights["eq_mhd"].b2_1,
                self.weights["eq_mhd"].b2_2,
                self.weights["eq_mhd"].b2_3,
            )

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: (rot_B(e1, e2, e3) @ self.Ginv(e1, e2, e3))[:, :, :, m, n],
                    ]

            self._T1 = self.create_basis_op(
                fun,
                "Hcurl",
                "Hcurl",
                name="T1",
            )

        return self._T1

    @property
    def T2(self):
        r"""Basis projection operator

        .. math::

            \mathcal{T}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\frac{\mathcal R^B_{\mu, \nu}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^B_{\mu, \nu} := \epsilon_{\mu \alpha \nu}\, B^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^B \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium magnetic field (2-form).
        """
        if not hasattr(self, "_T2"):
            rot_B = RotationMatrix(
                self.weights["eq_mhd"].b2_1,
                self.weights["eq_mhd"].b2_2,
                self.weights["eq_mhd"].b2_3,
            )

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: rot_B(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3),
                    ]

            self._T2 = self.create_basis_op(
                fun,
                "Hdiv",
                "Hcurl",
                name="T2",
            )

        return self._T2

    @property
    def Sv(self):
        r"""Basis projection operator

        .. math::

            \mathcal{S}^v_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{p}^3_{\text{eq}} \Lambda^{0,\nu}_{mno} \right] \,.
        """
        if not hasattr(self, "_Sv"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].p3(e1, e2, e3) if m == n else 0 * e1,
                    ]

            self._Sv = self.create_basis_op(
                fun,
                "H1vec",
                "Hdiv",
                name="Sv",
            )

        return self._Sv

    @property
    def S1(self):
        r"""Basis projection operator

        .. math::

            \mathcal{S}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\hat{p}^3_{\text{eq}}G^{-1}_{\mu,\nu}\Lambda^1_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_S1"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].p3(e1, e2, e3)
                        * self.Ginv(e1, e2, e3)[:, :, :, m, n],
                    ]

            self._S1 = self.create_basis_op(
                fun,
                "Hcurl",
                "Hdiv",
                name="S1",
            )

        return self._S1

    @property
    def S2(self):
        r"""Basis projection operator

        .. math::

            \mathcal{S}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        """
        if not hasattr(self, "_S2"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].p3(
                            e1,
                            e2,
                            e3,
                        )
                        / self.sqrt_g(e1, e2, e3)
                        if m == n
                        else 0 * e1,
                    ]

            self._S2 = self.create_basis_op(
                fun,
                "Hdiv",
                "Hdiv",
                name="S2",
            )

        return self._S2

    @property
    def S11(self):
        r"""Basis projection operator

        .. math::

            \mathcal{S}^{11}_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[ \hat{p}^0_{\text{eq}} \Lambda^1_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_S11"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].p0(e1, e2, e3) if m == n else 0 * e1,
                    ]

            self._S11 = self.create_basis_op(
                fun,
                "Hcurl",
                "Hcurl",
                name="S11",
            )

        return self._S11

    @property
    def S21(self):
        r"""Basis projection operator

        .. math::

            \mathcal{S}^{21}_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[ \frac{G_{\mu, \nu}}{\sqrt{g}} \Lambda^2_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_S21"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.G(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3),
                    ]

            self._S21 = self.create_basis_op(
                fun,
                "Hdiv",
                "Hcurl",
                name="S21",
            )

        return self._S21

    @property
    def S21p(self):
        r"""Basis projection operator

        .. math::

            \mathcal{S}^{21p}_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[ \frac{G_{\mu, \nu}}{\sqrt{g}} \Lambda^2_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_S21p"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].p0(
                            e1,
                            e2,
                            e3,
                        )
                        * self.G(e1, e2, e3)[:, :, :, m, n]
                        / self.sqrt_g(e1, e2, e3),
                    ]

            self._S21p = self.create_basis_op(
                fun,
                "Hdiv",
                "Hcurl",
                name="S21p",
            )
        return self._S21p

    @property
    def Uv(self):
        r"""Basis projection operator

        .. math::

            \mathcal{U}^v_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\sqrt{g} \, \Lambda^{0, \nu}_{mno} \right] \,.
        """
        if not hasattr(self, "_Uv"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.sqrt_g(e1, e2, e3) if m == n else 0 * e1,
                    ]

            self._Uv = self.create_basis_op(
                fun,
                "H1vec",
                "Hdiv",
                name="Uv",
            )

        return self._Uv

    @property
    def U1(self):
        r"""Basis projection operator

        .. math::

            \mathcal{U}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[ \sqrt{g} \, G^{-1}_{\mu, \nu} \Lambda^1_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_U1"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.sqrt_g(e1, e2, e3) * self.Ginv(e1, e2, e3)[:, :, :, m, n],
                    ]

            self._U1 = self.create_basis_op(
                fun,
                "Hcurl",
                "Hdiv",
                name="U1",
            )

        return self._U1

    @property
    def Xv(self):
        r"""Basis projection operator

        .. math::

            \mathcal{X}^v_{(\mu,ijk),(\nu,mno)} := \hat{\Pi}^{0, \mu}_{ijk} \left[ DF_{\mu, \nu}\Lambda^{0, \nu}_{mno} \right] \,.
        """
        if not hasattr(self, "_Xv"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.DF(e1, e2, e3)[:, :, :, m, n],
                    ]

            self._Xv = self.create_basis_op(
                fun,
                "H1vec",
                "H1vec",
                name="Xv",
            )

        return self._Xv

    @property
    def X1(self):
        r"""Basis projection operator

        .. math::

            \mathcal{X}^1_{(\mu, ijk),(\nu, mno)} := \hat{\Pi}^{0, \mu}_{ijk} \left[ DF^{-\top}_{\mu, \nu}\Lambda^1_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_X1"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.DFinvT(e1, e2, e3)[:, :, :, m, n],
                    ]

            self._X1 = self.create_basis_op(
                fun,
                "Hcurl",
                "H1vec",
                name="X1",
            )

        return self._X1

    @property
    def X2(self):
        r"""Basis projection operator

        .. math::

            \mathcal{X}^2_{(\mu, ijk),(\nu, mno)} := \hat{\Pi}^{0, \mu}_{ijk} \left[ \frac{DF_{\mu, \nu}}{\sqrt{g}} \Lambda^2_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_X2"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.DF(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3),
                    ]

            self._X2 = self.create_basis_op(
                fun,
                "Hdiv",
                "H1vec",
                name="X2",
            )

        return self._X2

    @property
    def W1(self):
        r"""Basis projection operator

        .. math::

            \mathcal{W}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\frac{\hat{\rho}^3_{\text{eq}}}{\sqrt{g}}\Lambda^1_{(\nu, mno)} \right] \,.
        """
        if not hasattr(self, "_W1"):
            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: self.weights["eq_mhd"].n3(
                            e1,
                            e2,
                            e3,
                        )
                        / self.sqrt_g(e1, e2, e3)
                        if m == n
                        else 0 * e1,
                    ]

            self._W1 = self.create_basis_op(
                fun,
                "Hcurl",
                "Hcurl",
                name="W1",
            )

        return self._W1

    @property
    def R1(self):
        r"""Basis projection operator

        .. math::

            \mathcal{R}^1_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^1_{(\mu, ijk)} \left[\frac{\mathcal R^J_{\mu, \nu}}{\sqrt{g}}\Lambda^2_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^J_{\mu, \nu} := \epsilon_{\mu \alpha \nu}\, J^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec J^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium current density (2-form).
        """

        if not hasattr(self, "_R1"):
            rot_J = RotationMatrix(
                self.weights["eq_mhd"].j2_1,
                self.weights["eq_mhd"].j2_2,
                self.weights["eq_mhd"].j2_3,
            )

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: rot_J(e1, e2, e3)[:, :, :, m, n] / self.sqrt_g(e1, e2, e3),
                    ]

            self._R1 = self.create_basis_op(
                fun,
                "Hdiv",
                "Hcurl",
                name="R1",
            )

        return self._R1

    @property
    def R2(self):
        r"""Basis projection operator

        .. math::

            \mathcal{R}^2_{(\mu, ijk), (\nu, mno)} := \hat{\Pi}^2_{(\mu, ijk)} \left[\mathcal R^J_{\mu, \beta} G^{-1}_{\beta, \nu} \Lambda^2_{(\nu, mno)} \right] \,.

        with the rotation matrix

        .. math::

            \mathcal R^J_{\mu, \beta} := \epsilon_{\mu \alpha \beta}\, J^2_{\textnormal{eq}, \alpha}\,,\qquad s.t. \qquad \mathcal R^J \vec v = \vec J^2_{\textnormal{eq}} \times \vec v\,,

        where :math:`\epsilon_{\mu \alpha \beta}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \alpha}` is the :math:`\alpha`-component of the MHD equilibrium current density (2-form).
        """
        if not hasattr(self, "_R2"):
            rot_J = RotationMatrix(
                self.weights["eq_mhd"].j2_1,
                self.weights["eq_mhd"].j2_2,
                self.weights["eq_mhd"].j2_3,
            )

            fun = []
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: (self.Ginv(e1, e2, e3) @ rot_J(e1, e2, e3))[:, :, :, m, n],
                    ]

            self._R2 = self.create_basis_op(
                fun,
                "Hdiv",
                "Hdiv",
                name="R2",
            )

        return self._R2

    @property
    def PB(self):
        r"""
        Basis projection operator

        .. math::

            \mathcal P^b_{ijk, (\mu, mno)} := \hat \Pi^0_{ijk} \left[\frac{1}{\sqrt g} \hat{b}^1_{\text{eq},\mu} \cdot \Lambda^2_{\mu, mno}\right]\,.
        """
        if not hasattr(self, "_PB"):
            fun = [[]]
            for m in range(3):
                fun[-1] += [
                    lambda e1, e2, e3, m=m: self.weights["eq_mhd"].unit_b1(e1, e2, e3)[m] / self.sqrt_g(e1, e2, e3),
                ]

            self._PB = self.create_basis_op(
                fun,
                "Hdiv",
                "H1",
                name="PB",
            )

        return self._PB

    ##########################################
    # Wrapper around BasisProjectionOperator #
    ##########################################
    def create_basis_op(
        self,
        fun: list,
        V_id: str,
        W_id: str,
        assemble: bool = False,
        name: str = None,
    ):
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

        assemble: bool
            Whether to assemble the DOF matrix.

        name: bstr
            Name of the operator.

        Returns
        -------
        out : A BasisProjectionOperator object.
        """

        assert isinstance(fun, list)

        if W_id in {"H1", "L2"}:
            assert len(fun) == 1
        else:
            assert len(fun) == 3

        for row in fun:
            assert isinstance(row, list)
            if V_id in {"H1", "L2"}:
                assert len(row) == 1
            else:
                assert len(row) == 3

        V_form = self.derham.space_to_form[V_id]
        W_form = self.derham.space_to_form[W_id]

        if self.derham.with_local_projectors:
            out = BasisProjectionOperatorLocal(
                self.derham.P[W_form],
                self.derham.Vh_fem[V_form],
                fun,
                self.derham.extraction_ops[V_form],
                self.derham.boundary_ops[V_form],
                self.derham.extraction_ops[W_form],
                self.derham.boundary_ops[W_form],
                transposed=False,
            )
        else:
            out = BasisProjectionOperator(
                self.derham.P[W_form],
                self.derham.Vh_fem[V_form],
                fun,
                V_extraction_op=self.derham.extraction_ops[V_form],
                V_boundary_op=self.derham.boundary_ops[V_form],
                transposed=False,
                polar_shift=self.domain.pole,
            )

        if assemble:
            if MPI.COMM_WORLD.Get_rank() == 0 and self.verbose:
                print(f'\nAssembling BasisProjectionOperator "{name}" with V={V_id}, W={W_id}.')
            out.assemble(verbose=self.verbose)

        if MPI.COMM_WORLD.Get_rank() == 0 and self.verbose:
            print("Done.")

        return out


class BasisProjectionOperatorLocal(LinOpWithTransp):
    r"""
    Class for assembling basis projection operators in 3d, based on local projectors.

    A basis projection operator :math:`\mathcal P: \mathbb R^{N_\alpha} \to \mathbb R^{N_\beta}` is defined by the matrix

    .. math::

        \mathcal P_{(\mu, ijk),(\nu, mno)} = \hat \Pi^\beta_{\mu, ijk} \left( A_{\mu,\nu}\,\Lambda^\alpha_{\nu, mno} \right)\,,

    where the weight fuction :math:`A` is a tensor of rank 0, 1 or 2, depending on domain and co-domain of the operator, and
    :math:`\Lambda^\alpha_{\nu, mno}` is the B-spline basis function with tensor-product index :math:`mno` of the
    :math:`\nu`-th component in the space :math:`V^\alpha_h`. The operator :math:`\hat \Pi^\beta: V^\beta \to \mathbb R^{N_\beta}`
    is a local commuting projector from the continuous space
    into the space of coefficients.

    Finally, extraction and boundary operators can be applied to the basis projection operator matrix, :math:`B_P * E_P * \mathcal P * E_V^T * B_V^T`.

    Parameters
    ----------
    P : struphy.feec.projectors.CommutingProjectorLocal
        Local commuting projector mapping into TensorFemSpace/VectorFemSpace W = P.space (codomain of operator).

    V : psydac.fem.basic.FemSpace
        Finite element spline space (domain, input space).

    weights : list
        Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.

    V_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of V.

    V_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions on V.

    P_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of the domain of P.

    P_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions on the domain of P.

    transposed : bool
        Whether to assemble the transposed operator.
    """

    def __init__(
        self,
        P: CommutingProjectorLocal,
        V: FemSpace,
        weights: list,
        V_extraction_op: PolarExtractionOperator | IdentityOperator = None,
        V_boundary_op: BoundaryOperator | IdentityOperator = None,
        P_extraction_op: PolarExtractionOperator | IdentityOperator = None,
        P_boundary_op: BoundaryOperator | IdentityOperator = None,
        transposed: bool = False,
    ):
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

        self._P = P
        self._V = V

        # set extraction operators
        if P_extraction_op is not None:
            self._P_extraction_op = P_extraction_op
        else:
            self._P_extraction_op = IdentityOperator(P.coeff_space)

        if V_extraction_op is not None:
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.coeff_space)

        # set boundary operators
        if P_boundary_op is not None:
            self._P_boundary_op = P_boundary_op
        else:
            self._P_boundary_op = IdentityOperator(
                self._P_extraction_op.domain,
            )

        if V_boundary_op is not None:
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(
                self._V_extraction_op.domain,
            )

        self._weights = weights
        self._transposed = transposed
        self._dtype = V.coeff_space.dtype

        # set domain and codomain symbolic names
        self._P_name = self._P.space_id

        if hasattr(V.symbolic_space, "name"):
            self._V_name = V.symbolic_space.name
        elif isinstance(V.symbolic_space, str):
            self._V_name = V.symbolic_space
        else:
            self._V_name = "H1vec"

        if transposed:
            self._domain_symbolic_name = self._P_name
            self._codomain_symbolic_name = self._V_name
        else:
            self._domain_symbolic_name = self._V_name
            self._codomain_symbolic_name = self._P_name

        # Are both space scalar spaces : useful to know if _mat will be Stencil or Block Matrix
        self._is_scalar = True
        if not isinstance(V, TensorFemSpace):
            self._is_scalar = False
            self._mpi_comm = V.coeff_space.spaces[0].cart.comm
        else:
            self._mpi_comm = V.coeff_space.cart.comm

        if not isinstance(P.fem_space, TensorFemSpace):
            self._is_scalar = False

        # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(V, TensorFemSpace):
            self._Vspaces = [V.coeff_space]
            self._V1ds = [V.spaces]
            self._VNbasis = xp.array([self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis])
        else:
            self._Vspaces = V.coeff_space
            self._V1ds = [comp.spaces for comp in V.spaces]
            self._VNbasis = xp.array(
                [
                    [self._V1ds[0][0].nbasis, self._V1ds[0][1].nbasis, self._V1ds[0][2].nbasis],
                    [
                        self._V1ds[1][0].nbasis,
                        self._V1ds[1][1].nbasis,
                        self._V1ds[1][2].nbasis,
                    ],
                    [self._V1ds[2][0].nbasis, self._V1ds[2][1].nbasis, self._V1ds[2][2].nbasis],
                ],
            )

        # output space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(P.fem_space, TensorFemSpace):
            self._Wspaces = [P.fem_space.coeff_space]
            self._W1ds = [P.fem_space.spaces]
            self._periodic = P._periodic
        else:
            self._Wspaces = P.fem_space.coeff_space
            self._W1ds = [comp.spaces for comp in P.fem_space.spaces]
            self._periodic = P._periodic[0]

        # We get the starts and ends of the Projector. They are the same as the starts and end for the rows of the StencilMatrix or BlockLinearOperator

        self._starts = self._P._starts
        self._ends = self._P._ends
        self._pds = self._P._pds
        # Degree of the B-splines
        self._p = self._P._p

        # ============= create and assemble the Basis Projection Operator matrix =======
        if self._is_scalar:
            self._mat = StencilMatrix(V.coeff_space, P.fem_space.coeff_space)
        else:
            self._mat = BlockLinearOperator(
                V.coeff_space,
                P.fem_space.coeff_space,
            )

        self._mat = self.assemble()
        # ========================================================

        # build the transposed matrix and applied extraction and boundary operators
        if transposed:
            self._mat_T = self._mat.T
            self._operator = (
                self._V_boundary_op
                @ self._V_extraction_op
                @ self._mat_T
                @ self._P_extraction_op.T
                @ self._P_boundary_op.T
            )
        else:
            self._operator = (
                self._P_boundary_op
                @ self._P_extraction_op
                @ self._mat
                @ self._V_extraction_op.T
                @ self._V_boundary_op.T
            )

        # set domain and codomain
        if transposed:
            self._domain = self._P.coeff_space
            self._codomain = self._V.coeff_space
        else:
            self._domain = self._V.coeff_space
            self._codomain = self._P.coeff_space

    @property
    def domain(self):
        """Domain vector space (input) of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain vector space (input) of the operator."""
        return self._codomain

    @property
    def dtype(self):
        """Datatype of the operator."""
        return self._dtype

    @property
    def tosparse(self):
        return self._mat.tosparse()

    @property
    def toarray(self):
        return self._mat.toarray()

    @property
    def transposed(self):
        """If the transposed operator is in play."""
        return self._transposed

    def dot(self, v, out=None):
        """
        Applies the basis projection operator to the FE coefficients v.

        Parameters
        ----------
        v : psydac.linalg.basic.Vector
            Vector the operator shall be applied to.

        out : psydac.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        Returns
        -------
         out : psydac.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self.domain

        if out is None:
            out = self._operator.dot(v)
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
            self._operator.dot(v, out=out)

        return out

    def transpose(self):
        """
        Returns the transposed operator.
        """
        return BasisProjectionOperatorLocal(
            self._P,
            self._V,
            self._weights,
            self._V_extraction_op,
            self._V_boundary_op,
            self._P_extraction_op,
            self._P_boundary_op,
            not self.transposed,
        )

    def update_weights(self, weights):
        """Updates self.weights and computes new BasisProjectionOperatorLocal matrix.

        Parameters
        ----------
        weights : list
            Weight function(s) (callables) in a 2d list of shape corresponding to number of components of domain/codomain.
        """

        self._weights = weights

        # assemble tensor-product dof matrix
        self._mat = self.assemble()

        # only need to update the transposed in case where it's needed
        if self._transposed:
            self._mat_T = self._mat.T

    def assemble(self, verbose=False):
        """
        Assembles the BasisProjectionOperatorLocal. And
        store it in self._mat.
        """

        # get the needed data :
        V = self._V
        P = self._P
        weights = self._weights

        # We determine where we have B-splines and where D-splines.
        if self._V_name == "H1":
            BoD = ["B", "B", "B"]
        elif self._V_name == "Hcurl":
            BoD = [["D", "B", "B"], ["B", "D", "B"], ["B", "B", "D"]]
        elif self._V_name == "Hdiv":
            BoD = [["B", "D", "D"], ["D", "B", "D"], ["D", "D", "B"]]
        elif self._V_name == "L2":
            BoD = ["D", "D", "D"]
        elif self._V_name in ("H1H1H1", "H1vec"):
            BoD = [["B", "B", "B"], ["B", "B", "B"], ["B", "B", "B"]]
        else:
            raise Exception("The FE space name for the input space must be H1, Hcurl, Hdiv, L2 or H1H1H1 or H1vec.")

        if isinstance(self._mat, StencilMatrix):
            # We get the B and D spline indices this MPI rank must compute
            eval_indices_B = P._Basis_functions_indices_B
            eval_indices_D = P._Basis_functions_indices_D
            if self._V_name == "H1":
                eval_indices = eval_indices_B
            elif self._V_name == "L2":
                eval_indices = eval_indices_D

            # We only use this counter to know if we are calling the Projection for the very first time
            counter = 0
            for col0 in eval_indices[0]:
                for col1 in eval_indices[1]:
                    for col2 in eval_indices[2]:
                        if counter == 0:
                            coeff, weigths_dof = P(
                                weights[0][0],
                                weighted=True,
                                B_or_D=BoD,
                                basis_indices=[col0, col1, col2],
                                first_go=True,
                            )
                        else:
                            coeff = P(
                                weights[0][0],
                                weighted=True,
                                B_or_D=BoD,
                                basis_indices=[
                                    col0,
                                    col1,
                                    col2,
                                ],
                                first_go=False,
                                pre_computed_dofs=weigths_dof,
                            )
                        counter += 1

                        assemble_basis_projection_operator_local(
                            self._starts,
                            self._ends,
                            self._pds,
                            self._periodic,
                            self._p,
                            xp.array([col0, col1, col2]),
                            self._VNbasis,
                            self._mat._data,
                            coeff,
                            P._rows_B_or_D_splines_0[BoD[0]][P._translation_indices_B_or_D_splines_0[BoD[0]][col0]],
                            P._rows_B_or_D_splines_1[BoD[1]][P._translation_indices_B_or_D_splines_1[BoD[1]][col1]],
                            P._rows_B_or_D_splines_2[BoD[2]][P._translation_indices_B_or_D_splines_2[BoD[2]][col2]],
                            P._rowe_B_or_D_splines_0[BoD[0]][P._translation_indices_B_or_D_splines_0[BoD[0]][col0]],
                            P._rowe_B_or_D_splines_1[BoD[1]][P._translation_indices_B_or_D_splines_1[BoD[1]][col1]],
                            P._rowe_B_or_D_splines_2[BoD[2]][P._translation_indices_B_or_D_splines_2[BoD[2]][col2]],
                        )

        elif self._P_name == "H1" or self._P_name == "L2":
            # We get the B and D spline indices this MPI rank must compute
            eval_indices_B = P._Basis_functions_indices_B
            eval_indices_D = P._Basis_functions_indices_D

            if self._V_name == "Hcurl":
                eval_block_0 = [eval_indices_D[0], eval_indices_B[1], eval_indices_B[2]]
                eval_block_1 = [eval_indices_B[0], eval_indices_D[1], eval_indices_B[2]]
                eval_block_2 = [eval_indices_B[0], eval_indices_B[1], eval_indices_D[2]]

            elif self._V_name == "Hdiv":
                eval_block_0 = [eval_indices_B[0], eval_indices_D[1], eval_indices_D[2]]
                eval_block_1 = [eval_indices_D[0], eval_indices_B[1], eval_indices_D[2]]
                eval_block_2 = [eval_indices_D[0], eval_indices_D[1], eval_indices_B[2]]

            elif self._V_name in ("H1H1H1", "H1vec"):
                eval_block_0 = [eval_indices_B[0], eval_indices_B[1], eval_indices_B[2]]
                eval_block_1 = [eval_indices_B[0], eval_indices_B[1], eval_indices_B[2]]
                eval_block_2 = [eval_indices_B[0], eval_indices_B[1], eval_indices_B[2]]
            else:
                raise Exception("The input space name is not defined.")

            eval_blocks = [eval_block_0, eval_block_1, eval_block_2]
            # Filling the hh-th block
            for hh in range(3):
                Aux = StencilMatrix(self._Vspaces[hh], self._Wspaces[0])
                counter = 0
                for col0 in eval_blocks[hh][0]:
                    for col1 in eval_blocks[hh][1]:
                        for col2 in eval_blocks[hh][2]:
                            if counter == 0:
                                coeff, weigths_dof = P(
                                    weights[0][hh],
                                    weighted=True,
                                    B_or_D=BoD[hh],
                                    basis_indices=[
                                        col0,
                                        col1,
                                        col2,
                                    ],
                                    first_go=True,
                                )
                            else:
                                coeff = P(
                                    weights[0][hh],
                                    weighted=True,
                                    B_or_D=BoD[hh],
                                    basis_indices=[
                                        col0,
                                        col1,
                                        col2,
                                    ],
                                    first_go=False,
                                    pre_computed_dofs=weigths_dof,
                                )
                            counter += 1

                            assemble_basis_projection_operator_local(
                                self._starts,
                                self._ends,
                                self._pds,
                                self._periodic,
                                self._p,
                                xp.array(
                                    [
                                        col0,
                                        col1,
                                        col2,
                                    ],
                                ),
                                self._VNbasis[hh],
                                Aux._data,
                                coeff,
                                P._rows_B_or_D_splines_0[BoD[hh][0]][
                                    P._translation_indices_B_or_D_splines_0[BoD[hh][0]][col0]
                                ],
                                P._rows_B_or_D_splines_1[BoD[hh][1]][
                                    P._translation_indices_B_or_D_splines_1[BoD[hh][1]][col1]
                                ],
                                P._rows_B_or_D_splines_2[BoD[hh][2]][
                                    P._translation_indices_B_or_D_splines_2[BoD[hh][2]][col2]
                                ],
                                P._rowe_B_or_D_splines_0[BoD[hh][0]][
                                    P._translation_indices_B_or_D_splines_0[BoD[hh][0]][col0]
                                ],
                                P._rowe_B_or_D_splines_1[BoD[hh][1]][
                                    P._translation_indices_B_or_D_splines_1[BoD[hh][1]][col1]
                                ],
                                P._rowe_B_or_D_splines_2[BoD[hh][2]][
                                    P._translation_indices_B_or_D_splines_2[BoD[hh][2]][col2]
                                ],
                            )

                self._mat[(0, hh)] = Aux

        elif self._V_name == "H1" or self._V_name == "L2":
            # We get the B and D spline indices this MPI rank must compute
            eval_indices_B = P._Basis_function_indices_mark_B
            eval_indices_D = P._Basis_function_indices_mark_D

            if self._V_name == "H1":
                eval_indices = eval_indices_B
            elif self._V_name == "L2":
                eval_indices = eval_indices_D

            Aux0 = StencilMatrix(self._Vspaces[0], self._Wspaces[0])
            Aux1 = StencilMatrix(self._Vspaces[0], self._Wspaces[1])
            Aux2 = StencilMatrix(self._Vspaces[0], self._Wspaces[2])
            Aux = [Aux0, Aux1, Aux2]
            counter = 0
            for col0 in eval_indices[0]:
                for col1 in eval_indices[1]:
                    for col2 in eval_indices[2]:
                        if counter == 0:
                            coeff, weigths_dof = P(
                                [weights[0][0], weights[1][0], weights[2][0]],
                                weighted=True,
                                B_or_D=BoD,
                                basis_indices=[
                                    col0,
                                    col1,
                                    col2,
                                ],
                                first_go=True,
                            )
                        else:
                            coeff = P(
                                [weights[0][0], weights[1][0], weights[2][0]],
                                weighted=True,
                                B_or_D=BoD,
                                basis_indices=[col0, col1, col2],
                                first_go=False,
                                pre_computed_dofs=weigths_dof,
                            )
                        counter += 1

                        for h in range(3):
                            assemble_basis_projection_operator_local(
                                self._starts[h],
                                self._ends[h],
                                self._pds[h],
                                self._periodic,
                                self._p,
                                xp.array(
                                    [
                                        col0,
                                        col1,
                                        col2,
                                    ],
                                ),
                                self._VNbasis,
                                Aux[h]._data,
                                coeff[h],
                                P._rows_block_B_or_D_splines[0][h][BoD[0]][
                                    P._translation_indices_block_B_or_D_splines[0][h][BoD[0]][col0]
                                ],
                                P._rows_block_B_or_D_splines[1][h][BoD[1]][
                                    P._translation_indices_block_B_or_D_splines[1][h][BoD[1]][col1]
                                ],
                                P._rows_block_B_or_D_splines[2][h][BoD[2]][
                                    P._translation_indices_block_B_or_D_splines[2][h][BoD[2]][col2]
                                ],
                                P._rowe_block_B_or_D_splines[0][h][BoD[0]][
                                    P._translation_indices_block_B_or_D_splines[0][h][BoD[0]][col0]
                                ],
                                P._rowe_block_B_or_D_splines[1][h][BoD[1]][
                                    P._translation_indices_block_B_or_D_splines[1][h][BoD[1]][col1]
                                ],
                                P._rowe_block_B_or_D_splines[2][h][BoD[2]][
                                    P._translation_indices_block_B_or_D_splines[2][h][BoD[2]][col2]
                                ],
                            )

            for h in range(3):
                self._mat[(h, 0)] = Aux[h]

        else:
            # We get the B and D spline indices this MPI rank must compute
            eval_indices_B = P._Basis_function_indices_mark_B
            eval_indices_D = P._Basis_function_indices_mark_D

            if self._V_name == "Hcurl":
                eval_block_0 = [eval_indices_D[0], eval_indices_B[1], eval_indices_B[2]]
                eval_block_1 = [eval_indices_B[0], eval_indices_D[1], eval_indices_B[2]]
                eval_block_2 = [eval_indices_B[0], eval_indices_B[1], eval_indices_D[2]]
            elif self._V_name == "Hdiv":
                eval_block_0 = [eval_indices_B[0], eval_indices_D[1], eval_indices_D[2]]
                eval_block_1 = [eval_indices_D[0], eval_indices_B[1], eval_indices_D[2]]
                eval_block_2 = [eval_indices_D[0], eval_indices_D[1], eval_indices_B[2]]

            elif self._V_name in ("H1H1H1", "H1vec"):
                eval_block_0 = [eval_indices_B[0], eval_indices_B[1], eval_indices_B[2]]
                eval_block_1 = [eval_indices_B[0], eval_indices_B[1], eval_indices_B[2]]
                eval_block_2 = [eval_indices_B[0], eval_indices_B[1], eval_indices_B[2]]
            else:
                raise Exception("The input space name is not defined.")

            eval_blocks = [eval_block_0, eval_block_1, eval_block_2]

            # Iterates over the input block entries
            for hh in range(3):
                Aux0 = StencilMatrix(self._Vspaces[hh], self._Wspaces[0])
                Aux1 = StencilMatrix(self._Vspaces[hh], self._Wspaces[1])
                Aux2 = StencilMatrix(self._Vspaces[hh], self._Wspaces[2])
                Aux = [Aux0, Aux1, Aux2]
                counter = 0
                for col0 in eval_blocks[hh][0]:
                    for col1 in eval_blocks[hh][1]:
                        for col2 in eval_blocks[hh][2]:
                            if counter == 0:
                                coeff, weigths_dof = P(
                                    [weights[0][hh], weights[1][hh], weights[2][hh]],
                                    weighted=True,
                                    B_or_D=BoD[hh],
                                    basis_indices=[
                                        col0,
                                        col1,
                                        col2,
                                    ],
                                    first_go=True,
                                )
                            else:
                                coeff = P(
                                    [weights[0][hh], weights[1][hh], weights[2][hh]],
                                    weighted=True,
                                    B_or_D=BoD[hh],
                                    basis_indices=[
                                        col0,
                                        col1,
                                        col2,
                                    ],
                                    first_go=False,
                                    pre_computed_dofs=weigths_dof,
                                )
                            counter += 1

                            # Iterates over the output block entries
                            for h in range(3):
                                assemble_basis_projection_operator_local(
                                    self._starts[h],
                                    self._ends[h],
                                    self._pds[h],
                                    self._periodic,
                                    self._p,
                                    xp.array(
                                        [
                                            col0,
                                            col1,
                                            col2,
                                        ],
                                    ),
                                    self._VNbasis[hh],
                                    Aux[h]._data,
                                    coeff[h],
                                    P._rows_block_B_or_D_splines[0][h][BoD[hh][0]][
                                        P._translation_indices_block_B_or_D_splines[0][h][BoD[hh][0]][col0]
                                    ],
                                    P._rows_block_B_or_D_splines[1][h][BoD[hh][1]][
                                        P._translation_indices_block_B_or_D_splines[1][h][BoD[hh][1]][col1]
                                    ],
                                    P._rows_block_B_or_D_splines[2][h][BoD[hh][2]][
                                        P._translation_indices_block_B_or_D_splines[2][h][BoD[hh][2]][col2]
                                    ],
                                    P._rowe_block_B_or_D_splines[0][h][BoD[hh][0]][
                                        P._translation_indices_block_B_or_D_splines[0][h][BoD[hh][0]][col0]
                                    ],
                                    P._rowe_block_B_or_D_splines[1][h][BoD[hh][1]][
                                        P._translation_indices_block_B_or_D_splines[1][h][BoD[hh][1]][col1]
                                    ],
                                    P._rowe_block_B_or_D_splines[2][h][BoD[hh][2]][
                                        P._translation_indices_block_B_or_D_splines[2][h][BoD[hh][2]][col2]
                                    ],
                                )

                for h in range(3):
                    self._mat[(h, hh)] = Aux[h]

        self._mat.update_ghost_regions()
        return self._mat


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
        Weight function(s) (callables or xp.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain.

    V_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of V.

    V_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions on V.

    P_extraction_op : PolarExtractionOperator | IdentityOperator
        Extraction operator to polar sub-space of the domain of P.

    P_boundary_op : BoundaryOperator | IdentityOperator
        Boundary operator that sets essential boundary conditions on the domain of P.

    transposed : bool
        Whether to assemble the transposed operator.

    polar_shift : bool
        Whether there are metric coefficients contained in "weights" which are singular at eta1=0. If True, interpolation points at eta1=0 are shifted away from the singularity by 1e-5.

    use_cache : bool
        Whether to store some information computed in self.assemble for reuse. Set it to true if planned to update the weights later.
    """

    def __init__(
        self,
        P: CommutingProjector,
        V: FemSpace,
        weights: list,
        *,
        V_extraction_op: PolarExtractionOperator | IdentityOperator = None,
        V_boundary_op: BoundaryOperator | IdentityOperator = None,
        P_extraction_op: PolarExtractionOperator | IdentityOperator = None,
        P_boundary_op: BoundaryOperator | IdentityOperator = None,
        transposed: bool = False,
        polar_shift: bool = False,
        use_cache: bool = False,
    ):
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

        self._P = P
        self._V = V

        # set extraction operators
        if P_extraction_op is not None:
            self._P_extraction_op = P_extraction_op
        else:
            self._P_extraction_op = P.dofs_extraction_op

        if V_extraction_op is not None:
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.coeff_space)

        # set boundary operators
        if P_boundary_op is not None:
            self._P_boundary_op = P_boundary_op
        else:
            self._P_boundary_op = P.boundary_op

        if V_boundary_op is not None:
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(
                self._V_extraction_op.codomain,
            )

        self._weights = weights
        self._transposed = transposed
        self._polar_shift = polar_shift
        self._dtype = V.coeff_space.dtype
        self._use_cache = use_cache

        # Create cache
        if use_cache:
            self._cache = {}

        # set domain and codomain symbolic names
        if hasattr(P.space.symbolic_space, "name"):
            P_name = P.space.symbolic_space.name
        elif isinstance(P.space.symbolic_space, str):
            P_name = P.space.symbolic_space
        else:
            P_name = "H1vec"

        if hasattr(V.symbolic_space, "name"):
            V_name = V.symbolic_space.name
        elif isinstance(V.symbolic_space, str):
            V_name = V.symbolic_space
        else:
            V_name = "H1vec"

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
            self._mpi_comm = V.coeff_space.spaces[0].cart.comm
        else:
            self._mpi_comm = V.coeff_space.cart.comm

        if not isinstance(P.space, TensorFemSpace):
            self._is_scalar = False

        # ============= create and assemble tensor-product dof matrix =======
        if self._is_scalar:
            self._dof_mat = StencilMatrix(V.coeff_space, P.space.coeff_space)
        else:
            self._dof_mat = BlockLinearOperator(
                V.coeff_space,
                P.space.coeff_space,
            )

        self._dof_mat = self.assemble()
        # ========================================================

        # build composed linear operator BP * P * DOF * EV^T * BV^T or transposed
        if transposed:
            self._dof_mat_T = self._dof_mat.T
            self._dof_operator = (
                self._V_boundary_op
                @ self._V_extraction_op
                @ self._dof_mat_T
                @ self._P_extraction_op.T
                @ self._P_boundary_op.T
            )
            self._x0 = self._dof_operator.domain.zeros()
        else:
            self._dof_operator = (
                self._P_boundary_op
                @ self._P_extraction_op
                @ self._dof_mat
                @ self._V_extraction_op.T
                @ self._V_boundary_op.T
            )
            self._x0 = self._dof_operator.codomain.zeros()

        # set domain and codomain
        self._domain = self.dof_operator.domain
        self._codomain = self.dof_operator.codomain

        # temporary vectors for dot product
        self._tmp_dom = self._dof_operator.domain.zeros()
        self._tmp_codom = self._dof_operator.codomain.zeros()

    @property
    def domain(self):
        """Domain vector space (input) of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain vector space (input) of the operator."""
        return self._codomain

    @property
    def dtype(self):
        """Datatype of the operator."""
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    @property
    def transposed(self):
        """If the transposed operator is in play."""
        return self._transposed

    @property
    def dof_operator(self):
        """The degrees of freedom operator as composite linear operator containing polar extraction and boundary operators."""
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
            out = self.codomain.zeros()

            if self.transposed:
                # 1. apply inverse transposed inter-/histopolation matrix, 2. apply transposed dof operator
                out = self.dof_operator.dot(
                    self._P.solve(
                        v,
                        True,
                        apply_bc=True,
                    ),
                )
            else:
                # 1. apply dof operator, 2. apply inverse inter-/histopolation matrix
                out = self._P.solve(
                    self.dof_operator.dot(
                        v,
                    ),
                    False,
                    apply_bc=True,
                )

        assert isinstance(out, Vector)
        assert out.space == self.codomain

        if self.transposed:
            # 1. apply inverse transposed inter-/histopolation matrix, 2. apply transposed dof operator
            self._P.solve(v, True, apply_bc=True, out=self._tmp_dom, x0=self._x0)
            self._tmp_dom.copy(out=self._x0)
            self.dof_operator.dot(self._tmp_dom, out=out)
        else:
            # 1. apply dof operator, 2. apply inverse inter-/histopolation matrix
            self.dof_operator.dot(v, out=self._tmp_codom)
            self._P.solve(self._tmp_codom, False, apply_bc=True, out=out, x0=self._x0)
            out.copy(out=self._x0)

        return out

    def transpose(self, conjugate=False):
        """
        Returns the transposed operator.
        """
        return BasisProjectionOperator(
            self._P,
            self._V,
            self._weights,
            V_extraction_op=self._V_extraction_op,
            V_boundary_op=self._V_boundary_op,
            P_extraction_op=self._P_extraction_op,
            P_boundary_op=self._P_boundary_op,
            transposed=not self.transposed,
            polar_shift=self._polar_shift,
            use_cache=self._use_cache,
        )

    def update_weights(self, weights):
        """Updates self.weights and computes new DOF matrix.

        Parameters
        ----------
        weights : list
            Weight function(s) (callables or xp.ndarrays) in a 2d list of shape corresponding to number of components of domain/codomain.
        """

        self._weights = weights

        # assemble tensor-product dof matrix
        self._dof_mat = self.assemble()

        # only need to update the transposed in case where it's needed
        # (no need to recreate a new ComposedOperator)
        if self._transposed:
            self._dof_mat_T = self._dof_mat.transpose(out=self._dof_mat_T)

    def assemble(self, weights=None, verbose=False):
        """
        Assembles the tensor-product DOF matrix sigma_i(weights[i,j]*Lambda_j), where i=(i1, i2, ...)
        and j=(j1, j2, ...) depending on the number of spatial dimensions (1d, 2d or 3d). And
        store it in self._dof_mat.
        """
        rank = MPI.COMM_WORLD.Get_rank()

        # get the needed data :
        V = self._V
        P = self._P.projector_tensor
        if weights is None:
            weights = self._weights
        else:
            assert isinstance(weights, list)
            assert isinstance(weights[0], list)
        polar_shift = self._polar_shift

        # input space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(V, TensorFemSpace):
            _Vspaces = [V.coeff_space]
            _V1ds = [V.spaces]
        else:
            _Vspaces = V.coeff_space
            _V1ds = [comp.spaces for comp in V.spaces]

        # output space: 3d StencilVectorSpaces and 1d SplineSpaces of each component
        if isinstance(P.space, TensorFemSpace):
            _Wspaces = [P.space.coeff_space]
            _W1ds = [P.space.spaces]
        else:
            _Wspaces = P.space.coeff_space
            _W1ds = [comp.spaces for comp in P.space.spaces]

        # retrieve number of quadrature points of each component (=1 for interpolation)
        _nqs = [[P.grid_x[comp][direction].shape[1] for direction in range(V.ldim)] for comp in range(len(_W1ds))]

        # ouptut vector space (codomain), row of block
        for i, (Wspace, W1d, nq, weight_line) in enumerate(zip(_Wspaces, _W1ds, _nqs, weights)):
            _Wdegrees = [space.degree for space in W1d]

            # input vector space (domain), column of block
            for j, (Vspace, V1d, loc_weight) in enumerate(zip(_Vspaces, _V1ds, weight_line)):
                _starts_in = xp.array(Vspace.starts)
                _ends_in = xp.array(Vspace.ends)
                _pads_in = xp.array(Vspace.pads)

                _starts_out = xp.array(Wspace.starts)
                _ends_out = xp.array(Wspace.ends)
                _pads_out = xp.array(Wspace.pads)

                # use cached information if asked
                if self._use_cache:
                    if (i, j) in self._cache:
                        _ptsG, _wtsG, _spans, _bases, _subs = self._cache[
                            (
                                i,
                                j,
                            )
                        ]
                    else:
                        _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                            V1d,
                            W1d,
                            _starts_out,
                            _ends_out,
                            nq,
                            polar_shift,
                        )

                        self._cache[(i, j)] = (
                            _ptsG,
                            _wtsG,
                            _spans,
                            _bases,
                            _subs,
                        )
                else:
                    # no cache
                    _ptsG, _wtsG, _spans, _bases, _subs = prepare_projection_of_basis(
                        V1d,
                        W1d,
                        _starts_out,
                        _ends_out,
                        nq,
                        polar_shift,
                    )

                _ptsG = [pts.flatten() for pts in _ptsG]

                _Vnbases = [int(space.nbasis) for space in V1d]
                _Wnbases = [int(space.nbasis) for space in W1d]

                # Evaluate weight function at quadrature points
                # evaluate weight at quadrature points
                if callable(loc_weight):
                    PTS = xp.meshgrid(*_ptsG, indexing="ij")
                    mat_w = loc_weight(*PTS).copy()
                elif isinstance(loc_weight, xp.ndarray):
                    assert loc_weight.shape == (len(_ptsG[0]), len(_ptsG[1]), len(_ptsG[2]))
                    mat_w = loc_weight
                elif loc_weight is not None:
                    raise TypeError(
                        "weights must be xp.ndarray, callable or None",
                    )

                # Call the kernel if weight function is not zero or in the scalar case
                # to avoid calling _block of a StencilMatrix in the else

                not_weight_zero = xp.array(
                    int(loc_weight is not None and xp.any(xp.abs(mat_w) > 1e-14)),
                )

                if self._mpi_comm is not None:
                    self._mpi_comm.Allreduce(
                        MPI.IN_PLACE,
                        not_weight_zero,
                        op=MPI.LOR,
                    )

                if not_weight_zero or self._is_scalar:
                    # get cell of block matrix (don't instantiate if all zeros)
                    if self._is_scalar:
                        dofs_mat = self._dof_mat
                    else:
                        dofs_mat = self._dof_mat[i, j]

                    if dofs_mat is None:
                        # Maybe in a previous iteration we had more zeros
                        self._dof_mat[i, j] = StencilMatrix(
                            Vspace,
                            Wspace,
                            backend=PSYDAC_BACKEND_GPYCCEL,
                            precompiled=True,
                        )
                        dofs_mat = self._dof_mat[i, j]

                    kernel = Pyccelkernel(
                        getattr(
                            basis_projection_kernels,
                            "assemble_dofs_for_weighted_basisfuns_" + str(V.ldim) + "d",
                        ),
                    )

                    if rank == 0 and verbose:
                        print(f"Assemble block {i, j}")
                    kernel(
                        dofs_mat._data,
                        _starts_in,
                        _ends_in,
                        _pads_in,
                        _starts_out,
                        _ends_out,
                        _pads_out,
                        mat_w,
                        *_wtsG,
                        *_spans,
                        *_bases,
                        *_subs,
                        *_Vnbases,
                        *_Wnbases,
                        *_Wdegrees,
                    )

                    dofs_mat.set_backend(
                        backend=PSYDAC_BACKEND_GPYCCEL,
                        precompiled=True,
                    )

                    dofs_mat.update_ghost_regions()

                else:
                    self._dof_mat[i, j] = None

        return self._dof_mat


def prepare_projection_of_basis(V1d, W1d, starts_out, ends_out, n_quad=None, polar_shift=False):
    """Obtain knot span indices and basis functions evaluated at projection point sets of a given space.

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

    subs : 3-tuple of 1f int arrays
        Sub-interval indices (either 0 or 1). This index is 1 if an element has to be split for exact integration (even spline degree).
    """

    pts, wts, subs, spans, bases = [], [], [], [], []

    if n_quad is None:
        n_quad = [None] * 3

    # Loop over direction, prepare point sets and evaluate basis functions
    for d, (space_in, space_out, s, e) in enumerate(zip(V1d, W1d, starts_out, ends_out)):
        # point sets and weights for inter-/histopolation
        pts_i, wts_i, subs_i = get_pts_and_wts(
            space_out,
            s,
            e,
            n_quad=n_quad[d],
            polar_shift=d == 0 and polar_shift,
        )

        pts += [pts_i]
        wts += [wts_i]
        subs += [subs_i]

        # Knot span indices and V-basis functions evaluated at W-point sets
        s_i, b_i = get_span_and_basis(pts[-1], space_in)

        spans += [s_i]
        bases += [b_i]

    # print("#################################################")
    # print("#################################################")
    # print("W1d[0]:")
    # print(W1d[0])
    # print("W1d[1]:")
    # print(W1d[1])
    # print("W1d[2]:")
    # print(W1d[2])
    # print("pts :")
    # print(pts)
    # print("#################################################")
    # print("#################################################")

    return tuple(pts), tuple(wts), tuple(spans), tuple(bases), tuple(subs)


class CoordinateProjector(LinearOperator):
    r"""
    Class of projectors on one component of a :class:`~psydac.linalg.block.BlockVectorSpace`.
    Represent the projection on the :math:`\mu`-th component :

    .. math::

        \begin{align}
        P_\mu : \ & V_1 \times \ldots \times V_\mu \times \ldots \times V_n \longrightarrow V_\mu \,,
        \\[2mm]
        &\vec{x} = (x_1,\ldots,x_\mu,\ldots ,x_n) \mapsto x_\mu \,.
        \end{align}

    Parameters
    ----------
    mu : int
        The component on which to project.

    V : BlockVectorSpace | PolarDerhamSpace
        Domain, input space.

    Vmu : StencilVectorSpace | PolarDerhamSpace
        Codomain, out space, must be :math:`\mu`-th space of V.
    """

    def __init__(
        self,
        mu: int,
        V: BlockVectorSpace | PolarDerhamSpace,
        Vmu: StencilVectorSpace | PolarDerhamSpace,
    ):
        assert isinstance(mu, int)
        if isinstance(V, PolarDerhamSpace):
            assert V.parent_space.spaces[mu] == Vmu.parent_space
        else:
            assert V.spaces[mu] == Vmu

        self.dir = mu
        self._domain = V
        self._codomain = Vmu
        self._dtype = Vmu.dtype

    @property
    def domain(self):
        """Domain vector space (input) of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain vector space (input) of the operator."""
        return self._codomain

    @property
    def dtype(self):
        """Datatype of the operator."""
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return CoordinateInclusion(self.dir, self._domain, self._codomain)

    def dot(
        self,
        v: BlockVector | PolarVector,
        out=None,
    ):
        assert v.space == self._domain
        if isinstance(self.domain, PolarDerhamSpace):
            if out is not None:
                assert out.space == self._codomain
                out *= 0.0
            else:
                out = self.codomain.zeros()
            out._tp += v.tp.blocks[self.dir]
        else:
            if out is not None:
                assert out.space == self._codomain
                out *= 0.0
                out += v.blocks[self.dir]
            else:
                out = v.blocks[self.dir].copy()
        out.update_ghost_regions()  # TODO: this is usually not done within .dot, should maybe be removed?
        return out

    def idot(
        self,
        v: BlockVector | PolarVector,
        out: StencilVector | PolarVector,
    ):
        assert v.space == self._domain
        assert out.space == self._codomain
        if isinstance(self.domain, PolarDerhamSpace):
            out += v.tp.blocks[self.dir]
        else:
            out += v.blocks[self.dir]


class CoordinateInclusion(LinearOperator):
    r"""
    Class of inclusion operator from one component of a :class:`~psydac.linalg.block.BlockVectorSpace`.
    Represent the canonical inclusion on the :math:`\mu`-th component :

    .. math::

        \begin{align}
        I_\mu : \ &V_\mu \longrightarrow V_1 \times \ldots \times V_\mu \times \ldots \times V_n \,,
        \\[2mm]
        &x_\mu \mapsto \vec{x} = (0,\ldots,x_\mu,\ldots , 0) \,.
        \end{align}


    Parameters
    ----------
    mu : int
        The component on which to project.

    V : BlockVectorSpace | PolarDerhamSpace
        Codomain, out space.

    Vmu : StencilVectorSpace | PolarDerhamSpace
        Domain, in space, must be :math:`\mu`-th space of V.
    """

    def __init__(
        self,
        mu: int,
        V: BlockVectorSpace | PolarDerhamSpace,
        Vmu: StencilVectorSpace | PolarDerhamSpace,
    ):
        assert isinstance(mu, int)
        if isinstance(V, PolarDerhamSpace):
            assert V.parent_space.spaces[mu] == Vmu.parent_space
        else:
            assert V.spaces[mu] == Vmu

        self.dir = mu
        self._domain = Vmu
        self._codomain = V
        self._dtype = V.dtype

    @property
    def domain(self):
        """Domain vector space (input) of the operator."""
        return self._domain

    @property
    def codomain(self):
        """Codomain vector space (input) of the operator."""
        return self._codomain

    @property
    def dtype(self):
        """Datatype of the operator."""
        return self._dtype

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return CoordinateProjector(self.dir, self._codomain, self._domain)

    def dot(self, v: StencilVector | PolarVector, out=None):
        assert v.space == self._domain

        if isinstance(self.domain, PolarDerhamSpace):
            if out is not None:
                assert out.space == self._codomain
                out *= 0.0
            else:
                out = self._codomain.zeros()
            out._tp._blocks[self.dir] += v.tp

        else:
            if out is not None:
                assert out.space == self._codomain
                out *= 0.0
                out._blocks[self.dir] += v
            else:
                blocks = [sspace.zeros() for sspace in self.codomain.spaces]
                blocks[self.dir] = v.copy()
                out = BlockVector(self._codomain, blocks)

        out.update_ghost_regions()
        return out

    def idot(self, v: StencilVector | PolarVector, out: BlockVector | PolarVector):
        assert v.space == self._domain
        assert out.space == self._codomain
        out._blocks[self.dir] += v


def find_relative_col(col, row, Nbasis, periodic):
    """Compute the relative row position of a StencilMatrix from the global column and row positions.

    Parameters
    ----------
    col : int
        Global column index.

    row : int
        Global row index.

    Nbasis : int
        Number of B(or D)-splines for this particular dimension.

    periodic : bool
        True if we have periodic boundary conditions in this direction, otherwise False.

    Returns
    -------
    relativecol : int
        The relative column position of col with respect to the the current row of the StencilMatrix.

    """
    if not periodic:
        relativecol = col - row
    # In the periodic case we must account for the possible looping of the basis functions when computing the relative row postion
    else:
        if col <= row:
            if abs(col - row) <= abs(col + Nbasis - row):
                relativecol = col - row
            else:
                relativecol = col + Nbasis - row
        else:
            if abs(col - row) <= abs(col - Nbasis - row):
                relativecol = col - row
            else:
                relativecol = col - Nbasis - row
    return relativecol
