import inspect
import logging
from copy import deepcopy
from typing import Callable

import cunumpy as xp
from feectools.api.settings import PSYDAC_BACKEND_GPYCCEL
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from feectools.fem.tensor import FemSpace, TensorFemSpace
from feectools.fem.vector import VectorFemSpace
from feectools.linalg.basic import IdentityOperator, InverseLinearOperator, LinearOperator, Vector
from feectools.linalg.block import BlockLinearOperator, BlockVector
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilDiagonalMatrix, StencilMatrix, StencilVector

from struphy import equils
from struphy.feec import mass_kernels
from struphy.feec.linear_operators import BoundaryOperator, LinOpWithTransp
from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.feec.utilities import LocalProjectionMatrix, LocalRotationMatrix, get_quad_grids
from struphy.fields_background.base import MHDequilibrium
from struphy.geometry.base import Domain
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.polar.basic import PolarVector
from struphy.polar.linear_operators import PolarExtractionOperator
from struphy.utils.docstring_converter import auto_convert_docstring, info
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import __class_with_params_repr_no_defaults__

logger = logging.getLogger("struphy")


class WeightedMassOperators:
    r"""
    Collection of pre-defined :class:`struphy.feec.mass.WeightedMassOperator`.

    Parameters
    ----------
    derham : Derham
        Discrete de Rham sequence on the logical unit cube.

    domain : :ref:`avail_mappings`
        Mapping from logical unit cube to physical domain and corresponding metric coefficients.

    eq_mhd : MHDequilibrium | None
        MHD equilibrium object.

    matrix_free : bool
        If set to true will not compute the matrix associated with the operator but directly compute the product when called.
    """

    def __init__(
        self,
        derham: Derham,
        domain: Domain,
        eq_mhd: MHDequilibrium | None = None,
        matrix_free: bool = False,
    ):
        self._derham = derham
        self._domain = domain
        self._matrix_free = matrix_free
        self._eq_mhd = eq_mhd

        if self._eq_mhd is None:
            self._eq_mhd = equils.HomogenSlab()
        if not hasattr(self.eq_mhd, "_domain"):
            self._eq_mhd.domain = self._domain

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

    @property
    def derham(self) -> Derham:
        """Discrete de Rham sequence on the logical unit cube."""
        return self._derham

    @property
    def domain(self) -> Domain:
        """Mapping from the logical unit cube to the physical domain with corresponding metric coefficients."""
        return self._domain

    @property
    def eq_mhd(self) -> MHDequilibrium | None:
        """MHD equilibrium object."""
        return self._eq_mhd

    @property
    def matrix_free(self) -> bool:
        """If set to true will not compute the matrix associated with the operators but directly compute the dot product when called."""
        return self._matrix_free

    def info(self):
        print("The mass matrices of the Derham complex are:")
        self.M0.info()
        self.M1.info()
        self.M2.info()
        self.M3.info()
        print("Available mass operators as properties:")
        li = ""
        for name, method in inspect.getmembers(self.__class__, predicate=inspect.isdatadescriptor):
            if name.startswith("M") and isinstance(getattr(self.__class__, name), property):
                li += f"{name}, "
        li = li.rstrip(", ")
        print(f"{li}")
        print("\nTo see more details on a mass operator, call the info() method on it, e.g. mass_ops.M0.info().")

    #######################################################################
    # Mass matrices related to L2-scalar products in all 3d derham spaces #
    #######################################################################

    @auto_convert_docstring
    @property
    def M0(self):
        r"""
        Standard mass matrix for 0-forms (H1 space):

        .. math::

            \mathbb M^0_{ijk, mno} = \int \Lambda^0_{ijk}  \Lambda^0_{mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.
        """
        if not hasattr(self, "_M0"):
            self._M0 = self.create_weighted_mass(
                "H1",
                "H1",
                weights=("sqrt_g",),
                name="M0",
                assemble=True,
            )
        return self._M0

    @auto_convert_docstring
    @property
    def M1(self):
        r"""
        Standard mass matrix for 1-forms (Hcurl space) as 3x3 block matrix indexed by :math:`(\mu, \nu)`:

        .. math::

            \mathbb M^1_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^1_{\mu,ijk}\, G^{-1} \vec{\Lambda}^1_{\nu, mno} \sqrt{g}  \textnormal{d}\boldsymbol{\eta}.
        """

        if not hasattr(self, "_M1"):
            self._M1 = self.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=("Ginv", "sqrt_g"),
                name="M1",
                assemble=True,
            )

        return self._M1

    @auto_convert_docstring
    @property
    def M2(self):
        r"""
        Standard mass matrix for 2-forms (Hdiv space) as 3x3 block matrix indexed by :math:`(\mu, \nu)`:

        .. math::

            \mathbb M^2_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^2_{\mu,ijk} G \vec{\Lambda}^2_{\nu, mno} \frac{1}{\sqrt{g}} \textnormal{d}\boldsymbol{\eta}.
        """

        if not hasattr(self, "_M2"):
            self._M2 = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=("G", "1/sqrt_g"),
                name="M2",
                assemble=True,
            )

        return self._M2

    @auto_convert_docstring
    @property
    def M3(self):
        r"""
        Standard mass matrix for 3-forms (L2 space):

        .. math::

            \mathbb M^3_{ijk, mno} = \int \Lambda^3_{ijk}\,  \Lambda^3_{mno} \frac{1}{\sqrt{g}}\,  \textnormal{d}\boldsymbol{\eta}.
        """

        if not hasattr(self, "_M3"):
            self._M3 = self.create_weighted_mass(
                "L2",
                "L2",
                weights=("1/sqrt_g",),
                name="M3",
                assemble=True,
            )

        return self._M3

    @auto_convert_docstring
    @property
    def Mv(self):
        r"""
        Standard mass matrix for vector 0-forms (H1vec space) as 3x3 block matrix indexed by :math:`(\mu, \nu)`:

        .. math::

            \mathbb M^v_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^v_{\mu,ijk} G \vec{\Lambda}^v_{\nu, mno} \sqrt{g}  \textnormal{d}\boldsymbol{\eta}.
        """

        if not hasattr(self, "_Mv"):
            self._Mv = self.create_weighted_mass(
                "H1vec",
                "H1vec",
                weights=("G", "sqrt_g"),
                name="Mv",
                assemble=True,
            )

        return self._Mv

    ######################################
    # Predefined weighted mass operators #
    ######################################
    @auto_convert_docstring
    @property
    def M2stab_for_rot(self):
        r"""
        Stabilization matrix for the rot-problem u2 + B2 x u2 = G*f2 as 3x3 block matrix indexed by :math:`(\mu, \nu)`:

        .. math::

            \mathbb M^2_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^2_{\mu,ijk} \vec{\Lambda}^2_{\nu, mno} \frac{1}{\sqrt{g}} \textnormal{d}\boldsymbol{\eta}.
        """

        if not hasattr(self, "_M2stab_for_rot"):
            self._M2stab_for_rot = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=("Identity", "1/sqrt_g"),
                name="M2stab_for_rot",
                assemble=True,
            )

        return self._M2stab_for_rot

    @auto_convert_docstring
    @property
    def M1n(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,n}_{(\mu,ijk), (\nu,mno)} = \int n^0_{\textnormal{eq}}(\boldsymbol{\eta}) \vec{\Lambda}^1_{\mu,ijk} G^{-1} \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol{\eta})` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, "_M1n"):
            self._M1n = self.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    "Ginv",
                    "sqrt_g",
                    "eq_n0",
                ),
                name="M1n",
                assemble=True,
            )

        return self._M1n

    @auto_convert_docstring
    @property
    def M2n(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{2,n}_{(\mu,ijk), (\nu,mno)} = \int n^0_{\textnormal{eq}}(\boldsymbol{\eta}) \vec{\Lambda}^2_{\mu,ijk} G \vec{\Lambda}^2_{\nu,mno} \frac{1}{\sqrt{g}} \textnormal{d}\boldsymbol{\eta}.

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol{\eta})` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, "_M2n"):
            self._M2n = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=(
                    "G",
                    "1/sqrt_g",
                    "eq_n0",
                ),
                name="M2n",
                assemble=True,
            )

        return self._M2n

    @auto_convert_docstring
    @property
    def Mvn(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{v,n}_{(\mu,ijk), (\nu,mno)} = \int n^0_{\textnormal{eq}}(\boldsymbol{\eta}) \vec{\Lambda}^v_{\mu,ijk} G \vec{\Lambda}^v_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol{\eta})` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, "_Mvn"):
            self._Mvn = self.create_weighted_mass(
                "H1vec",
                "H1vec",
                weights=(
                    "G",
                    "sqrt_g",
                    "eq_n0",
                ),
                name="Mvn",
                assemble=True,
            )

        return self._Mvn

    @auto_convert_docstring
    @property
    def M1ninv(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,\frac{1}{n}}_{(\mu,ijk), (\nu,mno)} = \int \frac{1}{n^0_{\textnormal{eq}}(\boldsymbol{\eta})} \vec{\Lambda}^1_{\mu,ijk} G^{-1} \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol{\eta})` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, "_M1ninv"):
            self._M1ninv = self.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    "Ginv",
                    "sqrt_g",
                    lambda *etas: 1 / self.eq_mhd.n0(*etas),
                ),
                name="M1ninv",
                assemble=True,
            )

        return self._M1ninv

    @auto_convert_docstring
    @property
    def M1J(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,J}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^1_{\mu,ijk} G^{-1} \mathcal{R}(J) \vec{\Lambda}^2_{\nu,mno} \textnormal{d}\boldsymbol{\eta}.

        with the rotation matrix

        .. math::

            \mathcal{R}(J)_{\alpha,\nu} := \epsilon_{\alpha\beta\nu} J^2_{\textnormal{eq},\beta},\qquad s.t. \qquad \mathcal{R}(J) \vec{v} = \vec{J}^2_{\textnormal{eq}} \times \vec{v},

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium current density (2-form).
        """

        if not hasattr(self, "_M1J"):
            assert self.eq_mhd is not None, (
                "M1J requires an MHD equilibrium to be provided when initializing the WeightedMassOperators object."
            )
            rot_J = LocalRotationMatrix(
                self.eq_mhd.j2_1,
                self.eq_mhd.j2_2,
                self.eq_mhd.j2_3,
            )

            self._M1J = self.create_weighted_mass(
                "Hdiv",
                "Hcurl",
                weights=("Ginv", rot_J),
                name="M1J",
                assemble=True,
            )

        return self._M1J

    @auto_convert_docstring
    @property
    def M2J(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{2,J}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^2_{\mu,ijk} \mathcal{R}(J) \vec{\Lambda}^2_{\nu,mno} \frac{1}{\sqrt{g}} \textnormal{d}\boldsymbol{\eta}.

        with the rotation matrix

        .. math::

            \mathcal{R}(J)_{\alpha,\nu} := \epsilon_{\alpha\beta\nu} J^2_{\textnormal{eq},\beta},\qquad s.t. \qquad \mathcal{R}(J) \vec{v} = \vec{J}^2_{\textnormal{eq}} \times \vec{v},

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium current density (2-form).
        """

        if not hasattr(self, "_M2J"):
            assert self.eq_mhd is not None, (
                "M2J requires an MHD equilibrium to be provided when initializing the WeightedMassOperators object."
            )
            rot_J = LocalRotationMatrix(
                self.eq_mhd.j2_1,
                self.eq_mhd.j2_2,
                self.eq_mhd.j2_3,
            )

            self._M2J = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=(rot_J, "1/sqrt_g"),
                name="M2J",
                assemble=True,
            )

        return self._M2J

    @auto_convert_docstring
    @property
    def MvJ(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{v,J}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^v_{\mu,ijk} \mathcal{R}(J) \vec{\Lambda}^v_{\nu,mno} \textnormal{d}\boldsymbol{\eta}.

        with the rotation matrix

        .. math::

            \mathcal{R}(J)_{\alpha,\nu} := \epsilon_{\alpha\beta\nu} J^2_{\textnormal{eq},\beta},\qquad s.t. \qquad \mathcal{R}(J) \vec{v} = \vec{J}^2_{\textnormal{eq}} \times \vec{v},

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`J^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium current density (2-form).
        """

        if not hasattr(self, "_MvJ"):
            assert self.eq_mhd is not None, (
                "MvJ requires an MHD equilibrium to be provided when initializing the WeightedMassOperators object."
            )
            rot_J = LocalRotationMatrix(
                self.eq_mhd.j2_1,
                self.eq_mhd.j2_2,
                self.eq_mhd.j2_3,
            )

            self._MvJ = self.create_weighted_mass(
                "Hdiv",
                "H1vec",
                weights=(rot_J,),
                name="MvJ",
                assemble=True,
            )

        return self._MvJ

    @auto_convert_docstring
    @property
    def M2B_div0(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{2,B}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^2_{\mu,ijk} \mathcal{R}(B) \vec{\Lambda}^2_{\nu,mno} \frac{1}{\sqrt{g}} \textnormal{d}\boldsymbol{\eta}.

        with the rotation matrix

        .. math::

            \mathcal{R}(B)_{\alpha,\nu} := \epsilon_{\alpha\beta\nu} B^2_{\textnormal{eq},\beta},\qquad s.t. \qquad \mathcal{R}(B) \vec{v} = \vec{B}^2_{\textnormal{eq}} \times \vec{v},

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium magnetic field (2-form).
        """

        if not hasattr(self, "_M2B_div0"):
            assert self.eq_mhd is not None, (
                "M2B_div0 requires an MHD equilibrium to be provided when initializing the WeightedMassOperators object."
            )
            a_eq = self.derham.P1(
                [
                    self.eq_mhd.a1_1,
                    self.eq_mhd.a1_2,
                    self.eq_mhd.a1_3,
                ],
            )

            tmp_b2 = self.derham.curl.dot(a_eq)
            b02fun = self.derham.create_spline_function("b02", "Hdiv")
            b02fun.vector = tmp_b2

            def b02funx(x, y, z):
                return b02fun(
                    x,
                    y,
                    z,
                    local=True,
                )[0]

            def b02funy(x, y, z):
                return b02fun(
                    x,
                    y,
                    z,
                    local=True,
                )[1]

            def b02funz(x, y, z):
                return b02fun(
                    x,
                    y,
                    z,
                    local=True,
                )[2]

            rot_B = LocalRotationMatrix(
                b02funx,
                b02funy,
                b02funz,
            )

            self._M2B_div0 = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=(rot_B, "1/sqrt_g"),
                name="M2B_div0",
                assemble=True,
            )

        return self._M2B_div0

    @auto_convert_docstring
    @property
    def M2B(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{2,B}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^2_{\mu,ijk} \mathcal{R}(B) \vec{\Lambda}^2_{\nu,mno} \frac{1}{\sqrt{g}} \textnormal{d}\boldsymbol{\eta}.

        with the rotation matrix

        .. math::

            \mathcal{R}(B)_{\alpha,\nu} := \epsilon_{\alpha\beta\nu} B^2_{\textnormal{eq},\beta},\qquad s.t. \qquad \mathcal{R}(B) \vec{v} = \vec{B}^2_{\textnormal{eq}} \times \vec{v},

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium magnetic field (2-form).
        """

        if not hasattr(self, "_M2B"):
            assert self.eq_mhd is not None, (
                "M2B requires an MHD equilibrium to be provided when initializing the WeightedMassOperators object."
            )
            rot_B = LocalRotationMatrix(
                self.eq_mhd.b2_1,
                self.eq_mhd.b2_2,
                self.eq_mhd.b2_3,
            )

            self._M2B = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=(rot_B, "1/sqrt_g"),
                name="M2B",
                assemble=True,
            )

        return self._M2B

    @auto_convert_docstring
    @property
    def M2Bn(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{2,BN}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^2_{\mu,ijk} \mathcal{R}(B) \vec{\Lambda}^2_{\nu,mno} \frac{1}{n^0_{\textnormal{eq}}(\boldsymbol{\eta})} \frac{1}{\sqrt{g}} \textnormal{d}\boldsymbol{\eta}.

        with the rotation matrix

        .. math::

            \mathcal{R}(B)_{\alpha,\nu} := \epsilon_{\alpha\beta\nu} B^2_{\textnormal{eq},\beta},\qquad s.t. \qquad \mathcal{R}(B) \vec{v} = \vec{B}^2_{\textnormal{eq}} \times \vec{v},

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium magnetic field (2-form).
        """

        if not hasattr(self, "_M2Bn"):
            assert self.eq_mhd is not None, (
                "M2Bn requires an MHD equilibrium to be provided when initializing the WeightedMassOperators object."
            )
            a_eq = self.derham.P1(
                [
                    self.eq_mhd.a1_1,
                    self.eq_mhd.a1_2,
                    self.eq_mhd.a1_3,
                ],
            )

            tmp_b2 = self.derham.curl.dot(a_eq)
            b02fun = self.derham.create_spline_function("b02", "Hdiv")
            b02fun.vector = tmp_b2

            def b02funx(x, y, z):
                return b02fun(
                    x,
                    y,
                    z,
                    local=True,
                )[0]

            def b02funy(x, y, z):
                return b02fun(
                    x,
                    y,
                    z,
                    local=True,
                )[1]

            def b02funz(x, y, z):
                return b02fun(
                    x,
                    y,
                    z,
                    local=True,
                )[2]

            rot_B = LocalRotationMatrix(
                b02funx,
                b02funy,
                b02funz,
            )

            self._M2Bn = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=(
                    rot_B,
                    "1/sqrt_g",
                    "eq_n0",
                ),
                name="M2Bn",
                assemble=True,
            )

        return self._M2Bn

    @auto_convert_docstring
    @property
    def M1Bninv(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,B\frac{1}{n}}_{(\mu,ijk), (\nu,mno)} = \int \frac{1}{n^0_{\textnormal{eq}}(\boldsymbol{\eta})} \vec{\Lambda}^1_{\mu,ijk} G^{-1} \mathcal{R}(B)_{\alpha,\gamma} G^{-1}_{\gamma,\nu} \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.

        with the rotation matrix

        .. math::

            \mathcal{R}(B)_{\alpha,\nu} := \epsilon_{\alpha\beta\nu} B^2_{\textnormal{eq},\beta},\qquad s.t. \qquad \mathcal{R}(B) \vec{v} = \vec{B}^2_{\textnormal{eq}} \times \vec{v},

        where :math:`\epsilon_{\alpha \beta \nu}` stands for the Levi-Civita tensor and :math:`B^2_{\textnormal{eq}, \beta}` is the :math:`\beta`-component of the MHD equilibrium magnetic field (2-form).
        """

        if not hasattr(self, "_M1Bninv"):
            assert self.eq_mhd is not None, (
                "M1Bninv requires an MHD equilibrium to be provided when initializing the WeightedMassOperators object."
            )
            rot_B = LocalRotationMatrix(
                self.eq_mhd.b2_1,
                self.eq_mhd.b2_2,
                self.eq_mhd.b2_3,
            )

            self._M1Bninv = self.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    "Ginv",
                    rot_B,
                    "Ginv",
                    "sqrt_g",
                    lambda *etas: 1 / self.eq_mhd.n0(*etas),
                ),
                name="M1Bninv",
                assemble=True,
            )

        return self._M1Bninv

    @auto_convert_docstring
    @property
    def M1para(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,\parallel}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^1_{\mu,ijk} b_0 b_0^\top \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.
        """
        if not hasattr(self, "_M1para"):
            bb = LocalProjectionMatrix(self.eq_mhd.unit_bv_1, self.eq_mhd.unit_bv_2, self.eq_mhd.unit_bv_3)

            self._M1para = self.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    bb,
                    "sqrt_g",
                ),
                name="M1para",
                assemble=True,
            )
        return self._M1para

    @auto_convert_docstring
    @property
    def M1perp(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,\perp}_{(\mu,ijk), (\nu,mno)} = \int \vec{\Lambda}^1_{\mu,ijk} \left(G^{-1} - b_0 b_0^\top \right) \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.
        """
        if not hasattr(self, "_M1perp"):
            self._M1perp = self.M1.copy()
            self._M1perp -= self.M1para
        return self._M1perp

    @auto_convert_docstring
    @property
    def M1para_MHDeq(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,\parallel}_{(\mu,ijk), (\nu,mno)} = \int \frac{n^0_{\textnormal{eq}}(\boldsymbol{\eta})}{\|B_0(\boldsymbol{\eta})\|^2} \vec{\Lambda}^1_{\mu,ijk} b_0 b_0^\top \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.
        """
        if not hasattr(self, "_M1para_MHDeq"):
            bb = LocalProjectionMatrix(self.eq_mhd.unit_bv_1, self.eq_mhd.unit_bv_2, self.eq_mhd.unit_bv_3)

            self._M1para_MHDeq = self.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    bb,
                    lambda *etas: 1 / self.eq_mhd.absB0(*etas) ** 2,
                    "eq_n0",
                    "sqrt_g",
                ),
                name="M1para_MHDeq",
                assemble=True,
            )
        return self._M1para_MHDeq

    @auto_convert_docstring
    @property
    def M1_MHDeq(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1}_{(\mu,ijk), (\nu,mno)} = \int \frac{n^0_{\textnormal{eq}}(\boldsymbol{\eta})}{\|B_0(\boldsymbol{\eta})\|^2} \vec{\Lambda}^1_{\mu,ijk} G^{-1} \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.
        """
        if not hasattr(self, "_M1_MHDeq"):
            self._M1_MHDeq = self.create_weighted_mass(
                "Hcurl",
                "Hcurl",
                weights=(
                    "Ginv",
                    lambda *etas: 1 / self.eq_mhd.absB0(*etas) ** 2,
                    "eq_n0",
                    "sqrt_g",
                ),
                name="M1_MHDeq",
                assemble=True,
            )
        return self._M1_MHDeq

    @auto_convert_docstring
    @property
    def M1gyro(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^{1,\perp}_{(\mu,ijk), (\nu,mno)} = \int \frac{n^0_{\textnormal{eq}}(\boldsymbol{\eta})}{\|B_0(\boldsymbol{\eta})\|^2} \vec{\Lambda}^1_{\mu,ijk} \left(G^{-1} - b_0 b_0^\top \right) \vec{\Lambda}^1_{\nu,mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.
        """
        if not hasattr(self, "_M1gyro"):
            self._M1gyro = self.M1_MHDeq.copy()
            self._M1gyro -= self.M1para_MHDeq
        return self._M1gyro

    @auto_convert_docstring
    @property
    def M0ad(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^0_{ijk, mno} = \int n^0_{\textnormal{eq}}(\boldsymbol{\eta}) \Lambda^0_{ijk} \Lambda^0_{mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol{\eta})` is an MHD equilibrium density (0-form).
        """

        if not hasattr(self, "_M0ad"):
            self._M0ad = self.create_weighted_mass(
                "H1",
                "H1",
                weights=(
                    "eq_n0",
                    "sqrt_g",
                ),
                name="M0ad",
                assemble=True,
            )

        return self._M0ad

    @auto_convert_docstring
    @property
    def M0ad_withT(self):
        r"""
        Mass matrix

        .. math::

            \mathbb M^0_{ijk, mno} = \int \frac{n^0_{\textnormal{eq}}(\boldsymbol{\eta})}{T^0_{\textnormal{eq}}(\boldsymbol{\eta})} \Lambda^0_{ijk} \Lambda^0_{mno} \sqrt{g} \textnormal{d}\boldsymbol{\eta}.

        where :math:`n^0_{\textnormal{eq}}(\boldsymbol{\eta})` and :math:`T^0_{\textnormal{eq}}(\boldsymbol{\eta})` are MHD equilibrium density and electron temperature (0-forms), respectively.
        """
        if not hasattr(self, "_M0ad_withT"):
            self._M0ad_withT = self.create_weighted_mass(
                "H1",
                "H1",
                weights=(
                    "eq_n0",
                    lambda *etas: 1 / self.eq_mhd.t0(*etas),
                    "sqrt_g",
                ),
                name="M0ad_withT",
                assemble=True,
            )

        return self._M0ad_withT

    @property
    def WMM(self):
        if not hasattr(self, "_WMM"):
            self._WMM = self.H1vecMassMatrix_density(self.derham, self, self.domain)
        return self._WMM

    @property
    def WMMnew(self):
        if not hasattr(self, "_WMMnew"):
            spline = self.derham.create_spline_function("l2_field", "L2")
            self._WMMnew = self.create_weighted_mass(
                "H1vec",
                "H1vec",
                weights=("G", spline),
                name="WMMnew",
                assemble=False,
            )
        return self._WMMnew

    #######################################
    # Wrapper around WeightedMassOperator #
    #######################################
    def create_weighted_mass(
        self,
        V_id: str,
        W_id: str,
        *,
        name: str = None,
        weights: tuple | list | str | None = None,
        assemble: bool = False,
        transposed: bool = False,
    ):
        r"""Weighted mass matrix :math:`V^\alpha_h \to V^\beta_h` with given (matrix-valued) weight function :math:`W(\boldsymbol \eta)`:

        .. math::

            \mathbb M_{(\mu, ijk), (\nu, mno)}(W) = \int \Lambda^\beta_{\mu, ijk}\, W_{\mu,\nu}(\boldsymbol \eta)\,  \Lambda^\alpha_{\nu, mno} \,  \textnormal d \boldsymbol\eta.

        Here, :math:`\alpha \in \{0, 1, 2, 3, v\}` indicates the domain and :math:`\beta \in \{0, 1, 2, 3, v\}` indicates the co-domain
        of the operator.

        Parameters
        ----------
        V_id : str
            Specifier for the domain of the operator ('H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec').

        W_id : str
            Specifier for the co-domain of the operator ('H1', 'Hcurl', 'Hdiv', 'L2' or 'H1vec').

        name: str
            Name of the operator.

        weights : None | str | tuple | list
            Information about the weights/block structure of the operator.
            Four cases are possible:

                1. ``None`` : all blocks are allocated, disregarding zero-blocks or any symmetry.
                2. ``str``  : for square block matrices (V=W), a symmetry can be set in order to accelerate the assembly process.
                    Possible strings are ``symm`` (symmetric), ``asym`` (anti-symmetric) and ``diag`` (diagonal).
                3. ``1D tuple`` : most common and recommended input format.
                    Entries are processed from left to right and multiplied together.

                    Supported tuple entries are:

                    - Strings (predefined names):
                        ``'G'``, ``'Ginv'``, ``'DFinv'``, ``'DFinvT'``, ``'sqrt_g'``, ``'Identity'``.
                    - Strings from equilibrium methods:
                        ``'eq_<method_name>'`` for methods of :class:`~struphy.fields_background.base.MHDequilibrium`.
                    - Reciprocal strings:
                        ``'1/sqrt_g'``, ``'1/eq_n0'``, ``'1/eq_absB0'``.
                    - Nested ``3x3`` Python lists (constant matrix entries).
                    - Callables (including objects such as local rotation matrices) returning
                        either scalar values or ``3x3`` matrix values at quadrature points.
                    - :class:`~struphy.feec.psydac_derham.SplineFunction` instances.

                    Example:
                    ``weights=('Ginv', 'sqrt_g')``
                4. ``2D list`` : 2d list with the same number of rows/columns as the number of components of the domain/codomain spaces.
                    The entries can be either a) callables or b) xp.ndarrays representing the weights at the quadrature points.
                    If an entry is zero or ``None``, the corresponding block is set to ``None`` to accelerate the dot product.

        assemble: bool
            Whether to assemble the weighted mass matrix, i.e. computes the integrals with
            :class:`~struphy.feec.mass.WeightedMassOperators.assemble`

        transposed: bool
            Whether to assemble the transposed operator.

        Returns
        -------
        out : A WeightedMassOperator object.
        """
        logger.debug(f"\nCreating weighted mass matrix {name} from {V_id} to {W_id}.")

        spline_functions = {}
        if isinstance(weights, tuple):  # Case 3 (1D tuple)
            # save callables in lists for later evaluation at quadrature points
            f_call_scalars = []
            f_call_column_vector = None
            f_call_row_vector = None
            f_call_matrices = []

            for n, f in enumerate(weights):
                logger.debug(f"Processing weight #{n}")
                if isinstance(f, str):
                    # determine the callable and add to list f_call_
                    logger.debug(f"Processing string weight {f}.")
                    if f == "1/sqrt_g":
                        f_call = lambda e1, e2, e3: 1.0 / abs(self.domain.jacobian_det(e1, e2, e3))
                        f_call_scalars.append(f_call)
                    elif "eq_n0" in f:
                        f_call = getattr(self.eq_mhd, "n0")
                        f_call_scalars.append(f_call)
                    else:
                        if f == "G":
                            f_call = lambda e1, e2, e3: self.domain.metric(e1, e2, e3, change_out_order=True)
                            f_call_matrices.append(f_call)
                        elif f == "Ginv":
                            f_call = lambda e1, e2, e3: self.domain.metric_inv(e1, e2, e3, change_out_order=True)
                            f_call_matrices.append(f_call)
                        elif f == "DFinv":
                            f_call = lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3, change_out_order=True)
                            f_call_matrices.append(f_call)
                        elif f == "DFinvT":
                            f_call = lambda e1, e2, e3: self.domain.jacobian_inv(
                                e1, e2, e3, change_out_order=True, transposed=True
                            )
                            f_call_matrices.append(f_call)
                        elif f == "sqrt_g":
                            f_call = lambda e1, e2, e3: abs(self.domain.jacobian_det(e1, e2, e3))
                            f_call_scalars.append(f_call)
                        elif f == "Identity":

                            def f_call(e1, e2, e3):
                                """Identity callable."""
                                # to keep C-ordering the (3, 3)-part is in the last indices
                                out = xp.zeros((3, 3, e1.shape[0], e2.shape[1], e3.shape[2]), dtype=float)
                                out[0, 0] = 1.0
                                out[1, 1] = 1.0
                                out[2, 2] = 1.0
                                return xp.transpose(out, axes=(2, 3, 4, 0, 1))

                            f_call_matrices.append(f_call)
                        else:
                            raise NotImplementedError(
                                f"The option {f} is not available.",
                            )
                elif isinstance(f, list):
                    assert len(f) == 3
                    logger.debug(f"Processing nested list weight {f}.")
                    for fi in f:
                        assert isinstance(fi, list)
                        assert len(fi) == 3

                    # copy the values of the nested list to avoid any issues with references
                    values = tuple([tuple([value for value in fi_row]) for fi_row in f])

                    def f_call(e1, e2, e3):
                        """Nested list callable."""
                        out = xp.zeros((3, 3, e1.shape[0], e2.shape[1], e3.shape[2]), dtype=float)
                        for m in range(3):
                            for n in range(3):
                                logger.debug(f"{values[m][n] = }")
                                out[m, n] = values[m][n]
                        return xp.transpose(out, axes=(2, 3, 4, 0, 1))

                    f_call_matrices.append(f_call)
                elif isinstance(f, SplineFunction):
                    logger.debug(f"Processing SplineFunction weight {f}.")
                    spline_functions[f.name] = f
                    continue
                else:
                    # Input is a a matrix or a Rotation matrix etc.
                    logger.debug(f"Processing callable weight {f}.")
                    assert callable(f)

                    # determine the output dimension of the callable and add to list f_call_
                    xx, yy, zz = xp.meshgrid(
                        xp.linspace(0, 1, 1),
                        xp.linspace(0, 1, 2),
                        xp.linspace(0, 1, 3),
                        indexing="ij",
                    )
                    out_dim = f(xx, yy, zz).ndim
                    if out_dim == 3:
                        f_call_scalars.append(f)
                    elif out_dim == 4:
                        f_call_column_vector = f
                        f_call_row_vector = f
                    elif out_dim == 5:
                        f_call_matrices.append(f)
                    else:
                        raise ValueError(
                            f"Callable {f} has wrong output dimension {out_dim}.",
                        )

            # check that the dimensions of the callables are compatible with the domain and codomain spaces
            if f_call_column_vector is not None:
                assert V_id in ("Hcurl", "Hdiv", "H1vec")
                assert len(f_call_matrices) == 0
                assert f_call_row_vector is None
            if f_call_row_vector is not None:
                assert W_id in ("Hcurl", "Hdiv", "H1vec")
                assert len(f_call_matrices) == 0
                assert f_call_column_vector is None
            if len(f_call_matrices) > 0:
                assert V_id in ("Hcurl", "Hdiv", "H1vec")
                assert W_id in ("Hcurl", "Hdiv", "H1vec")
                assert f_call_column_vector is None
                assert f_call_row_vector is None

            # matrix-matrix multiplication of the callables in f_call_matrices to get a single callable
            if len(f_call_matrices) > 0:

                def f_call_matrix(e1, e2, e3):
                    """Matrix-matrix multiplication of the callables in f_call_matrices."""
                    out = f_call_matrices[0](e1, e2, e3)
                    if len(f_call_matrices) > 1:
                        for f in f_call_matrices[1:]:
                            # out = xp.einsum("...ij,...jk->...ik", out, f(e1, e2, e3))
                            out[:] = out @ f(e1, e2, e3)  # the 3x3 part must be in the last two indices
                    return out

            # get the evaluation points for the quadrature grid of the codomain space W_id
            assert W_id in self.derham.spline_attributes, (
                f"Spline attributes for the codomain space {W_id} not found in the Derham object !!"
            )

            quad_grid_pts = self.derham.spline_attributes[W_id].quad_grid_pts
            logger.debug(f"{len(quad_grid_pts) = }")

            weights_values = []
            integration_grids = []
            # loop over components of W_id (rows, equal to the number of entries in quad_grid_pts)
            for component in quad_grid_pts:
                grids_1d = [pts.flatten() for pts in component]
                grid_sizes = tuple([len(grid_1d) for grid_1d in grids_1d])
                logger.debug(f"Initializing {grid_sizes = }")
                integration_grids += [grids_1d]

                # loop over components of V_id (columns)
                if V_id in ("H1", "L2"):
                    weights_values += [[None]]
                elif V_id in ("Hcurl", "Hdiv", "H1vec"):
                    weights_values += [[None, None, None]]
                else:
                    raise ValueError(f"Unknown space identifier {V_id} for the domain.")
            logger.debug(f"Initialized {weights_values = }")

            # evaluate at quadrature points, loop over rows of W_id (components of the codomain)
            for m, grids_1d in enumerate(integration_grids):
                logger.debug(f"rows of {W_id}: {m}")
                E1, E2, E3, _ = Domain.prepare_eval_pts(*grids_1d)

                # matrix or vectors first
                if len(f_call_matrices) > 0:
                    tmp = f_call_matrix(E1, E2, E3)
                    logger.debug(f"Evaluated matrix callable with shape {tmp.shape = }")
                    logger.debug(f"max value: {xp.max(tmp)}, min value: {xp.min(tmp)}")
                    for n in range(len(weights_values[m])):
                        logger.debug(f"columns of {V_id}: {n}")
                        weights_values[m][n] = tmp[:, :, :, m, n]
                elif f_call_column_vector is not None:
                    tmp = f_call_column_vector(E1, E2, E3)
                    logger.debug(f"Evaluated column vector callable with shape {tmp.shape = }")
                    logger.debug(f"max value: {xp.max(tmp)}, min value: {xp.min(tmp)}")
                    for n in range(len(weights_values[m])):
                        logger.debug(f"columns of {V_id}: {n}")
                        weights_values[m][n] = tmp[:, :, :, m]
                elif f_call_row_vector is not None:
                    tmp = f_call_row_vector(E1, E2, E3)
                    logger.debug(f"Evaluated row vector callable with shape {tmp.shape = }")
                    logger.debug(f"max value: {xp.max(tmp)}, min value: {xp.min(tmp)}")
                    for n in range(len(weights_values[m])):
                        logger.debug(f"columns of {V_id}: {n}")
                        weights_values[m][n] = tmp[:, :, :, n]

                # then loop over scalars and multiply with the previous result
                for f_call in f_call_scalars:
                    tmp = f_call(E1, E2, E3)
                    logger.debug(f"Evaluated scalar callable with shape {tmp.shape = }")
                    logger.debug(f"max value: {xp.max(tmp)}, min value: {xp.min(tmp)}")
                    for n in range(len(weights_values[m])):
                        logger.debug(f"columns of {V_id}: {n}")
                        if weights_values[m][n] is None and m == n:
                            weights_values[m][n] = tmp
                        else:
                            weights_values[m][n] *= tmp

        else:
            logger.debug(f"Processing weights of type {type(weights)}.")
            weights_values = weights

        out = WeightedMassOperator(
            self.derham,
            self.derham.fem_spaces[V_id],
            self.derham.fem_spaces[W_id],
            name=name,
            V_extraction_op=self.derham.extraction_ops[V_id],
            W_extraction_op=self.derham.extraction_ops[W_id],
            V_boundary_op=self.derham.boundary_ops[V_id],
            W_boundary_op=self.derham.boundary_ops[W_id],
            weights_info=weights_values,
            spline_functions=spline_functions,
            transposed=transposed,
            matrix_free=self.matrix_free,
        )

        if assemble:
            out.assemble()

        return out

    #######################################
    # Aux classes (to be removed in TODO) #
    #######################################
    class H1vecMassMatrix_density:
        """Wrapper around a Weighted mass operator from H1vec to H1vec whose weights are given by a 3 form"""

        def __init__(self, derham, mass_ops, domain):
            self._massop = mass_ops.create_weighted_mass("H1vec", "H1vec")
            self.field = derham.create_spline_function("field", "L2")

            integration_grid = [grid_1d.flatten() for grid_1d in derham.V0splines.quad_grid_pts[0]]

            self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
                derham.prepare_eval_tp_fixed(
                    integration_grid,
                )
            )

            grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
            self._f_values = xp.zeros(grid_shape, dtype=float)

            metric = domain.metric(*integration_grid)
            self._mass_metric_term = deepcopy(metric)
            self._full_term_mass = deepcopy(metric)

            self.domain_symbolic_name = self.massop.domain_symbolic_name

        @property
        def massop(
            self,
        ):
            """The WeightedMassOperator"""
            return self._massop

        @property
        def inv(
            self,
        ):
            """The inverse WeightedMassOperator"""
            if not hasattr(self, "_inv"):
                self._create_inv()
            return self._inv

        def update_weight(self, coeffs):
            """Update the weighted mass matrix operator"""

            self.field.vector = coeffs
            f_values = self.field.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._f_values,
            )
            for i in range(3):
                for j in range(3):
                    self._full_term_mass[i, j] = f_values * self._mass_metric_term[i, j]

            self._massop.assemble(
                [
                    [self._full_term_mass[0, 0], self._full_term_mass[0, 1], self._full_term_mass[0, 2]],
                    [
                        self._full_term_mass[1, 0],
                        self._full_term_mass[
                            1,
                            1,
                        ],
                        self._full_term_mass[1, 2],
                    ],
                    [self._full_term_mass[2, 0], self._full_term_mass[2, 1], self._full_term_mass[2, 2]],
                ],
            )

            if hasattr(self, "_inv") and self.inv._options["pc"] is not None:
                self.inv._options["pc"].update_mass_operator(self.massop)

        def _create_inv(self, type="pcg", tol=1e-16, maxiter=500):
            """Inverse the  weighted mass matrix, preconditioner must be set outside
            via self._inv._options['pc'] = ..."""
            self._inv = inverse(
                self.massop,
                type,
                pc=None,
                tol=tol,
                maxiter=maxiter,
                recycle=True,
            )


class WeightedMassOperator(LinOpWithTransp):
    r"""
    Class for assembling weighted mass matrices in 3d.

    Weighted mass matrices :math:`\mathbb M^{\beta\alpha}: \mathbb R^{N_\alpha} \to \mathbb R^{N_\beta}`
    are of the general form

    .. math::

        \mathbb M^{\beta \alpha}_{(\mu,ijk),(\nu,mno)} = \int_{[0, 1]^3} \Lambda^\beta_{\mu,ijk} \, A_{\mu,\nu} \, \Lambda^\alpha_{\nu,mno} \, \textnormal d^3 \boldsymbol\eta\,,

    where the weight fuction :math:`A` is a tensor of rank 0, 1 or 2,
    depending on domain and co-domain of the operator,
    and :math:`\Lambda^\alpha_{\nu, mno}` is the B-spline basis function
    with tensor-product index :math:`mno` of the
    :math:`\nu`-th component in the space :math:`V^\alpha_h`.
    These matrices are sparse and stored in StencilMatrix format.

    Finally, :math:`\mathbb M^{\beta\alpha}` can be multiplied by
    :class:`~struphy.polar.linear_operators.PolarExtractionOperator`
    and :class:`~struphy.feec.linear_operators.BoundaryOperator`,
    :math:`\mathbb B\, \mathbb E\, \mathbb M^{\beta\alpha} \mathbb E^T \mathbb B^T`,
    to account for :ref:`polar_splines` and/or :ref:`feec_bcs`, respectively.

    Parameters
    ----------
    derham : Derham
        Struphy Derham object.

    V : TensorFemSpace | VectorFemSpace
        Tensor product spline space from feectools.fem.tensor (domain, input space).

    W : TensorFemSpace | VectorFemSpace
        Tensor product spline space from feectools.fem.tensor (codomain, output space).

    name : str
        Name of the operator.

    V_extraction_op : PolarExtractionOperator, optional
        Extraction operator to polar sub-space of V.

    W_extraction_op : PolarExtractionOperator, optional
        Extraction operator to polar sub-space of W.

    V_boundary_op : BoundaryOperator, optional
        Boundary operator that sets essential boundary conditions.

    W_boundary_op : BoundaryOperator, optional
        Boundary operator that sets essential boundary conditions.

    weights_info : NoneType | str | list
        Information about the weights/block structure of the operator.
        Three cases are possible:

        1. ``None`` : all blocks are allocated, disregarding zero-blocks or any symmetry.
        2. ``str``  : for square block matrices (V=W), a symmetry can be set in order to accelerate the assembly process. Possible strings are ``symm`` (symmetric), ``asym`` (anti-symmetric) and ``diag`` (diagonal).
        3. ``list`` : 2d list with the same number of rows/columns as the number of components of the domain/codomain spaces. The entries can be either a) callables or b) xp.ndarrays representing the weights at the quadrature points. If an entry is zero or ``None``, the corresponding block is set to ``None`` to accelerate the dot product.

    spline_functions : dict[str, SplineFunction]
        Dictionary of spline functions that are used as weights in the operator.
        The keys must be the names of the spline functions and the values the SplineFunction objects.

    transposed : bool
        Whether to assemble the transposed operator.

    matrix_free : bool
        If set to true will not compute the matrix associated with the operator but directly compute the product when called
    """

    def __init__(
        self,
        derham: Derham,
        V: TensorFemSpace | VectorFemSpace,
        W: TensorFemSpace | VectorFemSpace,
        name: str = None,
        V_extraction_op: PolarExtractionOperator | IdentityOperator = None,
        W_extraction_op: PolarExtractionOperator | IdentityOperator = None,
        V_boundary_op: BoundaryOperator | IdentityOperator = None,
        W_boundary_op: BoundaryOperator | IdentityOperator = None,
        weights_info: str | list = None,
        spline_functions: dict[str, SplineFunction] | None = None,
        transposed: bool = False,
        matrix_free: bool = False,
        nquads: tuple | list = None,
    ):
        logger.debug(f"{derham = }")
        logger.debug(f"{V = }")
        logger.debug(f"{W = }")
        logger.debug(f"{name = }")
        logger.debug(f"{V_extraction_op = }")
        logger.debug(f"{W_extraction_op = }")
        logger.debug(f"{V_boundary_op = }")
        logger.debug(f"{W_boundary_op = }")
        logger.debug(f"{type(weights_info) = }")
        logger.debug(f"{spline_functions = }")
        logger.debug(f"{transposed = }")
        logger.debug(f"{matrix_free = }")
        logger.debug(f"{nquads = }")

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

        self._derham = derham
        self._nquads = nquads

        self._V = V
        self._W = W
        self._name = name

        # spline functions that are used as weights in the operator, to be evaluated at quadrature points
        self._spline_functions = spline_functions if spline_functions is not None else {}

        # set basis extraction operators
        if V_extraction_op is not None:
            assert V_extraction_op.domain == V.coeff_space
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(V.coeff_space)

        if W_extraction_op is not None:
            assert W_extraction_op.domain == W.coeff_space
            self._W_extraction_op = W_extraction_op
        else:
            self._W_extraction_op = IdentityOperator(W.coeff_space)

        # set boundary operators
        if V_boundary_op is not None:
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(
                self._V_extraction_op.codomain,
            )

        if W_boundary_op is not None:
            self._W_boundary_op = W_boundary_op
        else:
            self._W_boundary_op = IdentityOperator(
                self._W_extraction_op.codomain,
            )

        self._weights_info = weights_info
        self._transposed = transposed
        self._matrix_free = matrix_free

        self._dtype = V.coeff_space.dtype

        # set domain and codomain symbolic names
        logger.debug(f"{V.symbolic_space = }")
        logger.debug(f"{W.symbolic_space = }")
        V_name = V.symbolic_space
        W_name = W.symbolic_space

        assert V_name in derham.spline_attributes, (
            f"Spline attributes for the domain space {V_name} not found in the Derham object !!"
        )
        assert W_name in derham.spline_attributes, (
            f"Spline attributes for the codomain space {W_name} not found in the Derham object !!"
        )

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

        # Are both space scalar spaces : useful to know if _dof_mat will be Stencil or Block Matrix
        self._is_scalar = False
        if isinstance(V, TensorFemSpace) and isinstance(W, TensorFemSpace):
            self._is_scalar = True

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

            assert V_name == W_name, "only square matrices (V=W) allowed!"
            assert (
                len(
                    V_name,
                )
                > 2
            ), "only block matrices with domain/codomain spaces Hcurl, Hdiv and H1vec are allowed!"

            if self._matrix_free:
                if weights_info == "symm":
                    blocks = [
                        [StencilMatrixFreeMassOperator(self.derham, Vs, Ws, nquads=self.nquads) for Vs in V.spaces]
                        for Ws in W.spaces
                    ]
                elif weights_info == "asym":
                    blocks = [
                        [
                            StencilMatrixFreeMassOperator(self.derham, Vs, Ws, nquads=self.nquads) if i != j else None
                            for j, Vs in enumerate(V.spaces)
                        ]
                        for i, Ws in enumerate(W.spaces)
                    ]
                elif weights_info == "diag":
                    blocks = [
                        [
                            StencilMatrixFreeMassOperator(self.derham, Vs, Ws, nquads=self.nquads) if i == j else None
                            for j, Vs in enumerate(V.spaces)
                        ]
                        for i, Ws in enumerate(W.spaces)
                    ]
                else:
                    raise NotImplementedError(
                        f"given symmetry {weights_info} is not implemented!",
                    )

            else:
                if weights_info == "symm":
                    blocks = [
                        [
                            StencilMatrix(
                                Vs.coeff_space,
                                Ws.coeff_space,
                                backend=PSYDAC_BACKEND_GPYCCEL,
                                precompiled=True,
                            )
                            for Vs in V.spaces
                        ]
                        for Ws in W.spaces
                    ]
                elif weights_info == "asym":
                    blocks = [
                        [
                            StencilMatrix(
                                Vs.coeff_space,
                                Ws.coeff_space,
                                backend=PSYDAC_BACKEND_GPYCCEL,
                                precompiled=True,
                            )
                            if i != j
                            else None
                            for j, Vs in enumerate(V.spaces)
                        ]
                        for i, Ws in enumerate(W.spaces)
                    ]
                elif weights_info == "diag":
                    blocks = [
                        [
                            StencilMatrix(
                                Vs.coeff_space,
                                Ws.coeff_space,
                                backend=PSYDAC_BACKEND_GPYCCEL,
                                precompiled=True,
                            )
                            if i == j
                            else None
                            for j, Vs in enumerate(V.spaces)
                        ]
                        for i, Ws in enumerate(W.spaces)
                    ]
                else:
                    raise NotImplementedError(
                        f"given symmetry {weights_info} is not implemented!",
                    )

            self._mat = BlockLinearOperator(
                V.coeff_space,
                W.coeff_space,
                blocks=blocks,
            )

            # set zero default weights with same block structure as block matrix
            self._weights = []
            for block_row in blocks:
                self._weights += [[]]
                for block in block_row:
                    if block is None:
                        self._weights[-1] += [None]
                    else:
                        self._weights[-1] += [lambda *etas: 0 * etas[0]]

        # OR initialize blocks according to given weights by identifying zero blocks
        else:
            self._symmetry = None

            blocks = []
            self._weights = []

            # loop over codomain spaces (rows)
            for a, wspace in enumerate(Wspaces):
                blocks += [[]]
                self._weights += [[]]

                # quadrature points
                pts = [points.flatten() for points in derham.spline_attributes[W_name].quad_grid_pts[a]]

                # spline functions as weights: prepare for assembly on quad grid
                grid_shape = tuple([len(pt) for pt in pts])
                self.spline_values = {}
                self.spans = {}
                self.bases = {}
                for name, spline in self.spline_functions.items():
                    assert isinstance(spline, SplineFunction), (
                        f"The entry {name} in spline_functions must be a SplineFunction object."
                    )
                    self.spline_values[name] = xp.zeros(grid_shape, dtype=float)
                    self.spans[name], bns, bds = derham.prepare_eval_tp_fixed(pts)
                    if spline.space_id == "H1":
                        self.bases[name] = bns
                    elif spline.space_id == "L2":
                        self.bases[name] = bds
                    else:
                        raise NotImplementedError(
                            f"Spline functions in spline_functions must be defined on H1 or L2 spaces, but {spline.space_id} was given for the spline function {name}.",
                        )

                # loop over domain spaces (columns)
                for b, vspace in enumerate(Vspaces):
                    # set zero default weights if weights is None
                    if weights_info is None:
                        if self._matrix_free:
                            blocks[-1] += [
                                StencilMatrixFreeMassOperator(self.derham, vspace, wspace, self.nquads),
                            ]
                        else:
                            blocks[-1] += [
                                StencilMatrix(
                                    vspace.coeff_space,
                                    wspace.coeff_space,
                                    backend=PSYDAC_BACKEND_GPYCCEL,
                                    precompiled=True,
                                ),
                            ]
                        self._weights[-1] += [lambda *etas: 0 * etas[0]]

                    else:
                        if weights_info[a][b] is None:
                            blocks[-1] += [None]
                            self._weights[-1] += [None]

                        else:
                            if callable(weights_info[a][b]):
                                PTS = xp.meshgrid(*pts, indexing="ij")
                                mat_w = weights_info[a][b](*PTS).copy()
                            elif isinstance(weights_info[a][b], xp.ndarray):
                                mat_w = weights_info[a][b]

                            logger.debug(f"{mat_w.shape = } and {[pt.size for pt in pts] = }.")
                            assert mat_w.shape == tuple(
                                [pt.size for pt in pts],
                            )

                            if xp.any(xp.abs(mat_w) > 1e-14):
                                if self._matrix_free:
                                    blocks[-1] += [
                                        StencilMatrixFreeMassOperator(
                                            self.derham,
                                            vspace,
                                            wspace,
                                            weights=weights_info[a][b],
                                            nquads=self.nquads,
                                        ),
                                    ]
                                else:
                                    blocks[-1] += [
                                        StencilMatrix(
                                            vspace.coeff_space,
                                            wspace.coeff_space,
                                            backend=PSYDAC_BACKEND_GPYCCEL,
                                            precompiled=True,
                                        ),
                                    ]
                                self._weights[-1] += [weights_info[a][b]]
                            else:
                                blocks[-1] += [None]
                                self._weights[-1] += [None]

            if len(blocks) == len(blocks[0]) == 1:
                if blocks[0][0] is None:
                    if self._matrix_free:
                        self._mat = StencilMatrixFreeMassOperator(
                            self.derham,
                            vspace,
                            wspace,
                            nquads=self.nquads,
                        )
                    else:
                        self._mat = StencilMatrix(
                            vspace.coeff_space,
                            wspace.coeff_space,
                            backend=PSYDAC_BACKEND_GPYCCEL,
                            precompiled=True,
                        )
                else:
                    self._mat = blocks[0][0]
            else:
                self._mat = BlockLinearOperator(
                    V.coeff_space,
                    W.coeff_space,
                    blocks=blocks,
                )

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

        self._W_extraction_op_T = self._W_extraction_op.T
        self._W_boundary_op_T = self._W_boundary_op.T
        self._V_extraction_op_T = self._V_extraction_op.T
        self._V_boundary_op_T = self._V_boundary_op.T

        # TODO: maybe remove since this is done in the .dot() explicitly
        # build composite linear operators BW * EW * M * EV^T * BV^T, resp. IDV * EV * M^T * EW^T * IDW^T
        if self._transposed:
            self._M = self._V_extraction_op @ self._mat @ self._W_extraction_op_T
            self._M0 = self._V_boundary_op @ self._M @ self._W_boundary_op_T
        else:
            self._M = self._W_extraction_op @ self._mat @ self._V_extraction_op_T
            self._M0 = self._W_boundary_op @ self._M @ self._V_boundary_op_T

        # set domain and codomain
        self._domain = self._M.domain
        self._codomain = self._M.codomain

        # allocate temporaries for .dot()
        self._temp_WB = self._W_boundary_op.domain.zeros()
        self._temp_WE = self._W_extraction_op.domain.zeros()
        self._temp_VB = self._V_boundary_op.domain.zeros()
        self._temp_VE = self._V_extraction_op.domain.zeros()
        self._temp_mat = self._mat.domain.zeros()

        # load assembly kernel
        if not self._matrix_free:
            self._assembly_kernel = Pyccelkernel(
                getattr(
                    mass_kernels,
                    "kernel_" + str(self._V.ldim) + "d_mat",
                ),
            )

    @property
    def derham(self):
        return self._derham

    @property
    def domain(self):
        return self._domain

    @property
    def domain_symbolic_name(self) -> LiteralOptions.OptsFEECSpace:
        return self._domain_symbolic_name

    @property
    def codomain(self):
        return self._codomain

    @property
    def codomain_symbolic_name(self) -> LiteralOptions.OptsFEECSpace:
        return self._codomain_symbolic_name

    @property
    def name(self):
        return self._name

    @property
    def domain_femspace(self):
        return self._domain_femspace

    @property
    def codomain_femspace(self):
        return self._codomain_femspace

    @property
    def spline_functions(self):
        return self._spline_functions

    @property
    def dtype(self):
        return self._dtype

    def tosparse(self):
        if all(op is None for op in (self._W_extraction_op, self._V_extraction_op)):
            for bl in self._V_boundary_op.bc:
                for bc in bl:
                    assert not bc, logger.info(".tosparse() only works without boundary conditions at the moment")
            for bl in self._W_boundary_op.bc:
                for bc in bl:
                    assert not bc, logger.info(".tosparse() only works without boundary conditions at the moment")

            return self._mat.tosparse()
        elif all(isinstance(op, IdentityOperator) for op in (self._W_extraction_op, self._V_extraction_op)):
            return self._mat.tosparse()
        else:
            raise NotImplementedError()

    def toarray(self):
        if all(op is None for op in (self._W_extraction_op, self._V_extraction_op)):
            for bl in self._V_boundary_op.bc:
                for bc in bl:
                    assert not bc, logger.info(".toarray() only works without boundary conditions at the moment")
            for bl in self._W_boundary_op.bc:
                for bc in bl:
                    assert not bc, logger.info(".toarray() only works without boundary conditions at the moment")

            return self._mat.toarray()
        elif all(isinstance(op, IdentityOperator) for op in (self._W_extraction_op, self._V_extraction_op)):
            return self._mat.toarray()
        else:
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
    def nquads(self):
        if self._nquads is None:
            return self.derham.nquads
        else:
            return self._nquads

    @property
    def symmetry(self):
        return self._symmetry

    @property
    def weights(self):
        return self._weights

    def dot(self, v, out=None, apply_bc=True):
        """Dot product of the operator with a vector.

        Parameters
        ----------
        v : feectools.linalg.basic.Vector
            The input (domain) vector.

        out : feectools.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        apply_bc : bool
            Whether to apply the boundary operators (True) or not (False).

        Returns
        -------
        out : feectools.linalg.basic.Vector
            The output (codomain) vector.
        """

        assert isinstance(v, Vector)
        assert v.space == self.domain

        # newly created output vector
        if out is None:
            out = self.codomain.zeros()
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain

        if apply_bc:
            if self._transposed:
                self._W_boundary_op_T.dot(v, out=self._temp_WB)
                self._W_extraction_op_T.dot(self._temp_WB, out=self._temp_mat)
                self._mat.dot(self._temp_mat, out=self._temp_VE)
                self._V_extraction_op.dot(self._temp_VE, out=self._temp_VB)
                out = self._V_boundary_op.dot(self._temp_VB, out=out)
            else:
                self._V_boundary_op_T.dot(v, out=self._temp_VB)
                self._V_extraction_op_T.dot(self._temp_VB, out=self._temp_mat)
                self._mat.dot(self._temp_mat, out=self._temp_WE)
                self._W_extraction_op.dot(self._temp_WE, out=self._temp_WB)
                out = self._W_boundary_op.dot(self._temp_WB, out=out)
        else:
            if self._transposed:
                self._W_extraction_op_T.dot(v, out=self._temp_mat)
                self._mat.dot(self._temp_mat, out=self._temp_VE)
                out = self._V_extraction_op.dot(self._temp_VE, out=out)
            else:
                self._V_extraction_op_T.dot(v, out=self._temp_mat)
                self._mat.dot(self._temp_mat, out=self._temp_WE)
                out = self._W_extraction_op.dot(self._temp_WE, out=out)

        return out

    def transpose(self, conjugate=False):
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
            M = WeightedMassOperator(
                self.derham,
                self._V,
                self._W,
                name=self.name + "T",
                V_extraction_op=self._V_extraction_op,
                W_extraction_op=self._W_extraction_op,
                V_boundary_op=self._V_boundary_op,
                W_boundary_op=self._W_boundary_op,
                weights_info=weights,
                transposed=not self._transposed,
                matrix_free=self._matrix_free,
            )

            M.assemble()

        else:
            M = WeightedMassOperator(
                self.derham,
                self._V,
                self._W,
                name=self.name + "T",
                V_extraction_op=self._V_extraction_op,
                W_extraction_op=self._W_extraction_op,
                V_boundary_op=self._V_boundary_op,
                W_boundary_op=self._W_boundary_op,
                weights_info=self._symmetry,
                transposed=not self._transposed,
                matrix_free=self._matrix_free,
            )

            M.assemble(weights=weights)

        return M

    def assemble(self, weights=None, clear=True):
        r"""
        Assembles the weighted mass matrix, i.e. computes the integrals

        .. math::

            \mathbb M^{\beta \alpha}_{(\mu,ijk),(\nu,mno)} = \int_{[0, 1]^3} \Lambda^\beta_{\mu,ijk} \, A_{\mu,\nu} \, \Lambda^\alpha_{\nu,mno} \, \textnormal d^3 \boldsymbol\eta\,.

        The integration is performed with Gauss-Legendre quadrature over the logical domain.

        Parameters
        ----------
        weights : list | NoneType
            Weight function(s) (callables or xp.ndarrays) in a 2d list of shape corresponding to
            number of components of domain/codomain.
            If ``weights=None``, the weight is taken from the given weights in the
            instantiation of the object, else it will be overriden.

        clear : bool
            Whether to first set all data to zero before assembly. If False,
            the new contributions are added to existing ones.
        """

        if self._matrix_free:
            if weights is not None:
                if self._is_scalar:
                    self._mat.weights = weights[0][0]
                else:
                    for a, weights_row in enumerate(weights):
                        for b, weight in enumerate(weights_row):
                            if weight is not None:
                                assert callable(weight) or isinstance(
                                    weight,
                                    xp.ndarray,
                                )
                            self._mat[a, b].weights = weight

                self._weights = weights

        else:
            # clear data
            if clear:
                if isinstance(self._mat, StencilMatrix):
                    self._mat._data[:] = 0.0
                else:
                    for block_row in self._mat.blocks:
                        for block in block_row:
                            if block is not None:
                                block._data[:] = 0.0

            logger.debug(
                f'\nAssembling matrix of WeightedMassOperator "{self.name}" with V={self._domain_symbolic_name}, W={self._codomain_symbolic_name}.',
            )

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
                self._weights = weights

            V_name = self.domain_symbolic_name
            W_name = self.codomain_symbolic_name
            spline_attr = self.derham.spline_attributes

            # loop over codomain spaces (rows)
            for a, codomain_space in enumerate(codomain_spaces):
                # knot span indices of elements of local domain
                codomain_spans = spline_attr[W_name].quad_grid_spans[a]

                # global start spline index on process
                codomain_starts = [int(start) for start in codomain_space.coeff_space.starts]

                # pads (ghost regions)
                codomain_pads = codomain_space.coeff_space.pads

                # quadrature points
                pts = [points.flatten() for points in spline_attr[W_name].quad_grid_pts[a]]

                # global quadrature weights in format (local element, local weight)
                wts = spline_attr[W_name].quad_grid_wts[a]

                # evaluated basis functions at quadrature points of codomain space
                codomain_basis = spline_attr[W_name].quad_grid_bases[a]

                # loop over domain spaces (columns)
                for b, domain_space in enumerate(domain_spaces):
                    # skip None and redundant blocks (lower half for symmetric and anti-symmetric)
                    if not self._is_scalar:
                        if self._symmetry is not None and a > b:
                            continue

                    loc_weight = self._weights[a][b]

                    # evaluate weight at quadrature points
                    if callable(loc_weight):
                        PTS = xp.meshgrid(*pts, indexing="ij")
                        mat_w = loc_weight(*PTS).copy()
                    elif isinstance(loc_weight, xp.ndarray):
                        mat_w = loc_weight
                    elif loc_weight is not None:
                        raise TypeError(
                            "weights must be callable or xp.ndarray or None but is {}".format(
                                type(self._weights[a][b]),
                            ),
                        )

                    if loc_weight is not None:
                        assert mat_w.shape == tuple([pt.size for pt in pts])
                        # evalute splines and multiply
                        for name, spline in self.spline_functions.items():
                            logger.debug(
                                f"Maximum coefficient of spline {name}: {xp.max(xp.abs(spline.vector.toarray()))}"
                            )
                            values = spline.eval_tp_fixed_loc(
                                self.spans[name],
                                self.bases[name],
                                out=self.spline_values[name],
                            )
                            if xp.all(xp.abs(values) < 1e-14):
                                logger.warning(
                                    f"The spline weight {name} is close to zero at all quadrature points in the assembly of the weighted mass matrix {self.name}. Weights are not multiplied."
                                )
                                continue
                            mat_w *= values
                    else:
                        logger.debug(f"No weight for block {a, b}, setting mat_w to None.")
                        mat_w = None

                    not_weight_zero = xp.array(
                        int(loc_weight is not None and xp.any(xp.abs(mat_w) > 1e-14)),
                    )
                    if self.derham.comm is not None:
                        self.derham.comm.Allreduce(
                            MPI.IN_PLACE,
                            not_weight_zero,
                            op=MPI.LOR,
                        )

                    # evaluated basis functions at quadrature points of domain space
                    domain_basis = spline_attr[V_name].quad_grid_bases[b]

                    # assemble matrix (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)
                    if not_weight_zero or self._is_scalar:
                        # get cell of block matrix (don't instantiate if all zeros)
                        if self._is_scalar:
                            mat = self._mat
                            if loc_weight is None:
                                # in case it's none we still need to have zeros weights to call the kernel
                                mat_w = xp.zeros(
                                    tuple([pt.size for pt in pts]),
                                )
                        else:
                            mat = self._mat[a, b]

                        if mat is None:
                            # Maybe in a previous iteration we had more zeros
                            # Can only happen in the Block case
                            self._mat[a, b] = StencilMatrix(
                                domain_space.coeff_space,
                                codomain_space.coeff_space,
                                backend=PSYDAC_BACKEND_GPYCCEL,
                                precompiled=True,
                            )
                            mat = self._mat[a, b]

                        logger.debug(f"Assemble block {a, b}")

                        self._assembly_kernel(
                            *codomain_spans,
                            *codomain_space.degree,
                            *domain_space.degree,
                            *codomain_starts,
                            *codomain_pads,
                            *wts,
                            *codomain_basis,
                            *domain_basis,
                            mat_w,
                            mat._data,
                        )

                    else:
                        if clear:
                            self._mat[a, b] = None
                        else:
                            continue

            # exchange assembly data (accumulate ghost regions)
            self._mat.exchange_assembly_data()

            # copy data for symmetric/anti-symmetric block matrices
            if self.symmetry == "symm":
                self._mat.update_ghost_regions()

                self._mat[1, 0]._data[:] = self._mat[0, 1].T._data
                self._mat[2, 0]._data[:] = self._mat[0, 2].T._data
                self._mat[2, 1]._data[:] = self._mat[1, 2].T._data

            elif self.symmetry == "asym":
                self._mat.update_ghost_regions()

                self._mat[1, 0]._data[:] = -self._mat[0, 1].T._data
                self._mat[2, 0]._data[:] = -self._mat[0, 2].T._data
                self._mat[2, 1]._data[:] = -self._mat[1, 2].T._data

            logger.debug("Done.")

    def copy(self, out=None):
        """Create a copy of self, that can potentially be stored in a given WeightedMassOperator.

        Parameters
        ----------
        out : WeightedMassOperator(optional)
            The existing WeightedMassOperator in which we want to copy self.
        """
        if out is not None:
            assert isinstance(out, WeightedMassOperator)
            assert out.domain is self.domain
            assert out.codomain is self.codomain
        else:
            out = WeightedMassOperator(
                self.derham,
                V=self._V,
                W=self._W,
                V_extraction_op=self._V_extraction_op,
                W_extraction_op=self._W_extraction_op,
                V_boundary_op=self._V_boundary_op,
                W_boundary_op=self._W_boundary_op,
                weights_info=self._weights_info,
                transposed=self._transposed,
                matrix_free=self._matrix_free,
            )

        self._mat.copy(out=out._mat)
        return out

    def __imul__(self, a):
        self._mat *= a
        return self

    def __iadd__(self, M):
        assert M.domain is self.domain
        assert M.codomain is self.codomain

        if isinstance(M, WeightedMassOperator):
            self._mat += M._mat
            return self

        elif isinstance(M, LinearOperator):
            self._mat += M
            return self

        else:
            return LinearOperator.__add__(self, M)

    def __isub__(self, M):
        assert M.domain is self.domain
        assert M.codomain is self.codomain

        if isinstance(M, WeightedMassOperator):
            self._mat -= M._mat
            return self

        elif isinstance(M, LinearOperator):
            self._mat -= M
            return self

        else:
            return LinOpWithTransp.__sub__(self, M)

    def eval_quad(self, W, coeffs, out=None):
        """
        Evaluates a given FEM field defined by its coefficients at the L2 quadrature points.

        Parameters
        ----------
        W : TensorFemSpace | VectorFemSpace
            Tensor product spline space from feectools.fem.tensor.

        coeffs : StencilVector | BlockVector
            The coefficient vector corresponding to the FEM field. Ghost regions must be up-to-date!

        out : xp.ndarray | list/tuple of xp.ndarrays, optional
            If given, the result will be written into these arrays in-place. Number of outs must be compatible with number of components of FEM field.

        Returns
        -------
        out : xp.ndarray | list/tuple of xp.ndarrays
            The values of the FEM field at the quadrature points.
        """

        assert isinstance(W, (TensorFemSpace, VectorFemSpace))
        assert isinstance(coeffs, (StencilVector, BlockVector))
        assert W.coeff_space == coeffs.space

        # collect TensorFemSpaces for each component in tuple
        if isinstance(W, TensorFemSpace):
            Wspaces = (W,)
        else:
            Wspaces = W.spaces

        # prepare output
        if out is None:
            out = ()
            if isinstance(W, TensorFemSpace):
                out += (
                    xp.zeros(
                        [
                            q_grid[nquad].points.size
                            for q_grid, nquad in zip(get_quad_grids(W, nquads=self.nquads), self.nquads)
                        ],
                        dtype=float,
                    ),
                )
            else:
                for space in W.spaces:
                    out += (
                        xp.zeros(
                            [
                                q_grid[nquad].points.size
                                for q_grid, nquad in zip(
                                    get_quad_grids(space, nquads=self.nquads),
                                    self.nquads,
                                )
                            ],
                            dtype=float,
                        ),
                    )

        else:
            if isinstance(W, TensorFemSpace):
                assert isinstance(out, xp.ndarray)
                out = (out,)
            else:
                assert isinstance(out, (list, tuple))

        # load assembly kernel
        kernel = Pyccelkernel(getattr(mass_kernels, "kernel_" + str(W.ldim) + "d_eval"))

        # loop over components
        for a, wspace in enumerate(Wspaces):
            # knot span indices of elements of local domain
            spans = [
                quad_grid[nquad].spans
                for quad_grid, nquad in zip(get_quad_grids(wspace, nquads=self.nquads), self.nquads)
            ]

            # global start spline index on process
            starts = [int(start) for start in wspace.coeff_space.starts]

            # pads (ghost regions)
            pads = wspace.coeff_space.pads

            # global quadrature points (flattened) and weights in format (local element, local weight)
            pts = [
                quad_grid[nquad].points.flatten()
                for quad_grid, nquad in zip(get_quad_grids(wspace, nquads=self.nquads), self.nquads)
            ]
            wts = [
                quad_grid[nquad].weights
                for quad_grid, nquad in zip(get_quad_grids(wspace, nquads=self.nquads), self.nquads)
            ]

            # evaluated basis functions at quadrature points of codomain space
            basis = [
                quad_grid[nquad].basis
                for quad_grid, nquad in zip(get_quad_grids(wspace, nquads=self.nquads), self.nquads)
            ]

            if isinstance(coeffs, StencilVector):
                kernel(
                    *spans,
                    *wspace.degree,
                    *starts,
                    *pads,
                    *basis,
                    coeffs._data,
                    out[a],
                )
            else:
                kernel(
                    *spans,
                    *wspace.degree,
                    *starts,
                    *pads,
                    *basis,
                    coeffs[a]._data,
                    out[a],
                )

        if len(out) == 1:
            return out[0]
        else:
            return out

    def info(self, use_rst=False):
        return info(self, use_rst=use_rst)


class StencilMatrixFreeMassOperator(LinOpWithTransp):
    r"""Class implementing matrix-free weighted mass operators between StencilVectorSpaces.

    The result of the dot product with a spline function :math:`S_h` is computed as

    .. math::

        w^\mu_{ijk} = \int \Lambda_{\mu,ijk}\, S_h\, w(\boldsymbol\eta)\,\textrm d \boldsymbol \eta \,,

    where :math:`w(\boldsymbol\eta)` is a weight function (including the geometric weights).

    Should only be instanciated via `WeightedMassOperator`, where it's used to replace `StencilMatrix` when one does not want to assemble the matrix for cost reasons

    Parameters
    ----------
    V : TensorFemSpace
        Domain of the mass operator

    W : TensorFemSpace
        Codomain of the mass operator

    weights : callable | numpy.ndarry | None
        The weights of the mass operator
    """

    def __init__(self, derham, V, W, weights=None, nquads=None):
        self._V = V
        self._W = W
        self._domain = V.coeff_space
        self._codomain = W.coeff_space
        self._weights = weights

        self._derham = derham
        self._nquads = nquads

        self._dtype = V.coeff_space.dtype
        self._dot_kernel = Pyccelkernel(
            getattr(
                mass_kernels,
                "kernel_" + str(self._V.ldim) + "d_matrixfree",
            ),
        )

        self._diag_kernel = Pyccelkernel(
            getattr(
                mass_kernels,
                "kernel_" + str(self._V.ldim) + "d_diag",
            ),
        )

        shape = tuple(e - s + 1 for s, e in zip(V.coeff_space.starts, V.coeff_space.ends))
        self._diag_tmp = xp.zeros((shape))

        # knot span indices of elements of local domain
        self._codomain_spans = [
            quad_grid[nquad].spans for quad_grid, nquad in zip(get_quad_grids(self._W, nquads=self.nquads), self.nquads)
        ]

        # global start spline index on process
        self._codomain_starts = [int(start) for start in self._W.coeff_space.starts]
        # pads (ghost regions)
        self._codomain_pads = self._W.coeff_space.pads

        # evaluated basis functions at quadrature points of codomain space
        self._codomain_basis = [
            quad_grid[nquad].basis for quad_grid, nquad in zip(get_quad_grids(self._W, nquads=self.nquads), self.nquads)
        ]

        # knot span indices of elements of local domain
        self._domain_spans = [
            quad_grid[nquad].spans for quad_grid, nquad in zip(get_quad_grids(self._V, nquads=self.nquads), self.nquads)
        ]

        # global start spline index on process
        self._domain_starts = [int(start) for start in self._V.coeff_space.starts]

        # pads (ghost regions)
        self._domain_pads = self._V.coeff_space.pads

        # evaluated basis functions at quadrature points of domain space
        self._domain_basis = [
            quad_grid[nquad].basis for quad_grid, nquad in zip(get_quad_grids(self._V, nquads=self.nquads), self.nquads)
        ]

        # global quadrature points (flattened) and weights in format (local element, local weight)
        self._pts = [
            quad_grid[nquad].points.flatten()
            for quad_grid, nquad in zip(get_quad_grids(self._W, nquads=self.nquads), self.nquads)
        ]
        self._wts = [
            quad_grid[nquad].weights
            for quad_grid, nquad in zip(
                get_quad_grids(self._W, nquads=self.nquads),
                self.nquads,
            )
        ]

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def nquads(self):
        if self._nquads is None:
            return self.derham.nquads
        else:
            return self._nquads

    @property
    def derham(self):
        """Discrete de Rham sequence on the logical unit cube."""
        return self._derham

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def transpose(self, conjugate=False):
        return StencilMatrixFreeMassOperator(
            self._derham,
            self._codomain,
            self._domain,
            self._weights,
            nquads=self._nquads,
        )

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new):
        self._weights = new

    def dot(self, v, out=None):
        """
        Dot product of the operator with a vector. Direct computation (not using a StencilMatrix).

        Parameters
        ----------
        v : feectools.linalg.basic.Vector
            The input (domain) vector.

        out : feectools.linalg.basic.Vector, optional
            If given, the output will be written in-place into this vector.

        apply_bc : bool
            Whether to apply the boundary operators (True) or not (False).

        Returns
        -------
        out : feectools.linalg.basic.Vector
            The output (codomain) vector.
        """

        if out is None:
            out = self.codomain.zeros()
        else:
            assert isinstance(out, Vector)
            assert out.space == self.codomain
            out._data[:] = 0.0

        v.update_ghost_regions()

        # evaluate weight at quadrature points
        if callable(self._weights):
            PTS = xp.meshgrid(*self._pts, indexing="ij")
            mat_w = self._weights(*PTS).copy()
        elif isinstance(self._weights, xp.ndarray):
            mat_w = self._weights

        if self._weights is not None:
            assert mat_w.shape == tuple([pt.size for pt in self._pts])

            # call kernel (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)
            if xp.any(xp.abs(mat_w) > 1e-14):
                self._dot_kernel(
                    *self._codomain_spans,
                    *self._domain_spans,
                    *self._W.degree,
                    *self._V.degree,
                    *self._codomain_starts,
                    *self._domain_starts,
                    *self._codomain_pads,
                    *self._domain_pads,
                    *self._wts,
                    *self._codomain_basis,
                    *self._domain_basis,
                    mat_w,
                    out._data,
                    v._data,
                )

            out.exchange_assembly_data()
        return out

    def diagonal(self, inverse=False, sqrt=False, out=None):
        """
        Get the coefficients on the main diagonal as a StencilDiagonalMatrix object.

        Parameters
        ----------
        inverse : bool
            If True, get the inverse of the diagonal. (Default: False).

        sqrt : bool
            If True, get the square root of the diagonal. (Default: False).
            Can be combined with inverse to get the inverse square root

        out : StencilDiagonalMatrix
            If provided, write the diagonal entries into this matrix. (Default: None).

        Returns
        -------
        StencilDiagonalMatrix
            The matrix which contains the main diagonal of self (or its inverse).

        """
        # Check `inverse` argument
        assert isinstance(inverse, bool)

        # Only if domain == codomain
        assert self.domain == self.codomain

        # Determine domain and codomain of the StencilDiagonalMatrix
        V, W = self.domain, self.codomain

        # Check `out` argument
        if out is not None:
            assert isinstance(out, StencilDiagonalMatrix)
            assert out.domain is V
            assert out.codomain is W

        # evaluate weight at quadrature points
        if callable(self._weights):
            PTS = xp.meshgrid(*self._pts, indexing="ij")
            mat_w = self._weights(*PTS).copy()
        elif isinstance(self._weights, xp.ndarray):
            mat_w = self._weights

        diag = self._diag_tmp
        diag[:] = 0.0
        self._diag_kernel(
            *self._codomain_spans,
            *self._W.degree,
            *self._codomain_starts,
            *self._codomain_pads,
            *self._wts,
            *self._codomain_basis,
            mat_w,
            diag,
        )

        data = out._data if out else None

        # Calculate entries of StencilDiagonalMatrix
        if sqrt:
            diag = xp.sqrt(diag)

        if inverse:
            data = xp.divide(1, diag, out=data)
        elif out:
            xp.copyto(data, diag)
        else:
            data = diag.copy()

        # If needed create a new StencilDiagonalMatrix object
        if out is None:
            out = StencilDiagonalMatrix(V, W, data)

        return out


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

    solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by implicit or explicit operators.

    precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
        Preconditioner for the mass-matrix block.

    solver_params : SolverParameters, default=None
            Solver controls; defaults to ``SolverParameters()``.
    """

    def __init__(
        self,
        space_id: str,
        mass_ops: WeightedMassOperators,
        solver_name: LiteralOptions.OptsSymmSolver = "pcg",
        precond_name: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner",
        solver_params: SolverParameters = None,
    ):
        assert space_id in ("H1", "Hcurl", "Hdiv", "L2", "H1vec")

        # TODO: enable serialization of WeightedMassOperators
        # self.params = copy.deepcopy(locals())

        # TODO: move L2projector to its own file and avoid circular imports
        from struphy.feec import preconditioner

        if solver_params is None:
            solver_params = SolverParameters()

        self._space_id = space_id
        self._mass_ops = mass_ops
        self._space_key = mass_ops.derham.space_to_form[self.space_id]
        self._space = mass_ops.derham.fem_spaces[self.space_key]

        # mass matrix
        self._Mmat = getattr(self.mass_ops, "M" + self.space_key)

        # quadrature grid
        self._quad_grid_pts = self.mass_ops.derham.spline_attributes[self.space_key].quad_grid_pts

        if space_id in ("H1", "L2"):
            self._quad_grid_mesh = xp.meshgrid(
                *[pt.flatten() for pt in self.quad_grid_pts[0]],
                indexing="ij",
            )
            self._geom_weights = self.Mmat.weights[0][0]  # (*self.quad_grid_mesh)
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
                        self._geom_weights[-1] += [weight]  # (*mesh)]
                    else:
                        self._geom_weights[-1] += [xp.zeros_like(mesh[0])]

        # other quad grid info
        self._tensor_fem_spaces = self.mass_ops.derham.spline_attributes[self.space_key].tensor_spaces
        self._wts_l = self.mass_ops.derham.spline_attributes[self.space_key].quad_grid_wts
        self._spans_l = self.mass_ops.derham.spline_attributes[self.space_key].quad_grid_spans
        self._bases_l = self.mass_ops.derham.spline_attributes[self.space_key].quad_grid_bases

        # Preconditioner
        if precond_name is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, precond_name)
            pc = pc_class(self.Mmat)

        # solver
        self._solver = inverse(
            self.Mmat,
            solver_name,
            pc=pc,
            tol=solver_params.tol,
            maxiter=solver_params.maxiter,
            verbose=solver_params.verbose,
        )

    @property
    def params(self) -> dict:
        """Parameters passed to __init__(), as dictionary."""
        if not hasattr(self, "_params"):
            self._params = {}
        return self._params

    @params.setter
    def params(self, new):
        assert isinstance(new, dict)
        if "self" in new:
            new.pop("self")
        if "__class__" in new:
            new.pop("__class__")
        self._params = new

    @property
    def mass_ops(self) -> WeightedMassOperators:
        """Struphy mass operators object, see :ref:`mass_ops`.."""
        return self._mass_ops

    @property
    def space_id(self) -> str:
        """The ID of the space (H1, Hcurl, Hdiv, L2 or H1vec)."""
        return self._space_id

    @property
    def space_key(self) -> str:
        """The key of the space (0, 1, 2, 3 or v)."""
        return self._space_key

    @property
    def space(self) -> FemSpace:
        """The Derham finite element space (from ``Derham.fem_spaces``)."""
        return self._space

    @property
    def solver(self) -> InverseLinearOperator:
        """The iterative solver for the mass matrix."""
        return self._solver

    @property
    def Mmat(self) -> WeightedMassOperator:
        """The mass matrix of space."""
        return self._Mmat

    @property
    def quad_grid_pts(self) -> tuple[tuple[xp.ndarray]]:
        """List of quadrature points in each direction for integration over grid cells in format (ni, nq) = (cell, quadrature point)."""
        return self._quad_grid_pts

    @property
    def quad_grid_mesh(self) -> list[tuple[xp.ndarray]]:
        """Mesh grids of quad_grid_pts."""
        return self._quad_grid_mesh

    @property
    def geom_weights(self) -> list[list[xp.ndarray]]:
        """Geometric coefficients (e.g. Jacobians) evaluated at quad_grid_mesh, stored as list[list] either 1x1 or 3x3."""
        return self._geom_weights

    def __repr__(self):
        out = f"{self.__class__.__name__}(\n"
        for k, v in self.params.items():
            out += " " * 4
            out += f"{k}={v},\n"
        out += ")"
        return out

    def __repr_no_defaults__(self):
        return __class_with_params_repr_no_defaults__(self)

    def solve(self, rhs: StencilVector | BlockVector, out=None) -> StencilVector | BlockVector:
        """
        Solves the linear system M * x = rhs, where M is the mass matrix.

        Parameters
        ----------
        rhs : feectools.linalg.basic.vector
            The right-hand side of the linear system.

        out : feectools.linalg.basic.vector, optional
            If given, the result will be written into this vector in-place.

        Returns
        -------
        out : feectools.linalg.basic.vector
            Output vector (result of linear system).
        """

        assert isinstance(rhs, StencilVector) or isinstance(rhs, BlockVector)
        assert rhs.space == self.Mmat.domain

        if out is None:
            out = self.solver.dot(rhs)
        else:
            self.solver.dot(rhs, out=out)

        return out

    def get_dofs(
        self,
        fun: Callable | xp.ndarray | list[Callable | xp.ndarray] | tuple[Callable | xp.ndarray],
        dofs: StencilVector | BlockVector = None,
        apply_bc: bool = False,
        clear: bool = True,
    ) -> StencilVector | BlockVector:
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
        fun : Callable | xp.ndarray | list[Callable | xp.ndarray] | tuple[Callable | xp.ndarray]
            Weight function(s) (callables or xp.ndarrays) in a 1d list of shape corresponding to number of components.

        dofs : StencilVector | BlockVector, optional
            The vector for the output.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom.

        clear : bool, optional
            Whether to first set all data to zero before assembly. If False, the new contributions are added to existing ones in vec.

        Returns
        -------
        dofs : StencilVector | BlockVector
             The assembled degrees of freedom, before projection.
        """

        # evaluate fun at quad_grid or check array size
        if callable(fun):
            fun_weights = fun(*self.quad_grid_mesh)
        elif isinstance(fun, xp.ndarray):
            assert fun.shape == self.quad_grid_mesh[0].shape, (
                f"Expected shape {self.quad_grid_mesh[0].shape}, got {fun.shape =} instead."
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
            for mesh in self.quad_grid_mesh:
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

    def __call__(
        self,
        fun: Callable | list[Callable] | tuple[Callable],
        out: StencilVector | BlockVector = None,
        dofs: StencilVector | BlockVector = None,
        apply_bc: bool = False,
    ) -> StencilVector | BlockVector:
        """
        Applies projector to given callable(s).

        Parameters
        ----------
        fun : Callable | list[Callable] | tuple[Callable]
            The function to be projected. List of three callables for vector-valued functions.

        out : StencilVector | BlockVector, optional
            If given, the result will be written into this vector in-place.

        dofs : StencilVector | BlockVector, optional
            If given, the dofs will be written into this vector in-place.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom and coefficients.

        Returns
        -------
        coeffs : feectools.linalg.basic.vector
            The FEM spline coefficients after projection.
        """
        return self.solve(self.get_dofs(fun, dofs=dofs, apply_bc=apply_bc), out=out)


class AverageOperator(LinOpWithTransp):
    r"""
    Class for quadrature operators, performs the average of a `FeecVariable` along a given direction.
    For example along the :math:`\eta_3` direction, it applies the following linear operator :

    .. math::

        \mathbb M^{\alpha}_{(\mu,ijk),(\nu,mno)} = \delta_{i,m} \delta_{j,n} c_o

    with :math:`c_o=\int_0^1 N_{o}(\eta_3) \textnormal{d} \eta` and :math:`N_{o}` the B-spline function at the place `o`.
    In other words, it maps a spline function :math:`S_h` to the function obtained by averaging :math:`S_h` along the given direction, i.e. for the example of direction 3:

    .. math::

        S_h(\eta_1, \eta_2, \eta_3) = \sum_{mno} c_{mno} N_m(\eta_1) N_n(\eta_2) N_o(\eta_3) \quad \mapsto \quad \overline S_h(\eta_1, \eta_2, \eta_3) = \sum_{ijk} \overline c_{ij} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,,

    with

    .. math::

        \overline c_{ij} = \sum_o \delta_{i,m} \delta_{j,n} \, c_{mno} \int_0^1 N_{o}(\eta_3) \textnormal{d} \eta_3 .

    Parameters
    ----------
    derham : Derham
        The derham complexe that supports the space used

    space : str
        Identifier of the space on which the average is performed, either `"H1"`, `"Hcurl"`, `"Hdiv"`, `"L2"`, `"H1vec"`

    direction : int, optional
        The direction of the space along which the average is performed, either `0`, `1` or `2`.

    transposed : bool, optional
        Whether to take the transpose of the operator.
    """

    def __init__(
        self,
        derham: Derham,
        space: str = "H1",
        direction: int = 2,
        transposed: bool = False,
    ):

        if space not in derham.space_to_form:
            AssertionError("Must match a space of the derham complex")
        if space != "H1":
            NotImplementedError()
        space_id = "V" + derham.space_to_form[space]
        self._V = getattr(derham, space_id)  # StencilVectorSpace
        self._domain = getattr(derham, space_id)
        self._codomain = getattr(derham, space_id)
        self._pads = self._V.pads  # gets the number of ghost cells
        self._derham = derham
        self._dtype = self._domain.dtype
        self._transposed = transposed
        if direction == 0:
            self._directions = (0, 1, 2)
        elif direction == 1:
            self._directions = (1, 0, 2)
        elif direction == 2:
            self._directions = (2, 0, 1)
        else:
            raise ValueError("invalid direction id, must be 0, 1 or 2")

        comm = derham.comm
        # Selection of ranks for each subcomms regarding their position in the two perpendicular directions to the averaged direction.
        if not isinstance(comm, (MockComm, type(None))):
            rank = comm.Get_rank()
            nprocs = derham.domain_decomposition.nprocs
            dom_arr = derham.domain_array
            color1 = int(dom_arr[rank, 3 * self._directions[1]] * nprocs[self._directions[1]])
            color2 = int(dom_arr[rank, 3 * self._directions[2]] * nprocs[self._directions[2]])
            color = color1 * nprocs[self._directions[2]] + color2
            logger.debug(f"{dom_arr = }")
            self.subcomm = comm.Split(color=color, key=rank)

        # We allocate memory for the 2D temporary array for each process
        self._tmp = xp.zeros(
            (
                int(self._V.ends[self._directions[1]] - self._V.starts[self._directions[1]] + 1),
                int(self._V.ends[self._directions[2]] - self._V.starts[self._directions[2]] + 1),
            )
        )

        # We allocate memory for the weights (integrals of 1D B-splines)
        self._weights = xp.zeros(int(self._V.ends[self._directions[0]] - self._V.starts[self._directions[0]] + 1))

        self.allocate()

        # definition of subscripts for function xp.einsum
        if self._transposed:
            if self._directions[0] == 0:
                self._subscripts = ("ijk->jk", "ij,o->oij")
            if self._directions[0] == 1:
                self._subscripts = ("ijk->ik", "ij,o->ioj")
            if self._directions[0] == 2:
                self._subscripts = ("ijk->ij", "ij,o->ijo")
        else:
            if self._directions[0] == 0:
                self._subscripts = ("ojk,o->jk",)
            if self._directions[0] == 1:
                self._subscripts = ("iok,o->ik",)
            if self._directions[0] == 2:
                self._subscripts = ("ijo,o->ij",)

        # definition of slices
        sl_ghost = tuple(slice(p, -p) if p > 0 else slice(None) for p in self._pads)
        sl_broadcasting = [slice(None), slice(None), slice(None)]
        sl_broadcasting[self._directions[0]] = None
        self._slices = (sl_ghost, tuple(sl_broadcasting))

    def allocate(self):
        """Compute the weights, which are the integrals of 1D B-splines in the averaged direction"""
        knots = getattr(self.derham.args_derham, "tn" + str(self._directions[0] + 1))
        degree = self.derham.degree[self._directions[0]]

        i_begin, i_end = self._V.starts[self._directions[0]], self._V.ends[self._directions[0]] + 1
        if self.derham.bcs[self._directions[0]] is None:
            i_begin += self.derham.degree[self._directions[0]]
            i_end += self.derham.degree[self._directions[0]]
        # General formula for any distribution of knots for the integral of a B-spline function, thus works with periodic and clamped boundary conditions :
        self._weights[:] = (knots[i_begin + degree + 1 : i_end + degree + 1] - knots[i_begin:i_end]) / (degree + 1)

    @property
    def domain(self):
        return self._domain

    @property
    def codomain(self):
        return self._codomain

    @property
    def dtype(self):
        return self._dtype

    @property
    def derham(self):
        return self._derham

    @property
    def nquads(self):
        if self._nquads is None:
            return self.derham.nquads
        else:
            return self._nquads

    @property
    def tosparse(self):
        raise NotImplementedError()

    @property
    def toarray(self):
        raise NotImplementedError()

    def dot(self, v, out=None):

        # assert isinstance(v, StencilVector)
        # assert v.space == self.domain

        v.update_ghost_regions()

        if out is None:
            out = self.codomain.zeros()

        x = v._data[self._slices[0]]
        y = out._data[self._slices[0]]
        if self._transposed:
            xp.einsum(self._subscripts[0], x, out=self._tmp)
            if not isinstance(self.derham.comm, (MockComm, type(None))):
                self.subcomm.Allreduce(MPI.IN_PLACE, self._tmp, MPI.SUM)
            xp.einsum(self._subscripts[1], self._tmp, self._weights, out=y)
        else:
            xp.einsum(self._subscripts[0], x, self._weights, out=self._tmp)
            if not isinstance(self.derham.comm, (MockComm, type(None))):
                self.subcomm.Allreduce(MPI.IN_PLACE, self._tmp, MPI.SUM)
            y[:] = self._tmp[self._slices[1]]

        return out

    def transpose(self, conjugate=False):
        return AverageOperator(self.derham, self.domain, self._weights, transposed=not self._transposed)
