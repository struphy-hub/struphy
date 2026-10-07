import inspect
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

import cunumpy as xp
from cunumpy.kernels import PyccelKernel
from feectools.api.settings import PSYDAC_BACKEND_GPYCCEL
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI
from feectools.fem.tensor import FemSpace, TensorFemSpace
from feectools.fem.vector import VectorFemSpace
from feectools.linalg.basic import IdentityOperator, InverseLinearOperator, LinearOperator, Vector
from feectools.linalg.block import BlockLinearOperator, BlockVector
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilDiagonalMatrix, StencilMatrix, StencilVector
from scope_profiler import ProfileManager

from struphy import equils
from struphy.feec import mass_kernels
from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.feec.utilities import LocalProjectionMatrix, LocalRotationMatrix, get_quad_grids
from struphy.fields_background.base import MHDequilibrium
from struphy.geometry.base import Domain
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.polar.basic import PolarVector
from struphy.polar.linear_operators import PolarExtractionOperator
from struphy.utils.docstring_converter import auto_convert_docstring, info
from struphy.utils.utils import __class_with_params_repr_no_defaults__

logger = logging.getLogger("struphy")

# space identifiers with scalar-valued and vector-valued elements
_SCALAR_SPACES = ("H1", "L2")
_VECTOR_SPACES = ("Hcurl", "Hdiv", "H1vec")


@dataclass
class _ClassifiedWeights:
    """Factors of a 1D weights tuple (Case 3 in :meth:`WeightedMassOperators.create_weighted_mass`), sorted by kind.

    All callables take the evaluation points (e1, e2, e3) on a sparse meshgrid of shape (n1, n2, n3).
    """

    # callables returning shape (n1, n2, n3)
    scalars: list = field(default_factory=list)
    # matrix factors in multiplication order: callables returning shape (n1, n2, n3, 3, 3)
    # or constant arrays of shape (3, 3)
    matrices: list = field(default_factory=list)
    # callable returning shape (n1, n2, n3, 3), for maps from a scalar to a vector space
    col_vector: Callable | None = None
    # callable returning shape (n1, n2, n3, 3), for maps from a vector to a scalar space
    row_vector: Callable | None = None
    # SplineFunction factors, keyed by name; evaluated during assembly in WeightedMassOperator
    spline_functions: dict = field(default_factory=dict)


def _log_weight_stats(label: str, values):
    """Log shape and range of evaluated weights; the reductions are only computed if debug logging is enabled."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Evaluated {label} with {values.shape = }, max = {xp.max(values)}, min = {xp.min(values)}")


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
        self._dry_run = False

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

    @property
    def dry_run(self) -> bool:
        """If True, mass operators created from now on do not allocate (nor assemble) their
        stencil matrices; only their sizes are computed. Set temporarily by :meth:`estimate_mem`."""
        return self._dry_run

    def estimate_mem(
        self,
        names: tuple[str] = ("M0", "M1", "M2", "M3", "Mv"),
        print_report: bool = False,
    ) -> dict[str, int]:
        """Estimate the local (per-MPI-rank) memory footprint of mass matrices, in bytes,
        without allocating them.

        Each requested operator is created exactly as by the corresponding property (same weights,
        hence the same zero-block detection), but with ``dry_run=True``, so that only the sizes of
        its stencil matrices are computed, see
        :attr:`~struphy.feec.mass.WeightedMassOperator.nbytes`. Operators that have already been
        created (and hence allocated) report their actual size instead; dry-run operators are not
        kept in the cache.

        Parameters
        ----------
        names : tuple[str]
            Names of the mass operator properties to estimate, e.g. ``("M0", "M1")``.

        print_report : bool
            Whether to print the breakdown on MPI rank 0.

        Returns
        -------
        dict
            Mapping ``{name: local_bytes}``.
        """
        mem = {}

        cached_before = set(vars(self))
        self._dry_run = True
        try:
            for name in names:
                assert isinstance(getattr(type(self), name, None), property), (
                    f"'{name}' is not a mass operator property of {type(self).__name__}."
                )
                mem[name] = getattr(self, name).nbytes
        finally:
            self._dry_run = False

            # do not keep dry-run (unusable) operators in the cache, including the pieces
            # created for composite operators (e.g. M1 and M1para for M1perp)
            for attr in set(vars(self)) - cached_before:
                delattr(self, attr)

        if print_report and (self.derham.comm is None or self.derham.comm.Get_rank() == 0):
            print("\nESTIMATED MASS MATRIX MEMORY (local, rank 0):")
            for name, nbytes in mem.items():
                print(f"  {name}: {nbytes / 1e6:.2f} MB")

        return mem

    def allocated_mem(self) -> dict[str, int]:
        """Local (per-MPI-rank) memory footprint, in bytes, of the mass matrices that have
        actually been created so far (i.e. those whose property has been accessed)."""
        mem = {}
        for name, method in inspect.getmembers(type(self), predicate=inspect.isdatadescriptor):
            if isinstance(method, property) and hasattr(self, "_" + name):
                op = getattr(self, "_" + name)
                if isinstance(op, WeightedMassOperator):
                    mem[name] = op.nbytes
        return mem

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
    def M3p_inv(self):
        r"""
        Pressure-weighted mass matrix for 3-forms:

        .. math::

            \mathbb M^{3,p^{-1}}_{ijk,mno} =
            \int \Lambda^3_{ijk}\,\Lambda^3_{mno}
            \frac{1}{p_0\sqrt{g}}\,\mathrm d\boldsymbol\eta.

        Here :math:`p_0` is the equilibrium pressure.  This operator is used
        for the quadratic pressure energy of linear MHD perturbations.
        """
        if not hasattr(self, "_M3p_inv"):
            assert self.eq_mhd is not None, "M3p_inv requires an MHD equilibrium with positive pressure."

            def inv_p0(e1, e2, e3):
                return 1.0 / self.eq_mhd.p0(e1, e2, e3)

            self._M3p_inv = self.create_weighted_mass(
                "L2",
                "L2",
                weights=(inv_p0, "1/sqrt_g"),
                name="M3p_inv",
                assemble=True,
            )
        return self._M3p_inv

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
                    lambda *etas: self.eq_mhd.n0(*etas),
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
                    lambda *etas: self.eq_mhd.n0(*etas),
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
                    lambda *etas: self.eq_mhd.n0(*etas),
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
            # The equilibrium field itself, as in M2B: the curl of the projected vector potential (M2B_div0)
            # loses a uniform field in periodic directions, where the potential is a ramp.
            rot_B = LocalRotationMatrix(
                self.eq_mhd.b2_1,
                self.eq_mhd.b2_2,
                self.eq_mhd.b2_3,
            )

            self._M2Bn = self.create_weighted_mass(
                "Hdiv",
                "Hdiv",
                weights=(
                    rot_B,
                    "1/sqrt_g",
                    lambda *etas: 1 / self.eq_mhd.n0(*etas),
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
            self._M1perp = self.M1 - self.M1para
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
                    lambda *etas: self.eq_mhd.n0(*etas) / self.eq_mhd.absB0(*etas) ** 2,
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
                    lambda *etas: self.eq_mhd.n0(*etas) / self.eq_mhd.absB0(*etas) ** 2,
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
            self._M1gyro = self.M1_MHDeq - self.M1para_MHDeq
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
                    lambda *etas: self.eq_mhd.n0(*etas),
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
                    lambda *etas: self.eq_mhd.n0(*etas) / self.eq_mhd.t0(*etas),
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
        dry_run: bool = None,
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

                    - Strings (predefined names) for metric and Jacobian-related weights:
                        ``'G'``, ``'Ginv'``, ``'DFinv'``, ``'DFinvT'``, ``'sqrt_g'``, ``'1/sqrt_g'``, ``'Identity'``.
                    - Callables (including objects such as local rotation matrices) returning
                        either scalar values or ``3x3`` matrix values at quadrature points.
                    - Nested ``3x3`` Python lists (constant matrix entries).
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

        dry_run: bool
            Whether to create the operator without allocating (and assembling) its stencil matrices,
            for memory estimation only. If None (default), the value of the ``dry_run`` attribute of
            this :class:`WeightedMassOperators` object is used, see :meth:`estimate_mem`.

        Returns
        -------
        out : A WeightedMassOperator object.
        """
        if dry_run is None:
            dry_run = self.dry_run

        logger.debug(f"\nCreating weighted mass matrix {name} from {V_id} to {W_id} ({dry_run = }).")

        with ProfileManager.profile_region(f"weights eval for {name}"):
            if isinstance(weights, tuple):
                # Case 3 (1D tuple): evaluate the product of all factors at the quadrature points
                weights_values, spline_functions = self._eval_tuple_weights(weights, V_id, W_id)
            else:
                # Cases 1, 2 and 4 are passed on unchanged to WeightedMassOperator
                logger.debug(f"Processing weights of type {type(weights)}.")
                weights_values, spline_functions = weights, {}

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
            dry_run=dry_run,
        )

        # weights given at quadrature points or as spline functions are bound to this Derham
        grid_bound = len(spline_functions) > 0 or (
            isinstance(weights, list) and any(isinstance(w, xp.ndarray) for row in weights for w in row)
        )
        out._creation_info = (
            None
            if grid_bound
            else {
                "V_id": V_id,
                "W_id": W_id,
                "name": name,
                "weights": weights,
                "transposed": transposed,
                "is_transpose": False,
            }
        )

        if assemble and not dry_run:
            with ProfileManager.profile_region(f"assemble {name}"):
                out.assemble()

        return out

    ##################################################################
    # Evaluation of tuple weights (Case 3 in create_weighted_mass) #
    ##################################################################
    def _eval_tuple_weights(self, weights: tuple, V_id: str, W_id: str):
        """Evaluate a 1D tuple of weights at the quadrature points of the codomain ``W_id``.

        The tuple entries are factors which are multiplied together. They are first sorted
        into scalar, vector and matrix factors (see :meth:`_classify_weights`). On the quadrature
        grid of each component of ``W_id`` (= block row), the matrix factors are multiplied
        together (or the single vector factor is evaluated), and the result is then multiplied
        by the product of all scalar factors. SplineFunction factors are not evaluated here;
        they are passed on to :class:`WeightedMassOperator`, which evaluates them during assembly.

        The factors are evaluated once per distinct quadrature grid: if all components of ``W_id``
        share the same grid (the usual case), the evaluation is reused for all block rows.

        Returns
        -------
        weights_values : list[list[xp.ndarray | None]]
            Weights at the quadrature points, one C-contiguous block of shape (n1, n2, n3) per
            (component of ``W_id``, component of ``V_id``). ``None`` marks a zero block.

        spline_functions : dict
            SplineFunction factors, keyed by their name.
        """
        classified = self._classify_weights(weights, V_id, W_id)
        self._check_weight_compatibility(classified, V_id, W_id)

        # number of block columns = number of components of the domain space V_id
        if V_id in _SCALAR_SPACES:
            n_cols = 1
        elif V_id in _VECTOR_SPACES:
            n_cols = 3
        else:
            raise ValueError(f"Unknown space identifier {V_id} for the domain.")

        assert W_id in self.derham.spline_attributes, (
            f"Spline attributes for the codomain space {W_id} not found in the Derham object !!"
        )

        # one tuple of 1d quadrature grids per component of W_id (= block row)
        quad_grid_pts = self.derham.spline_attributes[W_id].quad_grid_pts
        logger.debug(f"{len(quad_grid_pts) = }")

        weights_values = []
        evaluated_grid = None
        for m, component in enumerate(quad_grid_pts):
            grids_1d = [pts.flatten() for pts in component]
            grid_shape = tuple(g.size for g in grids_1d)

            # evaluate the factors only if the grid differs from the one of the previous row
            reuse = evaluated_grid is not None and all(
                xp.array_equal(g, g_prev) for g, g_prev in zip(grids_1d, evaluated_grid)
            )
            if not reuse:
                logger.debug(f"Evaluating weights for block row {m} of {W_id} on grid of shape {grid_shape}.")
                mv, scalar = self._eval_factors(classified, grids_1d)
                evaluated_grid = grids_1d

            # a scalar array reused from a previous row must not be shared between blocks
            weights_values.append(
                self._extract_row(classified, mv, scalar, m, n_cols, grid_shape, copy_scalar=reuse),
            )

        return weights_values, classified.spline_functions

    def _classify_weights(self, weights: tuple, V_id: str, W_id: str) -> "_ClassifiedWeights":
        """Sort the entries of a 1D weights tuple into scalar, vector and matrix factors and SplineFunctions.

        Strings are converted to callables (or constant matrices), nested lists to constant matrices;
        for general callables the kind is determined from the number of dimensions of their output
        (see :meth:`_callable_output_dim`).
        """
        classified = _ClassifiedWeights()

        for n, f in enumerate(weights):
            logger.debug(f"Processing weight #{n}: {f}")

            if isinstance(f, str):
                # predefined metric/Jacobian weight
                f_call, kind = self._string_weight_callable(f)
                if kind == "matrix":
                    classified.matrices.append(f_call)
                else:
                    classified.scalars.append(f_call)

            elif isinstance(f, list):
                # constant 3x3 matrix given as nested list (copied, such that later changes to the list have no effect)
                if len(f) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in f):
                    raise ValueError(f"Nested list weight must be of shape 3x3, got {f}.")
                classified.matrices.append(xp.array(f, dtype=float))

            elif isinstance(f, SplineFunction):
                # evaluated during assembly in WeightedMassOperator
                classified.spline_functions[f.name] = f

            elif callable(f):
                # general callable (e.g. a function or a LocalRotationMatrix)
                out_dim = self._callable_output_dim(f)
                if out_dim == 3:
                    classified.scalars.append(f)
                elif out_dim == 4:
                    # a column vector maps a scalar space to a vector space, a row vector vice versa
                    if classified.col_vector is not None or classified.row_vector is not None:
                        raise ValueError(f"At most one vector-valued weight is allowed, got a second one {f}.")
                    if V_id in _SCALAR_SPACES and W_id in _VECTOR_SPACES:
                        classified.col_vector = f
                    elif V_id in _VECTOR_SPACES and W_id in _SCALAR_SPACES:
                        classified.row_vector = f
                    else:
                        raise ValueError(
                            f"Vector weight {f} is only supported for scalar<->vector maps; got {V_id}->{W_id}."
                        )
                elif out_dim == 5:
                    classified.matrices.append(f)
                else:
                    raise ValueError(f"Callable {f} has wrong output dimension {out_dim}.")

            else:
                raise TypeError(f"Unsupported weight {f} of type {type(f)}.")

        return classified

    def _string_weight_callable(self, key: str):
        """Return the weight and its kind (``'matrix'`` or ``'scalar'``) for a predefined string weight.

        Matrix callables return arrays of shape (n1, n2, n3, 3, 3), scalar callables of shape (n1, n2, n3).
        ``'Identity'`` is returned as constant matrix of shape (3, 3).
        """
        if key == "G":
            return lambda e1, e2, e3: self.domain.metric(e1, e2, e3, change_out_order=True), "matrix"
        elif key == "Ginv":
            return lambda e1, e2, e3: self.domain.metric_inv(e1, e2, e3, change_out_order=True), "matrix"
        elif key == "DFinv":
            return lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3, change_out_order=True), "matrix"
        elif key == "DFinvT":
            return (
                lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3, change_out_order=True, transposed=True),
                "matrix",
            )
        elif key == "Identity":
            return xp.eye(3), "matrix"
        elif key == "sqrt_g":
            return lambda e1, e2, e3: abs(self.domain.jacobian_det(e1, e2, e3)), "scalar"
        elif key == "1/sqrt_g":
            return lambda e1, e2, e3: 1.0 / abs(self.domain.jacobian_det(e1, e2, e3)), "scalar"
        else:
            raise NotImplementedError(f"The option {key} is not available.")

    @staticmethod
    def _callable_output_dim(f: Callable) -> int:
        """Number of dimensions of the output of ``f`` on a small 3d test grid.

        3 means scalar-valued, 4 vector-valued and 5 matrix-valued.
        """
        xx, yy, zz = xp.meshgrid(
            xp.linspace(0, 1, 1),
            xp.linspace(0, 1, 2),
            xp.linspace(0, 1, 3),
            indexing="ij",
        )
        return f(xx, yy, zz).ndim

    @staticmethod
    def _check_weight_compatibility(classified: "_ClassifiedWeights", V_id: str, W_id: str):
        """Check that the kinds of the weight factors fit to the domain ``V_id`` and the codomain ``W_id``."""
        if classified.col_vector is not None and not (
            V_id in _SCALAR_SPACES and W_id in _VECTOR_SPACES and len(classified.matrices) == 0
        ):
            raise ValueError("Column vector weight requires a scalar->vector map without matrix factors.")
        if classified.row_vector is not None and not (
            V_id in _VECTOR_SPACES and W_id in _SCALAR_SPACES and len(classified.matrices) == 0
        ):
            raise ValueError("Row vector weight requires a vector->scalar map without matrix factors.")
        if len(classified.matrices) > 0 and not (V_id in _VECTOR_SPACES and W_id in _VECTOR_SPACES):
            raise ValueError(f"Matrix weights require a vector->vector map, got {V_id}->{W_id}.")

    @staticmethod
    def _eval_matrix_product(matrices: list, E1, E2, E3):
        """Matrix product (left to right) of the matrix factors at the points (E1, E2, E3) of a sparse meshgrid.

        Constant factors of shape (3, 3) are multiplied as such, without being expanded to the grid.
        Returns an array of shape (n1, n2, n3, 3, 3), or of shape (3, 3) if all factors are constant.
        """
        out = None
        for f in matrices:
            factor = f if isinstance(f, xp.ndarray) else f(E1, E2, E3)
            # batched product over the grid; the 3x3 part is in the last two axes
            out = factor if out is None else out @ factor
        return out

    @staticmethod
    def _eval_factors(classified: "_ClassifiedWeights", grids_1d: list):
        """Evaluate the matrix/vector factor and the product of the scalar factors on the grid ``grids_1d``.

        Returns
        -------
        mv : xp.ndarray | None
            Matrix product of shape (n1, n2, n3, 3, 3) or (3, 3) (if constant), vector of shape
            (n1, n2, n3, 3), or None if there is neither a matrix nor a vector factor.

        scalar : xp.ndarray | None
            Product of the scalar factors, of shape (n1, n2, n3), or None if there are none.
        """
        E1, E2, E3, _ = Domain.prepare_eval_pts(*grids_1d)

        if classified.matrices:
            mv = WeightedMassOperators._eval_matrix_product(classified.matrices, E1, E2, E3)
            _log_weight_stats("matrix weight", mv)
        elif classified.col_vector is not None:
            mv = classified.col_vector(E1, E2, E3)
            _log_weight_stats("column vector weight", mv)
        elif classified.row_vector is not None:
            mv = classified.row_vector(E1, E2, E3)
            _log_weight_stats("row vector weight", mv)
        else:
            mv = None

        scalar = None
        for f in classified.scalars:
            val = f(E1, E2, E3)
            scalar = val if scalar is None else scalar * val
        if scalar is not None:
            _log_weight_stats("scalar weight", scalar)

        return mv, scalar

    @staticmethod
    def _extract_row(
        classified: "_ClassifiedWeights",
        mv,
        scalar,
        m: int,
        n_cols: int,
        grid_shape: tuple,
        copy_scalar: bool = False,
    ) -> list:
        """Blocks of row ``m`` of the weights, from the evaluated factors (see :meth:`_eval_factors`).

        Returns a list of ``n_cols`` new C-contiguous arrays of shape ``grid_shape``; ``None`` marks a zero block.
        If ``copy_scalar`` is True, a purely scalar block is a copy of ``scalar`` instead of ``scalar`` itself.
        """
        row = []
        for n in range(n_cols):
            # entry (m, n) of the matrix or vector factor; None if there is no such factor
            if classified.matrices:
                if mv.ndim == 2:
                    # constant matrix: exactly zero entries give zero blocks
                    if mv[m, n] == 0.0:
                        row.append(None)
                        continue
                    factor = float(mv[m, n])
                else:
                    factor = mv[:, :, :, m, n]
            elif classified.col_vector is not None:
                factor = mv[:, :, :, m]
            elif classified.row_vector is not None:
                factor = mv[:, :, :, n]
            else:
                factor = None

            if factor is None:
                # purely scalar weight: only the diagonal blocks are non-zero
                if scalar is not None and m == n:
                    row.append(scalar.copy() if copy_scalar else scalar)
                else:
                    row.append(None)
            elif scalar is None:
                row.append(xp.ascontiguousarray(xp.broadcast_to(factor, grid_shape), dtype=float))
            else:
                row.append(xp.ascontiguousarray(factor * scalar))

        return row

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


def _zero_weight(*etas):
    """Zero default weight of an allocated block of a :class:`WeightedMassOperator`; keeps the block
    structure until actual weights are passed to :meth:`WeightedMassOperator.assemble`."""
    return 0 * etas[0]


class WeightedMassOperator(LinearOperator):
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

    dry_run : bool
        If True, the (potentially large) stencil matrices of the operator are not allocated;
        only their sizes are computed, see :attr:`nbytes`. The block structure (which blocks are
        non-zero) is determined in exactly the same way as for a regular operator, but the operator
        can neither be assembled nor applied. Used to estimate the memory footprint of the FEEC
        matrices before allocating them, see
        :meth:`~struphy.feec.mass.WeightedMassOperators.estimate_mem`.

    nquads : tuple | list, optional
        Number of quadrature points per direction; defaults to ``derham.nquads``.
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
        dry_run: bool = False,
    ):
        logger.debug(
            f"__init__: {name = }, {transposed = }, {matrix_free = }, {nquads = }, {dry_run = }, "
            f"spline_functions={list(spline_functions or {})}"
        )

        if dry_run and transposed:
            raise ValueError("dry_run=True is not supported for transposed operators.")

        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

        self._derham = derham
        self._nquads = nquads
        self._V = V
        self._W = W
        self._name = name
        self._weights_info = weights_info
        self._transposed = transposed
        self._matrix_free = matrix_free
        self._dry_run = dry_run
        self._dtype = V.coeff_space.dtype

        # recipe for re-creating the operator with WeightedMassOperators.create_weighted_mass, see to_dict()
        self._creation_info: dict | None = None

        # spline functions that are used as weights in the operator, to be evaluated at quadrature points
        self._spline_functions = spline_functions if spline_functions is not None else {}

        self._init_projection_ops(V_extraction_op, W_extraction_op, V_boundary_op, W_boundary_op)
        self._init_domain_codomain_spaces()
        self._prepare_spline_weights()

        # allocate the (block) matrix M (V -> W) and set its weights (zero blocks are None)
        if isinstance(weights_info, str):
            self._symmetry = weights_info
            blocks, self._weights = self._init_blocks_from_symmetry(weights_info)
        else:
            self._symmetry = None
            blocks, self._weights = self._init_blocks_from_weights(weights_info)

        self._mat = self._wrap_blocks(blocks)

        # the weights are stored in the block order of the (possibly transposed) operator
        if transposed:
            self._mat = self._mat.transpose()
            self._weights = self._transpose_weights(self._weights)

        self._init_domain_codomain()

        if self._dry_run:
            # memory estimation only (see the nbytes property): skip the .dot() temporaries
            # and the assembly kernel; none of them is needed for sizing and both would allocate memory.
            return

        self._allocate_dot_temporaries()

        if not self._matrix_free:
            self._load_assembly_kernel()

    # ------------------------------------------------------------------
    # Helpers for __init__ (and assemble/transpose)
    # ------------------------------------------------------------------

    @staticmethod
    def _component_spaces(space: TensorFemSpace | VectorFemSpace) -> tuple[TensorFemSpace, ...]:
        """Scalar component spaces of ``space``, i.e. ``(space,)`` for a TensorFemSpace
        and ``space.spaces`` for a VectorFemSpace."""
        if isinstance(space, TensorFemSpace):
            return (space,)
        return space.spaces

    @staticmethod
    def _transpose_weights(weights: list) -> list:
        """Transpose a 2d list of block weights (``None`` entries are kept)."""
        return [[weights[n][m] for n in range(len(weights))] for m in range(len(weights[0]))]

    @staticmethod
    def _evaluate_weight(weight, pts: list, copy: bool = False):
        """Evaluate a block weight at the quadrature points.

        Parameters
        ----------
        weight : callable | xp.ndarray | None
            Weight function of the logical coordinates, or its values at the quadrature points.

        pts : list[xp.ndarray]
            1d arrays of the quadrature points in each direction.

        copy : bool
            Whether to return a copy of an xp.ndarray weight (callables are always evaluated into a new array).

        Returns
        -------
        mat_w : xp.ndarray | None
            The weight on the tensor-product quadrature grid, or ``None`` if ``weight`` is ``None``.
        """
        if weight is None:
            return None

        if callable(weight):
            PTS = xp.meshgrid(*pts, indexing="ij")
            mat_w = weight(*PTS).copy()
        elif isinstance(weight, xp.ndarray):
            mat_w = weight.copy() if copy else weight
        else:
            raise TypeError(f"Weights must be callable, xp.ndarray or None, but are {type(weight)}.")

        grid_shape = tuple(pt.size for pt in pts)
        if mat_w.shape != grid_shape:
            raise ValueError(f"Weight has shape {mat_w.shape}, but the quadrature grid is {grid_shape}.")
        return mat_w

    def _quad_pts(self, space_name: str, component: int) -> list:
        """1d arrays of the (local) quadrature points of the given component of the space ``space_name``."""
        return [points.flatten() for points in self.derham.spline_attributes[space_name].quad_grid_pts[component]]

    def _is_globally_nonzero(self, local_nonzero: bool) -> bool:
        """Logical OR of ``local_nonzero`` over all MPI ranks.

        This is a collective call: it must be reached on all ranks, such that all ranks take
        the same decision on whether a block is allocated/assembled."""
        flag = xp.array(bool(local_nonzero), dtype=bool)
        if self.derham.comm is not None:
            self.derham.comm.Allreduce(MPI.IN_PLACE, flag, op=MPI.LOR)
        return bool(flag)

    def _init_projection_ops(self, V_extraction_op, W_extraction_op, V_boundary_op, W_boundary_op):
        """Set the polar extraction operators :math:`\\mathbb E_V, \\mathbb E_W` and the boundary
        operators :math:`\\mathbb B_V, \\mathbb B_W` (identities if not given), and their transposes."""
        # basis extraction operators
        if V_extraction_op is not None:
            if not V_extraction_op.domain == self._V.coeff_space:
                raise ValueError("The domain of V_extraction_op must be the coefficient space of V.")
            self._V_extraction_op = V_extraction_op
        else:
            self._V_extraction_op = IdentityOperator(self._V.coeff_space)

        if W_extraction_op is not None:
            if not W_extraction_op.domain == self._W.coeff_space:
                raise ValueError("The domain of W_extraction_op must be the coefficient space of W.")
            self._W_extraction_op = W_extraction_op
        else:
            self._W_extraction_op = IdentityOperator(self._W.coeff_space)

        # boundary operators (act on the codomain of the extraction operators)
        if V_boundary_op is not None:
            if not V_boundary_op.domain == self._V_extraction_op.codomain:
                raise ValueError("The domain of V_boundary_op must be the codomain of the V extraction operator.")
            self._V_boundary_op = V_boundary_op
        else:
            self._V_boundary_op = IdentityOperator(self._V_extraction_op.codomain)

        if W_boundary_op is not None:
            if not W_boundary_op.domain == self._W_extraction_op.codomain:
                raise ValueError("The domain of W_boundary_op must be the codomain of the W extraction operator.")
            self._W_boundary_op = W_boundary_op
        else:
            self._W_boundary_op = IdentityOperator(self._W_extraction_op.codomain)

        self._V_extraction_op_T = self._V_extraction_op.T
        self._W_extraction_op_T = self._W_extraction_op.T
        self._V_boundary_op_T = self._V_boundary_op.T
        self._W_boundary_op_T = self._W_boundary_op.T

    def _init_domain_codomain_spaces(self):
        """Set the domain/codomain FEM spaces and their symbolic names (V and W swapped for transposed
        operators), and whether both spaces are scalar (then ``_mat`` is a single block)."""
        V_name = self._V.symbolic_space
        W_name = self._W.symbolic_space
        logger.debug(f"{V_name = }, {W_name = }")

        for space_name in (V_name, W_name):
            if space_name not in self.derham.spline_attributes:
                raise ValueError(f"Spline attributes for the space {space_name} not found in the Derham object.")

        if self._transposed:
            self._domain_femspace, self._domain_symbolic_name = self._W, W_name
            self._codomain_femspace, self._codomain_symbolic_name = self._V, V_name
        else:
            self._domain_femspace, self._domain_symbolic_name = self._V, V_name
            self._codomain_femspace, self._codomain_symbolic_name = self._W, W_name

        self._is_scalar = isinstance(self._V, TensorFemSpace) and isinstance(self._W, TensorFemSpace)

    def _new_block(self, vspace: TensorFemSpace, wspace: TensorFemSpace, weights=None):
        """Allocate a single (non-assembled) block mapping the scalar space ``vspace`` to ``wspace``.

        Returns a :class:`StencilMatrixFreeMassOperator` with the given ``weights`` for matrix-free
        operators, and a StencilMatrix otherwise (then ``weights`` is ignored; they are passed
        to the assembly kernel in :meth:`assemble`)."""
        if self._matrix_free:
            return StencilMatrixFreeMassOperator(self.derham, vspace, wspace, weights=weights, nquads=self.nquads)

        return StencilMatrix(
            vspace.coeff_space,
            wspace.coeff_space,
            backend=PSYDAC_BACKEND_GPYCCEL,
            precompiled=True,
            dry_run=self._dry_run,
        )

    def _init_blocks_from_symmetry(self, symmetry: str) -> tuple[list, list]:
        """Allocate the blocks of a square block operator (V = W) according to a given symmetry.

        Only the blocks allowed by the symmetry (all for ``symm``, off-diagonal for ``asym``, diagonal
        for ``diag``) are allocated, the others are ``None``. Allocated blocks get a zero default weight,
        the actual weights are passed later to :meth:`assemble`.

        Returns
        -------
        blocks, weights : list, list
            2d lists of the blocks and their weights, in the block order of the non-transposed operator.
        """
        V_name = self._V.symbolic_space
        W_name = self._W.symbolic_space
        if V_name != W_name:
            raise ValueError(f"A symmetry can only be given for square operators (V=W), but {V_name = } and {W_name = }.")
        if not isinstance(self._V, VectorFemSpace):
            raise ValueError(
                f"A symmetry can only be given for vector-valued spaces (Hcurl, Hdiv, H1vec), but {V_name = }."
            )

        # which blocks (i, j) are allocated for the given symmetry
        block_masks = {
            "symm": lambda i, j: True,
            "asym": lambda i, j: i != j,
            "diag": lambda i, j: i == j,
        }
        if symmetry not in block_masks:
            raise NotImplementedError(f"given symmetry {symmetry} is not implemented!")
        is_allocated = block_masks[symmetry]

        blocks = [
            [self._new_block(Vs, Ws) if is_allocated(i, j) else None for j, Vs in enumerate(self._V.spaces)]
            for i, Ws in enumerate(self._W.spaces)
        ]
        weights = [[_zero_weight if block is not None else None for block in row] for row in blocks]

        return blocks, weights

    def _init_blocks_from_weights(self, weights_info: list | None) -> tuple[list, list]:
        """Allocate the blocks according to the given weights.

        For ``weights_info=None`` all blocks are allocated with zero default weights. Otherwise a block
        is allocated only if its weight is non-zero on at least one MPI rank; globally zero blocks are
        set to ``None`` to accelerate the dot product.

        Returns
        -------
        blocks, weights : list, list
            2d lists of the blocks and their weights, in the block order of the non-transposed operator.
        """
        W_name = self._W.symbolic_space

        blocks = []
        weights = []

        # loop over codomain spaces (rows)
        for a, wspace in enumerate(self._component_spaces(self._W)):
            blocks += [[]]
            weights += [[]]

            pts = self._quad_pts(W_name, a)

            # loop over domain spaces (columns)
            for b, vspace in enumerate(self._component_spaces(self._V)):
                if weights_info is None:
                    blocks[-1] += [self._new_block(vspace, wspace)]
                    weights[-1] += [_zero_weight]
                    continue

                # A block can be locally zero on this MPI rank but non-zero on another rank.
                # We therefore check whether the block is globally non-zero before deciding
                # whether to allocate it. All ranks must make the same block-allocation decision,
                # otherwise exchange_assembly_data() will communicate incompatible block structures.
                loc_weight = weights_info[a][b]
                mat_w = self._evaluate_weight(loc_weight, pts)
                local_nonzero = mat_w is not None and bool(xp.any(xp.abs(mat_w) > 1e-14))

                if self._is_globally_nonzero(local_nonzero):
                    if loc_weight is None:
                        # The block is globally non-zero, but this rank has a locally zero weight.
                        # We still allocate the block and use a zero local weight array, such that
                        # the local matrix has the same structure as on the other MPI ranks.
                        loc_weight = xp.zeros(tuple(pt.size for pt in pts), dtype=float)

                    blocks[-1] += [self._new_block(vspace, wspace, weights=loc_weight)]
                    weights[-1] += [loc_weight]
                else:
                    blocks[-1] += [None]
                    weights[-1] += [None]

        return blocks, weights

    def _wrap_blocks(self, blocks: list):
        """Return the single block of a 1x1 operator, or a BlockLinearOperator (V -> W) otherwise.

        A zero 1x1 block is still allocated, such that the matrix is never ``None``."""
        if len(blocks) == len(blocks[0]) == 1:
            if blocks[0][0] is None:
                return self._new_block(self._component_spaces(self._V)[0], self._component_spaces(self._W)[0])
            return blocks[0][0]

        return BlockLinearOperator(self._V.coeff_space, self._W.coeff_space, blocks=blocks)

    def _prepare_spline_weights(self):
        """Prepare the evaluation of :attr:`spline_functions` at the quadrature points: knot spans,
        basis function values and output buffers, keyed by spline name. In :meth:`assemble`,
        the block weights are multiplied by the evaluated spline functions.

        The (local) quadrature grid is the same for all components of all spaces, since it is
        determined by the elements of the domain decomposition (this is also used in
        :mod:`struphy.feec.preconditioner`). Hence the spline functions are evaluated once,
        on the grid of the first codomain component, and used for all blocks."""
        pts = self._quad_pts(self._codomain_symbolic_name, 0)
        grid_shape = tuple(len(pt) for pt in pts)

        self._spline_values = {}
        self._spline_spans = {}
        self._spline_bases = {}
        for name, spline in self.spline_functions.items():
            if not isinstance(spline, SplineFunction):
                raise TypeError(f"The entry {name} in spline_functions must be a SplineFunction object.")
            self._spline_values[name] = xp.zeros(grid_shape, dtype=float)
            self._spline_spans[name], bns, bds = self.derham.prepare_eval_tp_fixed(pts)
            if spline.space_id == "H1":
                self._spline_bases[name] = bns
            elif spline.space_id == "L2":
                self._spline_bases[name] = bds
            else:
                raise NotImplementedError(
                    f"Spline functions in spline_functions must be defined on H1 or L2 spaces, but {spline.space_id} was given for the spline function {name}.",
                )

    def _init_domain_codomain(self):
        """Set domain and codomain of the operator, i.e. the codomains of the extraction operators
        of V and W (swapped for transposed operators). These are the domain and codomain of :attr:`M`."""
        if self._transposed:
            self._domain = self._W_extraction_op.codomain
            self._codomain = self._V_extraction_op.codomain
        else:
            self._domain = self._V_extraction_op.codomain
            self._codomain = self._W_extraction_op.codomain

    def _allocate_dot_temporaries(self):
        """Allocate the intermediate vectors used in :meth:`dot`."""
        self._temp_WB = self._W_boundary_op.domain.zeros()
        self._temp_WE = self._W_extraction_op.domain.zeros()
        self._temp_VB = self._V_boundary_op.domain.zeros()
        self._temp_VE = self._V_extraction_op.domain.zeros()
        self._temp_mat = self._mat.domain.zeros()

    def _load_assembly_kernel(self):
        """Load the pyccelized assembly kernel for the dimension of the problem (1d, 2d or 3d)."""
        self._assembly_kernel = PyccelKernel(
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
    def dry_run(self) -> bool:
        """Whether the operator was created for memory estimation only, i.e. without allocating
        its stencil matrices (in which case it can neither be assembled nor applied)."""
        return self._dry_run

    @property
    def nbytes(self) -> int:
        """Local (per-MPI-rank) memory footprint of the stencil matrices of this operator, in bytes.
        Also available for operators created with ``dry_run=True``, i.e. before/without allocation.
        Matrix-free operators do not store a matrix and return 0."""
        return int(getattr(self._mat, "nbytes", 0))

    @property
    def dtype(self):
        return self._dtype

    def _check_identity_extraction(self, method: str):
        """Raise if the operator has non-trivial (polar) extraction operators, which
        :meth:`to_sparse_mat_only` and :meth:`to_array_mat_only` do not support."""
        if not all(isinstance(op, IdentityOperator) for op in (self._W_extraction_op, self._V_extraction_op)):
            raise NotImplementedError(f".{method}() is not implemented for polar extraction operators.")

    def to_sparse_mat_only(self):
        """Sparse matrix of the (block) stencil matrix, without extraction and boundary operators.

        In contrast to :meth:`tosparse` (inherited from LinearOperator, computed column by column
        with :meth:`dot`), this is cheap but does not account for boundary conditions."""
        self._check_identity_extraction("to_sparse_mat_only")
        return self._mat.tosparse()

    def to_array_mat_only(self):
        """Dense array of the (block) stencil matrix, without extraction and boundary operators.

        In contrast to :meth:`toarray` (inherited from LinearOperator, computed column by column
        with :meth:`dot`), this is cheap but does not account for boundary conditions."""
        self._check_identity_extraction("to_array_mat_only")
        return self._mat.toarray()

    @property
    def M(self):
        """Composite operator :math:`\\mathbb E_W \\mathbb M \\mathbb E_V^T` (V and W swapped if transposed),
        built on first access (allocates temporaries). Note that :meth:`dot` does not use it."""
        if not hasattr(self, "_M"):
            if self._transposed:
                self._M = self._V_extraction_op @ self._mat @ self._W_extraction_op_T
            else:
                self._M = self._W_extraction_op @ self._mat @ self._V_extraction_op_T
        return self._M

    @property
    def M0(self):
        """Composite operator :math:`\\mathbb B_W \\mathbb E_W \\mathbb M \\mathbb E_V^T \\mathbb B_V^T`
        (V and W swapped if transposed), built on first access (allocates temporaries)."""
        if not hasattr(self, "_M0"):
            if self._transposed:
                self._M0 = self._V_boundary_op @ self.M @ self._W_boundary_op_T
            else:
                self._M0 = self._W_boundary_op @ self.M @ self._V_boundary_op_T
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
            weights = self._transpose_weights(self._weights)
        else:
            weights = self._weights

        name = self.name + "T" if self.name is not None else None

        M = WeightedMassOperator(
            self.derham,
            self._V,
            self._W,
            name=name,
            V_extraction_op=self._V_extraction_op,
            W_extraction_op=self._W_extraction_op,
            V_boundary_op=self._V_boundary_op,
            W_boundary_op=self._W_boundary_op,
            weights_info=weights if self._symmetry is None else self._symmetry,
            spline_functions=self._spline_functions,
            transposed=not self._transposed,
            matrix_free=self._matrix_free,
            nquads=self._nquads,
        )

        # weights of M in its own (transposed) block order
        M._weights = self._transpose_weights(self._weights)

        if self._creation_info is not None:
            M._creation_info = dict(self._creation_info, is_transpose=not self._creation_info["is_transpose"])

        if self._matrix_free:
            if self._symmetry is not None:
                M.assemble(weights=M._weights)
        else:
            # transpose the assembled data instead of re-assembling from the weights: the data need not
            # stem from self._weights (e.g. accumulation matrices, which are filled by the particles)
            self._mat.transpose(out=M._mat)

            # remove blocks of M that are zero in self
            if isinstance(self._mat, BlockLinearOperator):
                for a, b in M._mat.nonzero_block_indices:
                    if self._mat[b, a] is None:
                        M._mat[a, b] = None

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
        assert not self._dry_run, (
            "A dry-run operator has no matrix data and cannot be assembled (memory estimation only)."
        )
        if weights is not None or not clear:
            # the data no longer stems from the creation recipe
            self._creation_info = None

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
            domain_spaces = self._component_spaces(self._domain_femspace)
            codomain_spaces = self._component_spaces(self._codomain_femspace)

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
                pts = self._quad_pts(W_name, a)

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

                    # evaluate weight at quadrature points (copy: spline factors below must not
                    # modify the stored geometric weight, assembly can be repeated as density changes)
                    mat_w = self._evaluate_weight(loc_weight, pts, copy=True)

                    if loc_weight is not None:
                        # evalute splines and multiply
                        for name, spline in self.spline_functions.items():
                            logger.debug(
                                f"Maximum coefficient of spline {name}: {xp.max(xp.abs(spline.vector.toarray()))}"
                            )
                            values = spline.eval_tp_fixed_loc(
                                self._spline_spans[name],
                                self._spline_bases[name],
                                out=self._spline_values[name],
                            )
                            if xp.all(xp.abs(values) < 1e-14):
                                logger.warning(
                                    f"The spline weight {name} is close to zero at all quadrature points in the assembly of the weighted mass matrix {self.name}. Weights are not multiplied."
                                )
                                continue
                            mat_w *= values
                    else:
                        logger.debug(f"No weight for block {a, b}, setting mat_w to None.")

                    not_weight_zero = self._is_globally_nonzero(
                        loc_weight is not None and bool(xp.any(xp.abs(mat_w) > 1e-14)),
                    )

                    # evaluated basis functions at quadrature points of domain space
                    domain_basis = spline_attr[V_name].quad_grid_bases[b]

                    # assemble matrix (if mat_w is not zero) by calling the appropriate kernel (1d, 2d or 3d)
                    if not_weight_zero or self._is_scalar:
                        # get cell of block matrix (don't instantiate if all zeros)
                        if self._is_scalar:
                            mat = self._mat
                            if loc_weight is None:
                                # not_weight_zero is global after the MPI reduction. Hence this rank may
                                # enter the assembly branch even when its own local weight is None.
                                # In that case we assemble a zero local contribution, but we must still
                                # provide a correctly shaped array to the pyccel kernel.
                                mat_w = xp.zeros(
                                    tuple([pt.size for pt in pts]),
                                )
                        else:
                            mat = self._mat[a, b]

                            # block case: after the MPI Allreduce, this block may be globally
                            # non-zero even if it is locally zero on this rank.
                            if mat_w is None:
                                mat_w = xp.zeros(tuple([pt.size for pt in pts]))

                        # This can happen for block matrices if the block was previously considered
                        # zero locally, but is now required because it is non-zero on at least one
                        # MPI rank. The block must exist on all ranks before assembly/exchange.
                        if mat is None:
                            # Maybe in a previous iteration we had more zeros
                            # Can only happen in the Block case
                            self._mat[a, b] = self._new_block(domain_space, codomain_space)
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

    @property
    def is_reconstructible(self) -> bool:
        """Whether the operator can be re-created from :meth:`to_dict` (e.g. on another Derham).

        True for operators created by :meth:`WeightedMassOperators.create_weighted_mass` (and their transposes)
        whose data has not been modified afterwards (by ``assemble(weights=...)``, in-place arithmetic, ...).
        """
        return self._creation_info is not None

    def to_dict(self) -> dict:
        """Recipe for re-creating the operator with :meth:`WeightedMassOperators.create_weighted_mass`.

        The weights are stored as given at creation. The dictionary is JSON serializable if they are
        strings (``'Ginv'``, ``'sqrt_g'``, ...) or nested lists of numbers; callables are kept as objects.
        Re-create the operator (on any Derham) with :meth:`from_dict`.
        """
        if self._creation_info is None:
            raise ValueError(
                f"WeightedMassOperator {self.name!r} cannot be serialized: it was not created by "
                "WeightedMassOperators.create_weighted_mass or its data was modified afterwards."
            )
        params = dict(self._creation_info)
        # tuple (1D product of weights) and 2D list (block weights) are different formats; store the
        # tuple as a list (JSON) and record its type, since its entries may themselves be (3x3) lists
        params["weights_is_tuple"] = isinstance(params["weights"], tuple)
        if params["weights_is_tuple"]:
            params["weights"] = list(params["weights"])
        return {
            "type": self.__class__.__name__,
            "params": params,
        }

    @classmethod
    def from_dict(cls, dct: dict, mass_ops: "WeightedMassOperators") -> "WeightedMassOperator":
        """Re-create a :class:`WeightedMassOperator` from :meth:`to_dict` with the given collection.

        Parameters
        ----------
        dct : dict
            Output of :meth:`to_dict`.

        mass_ops : WeightedMassOperators
            Collection providing the Derham, domain and matrix_free option of the new operator.
        """
        assert dct["type"] == cls.__name__
        params = dct["params"]
        name = params["name"]
        weights = params["weights"]
        if params["weights_is_tuple"]:
            weights = tuple(weights)

        out = mass_ops.create_weighted_mass(
            params["V_id"],
            params["W_id"],
            name=name,
            weights=weights,
            assemble=True,
            transposed=params["transposed"],
        )
        return out.T if params["is_transpose"] else out

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
                name=self.name,
                V_extraction_op=self._V_extraction_op,
                W_extraction_op=self._W_extraction_op,
                V_boundary_op=self._V_boundary_op,
                W_boundary_op=self._W_boundary_op,
                weights_info=self._weights_info,
                spline_functions=self._spline_functions,
                transposed=self._transposed,
                matrix_free=self._matrix_free,
                nquads=self._nquads,
            )
            # current weights (they may have been changed by assemble(weights=...))
            out._weights = [list(row) for row in self._weights]

        self._mat.copy(out=out._mat)

        if self._creation_info is None:
            out._creation_info = None
        else:
            out._creation_info = dict(self._creation_info)  # to create a separate dictionary

        return out

    def __imul__(self, a):
        self._mat *= a
        self._creation_info = None
        return self

    def __iadd__(self, M):
        assert M.domain is self.domain
        assert M.codomain is self.codomain

        if isinstance(M, WeightedMassOperator):
            self._creation_info = None
            self._mat += M._mat
            return self

        elif isinstance(M, LinearOperator):
            self._creation_info = None
            self._mat += M
            return self

        else:
            return LinearOperator.__add__(self, M)

    def __isub__(self, M):
        assert M.domain is self.domain
        assert M.codomain is self.codomain

        if isinstance(M, WeightedMassOperator):
            self._creation_info = None
            self._mat -= M._mat
            return self

        elif isinstance(M, LinearOperator):
            self._creation_info = None
            self._mat -= M
            return self

        else:
            return LinearOperator.__sub__(self, M)

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
        kernel = PyccelKernel(getattr(mass_kernels, "kernel_" + str(W.ldim) + "d_eval"))

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


class StencilMatrixFreeMassOperator(LinearOperator):
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
        self._dot_kernel = PyccelKernel(
            getattr(
                mass_kernels,
                "kernel_" + str(self._V.ldim) + "d_matrixfree",
            ),
        )

        self._diag_kernel = PyccelKernel(
            getattr(
                mass_kernels,
                "kernel_" + str(self._V.ldim) + "d_diag",
            ),
        )

        # temporary with ghost regions for the diagonal (contributions to other processes are exchanged)
        self._diag_tmp = W.coeff_space.zeros()

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

    def transpose(self, conjugate=False):
        return StencilMatrixFreeMassOperator(
            self._derham,
            self._W,
            self._V,
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

        diag_tmp = self._diag_tmp
        diag_tmp._data[:] = 0.0
        if self._weights is not None:
            self._diag_kernel(
                *self._codomain_spans,
                *self._W.degree,
                *self._codomain_starts,
                *self._codomain_pads,
                *self._wts,
                *self._codomain_basis,
                mat_w,
                diag_tmp._data,
            )
            diag_tmp.exchange_assembly_data()

        # entries owned by this process (without ghost regions)
        idx = tuple(slice(p * m, -p * m) if p != 0 else slice(None) for p, m in zip(W.pads, W.shifts))
        diag = diag_tmp._data[idx]

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

        # basis extraction operator (tensor-product --> polar dofs) and tensor-product vector for assembly
        self._extraction_op = self.mass_ops.derham.extraction_ops[self.space_key]
        if self.mass_ops.derham.polar_splines:
            self._dofs_tp = self.space.coeff_space.zeros()

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

    def solve(
        self,
        rhs: StencilVector | BlockVector | PolarVector,
        out=None,
    ) -> StencilVector | BlockVector | PolarVector:
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

        assert isinstance(rhs, (StencilVector, BlockVector, PolarVector))
        assert rhs.space == self.Mmat.domain

        if out is None:
            out = self.solver.dot(rhs)
        else:
            self.solver.dot(rhs, out=out)

        return out

    def get_dofs(
        self,
        fun: Callable | xp.ndarray | list[Callable | xp.ndarray] | tuple[Callable | xp.ndarray],
        dofs: StencilVector | BlockVector | PolarVector = None,
        apply_bc: bool = False,
        clear: bool = True,
    ) -> StencilVector | BlockVector | PolarVector:
        r"""
        Assembles (in 3d) the Stencil-/Block-/PolarVector

        .. math::

            V_{ijk} = \int f * w_\textrm{geom} * \Lambda^\alpha_{ijk}\,\textrm d \boldsymbol \eta = \left( f\,, \Lambda^\alpha_{ijk}\right)_{L^2}\,,

        where :math:`\Lambda^\alpha_{ijk}` are the basis functions of :math:`V_h^\alpha`,
        :math:`f` is an :math:`\alpha`-form proxy function and :math:`w_\textrm{geom}` stand for metric coefficients.

        Note that any geometric terms (e.g. Jacobians) in the L2 scalar product are automatically assembled
        into :math:`w_\textrm{geom}`, depending on the space of :math:`\alpha`-forms.

        For polar splines, the tensor-product vector is mapped to the polar sub-space with the basis extraction operator
        (the polar basis functions are linear combinations of the tensor-product ones).

        The integration is performed with Gauss-Legendre quadrature over the whole logical domain.

        Parameters
        ----------
        fun : Callable | xp.ndarray | list[Callable | xp.ndarray] | tuple[Callable | xp.ndarray]
            Weight function(s) (callables or xp.ndarrays) in a 1d list of shape corresponding to number of components.

        dofs : StencilVector | BlockVector | PolarVector, optional
            The vector for the output. Either an element of the domain of the mass matrix (PolarVector for polar splines),
            or a tensor-product Stencil-/BlockVector, in which case no basis extraction is performed.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom.

        clear : bool, optional
            Whether to first set all data to zero before assembly. If False, the new contributions are added to existing ones in vec.

        Returns
        -------
        dofs : StencilVector | BlockVector | PolarVector
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
            dofs = self.Mmat.codomain.zeros()
        else:
            assert isinstance(dofs, (StencilVector, BlockVector, PolarVector))
            assert dofs.space in (self.Mmat.codomain, self.space.coeff_space)

        # for polar splines, assemble into tensor-product vector first
        is_polar = isinstance(dofs, PolarVector)
        if is_polar:
            vec = self._dofs_tp
        else:
            vec = dofs

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
        if clear or is_polar:
            if isinstance(vec, StencilVector):
                vec._data[:] = 0.0
            else:
                for block in vec.blocks:
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

            if isinstance(vec, StencilVector):
                mass_kernels.kernel_3d_vec(
                    *spans,
                    *fem_space.degree,
                    *starts,
                    *pads,
                    *wts,
                    *basis,
                    mat_w,
                    vec._data,
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
                    vec[a]._data,
                )

        # exchange assembly data (accumulate ghost regions) and update ghost regions
        vec.exchange_assembly_data()
        vec.update_ghost_regions()

        # apply basis extraction operator (tensor-product --> polar)
        if is_polar:
            if clear:
                self._extraction_op.dot(vec, out=dofs)
            else:
                dofs += self._extraction_op.dot(vec)
            dofs.update_ghost_regions()

        # apply boundary operator
        if apply_bc:
            dofs = self.mass_ops.derham.boundary_ops[self.space_key].dot(dofs)

        return dofs

    def __call__(
        self,
        fun: Callable | list[Callable] | tuple[Callable],
        out: StencilVector | BlockVector | PolarVector = None,
        dofs: StencilVector | BlockVector | PolarVector = None,
        apply_bc: bool = False,
    ) -> StencilVector | BlockVector | PolarVector:
        """
        Applies projector to given callable(s).

        Parameters
        ----------
        fun : Callable | list[Callable] | tuple[Callable]
            The function to be projected. List of three callables for vector-valued functions.

        out : StencilVector | BlockVector | PolarVector, optional
            If given, the result will be written into this vector in-place.

        dofs : StencilVector | BlockVector | PolarVector, optional
            If given, the dofs will be written into this vector in-place.

        apply_bc : bool, optional
            Whether to apply essential boundary conditions to degrees of freedom and coefficients.

        Returns
        -------
        coeffs : feectools.linalg.basic.vector
            The FEM spline coefficients after projection.
        """
        return self.solve(self.get_dofs(fun, dofs=dofs, apply_bc=apply_bc), out=out)


class AverageOperator(LinearOperator):
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

    nquads : list[int], optional
        Number of quadrature points in each direction. If not given, those of the derham complex are used.
    """

    def __init__(
        self,
        derham: Derham,
        space: str = "H1",
        direction: int = 2,
        transposed: bool = False,
        nquads: list[int] | None = None,
    ):

        if space not in derham.space_to_form:
            raise AssertionError("Must match a space of the derham complex")
        if space != "H1":
            raise NotImplementedError("AverageOperator is only implemented for space H1")
        space_id = "V" + derham.space_to_form[space]
        self._V = getattr(derham, space_id)  # StencilVectorSpace
        self._domain = getattr(derham, space_id)
        self._codomain = getattr(derham, space_id)
        self._pads = self._V.pads  # gets the number of ghost cells
        self._derham = derham
        self._dtype = self._domain.dtype
        self._space = space
        self._direction = direction
        self._transposed = transposed
        self._nquads = nquads
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
            coords = derham.domain_decomposition.coords
            color1 = int(coords[self._directions[1]])
            color2 = int(coords[self._directions[2]])
            color = color1 * nprocs[self._directions[2]] + color2
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
        return AverageOperator(
            self.derham, self._space, self._direction, transposed=not self._transposed, nquads=self._nquads
        )
