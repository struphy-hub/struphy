from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from feectools.linalg.basic import IdentityOperator
from feectools.linalg.stencil import StencilVector

from struphy.feec.mass import AverageOperator
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import ParticlesToGrid
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.utils.utils import check_option


class PoissonAdiabaticGyrokinetic(ImplicitDiffusion):
    r"""
    Weak discretization of the Poisson equation, with adabatic response of electrons.

    Find :math:`\phi \in H^1` such that

    .. math::

        \frac{1}{Z\epsilon^2} \int_\Omega \frac{n_0}{T_0} \psi\, \phi\,\textrm d \mathbf x + \int_\Omega \frac{n_0}{|B_0|²} \nabla \psi^\top \,\mathbb D \nabla \phi \,\textrm d \mathbf x = \sum_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`\mathbb D` can be an anisotropic (gyrokinetic) diffusion operator (or the identity),
    :math:`\epsilon \in \mathbb R` is the gyrokinetic ratio defined in units, and Z the charge number of ions.
    Boundary terms from integration by parts are assumed to vanish.

    The equation is discretized as

    .. math::

        \left( \frac{1}{Z\epsilon^2}\,\mathbb M^0_ad + \mathbb G^\top \mathbb M^1_\textnormal{gyro} \mathbb G \right)\, \boldsymbol\phi^{n+1} = \sum_i(\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`\mathbb M^1_\textnormal{gyro}` is the gyrokinetic mass matrix
    and :math:`\mathbb S` is a stabilization matrix.

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    stab_eps : float
        Stabilization parameter multiplied on stab_mat (default=0.0).

    stab_mat : str
        Name of the stabilizing matrix.

    rho : FEECVariable or Callable or ParticlesToGrid or list
        (List of) right-hand side source term(s) of the Poisson problem (optional, can be set with
        a setter later). See :class:`~struphy.propagators.implicit_diffusion.ImplicitDiffusion` for
        the accepted entries; particle sources are passed as
        :class:`~struphy.pic.accumulation.particles_to_grid.ParticlesToGrid`.

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    solver : dict
        Parameters for the iterative solver (see ``__init__`` for details).
    """

    def __init__(
        self,
        rho: FEECVariable | Callable | ParticlesToGrid | list = None,
        rho_coeffs: float | list = None,
        epsilon: float = 1.0,
        Z: int = 1,
        diagnostic: FEECVariable | None = None,
    ):
        """
        Parameters
        ----------
        see ImplicitDiffusion.__init__ docstring for other parameters.

        epsilon : float
            Gyrokinetic parameter, appears in gyrokinetic Poisson equation

        Z : int
            Charge number for ions, appears in gyrokinetic Poisson equation
        """
        super().__init__(rho=rho, rho_coeffs=rho_coeffs, diagnostic=diagnostic)
        self.epsilon = epsilon
        self.Z = Z

    @dataclass
    class Options:
        r"""Configuration options for :class:`PoissonAdiabaticGyrokinetic`.

        Parameters
        ----------
        epsilon : float, default=1.0
            Gyrokinetic parameter

        Z : float, default=1.0
            Charge number of ions

        stab_mat : {"M0", "M0ad", "Id"}, default="Id"
            Stabilization matrix multiplied by ``stab_eps``.

            - ``"M0"``: standard weighted 0-form mass operator.
            - ``"M0ad"``: adiabatic-electron weighted 0-form mass operator.
            - ``"Id"``: identity operator.

        which_geometry: {"cylindrical", "toroidal"}, default="cylindrical"
            Geometry of the problem, determines the meaning of `<\phi>` in the stabilization term.

        diffusion_mat : {"M1", "M1perp", "M1gyro"}, defaults="M1gyro"
            Diffusion matrix.

        rho : FEECVariable or Callable or ParticlesToGrid or list, default=None
            Right-hand side source term(s) of the Poisson problem.
            Accepted entries are:

            - ``None``: zero source.
            - ``FEECVariable`` in ``H1``.
            - ``Callable`` to be projected to ``H1`` via ``L2Projector``.
            - :class:`~struphy.pic.accumulation.particles_to_grid.ParticlesToGrid`, describing a
              particle-to-grid (charge/current) deposition.
            - a ``list`` containing any mix of the entries above.

        rho_coeffs : float or list, default=None
            Multiplicative coefficient(s) applied to ``rho``.
            If ``None``, coefficients default to ``1.0`` for all sources.

        x0 : StencilVector, default=None
            Initial guess for the iterative linear solver.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver passed to
            :func:`psydac.linalg.solvers.inverse`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Name of the preconditioner configuration.
            Currently this class inherits the same behavior as
            :class:`ImplicitDiffusion`, where ``pc=None`` is used internally.

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.

        filter_params : dict[PICVariable | SPHVariable, FilterParameters], default=None
            If not None, specifies a filter to the accumulation of a specific variable.

        Notes
        -----
        ``Poisson.Options`` reuses :class:`ImplicitDiffusion` internals by
        enforcing
        ``sigma_2 = 0.0``, ``sigma_3 = 1.0``, ``divide_by_dt = False`` and
        ``diffusion_mat = "M1"`` in ``__post_init__``.
        """

        # specific literals
        OptsStabMat = Literal["M0", "M0ad", "Id", "M0ad_withT"]
        OptsDiffusionMat = Literal["M1", "M1perp", "M1gyro"]
        OptsGeometry = Literal["cylindrical", "toroidal"]
        # propagator options
        stab_mat: OptsStabMat = "M0ad_withT"
        diffusion_mat: OptsDiffusionMat = "M1gyro"
        which_geometry: OptsGeometry = "cylindrical"
        x0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: dict[PICVariable | SPHVariable, FilterParameters] = None

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    def allocate(self):
        epsilon = self.epsilon
        Z = self.Z
        self.options.sigma_1 = 1 / epsilon**2 / self.Z
        self.options.sigma_2 = 0.0
        self.options.sigma_3 = 1 / epsilon
        self.options.divide_by_dt = False
        super().allocate()
        if self.options.which_geometry == "cylindrical":
            average_mat = AverageOperator(self.derham, "H1", 2)
            self._stab_mat = self._stab_mat @ (
                IdentityOperator(self._stab_mat.domain, self._stab_mat.codomain) - average_mat
            )
        elif self.options.which_geometry == "toroidal":
            average_mat1 = AverageOperator(self.derham, "H1", 1)
            average_mat2 = AverageOperator(self.derham, "H1", 2)
            self._stab_mat = self._stab_mat @ (
                IdentityOperator(self._stab_mat.domain, self._stab_mat.codomain) - (average_mat1 @ average_mat2)
            )

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
