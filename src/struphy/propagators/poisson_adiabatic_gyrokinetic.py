from dataclasses import dataclass
from typing import Callable, Literal

from feectools.linalg.stencil import StencilVector

from struphy.feec.mass import AverageOperator
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.pic.base import Particles
from struphy.propagators.implicit_diffusion import ImplicitDiffusion
from struphy.utils.utils import check_option


class PoissonAdiabaticGyrokinetic(ImplicitDiffusion):
    r"""
    Weak discretization of the Poisson equation, with adabatic response of electrons.

    Find :math:`\phi \in H^1` such that

    .. math::

        \frac{1}{Z\epsilon^2} \int_\Omega \frac{n_0}{T_0} \psi\, \phi\,\textrm d \mathbf x + \int_\Omega \frac{n_0}{|B_0|²} \nabla \psi^\top \, \nabla \phi \,\textrm d \mathbf x = \sum_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`\epsilon \in \mathbb R` is the gyrokinetic ratio defined in units, and Z the charge number of ions.
    Boundary terms from integration by parts are assumed to vanish.

    The equation is discretized as

    .. math::

        \left( \frac{1}{Z\epsilon^2}\,\mathbb M^0_ad + \mathbb G^\top \mathbb M^1 \mathbb G \right)\, \boldsymbol\phi^{n+1} = \sum_i(\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`\mathbb M^1` is the :math:`H(\textnormal{curl})`-mass matrix
    and :math:`\mathbb S` is a stabilization matrix.

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    stab_eps : float
        Stabilization parameter multiplied on stab_mat (default=0.0).

    stab_mat : str
        Name of the stabilizing matrix.

    rho : StencilVector or tuple or list
        (List of) right-hand side FE coefficients of a 0-form (optional, can be set with a setter later).
        Can be either a) StencilVector or b) 2-tuple, or a list of those.
        In case b) the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    solver : dict
        Parameters for the iterative solver (see ``__init__`` for details).
    """

    @dataclass
    class Options:
        """Configuration options for :class:`PoissonAdiabaticGyrokinetic`.

        Parameters
        ----------
        stab_mat : {"M0", "M0ad", "Id"}, default="Id"
            Stabilization matrix multiplied by ``stab_eps``.

            - ``"M0"``: standard weighted 0-form mass operator.
            - ``"M0ad"``: adiabatic-electron weighted 0-form mass operator.
            - ``"Id"``: identity operator.

        which_geometry: {"cylindrical", "toroidal"}, default="cylindrical"
            Geometry of the problem, determines the meaning of `<\phi>` in the stabilization term.

        diffusion_mat : {"M1", "M1perp", "M1gyro"}, defaults="M1gyro"
            Diffusion matrix.

        rho : FEECVariable or Callable or tuple or list, default=None
            Right-hand side source term(s) of the Poisson problem.
            Accepted entries are:

            - ``None``: zero source.
            - ``FEECVariable`` in ``H1``.
            - ``Callable`` to be projected to ``H1`` via ``L2Projector``.
            - ``AccumulatorVector``.
            - a ``list`` containing any mix of the entries above.

            The tuple form is accepted by typing for compatibility with other
            propagator interfaces that pair particle data with accumulators.

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
        which_geometry: OptsGeometry = "cylindrical"
        diffusion_mat: OptsDiffusionMat = "M1gyro"
        rho: FEECVariable | Callable | tuple[AccumulatorVector, Particles] | list = None
        rho_coeffs: float | list = None
        x0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            # Poisson solve (-> set some params of parent class)
            self.sigma_1 = 1.0
            self.sigma_2 = 0.0
            self.sigma_3 = 1.0
            self.divide_by_dt = False

    def allocate(self):
        super().allocate()
        if self.options.which_geometry == "cylindrical":
            average_mat = AverageOperator(self.derham, "H1", 2)
            temp = self._stab_mat.copy() @ average_mat
            self._stab_mat -= temp
        elif self.options.which_geometry == "toroidal":
            average_mat1 = AverageOperator(self.derham, "H1", 1)
            average_mat2 = AverageOperator(self.derham, "H1", 2)
            temp = self._stab_mat.copy() @ average_mat1 @ average_mat2
            self._stab_mat -= temp

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new
