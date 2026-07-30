import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilVector
from line_profiler import profile

from struphy.feec.mass import L2Projector, WeightedMassOperator
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector
from struphy.pic.base import Particles
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class CurlCurlSolve(Propagator):
    r"""
    Weak discretization of the curl-curl problem.

    Find :math:`\mathbf E \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \nabla \times \mathbf F \cdot \nabla \times \mathbf E\,\textrm d \mathbf x - \sigma \int_\Omega \mathbf F \cdot \mathbf E\,\textrm d \mathbf x = \sum_i \int_\Omega \mathbf F \cdot \mathbf J _i\,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,

    where :math:`\mathbf J _i:\Omega \to \mathbb R^3` are real-valued and
    :math:`\sigma \in \mathbb R / \{0\}` is a scalar.
    Boundary terms from integration by parts are assumed to vanish.
    The equation is discretized as

    .. math::

        \left( \mathbb C^\top \cdot \mathbb M^2 \cdot \mathbb C - \sigma \mathbb M^1 \right)\, \boldsymbol e^{n+1} =\sum_i \boldsymbol j_i\,,

    where :math:`\mathbb M^1` and :math:`\mathbb M^2` are :class:`WeightedMassOperators <struphy.feec.mass.WeightedMassOperators>` and :math:`\boldsymbol j_i`
    is the vector of coefficients of the projection of :math:`\mathbf J_i` into the discrete space :math:`V^1_h\subset H(\textnormal{curl})`.
    """

    class Variables:
        """
        Attributes
        ----------
        e : FEECVariable
            Vector-valued solution in ``"Hcurl"`` space.
        """

        def __init__(self):
            self._e: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

    def __init__(
        self,
        j: FEECVariable | tuple[Callable, Callable, Callable] | tuple[AccumulatorVector, Particles] | list = None,
        j_coeffs: float | list = None,
    ):
        """
        Parameters
        ----------
        j : FEECVariable or tuple of Callables or tuple or list, default=None
            Source term(s) on the right-hand side.
            Accepted entries are:

            - ``None``: zero source.
            - ``FEECVariable`` in ``"Hcurl"``.
            - ``tuple`` of three ``Callable``s to be projected to ``"Hcurl"`` via
              ``L2Projector``.
            - ``AccumulatorVector``.
            - a ``list`` containing any mix of the entries above.

            The tuple form is accepted by typing for compatibility with other
            propagator interfaces that pair particle data with accumulators.

        j_coeffs : float or list, default=None
            Multiplicative coefficient(s) for ``j`` sources.
            If a scalar is provided, it is applied to a single source.
            If a sequence is provided, its length must match the number of
            collected sources.
            If ``None``, all coefficients default to ``1.0``.
        """
        self.variables = self.Variables()
        self.j = j
        self.j_coeffs = j_coeffs

    @dataclass
    class Options:
        """Configuration options for :class:`CurlCurlSolve`.

        Parameters
        ----------
        sigma : float, default=1.0
            Coefficient multiplying the stabilization/mass contribution on the
            left-hand side.

        stab_mat : {"M1", "Id"}, default="M1"
            Stabilization operator used in the term weighted by ``sigma``.

            - ``"M1"``: standard weighted 1-form mass operator.
            - ``"Id"``: identity operator.

        diffusion_mat : "M2" or WeightedMassOperator, default="M2"
            Diffusion metric in the bilinear form
            ``curl.T @ diffusion_mat @ curl``.
            You can pass the name of a pre-built operator in ``mass_ops`` or a
            custom ``WeightedMassOperator`` compatible with the codomain of
            ``curl``.

        x0 : StencilVector, default=None
            Initial guess for the iterative linear solver.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver passed to
            :func:`psydac.linalg.solvers.inverse`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Name of the preconditioner configuration.
            Currently this class sets ``pc=None`` internally, so this option is
            reserved for compatibility and future extensions.

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.
        """

        # specific literals
        OptsStabMat = Literal["M1", "Id"]
        OptsDiffusionMat = Literal["M2"]
        # propagator options
        sigma: float = 1.0
        stab_mat: OptsStabMat = "M1"
        diffusion_mat: OptsDiffusionMat = "M2"
        x0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.diffusion_mat, self.OptsDiffusionMat)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        self._options = new

    @profile
    def allocate(self, verbose: bool = False):
        # always stabilize
        if xp.abs(self.options.sigma) < 1e-14:
            self.options.sigma = 1e-14
            if MPI.COMM_WORLD.Get_rank() == 0:
                logger.info(f"Running Curl-Curl solve with {self.options.sigma =}")

        # model parameters
        self._sigma = self.options.sigma

        e = self.variables.e.spline.vector

        # collect rhs
        def verify_rhs(j) -> StencilVector | FEECVariable | AccumulatorVector:
            """Perform preliminary operations on j to compute the rhs and return the result."""
            if j is None:
                rhs = e.space.zeros()
            elif isinstance(j, FEECVariable):
                assert j.space == "Hcurl"
                rhs = j
            elif isinstance(j, AccumulatorVector):
                rhs = j
            elif isinstance(j, tuple[Callable, Callable, Callable]):
                assert (
                    len(
                        j,
                    )
                    == 3
                )
                rhs = L2Projector("Hcurl", self.mass_ops).get_dofs(j, apply_bc=True)
            else:
                raise TypeError(f"{type(j) =} is not accepted.")

            return rhs

        j = self.j
        if isinstance(j, list):
            self._sources = []
            for ji in j:
                self._sources += [verify_rhs(ji)]
        else:
            self._sources = [verify_rhs(j)]

        # coeffs of rhs
        if self.j_coeffs is not None:
            if isinstance(self.j_coeffs, (list, tuple)):
                self._coeffs = self.j_coeffs
            else:
                self._coeffs = [self.j_coeffs]
            assert len(self._coeffs) == len(self._sources)
        else:
            self._coeffs = [1.0 for src in self.sources]

        # initial guess and solver params
        self._x0 = self.options.x0
        self._info = self.options.solver_params.info

        if self.options.stab_mat == "Id":
            stab_mat = IdentityOperator(e.space)
        else:
            stab_mat = getattr(self.mass_ops, self.options.stab_mat)

        if isinstance(self.options.diffusion_mat, str):
            diffusion_mat = getattr(self.mass_ops, self.options.diffusion_mat)
        else:
            diffusion_mat = self.options.diffusion_mat
            assert isinstance(diffusion_mat, WeightedMassOperator)
            assert diffusion_mat.domain == self.derham.curl.codomain
            assert diffusion_mat.codomain == self.derham.curl.codomain

        # Set lhs matrices (without dt)
        self._stab_mat = stab_mat
        self._diffusion_op = self.derham.curl.T @ diffusion_mat @ self.derham.curl

        # preconditioner and solver for Ax=b
        if self.options.precond is None:
            pc = None
        else:
            # TODO: waiting for multigrid preconditioner
            pc = None

        # solver just with A_2, but will be set during call with dt
        self._solver = inverse(
            self._diffusion_op,
            self.options.solver,
            pc=pc,
            x0=self.x0,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
            recycle=self.options.solver_params.recycle,
        )

        # allocate memory for solution
        self._tmp = e.space.zeros()
        self._rhs = e.space.zeros()
        self._tmp_src = e.space.zeros()

    @property
    def sources(self) -> list[StencilVector | FEECVariable | AccumulatorVector]:
        """
        Right-hand side of the equation (sources).
        """
        return self._sources

    @property
    def coeffs(self) -> list[float]:
        """
        Same length as self.sources. Coefficients multiplied with sources before solve (default is 1.0).
        """
        return self._coeffs

    @property
    def x0(self):
        """
        feectools.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First guess of the iterative solver.
        """
        return self.options.x0

    @x0.setter
    def x0(self, value: StencilVector):
        """In-place setter for StencilVector/PolarVector. First guess of the iterative solver."""
        assert value.space == self.derham.V1
        assert value.space.symbolic_space == "Hcurl", (
            f"Right-hand side must be in Hcurl, but is in {value.space.symbolic_space}."
        )

        if self.options.x0 is None:
            self.options.x0 = value
        else:
            self.options.x0[:] = value[:]

    @profile
    def __call__(self, dt):
        # compute rhs
        self._rhs *= 0.0
        for src, coeff in zip(self.sources, self.coeffs):
            if isinstance(src, StencilVector):
                self._rhs += coeff * src
            elif isinstance(src, FEECVariable):
                v = src.spline.vector
                self._rhs += coeff * self.mass_ops.M1.dot(v, out=self._tmp_src)
            elif isinstance(src, AccumulatorVector):
                src()  # accumulate
                self._rhs += coeff * src.vectors[0]

        # compute lhs
        self._solver.linop = self._diffusion_op - self._sigma * self._stab_mat

        # solve
        out = self._solver.solve(self._rhs, out=self._tmp)
        info = self._solver._info

        if self._info:
            logger.info(info)

        self.update_feec_variables(e=out)
