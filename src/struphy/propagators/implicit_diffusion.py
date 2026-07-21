import logging
from dataclasses import dataclass
from typing import Callable, Literal

import cunumpy as xp
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.solvers import inverse
from feectools.linalg.stencil import StencilVector
from line_profiler import profile

from struphy.feec.mass import L2Projector, WeightedMassOperator
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable
from struphy.pic.accumulation.particles_to_grid import AccumulatorVector, ParticlesToGrid
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option
from struphy.pic.accumulation.filter import FilterParameters

logger = logging.getLogger("struphy")

opts_feec_space = LiteralOptions.OptsFEECSpace


class ImplicitDiffusion(Propagator):
    r"""
    Weak, implicit discretization of the diffusion (or heat) equation (can be used as a Poisson solver too).

    Find :math:`\phi \in H^1` such that

    .. math::

        \int_\Omega \psi\, n_0(\mathbf x)\frac{\partial \phi}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \psi^\top D_0(\mathbf x) \nabla \phi \,\textrm d \mathbf x = \sum_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`n_0, \rho_i:\Omega \to \mathbb R` are real-valued functions and
    :math:`D_0:\Omega \to \mathbb R^{3\times 3}`
    is a positive diffusion matrix.
    Boundary terms from integration by parts are assumed to vanish.
    The equation is discretized as

    .. math::

        \left( \frac{\sigma_1}{\Delta t} \mathbb M^0_{n_0} + \mathbb G^\top \mathbb M^1_{D_0} \mathbb G \right)\, \boldsymbol\phi^{n+1} = \frac{\sigma_2}{\Delta t} \mathbb M^0_{n_0} \boldsymbol\phi^{n} + \frac{\sigma_3}{\Delta t} \sum_i(\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`M^0_{n_0}` and :math:`M^1_{D_0}` are :class:`WeightedMassOperators <struphy.feec.mass.WeightedMassOperators>`
    and :math:`\sigma_1, \sigma_2, \sigma_3 \in \mathbb R` are artificial parameters that can be tuned to
    change the model (see Notes).

    Notes
    -----

    * :math:`\sigma_1=\sigma_2=0` and :math:`\sigma_3 = \Delta t`: **Poisson solver** with a given charge density :math:`\sum_i\rho_i`.
    * :math:`\sigma_2=0` and :math:`\sigma_1 = \sigma_3 = \Delta t` : Poisson with **adiabatic electrons**.
    * :math:`\sigma_1=\sigma_2=1` and :math:`\sigma_3 = 0`: **Implicit heat equation**.
    """

    class Variables:
        """
        Attributes
        ----------
        phi : FEECVariable
            Scalar solution field in H^1 space.
        """

        def __init__(self):
            self._phi: FEECVariable = None

        @property
        def phi(self) -> FEECVariable:
            return self._phi

        @phi.setter
        def phi(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._phi = new

    def __init__(
        self,
        rho: FEECVariable | ParticlesToGrid | Callable | list = None,
        rho_coeffs: float | list = None,
        diagnostic: FEECVariable | None = None,
    ):
        """
        Parameters
        ----------
        rho : FEECVariable or Callable or ParticlesToGrid or list, default=None
            Source term(s) on the right-hand side.
            Accepted entries are:

            - ``None``: zero source.
            - ``FEECVariable`` in ``H1``.
            - ``Callable`` to be projected to ``H1`` via ``L2Projector``.
            - :class:`~struphy.pic.accumulation.particles_to_grid.ParticlesToGrid`, describing a
              particle-to-grid (charge/current) deposition. An
              :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector` is built from
              it at ``allocate()``.
            - a ``list`` containing any mix of the entries above.

        rho_coeffs : float or list, default=None
            Multiplicative coefficient(s) for ``rho`` sources.
            If a scalar is provided, it is applied to a single source.
            If a sequence is provided, its length must match the number of
            collected sources.
            If ``None``, all coefficients default to ``1.0``.

        diagnostic : FEECVariable, default=None
            If not None, updates at each call to the propagator, takes the value of the right-hand side.
            Otherwise does not provide diagnostic.
        """
        if isinstance(rho, list):
            for r in rho:
                if isinstance(r, ParticlesToGrid):
                    assert r.accum_space is not None, "If rho contains a PICVariable, accum_space must be provided."
                    assert r.accum_kernel is not None, "If rho contains a PICVariable, accum_kernel must be provided."
        elif isinstance(rho, ParticlesToGrid):
            assert rho.accum_space is not None, "If rho is a PICVariable, accum_space must be provided."
            assert rho.accum_kernel is not None, "If rho is a PICVariable, accum_kernel must be provided."
        
        self.variables = self.Variables()
        self.rho = rho
        self.rho_coeffs = rho_coeffs
        self.diagnostic = diagnostic
            
    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`ImplicitDiffusion`.

        Parameters
        ----------
        sigma_1 : float, default=1.0
            Coefficient multiplying the stabilization/mass contribution on the
            left-hand side.

            With ``divide_by_dt=True`` it is interpreted as ``sigma_1 / dt`` in
            the assembled linear system.

        sigma_2 : float, default=0.0
            Coefficient multiplying the previous solution contribution
            ``stab_mat * phi^n`` on the right-hand side.

            With ``divide_by_dt=True`` it is interpreted as ``sigma_2 / dt``.

        sigma_3 : float, default=1.0
            Coefficient multiplying the source terms ``rho`` on the
            right-hand side.

            With ``divide_by_dt=True`` it is interpreted as ``sigma_3 / dt``.

        divide_by_dt : bool, default=False
            If ``True``, divide ``sigma_1``, ``sigma_2`` and ``sigma_3`` by the
            time step passed to ``__call__(dt)`` before assembling the system.

        stab_mat : {"M0", "M0ad", "Id"}, default="M0"
            Stabilization operator used in the term weighted by ``sigma_1`` and
            ``sigma_2``.

            - ``"M0"``: standard weighted 0-form mass operator.
            - ``"M0ad"``: adiabatic-electron weighted 0-form mass operator.
            - ``"Id"``: identity operator.

        diffusion_mat : {"M1", "M1perp"} or WeightedMassOperator, default="M1"
            Diffusion metric in the bilinear form
            ``grad.T @ diffusion_mat @ grad``.
            You can pass the name of a pre-built operator in ``mass_ops`` or a
            custom ``WeightedMassOperator`` compatible with the codomain of
            ``grad``.

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
            
        filter_params : dict[PICVariable | SPHVariable, FilterParameters], default=None
            If not None, specifies a filter to the accumulation of a specific variable.
            Keyed by the ``pic_variable`` of the corresponding
            :class:`~struphy.pic.accumulation.particles_to_grid.ParticlesToGrid` source in ``rho``.

        Note
        ----
        The accumulation kernel for a particle source is not configured here:
        it is carried by the :class:`~struphy.pic.accumulation.particles_to_grid.ParticlesToGrid`
        instance passed as (part of) ``rho``.
        """

        # specific literals
        OptsStabMat = Literal["M0", "M0ad", "Id"]
        OptsDiffusionMat = Literal["M1", "M1perp"]
        # propagator options
        sigma_1: float = 1.0
        sigma_2: float = 0.0
        sigma_3: float = 1.0
        divide_by_dt: bool = False
        stab_mat: OptsStabMat = "M0"
        diffusion_mat: OptsDiffusionMat = "M1"
        x0: StencilVector = None
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: dict[PICVariable | SPHVariable, FilterParameters] = None

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
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

    @profile
    def allocate(self):
        # always stabilize
        if xp.abs(self.options.sigma_1) < 1e-14:
            self.options.sigma_1 = 1e-14
            logger.warning(f"Stabilizing Poisson solve with {self.options.sigma_1 =}")

        # model parameters
        self._sigma_1 = self.options.sigma_1
        self._sigma_2 = self.options.sigma_2
        self._sigma_3 = self.options.sigma_3
        self._divide_by_dt = self.options.divide_by_dt

        phi = self.variables.phi.spline.vector

        # collect rhs
        def verify_rhs(rho) -> StencilVector | FEECVariable | AccumulatorVector:
            """Perform preliminary operations on rho to compute the rhs and return the result."""
            
            if rho is None:
                rhs = phi.space.zeros()
                
            elif isinstance(rho, FEECVariable):
                assert rho.space == "H1"
                rhs = rho
                
            elif isinstance(rho, ParticlesToGrid):
                filter_params = None
                if self.options.filter_params is not None:
                    filter_params = self.options.filter_params.get(rho.pic_variable)

                rhs = AccumulatorVector(
                    rho.pic_variable.particles,
                    rho.accum_space,
                    rho.accum_kernel,
                    Propagator.mass_ops,
                    Propagator.domain.args_domain,
                    filter_params=filter_params,
                )
                
                if not rhs.particles.control_variate and rhs.space_id == "H1":
                    l2_proj = L2Projector("H1", Propagator.mass_ops)
                    f0e = -rho.pic_variable.species.charge_number * rhs.particles.f0
                    rho_eh = FEECVariable(space="H1")
                    rho_eh.allocate(derham=Propagator.derham, domain=Propagator.domain)
                    rho_eh.spline.vector = l2_proj.get_dofs(f0e.n)
                    return [rhs, rho_eh]
                
            elif isinstance(rho, Callable):
                rhs = L2Projector("H1", self.mass_ops).get_dofs(rho, apply_bc=True)
            
            else:
                raise TypeError(f"{type(rho) =} is not accepted.")

            return [rhs]

        rho = self.rho
        if isinstance(rho, list):
            self._sources = []
            for n, r in enumerate(rho):
                tmp = verify_rhs(r)
                if len(tmp) == 2 and self.rho_coeffs is not None:
                    if len(self.rho_coeffs) < len(rho):
                        self.rho_coeffs.insert(n + 1, self.rho_coeffs[n])
                self._sources += tmp
        else:
            tmp = verify_rhs(rho)
            if len(tmp) == 2 and self.rho_coeffs is not None:
                if not isinstance(self.rho_coeffs, (list, tuple)):
                    self.rho_coeffs = [self.rho_coeffs, self.rho_coeffs]
            self._sources = tmp

        # coeffs of rhs
        if self.rho_coeffs is not None:
            if isinstance(self.rho_coeffs, (list, tuple)):
                self._coeffs = self.rho_coeffs
            else:
                self._coeffs = [self.rho_coeffs]
            assert len(self._coeffs) == len(self._sources)
        else:
            self._coeffs = [1.0 for src in self.sources]

        # initial guess and solver params
        self._x0 = self.options.x0
        self._info = self.options.solver_params.info

        if self.options.stab_mat == "Id":
            stab_mat = IdentityOperator(phi.space)
        else:
            stab_mat = getattr(self.mass_ops, self.options.stab_mat)

        if isinstance(self.options.diffusion_mat, str):
            diffusion_mat = getattr(self.mass_ops, self.options.diffusion_mat)
        else:
            diffusion_mat = self.options.diffusion_mat
            assert isinstance(diffusion_mat, WeightedMassOperator)
            assert diffusion_mat.domain == self.derham.grad.codomain
            assert diffusion_mat.codomain == self.derham.grad.codomain

        # Set lhs matrices (without dt)
        self._stab_mat = stab_mat
        self._diffusion_op = self.derham.grad.T @ diffusion_mat @ self.derham.grad

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
        self._tmp = phi.space.zeros()
        self._rhs = phi.space.zeros()
        self._rhs2 = phi.space.zeros()
        self._tmp_src = phi.space.zeros()

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
        assert value.space == self.derham.V0
        assert value.space.symbolic_space == "H1", (
            f"Right-hand side must be in H1, but is in {value.space.symbolic_space}."
        )

        if self.options.x0 is None:
            self.options.x0 = value
        else:
            self.options.x0[:] = value[:]

    @profile
    def __call__(self, dt):
        # set parameters
        if self._divide_by_dt:
            sig_1 = self._sigma_1 / dt
            sig_2 = self._sigma_2 / dt
            sig_3 = self._sigma_3 / dt
        else:
            sig_1 = self._sigma_1
            sig_2 = self._sigma_2
            sig_3 = self._sigma_3

        # compute rhs
        phin = self.variables.phi.spline.vector
        rhs = self._stab_mat.dot(phin, out=self._rhs)
        rhs *= sig_2

        self._rhs2 *= 0.0
        for src, coeff in zip(self.sources, self.coeffs):
            if isinstance(src, StencilVector):
                self._rhs2 += sig_3 * coeff * src
            elif isinstance(src, FEECVariable):
                v = src.spline.vector
                self._rhs2 += sig_3 * coeff * self.mass_ops.M0.dot(v, out=self._tmp_src)
            elif isinstance(src, AccumulatorVector):
                if src.particles.control_variate:
                    src.particles.update_weights()
                src()  # accumulate
                self._rhs2 += sig_3 * coeff * src.vectors[0]

        rhs += self._rhs2

        if self.diagnostic is not None:
            proj = L2Projector("H1", self.mass_ops)
            self.diagnostic.spline.vector = proj.solve(rhs)

        # compute lhs
        self._solver.linop = sig_1 * self._stab_mat + self._diffusion_op

        # solve
        out = self._solver.solve(rhs, out=self._tmp)
        info = self._solver._info

        if self._info:
            logger.info(info)

        self.update_feec_variables(phi=out)
