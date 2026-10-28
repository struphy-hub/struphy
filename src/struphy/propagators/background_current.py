import logging
from dataclasses import dataclass

from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class BackgroundCurrent(Propagator):
    r"""Force of the equilibrium current on the magnetic field perturbation in linear MHD (zero-flow equilibrium):
    the part of the Lorentz force that is not in the energy conserving shear-Alfvén step
    (:class:`~struphy.propagators.shear_alfven_propagator.ShearAlfvenPropagator`,
    :class:`~struphy.propagators.shear_alfven_curl_curl_propagator.ShearAlfvenCurlCurlPropagator`).

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})` such that

    .. math::

        \int_\Omega \rho_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \,\textnormal d \mathbf x
        = \int_\Omega \left[ (\nabla \times \mathbf B_0) \times \tilde{\mathbf B} \right] \cdot \mathbf V \,\textnormal d \mathbf x
        \qquad \forall \, \mathbf V \in H(\textnormal{div})\,,

    at a fixed magnetic field perturbation :math:`\tilde{\mathbf B}` (the constructor argument ``b_field``), either in
    :math:`H(\textnormal{div})` or in :math:`H(\textnormal{curl})`, with the equilibrium density :math:`\rho_0` and
    magnetic field :math:`\mathbf B_0`. This is the :math:`\rho_0`-weighted :math:`L^2` projection of
    :math:`\rho_0 \partial_t \tilde{\mathbf U} = \mathbf J_0 \times \tilde{\mathbf B}`, :math:`\mathbf J_0 = \nabla \times \mathbf B_0`.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^2_h` and
    :math:`\mathbf b` of :math:`\tilde{\mathbf B}_h \in V^\beta_h` (:math:`\beta = 2` or :math:`1`),

    .. math::

        \mathbb M^2_{\rho_0} \dot{\mathbf u} = \mathbb M^{2\beta}_{J_0} \mathbf b\,,

    with the density-weighted mass matrix :math:`\mathbb M^2_{\rho_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2n`)
    and the current mass matrix :math:`\mathbb M^{2\beta}_{J_0}` of
    :math:`\int (\mathbf J_0 \times \mathbf W) \cdot \mathbf V \,\textnormal d \mathbf x`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M2J` for :math:`\beta = 2`,
    :attr:`~struphy.feec.mass.WeightedMassOperators.M21J` for :math:`\beta = 1`).

    **Time discretization.** Exact in time, the only option, since the right-hand side does not change during the
    step:

    .. math::

        \mathbf u^{n+1} = \mathbf u^n + \Delta t \, (\mathbb M^2_{\rho_0})^{-1} \mathbb M^{2\beta}_{J_0} \mathbf b\,,

    where :math:`(\mathbb M^2_{\rho_0})^{-1}` is applied with ``Options.solver`` and ``Options.precond``.
    """

    class Variables:
        """Container for variables advanced by :class:`BackgroundCurrent`.

        Attributes
        ----------
        u : FEECVariable
            Velocity perturbation variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._u = new

    def __init__(self, b_field: FEECVariable = None):
        r"""
        Parameters
        ----------
        b_field : FEECVariable, default=None
            Magnetic field perturbation (not updated), as a 2-form (``"Hdiv"`` space) or a 1-form (``"Hcurl"`` space).
            If ``None``, an empty ``FEECVariable(space="Hdiv")`` is created and allocated internally.
            Otherwise the given variable is treated as shared (e.g. the model's ``em_fields.b_field``)
            and must already be allocated when :meth:`allocate` is called; it is not re-allocated.
        """
        assert b_field is None or b_field.space in ("Hcurl", "Hdiv"), "b_field must be in Hcurl or Hdiv."
        self.variables = self.Variables()
        self.b_field = b_field if b_field is not None else FEECVariable(space="Hdiv")
        self._allocate_b_field = b_field is None

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`BackgroundCurrent`.

        Parameters
        ----------
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Iterative solver of the ``M2n`` inversion.
        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the ``M2n`` inversion.
        solver_params : SolverParameters, default=None
            Solver controls; defaults to ``SolverParameters()``.
        """

        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
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
        self._info = self.options.solver_params.info

        if self.b_field.space == "Hcurl":
            self._MJ = self.mass_ops.M21J
        else:
            self._MJ = self.mass_ops.M2J

        if self.options.precond is None:
            pc = None
        else:
            pc = getattr(preconditioner, self.options.precond)(self.mass_ops.M2n)
        self._M2n_inv = inverse(
            self.mass_ops.M2n,
            self.options.solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

        # only allocate the private default; a shared b_field is allocated (with initial conditions) by the model
        if self._allocate_b_field:
            self.b_field.allocate(self.derham, self.domain)
        self._b = self.b_field.spline.vector

        self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
        self._u_tmp2 = self.variables.u.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        un = self.variables.u.spline.vector

        force = self._MJ.dot(self._b, out=self._u_tmp1)
        un1 = self._M2n_inv.dot(force, out=self._u_tmp2)
        un1 *= dt
        un1 += un

        diffs = self.update_feec_variables(u=un1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            info = self._M2n_inv.get_info()
            logger.info(f"Status     for BackgroundCurrent: {info['success']}")
            logger.info(f"Iterations for BackgroundCurrent: {info['niter']}")
            logger.info(f"Maxdiff u2 for BackgroundCurrent: {diffs['u']}")
            logger.info("")
