import logging
from dataclasses import dataclass
from typing import Literal

from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class MaxwellWeakAmpere(Propagator):
    r"""FEEC discretization of the following equations:
    find :math:`\mathbf E \in H(\textnormal{curl})` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \int_{\Omega} \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \, \textrm d \mathbf x - \int_{\Omega} \mathbf B \cdot \nabla \times \mathbf F \,\textrm d \mathbf x &= 0\,, \qquad \forall \, \mathbf F \in H(\textnormal{curl})
        \\[2mm]
        \frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E &= 0\,.

    Ampère's law holds weakly, Faraday's law strongly. With the FE coefficients
    :math:`\mathbf e \in \mathbb R^{N_1}` of :math:`\mathbf E \in V^1_h` and :math:`\mathbf b \in \mathbb R^{N_2}`
    of :math:`\mathbf B \in V^2_h`, the semi-discrete system reads

    .. math::

        \mathbb M_1 \dot{\mathbf e} = \mathbb C^\top \mathbb M_2 \mathbf b\,, \qquad
        \dot{\mathbf b} = - \mathbb C \mathbf e\,,

    with the mass matrices :math:`\mathbb M_1`, :math:`\mathbb M_2`
    (:class:`~struphy.feec.mass.WeightedMassOperators`) and the discrete curl :math:`\mathbb C`
    (:attr:`~struphy.feec.psydac_derham.Derham.curl`). It conserves the energy
    :math:`\frac 12 \mathbf e^\top \mathbb M_1 \mathbf e + \frac 12 \mathbf b^\top \mathbb M_2 \mathbf b`
    and the discrete divergence :math:`\mathbb D \mathbf b` (since :math:`\mathbb D \mathbb C = 0`).

    :ref:`time_discret`:

    - implicit: Crank-Nicolson (implicit mid-point),

      .. math::

          \mathbb M_1 \frac{\mathbf e^{n+1} - \mathbf e^n}{\Delta t} = \mathbb C^\top \mathbb M_2 \frac{\mathbf b^{n+1} + \mathbf b^n}{2}\,,
          \qquad
          \frac{\mathbf b^{n+1} - \mathbf b^n}{\Delta t} = - \mathbb C \frac{\mathbf e^{n+1} + \mathbf e^n}{2}\,,

      which conserves both invariants exactly (up to the solver tolerance). The magnetic field is eliminated by
      :class:`~struphy.linear_algebra.schur_solver.SchurSolver` with :math:`x = \mathbf e`, :math:`y = \mathbf b`,
      :math:`A = \mathbb M_1`, :math:`B = -\frac 12 \mathbb C^\top \mathbb M_2` and :math:`C = \frac 12 \mathbb C`,
      which gives the curl-curl system

      .. math::

          \left(\mathbb M_1 + \frac{\Delta t^2}{4} \mathbb C^\top \mathbb M_2 \mathbb C\right) \mathbf e^{n+1}
          = \left(\mathbb M_1 - \frac{\Delta t^2}{4} \mathbb C^\top \mathbb M_2 \mathbb C\right) \mathbf e^n
          + \Delta t \, \mathbb C^\top \mathbb M_2 \mathbf b^n\,,

      solved with ``solver`` and ``precond``. By default (``precond="StiffnessPreconditioner"``) the Schur operator
      is preconditioned by :class:`~struphy.feec.preconditioner.StiffnessPreconditioner` for
      :math:`\mathbb C^\top \left(\frac{\Delta t^2}{4} \mathbb M_2\right) \mathbb C + \mathbb M_1`, built for the
      current time step (:meth:`~struphy.propagators.maxwell_weak_ampere.MaxwellWeakAmpere.prepare_time_step`). Then
      :math:`\mathbf b^{n+1} = \mathbf b^n - \frac{\Delta t}{2} \mathbb C (\mathbf e^{n+1} + \mathbf e^n)`.

    - explicit: Runge-Kutta method from :class:`~struphy.ode.utils.ButcherTableau` for the system
      :math:`\dot{\mathbf e} = \mathbb M_1^{-1} \mathbb C^\top \mathbb M_2 \mathbf b`,
      :math:`\dot{\mathbf b} = - \mathbb C \mathbf e`, where :math:`\mathbb M_1^{-1}` is applied with ``solver`` and
      ``precond`` in each stage (``"StiffnessPreconditioner"`` is replaced by the mass matrix preconditioner of
      :math:`\mathbb M_1` here). The time step is limited by the CFL condition of light waves.
    """

    class Variables:
        """Container for Maxwell equation variables.

        Attributes
        ----------
        e : FEECVariable
            Electric field in H(curl) space.
        b : FEECVariable
            Magnetic field in H(div) space.
        """

        def __init__(self):
            self._e: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`~struphy.propagators.maxwell_weak_ampere.MaxwellWeakAmpere`.

        Parameters
        ----------
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping scheme to use for the Maxwell equations.

            - ``"implicit"``: Crank-Nicolson (implicit mid-point) scheme.
            - ``"explicit"``: explicit Runge-Kutta methods from ``ButcherTableau``.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Name of the symmetric iterative solver passed to
            ``feectools.linalg.solvers.inverse``.

        precond : LiteralOptions.OptsMaxwellPrecond, default="StiffnessPreconditioner"
            Preconditioner of the linear solves.

            - ``"StiffnessPreconditioner"``: Kronecker (fast diagonalization) approximation of the inverse of the
              Schur operator :math:`\mathbb M_1 + \frac{\Delta t^2}{4} \mathbb C^\top \mathbb M_2 \mathbb C`
              (:class:`~struphy.feec.preconditioner.StiffnessPreconditioner` with ``derivative="curl"``; with polar
              splines a :class:`~struphy.feec.preconditioner.PolarStiffnessPreconditioner`), rebuilt when the time
              step changes. For ``algo="explicit"``, the mass matrix preconditioner of :math:`\mathbb M_1` is used
              instead. (With polar splines, the implicit step currently fails in the
              :class:`~struphy.linear_algebra.schur_solver.SchurSolver` with any preconditioner.)
            - ``"MassMatrixPreconditioner"``, ``"PolarMassMatrixPreconditioner"``: preconditioner of
              :math:`\mathbb M_1`, better for small time steps :math:`\Delta t \lesssim h`, e.g. in CFL-limited
              PIC runs.
            - ``None``: no preconditioner (set automatically, with a warning, for ``solver="cg"``, which takes no
              preconditioner).

        solver_params : SolverParameters, default=None
            Iterative-solver controls (for example ``tol``, ``maxiter``,
            ``verbose``, ``info``, ``recycle``).
            If ``None``, defaults to ``SolverParameters()``.

        butcher : ButcherTableau, default=None
            Butcher tableau for explicit Runge-Kutta methods.
            Only used when ``algo="explicit"``.
            If ``None``, defaults to ``ButcherTableau()``.

        Notes
        -----
        System size reduction is performed via :class:`~struphy.linear_algebra.schur_solver.SchurSolver` for
        implicit time stepping to eliminate the magnetic field and reduce
        the system to the electric field equation.
        """

        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        algo: OptsAlgo = "implicit"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMaxwellPrecond = "StiffnessPreconditioner"
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMaxwellPrecond)
            if self.solver == "cg" and self.precond is not None:
                logger.warning(f'solver="cg" takes no preconditioner; ignoring precond="{self.precond}".')
                self.precond = None

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.algo == "explicit" and self.butcher is None:
                self.butcher = ButcherTableau()

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
        # obtain needed matrices
        M1 = self.mass_ops.M1
        M2 = self.mass_ops.M2
        curl = self.derham.curl

        # Preconditioner (the one of the Schur operator depends on dt and is built in prepare_time_step)
        self._precond = self.options.precond
        self._schur_precond_dt = None
        if self._precond == "StiffnessPreconditioner" and self.options.algo == "explicit":
            self._precond = "PolarMassMatrixPreconditioner" if self.derham.polar_splines else "MassMatrixPreconditioner"
        if self._precond in (None, "StiffnessPreconditioner"):
            pc = None
        else:
            pc_class = getattr(preconditioner, self._precond)
            pc = pc_class(M1)

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info
            # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
            _A = M1

            # no dt
            self._B = -1 / 2 * curl.T @ M2
            self._C = 1 / 2 * curl

            # Instantiate Schur solver (constant in this case)
            _BC = self._B @ self._C

            self._schur_solver = SchurSolver(
                _A,
                _BC,
                self.options.solver,
                precond=pc,
                solver_params=self.options.solver_params,
            )

            # pre-allocate arrays
            self._byn = self._B.codomain.zeros()
        else:
            self._info = False

            # define vector field
            M1_inv = inverse(
                M1,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            weak_curl = M1_inv @ curl.T @ M2

            # allocate output of vector field
            out1 = self.variables.e.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                weak_curl.dot(y2, out=out)
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                curl.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.e.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

        # allocate place-holder vectors to avoid temporary array allocations in __call__
        self._e_tmp1 = self.variables.e.spline.vector.space.zeros()
        self._e_tmp2 = self.variables.e.spline.vector.space.zeros()
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

    def prepare_time_step(self, dt: float):
        """Stiffness preconditioner of the Schur operator M1 + dt^2/4 C^T M2 C for the time step ``dt``: built in the
        setup, rebuilt when dt changes."""
        if self._precond == "StiffnessPreconditioner" and dt != self._schur_precond_dt:
            self._schur_solver.precond = preconditioner.StiffnessPreconditioner(
                self.mass_ops, "curl", sigma=1.0, mid=(dt**2 / 4) * self.mass_ops.M2, stab="M1"
            )
            self._schur_precond_dt = dt

    @profile
    def __call__(self, dt):
        # current FE coeffs
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        if self.options.algo == "implicit":
            # preconditioner for the current time step (no-op unless dt changed since the setup)
            self.prepare_time_step(dt)

            # solve for new e coeffs
            self._B.dot(bn, out=self._byn)

            en1, info = self._schur_solver(en, self._byn, dt, out=self._e_tmp1)

            # new b coeffs
            _e = en.copy(out=self._e_tmp2)
            _e += en1
            bn1 = self._C.dot(_e, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(e=en1, b=bn1)
        else:
            self._ode_solver(0.0, dt)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                logger.info(f"Status     for Maxwell: {info['success']}")
                logger.info(f"Iterations for Maxwell: {info['niter']}")
                logger.info(f"Maxdiff e for Maxwell: {diffs['e']}")
                logger.info(f"Maxdiff b for Maxwell: {diffs['b']}")
                logger.info("")
