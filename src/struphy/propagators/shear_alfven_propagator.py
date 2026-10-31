import logging
from dataclasses import dataclass
from typing import Literal

from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.density_weighted import VelocityMass
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.multigrid.preconditioner import MultiGridOptions, MultiGridPreconditioner
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ShearAlfvenPropagator(Propagator):
    r"""Shear-Alfvén step of linear MHD (zero-flow equilibrium) with the magnetic field in :math:`H(\textnormal{div})`.

    **Continuous problem.** Find :math:`\tilde{\mathbf{U}} \in V_U` with
    :math:`V_U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` (``Options.u_space``) and
    :math:`\tilde{\mathbf{B}} \in H(\textnormal{div})` such that

    .. math::

        \int_{\Omega} \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x &= \int_{\Omega} \tilde{\mathbf{B}} \cdot \nabla \times (\mathbf{B}_0 \times \mathbf V) \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in V_U\,,
        \\[2mm]
        \frac{\partial \tilde{\mathbf{B}}}{\partial t} &= \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)\,.

    The momentum equation is weak (integration by parts of the Lorentz force), the induction equation holds strongly
    in :math:`H(\textnormal{div})`. Essential boundary conditions impose :math:`\tilde{\mathbf B} \cdot \mathbf n = 0`
    (and the corresponding condition on :math:`\tilde{\mathbf U}`) strongly.

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^\alpha_h`
    (:math:`\alpha = 1, 2, v` for ``u_space`` ``"Hcurl"``, ``"Hdiv"``, ``"H1vec"``) and :math:`\mathbf b` of
    :math:`\tilde{\mathbf B}_h \in V^2_h`,

    .. math::

        \mathbb M^\alpha_{\rho_0} \dot{\mathbf u} = (\mathbb T^\alpha)^\top \mathbb C^\top \mathbb M^2 \mathbf b\,, \qquad
        \dot{\mathbf b} = - \mathbb C \mathbb T^\alpha \mathbf u\,,

    with the discrete curl :math:`\mathbb C` (:attr:`~struphy.feec.psydac_derham.Derham.curl`), the mass matrix
    :math:`\mathbb M^2` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2`), the density-weighted mass matrix
    :math:`\mathbb M^\alpha_{\rho_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M1n`,
    :attr:`~struphy.feec.mass.WeightedMassOperators.M2n`, :attr:`~struphy.feec.mass.WeightedMassOperators.Mvn`)
    and the basis projection operator :math:`\mathbb T^\alpha \approx \hat \Pi^1[\mathbf B_0 \times \,\cdot\,]`
    (:attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.T1`,
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.T2`,
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.Tv`). The induction equation holds exactly in
    :math:`V^2_h`, so the discrete divergence of :math:`\mathbf b` stays constant, and the system conserves the
    perturbed energy :math:`\frac 12 \mathbf u^\top \mathbb M^\alpha_{\rho_0} \mathbf u + \frac 12 \mathbf b^\top \mathbb M^2 \mathbf b`.

    **Time discretization.** Selected by ``Options.algo``:

    - ``algo="implicit"`` (default): Crank-Nicolson (implicit mid-point), which conserves the perturbed energy exactly
      (up to the solver tolerances). The magnetic field is eliminated with
      :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (:math:`x = \mathbf u`, :math:`y = \mathbf b`), with
      the Schur complement

      .. math::

          S = \mathbb M^\alpha_{\rho_0} + \frac{\Delta t^2}{4} (\mathbb T^\alpha)^\top \mathbb C^\top \mathbb M^2 \mathbb C \mathbb T^\alpha\,,

      solved by ``Options.solver`` with the preconditioner ``Options.precond`` (by default the Kronecker
      :class:`~struphy.feec.preconditioner.ShearAlfvenSchurPreconditioner`; also multigrid or mass matrix
      preconditioners, see :class:`Options`); then
      :math:`\mathbf b^{n+1} = \mathbf b^n - \frac{\Delta t}{2} \mathbb C \mathbb T^\alpha (\mathbf u^{n+1} + \mathbf u^n)`.
    - ``algo="explicit"``: explicit Runge-Kutta method of the semi-discrete system, with the tableau
      ``Options.butcher`` (:class:`~struphy.ode.utils.ButcherTableau`); each stage inverts
      :math:`\mathbb M^\alpha_{\rho_0}` with ``Options.solver``.

    **Perturbative models** (constructor argument ``rho_field``, the density perturbation :math:`\tilde \rho`, only
    with ``u_space="Hdiv"``): the inertia is that of the total density, :math:`\mathbb M^2_{\rho_0} \to \mathbb
    M^2_\rho`, :math:`\rho = \rho_0 + \tilde \rho^n` (:class:`~struphy.feec.density_weighted.VelocityMass`, assembled
    at :math:`t^n`; the density does not change in this step, so the energy is still conserved exactly). The
    preconditioners are still built from :math:`\mathbb M^2_{\rho_0}`.
    """

    class Variables:
        """Container for variables advanced by :class:`ShearAlfvenPropagator`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        b : FEECVariable
            Magnetic-field variable in ``"Hdiv"`` space.
        """

        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self, rho_field: FEECVariable = None):
        r"""
        Parameters
        ----------
        rho_field : FEECVariable, default=None
            Density perturbation in ``"L2"`` space (not updated): the inertia is that of the total density
            :math:`\rho_0 + \tilde \rho` (perturbative models, only with ``Options.u_space="Hdiv"``). If ``None``, the
            inertia is that of :math:`\rho_0`.
        """
        assert rho_field is None or rho_field.space == "L2", "rho_field must be in L2."
        self.variables = self.Variables()
        self.rho_field = rho_field

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`ShearAlfvenPropagator`.

        Parameters
        ----------
        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the velocity variable.
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping algorithm.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used by implicit or explicit operators.
        precond : LiteralOptions.OptsShearAlfvenPrecond, default="StiffnessPreconditioner"
            Preconditioner of the linear solve: of the mass-matrix block (``"MassMatrixPreconditioner"``,
            ``"PolarMassMatrixPreconditioner"``), or of the whole Schur operator
            :math:`\mathbb M^2_{\rho_0} + \frac{\Delta t^2}{4} \mathbb T^\top \mathbb C^\top \mathbb M^2 \mathbb C \mathbb T`:
            a geometric multigrid V-cycle (``"MultiGrid"``,
            :class:`~struphy.linear_algebra.multigrid.preconditioner.MultiGridPreconditioner`) or the Kronecker
            (fast diagonalization) preconditioner (``"StiffnessPreconditioner"``, the default,
            :class:`~struphy.feec.preconditioner.ShearAlfvenSchurPreconditioner`: block Jacobi in the velocity
            components with the auxiliary space correction of the :math:`\mathbf E \times \mathbf B` flows if
            ``exb_correction``, see there). Both need the implicit algorithm with ``solver="pcg"`` or ``"pcg_sr"``. The Kronecker
            preconditioner also needs ``u_space="Hdiv"``, tensor-product splines (no polar splines) and the global
            commuting projectors; otherwise ``"StiffnessPreconditioner"`` falls back (with a warning) to
            ``"MassMatrixPreconditioner"`` (``"PolarMassMatrixPreconditioner"`` with polar splines). The Schur operator
            depends on the time step, so these preconditioners are built in the setup (:meth:`prepare_time_step`) and
            updated if the time step changes.
            The multigrid preconditioner is experimental: on the TAE example (HollowTorus, 8x32x4) it needs more PCG
            iterations than the mass matrix preconditioner and is much slower (the smoother does not treat the large
            kernel of the curl term on 2-forms). The Kronecker preconditioner (with ``exb_correction``) needs about 11
            instead of 31 PCG iterations per step there (dt = 0.1), with larger gains for larger time steps.
        exb_correction : bool, default=True
            For ``precond="StiffnessPreconditioner"``: add the auxiliary space correction
            :math:`\mathbb N P_\phi \mathbb N^\top` (:class:`~struphy.feec.preconditioner.AuxiliarySpacePreconditioner`)
            of the :math:`\mathbf E \times \mathbf B` flows :math:`\mathbf B_0 \times \nabla \phi`,
            :math:`\mathbb N = \hat \Pi^2[\mathbf B_0 \times \,\cdot\,] \, \mathbb G` with :math:`\phi \in V^0_h`. The block
            Jacobi preconditioner in the velocity components does not reduce the incompressible shear Alfven modes
            with small parallel wave number (their velocity components cancel in the compression), which this
            correction resolves; :math:`P_\phi` is the Kronecker (fast diagonalization) preconditioner of
            :math:`\mathbb N^\top S \mathbb N` on :math:`V^0_h`.
        multigrid : MultiGridOptions, default=None
            Options of the multigrid preconditioner (if ``precond="MultiGrid"``); defaults to
            ``MultiGridOptions()``.
        solver_params : SolverParameters, default=None
            Solver controls; defaults to ``SolverParameters()``.
        butcher : ButcherTableau, default=None
            Explicit Runge-Kutta tableau when ``algo="explicit"``.
        """

        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        algo: OptsAlgo = "implicit"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsShearAlfvenPrecond = "StiffnessPreconditioner"
        exb_correction: bool = True
        multigrid: MultiGridOptions = None
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsShearAlfvenPrecond)
            if self.precond == "MultiGrid":
                assert self.algo == "implicit", f'precond="{self.precond}" requires algo="implicit".'
                assert self.solver in ("pcg", "pcg_sr"), f'precond="{self.precond}" requires solver="pcg" or "pcg_sr".'
            if self.precond == "MultiGrid" and self.multigrid is None:
                self.multigrid = MultiGridOptions()

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
        u_space = self.options.u_space

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = "M" + self.derham.space_to_form[u_space] + "n"
        id_T = "T" + self.derham.space_to_form[u_space]

        # call operators (with rho_field: the mass matrix of the total density, assembled in __call__)
        assert self.rho_field is None or u_space == "Hdiv", 'rho_field needs Options(u_space="Hdiv").'
        self._velocity_mass = (
            VelocityMass(
                self.derham,
                self.domain,
                self.mass_ops,
                self.projected_equil if self.rho_field is not None else None,
                rho_field=self.rho_field,
            )
            if u_space == "Hdiv"
            else None
        )
        _A = getattr(self.mass_ops, id_M) if self._velocity_mass is None else self._velocity_mass.M
        _T = getattr(self.basis_ops, id_T)
        _M2 = self.mass_ops.M2
        curl = self.derham.curl

        # Preconditioner (the preconditioners of the Schur operator depend on dt and are built in prepare_time_step)
        self._id_M = id_M
        self._multigrid_dt = None
        self._schur_precond_dt = None
        # the requested preconditioner, or the fallback if the Kronecker one is not possible here
        self._precond = preconditioner.ShearAlfvenSchurPreconditioner.resolve(
            self.options.precond, self.derham, self.options.algo, self.options.solver, self.options.u_space
        )
        if self._precond == "StiffnessPreconditioner":
            # Kronecker operators (built once, independent of dt)
            self._schur_precond = preconditioner.ShearAlfvenSchurPreconditioner(
                self.derham, self.mass_ops, self.basis_ops, _T, id_M, exb_correction=self.options.exb_correction
            )
        if self._precond in (None, "MultiGrid", "StiffnessPreconditioner"):
            pc = None
        else:
            pc_class = getattr(preconditioner, self._precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info

            self._B = -1 / 2 * _T.T @ curl.T @ _M2
            self._C = 1 / 2 * curl @ _T

            # instantiate Schur solver (constant in this case)
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
            A_inv = inverse(
                _A,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            _f1 = A_inv @ _T.T @ curl.T @ _M2
            _f2 = curl @ _T

            # allocate output of vector field
            out1 = self.variables.u.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                _f1.dot(y2, out=out)
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                _f2.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.u.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
        self._u_tmp2 = self.variables.u.spline.vector.space.zeros()
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

    def prepare_time_step(self, dt: float):
        """Preconditioner of the Schur operator for the time step ``dt`` (multigrid or Kronecker): built in the setup,
        updated or rebuilt when dt changes."""
        if self.options.algo != "implicit":
            return
        # the multigrid preconditioner of the Schur operator A - dt^2 B C: updated (re-using the coarse mass and
        # derivative operators) when dt changes
        if self._precond == "MultiGrid" and dt != self._multigrid_dt:
            schur = self._schur_solver.A - dt**2 * self._schur_solver.BC
            if self._schur_solver.precond is None:
                self._schur_solver.precond = MultiGridPreconditioner(
                    schur, self.derham, self.domain, self.options.multigrid, mass_ops=self.mass_ops
                )
            else:
                self._schur_solver.precond.update(schur)
            self._multigrid_dt = dt

        # the Kronecker preconditioner of S = M + dt^2/4 T^T C^T M2 C T (plus the E x B correction), rebuilt when dt
        # changes
        if self._precond == "StiffnessPreconditioner" and dt != self._schur_precond_dt:
            self._schur_solver.precond = self._schur_precond.build(dt)
            self._schur_precond_dt = dt

    @profile
    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # inertia of the total density at t^n (perturbative models)
        if self._velocity_mass is not None:
            self._velocity_mass.update()

        if self.options.algo == "implicit":
            # preconditioner for the current time step (no-op unless dt changed since the setup)
            self.prepare_time_step(dt)

            # solve for new u coeffs
            byn = self._B.dot(bn, out=self._byn)

            un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

            # new b coeffs
            _u = un.copy(out=self._u_tmp2)
            _u += self._u_tmp1
            bn1 = self._C.dot(_u, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(u=un1, b=bn1)
        else:
            self._ode_solver(0.0, dt)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                logger.info(f"Status     for ShearAlfven: {info['success']}")
                logger.info(f"Iterations for ShearAlfven: {info['niter']}")
                logger.info(f"Maxdiff up for ShearAlfven: {diffs['u']}")
                logger.info(f"Maxdiff b2 for ShearAlfven: {diffs['b']}")
                logger.info("")
