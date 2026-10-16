import logging
from dataclasses import dataclass
from typing import Literal

from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.preconditioner import HiptmairXuPreconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class ShearAlfvenCurlCurlPropagator(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde{\mathbf U} \in H(\textnormal{div})` and :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \left[(\nabla \times \tilde{\mathbf B}) \times \mathbf B_0\right] \cdot \mathbf V \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in H(\textnormal{div})\,,
        \\[2mm]
        &\int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x = - \int_\Omega \left[(\nabla \times \mathbf C) \times \mathbf B_0\right] \cdot \tilde{\mathbf{U}} \,\textrm d \mathbf x \qquad \forall \, \mathbf C \in H(\textrm{curl})\,.

    The momentum equation holds strongly up to the :math:`\rho_0`-weighted :math:`L^2` projection onto
    :math:`H(\textnormal{div})`, the induction equation is weak. In discrete form,

    .. math::

        \mathbb M^\rho_2 \dot{\mathbf u} = - \mathbb M^{2,B} \mathbb C \mathbf b\,, \qquad
        \mathbb M_1 \dot{\mathbf b} = - \mathbb C^\top \mathbb M^{2,B} \mathbf u\,,

    where :math:`\mathbb M^{2,B}` (antisymmetric) is :attr:`~struphy.feec.mass.WeightedMassOperators.M2B`, such that the
    energy :math:`\frac 12 \mathbf u^\top \mathbb M^\rho_2 \mathbf u + \frac 12 \mathbf b^\top \mathbb M_1 \mathbf b`
    is conserved. This is the discrete system of :class:`~struphy.propagators.shear_alfven_b1.ShearAlfvenB1`,
    with the velocity instead of the magnetic field eliminated in the implicit solve.

    :ref:`time_discret`:

    - implicit: Crank-Nicolson (implicit mid-point). System size reduction via
      :class:`~struphy.linear_algebra.schur_solver.SchurSolver` with :math:`x = \mathbf b`, :math:`y = \mathbf u`;
      the Schur complement is the curl-curl operator

      .. math::

          S = \mathbb M_1 + \frac{\Delta t^2}{4} \mathbb C^\top (\mathbb M^{2,B})^\top (\mathbb M^\rho_2)^{-1} \mathbb M^{2,B} \mathbb C\,,

      where :math:`(\mathbb M^\rho_2)^{-1}` is an inner iterative solve. With ``precond="HiptmairXu"``, :math:`S` is
      preconditioned by :class:`~struphy.feec.preconditioner.HiptmairXuPreconditioner` for
      :math:`\frac{\Delta t^2}{4}\left(\mathbb C^\top \mathbb M^{2,\textnormal{gyro}} \mathbb C + \frac{4}{\Delta t^2} \mathbb M_1\right)`,
      where :attr:`~struphy.feec.mass.WeightedMassOperators.M2gyro` is the mass matrix of the continuous counterpart
      :math:`\int \frac{1}{\rho_0} \left[(\nabla \times \mathbf C) \times \mathbf B_0\right] \cdot \left[(\nabla \times \tilde{\mathbf B}) \times \mathbf B_0\right] \textnormal d \mathbf x`
      of the curl-curl term.
    - explicit: explicit RK methods from ButcherTableau.
    """

    class Variables:
        """Container for variables advanced by :class:`ShearAlfvenCurlCurlPropagator`.

        Attributes
        ----------
        u : FEECVariable
            Velocity variable in ``"Hdiv"`` space.
        b : FEECVariable
            Magnetic-field variable in ``"Hcurl"`` space.
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
            assert new.space == "Hdiv"
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`ShearAlfvenCurlCurlPropagator`.

        Parameters
        ----------
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping algorithm.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver of the Schur system in ``b`` (implicit) or of ``M1`` (explicit).
        precond : LiteralOptions.OptsShearAlfvenCurlCurlPrecond, default="HiptmairXu"
            Preconditioner of the Schur system: Hiptmair-Xu for the curl-curl operator with ``M2gyro`` in the middle
            (``"HiptmairXu"``, implicit algorithm with ``solver="pcg"`` only, no polar splines; it depends on the time
            step, so it is built at the first time step and rebuilt when the time step changes), or a preconditioner
            of the mass matrix ``M1`` (``"MassMatrixPreconditioner"``, ``"PolarMassMatrixPreconditioner"``).
        hx_aux_weight : float, default=1.0
            Factor of the vector auxiliary space correction of the Hiptmair-Xu preconditioner (``aux_weight`` of
            :class:`~struphy.feec.preconditioner.HiptmairXuPreconditioner`, 0 switches it off; ``M2gyro`` vanishes
            along the background field, so the isotropic vector Laplacian of the correction does not match the
            curl-curl term).
        solver_params : SolverParameters, default=None
            Solver controls of the Schur system; defaults to ``SolverParameters()``.
        solver_M2n : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver of the inner ``M2n`` inversion.
        precond_M2n : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner of the inner ``M2n`` inversion.
        solver_params_M2n : SolverParameters, default=None
            Solver controls of the inner ``M2n`` inversion; defaults to ``SolverParameters(tol=1e-14)`` (the outer
            Krylov solver needs an accurate inner inverse).
        butcher : ButcherTableau, default=None
            Explicit Runge-Kutta tableau when ``algo="explicit"``.
        """

        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        algo: OptsAlgo = "implicit"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsShearAlfvenCurlCurlPrecond = "HiptmairXu"
        hx_aux_weight: float = 1.0
        solver_params: SolverParameters = None
        solver_M2n: LiteralOptions.OptsSymmSolver = "pcg"
        precond_M2n: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params_M2n: SolverParameters = None
        butcher: ButcherTableau = None

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsShearAlfvenCurlCurlPrecond)
            check_option(self.solver_M2n, LiteralOptions.OptsSymmSolver)
            check_option(self.precond_M2n, LiteralOptions.OptsMassPrecond)
            if self.precond == "HiptmairXu":
                assert self.algo == "implicit", 'precond="HiptmairXu" requires algo="implicit".'
                assert self.solver == "pcg", 'precond="HiptmairXu" requires solver="pcg".'

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.solver_params_M2n is None:
                self.solver_params_M2n = SolverParameters(tol=1e-14)

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
        if self.options.precond == "HiptmairXu" and self.derham.polar_splines:
            raise NotImplementedError('precond="HiptmairXu" does not support polar splines.')

        _M1 = self.mass_ops.M1
        _M2B = self.mass_ops.M2B
        curl = self.derham.curl

        # inverse of M2n (inner solve)
        if self.options.precond_M2n is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond_M2n)
            pc = pc_class(self.mass_ops.M2n)

        M2n_inv = inverse(
            self.mass_ops.M2n,
            self.options.solver_M2n,
            pc=pc,
            tol=self.options.solver_params_M2n.tol,
            maxiter=self.options.solver_params_M2n.maxiter,
            verbose=self.options.solver_params_M2n.verbose,
        )

        # preconditioner of the mass matrix M1 (Hiptmair-Xu depends on dt and is built in __call__)
        self._hx_dt = None
        if self.options.precond is None or self.options.precond == "HiptmairXu":
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(_M1)

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info

            # define block matrix [[A B], [C I]] (without time step size dt in the diagonals), x = b, y = u
            self._B = 1 / 2 * curl.T @ _M2B
            self._C = 1 / 2 * M2n_inv @ _M2B @ curl

            # instantiate Schur solver (constant in this case)
            _BC = self._B @ self._C

            self._schur_solver = SchurSolver(
                _M1,
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
                _M1,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            _f1 = M2n_inv @ _M2B @ curl
            _f2 = M1_inv @ curl.T @ _M2B

            # allocate output of vector field
            out1 = self.variables.u.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                _f1.dot(y2, out=out)
                out *= -1.0
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
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()
        self._b_tmp2 = self.variables.b.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        if self.options.algo == "implicit":
            # Hiptmair-Xu preconditioner of S ~ dt^2/4 (C^T M2gyro C + 4/dt^2 M1), for the current time step: built
            # at the first time step, rebuilt when dt changes (sigma = 4/dt^2 with the unscaled M2gyro in the middle,
            # as the vector auxiliary space correction of Hiptmair-Xu assumes a unit-scale curl-curl term)
            if self.options.precond == "HiptmairXu" and dt != self._hx_dt:
                sigma = 4.0 / dt**2
                hx = HiptmairXuPreconditioner(
                    None, self.mass_ops, sigma=sigma, mid="M2gyro", aux_weight=self.options.hx_aux_weight
                )
                self._schur_solver.precond = sigma * hx
                self._hx_dt = dt

            # solve for new b coeffs
            byn = self._B.dot(un, out=self._byn)

            bn1, info = self._schur_solver(bn, byn, dt, out=self._b_tmp1)

            # new u coeffs
            _b = bn.copy(out=self._b_tmp2)
            _b += bn1
            un1 = self._C.dot(_b, out=self._u_tmp1)
            un1 *= -dt
            un1 += un

            diffs = self.update_feec_variables(u=un1, b=bn1)
        else:
            self._ode_solver(0.0, dt)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for ShearAlfvenCurlCurl: {info['success']}")
            logger.info(f"Iterations for ShearAlfvenCurlCurl: {info['niter']}")
            logger.info(f"Maxdiff up for ShearAlfvenCurlCurl: {diffs['u']}")
            logger.info(f"Maxdiff b1 for ShearAlfvenCurlCurl: {diffs['b']}")
            logger.info("")
