import logging
from dataclasses import dataclass
from typing import Literal

from feectools.linalg.block import BlockVector
from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI
from scope_profiler import ProfileManager

from struphy.feec import preconditioner
from struphy.feec.kronecker_block import basis_projection_operator, derivative_operator
from struphy.feec.preconditioner import HiptmairXuPreconditioner
from struphy.feec.utilities import LocalRotationMatrix
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
    r"""Shear-Alfvén step of linear MHD (zero-flow equilibrium) with the magnetic field in :math:`H(\textnormal{curl})`.

    **Continuous problem.** Find :math:`\tilde{\mathbf U} \in H(\textnormal{div})` and
    :math:`\tilde{\mathbf B} \in H(\textnormal{curl})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \left[(\nabla \times \tilde{\mathbf B}) \times \mathbf B_0\right] \cdot \mathbf V \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in H(\textnormal{div})\,,
        \\[2mm]
        &\int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x = - \int_\Omega \left[(\nabla \times \mathbf C) \times \mathbf B_0\right] \cdot \tilde{\mathbf{U}} \,\textrm d \mathbf x \qquad \forall \, \mathbf C \in H(\textrm{curl})\,.

    The momentum equation is the :math:`\rho_0`-weighted :math:`L^2` projection of the Lorentz force onto
    :math:`H(\textnormal{div})`; the induction equation is weak, its boundary term is dropped, which imposes
    :math:`\tilde{\mathbf U} \cdot \mathbf n = 0` weakly (free boundary conditions of the de Rham sequence;
    essential ones set the tangential components of :math:`\tilde{\mathbf B}` to zero on the boundary).

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf u \in \mathbb R^{N_2}` of
    :math:`\tilde{\mathbf U}_h \in V^2_h` and :math:`\mathbf b \in \mathbb R^{N_1}` of
    :math:`\tilde{\mathbf B}_h \in V^1_h`,

    .. math::

        \mathbb M^{2,n} \dot{\mathbf u} = - \mathbb M^{2,B} \mathbb C \mathbf b\,, \qquad
        \mathbb M^1 \dot{\mathbf b} = - \mathbb C^\top \mathbb M^{2,B} \mathbf u\,,

    with the discrete curl :math:`\mathbb C` (:attr:`~struphy.feec.psydac_derham.Derham.curl`), the mass matrix
    :math:`\mathbb M^1` (:attr:`~struphy.feec.mass.WeightedMassOperators.M1`), the density-weighted mass matrix of
    2-forms :math:`\mathbb M^{2,n}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2n`) and the antisymmetric
    mass matrix :math:`\mathbb M^{2,B}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M2B`) of
    :math:`\int (\mathbf B_0 \times \mathbf W) \cdot \mathbf V \,\textnormal d \mathbf x`. This is the semi-discrete
    system of :class:`~struphy.propagators.shear_alfven_b1.ShearAlfvenB1`. It conserves the perturbed energy
    :math:`\frac 12 \mathbf u^\top \mathbb M^{2,n} \mathbf u + \frac 12 \mathbf b^\top \mathbb M^1 \mathbf b`.

    **Time discretization.** Selected by ``Options.algo`` and the constructor argument ``projected``:

    - ``algo="implicit"``, ``projected=True``: Crank-Nicolson (implicit mid-point) of the semi-discrete system, which
      conserves the perturbed energy exactly (up to the solver tolerances). The velocity is eliminated with
      :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (:math:`x = \mathbf b`, :math:`y = \mathbf u`), with
      the Schur complement

      .. math::

          S = \mathbb M^1 + \frac{\Delta t^2}{4} \mathbb C^\top (\mathbb M^{2,B})^\top (\mathbb M^{2,n})^{-1} \mathbb M^{2,B} \mathbb C\,,

      where :math:`(\mathbb M^{2,n})^{-1}` is an inner iterative solve (``solver_M2n``, ``precond_M2n``). The
      curl-curl operator with :math:`\mathbb M^{2,\textnormal{gyro}}` below is not spectrally equivalent to :math:`S`
      (it overestimates :math:`S` on the fields whose Lorentz force is removed by the projection), so
      ``precond="HiptmairXu"`` does not pay off here; use ``precond="MassMatrixPreconditioner"``.
    - ``algo="implicit"``, ``projected=False`` (default): Crank-Nicolson **without the projection in the curl-curl
      term**: the velocity update
      :math:`\tilde{\mathbf U}^{n+1} = \tilde{\mathbf U}^n + \frac{\Delta t}{2 \rho_0} (\nabla \times
      (\tilde{\mathbf B}^{n+1} + \tilde{\mathbf B}^n)) \times \mathbf B_0` is inserted into the weak induction
      equation before the projection, which gives the curl-curl system

      .. math::

          \left(\mathbb M^1 + \frac{\Delta t^2}{4} \mathbb C^\top \mathbb M^{2,\textnormal{gyro}} \mathbb C\right) \mathbf b^{n+1}
          = \left(\mathbb M^1 - \frac{\Delta t^2}{4} \mathbb C^\top \mathbb M^{2,\textnormal{gyro}} \mathbb C\right) \mathbf b^n
          - \Delta t \, \mathbb C^\top \mathbb M^{2,B} \mathbf u^n\,,

      with :attr:`~struphy.feec.mass.WeightedMassOperators.M2gyro`, the mass matrix of
      :math:`\int \frac{1}{\rho_0} \left[(\nabla \times \mathbf C) \times \mathbf B_0\right] \cdot \left[(\nabla \times \tilde{\mathbf B}) \times \mathbf B_0\right] \textnormal d \mathbf x`.
      The velocity is then updated with the projection,
      :math:`\mathbf u^{n+1} = \mathbf u^n - \frac{\Delta t}{2} (\mathbb M^{2,n})^{-1} \mathbb M^{2,B} \mathbb C (\mathbf b^{n+1} + \mathbf b^n)`.
      The perturbed energy does not increase, but is not exactly conserved (the projection error of the Lorentz force
      enters): numerical damping that vanishes with the resolution along :math:`\mathbf B_0`. The system is solved
      by :class:`~struphy.linear_algebra.schur_solver.SchurSolver` with :math:`x = \mathbf b`, :math:`y = \mathbf u`.
      With ``precond="HiptmairXu"`` (default), it is preconditioned by
      :class:`~struphy.feec.preconditioner.HiptmairXuPreconditioner` for the curl-curl operator (scaled by
      :math:`\sigma = 4/\Delta t^2`; TAE example 8x32x4: 37 instead of 706 PCG iterations per step for
      :math:`\Delta t = 2`), optionally (``aux_correction``) plus the field-aligned auxiliary space corrections of
      :class:`~struphy.feec.preconditioner.AuxiliarySpacePreconditioner`: perpendicular perturbations
      :math:`\mathbf B_0 \times \nabla \psi` and parallel perturbations :math:`\phi \mathbf B_0` with
      :math:`\psi, \phi \in V^0_h`, whose operators are approximated by the Kronecker
      :class:`~struphy.feec.preconditioner.StiffnessPreconditioner`.
    - ``algo="explicit"``: explicit Runge-Kutta method of the semi-discrete system, with the tableau
      ``Options.butcher`` (:class:`~struphy.ode.utils.ButcherTableau`); ``projected`` has no effect.

    Parameters
    ----------
    projected : bool, default=False
        Whether the curl-curl term of the implicit step contains the projection of the Lorentz force (exactly energy
        conserving step, see above).
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

    def __init__(self, projected: bool = False):
        self.variables = self.Variables()
        # whether the curl-curl term contains the projection of the Lorentz force (exact energy conservation)
        self._projected = projected

    @dataclass(repr=False)
    class Options(OptionsBase):
        r"""Configuration options for :class:`ShearAlfvenCurlCurlPropagator`.

        Parameters
        ----------
        algo : {"implicit", "explicit"}, default="implicit"
            Time stepping algorithm.
        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Solver of the Schur system in ``b`` (implicit) or of ``M1`` (explicit).
        precond : LiteralOptions.OptsShearAlfvenCurlCurlPrecond, default="HiptmairXu"
            Preconditioner of the Schur system: Hiptmair-Xu for the curl-curl operator with ``M2gyro`` in the middle
            (``"HiptmairXu"``, implicit algorithm with ``solver="pcg"`` only, no polar splines; it depends on the time
            step, so it is built in the setup (:meth:`prepare_time_step`) and rebuilt when the time step changes), or a
            preconditioner
            of the mass matrix ``M1`` (``"MassMatrixPreconditioner"``, ``"PolarMassMatrixPreconditioner"``; recommended
            with ``projected=True``). With ``algo="explicit"``, ``"HiptmairXu"`` is replaced by
            ``"MassMatrixPreconditioner"`` (for ``M1``).
        aux_correction : bool, default=False
            With ``precond="HiptmairXu"``: add the field-aligned auxiliary space corrections of the perturbations
            :math:`\mathbf B_0 \times \nabla \psi` and :math:`\phi \mathbf B_0` (:math:`\psi, \phi \in V^0_h`), which the
            Hiptmair-Xu preconditioner of the anisotropic curl-curl operator reduces only slowly. They lower the
            condition number (TAE geometry, 4x12x3, dt = 1: 34 to 19) but hardly the PCG iterations (tilted slab
            16x16x8, dt = 2: 148 to 125; TAE example 8x32x4: unchanged), at about twice the cost per iteration.
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
        aux_correction: bool = False
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
            if self.aux_correction and self.precond != "HiptmairXu":
                raise ValueError('aux_correction=True requires precond="HiptmairXu".')
            if self.precond == "HiptmairXu":
                if self.algo == "explicit":
                    self.precond = "MassMatrixPreconditioner"
                else:
                    assert self.solver in ("pcg", "pcg_sr"), 'precond="HiptmairXu" requires solver="pcg" or "pcg_sr".'

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
        if self.options.precond == "HiptmairXu" and self.options.aux_correction:
            assert not self.derham.with_local_projectors, "The auxiliary space correction needs the global projectors."

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

            # S = A - dt^2 BC: with the projection, BC = 1/4 C^T M2B M2n^-1 M2B C; without, the curl-curl term
            # -1/4 C^T M2gyro C of the continuous Lorentz force (M2B M2n^-1 M2B ~ -M2gyro, M2B antisymmetric)
            if self._projected:
                _BC = self._B @ self._C
            else:
                _BC = -1 / 4 * curl.T @ self.mass_ops.M2gyro @ curl

            if self.options.precond == "HiptmairXu" and self.options.aux_correction:
                self._allocate_aux_spaces()

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

    def _allocate_aux_spaces(self):
        """Maps N of the auxiliary spaces (psi -> B0 x grad psi, phi -> phi B0, as 1-forms) and their Kronecker
        approximations (independent of dt)."""
        eq = self.basis_ops.weights["eq_mhd"]
        domain = self.domain
        rot_b2 = LocalRotationMatrix(eq.b2_1, eq.b2_2, eq.b2_3)

        # covariant components of B0 x grad psi: eps_ijk b2_j (G^-1 grad psi)_k
        def weight(e1, e2, e3, m, n):
            Ginv = domain.metric_inv(e1, e2, e3, change_out_order=True)
            R = rot_b2(e1, e2, e3)
            return sum(R[..., m, k] * Ginv[..., k, n] for k in range(3))

        fun = [[(lambda e1, e2, e3, m=m, n=n: weight(e1, e2, e3, m, n)) for n in range(3)] for m in range(3)]
        Pi_perp = self.basis_ops.create_basis_op(fun, "Hcurl", "Hcurl", assemble=True, name="Pi_B0xgrad")
        # covariant components of phi B0: phi b1
        Pi_par = self.basis_ops.create_basis_op(
            [[eq.b1_1], [eq.b1_2], [eq.b1_3]], "H1", "Hcurl", assemble=True, name="Pi_phiB0"
        )

        grad = self.derham.grad
        kron_curl = derivative_operator(self.derham, "curl")
        kron_perp = basis_projection_operator(Pi_perp, self.derham) @ derivative_operator(self.derham, "grad")
        kron_par = basis_projection_operator(Pi_par, self.derham)
        # (N, N^T, [N~, C N~]) per auxiliary space
        self._aux_spaces = [
            (Pi_perp @ grad, grad.T @ Pi_perp.T, [kron_perp, kron_curl @ kron_perp]),
            (Pi_par, Pi_par.T, [kron_par, kron_curl @ kron_par]),
        ]

    @profile
    def _build_precond(self, dt: float):
        """Hiptmair-Xu preconditioner of S ~ dt^2/4 (C^T M2gyro C + 4/dt^2 M1) (sigma = 4/dt^2 with the unscaled M2gyro in
        the middle, as the vector auxiliary space correction of Hiptmair-Xu assumes a unit-scale curl-curl term), plus
        the field-aligned auxiliary space corrections N P N^T, P the Kronecker preconditioner of
        N^T (M1 + dt^2/4 C^T M2gyro C) N on V0."""
        sigma = 4.0 / dt**2
        with ProfileManager.profile_region("init HiptmairXu"):
            hx = HiptmairXuPreconditioner(
                None, self.mass_ops, sigma=sigma, mid="M2gyro", aux_weight=self.options.hx_aux_weight
            )
        pc = sigma * hx
        if self.options.aux_correction:
            with ProfileManager.profile_region("add auxiliary space corrections"):
                mid = (dt**2 / 4) * self.mass_ops.M2gyro
                corrections = [
                    (N, NT, preconditioner.StiffnessPreconditioner(self.mass_ops, ops, sigma=0.0, mid=["M1", mid]))
                    for N, NT, ops in self._aux_spaces
                ]
                pc = preconditioner.AuxiliarySpacePreconditioner(pc, corrections)
        return pc

    def prepare_time_step(self, dt: float):
        """Hiptmair-Xu preconditioner of S ~ dt^2/4 (C^T M2gyro C + 4/dt^2 M1) for the time step ``dt``: built in the
        setup, rebuilt when dt changes (sigma = 4/dt^2 with the unscaled M2gyro in the middle, as the vector auxiliary
        space correction of Hiptmair-Xu assumes a unit-scale curl-curl term)."""
        if self.options.algo == "implicit" and self.options.precond == "HiptmairXu" and dt != self._hx_dt:
            with ProfileManager.profile_region("build_HiptmairXu_precond"):
                self._schur_solver.precond = self._build_precond(dt)
            self._hx_dt = dt

    @profile
    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        if self.options.algo == "implicit":
            # Hiptmair-Xu preconditioner for the current time step (no-op unless dt changed since the setup)
            self.prepare_time_step(dt)

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
