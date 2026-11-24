import logging
from dataclasses import dataclass

from feectools.linalg.solvers import inverse
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.feec.utilities import LocalRotationMatrix
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class MagnetosonicCrossB0(Propagator):
    r"""FEEC discretization of the same equations as :class:`~struphy.propagators.magnetosonic.Magnetosonic`:
    find :math:`\tilde{\rho} \in L^2, \tilde{\mathbf U} \in H(\textnormal{div}), \tilde p \in L^2` such that

    .. math::

        \frac{\partial \tilde{\rho}}{\partial t} + \nabla\cdot(\rho_0 \tilde{\mathbf{U}}) &= 0
        \\[2mm]
        \int_{\Omega} \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textrm{d}\mathbf x - \int_{\Omega} \tilde p\,\nabla\cdot\mathbf V\,\textrm{d}\mathbf x &= \int_{\Omega} \mathbf J_0 \times \tilde{\mathbf{B}}\cdot\mathbf V\,\textrm{d}\mathbf x \qquad \forall\,\mathbf V \in H(\textnormal{div})
        \\[2mm]
        \frac{\partial \tilde p}{\partial t} + \tilde{\mathbf U}\cdot\nabla p_0 + \frac{5}{3}\,p_0\nabla\cdot\tilde{\mathbf{U}} &= 0

    where :math:`\tilde{\mathbf B}` is a 2-form that is not updated here. Unlike in ``Magnetosonic``, the equilibrium
    terms are written through :math:`\mathbf B_0 \times`, which requires :math:`\mathbf B_0 \cdot \nabla p_0 = 0`:

    .. math::

        \mathbf J_0 \times \tilde{\mathbf B} \cdot \mathbf V
        &= -c\,(\mathbf B_0 \times \mathbf V) \cdot \tilde{\mathbf B} + \mathbf J_\perp \times \tilde{\mathbf B} \cdot \mathbf V\,,
        \qquad c = \frac{\mathbf J_0 \cdot \mathbf B_0}{|\mathbf B_0|^2}\,,\qquad \mathbf J_\perp = \mathbf J_0 - c\,\mathbf B_0\,,
        \\[2mm]
        \tilde{\mathbf U} \cdot \nabla p_0 &= \mathbf J_p \cdot (\mathbf B_0 \times \tilde{\mathbf U})\,,
        \qquad \mathbf J_p = \frac{\mathbf B_0 \times \nabla p_0}{|\mathbf B_0|^2}\,.

    :math:`\mathbf B_0 \times` is discretized with
    :attr:`~struphy.feec.basis_projection_ops.BasisProjectionOperators.T2` operator, exploiting the fact that
    :math:`\mathbf J_0` is mostly parallel to :math:`\mathbf B_0` and :math:`\nabla p_0` is perpendicular to
    :math:`\mathbf B_0`.

    Time discretization: Crank-Nicolson, reduced with :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \mathbb M^2_{\rho_0} \frac{\mathbf u^{n+1} - \mathbf u^n}{\Delta t}
        &= \mathbb D^\top \mathbb M^3 \frac{\mathbf p^{n+1} + \mathbf p^n}{2} + \mathbb M^J \mathbf b\,,
        \\[2mm]
        \frac{\mathbf p^{n+1} - \mathbf p^n}{\Delta t}
        &= -\left(\mathbb G + \frac{5}{3} \mathbb K^3 \mathbb D\right) \frac{\mathbf u^{n+1} + \mathbf u^n}{2}\,,
        \\[2mm]
        \frac{\mathbf n^{n+1} - \mathbf n^n}{\Delta t}
        &= -\mathbb D \mathbb Q^2 \frac{\mathbf u^{n+1} + \mathbf u^n}{2}\,.
    """

    class Variables:
        """Container for variables advanced by :class:`MagnetosonicCrossB0`.

        Attributes
        ----------
        n : FEECVariable
            Density perturbation variable in ``"L2"`` space.
        u : FEECVariable
            Velocity perturbation variable in ``"Hdiv"`` space.
        p : FEECVariable
            Pressure perturbation variable in ``"L2"`` space.
        """

        def __init__(self):
            self._n: FEECVariable = None
            self._u: FEECVariable = None
            self._p: FEECVariable = None

        @property
        def n(self) -> FEECVariable:
            return self._n

        @n.setter
        def n(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._n = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._u = new

        @property
        def p(self) -> FEECVariable:
            return self._p

        @p.setter
        def p(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._p = new

    def __init__(self, b_field: FEECVariable = None):
        r"""
        Parameters
        ----------
        b_field : FEECVariable, default=None
            Magnetic field perturbation, as a 2-form (``"Hdiv"`` space), entering the Lorentz force
            :math:`(\nabla \times \mathbf B_0) \times \tilde{\mathbf B}`.
            If ``None``, an empty ``FEECVariable(space="Hdiv")`` is created and allocated internally.
            Otherwise the given variable is treated as shared (e.g. the model's ``em_fields.b_field``)
            and must already be allocated when :meth:`allocate` is called; it is not re-allocated.
        """
        assert b_field is None or b_field.space == "Hdiv", "b_field must be in Hdiv."
        self.variables = self.Variables()
        self.b_field = b_field if b_field is not None else FEECVariable(space="Hdiv")
        self._allocate_b_field = b_field is None

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`MagnetosonicCrossB0`.

        Parameters
        ----------
        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            Iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the mass matrix block associated with
            the velocity variable ``u``.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.
        """

        solver: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, LiteralOptions.OptsGenSolver)
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

        div, M3 = self.derham.div, self.mass_ops.M3
        T2 = self.basis_ops.T2

        _gradp0_T2 = self._weighted_mass_gradp0() @ T2
        _Kdiv = 5 / 3 * self.basis_ops.K3 @ div
        self._B = -1 / 2.0 * div.T @ M3
        self._M3_inv = inverse(
            M3,
            "pcg",
            pc=preconditioner.MassMatrixPreconditioner(M3),
            tol=1e-12,
            maxiter=self.options.solver_params.maxiter,
        )
        self._C = 1 / 2.0 * (_Kdiv + self._M3_inv @ _gradp0_T2)

        self._MJ = self._jxb_operator()
        self._DQ = div @ self.basis_ops.Q2

        if self._allocate_b_field:
            self.b_field.allocate(self.derham, self.domain)
        self._b = self.b_field.spline.vector

        _A = self.mass_ops.M2n
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(_A)

        _BC = self._B @ (1 / 2.0 * _Kdiv) + (-1 / 4.0) * div.T @ _gradp0_T2

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # allocate dummy vectors to avoid temporary array allocations
        self._u_tmp1 = self.variables.u.spline.vector.space.zeros()
        self._u_tmp2 = self.variables.u.spline.vector.space.zeros()
        self._p_tmp1 = self.variables.p.spline.vector.space.zeros()
        self._n_tmp1 = self.variables.n.spline.vector.space.zeros()

        self._byn1 = self._B.codomain.zeros()
        self._byn2 = self._B.codomain.zeros()

    def _parallel_current_factor(self, e1, e2, e3):
        """The factor c = (J0.B0)/|B0|^2, from the 2-form of J0 and the 1-form of B0."""
        eq = self.mass_ops.eq_mhd
        j_funs = (eq.j2_1, eq.j2_2, eq.j2_3)
        b_funs = (eq.b1_1, eq.b1_2, eq.b1_3)
        j_dot_b = sum(j(e1, e2, e3) * b(e1, e2, e3) for j, b in zip(j_funs, b_funs))
        sqrt_g = self.domain.jacobian_det(e1, e2, e3, squeeze_out=False)
        return j_dot_b / (sqrt_g * eq.absB0(e1, e2, e3) ** 2)

    def _weighted_mass_gradp0(self):
        """Mass matrix from 1-forms to 3-forms such that q^T M e = int q J_p . E dx, for a 3-form q and a 1-form E."""
        eq = self.mass_ops.eq_mhd

        def weight(j_dia2_k):
            return lambda e1, e2, e3: j_dia2_k(e1, e2, e3) / self.domain.jacobian_det(e1, e2, e3, squeeze_out=False)

        return self.mass_ops.create_weighted_mass(
            "Hcurl",
            "L2",
            weights=[[weight(eq.j_dia2_1), weight(eq.j_dia2_2), weight(eq.j_dia2_3)]],
            name="M31Jp",
            assemble=True,
        )

    def _jxb_operator(self):
        """The operator M^J of the force J0 x b on a 2-form b: - T2^T M21(c) + M2(J_perp)."""
        eq = self.mass_ops.eq_mhd
        j_funs = (eq.j2_1, eq.j2_2, eq.j2_3)
        b_funs = (eq.b2_1, eq.b2_2, eq.b2_3)
        c = self._parallel_current_factor

        diag = [[c if m == n else None for n in range(3)] for m in range(3)]
        M21_c = self.mass_ops.create_weighted_mass("Hdiv", "Hcurl", weights=diag, name="M21Jpar", assemble=True)

        j_perp = [
            (lambda e1, e2, e3, k=k: j_funs[k](e1, e2, e3) - c(e1, e2, e3) * b_funs[k](e1, e2, e3)) for k in range(3)
        ]
        M2J_perp = self.mass_ops.create_weighted_mass(
            "Hdiv",
            "Hdiv",
            weights=(LocalRotationMatrix(*j_perp), "1/sqrt_g"),
            name="M2Jperp",
            assemble=True,
        )

        return -1.0 * self.basis_ops.T2.T @ M21_c + M2J_perp

    @profile
    def __call__(self, dt):
        # current FE coeffs
        nn = self.variables.n.spline.vector
        un = self.variables.u.spline.vector
        pn = self.variables.p.spline.vector

        # solve for new u coeffs (no tmps created here)
        byn1 = self._B.dot(pn, out=self._byn1)
        byn2 = self._MJ.dot(self._b, out=self._byn2)
        byn2 *= 1 / 2
        byn1 -= byn2

        un1, info = self._schur_solver(un, byn1, dt, out=self._u_tmp1)

        # new p, n coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        pn1 = self._C.dot(_u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        nn1 = self._DQ.dot(_u, out=self._n_tmp1)
        nn1 *= -dt / 2
        nn1 += nn

        diffs = self.update_feec_variables(n=nn1, u=un1, p=pn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for MagnetosonicCrossB0: {info['success']}")
            logger.info(f"Iterations for MagnetosonicCrossB0: {info['niter']}")
            logger.info(f"Maxdiff n3 for MagnetosonicCrossB0: {diffs['n']}")
            logger.info(f"Maxdiff up for MagnetosonicCrossB0: {diffs['u']}")
            logger.info(f"Maxdiff p3 for MagnetosonicCrossB0: {diffs['p']}")
            logger.info("")
