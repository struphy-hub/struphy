import logging
from dataclasses import dataclass

from feectools.linalg.basic import IdentityOperator
from line_profiler import profile
from maybempi import MPI

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class Magnetosonic(Propagator):
    r"""Magnetosonic step of linear MHD (zero-flow equilibrium): density, velocity and pressure perturbations at a
    fixed magnetic field perturbation.

    **Continuous problem.** Find :math:`\tilde{\rho} \in L^2`, :math:`\tilde{\mathbf U} \in V_U` with
    :math:`V_U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` (``Options.u_space``) and
    :math:`\tilde p \in L^2` such that

    .. math::

        \frac{\partial \tilde{\rho}}{\partial t} + \nabla\cdot(\rho_0 \tilde{\mathbf{U}}) &= 0\,,
        \\[2mm]
        \int_{\Omega} \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textrm{d}\mathbf x - \int_{\Omega} \tilde p\,\nabla\cdot\mathbf V\,\textrm{d}\mathbf x &= \int_{\Omega} (\nabla\times\mathbf{B}_0)\times\tilde{\mathbf{B}}\cdot\mathbf V\,\textrm{d}\mathbf x \qquad \forall\,\mathbf V \in V_U\,,
        \\[2mm]
        \frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}}) + \frac{2}{3}\,p_0\nabla\cdot\tilde{\mathbf{U}} &= 0\,.

    The continuity and pressure equations hold strongly in :math:`L^2`, the momentum equation is weak (integration by
    parts of the pressure gradient; with free boundary conditions this imposes :math:`\tilde p = 0` weakly). For
    :math:`V_U = H(\textnormal{curl})` or :math:`(H^1)^3`, the divergence acts on the projection of the velocity
    (and of :math:`\mathbf V`) onto :math:`H(\textnormal{div})` (operator :math:`\mathbb U^\alpha` below). The
    magnetic field perturbation :math:`\tilde{\mathbf B}` is not updated (it enters only the Lorentz force); it is the
    constructor argument ``b_field``, either a 2-form (:math:`H(\textnormal{div})`, default) or a 1-form
    (:math:`H(\textnormal{curl})`, only for :math:`V_U = H(\textnormal{div})`).

    **Semi-discrete problem.** With the FE coefficients :math:`\mathbf n`, :math:`\mathbf p` of
    :math:`\tilde \rho_h, \tilde p_h \in V^3_h`, :math:`\mathbf u` of :math:`\tilde{\mathbf U}_h \in V^\alpha_h`
    (:math:`\alpha = 1, 2, v` for ``u_space`` ``"Hcurl"``, ``"Hdiv"``, ``"H1vec"``) and :math:`\mathbf b` of
    :math:`\tilde{\mathbf B}_h \in V^\beta_h` (:math:`\beta = 2` or :math:`1`),

    .. math::

        \dot{\mathbf n} = - \mathbb D \mathbb Q^\alpha \mathbf u\,, \qquad
        \mathbb M^\alpha_{\rho_0} \dot{\mathbf u} = (\mathbb D \mathbb U^\alpha)^\top \mathbb M^3 \mathbf p + \mathbb M^{\alpha\beta}_{J_0} \mathbf b\,, \qquad
        \dot{\mathbf p} = - \left(\mathbb D \mathbb S^\alpha + \frac 23 \mathbb K^3 \mathbb D \mathbb U^\alpha\right) \mathbf u\,,

    with the discrete divergence :math:`\mathbb D` (:attr:`~struphy.feec.psydac_derham.Derham.div`), the mass matrix
    :math:`\mathbb M^3` (:attr:`~struphy.feec.mass.WeightedMassOperators.M3`), the density-weighted mass matrix
    :math:`\mathbb M^\alpha_{\rho_0}` (:attr:`~struphy.feec.mass.WeightedMassOperators.M1n`,
    :attr:`~struphy.feec.mass.WeightedMassOperators.M2n`, :attr:`~struphy.feec.mass.WeightedMassOperators.Mvn`), the
    current mass matrix :math:`\mathbb M^{\alpha\beta}_{J_0}` of :math:`\int (\mathbf J_0 \times \mathbf W) \cdot \mathbf V
    \,\textnormal d \mathbf x`, :math:`\mathbf J_0 = \nabla \times \mathbf B_0`
    (:attr:`~struphy.feec.mass.WeightedMassOperators.M1J`, :attr:`~struphy.feec.mass.WeightedMassOperators.M2J`,
    :attr:`~struphy.feec.mass.WeightedMassOperators.MvJ`; :attr:`~struphy.feec.mass.WeightedMassOperators.M21J` for
    the 1-form :math:`\tilde{\mathbf B}`), and the basis projection operators
    (:class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`)
    :math:`\mathbb Q^\alpha \approx \hat \Pi^2[\rho_0 \,\cdot\,]` (``Q1``, ``Q2``, ``Qv``),
    :math:`\mathbb S^\alpha \approx \hat \Pi^2[p_0 \,\cdot\,]` (``S1``, ``S2``, ``Sv``),
    :math:`\mathbb K^3 \approx \hat \Pi^3[p_0 \,\cdot\,]` (``K3``) and :math:`\mathbb U^\alpha \approx \hat \Pi^2[\,\cdot\,]`
    (``U1``, ``Uv``; the identity for :math:`\alpha = 2`).

    **Time discretization.** Crank-Nicolson (implicit mid-point), the only option. The pressure is eliminated with
    :class:`~struphy.linear_algebra.schur_solver.SchurSolver` (:math:`x = \mathbf u`, :math:`y = \mathbf p`), with
    the Schur complement

    .. math::

        S = \mathbb M^\alpha_{\rho_0} + \frac{\Delta t^2}{4} (\mathbb D \mathbb U^\alpha)^\top \mathbb M^3
        \left(\mathbb D \mathbb S^\alpha + \frac 23 \mathbb K^3 \mathbb D \mathbb U^\alpha\right)\,,

    solved by ``Options.solver`` (non-symmetric, default ``"pbicgstab"``) with the mass matrix preconditioner
    ``Options.precond`` of :math:`\mathbb M^\alpha_{\rho_0}`; then :math:`\mathbf p^{n+1}` and :math:`\mathbf n^{n+1}`
    follow from the mid-point velocity :math:`(\mathbf u^{n+1} + \mathbf u^n)/2`.
    """

    class Variables:
        """Container for variables advanced by :class:`Magnetosonic`.

        Attributes
        ----------
        n : FEECVariable
            Density perturbation variable in ``"L2"`` space.
        u : FEECVariable
            Velocity perturbation variable in one of ``"Hcurl"``, ``"Hdiv"``,
            or ``"H1vec"``.
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
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
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
            Magnetic field perturbation, as a 2-form (``"Hdiv"`` space) or a 1-form (``"Hcurl"`` space, requires
            ``u_space="Hdiv"``), entering the Lorentz force :math:`(\nabla \times \mathbf B_0) \times \tilde{\mathbf B}`.
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
        """Configuration options for :class:`Magnetosonic`.

        Parameters
        ----------
        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the velocity variable ``u``.

        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            Iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the mass matrix block associated with
            ``u_space``.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.
        """

        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        solver: LiteralOptions.OptsGenSolver = "pbicgstab"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
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
        u_space = self.options.u_space

        self._info = self.options.solver_params.info
        self._bc = self.derham.dirichlet_bc

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = "M" + self.derham.space_to_form[u_space] + "n"
        id_MJ = "M" + self.derham.space_to_form[u_space] + "J"

        if u_space == "Hcurl":
            id_S, id_U, id_K, id_Q = "S1", "U1", "K3", "Q1"
        elif u_space == "Hdiv":
            id_S, id_U, id_K, id_Q = "S2", None, "K3", "Q2"
        elif u_space == "H1vec":
            id_S, id_U, id_K, id_Q = "Sv", "Uv", "K3", "Qv"

        _A = getattr(self.mass_ops, id_Mn)
        _S = getattr(self.basis_ops, id_S)
        _K = getattr(self.basis_ops, id_K)

        if id_U is None:
            _U = IdentityOperator(self.variables.u.spline.vector.space)
            _UT = IdentityOperator(self.variables.u.spline.vector.space)
        else:
            _U = getattr(self.basis_ops, id_U)
            _UT = _U.T

        self._B = -1 / 2.0 * _UT @ self.derham.div.T @ self.mass_ops.M3
        # the factor 1/2 of Crank-Nicolson multiplies the whole pressure operator div(p0 u) + 2/3 p0 div u
        self._C = 1 / 2.0 * (self.derham.div @ _S + 2 / 3 * _K @ self.derham.div @ _U)

        if self.b_field.space == "Hcurl":
            if u_space != "Hdiv":
                raise NotImplementedError('A magnetic field in Hcurl requires u_space="Hdiv".')
            self._MJ = self.mass_ops.M21J
        else:
            self._MJ = getattr(self.mass_ops, id_MJ)
        self._DQ = self.derham.div @ getattr(self.basis_ops, id_Q)

        # only allocate the private default; a shared b_field is allocated (with initial conditions) by the model
        if self._allocate_b_field:
            self.b_field.allocate(self.derham, self.domain)
        self._b = self.b_field.spline.vector

        # preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # instantiate Schur solver (constant in this case)
        _BC = self._B @ self._C

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
        self._b_tmp1 = self._b.space.zeros()

        self._byn1 = self._B.codomain.zeros()
        self._byn2 = self._B.codomain.zeros()

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

        # new p, n, b coeffs (no tmps created here)
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
            logger.info(f"Status     for Magnetosonic: {info['success']}")
            logger.info(f"Iterations for Magnetosonic: {info['niter']}")
            logger.info(f"Maxdiff n3 for Magnetosonic: {diffs['n']}")
            logger.info(f"Maxdiff up for Magnetosonic: {diffs['u']}")
            logger.info(f"Maxdiff p3 for Magnetosonic: {diffs['p']}")
            logger.info("")
