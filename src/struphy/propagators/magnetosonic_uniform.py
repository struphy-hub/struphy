import logging
from dataclasses import dataclass

from feectools.ddm.mpi import mpi as MPI
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class MagnetosonicUniform(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde \rho \in L^2, \tilde{\mathbf U} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}, \tilde p \in L^2` such that

    .. math::
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,,

        \int \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textrm d \mathbf x  - \int \tilde p\, \nabla \cdot \mathbf V \,\textrm d \mathbf x
        = 0
        \qquad \forall \ \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

        &\frac{\partial \tilde p}{\partial t}
        + \frac{5}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1}_i - \mathbf p^n_i \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_2)^{-1} \mathbb D^\top \mathbb M_3 \\ - \gamma \mathcal K^3 \mathbb D & 0 \end{bmatrix}
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1}_i + \mathbf p^n_i) \end{bmatrix} ,

    where :math:`\mathbb M^\rho_2`  is a weighted mass matrix in 2-space,
    the weight being the MHD equilibirum density :math:`\rho_0`. Furthermore, :math:`\mathcal K^3` is the basis projection operator given by :

    .. math::

        \mathcal{K}^3_{ijk,mno} := \hat{\Pi}^3_{ijk} \left[ \frac{\hat{p}^3_{\text{eq}}}{\sqrt{g}}\Lambda^3_{mno} \right] \,.
    The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.

    Decoupled density update:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathcal Q \mathbb D  (\mathbf u^{n+1} + \mathbf u^n) \,.

    Parameters
    ----------
    n : feectools.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    u : feectools.linalg.block.BlockVector
        FE coefficients of MHD velocity 2-form.

    p : feectools.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    """

    class Variables:
        """Container for variables advanced by :class:`MagnetosonicUniform`.

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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`MagnetosonicUniform`.

        Parameters
        ----------
        solver : LiteralOptions.OptsGenSolver, default="pbicgstab"
            Iterative solver used by :class:`SchurSolver`.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner applied to the ``M2n`` block.

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
    def allocate(self, verbose: bool = False):
        self._info = self.options.solver_params.info
        self._bc = self.derham.dirichlet_bc

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_Mn = "M2n"
        id_K, id_Q = "K3", "Q3"

        _A = getattr(self.mass_ops, id_Mn)
        _K = getattr(self.basis_ops, id_K)

        self._B = -1 / 2.0 * self.derham.div.T @ self.mass_ops.M3
        self._C = 5 / 6.0 * _K @ self.derham.div

        self._QD = getattr(self.basis_ops, id_Q) @ self.derham.div

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
        n = self.variables.n.spline.vector
        u = self.variables.u.spline.vector
        p = self.variables.p.spline.vector

        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._p_tmp1 = p.space.zeros()
        self._n_tmp1 = n.space.zeros()

        self._byn1 = self._B.codomain.zeros()

    @profile
    def __call__(self, dt):
        # current variables
        nn = self.variables.n.spline.vector
        un = self.variables.u.spline.vector
        pn = self.variables.p.spline.vector

        # solve for new u coeffs
        byn1 = self._B.dot(pn, out=self._byn1)

        un1, info = self._schur_solver(un, byn1, dt, out=self._u_tmp1)

        # new p, n, b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        pn1 = self._C.dot(_u, out=self._p_tmp1)
        pn1 *= -dt
        pn1 += pn

        nn1 = self._QD.dot(_u, out=self._n_tmp1)
        nn1 *= -dt / 2.0
        nn1 += nn

        # write new coeffs into self.feec_vars
        diffs = self.update_feec_variables(n=nn1, u=un1, p=pn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Status     for Magnetosonic: {info['success']}")
            logger.info(f"Iterations for Magnetosonic: {info['niter']}")
            logger.info(f"Maxdiff n3 for Magnetosonic: {diffs['n']}")
            logger.info(f"Maxdiff up for Magnetosonic: {diffs['u']}")
            logger.info(f"Maxdiff p3 for Magnetosonic: {diffs['p']}")
            logger.info("")
