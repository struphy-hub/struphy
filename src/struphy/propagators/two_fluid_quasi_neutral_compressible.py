import logging
from dataclasses import dataclass
from typing import Callable, get_args
from warnings import warn

from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse

from struphy.feec.boundary_mass import BoundaryIntegralOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.geometry.utilities import TransformedPformComponent
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option
from struphy.feec.preconditioner import MassMatrixPreconditioner
from struphy.initial.base import Perturbation

logger = logging.getLogger("struphy")


class TwoFluidQuasiNeutralCompressible(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the uniform-density quasi-neutral
    two-fluid model in H(curl)/H1 spaces.

    Finds :math:`u_i, u_e \in H(\mathrm{curl})` and :math:`\phi \in H^1` such that:

    .. math::

        \partial_t (u_i, v_i)
        + (\nabla\phi, v_i)
        - \frac{1}{\varepsilon}(u_i \times B, v_i)
        + \nu_i (\mathrm{curl}\, u_i, \mathrm{curl}\, v_i)
        - \nu_i (\nabla\omega_i, v_i)
        - \int_{\partial\Omega} v_i \cdot (g_i \times n)\,dS
        &= (f_i, v_i) \\
        - (\nabla\phi, v_e)
        + \frac{1}{\mu\varepsilon}(u_e \times B, v_e)
        + \mu\nu_e (\mathrm{curl}\, u_e, \mathrm{curl}\, v_e)
        - \mu\nu_e (\nabla\omega_e, v_e)
        - \int_{\partial\Omega} v_e \cdot (g_e \times n)\,dS
        &= (f_e, v_e) \\
        (\omega_i, \alpha_i) + (u_i, \nabla\alpha_i) &= \mathbb{B}^0 g_i \\
        (\omega_e, \alpha_e) + (u_e, \nabla\alpha_e) &= \mathbb{B}^0 g_e \\
        (u_i - u_e, \nabla\psi) &= \mathbb{B}^0(g_i - g_e)

    The normal trace is enforced weakly via :math:`\mathbb{B}^0` (scalar H1 boundary mass).
    The tangential trace is enforced strongly via essential BCs on the H(curl) space.

    :ref:`time_discret`: fully implicit Euler.
    """

    # =========================================================================
    ### State variables
    # =========================================================================

    class Variables:
        """Container for variables advanced by :class:`TwoFluidQuasiNeutralHCurl`.

        Attributes
        ----------
        u : FEECVariable or None
            Ion velocity in ``"Hcurl"`` space.
        ue : FEECVariable or None
            Electron velocity in ``"Hcurl"`` space.
        phi : FEECVariable or None
            Electrostatic potential in ``"H1"`` space.
        """

        def __init__(self) -> None:
            self._u: FEECVariable | None = None
            self._ue: FEECVariable | None = None
            self._phi: FEECVariable | None = None

        @property
        def u(self) -> FEECVariable | None:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._u = new

        @property
        def ue(self) -> FEECVariable | None:
            return self._ue

        @ue.setter
        def ue(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._ue = new

        @property
        def phi(self) -> FEECVariable | None:
            return self._phi

        @phi.setter
        def phi(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._phi = new

    def __init__(self, allocate_variables: bool = False):
        self.variables = self.Variables()

        if allocate_variables:
            self.variables.u = FEECVariable(space="Hcurl")
            self.variables.ue = FEECVariable(space="Hcurl")
            self.variables.phi = FEECVariable(space="H1")

            self.variables.u.allocate(derham=self.derham, domain=self.domain, equil=self.projected_equil.equil)
            self.variables.ue.allocate(derham=self.derham, domain=self.domain, equil=self.projected_equil.equil)
            self.variables.phi.allocate(derham=self.derham, domain=self.domain, equil=self.projected_equil.equil)

    # =========================================================================
    ### Options
    # =========================================================================

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`TwoFluidQuasiNeutralHCurl`.

        Parameters
        ----------
        nu : float, default=1.0
            Ion viscosity coefficient.
        nu_e : float, default=1.0
            Electron viscosity coefficient.
        mu : float, default=1.0
            Electron-to-ion mass ratio (mu = m_e / m_i).
        eps_norm : float or None, default=None
            Normalization parameter epsilon (ion cyclotron parameter).
        source_u : Callable or None
            Source term for ion momentum equation.
        source_ue : Callable or None
            Source term for electron momentum equation.
        essential_u : list[Perturbation] or None
            Essential (tangential) boundary data for ions — enforced strongly via
            the constrained H(curl) space. Set via ``variables.u.lifting_function``.
        essential_ue : list[Perturbation] or None
            Essential (tangential) boundary data for electrons — enforced strongly.
        solver : str, default="gmres"
            Linear solver for the saddle point system.
        solver_params : SolverParameters or None

        Notes
        -----
        The normal trace is enforced weakly via ``NormalBoundaryMass(data_space="Hcurl")``,
        using the H(curl) lifting spline (``variables.u.spline_lift``) directly.
        No separate normal boundary data needs to be supplied.
        """

        nu: float = 1.0
        nu_e: float = 1.0
        mu: float = 1.0
        eps_norm: float | None = None

        source_u: Callable | None = None
        source_ue: Callable | None = None

        # tangential BC: enforced strongly on H(curl)
        essential_u: list[Perturbation] | Perturbation | None = None
        essential_ue: list[Perturbation] | Perturbation | None = None

        solver: LiteralOptions.OptsGenSolver = "gmres"
        solver_params: SolverParameters | None = None

        def __post_init__(self):
            if self.source_u is None:
                warn("No source_u specified — defaulting to zero.")
            if self.source_ue is None:
                warn("No source_ue specified — defaulting to zero.")
            if self.eps_norm is None:
                warn("No eps_norm specified — will default to ion cyclotron parameter epsilon in allocate.")

            if self.nu < 0:
                raise ValueError(f"nu must be non-negative, got {self.nu}")
            if self.nu_e < 0:
                raise ValueError(f"nu_e must be non-negative, got {self.nu_e}")
            if self.mu <= 0:
                raise ValueError(f"mu must be positive, got {self.mu}")
            if self.eps_norm is not None and self.eps_norm <= 0:
                raise ValueError(f"eps_norm must be positive, got {self.eps_norm}")

            check_option(self.solver, LiteralOptions.OptsGenSolver, LiteralOptions.OptsSaddlePointSolver)
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

    # =========================================================================
    ### Allocate
    # =========================================================================

    def allocate(self):

        self._rank = self.derham.comm.Get_rank() if self.derham.comm is not None else 0
        self._dt = None

        if self.options.eps_norm is None:
            self._options.eps_norm = self.variables.u.species.equation_params.epsilon

        # ---- lifting (derham_lift is unconstrained, self.derham is constrained) ---
        self._has_lifting_u = self.variables.u.derham_lift is not None
        self._has_lifting_ue = self.variables.ue.derham_lift is not None

        self._derham_lift_u = self.variables.u.derham_lift if self._has_lifting_u else self.derham
        self._derham_lift_ue = self.variables.ue.derham_lift if self._has_lifting_ue else self.derham

        # ---- solution spline in constrained space ---
        self._u_0 = self.derham.create_spline_function("u", space_id="Hcurl")

        # boundary splines (tangential lifting g_i, g_e) in unconstrained H(curl)
        self._boundary_spline_u = (
            self.variables.u.spline_lift.vector
            if self._has_lifting_u
            else self._derham_lift_u.coeff_spaces["1"].zeros()
        )
        self._boundary_spline_ue = (
            self.variables.ue.spline_lift.vector
            if self._has_lifting_ue
            else self._derham_lift_ue.coeff_spaces["1"].zeros()
        )  # TODO Naming is weird!

        self._hcurl_b_op_u = (
            self.variables.u.boundary_op_lift
            if self._has_lifting_u
            else IdentityOperator(self.derham.coeff_spaces["1"])
        )
        self._hcurl_b_op_ue = (
            self.variables.ue.boundary_op_lift
            if self._has_lifting_ue
            else IdentityOperator(self.derham.coeff_spaces["1"])
        )

        # ---- pre-allocated RHS vectors ---
        self._rhs_vec_u = self.derham.create_spline_function("rhs_vec_u", space_id="Hcurl")
        self._rhs_vec_ue = self.derham.create_spline_function("rhs_vec_ue", space_id="Hcurl")
        self._rhs_vec_phi = self.derham.create_spline_function("rhs_vec_phi", space_id="H1")

        # ---- source terms projected onto unconstrained H(curl) ---
        self._src_u = self._derham_lift_u.create_spline_function("rhs_u", space_id="Hcurl")
        self._src_ue = self._derham_lift_ue.create_spline_function("rhs_ue", space_id="Hcurl")

        for rhs, source, derham_lift in [
            (self._src_u, self.options.source_u, self._derham_lift_u),
            (self._src_ue, self.options.source_ue, self._derham_lift_ue),
        ]:
            if source is not None:
                fun_vec = [lambda x, y, z, f=source, c=c: f(x, y, z)[c] for c in range(3)]
                fun = [
                    TransformedPformComponent(
                        fun_vec, "physical", "1", comp=comp, domain=self.domain,
                    )
                    for comp in range(3)
                ]
                rhs.vector = derham_lift.projectors["1"](fun)

        # ---- mass operators ---
        self._mass_ops_lift_u = WeightedMassOperators(self._derham_lift_u, self.domain, eq_mhd=self.mass_ops.eq_mhd)
        self._mass_ops_lift_ue = WeightedMassOperators(self._derham_lift_ue, self.domain, eq_mhd=self.mass_ops.eq_mhd)

        # unconstrained operators (for RHS assembly with lifting)
        self._M1_u = self._mass_ops_lift_u.M1
        self._M0_u = self._mass_ops_lift_u.M0
        self._M1B_u = self._mass_ops_lift_u.M1B
        self._curl_u = self._derham_lift_u.curl
        self._div_u = self._derham_lift_u.div

        self._M1_ue = self._mass_ops_lift_ue.M1
        self._M0_ue = self._mass_ops_lift_ue.M0
        self._M1B_ue = self._mass_ops_lift_ue.M1B
        self._curl_ue = self._derham_lift_ue.curl
        self._div_ue = self._derham_lift_ue.div

        self._mass_pc_u = MassMatrixPreconditioner(mass_operator=self._M0_u)
        self._M0inv_u = inverse(self._M0_u, "pcg", pc=self._mass_pc_u, tol=1e-10, maxiter=1000, recycle=True)

        self._mass_pc_ue = MassMatrixPreconditioner(mass_operator=self._M0_ue)
        self._M0inv_ue = inverse(self._M0_ue, "pcg", pc=self._mass_pc_ue, tol=1e-10, maxiter=1000, recycle=True)

        self._lapl_u = (
            self._curl_u.T @ self._mass_ops_lift_u.M2 @ self._curl_u
            + self._M1_u @ self._div_u.T @ self._M0inv_u @ self._div_u @ self._M1_u
        )
        self._lapl_ue = (
            self._curl_ue.T @ self._mass_ops_lift_ue.M2 @ self._curl_ue
            + self._M1_ue @ self._div_ue.T @ self._M0inv_ue @ self._div_ue @ self._M1_ue
        )

        self._A_i = (
            - self._M1B_u / self.options.eps_norm
            + self.options.nu * self._lapl_u
        )
        self._A_e = (
            self._M1B_ue / (self.options.mu * self.options.eps_norm)
            + self.options.mu * self.options.nu_e * self._lapl_ue
        )

        # ---- constrained operators (for system matrix) ---
        self._M1 = self.mass_ops.M1
        self._M0 = self.mass_ops.M0
        self._M1B = self.mass_ops.M1B
        self._curl = self.derham.curl
        self._div = self.derham.div

        self._mass_pc = MassMatrixPreconditioner(mass_operator=self._M0)
        self._M0inv = inverse(self._M0, "pcg", pc=self._mass_pc, tol=1e-10, maxiter=1000, recycle=True)

        self._lapl_v0 = (
            self._curl.T @ self.mass_ops.M2 @ self._curl
            + self._M1 @ self._div.T @ self._M0inv @ self._div @ self._M1
        )

        self._A11 = -self._M1B / self.options.eps_norm + self.options.nu * self._lapl_v0
        self._A22 = (
            self._M1B / (self.options.mu * self.options.eps_norm)
            + self.options.mu * self.options.nu_e * self._lapl_v0
        )

        # ---- normal boundary mass: int_{dOmega} (g.n) * alpha dS ---
        bnd_ops_u = BoundaryIntegralOperators(self._mass_ops_lift_u, active_faces=[True] * 6)
        self._B0_normal_u = bnd_ops_u.normal(data_space="Hcurl", test_space="H1")

        bnd_ops_ue = BoundaryIntegralOperators(self._mass_ops_lift_ue, active_faces=[True] * 6)
        self._B0_normal_ue = bnd_ops_ue.normal(data_space="Hcurl", test_space="H1")

        # ---- saddle point system: B = D M1, B^T = M1 D^T ---
        self._B = self._div @ self._M1

        self._block_domain = BlockVectorSpace(self.derham.coeff_spaces["1"], self.derham.coeff_spaces["1"])
        self._block_codomain_B = self.derham.coeff_spaces["0"]

        # B = [B1, -B2] acting on [u_i, u_e]
        self._B = BlockLinearOperator(
            self._block_domain, self._block_codomain_B,
            blocks=[[self._B, -self._B]],
        )

        self._block_domain_M = BlockVectorSpace(self._block_domain, self._block_codomain_B)

        _A_init = BlockLinearOperator(
            self._block_domain, self._block_domain,
            blocks=[[self._A11, None], [None, self._A22]],
        )
        _M_init = BlockLinearOperator(
            self._block_domain_M, self._block_domain_M,
            blocks=[[_A_init, self._B.T], [self._B, None]],
        )

        if self.options.solver in get_args(LiteralOptions.OptsSaddlePointSolver):
            self._Minv = inverse(
                _M_init,
                self.options.solver,
                A11=self._A11,
                A22=self._A22,
                B1=self._B,
                B2=-self._B2,
                recycle=self.options.solver_params.recycle,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
        else:
            self._Minv = inverse(
                _M_init,
                self.options.solver,
                recycle=self.options.solver_params.recycle,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )

        self._RHS = BlockVector(
            self._block_domain_M,
            blocks=[
                BlockVector(self._block_domain, blocks=[self._rhs_vec_u.vector, self._rhs_vec_ue.vector]),
                self._rhs_vec_phi.vector,
            ],
        )
        self._SOL = self._block_domain_M.zeros()

    # =========================================================================
    ### Time step
    # =========================================================================

    def __call__(self, dt):

        # --- rebuild system matrix if dt changed ---
        if dt != self._dt:
            self._dt = dt
            _A11 = self._A11 + self._M1 / dt
            _A = BlockLinearOperator(
                self._block_domain, self._block_domain,
                blocks=[[_A11, None], [None, self._A22]]
            )
            _M = BlockLinearOperator(
                self._block_domain_M, self._block_domain_M,
                blocks=[[_A, self._B.T], [self._B, None]]
            )
            self._Minv.linop = _M
            
            if self.options.solver in get_args(LiteralOptions.OptsSaddlePointSolver):
                self._Minv.update_A11(_A11)


        # --- copy current homogeneous solution ---
        self._u_0.vector = self.variables.u.spline.vector

        # --- assemble RHS for ions ---
        self._rhs_vec_u.vector = (
            self._hcurl_b_op_u.dot(
                self._M1_u.dot(self._src_u.vector)
                - self._A_i.dot(self._boundary_spline_u)
                - self._M1_u.dot(self._boundary_spline_u) / dt
                + self.options.nu * self._M1_u.dot(self._div_u.T.dot(self._M0inv_u.dot(
                    self._B0_normal_u.dot(self._boundary_spline_u)
                    - self._div_u.dot(self._M1_u.dot(self._boundary_spline_u))
                )))
            )
            + self._M1.dot(self._u_0.vector) / dt
        )

        # --- assemble RHS for electrons ---
        self._rhs_vec_ue.vector = (
            self._hcurl_b_op_ue.dot(
                self._M1_ue.dot(self._src_ue.vector)
                - self._A_e.dot(self._boundary_spline_ue)
                + self.options.mu * self.options.nu_e * self._M1_ue.dot(self._div_ue.T.dot(self._M0inv_ue.dot(
                    self._B0_normal_ue.dot(self._boundary_spline_ue)
                    - self._div_ue.dot(self._M1_ue.dot(self._boundary_spline_ue))
                )))
            )
        )

        # --- assemble RHS for quasineutrality ---
        self._rhs_vec_phi.vector = (
            self._B0_normal_u.dot(self._boundary_spline_u)
            - self._B0_normal_ue.dot(self._boundary_spline_ue)
            - self._div.dot(self._M1.dot(self._boundary_spline_u - self._boundary_spline_ue))
        )

        # --- build block RHS and solve ---
        self._Minv.dot(
            BlockVector(
                self._block_domain_M,
                blocks=[
                    BlockVector(self._block_domain, blocks=[self._rhs_vec_u.vector, self._rhs_vec_ue.vector]),
                    self._rhs_vec_phi.vector,
                ],
            ),
            out=self._SOL,
        )

        info = self._Minv.get_info()

        # --- update FEEC variables ---
        max_diffs = self.update_feec_variables(u=self._SOL[0][0], ue=self._SOL[0][1], phi=self._SOL[1])

        if self.options.solver_params.info and self._rank == 0:
            print(f"Status: {info['success']}, Iterations: {info['niter']}")
            print(f"Max diffs: {max_diffs}")