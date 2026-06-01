import logging
from dataclasses import dataclass
from typing import Callable, get_args
from warnings import warn

from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import IdentityOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse

from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.geometry.utilities import TransformedPformComponent
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable
from struphy.propagators.base import Propagator
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


class TwoFluidQuasiNeutralFull(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in H(\textnormal{div})`, :math:`\mathbf u_e \in H(\textnormal{div})` and  :math:`\mathbf \phi \in L^2` such that

    .. math::

        \int_{\Omega} \partial_t  \mathbf{u}\cdot \mathbf{v} \, \textrm d\mathbf{x} &=  \int_{\Omega}  \phi \nabla \! \cdot \! \mathbf{v} \, \textrm d\mathbf{x}  + \int_{\Omega}  \mathbf{u}\! \times \! \mathbf{B}_0 \cdot \mathbf{v} \, \textrm d\mathbf{x} + \nu \int_{\Omega} \nabla \mathbf{u}\! : \! \nabla \mathbf{v} \, \textrm d\mathbf{x} + \int_{\Omega} f \mathbf{v} \, \textrm d\mathbf{x} \qquad \forall \, \mathbf{v} \in H(\textrm{div}) \,.
        \\[2mm]
        0 &= - \int_{\Omega} \phi \nabla \! \cdot \! \mathbf{v_e} \, \textrm d\mathbf{x} - \int_{\Omega} \mathbf{u_e} \! \times \! \mathbf{B}_0 \cdot \mathbf{v_e} \, \textrm d\mathbf{x}  + \nu_e \int_{\Omega} \nabla \mathbf{u_e}  \!: \! \nabla \mathbf{v_e} \, \textrm d\mathbf{x} + \int_{\Omega} f_e \mathbf{v_e} \, \textrm d\mathbf{x} \qquad \forall \ \mathbf{v_e} \in H(\textrm{div}) \,.
        \\[2mm]
        0 &= \int_{\Omega} \psi \nabla \cdot (\mathbf{u}-\mathbf{u_e}) \, \textrm d\mathbf{x} \qquad \forall \, \psi \in L^2 \,.

    :ref:`time_discret`: fully implicit.
    """

    # =========================================================================
    ### State variables (ion velocity u, electron velocity ue, pressure phi)
    # =========================================================================

    class Variables:
        """Container for variables advanced by :class:`TwoFluidQuasiNeutralFull`.

        Attributes
        ----------
        u : FEECVariable or None
            Ion velocity variable in ``"Hdiv"`` space.
        ue : FEECVariable or None
            Electron velocity variable in ``"Hdiv"`` space.
        phi : FEECVariable or None
            Electrostatic potential variable in ``"L2"`` space.
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
            assert new.space == "Hdiv"
            self._u = new

        @property
        def ue(self) -> FEECVariable | None:
            return self._ue

        @ue.setter
        def ue(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._ue = new

        @property
        def phi(self) -> FEECVariable | None:
            return self._phi

        @phi.setter
        def phi(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._phi = new

    def __init__(self, allocate_variables: bool = False):
        self.variables = self.Variables()
        
        if allocate_variables:
            self.variables.u = FEECVariable(space="Hdiv")
            self.variables.ue = FEECVariable(space="Hdiv")
            self.variables.phi = FEECVariable(space="L2")
            
            self.variables.u.allocate(derham=self.derham, domain=self.domain, equil=self.projected_equil.equil)
            self.variables.ue.allocate(derham=self.derham, domain=self.domain, equil=self.projected_equil.equil)
            self.variables.phi.allocate(derham=self.derham, domain=self.domain, equil=self.projected_equil.equil)


    # =========================================================================
    ### Options
    # =========================================================================

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`TwoFluidQuasiNeutralFull`.

        Parameters
        ----------
        nu : float, default=1.0
            Ion viscosity coefficient.
        nu_e : float, default=1.0
            Electron viscosity coefficient.
        eps_norm : float, default=1e-3
            Normalization/scaling parameter in Lorentz coupling terms.
        source_u : Callable or None, default=None
            Source term for ion momentum equation.
        source_ue : Callable or None, default=None
            Source term for electron momentum equation.
        stab_sigma : float or None, default=None
            Optional stabilization coefficient for electron block.
        solver : LiteralOptions.OptsGenSolver, default="gmres"
            Linear/saddle-point solver used for the global system.
        solver_params : SolverParameters or None, default=None
            Solver controls.
        """

        nu: float = 1.0
        nu_e: float = 1.0
        eps_norm: float | None = None

        source_u: Callable | None = None
        source_ue: Callable | None = None

        stab_sigma: float = 0.0
        solver: LiteralOptions.OptsGenSolver = "gmres"
        solver_params: SolverParameters | None = None

        def __post_init__(self):
            # --- warn if no source terms ---
            if self.source_u is None:
                warn("No source_u specified — defaulting to zero.")
            if self.source_ue is None:
                warn("No source_ue specified — defaulting to zero.")
            if self.eps_norm is None:
                warn("No eps_norm specified — will default to ion cyclotron parameter epsilon in allocate.")

            # --- physical parameter sanity checks ---
            if self.nu < 0:
                raise ValueError(f"nu must be non-negative, got {self.nu}")
            if self.nu_e < 0:
                raise ValueError(f"nu_e must be non-negative, got {self.nu_e}")
            if self.eps_norm is not None and self.eps_norm <= 0:
                raise ValueError(f"eps_norm must be positive, got {self.eps_norm}")

            # --- defaults ---
            if self.stab_sigma is None:
                warn("stab_sigma not specified, defaulting to 0.0")
                self.stab_sigma = 0.0

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

        # ---- solution splines (constrained) and u in unconstrained space -----
        self._u_0 = self.derham.create_spline_function("u", space_id="Hdiv")

        # boundary splines (u', ue') in unconstrained space — zero vectors if no lifting
        self._boundary_spline_u = (
            self.variables.u.boundary_spline.vector
            if self._has_lifting_u
            else self._derham_lift_u.coeff_spaces["2"].zeros()
        )
        self._boundary_spline_ue = (
            self.variables.ue.boundary_spline.vector
            if self._has_lifting_ue
            else self._derham_lift_ue.coeff_spaces["2"].zeros()
        )

        # boundary operators
        self._b_op_u = (
            self.variables.u.boundary_op_lift
            if self._has_lifting_u
            else IdentityOperator(self.derham.coeff_spaces["2"])
        )
        self._b_op_ue = (
            self.variables.ue.boundary_op_lift
            if self._has_lifting_ue
            else IdentityOperator(self.derham.coeff_spaces["2"])
        )

        # pre-allocated RHS vectors (constrained, after boundary operator)
        self._rhs_vec_u = self.derham.create_spline_function("rhs_vec_u", space_id="Hdiv")
        self._rhs_vec_ue = self.derham.create_spline_function("rhs_vec_ue", space_id="Hdiv")
        self._rhs_vec_phi = self.derham.create_spline_function("rhs_vec_phi", space_id="L2")

        self._div_boundary_u = self.derham.create_spline_function("div_boundary_u", space_id="L2")
        self._div_boundary_ue = self.derham.create_spline_function("div_boundary_ue", space_id="L2")

        # ---- source terms projected onto unconstrained space -----------------
        self._src_u = self._derham_lift_u.create_spline_function("rhs_u", space_id="Hdiv")
        self._src_ue = self._derham_lift_ue.create_spline_function("rhs_ue", space_id="Hdiv")

        for rhs, source, derham_lift in [
            (self._src_u, self.options.source_u, self._derham_lift_u),
            (self._src_ue, self.options.source_ue, self._derham_lift_ue),
        ]:
            if source is not None:
                fun_vec = [lambda x, y, z, f=source, c=c: f(x, y, z)[c] for c in range(3)]
                fun = [
                    TransformedPformComponent(
                        fun_vec,
                        "physical",
                        "2",
                        comp=comp,
                        domain=self.domain,
                    )
                    for comp in range(3)
                ]
                rhs.vector = derham_lift.projectors["2"](fun)

        # ---- unconstrained mass/basis operators (for RHS assembly) -----------

        self._mass_ops_lift_u = WeightedMassOperators(
            self._derham_lift_u,
            self.domain,
            eq_mhd=self.mass_ops.eq_mhd,
        )
        self._mass_ops_lift_ue = WeightedMassOperators(
            self._derham_lift_ue,
            self.domain,
            eq_mhd=self.mass_ops.eq_mhd,
        )
        self._basis_ops_lift_u = BasisProjectionOperators(
            self._derham_lift_u,
            self.domain,
            verbose=self.options.solver_params.verbose,
            eq_mhd=self.basis_ops.weights["eq_mhd"],
        )
        self._basis_ops_lift_ue = BasisProjectionOperators(
            self._derham_lift_ue,
            self.domain,
            eq_mhd=self.basis_ops.weights["eq_mhd"],
        )

        self._M2_u = self._mass_ops_lift_u.M2
        self._M2B_u = -self._mass_ops_lift_u.M2B
        self._div_u = self._derham_lift_u.div
        self._curl_u = self._derham_lift_u.curl
        self._S21_u = self._basis_ops_lift_u.S21

        self._lapl_u = (
            self._div_u.T @ self._mass_ops_lift_u.M3 @ self._div_u
            + self._S21_u.T @ self._curl_u.T @ self._M2_u @ self._curl_u @ self._S21_u
        )

        self._A11_u = -self._M2B_u / self.options.eps_norm + self.options.nu * self._lapl_u

        self._M2_ue = self._mass_ops_lift_ue.M2
        self._M2B_ue = -self._mass_ops_lift_ue.M2B
        self._div_ue = self._derham_lift_ue.div
        self._curl_ue = self._derham_lift_ue.curl
        self._S21_ue = self._basis_ops_lift_ue.S21

        self._lapl_ue = (
            self._div_ue.T @ self._mass_ops_lift_ue.M3 @ self._div_ue
            + self._S21_ue.T @ self._curl_ue.T @ self._M2_ue @ self._curl_ue @ self._S21_ue
        )

        self._A22_ue = (
            self.options.stab_sigma * IdentityOperator(self._derham_lift_ue.coeff_spaces["2"])
            + self._M2B_ue / self.options.eps_norm
            + self.options.nu_e * self._lapl_ue
        )

        # ---- constrained operators (for system matrix, built from self.derham) ---

        self._M2 = self.mass_ops.M2
        self._M3 = self.mass_ops.M3
        self._M2B = -self.mass_ops.M2B
        self._div = self.derham.div
        self._curl = self.derham.curl
        self._S21 = self.basis_ops.S21

        self._lapl_v0 = (
            self._div.T @ self._M3 @ self._div + self._S21.T @ self._curl.T @ self._M2 @ self._curl @ self._S21
        )

        self._A11 = -self._M2B / self.options.eps_norm + self.options.nu * self._lapl_v0
        self._A22 = (
            self.options.stab_sigma * IdentityOperator(self.derham.coeff_spaces["2"])
            + self._M2B / self.options.eps_norm
            + self.options.nu_e * self._lapl_v0
        )

        # ---- block saddle-point system ----------------------------------------

        self._block_domain = BlockVectorSpace(self.derham.coeff_spaces["2"], self.derham.coeff_spaces["2"])
        self._block_codomain_B = self.derham.coeff_spaces["3"]

        self._B1 = -self._M3 @ self._div
        self._B2 = self._M3 @ self._div

        self._B = BlockLinearOperator(self._block_domain, self._block_codomain_B, blocks=[[self._B1, self._B2]])

        self._block_domain_M = BlockVectorSpace(self._block_domain, self._block_codomain_B)

        _A_init = BlockLinearOperator(
            self._block_domain, self._block_domain, blocks=[[self._A11, None], [None, self._A22]]
        )
        _M_init = BlockLinearOperator(
            self._block_domain_M, self._block_domain_M, blocks=[[_A_init, self._B.T], [self._B, None]]
        )

        if self.options.solver in get_args(LiteralOptions.OptsSaddlePointSolver):
            self._Minv = inverse(
                _M_init,
                self.options.solver,
                A11=self._A11,
                A22=self._A22,
                B1=self._B1,
                B2=self._B2,
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

        # --- copy current homogeneous solution ---
        self._u_0.vector = self.variables.u.spline.vector

        # --- assemble RHS fully in unconstrained space, then enforce essential BCs ---
        self._rhs_vec_u.vector = (
            self._b_op_u.dot(
                (
                    self._M2_u.dot(self._src_u.vector)
                    - self._A11_u.dot(self._boundary_spline_u)
                    - self._M2_u.dot(self._boundary_spline_u) / dt
                )
            )
            + self._M2.dot(self._u_0.vector) / dt
        )

        self._rhs_vec_ue.vector = self._b_op_ue.dot(
            (self._M2_ue.dot(self._src_ue.vector) - self._A22_ue.dot(self._boundary_spline_ue))
        )

        self._div_boundary_u.vector = self._div_u.dot(self._boundary_spline_u)
        self._div_boundary_ue.vector = self._div_ue.dot(self._boundary_spline_ue)

        self._rhs_vec_phi.vector = self.mass_ops.M3.dot(self._div_boundary_u.vector) - self.mass_ops.M3.dot(
            self._div_boundary_ue.vector
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
