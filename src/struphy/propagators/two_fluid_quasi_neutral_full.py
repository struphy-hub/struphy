import logging
from dataclasses import dataclass
from typing import Callable, Literal, get_args
from warnings import warn

from feectools.api.essential_bc import apply_essential_bc_stencil
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.basic import ComposedLinearOperator, IdentityOperator, ZeroOperator
from feectools.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from feectools.linalg.solvers import inverse

from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator,
    BasisProjectionOperatorLocal,
    BasisProjectionOperators,
    CoordinateProjector,
)
from struphy.feec.mass import L2Projector, WeightedMassOperator, WeightedMassOperators
from struphy.io.options import LiteralOptions
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
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

    def __init__(self):
        self.variables = self.Variables()

    # =========================================================================
    ### Options
    # =========================================================================

    @dataclass
    class Options:
        """Configuration options for :class:`TwoFluidQuasiNeutralFull`.

        Parameters
        ----------
        nu : float, default=1.0
            Ion viscosity coefficient.
        nu_e : float, default=1.0
            Electron viscosity coefficient.
        eps_norm : float, default=1e-3
            Normalization/scaling parameter in Lorentz coupling terms.
        boundary_data_u : dict[tuple[int, int], Callable] or None, default=None
            Inhomogeneous Dirichlet data for ion velocity faces.
        boundary_data_ue : dict[tuple[int, int], Callable] or None, default=None
            Inhomogeneous Dirichlet data for electron velocity faces.
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
        eps_norm: float = 1e-3

        boundary_data_u: dict[tuple[int, int], Callable] | None = None
        boundary_data_ue: dict[tuple[int, int], Callable] | None = None

        source_u: Callable | None = None
        source_ue: Callable | None = None

        stab_sigma: float | None = None

        solver: LiteralOptions.OptsGenSolver = "gmres"
        solver_params: SolverParameters | None = None

        def __post_init__(self):
            # --- physical parameter sanity checks ---
            if self.nu < 0:
                raise ValueError(f"nu must be non-negative, got {self.nu}")
            if self.nu_e < 0:
                raise ValueError(f"nu_e must be non-negative, got {self.nu_e}")
            if self.eps_norm <= 0:
                raise ValueError(f"eps_norm must be positive, got {self.eps_norm}")

            # --- warn if no source terms ---
            if self.source_u is None:
                warn("No source_u specified — defaulting to zero.")
            if self.source_ue is None:
                warn("No source_ue specified — defaulting to zero.")

            # --- defaults ---
            if self.stab_sigma is None:
                warn("stab_sigma not specified, defaulting to 0.0")
                self.stab_sigma = 0.0

            check_option(self.solver, LiteralOptions.OptsGenSolver, LiteralOptions.OptsSaddlePointSolver)
            if self.solver_params is None:
                self.solver_params = SolverParameters()

    @property
    def options(self) -> Options:
        assert hasattr(self, "_options"), "Options not set."
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                logger.info(f"  {k}: {v}")
        self._options = new

    # =========================================================================
    ### Boundary condition helpers
    # =========================================================================

    def _get_dirichlet_faces(self):
        """Infer which faces have Dirichlet BCs by comparing derham and derham_v0.

        A face is Dirichlet if it is unclamped in derham but clamped in derham_v0
        (i.e. lifting is True there).
        """
        faces = []
        derham = self.derham
        derham_v0 = derham

        if derham_v0 is None:
            return faces

        bc = derham.dirichlet_bc
        bc_v0 = derham_v0.dirichlet_bc

        for d in range(3):
            if derham.spl_kind[d]:
                continue  # periodic axis, no Dirichlet
            for s, side in enumerate((-1, 1)):
                # clamped in v0 but not in derham => this is a lifted (inhom Dirichlet) face
                unclamped = not bc[d][s]
                clamped_v0 = bc_v0[d][s] if bc_v0 is not None else False
                if unclamped and clamped_v0:
                    faces.append((d, side))
                # clamped in both => homogeneous Dirichlet, also need to zero DOFs
                elif bc[d][s] and clamped_v0:
                    faces.append((d, side))
        return faces

    def _apply_essential_bc(self, vec):
        """Zero out Dirichlet DOFs, inferred from derham vs derham_v0."""
        for d, side in self._dirichlet_faces:
            apply_essential_bc_stencil(vec[0], axis=d, ext=side, order=0)

    # =========================================================================
    ### Allocate
    # =========================================================================

    def allocate(self, verbose=False):

        self.verbose = verbose
        self._rank = self.derham.comm.Get_rank() if self.derham.comm is not None else 0
        self._dt = None

        # ---- v0 de Rham complex (from derham.derham_v0) ----------------------
        self._derham_v0 = self.derham

        self._mass_ops_v0 = WeightedMassOperators(
            self._derham_v0,
            self.domain,
            eq_mhd=self.mass_ops.eq_mhd,
        )
        self._basis_ops_v0 = BasisProjectionOperators(
            self._derham_v0,
            self.domain,
            verbose=self.options.solver_params.verbose,
            eq_mhd=self.basis_ops.weights["eq_mhd"],
        )

        # ---- Dirichlet faces (inferred from derham vs derham_v0) -------------

        self._dirichlet_faces = self._get_dirichlet_faces()

        # ---- unconstrained operators (for RHS assembly) ----------------------

        self._M2 = self.mass_ops.M2
        self._M2B = -self.mass_ops.M2B
        self._div = self.derham.div
        self._curl = self.derham.curl
        self._S21 = self.basis_ops.S21

        self._lapl = (
            self._div.T @ self.mass_ops.M3 @ self._div + self._S21.T @ self._curl.T @ self._M2 @ self._curl @ self._S21
        )

        self._A11 = -self._M2B / self.options.eps_norm + self.options.nu * self._lapl
        self._A22 = (
            -self.options.stab_sigma * IdentityOperator(self.derham.V2)
            + self._M2B / self.options.eps_norm
            + self.options.nu_e * self._lapl
        )

        # ---- constrained operators (for system matrix) -----------------------

        self._M2_v0 = self._mass_ops_v0.M2
        self._M3_v0 = self._mass_ops_v0.M3
        self._M2B_v0 = -self._mass_ops_v0.M2B
        self._div_v0 = self._derham_v0.div
        self._curl_v0 = self._derham_v0.curl
        self._S21_v0 = self._basis_ops_v0.S21

        self._lapl_v0 = (
            self._div_v0.T @ self._M3_v0 @ self._div_v0
            + self._S21_v0.T @ self._curl_v0.T @ self._M2_v0 @ self._curl_v0 @ self._S21_v0
        )

        self._A11_v0 = -self._M2B_v0 / self.options.eps_norm + self.options.nu * self._lapl_v0
        self._A22_v0 = (
            -self.options.stab_sigma * IdentityOperator(self._derham_v0.V2)
            + self._M2B_v0 / self.options.eps_norm
            + self.options.nu_e * self._lapl_v0
        )

        # ---- block saddle-point system ----------------------------------------

        self._block_domain_v0 = BlockVectorSpace(self._derham_v0.V2, self._derham_v0.V2)
        self._block_codomain_v0 = self._block_domain_v0
        self._block_codomain_B_v0 = self._derham_v0.V3

        self._B1_v0 = -self._M3_v0 @ self._div_v0
        self._B2_v0 = self._M3_v0 @ self._div_v0

        self._B_v0 = BlockLinearOperator(
            self._block_domain_v0, self._block_codomain_B_v0, blocks=[[self._B1_v0, self._B2_v0]]
        )

        self._block_domain_M = BlockVectorSpace(self._block_domain_v0, self._block_codomain_B_v0)

        _A_init = BlockLinearOperator(
            self._block_domain_v0, self._block_codomain_v0, blocks=[[self._A11_v0, None], [None, self._A22_v0]]
        )
        _M_init = BlockLinearOperator(
            self._block_domain_M, self._block_domain_M, blocks=[[_A_init, self._B_v0.T], [self._B_v0, None]]
        )

        if self.options.solver in get_args(LiteralOptions.OptsSaddlePointSolver):
            self._Minv = inverse(
                _M_init,
                self.options.solver,
                A11=self._A11_v0,
                A22=self._A22_v0,
                B1=self._B1_v0,
                B2=self._B2_v0,
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

        # ---- projector -------------------------------------------------------

        self._projector = L2Projector(space_id="Hdiv", mass_ops=self.mass_ops)

        # ---- solution spline functions (unconstrained) -----------------------

        self._u = self.derham.create_spline_function("u", space_id="Hdiv")
        self._ue = self.derham.create_spline_function("ue", space_id="Hdiv")
        self._phi = self.derham.create_spline_function("phi", space_id="L2")

        # ---- BC lifts (unconstrained) ----------------------------------------

        self._u_prime = self.derham.create_spline_function("u_prime", space_id="Hdiv")
        self._ue_prime = self.derham.create_spline_function("ue_prime", space_id="Hdiv")

        for u_prime, boundary_data in [
            (self._u_prime, self.options.boundary_data_u),
            (self._ue_prime, self.options.boundary_data_ue),
        ]:
            if boundary_data is None:
                continue
            for (d, side), f_bc in boundary_data.items():
                if (d, side) in self._dirichlet_faces:
                    bc_pulled = lambda *etas, f=f_bc: self.domain.pull(
                        [
                            lambda x, y, z, f=f: f(x, y, z)[0],
                            lambda x, y, z, f=f: f(x, y, z)[1],
                            lambda x, y, z, f=f: f(x, y, z)[2],
                        ],
                        *etas,
                        kind="2",
                    )
                    _vec = self._projector(
                        [
                            lambda *etas: bc_pulled(*etas)[0],
                            lambda *etas: bc_pulled(*etas)[1],
                            lambda *etas: bc_pulled(*etas)[2],
                        ]
                    )
                    for d2, side2 in self._dirichlet_faces:
                        if (d2, side2) != (d, side):
                            apply_essential_bc_stencil(_vec[0], axis=d2, ext=side2, order=0)
                    u_prime.vector += _vec

        self._u_prime_v0 = self._derham_v0.create_spline_function("u_prime_v0", space_id="Hdiv")
        self._ue_prime_v0 = self._derham_v0.create_spline_function("ue_prime_v0", space_id="Hdiv")

        self._u_prime_v0.vector = self._u_prime.vector
        self._ue_prime_v0.vector = self._ue_prime.vector

        # ---- projected source terms (unconstrained) --------------------------

        self._rhs_u = self.derham.create_spline_function("rhs_u", space_id="Hdiv")
        self._rhs_ue = self.derham.create_spline_function("rhs_ue", space_id="Hdiv")

        for rhs, source in [(self._rhs_u, self.options.source_u), (self._rhs_ue, self.options.source_ue)]:
            if source is not None:
                src_pulled = lambda *etas, f=source: self.domain.pull(
                    [
                        lambda x, y, z, f=f: f(x, y, z)[0],
                        lambda x, y, z, f=f: f(x, y, z)[1],
                        lambda x, y, z, f=f: f(x, y, z)[2],
                    ],
                    *etas,
                    kind="2",
                )
                rhs.vector = self._projector.get_dofs(
                    [
                        lambda *etas: src_pulled(*etas)[0],
                        lambda *etas: src_pulled(*etas)[1],
                        lambda *etas: src_pulled(*etas)[2],
                    ]
                )

        # ---- pre-allocated RHS vectors (v0, reused each time step) -----------

        self._rhs_vec_u = self._derham_v0.create_spline_function("rhs_vec_u", space_id="Hdiv")
        self._rhs_vec_ue = self._derham_v0.create_spline_function("rhs_vec_ue", space_id="Hdiv")

    # =========================================================================
    ### Time step
    # =========================================================================

    def __call__(self, dt):

        # --- copy current state ---
        self._u.vector = self.variables.u.spline.vector
        self._ue.vector = self.variables.ue.spline.vector

        # --- rebuild system matrix if dt changed ---
        if dt != self._dt:  #  TODO change uzawa A11 block too
            self._dt = dt
            _A = BlockLinearOperator(
                self._block_domain_v0,
                self._block_codomain_v0,
                blocks=[[self._A11_v0 + self._M2_v0 / dt, None], [None, self._A22_v0]],
            )

            _M = BlockLinearOperator(
                self._block_domain_M, self._block_domain_M, blocks=[[_A, self._B_v0.T], [self._B_v0, None]]
            )
            self._Minv.linop = _M

        # --- assemble RHS in unconstrained space, then zero boundary DOFs ---
        # ion:      F1 = rhs_u + M2/dt * u - (A11 + M2/dt) * u'
        # electron: F2 = rhs_ue - A22 * ue'
        self._rhs_vec_u.vector = (
            self._rhs_u.vector  # TODO boundary operator
            + self._M2.dot(self._u.vector) / dt
            - self._A11.dot(self._u_prime.vector)
            - self._M2.dot(self._u_prime.vector) / dt
        )
        self._rhs_vec_ue.vector = self._rhs_ue.vector - self._A22.dot(self._ue_prime.vector)

        self._apply_essential_bc(self._rhs_vec_u.vector)
        self._apply_essential_bc(self._rhs_vec_ue.vector)

        # --- build block RHS and solve ---
        _F = BlockVector(self._block_domain_v0, blocks=[self._rhs_vec_u.vector, self._rhs_vec_ue.vector])
        _RHS = BlockVector(self._block_domain_M, blocks=[_F, self._block_codomain_B_v0.zeros()])

        _sol = self._Minv.dot(_RHS)
        info = self._Minv.get_info()

        # --- reconstruct full solution: u = u_0 + u' ---
        self._u.vector = _sol[0][0] + self._u_prime_v0.vector
        self._ue.vector = _sol[0][1] + self._ue_prime_v0.vector
        self._phi.vector = _sol[1]

        # --- update FEEC variables ---
        max_diffs = self.update_feec_variables(u=self._u.vector, ue=self._ue.vector, phi=self._phi.vector)

        if self.options.solver_params.info and self._rank == 0:
            logger.info(f"Status: {info['success']}, Iterations: {info['niter']}")
            logger.info(f"Max diffs: {max_diffs}")
            logger.info(f"Status: {info['success']}, Iterations: {info['niter']}")
            logger.info(f"Max diffs: {max_diffs}")
