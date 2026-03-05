
import copy
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, get_args, cast
from warnings import warn

import cunumpy as xp
import scipy as sc
from line_profiler import profile
from matplotlib import pyplot as plt
from numpy import zeros
from psydac.api.essential_bc import apply_essential_bc_stencil
from psydac.ddm.mpi import mpi as MPI
from psydac.linalg.basic import ComposedLinearOperator, IdentityOperator, ZeroOperator, InverseLinearOperator
from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.solvers import inverse
from psydac.linalg.stencil import StencilVector

import struphy.feec.utilities as util
from struphy.examples.restelli2018 import callables
from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator, BasisProjectionOperatorLocal,
    BasisProjectionOperators, CoordinateProjector,
)
from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner, MassMatrixPreconditioner
from struphy.feec.projectors import L2Projector
from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.feec.variational_utilities import (
    BracketOperator, Hdiv0_transport_operator, InternalEnergyEvaluator,
    KineticEnergyEvaluator, Pressure_transport_operator,
)
from struphy.fields_background.equils import set_defaults
from struphy.geometry.utilities import TransformedPformComponent
from struphy.initial import perturbations
from struphy.io.options import (
    OptsDirectSolver, OptsGenSolver, OptsMassPrecond, OptsNonlinearSolver,
    OptsSaddlePointSolver, OptsSymmSolver, OptsVecSpace, check_option,
)
from struphy.io.setup import descend_options_dict
from struphy.kinetic_background.base import Maxwellian
from struphy.kinetic_background.maxwellians import GyroMaxwellian2D, Maxwellian3D
from struphy.linear_algebra.saddle_point import SaddlePointSolver
from struphy.linear_algebra.schur_solver import SchurSolver, SchurSolverFull
from struphy.linear_algebra.solver import NonlinearSolverParameters, SolverParameters
from struphy.models.species import Species
from struphy.models.variables import FEECVariable, PICVariable, SPHVariable, Variable
from struphy.ode.solvers import ODEsolverFEEC
from struphy.ode.utils import ButcherTableau, OptsButcher
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.base import Particles
from struphy.pic.particles import Particles5D, Particles6D
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel


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

    class Variables():
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
    class Options():

        nu: float | None = None
        nu_e: float | None = None
        eps_norm: float | None = None

        # boundary conditions per species
        # supported kinds: "periodic", "dirichlet"
        # future:          "neumann", "robin"
        boundary_conditions_u:  dict[tuple[int, int], Literal["periodic", "dirichlet"]] | None = None
        boundary_conditions_ue: dict[tuple[int, int], Literal["periodic", "dirichlet"]] | None = None
        boundary_data_u:        dict[tuple[int, int], Callable] | None = None
        boundary_data_ue:       dict[tuple[int, int], Callable] | None = None

        source_u:  Callable | None = None
        source_ue: Callable | None = None

        stab_sigma: float | None = None

        solver: OptsGenSolver = "gmres"
        solver_params: SolverParameters | None = None

        def __post_init__(self):

            # --- required parameters ---
            assert self.nu                    is not None, "nu must be specified"
            assert self.nu_e                  is not None, "nu_e must be specified"
            assert self.eps_norm              is not None, "eps_norm must be specified"
            assert self.boundary_conditions_u  is not None, "boundary_conditions_u must be specified"
            assert self.boundary_conditions_ue is not None, "boundary_conditions_ue must be specified"

            # --- physical parameter sanity checks ---
            if self.nu < 0:
                raise ValueError(f"nu must be non-negative, got {self.nu}")
            if self.nu_e < 0:
                raise ValueError(f"nu_e must be non-negative, got {self.nu_e}")
            if self.eps_norm <= 0:
                raise ValueError(f"eps_norm must be positive, got {self.eps_norm}")

            # --- check all axes are covered ---
            for name, bcs in [("boundary_conditions_u",  self.boundary_conditions_u),
                               ("boundary_conditions_ue", self.boundary_conditions_ue)]:
                for d in range(3):
                    for side in (-1, 1):
                        assert (d, side) in bcs, \
                            f"{name} is missing entry for axis {d} side {side}"

            # --- periodic consistency: periodic must be paired on both sides ---
            for name, bcs in [("boundary_conditions_u",  self.boundary_conditions_u),
                               ("boundary_conditions_ue", self.boundary_conditions_ue)]:
                for (d, side), kind in bcs.items():
                    if kind == "periodic":
                        assert bcs.get((d, -side)) == "periodic", \
                            f"{name}: axis {d} side {side} is periodic but opposite side is not"

            # --- ions and electrons must agree on which axes are periodic ---
            for d in range(3):
                u_left   = self.boundary_conditions_u.get((d, -1))
                ue_left  = self.boundary_conditions_ue.get((d, -1))
                u_right  = self.boundary_conditions_u.get((d,  1))
                ue_right = self.boundary_conditions_ue.get((d,  1))
                u_periodic  = (u_left  == "periodic")
                ue_periodic = (ue_left == "periodic")
                if u_periodic != ue_periodic:
                    raise ValueError(
                        f"Axis {d}: ions and electrons must both be periodic or both non-periodic, "
                        f"got u={u_left}/{u_right}, ue={ue_left}/{ue_right}"
                    )

            # --- warn for Dirichlet faces with no boundary data ---
            for species, bcs, data, label in [
                ("u",  self.boundary_conditions_u,  self.boundary_data_u,  "boundary_data_u"),
                ("ue", self.boundary_conditions_ue, self.boundary_data_ue, "boundary_data_ue"),
            ]:
                has_dirichlet = any(v == "dirichlet" for v in bcs.values())
                if has_dirichlet:
                    if data is None:
                        warn(f"Dirichlet BCs specified for {species} but no {label} given "
                             f"— defaulting to homogeneous Dirichlet on all faces.")
                    else:
                        for (d, side), kind in bcs.items():
                            if kind == "dirichlet" and (d, side) not in data:
                                warn(f"No {label} given for axis {d} side {side} "
                                     f"— defaulting to homogeneous Dirichlet.")

            # --- warn if no source terms ---
            if self.source_u is None:
                warn("No source_u specified — defaulting to zero.")
            if self.source_ue is None:
                warn("No source_ue specified — defaulting to zero.")

            # --- defaults ---
            if self.stab_sigma is None:
                warn("stab_sigma not specified, defaulting to 0.0")
                self.stab_sigma = 0.0

            check_option(self.solver, OptsGenSolver)
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
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    # =========================================================================
    ### Boundary condition helpers
    # =========================================================================

    def _bc_to_dirichlet_flags(self, boundary_conditions, spl_kind):

        dirichlet_bc = []
        for d in range(3):
            if spl_kind[d]:  # periodic spline — no clamping ever
                dirichlet_bc.append((False, False))
            else:
                left  = boundary_conditions.get((d, -1)) == "dirichlet"
                right = boundary_conditions.get((d,  1)) == "dirichlet"
                dirichlet_bc.append((left, right))
        return tuple(tuple(bc) for bc in dirichlet_bc)

    def _apply_boundary_conditions(self, vec, boundary_conditions):
        """Zero out Dirichlet DOFs on the given stencil vector."""
        for (d, side), kind in boundary_conditions.items():
            if kind == "dirichlet":
                apply_essential_bc_stencil(vec[0], axis=d, ext=side, order=0)
            # future: neumann and robin require no zeroing here

    # =========================================================================
    ### Allocate
    # =========================================================================

    def allocate(self):

        self._rank = self.derham.comm.Get_rank() if self.derham.comm is not None else 0
        self._dt   = None

        # ---- constrained (v0) de Rham complex --------------------------------

        _dirichlet_u  = self._bc_to_dirichlet_flags(self.options.boundary_conditions_u,  self.derham.spl_kind)
        _dirichlet_ue = self._bc_to_dirichlet_flags(self.options.boundary_conditions_ue, self.derham.spl_kind)
        _dirichlet_bc = tuple(
            (l_u or l_ue, r_u or r_ue)
            for (l_u, r_u), (l_ue, r_ue) in zip(_dirichlet_u, _dirichlet_ue)
        )

        self._derham_v0    = Derham(
            self.derham.Nel, self.derham.p, self.derham.spl_kind,
            domain=self.domain, dirichlet_bc=_dirichlet_bc,
        )
        self._mass_ops_v0  = WeightedMassOperators(
            self._derham_v0, self.domain,
            verbose=self.options.solver_params.verbose,
            eq_mhd=self.mass_ops.weights["eq_mhd"],
        )
        self._basis_ops_v0 = BasisProjectionOperators(
            self._derham_v0, self.domain,
            verbose=self.options.solver_params.verbose,
            eq_mhd=self.basis_ops.weights["eq_mhd"],
        )

        # ---- unconstrained operators (for RHS assembly) ----------------------

        self._M2   = self.mass_ops.M2
        self._M2B  = - self.mass_ops.M2B
        self._div  = self.derham.div
        self._curl = self.derham.curl
        self._S21  = self.basis_ops.S21

        self._lapl = (self._div.T @ self.mass_ops.M3 @ self._div
                    + self._S21.T @ self._curl.T @ self._M2 @ self._curl @ self._S21)

        self._A11 = - self._M2B / self.options.eps_norm + self.options.nu   * self._lapl
        self._A22 = (- self.options.stab_sigma * IdentityOperator(self._A11.domain)
                    + self._M2B / self.options.eps_norm + self.options.nu_e * self._lapl)

        # ---- constrained operators (for system matrix) -----------------------

        self._M2_v0   = self._mass_ops_v0.M2
        self._M3_v0   = self._mass_ops_v0.M3
        self._M2B_v0  = - self._mass_ops_v0.M2B
        self._div_v0  = self._derham_v0.div
        self._curl_v0 = self._derham_v0.curl
        self._S21_v0  = self._basis_ops_v0.S21

        self._lapl_v0 = (self._div_v0.T @ self._M3_v0 @ self._div_v0
                       + self._S21_v0.T @ self._curl_v0.T @ self._M2_v0 @ self._curl_v0 @ self._S21_v0)

        self._A11_v0 = - self._M2B_v0 / self.options.eps_norm + self.options.nu   * self._lapl_v0
        self._A22_v0 = (- self.options.stab_sigma * IdentityOperator(self._A11_v0.domain)
                       + self._M2B_v0 / self.options.eps_norm + self.options.nu_e * self._lapl_v0)

        # ---- block saddle-point system ----------------------------------------

        self._block_domain_v0     = BlockVectorSpace(self._A11_v0.domain, self._A22_v0.domain)
        self._block_codomain_v0   = self._block_domain_v0
        self._block_codomain_B_v0 = self._M3_v0.codomain

        self._B1_v0 = - self._M3_v0 @ self._div_v0
        self._B2_v0 =   self._M3_v0 @ self._div_v0

        self._B_v0 = BlockLinearOperator(
            self._block_domain_v0, self._block_codomain_B_v0,
            blocks=[[self._B1_v0, self._B2_v0]]
        )

        self._block_domain_M = BlockVectorSpace(self._block_domain_v0, self._block_codomain_B_v0)

        _A_init = BlockLinearOperator(
            self._block_domain_v0, self._block_codomain_v0,
            blocks=[[self._A11_v0, None], [None, self._A22_v0]]
        )
        _M_init = BlockLinearOperator(
            self._block_domain_M, self._block_domain_M,
            blocks=[[_A_init, self._B_v0.T], [self._B_v0, None]]
        )
        self._Minv = cast(InverseLinearOperator, inverse(
            _M_init, self.options.solver,
            x0=None,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        ))

        # ---- projector -------------------------------------------------------

        self._projector = L2Projector(space_id="Hdiv", mass_ops=self.mass_ops)

        # ---- solution spline functions (unconstrained) -----------------------

        self._u   = self.derham.create_spline_function("u",   space_id="Hdiv")
        self._ue  = self.derham.create_spline_function("ue",  space_id="Hdiv")
        self._phi = self.derham.create_spline_function("phi", space_id="L2")

        # ---- BC lifts (unconstrained) ----------------------------------------

        self._u_prime  = self.derham.create_spline_function("u_prime",  space_id="Hdiv")
        self._ue_prime = self.derham.create_spline_function("ue_prime", space_id="Hdiv")

        for u_prime, boundary_data, boundary_conditions in [
            (self._u_prime,  self.options.boundary_data_u,  self.options.boundary_conditions_u),
            (self._ue_prime, self.options.boundary_data_ue, self.options.boundary_conditions_ue),
        ]:
            if boundary_data is None:
                continue
            for (d, side), f_bc in boundary_data.items():
                if boundary_conditions.get((d, side)) == "dirichlet":
                    bc_pulled = lambda *etas, f=f_bc: self.domain.pull(
                        [lambda x,y,z, f=f: f(x,y,z)[0],
                         lambda x,y,z, f=f: f(x,y,z)[1],
                         lambda x,y,z, f=f: f(x,y,z)[2]],
                        *etas, kind="2")
                    _vec = self._projector([lambda *etas: bc_pulled(*etas)[0],
                                           lambda *etas: bc_pulled(*etas)[1],
                                           lambda *etas: bc_pulled(*etas)[2]])
                    for (d2, side2), kind2 in boundary_conditions.items():
                        if kind2 == "dirichlet" and (d2, side2) != (d, side):
                            apply_essential_bc_stencil(_vec[0], axis=d2, ext=side2, order=0)
                    u_prime.vector += _vec

        self._u_prime_v0  = self._derham_v0.create_spline_function("u_prime_v0",  space_id="Hdiv")
        self._ue_prime_v0 = self._derham_v0.create_spline_function("ue_prime_v0", space_id="Hdiv")

        self._u_prime_v0.vector = self._u_prime.vector
        self._ue_prime_v0.vector = self._ue_prime.vector

        # ---- projected source terms (unconstrained) --------------------------

        self._rhs_u  = self.derham.create_spline_function("rhs_u",  space_id="Hdiv")
        self._rhs_ue = self.derham.create_spline_function("rhs_ue", space_id="Hdiv")

        for rhs, source in [(self._rhs_u, self.options.source_u), (self._rhs_ue, self.options.source_ue)]:
            if source is not None:
                src_pulled = lambda *etas, f=source: self.domain.pull(
                    [lambda x,y,z, f=f: f(x,y,z)[0],
                     lambda x,y,z, f=f: f(x,y,z)[1],
                     lambda x,y,z, f=f: f(x,y,z)[2]],
                    *etas, kind="2")
                rhs.vector = self._projector.get_dofs([lambda *etas: src_pulled(*etas)[0],
                                                       lambda *etas: src_pulled(*etas)[1],
                                                       lambda *etas: src_pulled(*etas)[2]])

        # ---- pre-allocated RHS vectors (v0, reused each time step) -----------

        self._rhs_vec_u  = self._derham_v0.create_spline_function("rhs_vec_u",  space_id="Hdiv")
        self._rhs_vec_ue = self._derham_v0.create_spline_function("rhs_vec_ue", space_id="Hdiv")

    # =========================================================================
    ### Time step
    # =========================================================================

    def __call__(self, dt):

        # --- copy current state ---
        self._u.vector  = self.variables.u.spline.vector
        self._ue.vector = self.variables.ue.spline.vector

        # --- rebuild system matrix if dt changed ---
        if dt != self._dt:
            self._dt = dt
            _A = BlockLinearOperator(
                self._block_domain_v0, self._block_codomain_v0,
                blocks=[[self._A11_v0 + self._M2_v0 / dt, None], [None, self._A22_v0]]
            )
            _M = BlockLinearOperator(
                self._block_domain_M, self._block_domain_M,
                blocks=[[_A, self._B_v0.T], [self._B_v0, None]]
            )
            self._Minv.linop = _M

        # --- assemble RHS in unconstrained space, then zero boundary DOFs ---
        # ion:      F1 = rhs_u + M2/dt * u - (A11 + M2/dt) * u'
        # electron: F2 = rhs_ue - A22 * ue'
        self._rhs_vec_u.vector  = (self._rhs_u.vector
                                   + self._M2.dot(self._u.vector) / dt
                                   - self._A11.dot(self._u_prime.vector)
                                   - self._M2.dot(self._u_prime.vector) / dt)
        self._rhs_vec_ue.vector = (self._rhs_ue.vector
                                   - self._A22.dot(self._ue_prime.vector))

        self._apply_boundary_conditions(self._rhs_vec_u.vector,  self.options.boundary_conditions_u)
        self._apply_boundary_conditions(self._rhs_vec_ue.vector, self.options.boundary_conditions_ue)

        # --- build block RHS and solve ---
        _F   = BlockVector(self._block_domain_v0,
                           blocks=[self._rhs_vec_u.vector, self._rhs_vec_ue.vector])
        _RHS = BlockVector(self._block_domain_M,
                           blocks=[_F, self._block_codomain_B_v0.zeros()])

        _sol = self._Minv.dot(_RHS)
        info = self._Minv.get_info()

        # --- reconstruct full solution: u = u_0 + u' ---
        self._u.vector   = _sol[0][0] + self._u_prime_v0.vector
        self._ue.vector  = _sol[0][1] + self._ue_prime_v0.vector
        self._phi.vector = _sol[1]

        # --- update FEEC variables ---
        max_diffs = self.update_feec_variables(
            u=self._u.vector, ue=self._ue.vector, phi=self._phi.vector
        )

        if self.options.solver_params.info and self._rank == 0:
            print(f"Status: {info['success']}, Iterations: {info['niter']}")
            print(f"Max diffs: {max_diffs}")