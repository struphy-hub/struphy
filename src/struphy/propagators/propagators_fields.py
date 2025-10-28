"Only FEEC variables are updated."

import copy
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal, get_args

import cunumpy as xp
import scipy as sc
from line_profiler import profile
from matplotlib import pyplot as plt
from numpy import zeros
from psydac.api.essential_bc import apply_essential_bc_stencil
from psydac.ddm.mpi import mpi as MPI
from psydac.linalg.basic import ComposedLinearOperator, IdentityOperator, ZeroOperator
from psydac.linalg.block import BlockLinearOperator, BlockVector, BlockVectorSpace
from psydac.linalg.solvers import inverse
from psydac.linalg.stencil import StencilVector

import struphy.feec.utilities as util
from struphy.examples.restelli2018 import callables
from struphy.feec import preconditioner
from struphy.feec.basis_projection_ops import (
    BasisProjectionOperator,
    BasisProjectionOperatorLocal,
    BasisProjectionOperators,
    CoordinateProjector,
)
from struphy.feec.linear_operators import BoundaryOperator
from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
from struphy.feec.preconditioner import MassMatrixDiagonalPreconditioner, MassMatrixPreconditioner
from struphy.feec.projectors import L2Projector
from struphy.feec.psydac_derham import Derham, SplineFunction
from struphy.feec.variational_utilities import (
    BracketOperator,
    Hdiv0_transport_operator,
    InternalEnergyEvaluator,
    KineticEnergyEvaluator,
    Pressure_transport_operator,
)
from struphy.fields_background.equils import set_defaults
from struphy.geometry.utilities import TransformedPformComponent
from struphy.initial import perturbations
from struphy.io.options import (
    OptsDirectSolver,
    OptsGenSolver,
    OptsMassPrecond,
    OptsNonlinearSolver,
    OptsSaddlePointSolver,
    OptsSymmSolver,
    OptsVecSpace,
    check_option,
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


class Maxwell(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf E \in H(\textnormal{curl})` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F \, \textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times \mathbf F \,\textrm d \mathbf x = 0\,, \qquad \forall \, \mathbf F \in H(\textnormal{curl}) \,.
        \\[2mm]
        &\frac{\partial \mathbf B}{\partial t} + \nabla\times\mathbf E = 0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.
    """

    class Variables:
        def __init__(self):
            self._e: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        algo: OptsAlgo = "implicit"
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        # obtain needed matrices
        M1 = self.mass_ops.M1
        M2 = self.mass_ops.M2
        curl = self.derham.curl

        # Preconditioner for M1 + ...
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(M1)

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info
            # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
            _A = M1

            # no dt
            self._B = -1 / 2 * curl.T @ M2
            self._C = 1 / 2 * curl

            # Instantiate Schur solver (constant in this case)
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
            M1_inv = inverse(
                M1,
                self.options.solver,
                pc=pc,
                tol=self.options.solver_params.tol,
                maxiter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )
            weak_curl = M1_inv @ curl.T @ M2

            # allocate output of vector field
            out1 = self.variables.e.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                weak_curl.dot(y2, out=out)
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                curl.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.e.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

        # allocate place-holder vectors to avoid temporary array allocations in __call__
        self._e_tmp1 = self.variables.e.spline.vector.space.zeros()
        self._e_tmp2 = self.variables.e.spline.vector.space.zeros()
        self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        # current FE coeffs
        en = self.variables.e.spline.vector
        bn = self.variables.b.spline.vector

        if self.options.algo == "implicit":
            # solve for new e coeffs
            self._B.dot(bn, out=self._byn)

            en1, info = self._schur_solver(en, self._byn, dt, out=self._e_tmp1)

            # new b coeffs
            _e = en.copy(out=self._e_tmp2)
            _e += en1
            bn1 = self._C.dot(_e, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(e=en1, b=bn1)
        else:
            self._ode_solver(0.0, dt)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                print("Status     for Maxwell:", info["success"])
                print("Iterations for Maxwell:", info["niter"])
                print("Maxdiff e for Maxwell:", diffs["e"])
                print("Maxdiff b for Maxwell:", diffs["b"])
                print()


class OhmCold(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf j \in H(\textnormal{curl})` and :math:`\mathbf E \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{1}{n_0} \frac{\partial \mathbf j}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x &= \frac{1}{\varepsilon} \int_\Omega \mathbf E \cdot \mathbf F \,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,
        \\[2mm]
        -\frac{\partial \mathbf E}{\partial t} &= \frac{\alpha^2}{\varepsilon} \mathbf j \,,

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`, such that

    .. math::

        \begin{bmatrix}
            \mathbf j^{n+1} - \mathbf j^n \\
            \mathbf e^{n+1} - \mathbf e^n
        \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix}
            0 & \frac{1}{\varepsilon} \mathbb M_{1/n_0}^{-1} \\
            - \frac{1}{\varepsilon} \mathbb M_{1/n_0}^{-1} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \alpha^2 \mathbb M_{1/n_0} (\mathbf j^{n+1} + \mathbf j^{n}) \\
            \mathbb M_1 (\mathbf e^{n+1} + \mathbf e^{n})
        \end{bmatrix} \,.
    """

    class Variables:
        def __init__(self):
            self._j: FEECVariable = None
            self._e: FEECVariable = None

        @property
        def j(self) -> FEECVariable:
            return self._j

        @j.setter
        def j(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._j = new

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        self._alpha = self.variables.j.species.equation_params.alpha
        self._epsilon = self.variables.j.species.equation_params.epsilon

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1ninv

        self._B = -1 / 2 * 1 / self._epsilon * self.mass_ops.M1  # no dt

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate Schur solver (constant in this case)
        _BC = 1 / 2 * self._alpha**2 / self._epsilon * self._B

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        j = self.variables.j.spline.vector
        e = self.variables.e.spline.vector

        self._tmp_j1 = j.space.zeros()
        self._tmp_j2 = j.space.zeros()
        self._tmp_e1 = e.space.zeros()
        self._tmp_e2 = e.space.zeros()

    def __call__(self, dt):
        # current variables
        jn = self.variables.j.spline.vector
        en = self.variables.e.spline.vector

        # in-place solution (no tmps created here)
        Ben = self._B.dot(en, out=self._tmp_e1)

        jn1, info = self._schur_solver(jn, Ben, dt, out=self._tmp_j1)

        en1 = jn.copy(out=self._tmp_j2)
        en1 += jn1
        en1 *= 1 / 2 * self._alpha**2 / self._epsilon
        en1 *= -dt
        en1 += en

        # write new coeffs into Propagator.variables
        diffs = self.update_feec_variables(e=en1, j=jn1)

        if self._info:
            print("Status     for OhmCold:", info["success"])
            print("Iterations for OhmCold:", info["niter"])
            print("Maxdiff e1 for OhmCold:", diffs["e"])
            print("Maxdiff j1 for OhmCold:", diffs["j"])
            print()


class JxBCold(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf j \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{1}{n_0} \frac{\partial \mathbf j}{\partial t} \cdot \mathbf F \,\textrm d \mathbf x
        = \frac{1}{\varepsilon} \int_\Omega \frac{1}{n_0} (\mathbf j \times \mathbf B_0) \cdot \mathbf F \,\textrm d \mathbf x \qquad \forall \,\mathbf F \in H(\textnormal{curl})\,,

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point), such that

    .. math::

        \mathbb M_{1/n_0} \left( \mathbf j^{n+1} - \mathbf j^n \right) = \frac{\Delta t}{2} \frac{1}{\varepsilon} \mathbb M_{B_0/n_0} \left( \mathbf j^{n+1} - \mathbf j^n \right)\,.
    """

    class Variables:
        def __init__(self):
            self._j: FEECVariable = None

        @property
        def j(self) -> FEECVariable:
            return self._j

        @j.setter
        def j(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._j = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        epsilon = self.variables.j.species.equation_params.epsilon

        # mass matrix in system (M - dt/2 * A)*j^(n + 1) = (M + dt/2 * A)*j^n
        self._M = self.mass_ops.M1ninv
        self._A = -1 / epsilon * self.mass_ops.M1Bninv  # no dt

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M1ninv)

        # Instantiate linear solver
        self._solver = inverse(
            self._M,
            self.options.solver,
            pc=pc,
            x0=self.variables.j.spline.vector,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_j = self._M.codomain.zeros()
        self._j_new = self.variables.j.spline.vector.space.zeros()

    @profile
    def __call__(self, dt):
        # current variables
        jn = self.variables.j.spline.vector

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = self._M - dt / 2.0 * self._A
        rhs = self._M + dt / 2.0 * self._A

        rhsv = rhs.dot(jn, out=self._rhs_j)

        self._solver.linop = lhs

        # solve linear system for updated j coefficients (in-place)
        jn1 = self._solver.solve(rhsv, out=self._j_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_dj = self.update_feec_variables(j=jn1)

        if self._info:
            print("Status     for FluidCold:", info["success"])
            print("Iterations for FluidCold:", info["niter"])
            print("Maxdiff j1 for FluidCold:", max_dj)
            print()


class ShearAlfven(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \tilde{\mathbf B } \cdot \nabla \times (\mathbf{B}_0 \times \mathbf V) \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

        &\frac{\partial \tilde{\mathbf{B}}}{\partial t} - \nabla\times(\tilde{\mathbf{U}} \times \mathbf{B}_0)
        = 0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} \mathcal {T^\alpha}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^\alpha} (\mathbb M^\rho_\alpha)^{-1} & 0 \end{bmatrix}
        \begin{bmatrix} {\mathbb M^\rho_\alpha}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    where :math:`\alpha \in \{1, 2, v\}` and :math:`\mathbb M^\rho_\alpha` is a weighted mass matrix in :math:`\alpha`-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density. The solution of the above system is based on the :ref:`Schur complement <schur_solver>`.
    """

    class Variables:
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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        u_space: OptsVecSpace = "Hdiv"
        algo: OptsAlgo = "implicit"
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        butcher: ButcherTableau = None

        def __post_init__(self):
            # checks
            check_option(self.u_space, OptsVecSpace)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        u_space = self.options.u_space

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        id_M = "M" + self.derham.space_to_form[u_space] + "n"
        id_T = "T" + self.derham.space_to_form[u_space]

        # call operators
        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)
        _M2 = self.mass_ops.M2
        curl = self.derham.curl

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
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

    @profile
    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        if self.options.algo == "implicit":
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
                print("Status     for ShearAlfven:", info["success"])
                print("Iterations for ShearAlfven:", info["niter"])
                print("Maxdiff up for ShearAlfven:", diffs["u"])
                print("Maxdiff b2 for ShearAlfven:", diffs["b"])
                print()


class ShearAlfvenB1(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{curl})` such that

    .. math::

        &\int_\Omega \rho_0\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textnormal d \mathbf x
        =\int_\Omega \tilde{\mathbf B } \cdot \nabla \times (\mathbf{B}_0 \times \mathbf V) \,\textnormal d \mathbf x \qquad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x - \int_\Omega \mathbf C \cdot\nabla\times   \left( \tilde{\mathbf{U}} \times \mathbf{B}_0 \right) \textrm d \mathbf x = 0 \qquad \forall \, \mathbf C \in H(\textrm{curl})\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_2)^{-1} \mathcal {T^2}^\top \mathbb C \mathbb M_1^{-1}\\ - \mathbb M_1^{-1} \mathbb C^\top \mathcal {T^2} (\mathbb M^\rho_2)^{-1} & 0 \end{bmatrix}
        \begin{bmatrix} {\mathbb M^\rho_2}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_1(\mathbf b^{n+1} + \mathbf b^n) \end{bmatrix} ,

    where :math:`\mathbb M^\rho_2` is a weighted mass matrix in 2-space, the weight being :math:`\rho_0`,
    the MHD equilibirum density.
    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hdiv")
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

    @dataclass
    class Options:
        # propagator options
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        solver_M1: OptsSymmSolver = "pcg"
        precond_M1: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params_M1: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)
            check_option(self.solver_M1, OptsSymmSolver)
            check_option(self.precond_M1, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.solver_params_M1 is None:
                self.solver_params_M1 = SolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info

        # define inverse of M1
        if self.options.precond_M1 is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond_M1)
            pc = pc_class(self.mass_ops.M1)

        M1_inv = inverse(
            self.mass_ops.M1,
            self.options.solver_M1,
            pc=pc,
            tol=self.options.solver_params_M1.tol,
            maxiter=self.options.solver_params_M1.maxiter,
            verbose=self.options.solver_params_M1.verbose,
        )

        # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M2n
        self._B = 1 / 2 * self.mass_ops.M2B @ self.derham.curl
        # I still have to invert M1
        self._C = 1 / 2 * M1_inv @ self.derham.curl.T @ self.mass_ops.M2B

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, "M2n"))

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
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector

        self._u_tmp1 = u.space.zeros()
        self._u_tmp2 = u.space.zeros()
        self._b_tmp1 = b.space.zeros()

        self._byn = self._B.codomain.zeros()

    @profile
    def __call__(self, dt):
        # current variables
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # solve for new u coeffs
        byn = self._B.dot(bn, out=self._byn)

        un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

        # new b coeffs (no tmps created here)
        _u = un.copy(out=self._u_tmp2)
        _u += un1
        bn1 = self._C.dot(_u, out=self._b_tmp1)
        bn1 *= -dt
        bn1 += bn

        # write new coeffs into self.feec_vars
        max_diffs = self.update_feec_variables(u=un1, b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            print("Status     for ShearAlfvenB1:", info["success"])
            print("Iterations for ShearAlfvenB1:", info["niter"])
            print("Maxdiff up for ShearAlfvenB1:", max_diffs["u"])
            print("Maxdiff b2 for ShearAlfvenB1:", max_diffs["b"])
            print()


class Hall(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf B \in H(\textnormal{curl})` such that

    .. math::

        \int_\Omega \frac{\partial \tilde{\mathbf{B}}}{\partial t} \cdot \mathbf C\,\textnormal d \mathbf x + \frac{1}{\varepsilon} \int_\Omega \nabla\times \mathbf C \cdot  \left( \frac{\nabla\times \tilde{\mathbf{B}}}{\rho_0}\times \mathbf{B}_0 \right) \textrm d \mathbf x = 0 \qquad \forall \, \mathbf C \in H(\textrm{curl})

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point):

    .. math::

        \mathbf b^{n+1} - \mathbf b^n
        = \frac{\Delta t}{2} \mathbb M_1^{-1} \mathbb C^\top  \mathbb M^{\mathcal{T},\rho}_2  \mathbb C  (\mathbf b^{n+1} + \mathbf b^n)  ,

    where :math:`\mathbb M^{\mathcal{T},\rho}_2` is a weighted mass matrix in 2-space, the weight being :math:`\frac{\mathcal{T}}{\rho_0}`,
    the MHD equilibirum density :math:`\rho_0` as a 0-form, and rotation matrix :math:`\mathcal{T} \vec v = \vec B^2_{\textnormal{eq}} \times \vec v\,,`.
    The solution of the above system is based on the Pre-conditioned Biconjugate Gradient Stabilized algortihm (PBiConjugateGradientStab).
    """

    class Variables:
        def __init__(self):
            self._b: FEECVariable = None

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

    @dataclass
    class Options:
        # propagator options
        solver: OptsGenSolver = "pbicgstab"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        epsilon_from: Species = None

        def __post_init__(self):
            # checks
            check_option(self.solver, OptsGenSolver)
            check_option(self.precond, OptsMassPrecond)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        if self.options.epsilon_from is None:
            epsilon = 1.0
        else:
            epsilon = self.options.epsilon_from.equation_params.epsilon

        self._info = self.options.solver_params.info
        self._tol = self.options.solver_params.tol
        self._maxiter = self.options.solver_params.maxiter
        self._verbose = self.options.solver_params.verbose

        # mass matrix in system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        id_M = "M1"
        id_M2Bn = "M2Bn"
        self._M = getattr(self.mass_ops, id_M)
        self._M2Bn = getattr(self.mass_ops, id_M2Bn)
        self._A = 1.0 / epsilon * self.derham.curl.T @ self._M2Bn @ self.derham.curl

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        # Instantiate linear solver
        self._solver = inverse(
            self._M,
            self.options.solver,
            pc=pc,
            x0=self.variables.b.spline.vector,
            tol=self._tol,
            maxiter=self._maxiter,
            verbose=self._verbose,
        )

        # allocate dummy vectors to avoid temporary array allocations
        self._rhs_b = self._M.codomain.zeros()
        self._b_new = self.variables.b.spline.vector.space.zeros()

    def __call__(self, dt):
        # current variables
        bn = self.variables.b.spline.vector

        # define system (M - dt/2 * A)*b^(n + 1) = (M + dt/2 * A)*b^n
        lhs = self._M - dt / 2.0 * self._A
        rhs = self._M + dt / 2.0 * self._A

        # solve linear system for updated b coefficients (in-place)
        rhs = rhs.dot(bn, out=self._rhs_b)
        self._solver.linop = lhs

        bn1 = self._solver.solve(rhs, out=self._b_new)
        info = self._solver._info

        # write new coeffs into self.feec_vars
        max_db = self.update_feec_variables(b=bn1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            print("Status     for Hall:", info["success"])
            print("Iterations for Hall:", info["niter"])
            print("Maxdiff b1 for Hall:", max_db["b"])
            print()


class Magnetosonic(Propagator):
    r"""
    :ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde \rho \in L^2, \tilde{\mathbf U} \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}, \tilde p \in L^2` such that

    .. math::
        &\frac{\partial \tilde \rho}{\partial t}+\nabla\cdot(\rho_0 \tilde{\mathbf{U}})=0\,,

        \int \rho_0&\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V\,\textrm d \mathbf x  - \int \tilde p\, \nabla \cdot \mathbf V \,\textrm d \mathbf x
        =\int (\nabla\times\mathbf{B}_0)\times \tilde{\mathbf{B}} \cdot \mathbf V\,\textrm d \mathbf x
        \qquad \forall \ \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

        &\frac{\partial \tilde p}{\partial t} + \nabla\cdot(p_0 \tilde{\mathbf{U}})
        + \frac{2}{3}\,p_0\nabla\cdot \tilde{\mathbf{U}}=0\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} \mathbf u^{n+1} - \mathbf u^n \\ \mathbf p^{n+1} - \mathbf p^n \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^\rho_\alpha)^{-1} {\mathcal U^\alpha}^\top \mathbb D^\top \mathbb M_3 \\ - \mathbb D \mathcal S^\alpha - (\gamma - 1) \mathcal K^\alpha \mathbb D \mathcal U^\alpha & 0 \end{bmatrix}
        \begin{bmatrix} (\mathbf u^{n+1} + \mathbf u^n) \\ (\mathbf p^{n+1} + \mathbf p^n) \end{bmatrix} + \begin{bmatrix} \Delta t (\mathbb M^\rho_\alpha)^{-1} \mathbb M^J_\alpha \mathbf b^n \\ 0 \end{bmatrix},

    where :math:`\alpha \in \{1, 2, v\}` and :math:`\mathcal U^2 = \mathbb Id`; moreover, :math:`\mathbb M^\rho_\alpha` and
    :math:`\mathbb M^J_\alpha` are weighted mass matrices in :math:`\alpha`-space,
    the weights being the MHD equilibirum density :math:`\rho_0`
    and the curl of the MHD equilibrium current density :math:`\mathbf J_0 = \nabla \times \mathbf B_0`.
    Density update is decoupled:

    .. math::

        \boldsymbol{\rho}^{n+1} = \boldsymbol{\rho}^n - \frac{\Delta t}{2} \mathbb D \mathcal Q^\alpha (\mathbf u^{n+1} + \mathbf u^n) \,.
    """

    class Variables:
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

    @dataclass
    class Options:
        b_field: FEECVariable = None
        u_space: OptsVecSpace = "Hdiv"
        solver: OptsGenSolver = "pbicgstab"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.u_space, OptsVecSpace)
            check_option(self.solver, OptsGenSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.b_field is None:
                self.b_field = FEECVariable(space="Hdiv")
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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

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
        self._C = 1 / 2.0 * self.derham.div @ _S + 2 / 3 * _K @ self.derham.div @ _U

        self._MJ = getattr(self.mass_ops, id_MJ)
        self._DQ = self.derham.div @ getattr(self.basis_ops, id_Q)

        self.options.b_field.allocate(self.derham, self.domain)
        self._b = self.options.b_field.spline.vector

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
            print("Status     for Magnetosonic:", info["success"])
            print("Iterations for Magnetosonic:", info["niter"])
            print("Maxdiff n3 for Magnetosonic:", diffs["n"])
            print("Maxdiff up for Magnetosonic:", diffs["u"])
            print("Maxdiff p3 for Magnetosonic:", diffs["p"])
            print()


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
    n : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

    u : psydac.linalg.block.BlockVector
        FE coefficients of MHD velocity 2-form.

    p : psydac.linalg.stencil.StencilVector
        FE coefficients of a discrete 3-form.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    """

    class Variables:
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

    @dataclass
    class Options:
        solver: OptsGenSolver = "pbicgstab"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, OptsGenSolver)
            check_option(self.precond, OptsMassPrecond)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
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
            print("Status     for Magnetosonic:", info["success"])
            print("Iterations for Magnetosonic:", info["niter"])
            print("Maxdiff n3 for Magnetosonic:", diffs["n"])
            print("Maxdiff up for Magnetosonic:", diffs["u"])
            print("Maxdiff p3 for Magnetosonic:", diffs["p"])
            print()


class FaradayExtended(Propagator):
    r"""Equations: Faraday's law

    .. math::
        \begin{align*}
        & \frac{\partial {\mathbf A}}{\partial t} = - \frac{\nabla \times (\nabla \times {\mathbf A} + {\mathbf B}_0) }{n} \times (\nabla \times {\mathbf A} + {\mathbf B}_0) - \frac{\int ({\mathbf A} - {\mathbf p}f \mathrm{d}{\mathbf p})}{n} \times (\nabla \times {\mathbf A} + {\mathbf B}_0), \\
        & n = \int f \mathrm{d}{\mathbf p}.
        \end{align*}

    Mid-point rule:

    .. math::
        \begin{align*}
        & \left[ \mathbb{M}_1 - \frac{\Delta t}{2} \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}}) \mathbb{M}_1^{-1} (\mathbb{P}_1^\top \mathbb{W} \mathbb{P}_1 + \mathbb{C}^\top \mathbb{M}_2 \mathbb{C} ) \right] {\mathbf a}^{n+1} \\
        & = \mathbb{M}_1 {\mathbf a}^n + \frac{\Delta t}{2} \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}}) \mathbb{M}_1^{-1} (\mathbb{P}_1^\top \mathbb{W} \mathbb{P}_1 + \mathbb{C}^\top \mathbb{M}_2 \mathbb{C} ) {\mathbf a}^{n+1} \\
        & - \Delta t \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}})  \mathbb{M}_1^{-1} \mathbb{P}_1^\top \mathbb{W} {\mathbf P}^n\\
        & + \Delta t \mathbb{F}(\hat{n}^0_h, {\mathbf a}^{n+\frac{1}{2}}) \mathbb{M}_1^{-1} \mathbb{C}^\top \mathbb{M}_2 {\mathbf b}_0\\
        & \mathbb{F}_{ij} = - \int \frac{1}{\hat{n}^0_h \sqrt{g}} G (\nabla \times {\mathbf A} + {\mathbf B}_0) \cdot (\Lambda^1_i \times \Lambda^1_j) \mathrm{d}{\boldsymbol \eta}.
        \end{align*}

    Parameters
    ---------- 
        a : psydac.linalg.block.BlockVector
            FE coefficients of vector potential.

        **params : dict
            Solver- and/or other parameters for this splitting step.
    """

    def __init__(self, a, **params):
        assert isinstance(a, (BlockVector, PolarVector))

        # parameters
        params_default = {
            "a_space": None,
            "beq": None,
            "particles": None,
            "quad_number": None,
            "shape_degree": None,
            "shape_size": None,
            "solver_params": None,
            "accumulate_density": None,
        }

        params = set_defaults(params, params_default)

        self._a = a
        self._a_old = self._a.copy()

        self._a_space = params["a_space"]
        assert self._a_space in {"Hcurl"}

        self._beq = params["beq"]

        self._particles = params["particles"]

        self._nqs = params["quad_number"]

        self.size1 = int(self.derham.domain_array[self.rank, int(2)])
        self.size2 = int(self.derham.domain_array[self.rank, int(5)])
        self.size3 = int(self.derham.domain_array[self.rank, int(8)])

        self.weight_1 = zeros(
            (self.size1 * self._nqs[0], self.size2 * self._nqs[1], self.size3 * self._nqs[2]),
            dtype=float,
        )
        self.weight_2 = zeros(
            (self.size1 * self._nqs[0], self.size2 * self._nqs[1], self.size3 * self._nqs[2]),
            dtype=float,
        )
        self.weight_3 = zeros(
            (self.size1 * self._nqs[0], self.size2 * self._nqs[1], self.size3 * self._nqs[2]),
            dtype=float,
        )

        self._weight_pre = [self.weight_1, self.weight_2, self.weight_3]

        self._ind = [
            [self.derham.indN[0], self.derham.indD[1], self.derham.indD[2]],
            [self.derham.indD[0], self.derham.indN[1], self.derham.indD[2]],
            [self.derham.indD[0], self.derham.indD[1], self.derham.indN[2]],
        ]

        # Initialize Accumulator object for getting density from particles
        self._pts_x = 1.0 / (2.0 * self.derham.Nel[0]) * xp.polynomial.legendre.leggauss(
            self._nqs[0],
        )[0] + 1.0 / (2.0 * self.derham.Nel[0])
        self._pts_y = 1.0 / (2.0 * self.derham.Nel[1]) * xp.polynomial.legendre.leggauss(
            self._nqs[1],
        )[0] + 1.0 / (2.0 * self.derham.Nel[1])
        self._pts_z = 1.0 / (2.0 * self.derham.Nel[2]) * xp.polynomial.legendre.leggauss(
            self._nqs[2],
        )[0] + 1.0 / (2.0 * self.derham.Nel[2])

        self._p_shape = params["shape_degree"]
        self._p_size = params["shape_size"]
        self._accum_density = params["accumulate_density"]

        # Initialize Accumulator object for getting the matrix and vector related with vector potential
        self._accum_potential = Accumulator(
            self.mass_ops,
            self.domain,
            self._a_space,
            "hybrid_fA_Arelated",
            add_vector=True,
            symmetry="symm",
        )

        self._solver_params = params["solver_params"]
        # preconditioner
        if self._solver_params["pc"] is None:
            self._pc = None
        else:
            pc_class = getattr(preconditioner, self._solver_params["pc"])
            self._pc = pc_class(self.mass_ops.M1)

        self._Minv = inverse(self.mass_ops.M1, tol=1e-8)
        self._CMC = self.derham.curl.T @ self.mass_ops.M2 @ self.derham.curl
        self._M1 = self.mass_ops.M1
        self._M2 = self.mass_ops.M2

    @property
    def variables(self):
        return [self._a]

    def __call__(self, dt):
        # the loop of fixed point iteration, 100 iterations at most.

        self._accum_density.accumulate(
            self._particles,
            xp.array(self.derham.Nel),
            xp.array(self._nqs),
            xp.array(
                self._pts_x,
            ),
            xp.array(self._pts_y),
            xp.array(self._pts_z),
            xp.array(self._p_shape),
            xp.array(self._p_size),
        )
        self._accum_potential.accumulate(self._particles)

        self._L2 = -dt / 2 * self._Minv @ (self._accum_potential._operators[0].matrix + self._CMC)
        self._RHS = -(self._L2.dot(self._a)) - dt * (
            self._Minv.dot(
                self._accum_potential._vectors[0] - self.derham.curl.T @ self._M2,
            ).dot(self._beq)
        )
        self._rhs = self._M1.dot(self._a)

        for _ in range(10):
            # print('+++++=====++++++', self._accum_density._operators[0].matrix._data)
            # set mid-value used in the fixed iteration
            curla_mid = (
                self.derham.curl.dot(
                    0.5 * (self._a_old + self._a),
                )
                + self._beq
            )
            curla_mid.update_ghost_regions()
            # initialize the curl A
            # remember to check ghost region of curla_mid
            util.create_weight_weightedmatrix_hybrid(
                curla_mid,
                self._weight_pre,
                self.derham,
                self._accum_density,
                self.domain,
            )
            # self._weight = [[None, self._weight_pre[2], -self._weight_pre[1]], [None, None, self._weight_pre[0]], [None, None, None]]
            self._weight = [
                [0.0 * self._weight_pre[k] for k in range(3)],
                [0.0 * self._weight_pre[k] for k in range(3)],
                [0.0 * self._weight_pre[k] for k in range(3)],
            ]
            # self._weight = [[self._weight_pre[0], self._weight_pre[2], self._weight_pre[1]], [self._weight_pre[2], self._weight_pre[1], self._weight_pre[0]], [self._weight_pre[1], self._weight_pre[0], self._weight_pre[2]]]
            HybridM1 = self.mass_ops.create_weighted_mass("Hcurl", "Hcurl", weights=self._weight, assemble=True)

            # next prepare for solving linear system
            _LHS = self._M1 + HybridM1 @ self._L2
            _RHS2 = HybridM1.dot(self._RHS) + self._rhs

            # TODO: unknown function 'pcg', use new solver API
            a_new, info = pcg(
                _LHS,
                _RHS2,
                self._pc,
                x0=self._a,
                tol=self._solver_params["tol"],
                maxiter=self._solver_params["maxiter"],
                verbose=self._solver_params["verbose"],
            )

            # write new coeffs into Propagator.variables
            max_da = self.feec_vars_update(a_new)
            print("++++====check_iteration_error=====+++++", max_da)
            # we can modify the diff function in in_place_update to get another type errors
            if max_da[0] < 10 ** (-6):
                break

    @classmethod
    def options(cls):
        dct = {}
        dct["solver"] = {
            "type": [
                ("pcg", "MassMatrixPreconditioner"),
                ("cg", None),
            ],
            "tol": 1.0e-8,
            "maxiter": 3000,
            "info": False,
            "verbose": False,
            "recycle": True,
        }
        return dct


class CurrentCoupling6DDensity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde{\mathbf{U}}  \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` such that

    .. math::

        &\int_\Omega \rho_0 \frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon} \int_\Omega n_\textnormal{h}\tilde{\mathbf{U}} \times(\mathbf{B}_0+\tilde{\mathbf{B}}) \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &n_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\,\textnormal{d}^3 \mathbf v\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point).
    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
        energetic_ions: PICVariable = None
        b_tilde: FEECVariable = None
        u_space: OptsVecSpace = "Hdiv"
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        boundary_cut: tuple = (0.0, 0.0, 0.0)

        def __post_init__(self):
            # checks
            check_option(self.u_space, OptsVecSpace)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)
            assert self.energetic_ions.space == "Particles6D"
            assert self.b_tilde.space == "Hdiv"

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._space_key_int = int(self.derham.space_to_form[self.options.u_space])

        particles = self.options.energetic_ions.particles
        u = self.variables.u.spline.vector
        self._b_eq = self.projected_equil.b2
        self._b_tilde = self.options.b_tilde.spline.vector

        # if self._particles.control_variate:

        #     # control variate method is only valid with Maxwellian distributions
        #     assert isinstance(self._particles.f0, Maxwellian)
        #     assert u_space == 'Hdiv'

        #     # evaluate and save nh0/|det(DF)| (push-forward) at quadrature points for control variate
        #     quad_pts = [quad_grid[nquad].points.flatten()
        #                 for quad_grid, nquad in zip(self.derham.Vh_fem['0']._quad_grids, self.derham.Vh_fem['0'].nquads)]

        #     self._nh0_at_quad = self.domain.push(
        #         self._particles.f0.n, *quad_pts, kind='3', squeeze_out=False)

        #     # memory allocation of magnetic field at quadrature points
        #     self._b_quad1 = xp.zeros_like(self._nh0_at_quad)
        #     self._b_quad2 = xp.zeros_like(self._nh0_at_quad)
        #     self._b_quad3 = xp.zeros_like(self._nh0_at_quad)

        #     # memory allocation for self._b_quad x self._nh0_at_quad * self._coupling_const
        #     self._mat12 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat13 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat23 = xp.zeros_like(self._nh0_at_quad)

        #     self._mat21 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat31 = xp.zeros_like(self._nh0_at_quad)
        #     self._mat32 = xp.zeros_like(self._nh0_at_quad)

        self._type = self.options.solver
        self._tol = self.options.solver_params.tol
        self._maxiter = self.options.solver_params.maxiter
        self._info = self.options.solver_params.info
        self._verbose = self.options.solver_params.verbose
        self._recycle = self.options.solver_params.recycle

        Ah = self.options.energetic_ions.species.mass_number
        Ab = self.variables.u.species.mass_number
        epsilon = self.options.energetic_ions.species.equation_params.epsilon

        self._coupling_const = Ah / Ab / epsilon

        self._boundary_cut_e1 = self.options.boundary_cut[0]

        # load accumulator
        self._accumulator = Accumulator(
            particles,
            self.options.u_space,
            Pyccelkernel(accum_kernels.cc_lin_mhd_6d_1),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=False,
            symmetry="asym",
            filter_params=self.options.filter_params,
        )

        # transposed extraction operator PolarVector --> BlockVector (identity map in case of no polar splines)
        self._E2T = self.derham.extraction_ops["2"].transpose()

        # mass matrix in system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        u_id = self.derham.space_to_form[self.options.u_space]
        self._M = getattr(self.mass_ops, "M" + u_id + "n")

        # preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self._M)

        # linear solver
        self._solver = inverse(
            self._M,
            self.options.solver,
            pc=pc,
            x0=self.variables.u.spline.vector,
            tol=self._tol,
            maxiter=self._maxiter,
            verbose=self._verbose,
            recycle=self._recycle,
        )

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()

        self._rhs_v = u.space.zeros()
        self._u_new = u.space.zeros()

    def __call__(self, dt):
        # pointer to old coefficients
        un = self.variables.u.spline.vector

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)

        if self._b_tilde is not None:
            self._b_full1 += self._b_tilde

        # extract coefficients to tensor product space (in-place)
        self._E2T.dot(self._b_full1, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        self._b_full2.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        # if self._particles.control_variate:

        #     # evaluate magnetic field at quadrature points (in-place)
        #     WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._b_full2,
        #                                    out=[self._b_quad1, self._b_quad2, self._b_quad3])

        #     self._mat12[:, :, :] = self._coupling_const * \
        #         self._b_quad3 * self._nh0_at_quad
        #     self._mat13[:, :, :] = -self._coupling_const * \
        #         self._b_quad2 * self._nh0_at_quad
        #     self._mat23[:, :, :] = self._coupling_const * \
        #         self._b_quad1 * self._nh0_at_quad

        #     self._mat21[:, :, :] = -self._mat12
        #     self._mat31[:, :, :] = -self._mat13
        #     self._mat32[:, :, :] = -self._mat23

        #     self._accumulator(self._b_full2[0]._data,
        #                       self._b_full2[1]._data,
        #                       self._b_full2[2]._data,
        #                       self._space_key_int,
        #                       self._coupling_const,
        #                       control_mat=[[None, self._mat12, self._mat13],
        #                                    [self._mat21, None,
        #                                     self._mat23],
        #                                    [self._mat31, self._mat32, None]])
        # else:
        self._accumulator(
            self._b_full2[0]._data,
            self._b_full2[1]._data,
            self._b_full2[2]._data,
            self._space_key_int,
            self._coupling_const,
            self._boundary_cut_e1,
        )

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = self._M - dt / 2 * self._accumulator.operators[0]
        rhs = self._M + dt / 2 * self._accumulator.operators[0]

        # solve linear system for updated u coefficients (in-place)
        rhs = rhs.dot(un, out=self._rhs_v)
        self._solver.linop = lhs

        un1 = self._solver.solve(rhs, out=self._u_new)
        info = self._solver._info

        # write new coeffs into Propagator.variables
        max_du = self.update_feec_variables(u=un1)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            print("Status     for CurrentCoupling6DDensity:", info["success"])
            print("Iterations for CurrentCoupling6DDensity:", info["niter"])
            print("Maxdiff up for CurrentCoupling6DDensity:", max_du)
            print()


class ShearAlfvenCurrentCoupling5D(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \left\{ 
            \begin{aligned} 
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \int \left(\tilde{\mathbf B} - \frac{A_\textnormal{h}}{A_b} \iint f^\text{vol} \mu \mathbf{b}_0\textnormal{d} \mu \textnormal{d} v_\parallel \right) \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf V) \, \textnormal{d} \mathbf{x} \quad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,, \,,
                \\
                &\frac{\partial \tilde{\mathbf B}}{\partial t} = - \nabla \times (\tilde{\mathbf B} \times \tilde{\mathbf U}) \,.
            \end{aligned}
        \right.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ \mathbf b^{n+1} - \mathbf b^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} \,.
        \begin{bmatrix} 
            0 & (\mathbb M^{2,n})^{-1} \mathcal {T^2}^\top \mathbb C^\top \\ - \mathbb C \mathcal {T^2} (\mathbb M^{2,n})^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            {\mathbb M^{2,n}}(\mathbf u^{n+1} + \mathbf u^n) \\ \mathbb M_2(\mathbf b^{n+1} + \mathbf b^n) + \sum_k^{N_p} \omega_k \mu_k \hat{\mathbf b}¹_0 (\boldsymbol \eta_k) \cdot \left(\frac{1}{\sqrt{g(\boldsymbol \eta_k)}} \vec \Lambda² (\boldsymbol \eta_k) \right)
        \end{bmatrix} \,,

    where 
    :math:`\mathcal{T}^2 = \hat \Pi \left[\frac{\tilde{\mathbf B}^2}{\sqrt{g} \times \vec \Lambda^2\right]`  and
    :math:`\mathbb M^{2,n}` is a :class:`~struphy.feec.mass.WeightedMassOperators` being weighted with :math:`\rho_\text{eq}`, the MHD equilibirum density. 
    """

    class Variables:
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

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsAlgo = Literal["implicit", "explicit"]
        # propagator options
        energetic_ions: PICVariable = None
        ep_scale: float = 1.0
        u_space: OptsVecSpace = "Hdiv"
        algo: OptsAlgo = "implicit"
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixDiagonalPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        butcher: ButcherTableau = None
        nonlinear: bool = True

        def __post_init__(self):
            # checks
            check_option(self.u_space, OptsVecSpace)
            check_option(self.algo, self.OptsAlgo)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)
            assert isinstance(self.energetic_ions, PICVariable)
            assert self.energetic_ions.space == "Particles5D"
            assert isinstance(self.ep_scale, float)
            assert isinstance(self.nonlinear, bool)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.filter_params is None:
                self.filter_params = FilterParameters()

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._u_form = self.derham.space_to_form[self.options.u_space]

        # call operatros
        id_M = "M" + self._u_form + "n"
        id_T = "T" + self._u_form

        _A = getattr(self.mass_ops, id_M)
        _T = getattr(self.basis_ops, id_T)
        M2 = self.mass_ops.M2
        curl = self.derham.curl
        PB = getattr(self.basis_ops, "PB")

        # define Accumulator and arguments
        self._ACC = AccumulatorVector(
            self.options.energetic_ions.particles,
            "H1",
            Pyccelkernel(accum_kernels_gc.gc_mag_density_0form),
            self.mass_ops,
            self.domain.args_domain,
            filter_params=self.options.filter_params,
        )

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        if self.options.nonlinear:
            # initialize operator TB
            self._initialize_projection_operator_TB()

            _T = _T + self._TB
            _TT = _T.T + self._TBT

        else:
            _TT = _T.T

        if self.options.algo == "implicit":
            self._info = self.options.solver_params.info

            # define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
            self._B = -1 / 2 * _TT @ curl.T @ M2
            self._B2 = -1 / 2 * _TT @ curl.T @ PB.T

            self._C = 1 / 2 * curl @ _T

            # Instantiate Schur solver (constant in this case)
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
            self._b_tmp1 = self.variables.b.spline.vector.space.zeros()

            self._byn = self._B.codomain.zeros()
            self._tmp_acc = self._B2.codomain.zeros()

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
            _f1 = A_inv @ _TT @ curl.T @ M2
            _f1_acc = A_inv @ _TT @ curl.T @ PB.T
            _f2 = curl @ _T

            # allocate output of vector field
            out_acc = self.variables.u.spline.vector.space.zeros()
            out1 = self.variables.u.spline.vector.space.zeros()
            out2 = self.variables.b.spline.vector.space.zeros()

            def f1(t, y1, y2, out: BlockVector = out1):
                _f1.dot(y2, out=out)
                _f1_acc.dot(self._ACC.vectors[0], out=out_acc)
                out += out_acc
                out.update_ghost_regions()
                return out

            def f2(t, y1, y2, out: BlockVector = out2):
                _f2.dot(y1, out=out)
                out *= -1.0
                out.update_ghost_regions()
                return out

            vector_field = {self.variables.u.spline.vector: f1, self.variables.b.spline.vector: f2}
            self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

    def __call__(self, dt):
        # update time-dependent operator TB
        if self.options.nonlinear:
            self._update_weights_TB()

        # current FE coeffs
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        # accumulate
        self._ACC(self.options.ep_scale)

        if self.options.algo == "implicit":
            # solve for new u coeffs (no tmps created here)
            byn = self._B.dot(bn, out=self._byn)
            b2acc = self._B2.dot(self._ACC.vectors[0], out=self._tmp_acc)
            byn += b2acc

            un1, info = self._schur_solver(un, byn, dt, out=self._u_tmp1)

            # new b coeffs (no tmps created here)
            _u = un.copy(out=self._u_tmp2)
            _u += un1
            bn1 = self._C.dot(_u, out=self._b_tmp1)
            bn1 *= -dt
            bn1 += bn

            diffs = self.update_feec_variables(u=un1, b=bn1)

        else:
            self._ode_solver(0.0, dt)

        if self._info and MPI.COMM_WORLD.Get_rank() == 0:
            if self.options.algo == "implicit":
                print("Status     for ShearAlfvenCurrentCoupling5D:", info["success"])
                print("Iterations for ShearAlfvenCurrentCoupling5D:", info["niter"])
                print("Maxdiff up for ShearAlfvenCurrentCoupling5D:", diffs["u"])
                print("Maxdiff b2 for ShearAlfvenCurrentCoupling5D:", diffs["b"])
                print()

    def _initialize_projection_operator_TB(self):
        r"""Initialize BasisProjectionOperator TB with the time-varying weight.

        .. math::

            \mathcal{T}^B_{(\mu,ijk),(\nu,mno)} := \hat \Pi¹_{(\mu,ijk)} \left[ \epsilon_{\mu \alpha \nu} \frac{\tilde{B}²_\alpha}{\sqrt{g}} \Lambda²_{\nu,mno} \right] \,.

        """

        # Call the projector and the space
        P1 = self.derham.P["1"]
        Vh = self.derham.Vh_fem[self._u_form]

        # Femfield for the field evaluation
        self._bf = self.derham.create_spline_function("bf", "Hdiv")

        # Initialize BasisProjectionOperator
        if self.derham._with_local_projectors:
            self._TB = BasisProjectionOperatorLocal(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=False,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )
            self._TBT = BasisProjectionOperatorLocal(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=True,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )
        else:
            self._TB = BasisProjectionOperator(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=False,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )
            self._TBT = BasisProjectionOperator(
                P1,
                Vh,
                [
                    [None, None, None],
                    [None, None, None],
                    [None, None, None],
                ],
                transposed=True,
                use_cache=True,
                polar_shift=True,
                V_extraction_op=self.derham.extraction_ops[self._u_form],
                V_boundary_op=self.derham.boundary_ops[self._u_form],
                P_boundary_op=self.derham.boundary_ops["1"],
            )

    def _update_weights_TB(self):
        """Updats time-dependent weights of the BasisProjectionOperator TB"""

        # Update Femfield
        self.variables.b.spline.vector.copy(out=self._bf.vector)

        # define callable weights
        def bf1(x, y, z):
            return self._bf(x, y, z, local=True)[0]

        def bf2(x, y, z):
            return self._bf(x, y, z, local=True)[1]

        def bf3(x, y, z):
            return self._bf(x, y, z, local=True)[2]

        from struphy.feec.utilities import RotationMatrix

        rot_B = RotationMatrix(bf1, bf2, bf3)

        fun = []

        if self._u_form == "v":
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: rot_B(e1, e2, e3)[:, :, :, m, n],
                    ]

        elif self._u_form == "1":
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: (
                            rot_B(e1, e2, e3)
                            @ self.domain.metric_inv(
                                e1,
                                e2,
                                e3,
                                change_out_order=True,
                                squeeze_out=False,
                            )
                        )[:, :, :, m, n],
                    ]

        else:
            for m in range(3):
                fun += [[]]
                for n in range(3):
                    fun[-1] += [
                        lambda e1, e2, e3, m=m, n=n: rot_B(e1, e2, e3)[:, :, :, m, n]
                        / abs(self.domain.jacobian_det(e1, e2, e3, squeeze_out=False)),
                    ]

        # update BasisProjectionOperator
        self._TB.update_weights(fun)
        self._TBT.update_weights(fun)


class CurrentCoupling5DDensity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \int \rho_0 \frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf{x} = \frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \iiint f^\text{vol} \left(1 - \frac{B_\parallel}{B^*_\parallel}\right) \tilde{\mathbf U} \times \mathbf B_f \cdot \mathbf V \, \textnormal{d} \mathbf{x} \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,

    FE coefficients update:

    .. math::

        \mathbf u^{n+1} - \mathbf u^n = -\frac{A_\textnormal{h}}{A_b} \frac{1}{\epsilon} \mathbb{L}²{\mathbb{B}}^\times_f \mathbb{N}(1/g) \mathbb{W} \mathbb{N}\left(1- \frac{\hat B^0_\parallel}{\hat B^{*0} _\parallel}\right) (\mathbb{L}²)^\top \frac{\Delta t}{2} \cdot (\mathbf u^{n+1} + \mathbf u^n) \,.

    For the detail explanation of the notations, see `2022_DriftKineticCurrentCoupling <https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2022_DriftKineticCurrentCoupling.md?ref_type=heads>`_.
    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
        energetic_ions: PICVariable = None
        b_tilde: FEECVariable = None
        ep_scale: float = 1.0
        u_space: OptsVecSpace = "Hdiv"
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None

        def __post_init__(self):
            # checks
            check_option(self.u_space, OptsVecSpace)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)
            assert isinstance(self.energetic_ions, PICVariable)
            assert self.energetic_ions.space == "Particles5D"
            assert isinstance(self.b_tilde, FEECVariable)
            assert isinstance(self.ep_scale, float)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.filter_params is None:
                self.filter_params = FilterParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        if self.options.u_space == "H1vec":
            self._u_form_int = 0
        else:
            self._u_form_int = int(self.derham.space_to_form[self.options.u_space])

        # call operatros
        id_M = "M" + self.derham.space_to_form[self.options.u_space] + "n"
        self._A = getattr(self.mass_ops, id_M)

        # magnetic equilibrium field
        unit_b1 = self.projected_equil.unit_b1
        curl_unit_b1 = self.projected_equil.curl_unit_b1
        self._b2 = self.projected_equil.b2

        # scaling factor
        epsilon = self.options.energetic_ions.species.equation_params.epsilon

        # temporary vectors to avoid memory allocation
        self._b_full = self._b2.space.zeros()
        self._rhs_v = self.variables.u.spline.vector.space.zeros()
        self._u_new = self.variables.u.spline.vector.space.zeros()

        # define Accumulator and arguments
        self._ACC = Accumulator(
            self.options.energetic_ions.particles,
            self.options.u_space,
            Pyccelkernel(accum_kernels_gc.cc_lin_mhd_5d_D),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=False,
            symmetry="asym",
            filter_params=self.options.filter_params,
        )

        self._args_accum_kernel = (
            epsilon,
            self.options.ep_scale,
            self._b_full[0]._data,
            self._b_full[1]._data,
            self._b_full[2]._data,
            unit_b1[0]._data,
            unit_b1[1]._data,
            unit_b1[2]._data,
            curl_unit_b1[0]._data,
            curl_unit_b1[1]._data,
            curl_unit_b1[2]._data,
            self._u_form_int,
        )

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        # linear solver
        self._A_inv = inverse(
            self._A,
            self.options.solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
        )

    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b2.copy(out=self._b_full)

        b_full += self.options.b_tilde.spline.vector
        b_full.update_ghost_regions()

        self._ACC(
            *self._args_accum_kernel,
        )

        # define system (M - dt/2 * A)*u^(n + 1) = (M + dt/2 * A)*u^n
        lhs = self._A - dt / 2 * self._ACC.operators[0]
        rhs = self._A + dt / 2 * self._ACC.operators[0]

        # solve linear system for updated u coefficients (in-place)
        rhs = rhs.dot(un, out=self._rhs_v)
        self._A_inv.linop = lhs

        _u = self._A_inv.solve(rhs, out=self._u_new)
        info = self._A_inv._info

        diffs = self.update_feec_variables(u=_u)

        if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
            print("Status     for CurrentCoupling5DDensity:", info["success"])
            print("Iterations for CurrentCoupling5DDensity:", info["niter"])
            print("Maxdiff up for CurrentCoupling5DDensity:", diffs["u"])
            print()


class ImplicitDiffusion(Propagator):
    r"""
    Weak, implicit discretization of the diffusion (or heat) equation (can be used as a Poisson solver too).

    Find :math:`\phi \in H^1` such that

    .. math::

        \int_\Omega \psi\, n_0(\mathbf x)\frac{\partial \phi}{\partial t}\,\textrm d \mathbf x + \int_\Omega \nabla \psi^\top D_0(\mathbf x) \nabla \phi \,\textrm d \mathbf x = \sum_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`n_0, \rho_i:\Omega \to \mathbb R` are real-valued functions and
    :math:`D_0:\Omega \to \mathbb R^{3\times 3}`
    is a positive diffusion matrix.
    Boundary terms from integration by parts are assumed to vanish.
    The equation is discretized as

    .. math::

        \left( \frac{\sigma_1}{\Delta t} \mathbb M^0_{n_0} + \mathbb G^\top \mathbb M^1_{D_0} \mathbb G \right)\, \boldsymbol\phi^{n+1} = \frac{\sigma_2}{\Delta t} \mathbb M^0_{n_0} \boldsymbol\phi^{n} + \frac{\sigma_3}{\Delta t} \sum_i(\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`M^0_{n_0}` and :math:`M^1_{D_0}` are :class:`WeightedMassOperators <struphy.feec.mass.WeightedMassOperators>`
    and :math:`\sigma_1, \sigma_2, \sigma_3 \in \mathbb R` are artificial parameters that can be tuned to
    change the model (see Notes).

    Notes
    -----

    * :math:`\sigma_1=\sigma_2=0` and :math:`\sigma_3 = \Delta t`: **Poisson solver** with a given charge density :math:`\sum_i\rho_i`.
    * :math:`\sigma_2=0` and :math:`\sigma_1 = \sigma_3 = \Delta t` : Poisson with **adiabatic electrons**.
    * :math:`\sigma_1=\sigma_2=1` and :math:`\sigma_3 = 0`: **Implicit heat equation**.
    """

    class Variables:
        def __init__(self):
            self._phi: FEECVariable = None

        @property
        def phi(self) -> FEECVariable:
            return self._phi

        @phi.setter
        def phi(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._phi = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsStabMat = Literal["M0", "M0ad", "Id"]
        OptsDiffusionMat = Literal["M1", "M1perp"]
        # propagator options
        sigma_1: float = 1.0
        sigma_2: float = 0.0
        sigma_3: float = 1.0
        divide_by_dt: bool = False
        stab_mat: OptsStabMat = "M0"
        diffusion_mat: OptsDiffusionMat = "M1"
        rho: FEECVariable | Callable | tuple[AccumulatorVector, Particles] | list = None
        rho_coeffs: float | list = None
        x0: StencilVector = None
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.diffusion_mat, self.OptsDiffusionMat)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        # always stabilize
        if xp.abs(self.options.sigma_1) < 1e-14:
            self.options.sigma_1 = 1e-14
            if MPI.COMM_WORLD.Get_rank() == 0:
                print(f"Stabilizing Poisson solve with {self.options.sigma_1 =}")

        # model parameters
        self._sigma_1 = self.options.sigma_1
        self._sigma_2 = self.options.sigma_2
        self._sigma_3 = self.options.sigma_3
        self._divide_by_dt = self.options.divide_by_dt

        phi = self.variables.phi.spline.vector

        # collect rhs
        def verify_rhs(rho) -> StencilVector | FEECVariable | AccumulatorVector:
            """Perform preliminary operations on rho to comute the rhs and return the result."""
            if rho is None:
                rhs = phi.space.zeros()
            elif isinstance(rho, FEECVariable):
                assert rho.space == "H1"
                rhs = rho
            elif isinstance(rho, AccumulatorVector):
                rhs = rho
            elif isinstance(rho, Callable):
                rhs = L2Projector("H1", self.mass_ops).get_dofs(rho, apply_bc=True)
            else:
                raise TypeError(f"{type(rho) =} is not accepted.")

            return rhs

        rho = self.options.rho
        if isinstance(rho, list):
            self._sources = []
            for r in rho:
                self._sources += [verify_rhs(r)]
        else:
            self._sources = [verify_rhs(rho)]

        # coeffs of rhs
        if self.options.rho_coeffs is not None:
            if isinstance(self.options.rho_coeffs, (list, tuple)):
                self._coeffs = self.options.rho_coeffs
            else:
                self._coeffs = [self.options.rho_coeffs]
            assert len(self._coeffs) == len(self._sources)
        else:
            self._coeffs = [1.0 for src in self.sources]

        # initial guess and solver params
        self._x0 = self.options.x0
        self._info = self.options.solver_params.info

        if self.options.stab_mat == "Id":
            stab_mat = IdentityOperator(phi.space)
        else:
            stab_mat = getattr(self.mass_ops, self.options.stab_mat)

        if isinstance(self.options.diffusion_mat, str):
            diffusion_mat = getattr(self.mass_ops, self.options.diffusion_mat)
        else:
            diffusion_mat = self.options.diffusion_mat
            assert isinstance(diffusion_mat, WeightedMassOperator)
            assert diffusion_mat.domain == self.derham.grad.codomain
            assert diffusion_mat.codomain == self.derham.grad.codomain

        # Set lhs matrices (without dt)
        self._stab_mat = stab_mat
        self._diffusion_op = self.derham.grad.T @ diffusion_mat @ self.derham.grad

        # preconditioner and solver for Ax=b
        if self.options.precond is None:
            pc = None
        else:
            # TODO: waiting for multigrid preconditioner
            pc = None

        # solver just with A_2, but will be set during call with dt
        self._solver = inverse(
            self._diffusion_op,
            self.options.solver,
            pc=pc,
            x0=self.x0,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
            recycle=self.options.solver_params.recycle,
        )

        # allocate memory for solution
        self._tmp = phi.space.zeros()
        self._rhs = phi.space.zeros()
        self._rhs2 = phi.space.zeros()
        self._tmp_src = phi.space.zeros()

    @property
    def sources(self) -> list[StencilVector | FEECVariable | AccumulatorVector]:
        """
        Right-hand side of the equation (sources).
        """
        return self._sources

    @property
    def coeffs(self) -> list[float]:
        """
        Same length as self.sources. Coefficients multiplied with sources before solve (default is 1.0).
        """
        return self._coeffs

    @property
    def x0(self):
        """
        psydac.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First guess of the iterative solver.
        """
        return self.options.x0

    @x0.setter
    def x0(self, value: StencilVector):
        """In-place setter for StencilVector/PolarVector. First guess of the iterative solver."""
        assert value.space == self.derham.Vh["0"]
        assert value.space.symbolic_space == "H1", (
            f"Right-hand side must be in H1, but is in {value.space.symbolic_space}."
        )

        if self.options.x0 is None:
            self.options.x0 = value
        else:
            self.options.x0[:] = value[:]

    @profile
    def __call__(self, dt):
        # set parameters
        if self._divide_by_dt:
            sig_1 = self._sigma_1 / dt
            sig_2 = self._sigma_2 / dt
            sig_3 = self._sigma_3 / dt
        else:
            sig_1 = self._sigma_1
            sig_2 = self._sigma_2
            sig_3 = self._sigma_3

        # compute rhs
        phin = self.variables.phi.spline.vector
        rhs = self._stab_mat.dot(phin, out=self._rhs)
        rhs *= sig_2

        self._rhs2 *= 0.0
        for src, coeff in zip(self.sources, self.coeffs):
            if isinstance(src, StencilVector):
                self._rhs2 += sig_3 * coeff * src
            elif isinstance(src, FEECVariable):
                v = src.spline.vector
                self._rhs2 += sig_3 * coeff * self.mass_ops.M0.dot(v, out=self._tmp_src)
            elif isinstance(src, AccumulatorVector):
                src()  # accumulate
                self._rhs2 += sig_3 * coeff * src.vectors[0]

        rhs += self._rhs2

        # compute lhs
        self._solver.linop = sig_1 * self._stab_mat + self._diffusion_op

        # solve
        out = self._solver.solve(rhs, out=self._tmp)
        info = self._solver._info

        if self._info:
            print(info)

        self.update_feec_variables(phi=out)


class Poisson(ImplicitDiffusion):
    r"""
    Weak discretization of the (stabilized) Poisson equation.

    Find :math:`\phi \in H^1` such that

    .. math::

        \epsilon \int_\Omega \psi\, \phi\,\textrm d \mathbf x + \int_\Omega \nabla \psi^\top \, \nabla \phi \,\textrm d \mathbf x = \sum_i \int_\Omega \psi\, \rho_i(\mathbf x)\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,,

    where :math:`\epsilon \in \mathbb R` is a stabilization parameter.
    Boundary terms from integration by parts are assumed to vanish.
    The equation is discretized as

    .. math::

        \left( \epsilon\,\mathbb S + \mathbb G^\top \mathbb M^1 \mathbb G \right)\, \boldsymbol\phi^{n+1} = \sum_i(\Lambda^0, \rho_i  )_{L^2}\,,

    where :math:`\mathbb M^1` is the :math:`H(\textnormal{curl})`-mass matrix
    and :math:`\mathbb S` is a stabilization matrix.

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    stab_eps : float
        Stabilization parameter multiplied on stab_mat (default=0.0).

    stab_mat : str
        Name of the stabilizing matrix.

    rho : StencilVector or tuple or list
        (List of) right-hand side FE coefficients of a 0-form (optional, can be set with a setter later).
        Can be either a) StencilVector or b) 2-tuple, or a list of those.
        In case b) the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    solver : dict
        Parameters for the iterative solver (see ``__init__`` for details).
    """

    @dataclass
    class Options:
        # specific literals
        OptsStabMat = Literal["M0", "M0ad", "Id"]
        # propagator options
        stab_eps: float = 0.0
        stab_mat: OptsStabMat = "Id"
        rho: FEECVariable | Callable | tuple[AccumulatorVector, Particles] | list = None
        rho_coeffs: float | list = None
        x0: StencilVector = None
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.stab_mat, self.OptsStabMat)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            # Poisson solve (-> set some params of parent class)
            self.sigma_1 = self.stab_eps
            self.sigma_2 = 0.0
            self.sigma_3 = 1.0
            self.divide_by_dt = False
            self.diffusion_mat = "M1"

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                if "sigma" not in k and k not in ("divide_by_dt", "diffusion_mat"):
                    print(f"  {k}: {v}")
        self._options = new


class VariationalMomentumAdvection(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        \int_{\Omega} \partial_t ( \rho  \mathbf{u}) \cdot \mathbf{v} \,\textrm d \mathbf x -
        \int_{\Omega} \rho \mathbf{u} \cdot  [\mathbf{u}, \mathbf{v}] \, \textrm d \mathbf x = 0 \,.

    On the logical domain:

    .. math::

        \int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \,\textrm d \boldsymbol \eta -
        \int_{\hat{\Omega}} \hat{\rho}^3 \hat{\mathbf{u}} \cdot G [\hat{\mathbf{u}}, \hat{\mathbf{v}}] \, \textrm d \boldsymbol \eta = 0 \,,

    which is discretized as

    .. math::

        \mathbb M^v[\hat{\rho}_h^n] \frac{\mathbf u^{n+1}- \mathbf u^n}{\Delta t} - \left(\sum_{\mu} (\hat{\Pi}^{0}[\hat{\mathbf u}_h^{n+1/2} \cdot \vec{\boldsymbol \Lambda}^1] \mathbb G P_{\mu} - \hat{\Pi}^0[\hat{\mathbf A}^1_{\mu,h} \cdot \vec{\boldsymbol \Lambda}^v])^\top P_i \right) \mathbb M^v[\hat{\rho}_h^n] \mathbf u^{n} = 0 ~ .

    where :math:`P_\mu` stand for the :class:`~struphy.feec.basis_projection_ops.CoordinateProjector` and the weights
    in the the two :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{n+1/2} = (\mathbf{u}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \,, \qquad \hat{\mathbf A}^1_{\mu,h} = \nabla P_\mu((\mathbf u^{n+1/2})^\top \vec{\boldsymbol \Lambda}^v)] \in V_h^1\,, \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.
    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._Mrho = self.mass_ops.WMM
        self._Mrho.inv._options["pc"] = MassMatrixDiagonalPreconditioner(self._Mrho.massop)

        self._initialize_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_diff = u.space.zeros()
        self._tmp__pc_diff = u.space.zeros()
        self._tmp_update = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_advection = u.space.zeros()

        self.brack = BracketOperator(self.derham, self._tmp_mn)
        self._dt2_brack = 2.0 * self.brack
        self.derivative = self._Mrho.massop + self._dt2_brack
        self.inv_derivative = inverse(
            self._Mrho.inv @ self.derivative,
            "gmres",
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

    def __call__(self, dt):
        if self._nonlin_solver.type == "Newton":
            self.__call_newton(dt)
        elif self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)

    def __call_newton(self, dt):
        # Initialize variable for Newton iteration
        un = self.variables.u.spline.vector
        mn = self._Mrho.massop.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self.options.nonlin_solver.tol
        err = tol + 1
        # Jacobian matrix for Newton solve
        self._dt2_brack._scalar = dt / 2
        if self._info:
            print()
            print("Newton iteration in VariationalMomentumAdvection")

        for it in range(self.options.nonlin_solver.maxiter):
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Compute the advection term
            advection = self.brack.dot(un12, out=self._tmp_advection)
            advection *= dt

            # Difference with the previous approximation :
            # diff = m^{n+1,r}-m^{n+1,r+1} = m^{n+1,r}-m^{n}+advection
            diff = mn1.copy(out=self._tmp_diff)
            diff -= mn
            diff += advection

            # Get error and stop if small enough
            err = self._get_error_newton(diff)

            if self._info:
                print("iteration : ", it, " error : ", err)
            if err < tol**2 or xp.isnan(err):
                break

            # Newton step
            pc_diff = self._Mrho.inv.dot(diff, out=self._tmp__pc_diff)
            update = self.inv_derivative.dot(pc_diff, out=self._tmp_update)
            if self._info:
                print(
                    "information on the linear solver : ",
                    self.inv_derivative._info,
                )
            un1 -= update
            mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)

        if it == self.options.nonlin_solver.maxiter - 1 or xp.isnan(err):
            print(
                f"!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(u=un1)

    def __call_picard(self, dt):
        # Initialize variable for Picard iteration
        un = self.variables.u.spline.vector
        mn = self._Mrho.massop.dot(un, out=self._tmp_mn)
        mn1 = mn.copy(out=self._tmp_mn1)
        un1 = un.copy(out=self._tmp_un1)
        tol = self.options.nonlin_solver.tol
        err = tol + 1
        # Jacobian matrix for Newton solve

        for it in range(self.options.nonlin_solver.maxiter):
            # Picard iteration
            if err < tol**2 or xp.isnan(err):
                break
            # half time step approximation
            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Compute the advection term
            advection = self.brack.dot(un12, out=self._tmp_advection)
            advection *= dt

            # Difference with the previous approximation :
            # diff = m^{n+1,r}-m^{n+1,r+1} = m^{n+1,r}-m^{n}+advection
            diff = mn1.copy(out=self._tmp_diff)
            diff -= mn
            diff += advection

            # Compute the norm of the difference
            err = self._Mrho.inv.dot_inner(self._tmp_diff, self._tmp_diff)

            # Update : m^{n+1,r+1} = m^n-advection
            mn1 = mn.copy(out=self._tmp_mn1)
            mn1 -= advection

            # Inverse the mass matrix to get the velocity
            un1 = self._Mrho.inv.dot(mn1, out=self._tmp_un1)

        if it == self.options.nonlin_solver.maxiter - 1 or xp.isnan(err):
            print(
                f"!!!WARNING: Maximum iteration in VariationalMomentumAdvection reached - not converged \n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(u=un1)

    def _initialize_mass(self):
        """Initialization of the mass matrix solver"""
        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

    def _get_error_newton(self, mn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        return err_u


class VariationalDensityEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\rho \in L^2` and  :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        &\partial_t \rho + \nabla \cdot ( \tilde{\rho} \mathbf u ) = 0 \,,
        \\[4mm]
        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x + \int_\Omega \left(\frac{|\mathbf u|^2}{2} - \frac{\partial(\rho \mathcal U(\rho))}{\partial \rho}\right) \nabla \cdot (\tilde{\rho} \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,.

    Where :math:`\tilde{\rho}` is either :math:`\rho` for full-f models, :math:`\rho_0` for linear models or :math:`\rho_0+\rho` for :math:`\delta f` models.

    In the case of linear model, the second equation is not updated.

    On the logical domain:

    .. math::

        \begin{align}
        &\partial_t \hat{\rho}^3 + \nabla \cdot ( \hat{\rho}^3 \hat{\mathbf{u}} ) = 0 ~ ,
        \\[4mm]
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        + \int_{\hat{\Omega}} \left( \frac{| DF \hat{\mathbf{u}} |^2}{2} - \frac{\partial (\hat{\rho}^3 \mathcal U)}{\partial \hat{\rho}^3} \right) \nabla \cdot (\hat{\rho}^3 \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        \end{align}

    where :math:`\mathcal U` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\frac{\mathbb M^v[\hat{\rho}_h^{n+1}] \mathbf u^{n+1}- \mathbb M^v[\hat{\rho}_h^n] \mathbf u^n}{\Delta t}
        + (\mathbb D \hat{\Pi}^{2}[\hat{\rho_h^{n}} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\left(\frac{DF \hat{\mathbf{u}}_h^{n+1} \cdot DF \hat{\mathbf{u}}_h^{n}}{2}
        - \frac{\hat{\rho}_h^{n+1}\mathcal U(\hat{\rho}_h^{n+1})-\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n})}{\hat{\rho}_h^{n+1}-\hat{\rho}_h^n} \right) = 0 ~ ,
        \\[2mm]
        &\frac{\boldsymbol \rho^{n+1}- \boldsymbol \rho^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{\rho_h^{n}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        \end{align}

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{\rho}_h^{k} = (\rho^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} .
    """

    class Variables:
        def __init__(self):
            self._rho: FEECVariable = None
            self._u: FEECVariable = None

        @property
        def rho(self) -> FEECVariable:
            return self._rho

        @rho.setter
        def rho(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._rho = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsModel = Literal[
            "pressureless",
            "barotropic",
            "full",
            "full_p",
            "full_q",
            "linear",
            "deltaf",
            "linear_q",
            "deltaf_q",
        ]
        # propagator options
        model: OptsModel = "barotropic"
        gamma: float = 5.0 / 3.0
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        s: FEECVariable = None

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        if self.options.model == "full":
            assert self.options.s is not None

        self._model = self.options.model
        self._gamma = self.options.gamma
        self._s = self.options.s
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize

        self._info = self.options.nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._Mrho = self.mass_ops.WMM
        self._Mrho.inv._options["pc"] = MassMatrixDiagonalPreconditioner(self._Mrho.massop)

        # Femfields for the projector
        self.rhof = self.derham.create_spline_function("rhof", "L2")
        self.rhof1 = self.derham.create_spline_function("rhof1", "L2")

        rho = self.variables.rho.spline.vector
        u = self.variables.u.spline.vector

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._kinetic_evaluator = KineticEnergyEvaluator(self.derham, self.domain, self.mass_ops)
        self._initialize_projectors_and_mass()
        if self._model in ["linear", "linear_q"]:
            rhotmp = self.projected_equil.n3
        elif self._model in ["deltaf", "deltaf_q"]:
            self._tmp_rho_deltaf = rho.space.zeros()
            rhotmp = rho.copy(out=self._tmp_rho_deltaf)
            rhotmp += self.projected_equil.n3
        else:
            rhotmp = rho
        self._update_weighted_MM(rhotmp)

        # bunch of temporaries to avoid allocating in the loop
        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_rhon1 = rho.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_rhon12 = rho.space.zeros()
        self._tmp_rhon_diff = rho.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_rho_advection = rho.space.zeros()
        self._linear_form_dl_drho = rho.space.zeros()

        # Compute the initial force in case we want to 'linearize' around a given equilibrium
        if self._linearize:
            self._compute_init_linear_form()

        if self._model in ["linear", "linear_q"]:
            self._update_Pirho(self.projected_equil.n3)

    @profile
    def __call__(self, dt):
        self.__call_newton(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""

        if self._info:
            print()
            print("Newton iteration in VariationalDensityEvolve")

        # Initial variables
        rhon = self.variables.rho.spline.vector
        un = self.variables.u.spline.vector

        if self._model in ["linear", "linear_q"]:
            advection = self.divPirho.dot(un, out=self._tmp_rho_advection)
            advection *= dt
            rhon1 = rhon.copy(out=self._tmp_rhon1)
            rhon1 -= advection
            self.update_feec_variables(rho=rhon1, u=un)
            return

        if self._model in ["deltaf", "deltaf_q"]:
            rho = rhon.copy(out=self._tmp_rho_deltaf)
            rho += self.projected_equil.n3
        else:
            rho = rhon
        self._update_weighted_MM(rho)
        mn = self._Mrho.massop.dot(un, out=self._tmp_mn)

        # Initialize variable for Newton iteration
        if self._model == "full":
            s = self._s.spline.vector
        else:
            s = None

        if self._model in ["deltaf", "deltaf_q"]:
            rho = rhon.copy(out=self._tmp_rho_deltaf)
            rho += self.projected_equil.n3
        else:
            rho = rhon
        self._update_Pirho(rho)

        rhon1 = rhon.copy(out=self._tmp_rhon1)
        rhon1 += self._tmp_rhon_diff
        if self._model in ["deltaf", "deltaf_q"]:
            rho = rhon1.copy(out=self._tmp_rho_deltaf)
            rho += self.projected_equil.n3
        else:
            rho = rhon1
        self._update_weighted_MM(rho)
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Newton iteration

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # rhon12 = rhon.copy(out=self._tmp_rhon12)
            # rhon12 += rhon1
            # rhon12 *= 0.5
            # if self._model == "deltaf":
            #     rhon12 += self.projected_equil.n3

            # self._update_Pirho(rhon12)

            # Update the linear form
            self._update_linear_form_dl_drho(rhon, rhon1, un, un1, s)

            # Compute the advection terms
            advection = self.divPirhoT.dot(
                self._linear_form_dl_drho,
                out=self._tmp_advection,
            )
            advection *= dt

            rho_advection = self.divPirho.dot(
                un12,
                out=self._tmp_rho_advection,
            )
            rho_advection *= dt

            # Get diff
            rhon_diff = rhon1.copy(out=self._tmp_rhon_diff)
            rhon_diff -= rhon
            rhon_diff += rho_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, rhon_diff)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt, rhon, rhon1, un, un1, s)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = rhon_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            rhon1 -= incr[1]

            # Multiply by the mass matrix to get the momentum

            if self._model in ["deltaf", "deltaf_q"]:
                rho = rhon1.copy(out=self._tmp_rho_deltaf)
                rho += self.projected_equil.n3
                self._update_weighted_MM(rho)
            else:
                self._update_weighted_MM(rhon1)

            mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            print(
                f"!!!Warning: Maximum iteration in VariationalDensityEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_rhon_diff = rhon1 - rhon
        self.update_feec_variables(rho=rhon1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import L2_transport_operator

        # Initialize the transport operator and transposed
        self.divPirho = L2_transport_operator(self.derham)
        self.divPirhoT = self.divPirho.T

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
            recycle=True,
        )

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.quad_grid_pts["0"]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        # tmps
        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
        self._rhof_values = xp.zeros(grid_shape, dtype=float)

        # Other mass matrices for newton solve
        self._M_drho = self.mass_ops.create_weighted_mass("L2", "L2")

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["3"],
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        # local version to avoid creating new version of LinearOperator every time
        self._I3 = IdentityOperator(self.derham.Vh_pol["3"])

        self._dt_pc_divPirhoT = 2 * (self.divPirhoT)
        self._dt2_pc_divPirhoT = 2 * (self.divPirhoT)
        self._dt2_divPirho = 2 * self.divPirho

        self._Jacobian[0, 0] = self._Mrho.massop + self._dt2_pc_divPirhoT @ self._kinetic_evaluator.M_un
        self._Jacobian[0, 1] = self._kinetic_evaluator.M_un1 + self._dt_pc_divPirhoT @ self._M_drho
        self._Jacobian[1, 0] = self._dt2_divPirho
        self._Jacobian[1, 1] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            "pbicgstab",
            pc=self._Mrho.inv,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver.tol,
        #                          maxiter=self._lin_solver.maxiter,
        #                          verbose=self._lin_solver.verbose,
        #                          recycle=True)

        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        # tmps
        self._eval_dl_drho = xp.zeros(grid_shape, dtype=float)

        self._uf_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)
        self._tmp_int_grid2 = xp.zeros(grid_shape, dtype=float)
        self._rhof_values = xp.zeros(grid_shape, dtype=float)
        self._rhof1_values = xp.zeros(grid_shape, dtype=float)

        if self._model == "full":
            self._tmp_de_drho = xp.zeros(grid_shape, dtype=float)
            gam = self._gamma
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                2 - gam,
            )
            self._proj_rho2_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                1 - gam,
            )
            self._proj_drho_metric_term = deepcopy(metric)

            if self._linearize:
                self._init_dener_drho = xp.zeros(grid_shape, dtype=float)

    def _update_Pirho(self, rho):
        """Update the weights of the `BasisProjectionOperator` Pirho"""

        self.divPirho.update_coeffs(rho)
        self.divPirhoT.update_coeffs(rho)

    def _update_weighted_MM(self, rho):
        """update the weighted mass matrix operator"""

        self._Mrho.update_weight(rho)

    def _update_linear_form_dl_drho(self, rhon, rhon1, un, un1, sn):
        """Update the linearform representing integration in V3 against kinetic energy"""

        self._kinetic_evaluator.get_u2_grid(un, un1, self._eval_dl_drho)

        if self._model == "barotropic":
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._rhof_values,
            )
            rhof1_values = self.rhof1.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._rhof1_values,
            )

            # self._eval_dl_drho -= (rhof_values + rhof1_values)/2
            rhof_values /= 2
            rhof1_values /= 2

            self._eval_dl_drho -= rhof_values
            self._eval_dl_drho -= rhof1_values

        if self._model == "full":
            self._energy_evaluator.evaluate_discrete_de_drho_grid(rhon, rhon1, sn, out=self._tmp_de_drho)

            self._tmp_int_grid *= 0
            self._tmp_int_grid += self._tmp_de_drho

            if self._linearize:
                self._tmp_int_grid -= self._init_dener_drho
            self._tmp_int_grid *= self._proj_rho2_metric_term

            # self._eval_dl_drho -= self._proj_rho2_metric_term * (self._energy_evaluator._DG_values + de_rhom_s)
            self._eval_dl_drho -= self._tmp_int_grid

        self._get_L2dofs_V3(self._eval_dl_drho, dofs=self._linear_form_dl_drho)

    def _compute_init_linear_form(self):
        if abs(self._gamma - 5 / 3) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_drho_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_monoatomic,
                out=self._init_dener_drho,
            )
        elif abs(self._gamma - 7 / 5) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_drho_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_diatomic,
                out=self._init_dener_drho,
            )
        else:
            raise ValueError("Gamma should be 7/5 or 5/3 for if you want to linearize")

    def _get_jacobian(self, dt, rhon, rhon1, un, un1, sn):
        self._kinetic_evaluator.assemble_M_un(un)
        self._kinetic_evaluator.assemble_M_un1(un1)

        if self._model == "barotropic":
            self._M_drho = -self.mass_ops.M3 / 2.0

        elif self._model == "full":
            self._energy_evaluator.evaluate_discrete_d2e_drho2_grid(rhon, rhon1, sn, out=self._tmp_int_grid)
            self._tmp_int_grid *= self._proj_drho_metric_term

            self._M_drho.assemble([[self._tmp_int_grid]], verbose=False)

        else:
            self._M_drho.assemble([[0.0 * self._tmp_int_grid]], verbose=False)

        # This way we can update only the scalar multiplying the operator and avoid creating multiple operators
        self._dt_pc_divPirhoT._scalar = dt
        self._dt2_pc_divPirhoT._scalar = dt / 2
        self._dt2_divPirho._scalar = dt / 2

    def _get_error_newton(self, mn_diff, rhon_diff):
        """Error for the newton method : max(|f(0)|,|f(1)|) where f is the function we're trying to nullify"""
        err_u = self._inv_Mv.dot_inner(
            self.derham.boundary_ops["v"].dot(mn_diff),
            mn_diff,
        )
        err_rho = self.mass_ops.M3.dot_inner(rhon_diff, rhon_diff)
        return max(err_rho, err_u)


class VariationalEntropyEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3` and :math:`s \in L^2` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \frac{\partial(\rho \mathcal U)}{\partial s} \nabla \cdot (s \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t s + \nabla \cdot ( s \mathbf u ) = 0 \,.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \left(\frac{\partial \hat{\rho}^3 \mathcal U}{\partial \hat{s}} \right) \nabla \cdot (\hat{s} \hat{\mathbf{v}}) \, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{s} + \nabla \cdot ( \hat{s} \hat{\mathbf{u}} ) = 0 ~ ,
        \end{align}

    where :math:`\mathcal U` depends on the chosen model. It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t} -
        (\mathbb D \hat{\Pi}^{2}[\hat{s_h^{n}} \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3\left( \frac{\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n+1})-\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n})}{\hat{s}_h^{n+1}-\hat{s}_h^n} \right) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf s^{n+1}- \mathbf s^n}{\Delta t} + \mathbb D \hat{\Pi}^{2}[\hat{s_h^{n}} \vec{\boldsymbol \Lambda}^v] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        \end{align}

    where :math:`\hat{l}^3(f)` denotes the vector representing the linear form :math:`v_h \mapsto \int_{\hat{\Omega}} f(\boldsymbol \eta) v_h(\boldsymbol \eta) d \boldsymbol \eta`, that is the vector with components

    .. math::
        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    and the weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{u}}_h^{k} = (\mathbf{u}^{k})^\top \vec{\boldsymbol \Lambda}^v \in (V_h^0)^3 \, \text{for k in} \{n, n+1/2, n+1\}, \qquad \hat{s}_h^{k} = (s^{k})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \, \text{for k in} \{n, n+1/2, n+1\} \qquad \hat{\rho}_h^{n} = (\rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \.
    """

    class Variables:
        def __init__(self):
            self._s: FEECVariable = None
            self._u: FEECVariable = None

        @property
        def s(self) -> FEECVariable:
            return self._s

        @s.setter
        def s(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._s = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsModel = Literal["full"]
        # propagator options
        model: OptsModel = "full"
        gamma: float = 5.0 / 3.0
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        rho: FEECVariable = None

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        if self.options.model == "full":
            assert self.options.rho is not None

        self._model = self.options.model
        self._gamma = self.options.gamma
        self._rho = self.options.rho
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._Mrho = self.mass_ops.WMM
        self._Mrho.inv._options["pc"] = MassMatrixDiagonalPreconditioner(self._Mrho.massop)

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        s = self.variables.s.spline.vector
        u = self.variables.u.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn12 = s.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_sn_diff = s.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn12 = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_s_advection = s.space.zeros()
        self._linear_form_dl_ds = s.space.zeros()
        if self._linearize:
            self._compute_init_linear_form()

    def __call__(self, dt):
        self.__call_newton(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        if self._info:
            print()
            print("Newton iteration in VariationalEntropyEvolve")
        sn = self.variables.s.spline.vector
        un = self.variables.u.spline.vector

        sn1 = sn.copy(out=self._tmp_sn1)
        # Initialize variable for Newton iteration
        rho = self._rho.spline.vector
        self._update_Pis(sn)

        mn = self._Mrho.massop.dot(un, out=self._tmp_mn)
        sn1 = sn.copy(out=self._tmp_sn1)
        sn1 += self._tmp_sn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Newton iteration

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self._update_linear_form_dl_ds(rho, sn, sn1)

            # Compute the advection terms
            advection = self.divPisT.dot(
                self._linear_form_dl_ds,
                out=self._tmp_advection,
            )
            advection *= dt

            s_advection = self.divPis.dot(
                un12,
                out=self._tmp_s_advection,
            )
            s_advection *= dt

            # Get diff
            sn_diff = sn1.copy(out=self._tmp_sn_diff)
            sn_diff -= sn
            sn_diff += s_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, sn_diff)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt, rho, sn, sn1)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = sn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            sn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            print(
                f"!!!Warning: Maximum iteration in VariationalEntropyEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )
        self._tmp_sn_diff = sn1 - sn
        self._tmp_un_diff = un1 - un
        self.update_feec_variables(s=sn1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and `CoordinateProjector` needed to compute the bracket term"""

        from struphy.feec.projectors import L2Projector
        from struphy.feec.variational_utilities import L2_transport_operator

        # Initialize the transport operator and transposed
        self.divPis = L2_transport_operator(self.derham)
        self.divPisT = self.divPis.T

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        # For Newton solve
        self._M_ds = self.mass_ops.create_weighted_mass("L2", "L2")

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["3"],
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        self._I3 = IdentityOperator(self.derham.Vh_pol["3"])

        # local version to avoid creating new version of LinearOperator every time
        self._dt_pc_divPisT = 2 * (self.divPisT)
        self._dt2_divPis = 2 * self.divPis

        self._Jacobian[0, 0] = self._Mrho.massop
        self._Jacobian[0, 1] = self._dt_pc_divPisT @ self._M_ds
        self._Jacobian[1, 0] = self._dt2_divPis
        self._Jacobian[1, 1] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            self.options.solver,
            pc=self._Mrho.inv,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver.tol,
        #                          maxiter=self._lin_solver.maxiter,
        #                          verbose=self._lin_solver.verbose,
        #                          recycle=True)

        # prepare for integration of linear form
        # L2-projector for V3
        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.quad_grid_pts["3"]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])
        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        if self._model == "full":
            self._tmp_de_ds = xp.zeros(grid_shape, dtype=float)
            if self._linearize:
                self._init_dener_ds = xp.zeros(grid_shape, dtype=float)

            gam = self._gamma
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                2 - gam,
            )
            self._proj_rho2_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                1 - gam,
            )
            self._proj_ds_metric_term = deepcopy(metric)

    def _update_Pis(self, s):
        """Update the weights of the `BasisProjectionOperator`"""

        self.divPis.update_coeffs(s)
        self.divPisT.update_coeffs(s)

    def _update_linear_form_dl_ds(self, rhon, sn, sn1):
        """Update the linear form representing integration in V3 against the derivative of the lagrangian"""

        if self._model == "full":
            self._energy_evaluator.evaluate_discrete_de_ds_grid(rhon, sn, sn1, out=self._tmp_de_ds)

            self._tmp_int_grid *= 0
            self._tmp_int_grid += self._tmp_de_ds

            if self._linearize:
                self._tmp_int_grid -= self._init_dener_ds
            self._tmp_int_grid *= self._proj_rho2_metric_term
            self._tmp_int_grid *= -1.0

        self._get_L2dofs_V3(self._tmp_int_grid, dofs=self._linear_form_dl_ds)

    def _compute_init_linear_form(self):
        if abs(self._gamma - 5 / 3) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_ds_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_monoatomic,
                out=self._init_dener_ds,
            )
        elif abs(self._gamma - 7 / 5) < 1e-3:
            self._energy_evaluator.evaluate_exact_de_ds_grid(
                self.projected_equil.n3,
                self.projected_equil.s3_diatomic,
                out=self._init_dener_ds,
            )
        else:
            raise ValueError("Gamma should be 7/5 or 5/3 for if you want to linearize")

    def _get_jacobian(self, dt, rhon, sn, sn1):
        if self._model == "full":
            self._energy_evaluator.evaluate_discrete_d2e_ds2_grid(rhon, sn, sn1, out=self._tmp_int_grid)
            self._tmp_int_grid *= self._proj_ds_metric_term

            self._M_ds.assemble([[self._tmp_int_grid]], verbose=False)

        # This way we can update only the scalar multiplying the operator and avoid creating multiple operators
        self._dt_pc_divPisT._scalar = dt
        self._dt2_divPis._scalar = dt / 2

    def _get_error_newton(self, mn_diff, sn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        err_rho = self.mass_ops.M3.dot_inner(sn_diff, sn_diff)
        return max(err_rho, err_u)


class VariationalMagFieldEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\mathbf \tilde{B} \times \mathbf v) \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \mathbf \tilde{B} \times \mathbf u ) = 0 \,.


    Where :math:`\tilde{\mathbf B}` is either :math:`\mathbf B` for full-f models, :math:`\mathbf B_0` for linear models or :math:`\mathbf B_0+\mathbf B` for :math:`\delta f` models.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ .
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[B_h^{n+1}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 B^{n+\frac{1}{2}} \big) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{B_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \end{align}

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+1/2})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \, \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        OptsModel = Literal["full", "full_p", "linear"]
        # propagator options
        model: OptsModel = "full"
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters(type="Newton")

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._model = self.options.model
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self._nonlin_solver.linearize

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._Mrho = self.mass_ops.WMM
        self._Mrho.inv._options["pc"] = MassMatrixDiagonalPreconditioner(self._Mrho.massop)

        # Projector
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_bn_weak_diff = b.space.zeros()

        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        if self._linearize:
            self._extracted_b2 = self.derham.extraction_ops["2"].dot(self.projected_equil.b2)

    def __call__(self, dt):
        self.__call_newton(dt)

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        if self._info:
            print()
            print("Newton iteration in VariationalMagFieldEvolve")
        # Compute implicit approximation of s^{n+1}
        un = self.variables.u.spline.vector
        bn = self.variables.b.spline.vector

        bn1 = bn.copy(out=self._tmp_bn1)
        # Initialize variable for Newton iteration

        self._update_Pib(bn)

        mn = self._Mrho.massop.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Newton iteration
            # half time step approximation
            bn12 = bn.copy(out=self._tmp_bn12)
            bn12 += bn1
            bn12 *= 0.5

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self._update_linear_form_dl_db()

            # Compute the advection terms
            if self._model == "linear":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection2 = self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += advection2

                b_advection = self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection,
                )
            else:
                advection = self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

            advection *= dt
            b_advection *= dt

            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error_newton(mn_diff, bn_diff)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            bn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            print(
                f"!!!Warning: Maximum iteration in VariationalMagFieldEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self.update_feec_variables(b=bn1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        self.curlPib = Hdiv0_transport_operator(self.derham)
        self.curlPibT = self.curlPib.T

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["2"],
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        self._I2 = IdentityOperator(self.derham.Vh_pol["2"])

        if self._model == "linear":
            # initialize the jacobian differently if linear model
            self._create_Pib0()

            self._linear_form_dl_db0 = self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT0 @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib0

        else:
            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib

        # local version to avoid creating new version of LinearOperator every time

        self._Jacobian[0, 0] = self._Mrho.massop
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            self.options.solver,
            pc=self._Mrho.inv,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #                          recycle=True)

    def _update_Pib(self, b):
        """Update the weights of the `BasisProjectionOperator`"""

        self.curlPib.update_coeffs(b)
        self.curlPibT.update_coeffs(b)

    def _create_Pib0(self):
        self.curlPib0 = Hdiv0_transport_operator(self.derham)
        self.curlPibT0 = self.curlPib0.T

        self.curlPib0.update_coeffs(self.projected_equil.b2)
        self.curlPibT0.update_coeffs(self.projected_equil.b2)

    def _update_linear_form_dl_db(self):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        if self._linearize:
            wb = self.mass_ops.M2.dot(self._tmp_bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(self._tmp_bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _get_error_newton(self, mn_diff, bn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        err_b = self.mass_ops.M2.dot_inner(bn_diff, bn_diff)
        return max(err_b, err_u)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2


class VariationalPBEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3`, :math:`p \in L^2` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf v) - \int_\Omega \frac{1}{\gamma -1} (\nabla \cdot (\tilde{p} \mathbf v))\,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \tilde{\mathbf B} \times \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t p + \nabla \cdot(\tilde{p} \mathbf u) + (\gamma - 1) \tilde{p} \nabla \cdot u = 0 \,.

    Where :math:`\tilde{\mathbf B}` (resp. :math:`\tilde{p}`) is either :math:`\mathbf B` (resp. :math:`p`) for full-f models, :math:`\mathbf B_0` (resp. :math:`p_0`) for linear models or :math:`\mathbf B_0+\mathbf B` (resp. :math:`p_0+p`) for :math:`\delta f` models.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\,
        - \frac{g}{\gamma -1} \nabla \cdot (\hat{p} \hat{v})  \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{p} + \nabla \cdot (\hat{p} \hat{u}) + (\gamma - 1 ) \hat{p} \nabla \cdot G \hat{u} = 0 \,
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 \mathbf B^{n+\frac{1}{2}} = 0 ~ ,
        - (\mathbb D \hat{\Pi}^{2}[\hat{p_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \hat{l}^3(\frac{g}{\gamma-1})
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf p^{n+1}- \mathbf p^n}{\Delta t} + \big(\mathbb D \hat{\Pi}^{2}[\hat{p_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v]]
        + (\gamma - 1)\hat{\Pi}^{3}[\hat{p_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^3] \mathbb D \mathcal{U}^v \big) \mathbf u^{n+1/2}= 0 ~ ,
        \\[2mm]
        \end{align}

    with

    .. math::

        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+\frac{1}{2}})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \,
        \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,
        \qquad \hat{p}_h^{n+1/2} = (\boldsymbol p^{n+1/2})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    and :math:`\mathcal{U}^v` is :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`.
    """

    class Variables:
        def __init__(self):
            self._p: FEECVariable = None
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def p(self) -> FEECVariable:
            return self._p

        @p.setter
        def p(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._p = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsModel = Literal["full_p", "linear", "deltaf"]
        # propagator options
        model: OptsModel = "full_p"
        gamma: float = 5.0 / 3.0
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        div_u: FEECVariable = None
        u2: FEECVariable = None
        pt3: FEECVariable = None
        bt2: FEECVariable = None

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._model = self.options.model
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize
        self._gamma = self.options.gamma

        if self.options.div_u is None:
            self._divu = None
        else:
            self._divu = self.options.div_u.spline.vector

        if self.options.u2 is None:
            self._u2 = None
        else:
            self._u2 = self.options.u2.spline.vector

        if self.options.pt3 is None:
            self._pt3 = None
        else:
            self._pt3 = self.options.pt3.spline.vector

        if self.options.bt2 is None:
            self._bt2 = None
        else:
            self._bt2 = self.options.bt2.spline.vector

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._Mrho = self.mass_ops.WMM
        self._Mrho.inv._options["pc"] = MassMatrixDiagonalPreconditioner(self._Mrho.massop)

        # Projector
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        p = self.variables.p.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un2 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_bn_weak_diff = b.space.zeros()
        self._tmp_pn1 = p.space.zeros()
        self._tmp_pn2 = p.space.zeros()
        self._tmp_pn_diff = p.space.zeros()
        self._tmp_pn_weak_diff = p.space.zeros()
        self._linear_form_dl_dp = p.space.zeros()
        self._tmp_pn12 = p.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._tmp_b_advection2 = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        self._tmp_p_advection = p.space.zeros()
        self._tmp_p_advection2 = p.space.zeros()

        self._create_linear_form_p()

        if self._linearize:
            self._extracted_b2 = self.derham.extraction_ops["2"].dot(self.projected_equil.b2)

    def __call__(self, dt):
        if self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)
        else:
            raise ValueError("Only Picard solver is implemented for VariationalPBEvolve")

    def __call_picard(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # In fact it is linear due to the explicit update, only one iteration will be done at each time step
        if self._info:
            print()
            print("Newton iteration in VariationalPBEvolve")

        un = self.variables.u.spline.vector
        pn = self.variables.p.spline.vector
        bn = self.variables.b.spline.vector

        self._update_Pib(bn)
        self._update_Projp(pn)

        mn = self._Mrho.massop.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        pn1 = pn.copy(out=self._tmp_pn1)
        pn1 += self._tmp_pn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Picard iteration
            # half time step approximation

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # pn12 = pn.copy(out=self._tmp_pn12)
            # pn12 += pn1
            # pn12 *= 0.5

            # self._update_Pib()
            # self._update_Projp()
            # Update the linear form
            self._update_linear_form_dl_db(bn, bn1)

            # Compute the advection terms
            if self._model == "linear":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                b_advection = self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                p_advection = self._transop_p0.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

            elif self._model == "deltaf":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection2,
                )

                advection += self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                b_advection += self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection2,
                )

                p_advection = self._transop_p0.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

                p_advection += self._transop_p.dot(
                    un12,
                    out=self._tmp_p_advection2,
                )

            else:
                advection = self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection2 = self._transop_pT.dot(self._linear_form_dl_dp, out=self._tmp_advection2)

                advection += advection2

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                p_advection = self._transop_p.dot(
                    un12,
                    out=self._tmp_p_advection,
                )

            advection *= dt
            b_advection *= dt
            p_advection *= dt

            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            # pn_diff = pn1.copy(out= self._tmp_pn_diff)
            # pn_diff -= pn
            # pn_diff += p_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            pn1 = pn.copy(out=self._tmp_pn1)
            pn1 -= p_advection

            # Get error
            err = self._get_error(mn_diff, bn_diff)  # , pn_diff)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            bn1 -= incr[1]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            print(
                f"!!!Warning: Maximum iteration in VariationalPBEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self._tmp_pn_diff = pn1 - pn
        self.update_feec_variables(p=pn1, b=bn1, u=un1)

        self._transop_p.div.dot(un12, out=self._divu)
        self._transop_p._Uv.dot(un1, out=self._u2)

        # Update the 2nd order variables

        if self._pt3 is not None:
            p_advection = self._transop_p.dot(
                un12,
                out=self._tmp_p_advection,
            )
            p_advection *= dt
            self._pt3 -= p_advection

        if self._bt2 is not None:
            b_advection = self.curlPib.dot(
                un12,
                out=self._tmp_b_advection,
            )
            b_advection *= dt
            self._bt2 -= b_advection

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        self.curlPib = Hdiv0_transport_operator(self.derham)
        self.curlPibT = self.curlPib.T
        self._transop_p = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma)
        self._transop_pT = self._transop_p.T

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.quad_grid_pts["3"]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        self._I2 = IdentityOperator(self.derham.Vh_pol["2"])

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["2"],
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        if self._model == "linear":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT0 @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib0

        elif self._model == "deltaf":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._full_curlPib = self.curlPib0 + self.curlPib
            self._full_curlPibT = self.curlPibT0 + self.curlPibT

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)

            self._mdt2_pc_curlPibT_M = 2 * (self._full_curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self._full_curlPib

        else:
            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs
        metric = self.domain.jacobian_det(
            *integration_grid,
        )

        self._energy_metric_term = deepcopy(metric)

        # local version to avoid creating new version of LinearOperator every time

        self._Jacobian[0, 0] = self._Mrho.massop
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2

        from struphy.linear_algebra.schur_solver import SchurSolverFull

        self._inv_Jacobian = SchurSolverFull(
            self._Jacobian,
            self.options.solver,
            pc=self._Mrho.inv,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],
        #
        #                     recycle=True)

    def _update_Pib(self, b):
        """Update the weights of the `BasisProjectionOperator`"""

        self.curlPib.update_coeffs(b)
        self.curlPibT.update_coeffs(b)

    def _create_Pib0(self):
        self.curlPib0 = Hdiv0_transport_operator(self.derham)
        self.curlPibT0 = self.curlPib.T
        self.curlPib0.update_coeffs(self.projected_equil.b2)
        self.curlPibT0.update_coeffs(self.projected_equil.b2)

    def _update_Projp(self, p):
        """Update the weights of the `BasisProjectionOperator`"""
        self._transop_p.update_coeffs(p)
        self._transop_pT.update_coeffs(p)

    def _create_transop0(self):
        """Update the weights of the `BasisProjectionOperator`"""

        self._transop_p0 = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma)
        self._transop_p0T = self._transop_p0.T
        self._transop_p0.update_coeffs(self.projected_equil.p3)
        self._transop_p0T.update_coeffs(self.projected_equil.p3)

    def _update_linear_form_dl_db(self, bn, bn1):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        if self._linearize:
            wb = self.mass_ops.M2.dot(bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _create_linear_form_p(self):
        """Update the linearform representing integration in V3 against pressure energy"""

        if self._model in ["full_p", "linear", "deltaf"]:
            self._tmp_int_grid *= 0.0
            self._tmp_int_grid -= 1.0 / (self._gamma - 1.0)
            self._tmp_int_grid *= self._energy_metric_term

        self._get_L2dofs_V3(self._tmp_int_grid, dofs=self._linear_form_dl_dp)

    def _get_error(self, mn_diff, bn_diff):  # , pn_diff):
        err_u = self._inv_Mv.dot_inner(self.derham.boundary_ops["v"].dot(mn_diff), mn_diff)
        err_b = self.mass_ops.M2.dot_inner(bn_diff, bn_diff)
        # weak_pn_diff = self.mass_ops.M3.dot(
        #     pn_diff,
        #     out=self._tmp_pn_weak_diff,
        # )
        # err_p = weak_pn_diff.dot(pn_diff)
        # print("err_b :"+str(err_b))
        # print("err_p :"+str(err_p))
        # print("err_u :"+str(err_u))
        return max(err_b, err_u)
        # return max(max(err_b, err_u),err_p)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2


class VariationalQBEvolve(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\mathbf u \in (H^1)^3`, :math:`p \in L^2` and :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x - \int_\Omega \mathbf B \cdot \nabla \times (\tilde{\mathbf B} \times \mathbf v) - \int_\Omega \frac{2 q}{\gamma -1} (\nabla \cdot (\tilde{q} \mathbf v))\,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3\,,
        \\[4mm]
        &\partial_t \mathbf B + \nabla \cdot ( \tilde{\mathbf B} \times \mathbf u ) = 0 \,,
        \\[4mm]
        &\partial_t q + \nabla \cdot(\tilde{q} \mathbf u) + (\gamma/2 - 1) \tilde{q} \nabla \cdot u = 0 \,.

    Where :math:`\tilde{\mathbf B}` (resp. :math:`\tilde{q}`) is either :math:`\mathbf B` (resp. :math:`q`) for full-f models, :math:`\mathbf B_0` (resp. :math:`q_0`) for linear models or :math:`\mathbf B_0+\mathbf B` (resp. :math:`q_0+q`) for :math:`\delta f` models.

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        - \int_{\hat{\Omega}} \hat{\mathbf{B}}^2 \cdot G \,\nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\,
        - \frac{q}{\gamma -1} \nabla \cdot (\hat{q} \hat{v})  \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{\mathbf{B}}^2 + \nabla \times (\hat{\mathbf{B}}^2 \times \hat{\mathbf{u}}) = 0 ~ ,
        \\[2mm]
        &\partial_t \hat{q} + \nabla \cdot (\hat{q} \hat{u}) + (\gamma/2 - 1 ) \hat{p} \nabla \cdot G \hat{u} = 0 \,
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        - (\mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n}} \cdot \vec{\boldsymbol \Lambda}^v])^\top \mathbb M^2 \mathbf B^{n+\frac{1}{2}}-
        \Big(\big(\mathbb D \hat{\Pi}^{2}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^v]]
        + (\gamma/2 - 1)\hat{\Pi}^{3}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^3] \mathbb D \mathcal{U}^v \big) \mathbf v \Big)^\top \hat{l}^3(\frac{2q^{n+\frac{1}{2}}}{\gamma-1}) = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf b^{n+1}- \mathbf b^n}{\Delta t} + \mathbb C \hat{\Pi}^{1}[\hat{\mathbf B_h^{n+\frac{1}{2}}} \cdot \vec{\boldsymbol \Lambda}^v]] \mathbf u^{n+1/2} = 0 ~ ,
        \\[2mm]
        &\frac{\mathbf q^{n+1}- \mathbf q^n}{\Delta t} + \big(\mathbb D \hat{\Pi}^{2}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^v]]
        + (\gamma/2 - 1)\hat{\Pi}^{3}[\hat{q_h^n} \cdot \vec{\boldsymbol \Lambda}^3] \mathbb D \mathcal{U}^v \big) \mathbf u^{n+1/2}= 0 ~ ,
        \\[2mm]
        \end{align}

    with

    .. math::

        \hat{l}^3(f)_{ijk}=\int_{\hat{\Omega}} f \Lambda^3_{ijk} \textrm d \boldsymbol \eta

    where weights in the the :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperator` and the :class:`~struphy.feec.mass.WeightedMassOperator` are given by

    .. math::

        \hat{\mathbf{B}}_h^{n+1/2} = (\mathbf{b}^{n+\frac{1}{2}})^\top \vec{\boldsymbol \Lambda}^2 \in V_h^2 \,
        \qquad \hat{\rho}_h^{n} = (\boldsymbol \rho^{n})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,
        \qquad \hat{q}_h^{n+1/2} = (\boldsymbol q^{n+1/2})^\top \vec{\boldsymbol \Lambda}^3 \in V_h^3 \,.

    and :math:`\mathcal{U}^v` is :class:`~struphy.feec.basis_projection_ops.BasisProjectionOperators`.
    """

    class Variables:
        def __init__(self):
            self._q: FEECVariable = None
            self._u: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def q(self) -> FEECVariable:
            return self._q

        @q.setter
        def q(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._q = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsModel = Literal["full_q", "linear_q", "deltaf_q"]
        # propagator options
        model: OptsModel = "full_q"
        gamma: float = 5.0 / 3.0
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        div_u: FEECVariable = None
        u2: FEECVariable = None
        qt3: FEECVariable = None
        bt2: FEECVariable = None

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._model = self.options.model
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize = self.options.nonlin_solver.linearize
        self._gamma = self.options.gamma

        if self.options.div_u is None:
            self._divu = None
        else:
            self._divu = self.options.div_u.spline.vector

        if self.options.u2 is None:
            self._u2 = None
        else:
            self._u2 = self.options.u2.spline.vector

        if self.options.qt3 is None:
            self._qt3 = None
        else:
            self._qt3 = self.options.qt3.spline.vector

        if self.options.bt2 is None:
            self._bt2 = None
        else:
            self._bt2 = self.options.bt2.spline.vector

        self._info = self._nonlin_solver.info and (self.rank == 0)

        self._Mrho = self.mass_ops.WMM
        self._Mrho.inv._options["pc"] = MassMatrixDiagonalPreconditioner(self._Mrho.massop)

        # Projector
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        q = self.variables.q.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_un_diff = u.space.zeros()
        self._tmp_bn_diff = b.space.zeros()
        self._tmp_un_weak_diff = u.space.zeros()
        self._tmp_bn_weak_diff = b.space.zeros()
        self._tmp_qn1 = q.space.zeros()
        self._tmp_qn2 = q.space.zeros()
        self._tmp_qn_diff = q.space.zeros()
        self._tmp_qn_weak_diff = q.space.zeros()
        self._linear_form_dl_dq = q.space.zeros()
        self._tmp_qn12 = q.space.zeros()
        self._tmp_mn = u.space.zeros()
        self._tmp_mn1 = u.space.zeros()
        self._tmp_mn_diff = u.space.zeros()
        self._tmp_advection = u.space.zeros()
        self._tmp_advection2 = u.space.zeros()
        self._tmp_b_advection = b.space.zeros()
        self._tmp_b_advection2 = b.space.zeros()
        self._linear_form_dl_db = b.space.zeros()

        self._tmp_q_advection = q.space.zeros()
        self._tmp_q_advection2 = q.space.zeros()

        if self._linearize:
            self._extracted_b2 = self.derham.extraction_ops["2"].dot(self.projected_equil.b2)
            self._extracted_q3 = self.derham.extraction_ops["3"].dot(self.projected_equil.q3)

    def __call__(self, dt):
        if self._nonlin_solver.type == "Picard":
            self.__call_picard(dt)
        else:
            raise ValueError("Only Picard solver is implemented for VariationalQBEvolve")

    def __call_picard(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # In fact it is linear due to the explicit update, only one iteration will be done at each time step
        if self._info:
            print()
            print("Newton iteration in VariationalQBEvolve")

        un = self.variables.u.spline.vector
        qn = self.variables.q.spline.vector
        bn = self.variables.b.spline.vector

        self._update_Pib(bn)
        self._update_Projq(qn)

        mn = self._Mrho.massop.dot(un, out=self._tmp_mn)
        bn1 = bn.copy(out=self._tmp_bn1)
        bn1 += self._tmp_bn_diff
        qn1 = qn.copy(out=self._tmp_qn1)
        qn1 += self._tmp_qn_diff
        un1 = un.copy(out=self._tmp_un1)
        un1 += self._tmp_un_diff
        mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)
        tol = self._nonlin_solver.tol
        err = tol + 1

        for it in range(self._nonlin_solver.maxiter):
            # Picard iteration
            # half time step approximation

            un12 = un.copy(out=self._tmp_un12)
            un12 += un1
            un12 *= 0.5

            # Update the linear form
            self._update_linear_form_dl_db(bn, bn1)
            self._update_linear_form_dl_dq(qn, qn1)

            # Compute the advection terms
            if self._model == "linear_q":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self._transop_q0T.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
                advection += self._transop_qT.dot(self._linear_form_dl_dq0, out=self._tmp_advection2)

                b_advection = self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                q_advection = self._transop_q0.dot(
                    un12,
                    out=self._tmp_q_advection,
                )

            elif self._model == "deltaf_q":
                advection = self.curlPibT0.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db0,
                    out=self._tmp_advection2,
                )

                advection += self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection2,
                )

                advection += self._transop_qT.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
                advection += self._transop_q0T.dot(self._linear_form_dl_dq, out=self._tmp_advection2)
                advection += self._transop_qT.dot(self._linear_form_dl_dq0, out=self._tmp_advection2)

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                b_advection += self.curlPib0.dot(
                    un12,
                    out=self._tmp_b_advection2,
                )

                q_advection = self._transop_q0.dot(
                    un12,
                    out=self._tmp_q_advection,
                )

                q_advection += self._transop_q.dot(
                    un12,
                    out=self._tmp_q_advection2,
                )

            else:
                advection = self.curlPibT.dot(
                    self._linear_form_dl_db,
                    out=self._tmp_advection,
                )

                advection += self._transop_qT.dot(self._linear_form_dl_dq, out=self._tmp_advection2)

                b_advection = self.curlPib.dot(
                    un12,
                    out=self._tmp_b_advection,
                )

                q_advection = self._transop_q.dot(
                    un12,
                    out=self._tmp_q_advection,
                )

            advection *= dt
            b_advection *= dt
            q_advection *= dt

            # Get diff
            bn_diff = bn1.copy(out=self._tmp_bn_diff)
            bn_diff -= bn
            bn_diff += b_advection

            qn_diff = qn1.copy(out=self._tmp_qn_diff)
            qn_diff -= qn
            qn_diff += q_advection

            mn_diff = mn1.copy(out=self._tmp_mn_diff)
            mn_diff -= mn
            mn_diff += advection

            # Get error
            err = self._get_error(mn_diff, bn_diff, qn_diff)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if err < tol**2 or xp.isnan(err):
                break

            # Derivative for Newton
            self._get_jacobian(dt)

            # Newton step
            self._tmp_f[0] = mn_diff
            self._tmp_f[1] = bn_diff
            self._tmp_f[2] = qn_diff

            incr = self._inv_Jacobian.dot(self._tmp_f, out=self._tmp_incr)
            if self._info:
                print(
                    "information on the linear solver : ",
                    self._inv_Jacobian._solver._info,
                )
            un1 -= incr[0]
            bn1 -= incr[1]
            qn1 -= incr[2]

            # Multiply by the mass matrix to get the momentum
            mn1 = self._Mrho.massop.dot(un1, out=self._tmp_mn1)

        if it == self._nonlin_solver.maxiter - 1 or xp.isnan(err):
            print(
                f"!!!Warning: Maximum iteration in VariationalPBEvolve reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self._tmp_un_diff = un1 - un
        self._tmp_bn_diff = bn1 - bn
        self._tmp_qn_diff = qn1 - qn
        self.update_feec_variables(q=qn1, b=bn1, u=un1)

        self._transop_q.div.dot(un12, out=self._divu)
        self._transop_q._Uv.dot(un1, out=self._u2)

        # Update the 2nd order variables

        if self._qt3 is not None:
            q_advection = self._transop_q.dot(
                un12,
                out=self._tmp_q_advection,
            )
            q_advection *= dt
            self._qt3 -= q_advection

        if self._bt2 is not None:
            b_advection = self.curlPib.dot(
                un12,
                out=self._tmp_b_advection,
            )
            b_advection *= dt
            self._bt2 -= b_advection

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        self.curlPib = Hdiv0_transport_operator(self.derham)
        self.curlPibT = self.curlPib.T
        self._transop_q = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma / 2.0)
        self._transop_qT = self._transop_q.T

        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.quad_grid_pts["3"]]

        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        # Inverse mass matrix needed to compute the error
        self.pc_Mv = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.Mv,
        )
        self._inv_Mv = inverse(
            self.mass_ops.Mv,
            "pcg",
            pc=self.pc_Mv,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        self._I2 = IdentityOperator(self.derham.Vh_pol["2"])
        self._I3 = IdentityOperator(self.derham.Vh_pol["3"])

        Jacs = BlockVectorSpace(
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["2"],
            self.derham.Vh_pol["3"],
        )

        self._tmp_f = Jacs.zeros()
        self._tmp_incr = Jacs.zeros()

        self._Jacobian = BlockLinearOperator(Jacs, Jacs)

        if self._model == "linear_q":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)
            self._linear_form_dl_dq0 = -2 / (self._gamma - 1.0) * self.mass_ops.M3.dot(self.projected_equil.q3)

            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT0 @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib0

            self._mdt2_pc_transopT_M = 2 * (self._transop_q0T @ self.mass_ops.M3)
            self._dt2_transop = 2 * self._transop_q0

        elif self._model == "deltaf_q":
            # initialize the jacobian differently if linear model
            self._create_Pib0()
            self._create_transop0()

            self._full_curlPib = self.curlPib0 + self.curlPib
            self._full_curlPibT = self.curlPibT0 + self.curlPibT

            self._full_transop = self._transop_q0 + self._transop_q
            self._full_transopT = self._transop_q0T + self._transop_qT

            self._linear_form_dl_db0 = -self.mass_ops.M2.dot(self.projected_equil.b2)
            self._linear_form_dl_dq0 = -2 / (self._gamma - 1.0) * self.mass_ops.M3.dot(self.projected_equil.q3)

            self._mdt2_pc_curlPibT_M = 2 * (self._full_curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self._full_curlPib

            self._mdt2_pc_transopT_M = 2 * (self._full_transopT @ self.mass_ops.M3)
            self._dt2_transop = 2 * self._full_transop

        else:
            self._mdt2_pc_curlPibT_M = 2 * (self.curlPibT @ self.mass_ops.M2)
            self._dt2_curlPib = 2 * self.curlPib

            self._mdt2_pc_transopT_M = 2 * (self._transop_qT @ self.mass_ops.M3)
            self._dt2_transop = 2 * self._transop_q

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs
        metric = self.domain.jacobian_det(
            *integration_grid,
        )

        self._energy_metric_term = deepcopy(metric)

        # local version to avoid creating new version of LinearOperator every time

        self._Jacobian[0, 0] = self._Mrho.massop
        self._Jacobian[0, 1] = self._mdt2_pc_curlPibT_M
        self._Jacobian[0, 2] = self._mdt2_pc_transopT_M
        self._Jacobian[1, 0] = self._dt2_curlPib
        self._Jacobian[1, 1] = self._I2
        self._Jacobian[2, 0] = self._dt2_transop
        self._Jacobian[2, 2] = self._I3

        from struphy.linear_algebra.schur_solver import SchurSolverFull3

        self._inv_Jacobian = SchurSolverFull3(
            self._Jacobian,
            self.options.solver,
            pc=self._Mrho.inv,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=self._lin_solver.verbose,
            recycle=True,
        )

        # self._inv_Jacobian = inverse(self._Jacobian,
        #                          'gmres',
        #                          tol=self._lin_solver['tol'],
        #                          maxiter=self._lin_solver['maxiter'],
        #                          verbose=self._lin_solver['verbose'],

        #                     recycle=True)

    def _update_Pib(self, b):
        """Update the weights of the `BasisProjectionOperator`"""

        self.curlPib.update_coeffs(b)
        self.curlPibT.update_coeffs(b)

    def _create_Pib0(self):
        self.curlPib0 = Hdiv0_transport_operator(self.derham)
        self.curlPibT0 = self.curlPib.T
        self.curlPib0.update_coeffs(self.projected_equil.b2)
        self.curlPibT0.update_coeffs(self.projected_equil.b2)

    def _update_Projq(self, q):
        """Update the weights of the `BasisProjectionOperator`"""
        self._transop_q.update_coeffs(q)
        self._transop_qT.update_coeffs(q)

    def _create_transop0(self):
        """Update the weights of the `BasisProjectionOperator`"""

        self._transop_q0 = Pressure_transport_operator(self.derham, self.domain, self.basis_ops.Uv, self._gamma / 2.0)
        self._transop_q0T = self._transop_q0.T
        self._transop_q0.update_coeffs(self.projected_equil.q3)
        self._transop_q0T.update_coeffs(self.projected_equil.q3)

    def _update_linear_form_dl_db(self, bn, bn1):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 *= 0.5
        if self._linearize:
            wb = self.mass_ops.M2.dot(bn12 - self._extracted_b2, out=self._linear_form_dl_db)
        else:
            wb = self.mass_ops.M2.dot(bn12, out=self._linear_form_dl_db)
        wb *= -1

    def _update_linear_form_dl_dq(self, qn, qn1):
        """Update the linearform representing integration in V2 derivative of the lagrangian"""
        qn12 = qn.copy(out=self._tmp_qn12)
        qn12 += qn1
        qn12 *= 0.5
        if self._linearize:
            wq = self.mass_ops.M3.dot(qn12 - self._extracted_q2, out=self._linear_form_dl_dq)
        else:
            wq = self.mass_ops.M3.dot(qn12, out=self._linear_form_dl_dq)
        wq *= -2 / (self._gamma - 1)

    def _get_error(self, mn_diff, bn_diff, qn_diff):
        err_u = self._inv_Mv.dot_inner(
            self.derham.boundary_ops["v"].dot(mn_diff),
            self.derham.boundary_ops["v"].dot(mn_diff),
        )
        err_b = self.mass_ops.M2.dot_inner(
            bn_diff,
            bn_diff,
        )
        err_q = self.mass_ops.M3.dot_inner(
            qn_diff,
            qn_diff,
        )
        # print("err_b :"+str(err_b))
        # print("err_p :"+str(err_p))
        # print("err_u :"+str(err_u))
        # return max(err_b, err_u)
        return max(max(err_b, err_u), err_q)

    def _get_jacobian(self, dt):
        self._mdt2_pc_curlPibT_M._scalar = -dt / 2
        self._dt2_curlPib._scalar = dt / 2
        self._mdt2_pc_transopT_M._scalar = -dt / (self._gamma - 1.0)
        self._dt2_transop._scalar = dt / 2


class VariationalViscosity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`s \in L^2` and  :math:`\mathbf u \in (H^1)^3` such that

    .. math::

        &\int_\Omega \partial_t (\rho \mathbf u) \cdot \mathbf v\,\textrm d \mathbf x + \int_\Omega (\mu + \mu_a(\mathbf x)) \nabla \mathbf u : \nabla \mathbf v \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in (H^1)^3 \,,
        \\[4mm]
        &\int_\Omega \frac{\partial \mathcal U}{\partial s} \partial_t s \, q \,\textrm d \mathbf x - \mu \int_\Omega |\nabla \mathbf u|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\,\text{if using } s,
        \\[4mm]
        &\int_\Omega \frac{1}{\gamma - 1} \partial_t p \, q\,\textrm d \mathbf x - \mu \int_\Omega |\nabla \mathbf u|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\, \text{if using } p.

    With :math:`\mu_a(\mathbf x) = \mu_a |\nabla \mathbf u(\mathbf x)|`

    On the logical domain:

    .. math::

        \begin{align}
        &\int_{\hat{\Omega}} \partial_t ( \hat{\rho}^3  \hat{\mathbf{u}}) \cdot G \hat{\mathbf{v}} \, \textrm d \boldsymbol \eta
        + \mu \int_{\hat{\Omega}} \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{v}}) \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta = 0 ~ ,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\hat{\rho} \hat{e}(\hat{\rho}, \hat{s})) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta -  \int_{\hat{\Omega}} (\mu + \mu_a(\boldsymbol \eta)) \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{u}}) \hat{w} \, \textrm d \boldsymbol \eta = 0 ~ , \text{if using } s,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\frac{1}{\gamma -1} \hat{p} ) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta - \int_{\hat{\Omega}} (\mu + \mu_a(\boldsymbol \eta)) \nabla (DF \hat{\mathbf{u}}) : \nabla (DF \hat{\mathbf{u}}) \hat{w} \, \textrm d \boldsymbol \eta = 0 ~, \text{if using } p.
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\mathbb M^v[\hat{\rho}_h^{n}] \frac{ \mathbf u^{n+1}-\mathbf u^n}{\Delta t}
        +  \sum_\nu (\mathbb G \mathcal{X}^v_\nu)^T (\mu \mathbb M_0 + \mu_a \mathbb M_0[|\nabla u|] \mathbb G \mathcal{X}^v_\nu \mathbf u^{n+1} = 0 ~ ,
        \\[2mm]
        &\frac{P^{3}(\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n}))- P^{3}(\hat{\rho}_h^{n}\mathcal U(\hat{\rho}_h^{n},\hat{s}_h^{n+1}))}{\Delta t} - \mu P^3(\sum_\nu DF \mathcal{X}^v_\nu \frac{ \mathbf u^{n+1}+\mathbf u^n}{2} \cdot DF \mathcal{X}^v_\nu \mathbf u^{n+1}) = 0 ~ , \text{if using } s,
        \\[2mm]
        &\frac{1}{\gamma -1}\frac{p^{n+1}- p^{n}}{\Delta t} - \mu P^3(\sum_\nu DF \mathcal{X}^v_\nu \frac{ \mathbf u^{n+1}+\mathbf u^n}{2} \cdot DF \mathcal{X}^v_\nu \mathbf u^{n+1}) = 0 ~ , \text{if using } p.
        \end{align}

    where $P^3$ denotes the $L^2$ projection in the last space of the de Rham sequence and the weights in :math:`\mathbb M_0[|\nabla u|]` are given by

    .. math::
        P^0(g \sqrt{\sum_\nu |(\mathbb G \mathcal{X}^v_\nu \mathbb u)^\top \vec{\boldsymbol \Lambda}^0 |^2]})^\top \vec{\boldsymbol \Lambda}^0 ~.

    """

    class Variables:
        def __init__(self):
            self._s: FEECVariable = None
            self._u: FEECVariable = None

        @property
        def s(self) -> FEECVariable:
            return self._s

        @s.setter
        def s(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._s = new

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1vec"
            self._u = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsModel = Literal["full", "full_p", "full_q", "linear_p", "linear_q", "deltaf_q"]
        # propagator options
        model: OptsModel = "full"
        gamma: float = 5.0 / 3.0
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixDiagonalPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        rho: FEECVariable = None
        pt3: FEECVariable = None
        mu: float = 0.0
        mu_a: float = 0.0
        alpha: float = 0.0

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters(type="Newton")

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._model = self.options.model
        self._gamma = self.options.gamma
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._mu_a = self.options.mu_a
        self._alpha = self.options.alpha
        self._mu = self.options.mu
        self._rho = self.options.rho
        self._pt3 = self.options.pt3

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        self._Mrho = self.mass_ops.WMM
        self._Mrho.inv._options["pc"] = MassMatrixDiagonalPreconditioner(self._Mrho.massop)

        # Femfields for the projector
        self.sf = self.derham.create_spline_function("sf", "L2")
        self.sf1 = self.derham.create_spline_function("sf1", "L2")
        self.uf1 = self.derham.create_spline_function("uf", "H1vec")
        self.uf12 = self.derham.create_spline_function("uf1", "H1vec")
        self.gu0f = self.derham.create_spline_function("gu0", "Hcurl")
        self.gu1f = self.derham.create_spline_function("gu1", "Hcurl")
        self.gu2f = self.derham.create_spline_function("gu2", "Hcurl")
        self.gu120f = self.derham.create_spline_function("gu120", "Hcurl")
        self.gu121f = self.derham.create_spline_function("gu121", "Hcurl")
        self.gu122f = self.derham.create_spline_function("gu122", "Hcurl")

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        u = self.variables.u.spline.vector
        s = self.variables.s.spline.vector

        self._tmp_un1 = u.space.zeros()
        self._tmp_un12 = u.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn_incr = s.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_gu0 = self.derham.Vh_pol["1"].zeros()
        self._tmp_gu1 = self.derham.Vh_pol["1"].zeros()
        self._tmp_gu2 = self.derham.Vh_pol["1"].zeros()
        self._tmp_gu120 = self.derham.Vh_pol["1"].zeros()
        self._tmp_gu121 = self.derham.Vh_pol["1"].zeros()
        self._tmp_gu122 = self.derham.Vh_pol["1"].zeros()
        self._linear_form_tot_e = s.space.zeros()
        self._linear_form_en1 = s.space.zeros()
        self.tot_rhs = s.space.zeros()

    def __call__(self, dt):
        if self._nonlin_solver.type == "Newton":
            self.__call_newton(dt)
        else:
            raise ValueError(
                "wrong value for solver type in VariationalViscosity",
            )

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Compute dissipation implicitely
        sn = self.variables.s.spline.vector
        un = self.variables.u.spline.vector

        if self._mu < 1.0e-15 and self._mu_a < 1.0e-15 and self._alpha < 1.0e-15:
            self.update_feec_variables(s=sn, u=un)
            return

        if self._info:
            print()
            print("Computing the dissipation in VariationalViscosity")

        # Update artificial viscosity weighted mass matrix
        total_viscosity = self._update_artificial_viscosity(un, dt)

        self._scaled_stiffness._scalar = dt * self._mu  # /2.
        self._scaled_Mv._scalar = dt * self._alpha
        # self.evol_op._multiplicants[1]._addends[0]._scalar = - dt*self._mu/2.
        un1 = self.evol_op.dot(un, out=self._tmp_un1)
        if self._info:
            print("information on the linear solver : ", self.inv_lop._info)

        if self._model == "linear_p" or (self._model == "linear_q" and self._nonlin_solver["fast"]):
            self.update_feec_variables(s=sn, u=un1)
            return

        # Energy balance term
        # 1) Pointwize energy change
        energy_change = self._get_energy_change(un, un1, dt, total_viscosity)
        # 2) Initial energy and linear form
        rho = self._rho
        if self._model in ["deltaf_q", "linear_q"]:
            self.sf.vector = self._pt3.spline.vector
        else:
            self.sf.vector = sn

        sf_values = self.sf.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_bd,
            out=self._sf_values,
        )

        if self._model == "full":
            rhof_values = self._energy_evaluator.eval_3form(rho, out=self._rhof_values)

            e_n = self._energy_evaluator.ener(
                rhof_values,
                sf_values,
                out=self._e_n,
            )

            e_n *= self._energy_metric

        elif self._model == "full_p":
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n *= 1.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        elif self._model in ["full_q"]:
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n **= 2
            e_n *= 1.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        elif self._model in ["linear_q", "deltaf_q"]:
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n *= self._q0_values
            e_n *= 2.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        energy_change += e_n

        self._get_L2dofs_V3(energy_change, dofs=self._linear_form_tot_e)

        # 3) Newton iteration
        sn1 = sn.copy(out=self._tmp_sn1)

        tol = self._nonlin_solver["tol"]
        err = tol + 1

        for it in range(self._nonlin_solver["maxiter"]):
            if self._model in ["deltaf_q", "linear_q"]:
                self.sf1.vector = self._pt3.spline.vector
            else:
                self.sf1.vector = sn1

            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._sf1_values,
            )

            if self._model == "full":
                e_n1 = self._energy_evaluator.ener(
                    rhof_values,
                    sf1_values,
                    out=self._e_n1,
                )
                e_n1 *= self._energy_metric

            elif self._model == "full_p":
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 *= 1.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            elif self._model in ["full_q"]:
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 **= 2
                e_n1 *= 1.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            elif self._model in ["linear_q", "deltaf_q"]:
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 *= self._q0_values
                e_n1 *= 2.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            self._get_L2dofs_V3(e_n1, dofs=self._linear_form_en1)

            self.tot_rhs *= 0.0
            self.tot_rhs -= self._linear_form_en1
            self.tot_rhs += self._linear_form_tot_e

            err = self._get_error_newton(self.tot_rhs)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if (err < tol**2 and it > 0) or xp.isnan(err):
                # force at least one iteration
                break

            if self._model == "full":
                deds = self._energy_evaluator.dener_ds(
                    rhof_values,
                    sf1_values,
                    out=self._de_s1_values,
                )
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]], verbose=False)
                self.pc_jac.update_mass_operator(self.M_de_ds)

            elif self._model in ["full_q", "linear_q", "deltaf_q"]:
                if self._model in ["deltaf_q", "linear_q"]:
                    sf1_values = self._q0_values

                deds = self._de_s1_values
                deds *= 0.0
                deds += sf1_values
                deds *= 2 / (self._gamma - 1.0)
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]], verbose=False)
                self.pc_jac.update_mass_operator(self.M_de_ds)

            incr = self.inv_jac.dot(self.tot_rhs, out=self._tmp_sn_incr)

            if self._info:
                print("information on the linear solver : ", self.inv_jac._info)

            if self._model in ["deltaf_q", "linear_q"]:
                self._pt3 += incr
            else:
                sn1 += incr

        if it == self._nonlin_solver["maxiter"] - 1 or xp.isnan(err):
            print(
                f"!!!Warning: Maximum iteration in VariationalViscosity reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(s=sn1, u=un1)

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        Xv = getattr(self.basis_ops, "Xv")
        Pcoord0 = CoordinateProjector(
            0,
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["0"],
        )
        Pcoord1 = CoordinateProjector(
            1,
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["0"],
        )
        Pcoord2 = CoordinateProjector(
            2,
            self.derham.Vh_pol["v"],
            self.derham.Vh_pol["0"],
        )

        M1 = self.mass_ops.M1
        self.M1_du = self.mass_ops.create_weighted_mass("Hcurl", "Hcurl")

        self.pc_M3 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M3,
        )
        self._inv_M3 = inverse(
            self.mass_ops.M3,
            "pcg",
            pc=self.pc_M3,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        self.M_de_ds = self.mass_ops.create_weighted_mass("L2", "L2")

        if self.options.precond is None:
            self.pc_jac = None
        else:
            pc_class = getattr(
                preconditioner,
                self.options.precond,
            )
            self.pc_jac = pc_class(self.M_de_ds)

        self.inv_jac = inverse(
            self.M_de_ds,
            "pcg",
            pc=self.pc_jac,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        grad = self.derham.grad_bcfree
        self.scalar_stiffness = grad.T @ M1 @ grad
        self.log_stiffness = (
            Pcoord0.T @ self.scalar_stiffness @ Pcoord0
            + Pcoord1.T @ self.scalar_stiffness @ Pcoord1
            + Pcoord2.T @ self.scalar_stiffness @ Pcoord2
        )

        self.phy_stiffness = Xv.T @ self.log_stiffness @ Xv

        self._scaled_stiffness = 0.00001 * self.phy_stiffness

        self.du_stiffness = grad.T @ self.M1_du @ grad
        self.du_log_stiffness = (
            Pcoord0.T @ self.du_stiffness @ Pcoord0
            + Pcoord1.T @ self.du_stiffness @ Pcoord1
            + Pcoord2.T @ self.du_stiffness @ Pcoord2
        )

        self.du_phy_stiffness = Xv.T @ self.du_log_stiffness @ Xv

        self._scaled_stiffness = 0.00001 * self.phy_stiffness

        self._scaled_Mv = 0.1 * self.mass_ops.Mv

        self.r_op = self._Mrho.massop  # - self._scaled_stiffness - self.du_phy_stiffness
        self.l_op = self._Mrho.massop + self._scaled_Mv + self._scaled_stiffness + self.du_phy_stiffness

        self.grad_0 = grad @ Pcoord0 @ Xv
        self.grad_1 = grad @ Pcoord1 @ Xv
        self.grad_2 = grad @ Pcoord2 @ Xv

        self.inv_lop = inverse(
            self.l_op,
            "pcg",
            pc=self._Mrho.inv,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        self.evol_op = self.inv_lop @ self.r_op
        # self.evol_op = IdentityOperator(self.derham.Vh_pol['v'])
        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.quad_grid_pts["3"]]
        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        self.integration_grid_gradient = [
            [self.integration_grid_bd[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
            [
                self.integration_grid_bn[0],
                self.integration_grid_bd[1],
                self.integration_grid_bn[2],
            ],
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bd[2]],
        ]

        self.integration_grid_u = [
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
            [
                self.integration_grid_bn[0],
                self.integration_grid_bn[1],
                self.integration_grid_bn[2],
            ],
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
        ]

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._guf0_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf2_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._guf120_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf121_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._guf122_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._uf1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._uf12_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._gu_sq_values = xp.zeros(grid_shape, dtype=float)
        self._u_sq_values = xp.zeros(grid_shape, dtype=float)
        self._gu_init_values = xp.zeros(grid_shape, dtype=float)

        self._sf_values = xp.zeros(grid_shape, dtype=float)
        self._sf1_values = xp.zeros(grid_shape, dtype=float)
        self._rhof_values = xp.zeros(grid_shape, dtype=float)

        self._e_n1 = xp.zeros(grid_shape, dtype=float)
        self._e_n = xp.zeros(grid_shape, dtype=float)

        self._de_s1_values = xp.zeros(grid_shape, dtype=float)

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        gam = self._gamma
        if self._model == "full":
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -gam,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                1 - gam,
            )
            self._energy_metric = deepcopy(metric)

        elif self._model == "full_p":
            metric = 1.0 / self.domain.jacobian_det(
                *integration_grid,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = (
                0
                * self.domain.jacobian_det(
                    *integration_grid,
                )
                + 1.0
            )
            self._energy_metric = deepcopy(metric)

            # no need to compute this every time step
            deds = self._de_s1_values
            deds *= 0.0
            deds += 1 / (self._gamma - 1.0)
            deds *= self._mass_metric_term

            self.M_de_ds.assemble([[deds]], verbose=False)
            self.pc_jac.update_mass_operator(self.M_de_ds)

        elif self._model in ["full_q", "linear_q", "deltaf_q"]:
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -2,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -1,
            )
            self._energy_metric = deepcopy(metric)

        metric = xp.power(
            self.domain.jacobian_det(
                *integration_grid,
            ),
            1,
        )
        self._sq_term_metric = deepcopy(metric)

        metric = self.domain.metric_inv(
            *integration_grid,
        ) * self.domain.jacobian_det(*integration_grid)
        self._mass_M1_metric = deepcopy(metric)

        if self._model in ["linear_q", "deltaf_q"]:
            self.sf1.vector = self.projected_equil.q3

            self._q0_values = self.sf1.eval_tp_fixed_loc(self.integration_grid_spans, self.integration_grid_bd)

        metric = self.domain.metric(
            *integration_grid,
        ) * self.domain.jacobian_det(*integration_grid)
        self._mass_Mv_metric = deepcopy(metric)

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

    def _get_error_newton(self, sn_diff):
        err_s = self._inv_M3.dot_inner(sn_diff, sn_diff)
        return err_s

    def _update_artificial_viscosity(self, un, dt):
        """Update the artificial viscosity as the norm of the gradient of un.
        Update the associated mass matrix and return the total viscosity for later computation"""
        gu0 = self.grad_0.dot(un, out=self._tmp_gu0)
        gu1 = self.grad_1.dot(un, out=self._tmp_gu1)
        gu2 = self.grad_2.dot(un, out=self._tmp_gu2)

        self.gu0f.vector = gu0
        self.gu1f.vector = gu1
        self.gu2f.vector = gu2

        gu0_v = self.gu0f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf0_values,
        )
        gu1_v = self.gu1f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf1_values,
        )
        gu2_v = self.gu2f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf2_values,
        )

        gu_sq_v = self._gu_init_values
        gu_sq_v *= 0.0
        for i in range(3):
            gu0_v[i] **= 2
            gu1_v[i] **= 2
            gu2_v[i] **= 2
            gu_sq_v += gu0_v[i]
            gu_sq_v += gu1_v[i]
            gu_sq_v += gu2_v[i]

        xp.sqrt(gu_sq_v, out=gu_sq_v)

        gu_sq_v *= dt * self._mu_a  # /2

        self.M1_du.assemble(
            [
                [
                    gu_sq_v * self._mass_M1_metric[0, 0],
                    gu_sq_v * self._mass_M1_metric[0, 1],
                    gu_sq_v * self._mass_M1_metric[0, 2],
                ],
                [
                    gu_sq_v * self._mass_M1_metric[1, 0],
                    gu_sq_v * self._mass_M1_metric[1, 1],
                    gu_sq_v * self._mass_M1_metric[1, 2],
                ],
                [
                    gu_sq_v * self._mass_M1_metric[2, 0],
                    gu_sq_v * self._mass_M1_metric[2, 1],
                    gu_sq_v * self._mass_M1_metric[2, 2],
                ],
            ],
            verbose=False,
        )

        # gu_sq_v *= 2.
        gu_sq_v += dt * self._mu

        return gu_sq_v

    def _get_energy_change(self, un, un1, dt, total_viscosity):
        """Return the total energy change caused by the viscosity"""
        un12 = un.copy(out=self._tmp_un12)
        un12 += un1
        un12 /= 2.0
        gu0 = self.grad_0.dot(un1, out=self._tmp_gu0)
        gu1 = self.grad_1.dot(un1, out=self._tmp_gu1)
        gu2 = self.grad_2.dot(un1, out=self._tmp_gu2)

        gu012 = self.grad_0.dot(un12, out=self._tmp_gu120)
        gu112 = self.grad_1.dot(un12, out=self._tmp_gu121)
        gu212 = self.grad_2.dot(un12, out=self._tmp_gu122)

        self.gu0f.vector = gu0
        self.gu1f.vector = gu1
        self.gu2f.vector = gu2

        self.gu120f.vector = gu012
        self.gu121f.vector = gu112
        self.gu122f.vector = gu212

        self.uf1.vector = un1
        self.uf12.vector = un12

        gu0_v = self.gu0f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf0_values,
        )
        gu1_v = self.gu1f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf1_values,
        )
        gu2_v = self.gu2f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf2_values,
        )

        gu120_v = self.gu120f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf120_values,
        )
        gu121_v = self.gu121f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf121_values,
        )
        gu122_v = self.gu122f.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_gradient,
            out=self._guf122_values,
        )

        u1_v = self.uf1.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_u,
            out=self._uf1_values,
        )
        u12_v = self.uf12.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_u,
            out=self._uf12_values,
        )

        gu_sq_v = self._gu_sq_values
        u_sq_v = self._u_sq_values
        gu_sq_v *= 0.0
        u_sq_v *= 0.0
        for i in range(3):
            for j in range(3):
                gu_sq_v += gu0_v[i] * self._mass_M1_metric[i, j] * gu120_v[j]
                gu_sq_v += gu1_v[i] * self._mass_M1_metric[i, j] * gu121_v[j]
                gu_sq_v += gu2_v[i] * self._mass_M1_metric[i, j] * gu122_v[j]
                u_sq_v += u1_v[i] * self._mass_Mv_metric[i, j] * u12_v[j]

        gu_sq_v *= total_viscosity
        u_sq_v *= dt * self._alpha
        gu_sq_v += u_sq_v

        return gu_sq_v


class VariationalResistivity(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`s \in L^2` and  :math:`\mathbf B \in H(\textrm{div})` such that

    .. math::

        &\int_\Omega \partial_t \mathbf B \cdot \mathbf v \,\textrm d \mathbf x + \int_\Omega (\eta + \eta_a(\mathbf x)) \nabla \times \mathbf B \cdot \nabla \times \mathbf v \,\textrm d \mathbf x = 0 \qquad \forall \, \mathbf v \in H(\textrm{div}) \,,
        \\[4mm]
        &\int_\Omega \frac{\partial \mathcal U}{\partial s} \partial_t s \, q\,\textrm d \mathbf x - \int_\Omega (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\, \text{if using } s,
        \\[4mm]
        &\int_\Omega \frac{1}{\gamma - 1} \partial_t p \, q\,\textrm d \mathbf x - \int_\Omega (\eta + \eta_a(\mathbf x)) |\nabla \times \mathbf B|^2 \, q \,\textrm d \mathbf x = 0 \qquad \forall \, q \in L^2\, \text{if using } p.

    With :math:`\eta_a(\mathbf x) = \eta_a |\nabla \times \mathbf B(\mathbf x)|`

    On the logical domain:

    .. math::

        \begin{align}
        &\partial_t \hat{\boldsymbol B} - \eta \Delta \hat{\boldsymbol B} = 0 ~ ,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\hat{\rho} \mathcal U(\hat{\rho}, \hat{s})) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta - \int_{\hat{\Omega}} (\eta + \eta_a(\mathbf \eta)) |DF^{-T}\tilde{\nabla} \times \hat{\boldsymbol B}|^2  \hat{w} \, \textrm d \boldsymbol \eta = 0 ~ , \text{if using } s,
        \\[2mm]
        &\int_{\hat{\Omega}} \partial_t (\frac{1}{\gamma -1} \hat{p} ) \hat{w} \,\frac{1}{\sqrt g}\, \textrm d \boldsymbol \eta - \int_{\hat{\Omega}} (\eta + \eta_a(\mathbf \eta)) |DF^{-T}\tilde{\nabla} \times \hat{\boldsymbol B}|^2  \hat{w} \, \textrm d \boldsymbol \eta = 0 ~, \text{if using } p.
        \end{align}

    It is discretized as

    .. math::

        \begin{align}
        &\frac{\mathbf B^{n+1}-\mathbf B^n}{\Delta t}
        + \, \mathbb C \mathbb M_1^{-1} (\eta M_1 + \eta_a M_1[|\nabla \times \mathbf B|]) M_1^{-1} \mathbb C^T \mathbb M_2  \mathbf B^{n+1} = 0 ~ ,
        \\[2mm]
        &\frac{P^{3}(\rho e(s^{n+1})- P^{3}(\rho e(s^{n}))}{\Delta t} - P^3((\eta + \eta_a(\mathbf x)) DF^{-T} \tilde{\mathbb C} \frac{ \mathbf B^{n+1}+\mathbf B^n}{2} \cdot DF^{-T} \tilde{\mathbb C} \mathbf B^{n+1}) = 0 ~ , \text{if using } s,
        \\[2mm]
        &\frac{1}{\gamma -1}\frac{p^{n+1}- p^{n}}{\Delta t} - P^3((\eta + \eta_a(\mathbf x)) DF^{-T} \tilde{\mathbb C} \frac{ \mathbf B^{n+1}+\mathbf B^n}{2} \cdot DF^{-T} \tilde{\mathbb C} \mathbf B^{n+1}) = 0 ~ , \text{if using } p.
        \end{align}

    where $P^3$ denotes the $L^2$ projection in the last space of the de Rham sequence.

    """

    class Variables:
        def __init__(self):
            self._s: FEECVariable = None
            self._b: FEECVariable = None

        @property
        def s(self) -> FEECVariable:
            return self._s

        @s.setter
        def s(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._s = new

        @property
        def b(self) -> FEECVariable:
            return self._b

        @b.setter
        def b(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._b = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsModel = Literal["full", "full_p", "full_q", "linear_p", "linear_q", "deltaf_q"]
        # propagator options
        model: OptsModel = "full"
        gamma: float = 5.0 / 3.0
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixDiagonalPreconditioner"
        solver_params: SolverParameters = None
        nonlin_solver: NonlinearSolverParameters = None
        linearize_current: bool = False
        rho: FEECVariable = None
        pt3: FEECVariable = None
        eta: float = 0.0
        eta_a: float = 0.0

        def __post_init__(self):
            # checks
            check_option(self.model, self.OptsModel)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.nonlin_solver is None:
                self.nonlin_solver = NonlinearSolverParameters(type="Newton")

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._model = self.options.model
        self._gamma = self.options.gamma
        self._eta = self.options.eta
        self._eta_a = self.options.eta_a
        self._lin_solver = self.options.solver_params
        self._nonlin_solver = self.options.nonlin_solver
        self._linearize_current = self.options.linearize_current
        self._rho = self.options.rho
        self._pt3 = self.options.pt3

        self._info = self._nonlin_solver.info and (MPI.COMM_WORLD.Get_rank() == 0)

        # Femfields for the projector
        self.rhof = self.derham.create_spline_function("rhof", "L2")
        self.sf = self.derham.create_spline_function("sf", "L2")
        self.sf1 = self.derham.create_spline_function("sf1", "L2")
        self.bf = self.derham.create_spline_function("Bf", "Hdiv")
        self.bf1 = self.derham.create_spline_function("Bf1", "Hdiv")
        self.cbf1 = self.derham.create_spline_function("cBf", "Hcurl")
        self.cbf12 = self.derham.create_spline_function("cBf", "Hcurl")

        # Projector
        self._energy_evaluator = InternalEnergyEvaluator(self.derham, self._gamma)
        self._initialize_projectors_and_mass()

        # bunch of temporaries to avoid allocating in the loop
        s = self.variables.s.spline.vector
        b = self.variables.b.spline.vector

        self._tmp_bn1 = b.space.zeros()
        self._tmp_bn12 = b.space.zeros()
        self._tmp_sn1 = s.space.zeros()
        self._tmp_sn_incr = s.space.zeros()
        self._tmp_sn_weak_diff = s.space.zeros()
        self._tmp_cb12 = self.derham.Vh_pol["1"].zeros()
        self._tmp_cb1 = self.derham.Vh_pol["1"].zeros()
        self._linear_form_tot_e = s.space.zeros()
        self._linear_form_en1 = s.space.zeros()
        self.tot_rhs = s.space.zeros()
        if True:  # self._linearize_current:
            self._extracted_b2 = self.derham.boundary_ops["2"].dot(
                self.derham.extraction_ops["2"].dot(self.projected_equil.b2),
            )

    def __call__(self, dt):
        if self._nonlin_solver.type == "Newton":
            self.__call_newton(dt)
        else:
            raise ValueError(
                "wrong value for solver type in VariationalResistivity",
            )

    def __call_newton(self, dt):
        """Solve the non linear system for updating the variables using Newton iteration method"""
        # Compute dissipation implicitely
        sn = self.variables.s.spline.vector
        bn = self.variables.b.spline.vector

        if self._eta < 1.0e-15 and self._eta_a < 1.0e-15:
            self.update_feec_variables(s=sn, b=bn)
            return

        if self._info:
            print()
            print("Computing the dissipation in VariationalResistivity")

        total_resistivity = self._update_artificial_resistivity(bn, dt)

        self._scaled_stiffness._scalar = dt * self._eta
        # self.evol_op._multiplicants[1]._addends[0]._scalar = -dt*self._eta/2.
        if self._linearize_current:
            bn1 = self.evol_op.dot(
                bn
                + dt
                * self._eta
                * self.curl.dot(
                    self.Tcurl.dot(self._extracted_b2),
                ),
                out=self._tmp_bn1,
            )
        else:
            bn1 = self.evol_op.dot(bn, out=self._tmp_bn1)
        if self._info:
            print("information on the linear solver : ", self.inv_lop._info)

        if self._model == "linear_p" or (self._model == "linear_q" and self._nonlin_solver["fast"]):
            self.update_feec_variables(s=sn, b=bn1)
            return

        # Energy balance term
        # 1) Pointwize energy change
        energy_change = self._get_energy_change(bn, bn1, total_resistivity)
        # 2) Initial energy and linear form
        rho = self._rho.spline.vector
        self.rhof.vector = rho
        if self._model in ["deltaf_q", "linear_q"]:
            self.sf.vector = self._pt3.spline.vector
        else:
            self.sf.vector = sn

        sf_values = self.sf.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_bd,
            out=self._sf_values,
        )

        if self._model == "full":
            rhof_values = self.rhof.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._rhof_values,
            )

            e_n = self._energy_evaluator.ener(
                rhof_values,
                sf_values,
                out=self._e_n,
            )

            e_n *= self._energy_metric

        elif self._model in ["full_p", "linear_p", "delta_p"]:
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n *= 1.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        elif self._model in ["full_q"]:
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n **= 2
            e_n *= 1.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        elif self._model in ["linear_q", "deltaf_q"]:
            e_n = self._e_n
            e_n *= 0.0
            e_n += sf_values
            e_n *= self._q0_values
            e_n *= 2.0 / (self._gamma - 1.0)
            e_n *= self._energy_metric

        energy_change += e_n

        self._get_L2dofs_V3(energy_change, dofs=self._linear_form_tot_e)

        # 3) Newton iteration
        sn1 = sn.copy(out=self._tmp_sn1)

        tol = self._nonlin_solver["tol"]
        err = tol + 1

        for it in range(self._nonlin_solver["maxiter"]):
            if self._model in ["deltaf_q", "linear_q"]:
                self.sf1.vector = self._pt3.spline.vector
            else:
                self.sf1.vector = sn1

            sf1_values = self.sf1.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_bd,
                out=self._sf1_values,
            )

            if self._model == "full":
                e_n1 = self._energy_evaluator.ener(
                    rhof_values,
                    sf1_values,
                    out=self._e_n1,
                )
                e_n1 *= self._energy_metric

            elif self._model in ["full_p", "linear_p", "delta_p"]:
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 *= 1.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            elif self._model in ["full_q"]:
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 **= 2
                e_n1 *= 1.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            elif self._model in ["linear_q", "deltaf_q"]:
                e_n1 = self._e_n1
                e_n1 *= 0.0
                e_n1 += sf1_values
                e_n1 *= self._q0_values
                e_n1 *= 2.0 / (self._gamma - 1.0)
                e_n1 *= self._energy_metric

            self._get_L2dofs_V3(e_n1, dofs=self._linear_form_en1)

            self.tot_rhs *= 0.0
            self.tot_rhs -= self._linear_form_en1
            self.tot_rhs += self._linear_form_tot_e

            err = self._get_error_newton(self.tot_rhs)

            if self._info:
                print("iteration : ", it, " error : ", err)

            if (err < tol**2 and it > 0) or xp.isnan(err):
                break

            if self._model == "full":
                deds = self._energy_evaluator.dener_ds(
                    rhof_values,
                    sf1_values,
                    out=self._de_s1_values,
                )
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]], verbose=False)
                self.pc_jac.update_mass_operator(self.M_de_ds)

            elif self._model in ["full_q", "linear_q", "deltaf_q"]:
                if self._model in ["deltaf_q", "linear_q"]:
                    sf1_values = self._q0_values
                deds = self._de_s1_values
                deds *= 0.0
                deds += sf1_values
                deds *= 2 / (self._gamma - 1.0)
                deds *= self._mass_metric_term

                self.M_de_ds.assemble([[deds]], verbose=False)
                self.pc_jac.update_mass_operator(self.M_de_ds)

            incr = self.inv_jac.dot(self.tot_rhs, out=self._tmp_sn_incr)

            if self._info:
                print("information on the linear solver : ", self.inv_jac._info)

            if self._model in ["deltaf_q", "linear_q"]:
                self._pt3 += incr
            else:
                sn1 += incr

        if it == self._nonlin_solver["maxiter"] - 1 or xp.isnan(err):
            print(
                f"!!!Warning: Maximum iteration in VariationalResistivity reached - not converged:\n {err =} \n {tol**2 =}",
            )

        self.update_feec_variables(s=sn1, b=bn1)

        # if self._pt3 is not None:
        #     bn12 = bn.copy(out=self._tmp_bn12)
        #     bn12 += bn1
        #     bn12 /= 2.0
        #     cb1 = self.Tcurl.dot(bn1, out=self._tmp_cb1)
        #     cb12 = self.Tcurl.dot(bn12, out=self._tmp_cb12)

        #     self.cbf12.vector = cb12
        #     self.cbf1.vector = cb1

        #     cb12_v = self.cbf12.eval_tp_fixed_loc(
        #         self.integration_grid_spans,
        #         self.integration_grid_curl,
        #         out=self._cb12_values,
        #     )
        #     cb1_v = self.cbf1.eval_tp_fixed_loc(
        #         self.integration_grid_spans,
        #         self.integration_grid_curl,
        #         out=self._cb1_values,
        #     )

        #     cb_sq_v = self._cb_sq_values
        #     cb_sq_v *= 0.0
        #     for i in range(3):
        #         for j in range(3):
        #             cb_sq_v += cb12_v[i] * self._sq_term_metric[i, j] * cb1_v[j]

        #     cb_sq_v *= self._cb_sq_values_init
        #     # 2) Initial energy and linear form
        #     self.sf.vector = self._pt3

        #     sf_values = self.sf.eval_tp_fixed_loc(
        #         self.integration_grid_spans,
        #         self.integration_grid_bd,
        #         out=self._sf_values,
        #     )

        #     e_n = self._e_n
        #     e_n *= 0.0
        #     e_n += sf_values
        #     e_n *= 1.0 / (self._gamma - 1.0)
        #     e_n *= self._energy_metric

        #     cb_sq_v += e_n

        #     self._get_L2dofs_V3(cb_sq_v, dofs=self._linear_form_tot_e)

        #     tol = self._nonlin_solver["tol"]
        #     err = tol + 1

        #     for it in range(self._nonlin_solver["maxiter"]):
        #         self.sf1.vector = self._pt3

        #         sf1_values = self.sf1.eval_tp_fixed_loc(
        #             self.integration_grid_spans,
        #             self.integration_grid_bd,
        #             out=self._sf1_values,
        #         )

        #         e_n1 = self._e_n1
        #         e_n1 *= 0.0
        #         e_n1 += sf1_values
        #         e_n1 *= 1.0 / (self._gamma - 1.0)
        #         e_n1 *= self._energy_metric

        #         self._get_L2dofs_V3(e_n1, dofs=self._linear_form_en1)

        #         self.tot_rhs *= 0.0
        #         self.tot_rhs -= self._linear_form_en1
        #         self.tot_rhs += self._linear_form_tot_e

        #         err = self._get_error_newton(self.tot_rhs)

        #         if self._info:
        #             print("iteration : ", it, " error : ", err)

        #         if (err < tol**2 and it > 0) or xp.isnan(err):
        #             break

        #         incr = self.inv_jac.dot(self.tot_rhs, out=self._tmp_sn_incr)

        #         if self._info:
        #             print("information on the linear solver : ", self.inv_jac._info)

        #         self._pt3 += incr

    def _initialize_projectors_and_mass(self):
        """Initialization of all the `BasisProjectionOperator` and needed to compute the bracket term"""

        pc_M1 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M1,
        )
        inv_M1 = inverse(
            self.mass_ops.M1,
            "pcg",
            pc=pc_M1,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        pc_M3 = preconditioner.MassMatrixDiagonalPreconditioner(
            self.mass_ops.M3,
        )
        self._inv_M3 = inverse(
            self.mass_ops.M3,
            "pcg",
            pc=pc_M3,
            tol=1e-16,
            maxiter=1000,
            verbose=False,
        )

        M2 = self.mass_ops.M2
        self.M_de_ds = self.mass_ops.create_weighted_mass("L2", "L2")

        D = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.M1_cb = self.mass_ops.create_weighted_mass("Hcurl", "Hcurl", weights=[D, "sqrt_g"])

        if self.options.precond is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner,
                self.options.precond,
            )
            self.pc_jac = pc_class(self.M_de_ds)

        self.inv_jac = inverse(
            self.M_de_ds,
            "pcg",
            pc=self.pc_jac,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        self.curl = self.derham.curl
        self.Tcurl = inv_M1 @ self.curl.T @ M2

        self.phy_stiffness = M2 @ self.curl @ inv_M1 @ self.curl.T @ M2
        self.phy_cb_stiffness = self.Tcurl.T @ self.M1_cb @ self.Tcurl

        self._scaled_stiffness = 0.00001 * self.phy_stiffness

        self.r_op = M2  # - self._scaled_stiffness
        self.l_op = M2 + self._scaled_stiffness + self.phy_cb_stiffness

        if self.options.precond is None:
            self.pc = None
        else:
            pc_class = getattr(
                preconditioner,
                self.options.precond,
            )
            self.pc = pc_class(M2)

        self.inv_lop = inverse(
            self.l_op,
            "pcg",
            pc=self.pc,
            tol=self._lin_solver.tol,
            maxiter=self._lin_solver.maxiter,
            verbose=False,
            recycle=True,
        )

        self.evol_op = self.inv_lop @ self.r_op
        # self.evol_op = IdentityOperator(self.derham.Vh_pol['v'])
        integration_grid = [grid_1d.flatten() for grid_1d in self.derham.quad_grid_pts["3"]]
        self.integration_grid_spans, self.integration_grid_bn, self.integration_grid_bd = (
            self.derham.prepare_eval_tp_fixed(
                integration_grid,
            )
        )

        self.integration_grid_curl = [
            [self.integration_grid_bd[0], self.integration_grid_bn[1], self.integration_grid_bn[2]],
            [
                self.integration_grid_bn[0],
                self.integration_grid_bd[1],
                self.integration_grid_bn[2],
            ],
            [self.integration_grid_bn[0], self.integration_grid_bn[1], self.integration_grid_bd[2]],
        ]

        grid_shape = tuple([len(loc_grid) for loc_grid in integration_grid])

        self._cb12_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]
        self._cb1_values = [xp.zeros(grid_shape, dtype=float) for i in range(3)]

        self._cb_sq_values = xp.zeros(grid_shape, dtype=float)
        self._cb_sq_values_init = xp.zeros(grid_shape, dtype=float)

        self._sf_values = xp.zeros(grid_shape, dtype=float)
        self._sf1_values = xp.zeros(grid_shape, dtype=float)
        self._rhof_values = xp.zeros(grid_shape, dtype=float)

        self._e_n1 = xp.zeros(grid_shape, dtype=float)
        self._e_n = xp.zeros(grid_shape, dtype=float)

        self._de_s1_values = xp.zeros(grid_shape, dtype=float)

        self._tmp_int_grid = xp.zeros(grid_shape, dtype=float)

        gam = self._gamma
        if self._model == "full":
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -gam,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                1 - gam,
            )
            self._energy_metric = deepcopy(metric)

        elif self._model in ["full_p", "linear_p", "delta_p"]:
            metric = 1.0 / self.domain.jacobian_det(
                *integration_grid,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = (
                0
                * self.domain.jacobian_det(
                    *integration_grid,
                )
                + 1.0
            )
            self._energy_metric = deepcopy(metric)

            # No need to compute this every iteration
            deds = self._de_s1_values
            deds *= 0.0
            deds += 1 / (self._gamma - 1.0)
            deds *= self._mass_metric_term

            self.M_de_ds.assemble([[deds]], verbose=False)
            self.pc_jac.update_mass_operator(self.M_de_ds)

        elif self._model in ["full_q", "linear_q", "deltaf_q"]:
            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -2,
            )
            self._mass_metric_term = deepcopy(metric)

            metric = xp.power(
                self.domain.jacobian_det(
                    *integration_grid,
                ),
                -1,
            )
            self._energy_metric = deepcopy(metric)

        if self._model in ["linear_q", "deltaf_q"]:
            self.sf1.vector = self.projected_equil.q3

            self._q0_values = self.sf1.eval_tp_fixed_loc(self.integration_grid_spans, self.integration_grid_bd)

        metric = self.domain.metric_inv(
            *integration_grid,
        ) * self.domain.jacobian_det(*integration_grid)
        self._sq_term_metric = deepcopy(metric)

        metric = self.domain.metric_inv(
            *integration_grid,
        )
        self._sq_term_metric_no_jac = deepcopy(metric)

        self._get_L2dofs_V3 = L2Projector("L2", self.mass_ops).get_dofs

    def _get_error_newton(self, sn_diff):
        err_s = self._inv_M3.dot_inner(sn_diff, sn_diff)
        return err_s

    def _update_artificial_resistivity(self, bn, dt):
        """Update the artificial resistivity as the norm of the gradient of un.
        Update the associated mass matrix and return the total resistivity for later computation"""
        if self._eta_a > 1e-15:
            cb = self.Tcurl.dot(bn, out=self._tmp_cb1)
            self.cbf1.vector = cb
            cb_v = self.cbf1.eval_tp_fixed_loc(
                self.integration_grid_spans,
                self.integration_grid_curl,
                out=self._cb1_values,
            )

            cb_sq_v = self._cb_sq_values_init
            cb_sq_v *= 0.0
            for i in range(3):
                for j in range(3):
                    cb_sq_v += cb_v[i] * self._sq_term_metric_no_jac[i, j] * cb_v[j]

            xp.sqrt(cb_sq_v, out=cb_sq_v)

            cb_sq_v *= dt * self._eta_a

            self.M1_cb.assemble(
                [
                    [
                        cb_sq_v * self._sq_term_metric[0, 0],
                        cb_sq_v * self._sq_term_metric[0, 1],
                        cb_sq_v * self._sq_term_metric[0, 2],
                    ],
                    [
                        cb_sq_v * self._sq_term_metric[1, 0],
                        cb_sq_v * self._sq_term_metric[1, 1],
                        cb_sq_v * self._sq_term_metric[1, 2],
                    ],
                    [
                        cb_sq_v * self._sq_term_metric[2, 0],
                        cb_sq_v * self._sq_term_metric[2, 1],
                        cb_sq_v * self._sq_term_metric[2, 2],
                    ],
                ],
                verbose=False,
            )

            cb_sq_v += dt * self._eta

        else:
            cb_sq_v = self._cb_sq_values_init
            cb_sq_v *= 0.0
            cb_sq_v += dt * self._eta

        return cb_sq_v

    def _get_energy_change(self, bn, bn1, total_resistivity):
        """Return the total energy change caused by the resistivity"""
        bn12 = bn.copy(out=self._tmp_bn12)
        bn12 += bn1
        bn12 /= 2.0
        if self._linearize_current:
            cb1 = self.Tcurl.dot(
                bn1 - self._extracted_b2,
                out=self._tmp_cb1,
            )
        else:
            cb1 = self.Tcurl.dot(bn1, out=self._tmp_cb1)

        if self._model in ["full", "full_p", "full_q"]:
            cb12 = self.Tcurl.dot(bn12, out=self._tmp_cb12)

        # elif self._model in ["linear_p", "linear_q"]:
        #     cb12 = self.Tcurl.dot(self._extracted_b2, out=self._tmp_cb12)

        elif self._model in ["delta_p", "deltaf_q", "linear_p", "linear_q"]:
            # bn12 += self._extracted_b2
            cb12 = self.Tcurl.dot(bn12, out=self._tmp_cb12)

        self.cbf12.vector = cb12
        self.cbf1.vector = cb1

        cb12_v = self.cbf12.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_curl,
            out=self._cb12_values,
        )
        cb1_v = self.cbf1.eval_tp_fixed_loc(
            self.integration_grid_spans,
            self.integration_grid_curl,
            out=self._cb1_values,
        )

        cb_sq_v = self._cb_sq_values
        cb_sq_v *= 0.0
        for i in range(3):
            for j in range(3):
                cb_sq_v += cb12_v[i] * self._sq_term_metric[i, j] * cb1_v[j]

        cb_sq_v *= total_resistivity

        return cb_sq_v


class TimeDependentSource(Propagator):
    r"""Propagates a source term :math:`S(t) \in V_h^n` of the form

    .. math::

        S(t) = \sum_{ijk} c_{ijk} \Lambda^n_{ijk} * h(\omega t)\,,

    where :math:`h(\omega t)` is one of the functions in Notes.

    Notes
    -----

    * :math:`h(\omega t) = \cos(\omega t)` (default)
    * :math:`h(\omega t) = \sin(\omega t)`
    """

    class Variables:
        def __init__(self):
            self._source: FEECVariable = None

        @property
        def source(self) -> FEECVariable:
            return self._source

        @source.setter
        def source(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._source = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsTimeSource = Literal["cos", "sin"]
        # propagator options
        omega: float = 2.0 * xp.pi
        hfun: OptsTimeSource = "cos"

        def __post_init__(self):
            # checks
            check_option(self.hfun, self.OptsTimeSource)

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        if self.options.hfun == "cos":

            def hfun(t):
                return xp.cos(self.options.omega * t)
        elif self.options.hfun == "sin":

            def hfun(t):
                return xp.sin(self.options.omega * t)
        else:
            raise NotImplementedError(f"{self.options.hfun =} not implemented.")

        self._hfun = hfun
        self._c0 = self.variables.source.spline.vector.copy()

    @profile
    def __call__(self, dt):
        # new coeffs
        cn1 = self._c0 * self._hfun(self.time_state[0])

        # write new coeffs into self.feec_vars
        # max_dc = self.feec_vars_update(cn1)
        self.update_feec_variables(source=cn1)


class AdiabaticPhi(Propagator):
    r"""
    Electrostatic potential for adiabatic electrons, computed from

    .. math::

        n_e = n_{e0}\,\exp \left( \frac{e \phi}{k_B T_e} \right) \approx n_{e0} \left( 1 + \frac{e \phi}{k_B T_{e0}} \right)\,,

    where :math:`n_{e0}` and :math:`T_{e0}` denote electron equilibrium density and temperature, respectively.

    This is solved in weak form: find :math:`\phi \in H^1` such that

    .. math::

        \int_\Omega \psi\, \frac{n_{e0}(\mathbf x)}{T_{e0}(\mathbf x)}\phi \,\textrm d \mathbf x  = \int_\Omega \psi\, (n_{e}(\mathbf x) - n_{e0}(\mathbf x))\,\textrm d \mathbf x \qquad \forall \ \psi \in H^1\,.

    The equation is discretized as

    .. math::

        \sigma_1 \mathbb M^0_{n/T} \boldsymbol \phi = (\Lambda^0, n_{e} - n_{e0} )_{L^2}\,,

    where :math:`M^0_{n/T}` is a :class:`~struphy.feec.mass.WeightedMassOperator` and :math:`\sigma_1`
    is a normalization parameter.

    Parameters
    ----------
    phi : StencilVector
        FE coefficients of the solution as a discrete 0-form.

    A_mat : WeightedMassOperator
        The matrix to invert.

    rho : StencilVector or tuple
        Right-hand side FE coefficients of a 0-form (optional, can be set with a setter later).
        Can be either a) StencilVector or b) 2-tuple.
        In case b) the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.

    sigma_1 : float
        Normalization parameter.

    x0 : StencilVector
        Initial guess for the iterative solver (optional, can be set with a setter later).

    **params : dict
        Parameters for the iterative solver (see ``__init__`` for details).
    """

    def __init__(
        self,
        phi: StencilVector,
        *,
        A_mat: WeightedMassOperator = "M0",
        rho: StencilVector | tuple = None,
        sigma_1: float = 1.0,
        x0: StencilVector = None,
        **params,
    ):
        assert phi.space == self.derham.Vh["0"]

        super().__init__(phi)

        # solver parameters
        params_default = {
            "type": ("pcg", "MassMatrixPreconditioner"),
            "tol": 1e-8,
            "maxiter": 3000,
            "info": False,
            "verbose": False,
            "recycle": False,
        }

        params = set_defaults(params, params_default)

        # collect rhs
        if rho is None:
            rho = phi.space.zeros()
        else:
            if isinstance(rho, tuple):
                assert isinstance(rho[0], AccumulatorVector)
                assert isinstance(rho[1], Particles)
                # assert rho[0].space_id == 'H1'
            else:
                assert rho.space == phi.space
        self._rho = rho

        # initial guess and solver params
        self._x0 = x0
        self._params = params
        A_mat = getattr(self.mass_ops, A_mat)

        # Set lhs matrices
        self._A = sigma_1 * A_mat

        # preconditioner and solver for Ax=b
        if params["type"][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, params["type"][1])
            pc = pc_class(A_mat)

        # solver just with A_2, but will be set during call with dt
        self._solver = inverse(
            self._A,
            params["type"][0],
            pc=pc,
            x0=self.x0,
            tol=params["tol"],
            maxiter=params["maxiter"],
            verbose=params["verbose"],
            recycle=params["recycle"],
        )

        # allocate memory for solution
        self._tmp = phi.space.zeros()
        self._tmp2 = phi.space.zeros()
        self._rhs = phi.space.zeros()
        self._rhs2 = phi.space.zeros()

    @property
    def rho(self):
        """
        Right-hand side FE coefficients of a 0-form.
        Can be either a) StencilVector or b) 2-tuple.
        In the latter case, the first tuple entry must be :class:`~struphy.pic.accumulation.particles_to_grid.AccumulatorVector`,
        and the second entry must be :class:`~struphy.pic.base.Particles`.
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        """In-place setter for StencilVector/PolarVector."""
        if isinstance(value, tuple):
            assert isinstance(value[0], AccumulatorVector)
            assert isinstance(value[1], Particles)
            self._rho = value
        else:
            assert value.space == self.derham.Vh["0"]
            self._rho[:] = value[:]

    @property
    def x0(self):
        """
        psydac.linalg.stencil.StencilVector or struphy.polar.basic.PolarVector. First guess of the iterative solver.
        """
        return self._x0

    @x0.setter
    def x0(self, value):
        """In-place setter for StencilVector/PolarVector. First guess of the iterative solver."""
        assert value.space == self.derham.Vh["0"]
        assert value.space.symbolic_space == "H1", (
            f"Right-hand side must be in H1, but is in {value.space.symbolic_space}."
        )

        if self._x0 is None:
            self._x0 = value
        else:
            self._x0[:] = value[:]

    def __call__(self, dt):
        self._rhs *= 0.0
        if isinstance(self._rho, tuple):
            self._rho[0]()  # accumulate
            self._rhs += self._rho[0].vectors[0]
        else:
            self._rhs += self._rho

        # solve
        out = self._solver.solve(self._rhs, out=self._tmp)
        info = self._solver._info

        if self._lin_solver["info"]:
            print(info)

        dphi = self.feec_vars_update(out)

    @classmethod
    def options(cls):
        dct = {}
        dct["solver"] = {
            "type": [
                ("pcg", "MassMatrixPreconditioner"),
                ("cg", None),
            ],
            "tol": 1.0e-8,
            "maxiter": 3000,
            "info": False,
            "verbose": False,
            "recycle": True,
        }
        return dct


class HasegawaWakatani(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`(n, \omega) \in H^1 \times H^1` such that

    .. math::

        &\int_\Omega\frac{\partial n}{\partial t} m \,\textrm d \mathbf x = \int_\Omega C(x, y)(\phi - n) \, m \,\textrm d \mathbf x - \int_\Omega \phi [n, m] \,\textrm d \mathbf x - \kappa \int_\Omega  \partial_y \phi \,m \,\textrm d \mathbf x - \nu \int_\Omega \nabla n \cdot \nabla m \,\textrm d \mathbf x \qquad \forall m \in H^1\,,
        \\[2mm]
        &\int_\Omega\frac{\partial \omega}{\partial t} \psi \,\textrm d \mathbf x = \int_\Omega C(x, y)(\phi - n) \, \psi \,\textrm d \mathbf x - \int_\Omega \phi [\omega, \psi] \,\textrm d \mathbf x - \nu \int_\Omega \nabla \omega \cdot \nabla \psi \,\textrm d \mathbf x \qquad \forall \psi \in H^1\,,

    where  :math:`\phi \in H^1` is a given stream function,
    :math:`C = C(x, y)`, :math:`\kappa` and :math:`\nu` are constants and
    :math:`[a, b] = \partial_x a \partial_y b - \partial_y a \partial_x b`.

    :ref:`time_discret`: explicit Runge-Kutta, see :class:`~struphy.ode.solvers.ODEsolverFEEC`.

    Parameters
    ----------
    n0 : StencilVector
        The density.

    omega0 : StencilVector
        The stream function.

    phi : SplineFuncion
        The potential.

    c_fun : str
        Defines the function c(x,y) in front of (phi - n).

    kappa, nu : float
        Equation parameters.

    algo : str
        See :class:`~struphy.ode.utils.ButcherTableau` for available algorithms.

    M0_solver : dict
        Solver parameters for M0 inversion.
    """

    class Variables:
        def __init__(self):
            self._n: FEECVariable = None
            self._omega: FEECVariable = None

        @property
        def n(self) -> FEECVariable:
            return self._n

        @n.setter
        def n(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._n = new

        @property
        def omega(self) -> FEECVariable:
            return self._omega

        @omega.setter
        def omega(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "H1"
            self._omega = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsCfun = Literal["const"]
        # propagator options
        phi: FEECVariable = None
        c_fun: OptsCfun = "const"
        kappa: float = 1.0
        nu: float = 0.01
        butcher: ButcherTableau = None
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.c_fun, self.OptsCfun)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)

            # defaults
            if self.solver_params is None:
                self.solver_params = SolverParameters()

            if self.butcher is None:
                self.butcher = ButcherTableau()

    @property
    def options(self) -> Options:
        if not hasattr(self, "_options"):
            self._options = self.Options()
        return self._options

    @options.setter
    def options(self, new):
        assert isinstance(new, self.Options)
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        # default phi
        if self.options.phi is None:
            self.options.phi = FEECVariable(space="H1")
            self.options.phi.allocate(derham=self.derham, domain=self.domain)

        self._phi = self.options.phi.spline
        self._phi.vector[:] = 1.0
        self._phi.vector.update_ghost_regions()

        # default c-function
        if self.options.c_fun == "const":
            c_fun = lambda e1, e2, e3: 0.0 + 0.0 * e1
        else:
            raise NotImplementedError(f"{self.options.c_fun =} is not available.")

        # expose equation parameters
        self._kappa = self.options.kappa
        self._nu = self.options.nu

        # get quadrature grid of V0
        pts = [grid.flatten() for grid in self.derham.quad_grid_pts["0"]]
        mesh_pts = xp.meshgrid(*pts, indexing="ij")

        # evaluate c(x, y) and metric coeff at local quadrature grid and multiply
        self._weights = c_fun(*mesh_pts)
        self._weights *= self.domain.jacobian_det(*mesh_pts)

        # evaluate phi at local quadrature grid
        self._spans, self._bns, self._bnd = self.derham.prepare_eval_tp_fixed(pts)
        self._phi_at_pts = self._phi.eval_tp_fixed_loc(self._spans, self._bns)

        # Jacobain at quad grid
        self._jac_det = self.domain.jacobian_det(*mesh_pts)
        self._jac_inv = self.domain.jacobian_inv(*mesh_pts, change_out_order=True)
        self._jac_invT = self.domain.jacobian_inv(*mesh_pts, change_out_order=True, transposed=True)

        # grad operator
        grad = self.derham.grad

        # mass operators
        M0 = self.mass_ops.M0
        M1 = self.mass_ops.M1
        M0c = self.mass_ops.create_weighted_mass(
            "H1",
            "H1",
            name="M0c",
            weights=[[self._weights]],
            assemble=True,
        )

        self._M1hw_weights = []
        for m in range(3):
            self._M1hw_weights += [[None, None, None]]

        self._phi_5d = xp.zeros((*self._phi_at_pts.shape, 3, 3), dtype=float)
        self._tmp_5d = xp.zeros((*self._phi_at_pts.shape, 3, 3), dtype=float)
        self._tmp_5dT = xp.zeros((3, 3, *self._phi_at_pts.shape), dtype=float)
        self._phi_5d[:, :, :, 0, 1] = self._phi_at_pts * self._jac_det
        self._phi_5d[:, :, :, 1, 0] = -self._phi_at_pts * self._jac_det
        self._tmp_5d[:] = self._jac_inv @ self._phi_5d @ self._jac_invT
        self._tmp_5dT[:] = xp.transpose(self._tmp_5d, axes=(3, 4, 0, 1, 2))

        self._M1hw_weights[0][1] = self._tmp_5dT[0, 1, :, :, :]
        self._M1hw_weights[1][0] = self._tmp_5dT[1, 0, :, :, :]

        # self._self._M1hw_weights = ["DFinv", [self._phi_5d], "DFinvT"]
        self._M1hw = self.mass_ops.create_weighted_mass(
            "Hcurl",
            "Hcurl",
            name="M1hw",
            weights=self._M1hw_weights,
            assemble=True,
        )

        # inverse M0 mass matrix
        solver = self.options.solver
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M0)
        # solver_params = deepcopy(M0_solver)  # need a copy to pop, otherwise testing fails
        # solver_params.pop("type")
        self._info = self.options.solver_params.info
        M0_inv = inverse(
            M0,
            solver,
            pc=pc,
            tol=self.options.solver_params.tol,
            maxiter=self.options.solver_params.maxiter,
            verbose=self.options.solver_params.verbose,
            recycle=self.options.solver_params.recycle,
        )

        # basis projection operator
        df_12 = lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3)[0, 1, :, :, :]
        df_22 = lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3)[1, 1, :, :, :]
        df_32 = lambda e1, e2, e3: self.domain.jacobian_inv(e1, e2, e3)[2, 1, :, :, :]
        fun = [[df_12, df_22, df_32]]
        # fun = [[None, lambda e1, e2, e3: 1.0 + 0.0 * e1, None]]
        self._BPO = self.basis_ops.create_basis_op(
            fun,
            "Hcurl",
            "H1",
            name="dy_phi",
            assemble=True,
        )
        # print(f"{self._BPO._dof_mat.blocks = }")

        # pre-allocated helper arrays
        n0 = self.variables.n.spline.vector
        omega0 = self.variables.omega.spline.vector

        self._tmp1 = n0.space.zeros()
        tmp2 = n0.space.zeros()
        self._tmp3 = n0.space.zeros()
        tmp4 = n0.space.zeros()
        tmp5 = n0.space.zeros()

        # rhs-callables for explicit ode solve
        terms1_n = -M0c + grad.T @ self._M1hw @ grad - self.options.nu * grad.T @ M1 @ grad
        terms1_phi = M0c
        terms1_phi_strong = -self.options.kappa * self._BPO @ grad

        terms2_omega = grad.T @ self._M1hw @ grad - self.options.nu * grad.T @ M1 @ grad
        terms2_n = -M0c
        terms2_phi = M0c

        out1 = n0.space.zeros()
        out2 = omega0.space.zeros()

        def f1(t, n, omega, out=out1):
            terms1_n.dot(n, out=self._tmp1)
            terms1_phi.dot(self._phi.vector, out=tmp2)
            self._tmp1 += tmp2
            M0_inv.dot(self._tmp1, out=out)
            terms1_phi_strong.dot(self._phi.vector, out=tmp2)
            out += tmp2
            out.update_ghost_regions()
            return out

        def f2(t, n, omega, out=out2):
            terms2_omega.dot(omega, out=self._tmp3)
            terms2_n.dot(n, out=tmp4)
            terms2_phi.dot(self._phi.vector, out=tmp5)
            self._tmp3 += tmp4
            self._tmp3 += tmp5
            M0_inv.dot(self._tmp3, out=out)
            out.update_ghost_regions()
            return out

        vector_field = {n0: f1, omega0: f2}
        self._ode_solver = ODEsolverFEEC(vector_field, butcher=self.options.butcher)

    def __call__(self, dt):
        # update time-dependent mass operator
        self._phi.eval_tp_fixed_loc(self._spans, self._bns, out=self._phi_at_pts)

        self._phi_5d[:, :, :, 0, 1] = self._phi_at_pts * self._jac_det
        self._phi_5d[:, :, :, 1, 0] = -self._phi_at_pts * self._jac_det
        self._tmp_5d[:] = self._jac_inv @ self._phi_5d @ self._jac_invT
        self._tmp_5dT[:] = xp.transpose(self._tmp_5d, axes=(3, 4, 0, 1, 2))

        self._M1hw_weights[0][1] = self._tmp_5dT[0, 1, :, :, :]
        self._M1hw_weights[1][0] = self._tmp_5dT[1, 0, :, :, :]

        self._M1hw.assemble(
            weights=self._M1hw_weights,
            verbose=False,
        )

        # solve with RK
        self._ode_solver(0.0, dt)


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

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None
            self._ue: FEECVariable = None
            self._phi: FEECVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._u = new

        @property
        def ue(self) -> FEECVariable:
            return self._ue

        @ue.setter
        def ue(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hdiv"
            self._ue = new

        @property
        def phi(self) -> FEECVariable:
            return self._phi

        @phi.setter
        def phi(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "L2"
            self._phi = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsDimension = Literal["1D", "2D", "Restelli", "Tokamak"]
        # propagator options
        nu: float = 1.0
        nu_e: float = 0.01
        eps_norm: float = 1.0
        solver: OptsGenSolver = "GMRES"
        solver_params: SolverParameters = None
        a: float = 1.0
        R0: float = 1.0
        B0: float = 10.0
        Bp: float = 12.0
        alpha: float = 0.1
        beta: float = 1.0
        stab_sigma: float = 1e-5
        variant: OptsSaddlePointSolver = "Uzawa"
        method_to_solve: OptsDirectSolver = "DirectNPInverse"
        preconditioner: bool = False
        spectralanalysis: bool = False
        lifting: bool = False
        dimension: OptsDimension = "2D"
        D1_dt: float = 1e-3

        def __post_init__(self):
            # checks
            check_option(self.solver, OptsGenSolver)
            check_option(self.variant, OptsSaddlePointSolver)
            check_option(self.method_to_solve, OptsDirectSolver)
            check_option(self.dimension, self.OptsDimension)

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
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\nNew options for propagator '{self.__class__.__name__}':")
            for k, v in new.__dict__.items():
                print(f"  {k}: {v}")
        self._options = new

    @profile
    def allocate(self):
        self._info = self.options.solver_params.info
        if self.derham.comm is not None:
            self._rank = self.derham.comm.Get_rank()
        else:
            self._rank = 0

        self._nu = self.options.nu
        self._nu_e = self.options.nu_e
        self._eps_norm = self.options.eps_norm
        self._a = self.options.a
        self._R0 = self.options.R0
        self._B0 = self.options.B0
        self._Bp = self.options.Bp
        self._alpha = self.options.alpha
        self._beta = self.options.beta
        self._stab_sigma = self.options.stab_sigma
        self._variant = self.options.variant
        self._method_to_solve = self.options.method_to_solve
        self._preconditioner = self.options.preconditioner
        self._dimension = self.options.dimension
        self._spectralanalysis = self.options.spectralanalysis
        self._lifting = self.options.lifting

        solver_params = self.options.solver_params

        # Lifting for nontrivial boundary conditions
        # derham had boundary conditions in eta1 direction, the following is in space Hdiv_0
        if self._lifting:
            self.derhamv0 = Derham(
                self.derham.Nel,
                self.derham.p,
                self.derham.spl_kind,
                domain=self.domain,
                dirichlet_bc=[[True, True], [False, False], [False, False]],
            )

            self._mass_opsv0 = WeightedMassOperators(
                self.derhamv0,
                self.domain,
                verbose=solver_params.verbose,
                eq_mhd=self.mass_ops.weights["eq_mhd"],
            )
            self._basis_opsv0 = BasisProjectionOperators(
                self.derhamv0,
                self.domain,
                verbose=solver_params.verbose,
                eq_mhd=self.basis_ops.weights["eq_mhd"],
            )
        else:
            self.derhamnumpy = Derham(
                self.derham.Nel,
                self.derham.p,
                self.derham.spl_kind,
                domain=self.domain,
                # dirichlet_bc=self.derham.dirichlet_bc,
                # nquads = self.derham._nquads,
                # nq_pr = self.derham._nq_pr,
                # comm = MPI.COMM_SELF, # self.derham._comm,
                # polar_ck= self.derham._polar_ck,
                # local_projectors=self.derham.with_local_projectors
            )

        # get forceterms for according dimension
        if self._dimension in ["2D", "1D"]:
            ### Manufactured solution ###
            _forceterm_logical = lambda e1, e2, e3: 0 * e1
            _funx = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Ions",
                comp="0",
                b0=self._B0,
                nu=self._nu,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
            )
            _funy = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Ions",
                comp="1",
                b0=self._B0,
                nu=self._nu,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
            )
            _funelectronsx = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Electrons",
                comp="0",
                b0=self._B0,
                nu_e=self._nu_e,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
            )
            _funelectronsy = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Electrons",
                comp="1",
                b0=self._B0,
                nu_e=self._nu_e,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
            )

            # get callable(s) for specified init type
            forceterm_class = [_funx, _funy, _forceterm_logical]
            forcetermelectrons_class = [_funelectronsx, _funelectronsy, _forceterm_logical]

            # pullback callable
            funx = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=0,
                domain=self.domain,
            )
            funy = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=1,
                domain=self.domain,
            )
            fun_electronsx = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=0,
                domain=self.domain,
            )
            fun_electronsy = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=1,
                domain=self.domain,
            )
            l2_proj = L2Projector(space_id="Hdiv", mass_ops=self.mass_ops)
            self._F1 = l2_proj([funx, funy, _forceterm_logical])
            self._F2 = l2_proj([fun_electronsx, fun_electronsy, _forceterm_logical])

        elif self._dimension == "Restelli":
            ### Restelli ###

            _forceterm_logical = lambda e1, e2, e3: 0 * e1
            _fun = getattr(callables, "RestelliForcingTerm")(
                B0=self._B0,
                nu=self._nu,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
                eps=self._eps_norm,
            )
            _funelectrons = getattr(callables, "RestelliForcingTerm")(
                B0=self._B0,
                nu=self._nu_e,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
                eps=self._eps_norm,
            )

            # get callable(s) for specified init type
            forceterm_class = [_forceterm_logical, _forceterm_logical, _fun]
            forcetermelectrons_class = [_forceterm_logical, _forceterm_logical, _funelectrons]

            # pullback callable
            fun_pb_1 = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=0,
                domain=self.domain,
            )
            fun_pb_2 = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=1,
                domain=self.domain,
            )
            fun_pb_3 = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=2,
                domain=self.domain,
            )
            fun_electrons_pb_1 = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=0,
                domain=self.domain,
            )
            fun_electrons_pb_2 = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=1,
                domain=self.domain,
            )
            fun_electrons_pb_3 = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=2,
                domain=self.domain,
            )
            if self._lifting:
                l2_proj = L2Projector(space_id="Hdiv", mass_ops=self._mass_opsv0)
            else:
                l2_proj = L2Projector(space_id="Hdiv", mass_ops=self.mass_ops)
            self._F1 = l2_proj([fun_pb_1, fun_pb_2, fun_pb_3], apply_bc=self._lifting)
            self._F2 = l2_proj([fun_electrons_pb_1, fun_electrons_pb_2, fun_electrons_pb_3], apply_bc=self._lifting)

            ### End Restelli ###

        elif self._dimension == "Tokamak":
            ### Tokamak geometry curl-free manufactured solution ###

            _forceterm_logical = lambda e1, e2, e3: 0 * e1
            _funx = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Ions",
                comp="0",
                b0=self._B0,
                nu=self._nu,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
            )
            _funy = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Ions",
                comp="1",
                b0=self._B0,
                nu=self._nu,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
            )
            _funz = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Ions",
                comp="2",
                b0=self._B0,
                nu=self._nu,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
            )
            _funelectronsx = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Electrons",
                comp="0",
                b0=self._B0,
                nu_e=self._nu_e,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
            )
            _funelectronsy = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Electrons",
                comp="1",
                b0=self._B0,
                nu_e=self._nu_e,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
            )
            _funelectronsz = getattr(callables, "ManufacturedSolutionForceterm")(
                species="Electrons",
                comp="2",
                b0=self._B0,
                nu_e=self._nu_e,
                dimension=self._dimension,
                stab_sigma=self._stab_sigma,
                eps=self._eps_norm,
                dt=self.options.D1_dt,
                a=self._a,
                Bp=self._Bp,
                alpha=self._alpha,
                beta=self._beta,
            )

            # get callable(s) for specified init type
            forceterm_class = [_funx, _funy, _funz]
            forcetermelectrons_class = [_funelectronsx, _funelectronsy, _funelectronsz]

            # pullback callable
            fun_pb_1 = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=0,
                domain=self.domain,
            )
            fun_pb_2 = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=1,
                domain=self.domain,
            )
            fun_pb_3 = TransformedPformComponent(
                forceterm_class,
                given_in_basis="physical",
                out_form="2",
                comp=2,
                domain=self.domain,
            )
            fun_electrons_pb_1 = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=0,
                domain=self.domain,
            )
            fun_electrons_pb_2 = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=1,
                domain=self.domain,
            )
            fun_electrons_pb_3 = TransformedPformComponent(
                forcetermelectrons_class,
                given_in_basis="physical",
                out_form="2",
                comp=2,
                domain=self.domain,
            )
            if self._lifting:
                l2_proj = L2Projector(space_id="Hdiv", mass_ops=self._mass_opsv0)
            else:
                l2_proj = L2Projector(space_id="Hdiv", mass_ops=self.mass_ops)
            self._F1 = l2_proj([fun_pb_1, fun_pb_2, fun_pb_3], apply_bc=self._lifting)
            self._F2 = l2_proj([fun_electrons_pb_1, fun_electrons_pb_2, fun_electrons_pb_3], apply_bc=self._lifting)

            ### End Tokamak geometry manufactured solution ###

        if self._variant == "GMRES":
            if self._lifting:
                self._M2 = getattr(self._mass_opsv0, "M2")
                self._M3 = getattr(self._mass_opsv0, "M3")
                self._M2B = -getattr(self._mass_opsv0, "M2B")
                self._div = self.derhamv0.div
                self._curl = self.derhamv0.curl
                self._S21 = self._basis_opsv0.S21
            else:
                self._M2 = getattr(self.mass_ops, "M2")
                self._M3 = getattr(self.mass_ops, "M3")
                self._M2B = -getattr(self.mass_ops, "M2B")
                self._div = self.derham.div
                self._curl = self.derham.curl
                self._S21 = self.basis_ops.S21

            # Define block matrix [[A BT], [B 0]] (without time step size dt in the diagonals)
            _A11 = (
                self._M2
                - self._M2B / self._eps_norm
                + self._nu
                * (self._div.T @ self._M3 @ self._div + self._S21.T @ self._curl.T @ self._M2 @ self._curl @ self._S21)
            )
            _A12 = None
            _A21 = _A12
            _A22 = (
                -self._stab_sigma * IdentityOperator(_A11.domain)
                + self._M2B / self._eps_norm
                + self._nu_e
                * (self._div.T @ self._M3 @ self._div + self._S21.T @ self._curl.T @ self._M2 @ self._curl @ self._S21)
            )
            _B1 = -self._M3 @ self._div
            _B2 = self._M3 @ self._div

            if _A12 is not None:
                assert _A11.codomain == _A12.codomain
            if _A21 is not None:
                assert _A22.codomain == _A21.codomain
            assert _B1.codomain == _B2.codomain
            if _A12 is not None:
                assert _A11.domain == _A12.domain == _B1.domain
            if _A21 is not None:
                assert _A21.domain == _A22.domain == _B2.domain
            assert _A22.domain == _B2.domain
            assert _A11.domain == _B1.domain

            self._block_domainA = BlockVectorSpace(_A11.domain, _A22.domain)
            self._block_codomainA = self._block_domainA
            self._block_domainB = self._block_domainA
            self._block_codomainB = _B2.codomain
            _blocksA = [[_A11, _A12], [_A21, _A22]]
            _A = BlockLinearOperator(self._block_domainA, self._block_codomainA, blocks=_blocksA)
            _blocksB = [[_B1, _B2]]
            _B = BlockLinearOperator(self._block_domainB, self._block_codomainB, blocks=_blocksB)
            _F = BlockVector(self._block_domainA, blocks=[self._F1, self._F2])  # missing M2/dt *un-1

        elif self._variant == "Uzawa":
            # Numpy
            if self._lifting:
                fun = []
                for m in range(3):
                    fun += [[]]
                    for n in range(3):
                        fun[-1] += [
                            lambda e1, e2, e3, m=m, n=n: self._basis_opsv0.G(e1, e2, e3)[:, :, :, m, n]
                            / self._basis_opsv0.sqrt_g(e1, e2, e3),
                        ]
                self._S21 = None
                if self.derhamv0.with_local_projectors:
                    self._S21 = BasisProjectionOperatorLocal(
                        self.derhamv0._Ploc["1"],
                        self.derhamv0.Vh_fem["2"],
                        fun,
                        transposed=False,
                    )

                if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
                    Vbc = self._mass_opsv0.M2._V_boundary_op.toarray_struphy()
                    Wbc = self._mass_opsv0.M2._W_boundary_op.toarray_struphy()
                    M2_mat = self._mass_opsv0.M2._mat.toarray()
                    self._M2np = Wbc @ M2_mat @ Vbc.T
                    Vbc = self._mass_opsv0.M3._V_boundary_op.toarray_struphy()
                    Wbc = self._mass_opsv0.M3._W_boundary_op.toarray_struphy()
                    M3_mat = self._mass_opsv0.M3._mat.toarray()
                    self._M3np = Wbc @ M3_mat @ Vbc.T
                    if isinstance(self.derhamv0.div, ComposedLinearOperator):
                        for mult in self.derhamv0.div.multiplicants:
                            if isinstance(mult, BlockLinearOperator):
                                if hasattr(self, "_Dnp"):
                                    self._Dnp = self._Dnp @ mult.toarray()
                                else:
                                    self._Dnp = mult.toarray()
                                # print(f"{type(mult.toarray())=}")   #with_pads = True
                            elif isinstance(mult, BoundaryOperator):
                                if hasattr(self, "_Dnp"):
                                    self._Dnp = self._Dnp @ mult.T.toarray_struphy()
                                else:
                                    self._Dnp = mult.toarray_struphy()
                    elif isinstance(self.derhamv0.div, BlockLinearOperator):
                        self._Dnp = self.derhamv0.div.toarray()
                    if isinstance(self.derhamv0.curl, ComposedLinearOperator):
                        for mult in self.derhamv0.curl.multiplicants:
                            if isinstance(mult, BlockLinearOperator):
                                if hasattr(self, "_Cnp"):
                                    self._Cnp = self._Cnp @ mult.toarray()
                                else:
                                    self._Cnp = mult.toarray()
                            elif isinstance(mult, BoundaryOperator):
                                if hasattr(self, "_Cnp"):
                                    self._Cnp = self._Cnp @ mult.T.toarray_struphy()
                                else:
                                    self._Cnp = mult.toarray_struphy()
                    elif isinstance(self.derhamv0.curl, BlockLinearOperator):
                        self._Dnp = self.derhamv0.curl.toarray()

                    if self._S21 is not None:
                        self._Hodgenp = self._S21.toarray
                    else:
                        self._Hodgenp = self._basis_opsv0.S21.toarray_struphy()  # self.basis_ops.S21.toarray
                    Vbc = self._mass_opsv0.M2B._V_boundary_op.toarray_struphy()
                    Wbc = self._mass_opsv0.M2B._W_boundary_op.toarray_struphy()
                    M2B_mat = -self._mass_opsv0.M2B._mat.toarray()  # - sign because of the definition of M2B
                    self._M2Bnp = Wbc @ M2B_mat @ Vbc.T
                elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
                    Vbc = self._mass_opsv0.M2._V_boundary_op.toarray_struphy(is_sparse=True)
                    Wbc = self._mass_opsv0.M2._W_boundary_op.toarray_struphy(is_sparse=True)
                    M2_mat = self._mass_opsv0.M2._mat.tosparse()
                    self._M2np = Wbc @ M2_mat @ Vbc.T
                    Vbc = self._mass_opsv0.M3._V_boundary_op.toarray_struphy(is_sparse=True)
                    Wbc = self._mass_opsv0.M3._W_boundary_op.toarray_struphy(is_sparse=True)
                    M3_mat = self._mass_opsv0.M3._mat.tosparse()
                    self._M3np = Wbc @ M3_mat @ Vbc.T
                    if self._S21 is not None:
                        self._Hodgenp = self._S21.tosparse
                    else:
                        self._Hodgenp = self._basis_opsv0.S21.toarray_struphy(is_sparse=True)
                    Vbc = self._mass_opsv0.M2B._V_boundary_op.toarray_struphy(is_sparse=True)
                    Wbc = self._mass_opsv0.M2B._W_boundary_op.toarray_struphy(is_sparse=True)
                    M2B_mat = self._mass_opsv0.M2B._mat.tosparse()
                    self._M2Bnp = -Wbc @ M2B_mat @ Vbc.T  # - sign because of the definition of M2B

                    if isinstance(self.derhamv0.div, ComposedLinearOperator):
                        for mult in self.derhamv0.div.multiplicants:
                            if isinstance(mult, BlockLinearOperator):
                                if hasattr(self, "_Dnp"):
                                    self._Dnp = self._Dnp @ mult.tosparse()
                                else:
                                    self._Dnp = mult.tosparse()
                            elif isinstance(mult, BoundaryOperator):
                                if hasattr(self, "_Dnp"):
                                    self._Dnp = self._Dnp @ mult.toarray_struphy(is_sparse=True)
                                else:
                                    self._Dnp = mult.toarray_struphy(is_sparse=True)
                    elif isinstance(self.derhamv0.div, BlockLinearOperator):
                        self._Dnp = self.derhamv0.div.tosparse()

                    if isinstance(self.derhamv0.curl, ComposedLinearOperator):
                        for mult in self.derhamv0.curl.multiplicants:
                            if isinstance(mult, BlockLinearOperator):
                                if hasattr(self, "_Cnp"):
                                    self._Cnp = self._Cnp @ mult.tosparse()
                                else:
                                    self._Cnp = mult.tosparse()
                            elif isinstance(mult, BoundaryOperator):
                                if hasattr(self, "_Cnp"):
                                    self._Cnp = self._Cnp @ mult.toarray_struphy(is_sparse=True)
                                else:
                                    self._Cnp = mult.toarray_struphy(is_sparse=True)
                    elif isinstance(self.derhamv0.curl, BlockLinearOperator):
                        self._Dnp = self.derhamv0.curl.tosparse()

            else:  # no lifting, use original Derham
                fun = []
                for m in range(3):
                    fun += [[]]
                    for n in range(3):
                        fun[-1] += [
                            lambda e1, e2, e3, m=m, n=n: self.basis_ops.G(e1, e2, e3)[:, :, :, m, n]
                            / self.basis_ops.sqrt_g(e1, e2, e3),
                        ]
                self._S21 = None
                if self.derham.with_local_projectors:
                    self._S21 = BasisProjectionOperatorLocal(
                        self.derham._Ploc["1"],
                        self.derham.Vh_fem["2"],
                        fun,
                        transposed=False,
                    )

                if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
                    Vbc = self.mass_ops.M2._V_boundary_op.toarray_struphy()
                    Wbc = self.mass_ops.M2._W_boundary_op.toarray_struphy()
                    M2_mat = self.mass_ops.M2._mat.toarray()
                    self._M2np = Wbc @ M2_mat @ Vbc.T
                    Vbc = self.mass_ops.M3._V_boundary_op.toarray_struphy()
                    Wbc = self.mass_ops.M3._W_boundary_op.toarray_struphy()
                    M3_mat = self.mass_ops.M3._mat.toarray()
                    self._M3np = Wbc @ M3_mat @ Vbc.T
                    self._Dnp = self.derhamnumpy.div.toarray()
                    self._Cnp = self.derhamnumpy.curl.toarray()

                    if self._S21 is not None:
                        self._Hodgenp = self._S21.toarray
                    else:
                        self._Hodgenp = self.basis_ops.S21.toarray_struphy()
                    Vbc = self.mass_ops.M2B._V_boundary_op.toarray_struphy()
                    Wbc = self.mass_ops.M2B._W_boundary_op.toarray_struphy()
                    M2B_mat = -self.mass_ops.M2B._mat.toarray()
                    self._M2Bnp = Wbc @ M2B_mat @ Vbc.T
                elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
                    self._M2np = self.mass_ops.M2.tosparse
                    self._M3np = self.mass_ops.M3.tosparse
                    if self._S21 is not None:
                        self._Hodgenp = self._S21.tosparse
                    else:
                        self._Hodgenp = self.basis_ops.S21.toarray_struphy(is_sparse=True)
                    self._M2Bnp = -self.mass_ops.M2B.tosparse

                    self._Dnp = self.derhamnumpy.div.tosparse()
                    self._Cnp = self.derhamnumpy.curl.tosparse()

            self._A11np_notimedependency = (
                self._nu
                * (
                    self._Dnp.T @ self._M3np @ self._Dnp
                    + 1.0 * self._Hodgenp.T @ self._Cnp.T @ self._M2np @ self._Cnp @ self._Hodgenp
                )
                - 1.0 * self._M2Bnp / self._eps_norm
            )
            A11np = self._M2np + self._A11np_notimedependency

            if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
                A11np += self._stab_sigma * xp.identity(A11np.shape[0])
                self.A22np = (
                    self._stab_sigma * xp.identity(A11np.shape[0])
                    + self._nu_e
                    * (
                        self._Dnp.T @ self._M3np @ self._Dnp
                        + self._Hodgenp.T @ self._Cnp.T @ self._M2np @ self._Cnp @ self._Hodgenp
                    )
                    + self._M2Bnp / self._eps_norm
                )
                self._A22prenp = (
                    xp.identity(self.A22np.shape[0]) * self._stab_sigma
                )  # + self._nu_e * (self._Dnp.T @ self._M3np @ self._Dnp)
            elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
                A11np += self._stab_sigma * sc.sparse.eye(A11np.shape[0], format="csr")
                self.A22np = (
                    self._stab_sigma * sc.sparse.eye(A11np.shape[0], format="csr")
                    + self._nu_e
                    * (
                        self._Dnp.T @ self._M3np @ self._Dnp
                        + self._Hodgenp.T @ self._Cnp.T @ self._M2np @ self._Cnp @ self._Hodgenp
                    )
                    + self._M2Bnp / self._eps_norm
                )
                self._A22prenp = self._stab_sigma * sc.sparse.eye(self.A22np.shape[0], format="csr")

            B1np = -self._M3np @ self._Dnp
            B2np = self._M3np @ self._Dnp
            self._B1np = B1np
            self._B2np = B2np
            self._F1np = self._F1.toarray()
            self._F2np = self._F2.toarray()
            _Anp = [A11np, self.A22np]
            _Bnp = [B1np, B2np]
            _Fnp = [self._F1np, self._F2np]
            self._A11prenp_notimedependency = self._nu * (self._Dnp.T @ self._M3np @ self._Dnp)
            _A11prenp = self._M2np + self._A11prenp_notimedependency
            _Anppre = [_A11prenp, self._A22prenp]

        if self._variant == "GMRES":
            self._solver_GMRES = SaddlePointSolver(
                A=_A,
                B=_B,
                F=_F,
                solver_name=self.options.solver,
                tol=self.options.solver_params.tol,
                max_iter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
                pc=None,
            )
            # Allocate memory for call
            self._untemp = u.space.zeros()

        elif self._variant == "Uzawa":
            self._solver_UzawaNumpy = SaddlePointSolver(
                Apre=_Anppre,
                A=_Anp,
                B=_Bnp,
                F=_Fnp,
                method_to_solve=self._method_to_solve,
                preconditioner=self._preconditioner,
                spectralanalysis=self.options.spectralanalysis,
                tol=self.options.solver_params.tol,
                max_iter=self.options.solver_params.maxiter,
                verbose=self.options.solver_params.verbose,
            )

    def __call__(self, dt):
        # current variables
        unfeec = self.variables.u.spline.vector
        uenfeec = self.variables.ue.spline.vector
        phinfeec = self.variables.phi.spline.vector

        if self._variant == "GMRES":
            if self._lifting:
                phinfeeccopy = self.derhamv0.create_spline_function("phi", space_id="L2")
                phinfeeccopy.vector = phinfeec
                # unfeec in space Hdiv, u0 in space Hdiv_0
                unfeeccopy = self.derhamv0.create_spline_function("u", space_id="Hdiv")
                u0 = self.derhamv0.create_spline_function("u", space_id="Hdiv")
                u_prime = self.derhamv0.create_spline_function("u", space_id="Hdiv")
                u0.vector = uenfeec
                unfeeccopy.vector = uenfeec
                apply_essential_bc_stencil(u0.vector[0], axis=0, ext=-1, order=0)
                apply_essential_bc_stencil(u0.vector[0], axis=0, ext=1, order=0)
                u_prime.vector = unfeeccopy.vector - u0.vector

                uenfeeccopy = self.derhamv0.create_spline_function("ue", space_id="Hdiv")
                ue0 = self.derhamv0.create_spline_function("ue", space_id="Hdiv")
                ue_prime = self.derhamv0.create_spline_function("ue", space_id="Hdiv")
                ue0.vector = uenfeec
                uenfeeccopy.vector = uenfeec
                apply_essential_bc_stencil(ue0.vector[0], axis=0, ext=-1, order=0)
                apply_essential_bc_stencil(ue0.vector[0], axis=0, ext=1, order=0)
                ue_prime.vector = uenfeeccopy.vector - ue0.vector

            _A11 = (
                self._M2 / dt
                - self._M2B / self._eps_norm
                + self._nu
                * (self._div.T @ self._M3 @ self._div + self._S21.T @ self._curl.T @ self._M2 @ self._curl @ self._S21)
            )
            _A12 = None
            _A21 = _A12
            _A22 = (
                self._nu_e
                * (self._div.T @ self._M3 @ self._div + self._S21.T @ self._curl.T @ self._M2 @ self._curl @ self._S21)
                + self._M2B / self._eps_norm
                - self._stab_sigma * IdentityOperator(_A11.domain)
            )

            if self._lifting:
                _A11prime = -self._M2B / self._eps_norm + self._nu * (
                    self.derhamv0.div.T @ self._M3 @ self.derhamv0.div
                    + self._basis_opsv0.S21.T
                    @ self.derhamv0.curl.T
                    @ self._M2
                    @ self.derhamv0.curl
                    @ self._basis_opsv0.S21
                )
                _A22prime = (
                    self._nu_e
                    * (
                        self.derhamv0.div.T @ self._M3 @ self.derhamv0.div
                        + self._basis_opsv0.S21.T
                        @ self.derhamv0.curl.T
                        @ self._M2
                        @ self.derhamv0.curl
                        @ self._basis_opsv0.S21
                    )
                    + self._M2B / self._eps_norm
                    - self._stab_sigma * IdentityOperator(_A11.domain)
                )
            _B1 = -self._M3 @ self._div
            _B2 = self._M3 @ self._div

            if _A12 is not None:
                assert _A11.codomain == _A12.codomain
            if _A21 is not None:
                assert _A22.codomain == _A21.codomain
            assert _B1.codomain == _B2.codomain
            if _A12 is not None:
                assert _A11.domain == _A12.domain == _B1.domain
            if _A21 is not None:
                assert _A21.domain == _A22.domain == _B2.domain
            assert _A22.domain == _B2.domain
            assert _A11.domain == _B1.domain

            _blocksA = [[_A11, _A12], [_A21, _A22]]
            _A = BlockLinearOperator(self._block_domainA, self._block_codomainA, blocks=_blocksA)
            _blocksB = [[_B1, _B2]]
            _B = BlockLinearOperator(self._block_domainB, self._block_codomainB, blocks=_blocksB)
            if self._lifting:
                _blocksF = [
                    self._M2.dot(self._F1) + self._M2.dot(u0.vector) / dt - _A11prime.dot(u_prime.vector),
                    self._M2.dot(self._F2) - _A22prime.dot(ue_prime.vector),
                ]
            else:
                _blocksF = [
                    self._M2.dot(self._F1) + self._M2.dot(unfeec) / dt,
                    self._M2.dot(self._F2),
                ]
            _F = BlockVector(self._block_domainA, blocks=_blocksF)

            # Imported solver
            self._solver_GMRES.A = _A
            self._solver_GMRES.B = _B
            self._solver_GMRES.F = _F

            if self._lifting:
                (
                    _sol1,
                    _sol2,
                    info,
                ) = self._solver_GMRES(u0.vector, ue0.vector, phinfeec)
                un = _sol1[0] + u_prime.vector
                uen = _sol1[1] + ue_prime.vector
                phin = _sol2
            else:
                (
                    _sol1,
                    _sol2,
                    info,
                ) = self._solver_GMRES(unfeec, uenfeec, phinfeec)
                un = _sol1[0]
                uen = _sol1[1]
                phin = _sol2
            # write new coeffs into self.feec_vars
            max_du, max_due, max_dphi = self.update_feec_variables(u=un, ue=uen, phi=phin)

        elif self._variant == "Uzawa":
            # Numpy
            A11np = self._M2np / dt + self._A11np_notimedependency
            if self._method_to_solve in ("DirectNPInverse", "InexactNPInverse"):
                A11np += self._stab_sigma * xp.identity(A11np.shape[0])
                _A22prenp = self._A22prenp
                A22np = self.A22np
            elif self._method_to_solve in ("SparseSolver", "ScipySparse"):
                A11np += self._stab_sigma * sc.sparse.eye(A11np.shape[0], format="csr")
                _A22prenp = self._A22prenp
                A22np = self.A22np

            # _Anp[1] and _Anppre[1] remain unchanged
            _Anp = [A11np, A22np]
            if self._preconditioner:
                _A11prenp = self._M2np / dt  # + self._A11prenp_notimedependency
                _Anppre = [_A11prenp, _A22prenp]

            if self._lifting:
                # unfeec in space Hdiv, u0 in space Hdiv_0
                unfeeccopy = self.derhamv0.create_spline_function("u", space_id="Hdiv")
                u0 = self.derhamv0.create_spline_function("u", space_id="Hdiv")
                u_prime = self.derham.create_spline_function("u", space_id="Hdiv")
                u0.vector = unfeec
                unfeeccopy.vector = unfeec
                apply_essential_bc_stencil(u0.vector[0], axis=0, ext=-1, order=0)
                apply_essential_bc_stencil(u0.vector[0], axis=0, ext=1, order=0)
                u_prime.vector = unfeeccopy.vector - u0.vector

                uenfeeccopy = self.derhamv0.create_spline_function("ue", space_id="Hdiv")
                ue0 = self.derhamv0.create_spline_function("ue", space_id="Hdiv")
                ue_prime = self.derhamv0.create_spline_function("ue", space_id="Hdiv")
                ue0.vector = uenfeec
                uenfeeccopy.vector = uenfeec
                apply_essential_bc_stencil(ue0.vector[0], axis=0, ext=-1, order=0)
                apply_essential_bc_stencil(ue0.vector[0], axis=0, ext=1, order=0)
                ue_prime.vector = uenfeeccopy.vector - ue0.vector

                _F1np = (
                    self._M2np @ self._F1np
                    + 1.0 / dt * self._M2np.dot(u0.vector.toarray())
                    - self._A11np_notimedependency.dot(u_prime.vector.toarray())
                )
                _F2np = self._M2np @ self._F2np - self.A22np.dot(ue_prime.vector.toarray())
                _Fnp = [_F1np, _F2np]
            else:
                _F1np = self._M2np @ self._F1np + 1.0 / dt * self._M2np.dot(unfeec.toarray())
                _F2np = self._M2np @ self._F2np
                _Fnp = [_F1np, _F2np]

            if self.rank == 0:
                if self._preconditioner:
                    self._solver_UzawaNumpy.Apre = _Anppre
                self._solver_UzawaNumpy.A = _Anp
                self._solver_UzawaNumpy.F = _Fnp
                if self._lifting:
                    un, uen, phin, info, residual_norms, spectralresult = self._solver_UzawaNumpy(
                        u0.vector,
                        ue0.vector,
                        phinfeec,
                    )

                    un += u_prime.vector.toarray()
                    uen += ue_prime.vector.toarray()
                else:
                    un, uen, phin, info, residual_norms, spectralresult = self._solver_UzawaNumpy(
                        unfeec,
                        uenfeec,
                        phinfeec,
                    )

                dimlist = [[shp - 2 * pi for shp, pi in zip(unfeec[i][:].shape, self.derham.p)] for i in range(3)]
                dimphi = [shp - 2 * pi for shp, pi in zip(phinfeec[:].shape, self.derham.p)]
                u_temp = BlockVector(self.derham.Vh["2"])
                ue_temp = BlockVector(self.derham.Vh["2"])
                phi_temp = StencilVector(self.derham.Vh["3"])
                test = 0
                for i, bl in enumerate(u_temp.blocks):
                    s = bl.starts
                    e = bl.ends
                    totaldim = dimlist[i][0] * dimlist[i][1] * dimlist[i][2]
                    test += totaldim
                    bl[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1] = un[
                        i * totaldim : (i + 1) * totaldim
                    ].reshape(*dimlist[i])

                for i, bl in enumerate(ue_temp.blocks):
                    s = bl.starts
                    e = bl.ends
                    totaldim = dimlist[i][0] * dimlist[i][1] * dimlist[i][2]
                    bl[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1] = uen[
                        i * totaldim : (i + 1) * totaldim
                    ].reshape(*dimlist[i])

                s = phi_temp.starts
                e = phi_temp.ends
                phi_temp[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1] = phin.reshape(*dimphi)
            else:
                print(f"TwoFluidQuasiNeutralFull is only running on one MPI.")

            # write new coeffs into self.feec_vars
            max_du, max_due, max_dphi = self.update_feec_variables(u=u_temp, ue=ue_temp, phi=phi_temp)

        if self._info and self._rank == 0:
            print("Status     for TwoFluidQuasiNeutralFull:", info["success"])
            print("Iterations for TwoFluidQuasiNeutralFull:", info["niter"])
            print("Maxdiff u for TwoFluidQuasiNeutralFull:", max_du)
            print("Maxdiff u_e for TwoFluidQuasiNeutralFull:", max_due)
            print("Maxdiff phi for TwoFluidQuasiNeutralFull:", max_dphi)
            print()
