"Particle and FEEC variables are updated."

from dataclasses import dataclass
from typing import Literal

import numpy as np
from line_profiler import profile
from mpi4py import MPI
from psydac.linalg.block import BlockVector
from psydac.linalg.solvers import inverse
from psydac.linalg.stencil import StencilVector

from struphy.feec import preconditioner
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.io.options import OptsGenSolver, OptsMassPrecond, OptsSymmSolver, OptsVecSpace, check_option
from struphy.io.setup import descend_options_dict
from struphy.kinetic_background.base import Maxwellian
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import DiscreteGradientSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic import utilities_kernels
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.particles import Particles5D, Particles6D
from struphy.pic.pushing import pusher_kernels, pusher_kernels_gc
from struphy.pic.pushing.pusher import Pusher
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator
from struphy.utils.arrays import xp
from struphy.utils.pyccel import Pyccelkernel


class VlasovAmpere(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf E \in H(\textnormal{curl})` and :math:`f` such that

    .. math::

        -& \int_\Omega \frac{\partial \mathbf E}{\partial t} \cdot \mathbf F\,\textrm d \mathbf x  =
        \frac{\alpha^2}{\varepsilon} \int_\Omega \int_{\mathbb{R}^3} f \mathbf{v} \cdot \mathbf F \, \text{d}^3 \mathbf{v} \,\textrm d \mathbf x \qquad \forall \, \mathbf F \in H(\textnormal{curl}) \,,
        \\[2mm]
        &\frac{\partial f}{\partial t} + \frac{1}{\varepsilon}\, \mathbf{E} 
            \cdot \frac{\partial f}{\partial \mathbf{v}} = 0 \,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`, such that

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right) \\
            \mathbf{V}^{n+1} - \mathbf{V}^n
        \end{bmatrix}
        =
        \frac{\Delta t}{2}
        \begin{bmatrix}
            0 & - \frac{\alpha^2}{\varepsilon} \mathbb L^1 \bar{DF^{-1}} \bar{\mathbf w} \\
            \frac{1}{\varepsilon} \bar{DF^{-\top}} \left(\mathbb L^1\right)^\top & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n \\
            \mathbf{V}^{n+1} + \mathbf{V}^n
        \end{bmatrix}

    based on the :class:`~struphy.linear_algebra.schur_solver.SchurSolver` with

    .. math::

        A = \mathbb M^1\,,\qquad B = \frac{\alpha^2}{2\varepsilon} \mathbb L^1 \bar{DF^{-1}} \bar{\mathbf w}\,,\qquad C = - \frac{1}{2\varepsilon} \bar{DF^{-\top}} \left(\mathbb L^1\right)^\top \,.

    The accumulation matrix and vector assembled in :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` are

    .. math::

        M = BC  \,,\qquad V = B \mathbf V \,.
    """

    class Variables:
        def __init__(self):
            self._e: FEECVariable = None
            self._ions: PICVariable = None

        @property
        def e(self) -> FEECVariable:
            return self._e

        @e.setter
        def e(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space == "Hcurl"
            self._e = new

        @property
        def ions(self) -> PICVariable:
            return self._ions

        @ions.setter
        def ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles6D"
            self._ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
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
        # scaling factors
        alpha = self.variables.ions.species.equation_params.alpha
        epsilon = self.variables.ions.species.equation_params.epsilon

        self._c1 = alpha**2 / epsilon
        self._c2 = 1.0 / epsilon

        self._info = self.options.solver_params.info

        # get accumulation kernel
        accum_kernel = Pyccelkernel(accum_kernels.vlasov_maxwell)

        # Initialize Accumulator object
        particles = self.variables.ions.particles

        self._accum = Accumulator(
            particles,
            "Hcurl",
            accum_kernel,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
        )

        # Create buffers to store temporarily e and its sum with old e
        self._e_tmp = self.derham.Vh["1"].zeros()
        self._e_scale = self.derham.Vh["1"].zeros()
        self._e_sum = self.derham.Vh["1"].zeros()

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(self.mass_ops.M1)

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1
        _BC = -self._accum.operators[0]

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

        # Instantiate particle pusher
        args_kernel = (
            self.derham.args_derham,
            self._e_sum.blocks[0]._data,
            self._e_sum.blocks[1]._data,
            self._e_sum.blocks[2]._data,
            self._c2,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_v_with_efield),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

    @profile
    def __call__(self, dt):
        # accumulate
        self._accum()

        # Update Schur solver
        self._schur_solver.BC = self._accum.operators[0]
        self._schur_solver.BC *= -self._c1 * self._c2 / 4.0

        # Vector for Schur solver
        self._e_scale *= 0.0
        self._e_scale += self._accum.vectors[0]
        self._e_scale *= self._c1 / 2.0

        # new e coeffs
        self._e_tmp, info = self._schur_solver(
            self.variables.e.spline.vector,
            self._e_scale,
            dt,
            out=self._e_tmp,
        )

        # mid-point e-field (no tmps created here)
        self._e_sum *= 0.0
        self._e_sum += self.variables.e.spline.vector
        self._e_sum += self._e_tmp
        self._e_sum *= 0.5

        # Update velocities
        self._pusher(dt)

        # update_weights
        if self.variables.ions.species.weights_params.control_variate:
            self.variables.ions.particles.update_weights()

        # write new coeffs into self.variables
        (max_de,) = self.update_feec_variables(e=self._e_tmp)

        # Print out max differences for weights and e-field
        if self._info:
            print("Status      for VlasovMaxwell:", info["success"])
            print("Iterations  for VlasovMaxwell:", info["niter"])
            print("Maxdiff e1  for VlasovMaxwell:", max_de)
            particles = self.variables.ions.particles
            buffer_idx = particles.bufferindex
            max_diff = xp.max(
                xp.abs(
                    xp.sqrt(
                        particles.markers_wo_holes[:, 3] ** 2
                        + particles.markers_wo_holes[:, 4] ** 2
                        + particles.markers_wo_holes[:, 5] ** 2,
                    )
                    - xp.sqrt(
                        particles.markers_wo_holes[:, buffer_idx + 3] ** 2
                        + particles.markers_wo_holes[:, buffer_idx + 4] ** 2
                        + particles.markers_wo_holes[:, buffer_idx + 5] ** 2,
                    ),
                ),
            )
            print("Maxdiff |v| for VlasovMaxwell:", max_diff)
            print()


class EfieldWeights(Propagator):
    r"""Solves the following substep

    .. math::

        \begin{align}
            & \frac{\partial \mathbf{E}}{\partial t} = - \frac{\alpha^2}{\varepsilon} \int \mathbf{v} f_1 \, \text{d} \mathbf{v} \,,
            \\[2mm]
            & \frac{\partial f_1}{\partial t} = \frac{1}{v_{\text{th}}^2 \varepsilon} \, \mathbf{E} \cdot \mathbf{v} f_0 \,,
        \end{align}

    which after discretization and in curvilinear coordinates reads

    .. math::

        \frac{\text{d}}{\text{d} t} w_p &= \frac{f_{0,p}}{s_{0, p}} \frac{1}{v_{\text{th}}^2 \varepsilon} \left[ DF^{-T} (\mathbb{\Lambda}^1)^T \mathbf{e} \right] \cdot \mathbf{v}_p \,,
        \\[2mm]
        \frac{\text{d}}{\text{d} t} \mathbb{M}_1 \mathbf{e} &= - \frac{\alpha^2}{\varepsilon} \frac 1N \sum_p w_p \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right) \,.

    This is solved using the Crank-Nicolson method

    .. math::

        \begin{bmatrix}
            \mathbb{M}_1 \left( \mathbf{e}^{n+1} - \mathbf{e}^n \right)
            \mathbf{W}^{n+1} - \mathbf{W}^n
        \end{bmatrix}
        =
        \frac{\Delta t}{2}
        \begin{bmatrix}
            0 & - \mathbb{E} \\
            \mathbb{W} & 0
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{e}^{n+1} + \mathbf{e}^n \\
            \mathbf{V}^{n+1} + \mathbf{V}^n
        \end{bmatrix} \,,

    where

    .. math::

        \mathbb{E} &= \frac{\alpha^2}{\varepsilon} \frac 1N \mathbb{\Lambda}^1 \cdot \left( DF^{-1} \mathbf{v}_p \right)  \,,
        \\[2mm]
        \mathbb{W} &= \frac{f_{0,p}}{s_{0,p}} \frac{1}{v_\text{th}^2 \varepsilon} \left( DF^{-1} \mathbf{v}_p \right) \cdot \left(\mathbb{\Lambda}^1\right)^T  \,,

    based on the :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.

    The accumulation matrix and vector assembled in :class:`~struphy.pic.accumulation.particles_to_grid.Accumulator` are

    .. math::

        BC = \mathbb{E} \mathbb{W} \, , \qquad By_n = \mathbb{E} \mathbf{W} \,.

    """

    @staticmethod
    def options(default=False):
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
        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(
        self,
        e: BlockVector,
        particles: Particles6D,
        *,
        alpha: float = 1.0,
        kappa: float = 1.0,
        f0: Maxwellian = None,
        solver=options(default=True)["solver"],
    ):
        super().__init__(e, particles)

        if f0 is None:
            f0 = Maxwellian3D()
        assert isinstance(f0, Maxwellian3D)

        self._alpha = alpha
        self._kappa = kappa
        self._f0 = f0
        assert self._f0.maxw_params["vth1"] == self._f0.maxw_params["vth2"] == self._f0.maxw_params["vth3"]
        self._vth = self._f0.maxw_params["vth1"]

        self._info = solver["info"]

        # Initialize Accumulator object
        self._accum = Accumulator(
            particles,
            "Hcurl",
            Pyccelkernel(accum_kernels.linear_vlasov_ampere),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
        )

        # Create buffers to store temporarily e and its sum with old e
        self._e_tmp = e.space.zeros()
        self._e_scale = e.space.zeros()
        self._e_sum = e.space.zeros()

        # marker storage
        self._f0_values = xp.zeros(particles.markers.shape[0], dtype=float)
        self._old_weights = xp.empty(particles.markers.shape[0], dtype=float)

        # ================================
        # ========= Schur Solver =========
        # ================================

        # Preconditioner
        if solver["type"][1] == None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver["type"][1])
            pc = pc_class(self.mass_ops.M1)

        # Define block matrix [[A B], [C I]] (without time step size dt in the diagonals)
        _A = self.mass_ops.M1
        _BC = self._alpha**2 * self._kappa**2 * self._accum.operators[0] / (4 * self._vth**2)

        # Instantiate Schur solver
        self._schur_solver = SchurSolver(
            _A,
            _BC,
            solver["type"][0],
            pc=pc,
            tol=solver["tol"],
            maxiter=solver["maxiter"],
            verbose=solver["verbose"],
        )

        # Instantiate particle pusher
        args_kernel = (
            self.derham.args_derham,
            self._e_sum.blocks[0]._data,
            self._e_sum.blocks[1]._data,
            self._e_sum.blocks[2]._data,
            self._f0_values,
            self._kappa,
            self._vth,
        )

        self._pusher = Pusher(
            particles,
            Pyccelkernel(pusher_kernels.push_weights_with_efield_lin_va),
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

    def __call__(self, dt):
        # evaluate f0 and accumulate
        self._f0_values[:] = self._f0(
            self.particles[0].markers[:, 0],
            self.particles[0].markers[:, 1],
            self.particles[0].markers[:, 2],
            self.particles[0].markers[:, 3],
            self.particles[0].markers[:, 4],
            self.particles[0].markers[:, 5],
        )

        self._accum(self._f0_values)

        # Update Schur solver
        self._schur_solver.BC = self._accum.operators[0]
        self._schur_solver.BC *= (-1) * self._alpha**2 * self._kappa**2 / (4 * self._vth**2)

        # Vector for schur solver
        self._e_scale *= 0.0
        self._e_scale += self._accum.vectors[0]
        self._e_scale *= self._alpha**2 * self._kappa / 2.0

        # new e-field (no tmps created here)
        self._e_tmp, info = self._schur_solver(
            xn=self.feec_vars[0],
            Byn=self._e_scale,
            dt=dt,
            out=self._e_tmp,
        )

        # Store old weights
        self._old_weights[~self.particles[0].holes] = self.particles[0].markers_wo_holes[:, 6]

        # Compute (e^{n+1} + e^n) (no tmps created here)
        self._e_sum *= 0.0
        self._e_sum += self.feec_vars[0]
        self._e_sum += self._e_tmp

        # Update weights
        self._pusher(dt)

        # write new coeffs into self.variables
        (max_de,) = self.feec_vars_update(self._e_tmp)

        # Print out max differences for weights and e-field
        if self._info:
            print("Status          for StepEfieldWeights:", info["success"])
            print("Iterations      for StepEfieldWeights:", info["niter"])
            print("Maxdiff    e1   for StepEfieldWeights:", max_de)
            max_diff = xp.max(
                xp.abs(
                    self._old_weights[~self.particles[0].holes]
                    - self.particles[0].markers[~self.particles[0].holes, 6],
                ),
            )
            print("Maxdiff weights for StepEfieldWeights:", max_diff)
            print()


class PressureCoupling6D(Propagator):
    r"""
    :ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde{\mathbf{U}}  \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and :math:`f` such that

    .. math::

        \int_\Omega \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = - \frac{A_\textnormal{h}}{A_\textnormal{b}} \nabla \cdot \tilde{\mathbb{P}}_{\textnormal{h},\perp} \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\frac{\partial f_\textnormal{h}}{\partial t} - \left(\nabla \tilde{\mathbf U}_\perp \cdot \mathbf{v} \right) \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\[2mm]
        &\tilde{\mathbb{P}}_{\textnormal{h},\perp} = \int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}_\perp \mathbf{v}_\perp^\top \,\textnormal{d}^3 \mathbf v\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} u^{n+1} - u^n \\ V^{n+1} - V^n \end{bmatrix}
        = \frac{\Delta t}{2} \begin{bmatrix} 0 & (\mathbb M^n)^{-1} V^\top (\bar {\mathcal X})^\top \mathbb G^\top (\bar {\mathbf \Lambda}^1)^\top \bar {DF}^{-1} \\ - {DF}^{-\top} \bar {\mathbf \Lambda}^1 \mathbb G \bar {\mathcal X} V (\mathbb M^n)^{-1} & 0 \end{bmatrix}
        \begin{bmatrix} {\mathbb M^n}(u^{n+1} + u^n) \\ \bar W (V^{n+1} + V^{n} \end{bmatrix} \,.
    """

    @staticmethod
    def options(default=False):
        dct = {}
        dct["use_perp_model"] = [True, False]
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
        dct["filter"] = {
            "use_filter": None,
            "modes": (1),
            "repeat": 1,
            "alpha": 0.5,
        }
        dct["boundary_cut"] = {
            "e1": 0.0,
            "e2": 0.0,
            "e3": 0.0,
        }
        dct["turn_off"] = False

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(
        self,
        particles: Particles5D,
        u: BlockVector | PolarVector,
        *,
        use_perp_model: bool = options(default=True)["use_perp_model"],
        u_space: str,
        solver: dict = options(default=True)["solver"],
        coupling_params: dict,
        filter: dict = options(default=True)["filter"],
        boundary_cut: dict = options(default=True)["boundary_cut"],
    ):
        super().__init__(particles, u)

        self._G = self.derham.grad
        self._GT = self.derham.grad.transpose()

        self._info = solver["info"]
        if self.derham.comm is None:
            self._rank = 0
        else:
            self._rank = self.derham.comm.Get_rank()

        assert u_space in {"Hcurl", "Hdiv", "H1vec"}

        if u_space == "Hcurl":
            id_Mn = "M1n"
            id_X = "X1"
        elif u_space == "Hdiv":
            id_Mn = "M2n"
            id_X = "X2"
        elif u_space == "H1vec":
            id_Mn = "Mvn"
            id_X = "Xv"

        if u_space == "H1vec":
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[u_space],
            )

        # Preconditioner
        if solver["type"][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver["type"][1])
            pc = pc_class(getattr(self.mass_ops, id_Mn))

        # Call the accumulation and Pusher class
        if use_perp_model:
            accum_ker = Pyccelkernel(accum_kernels.pc_lin_mhd_6d)
            pusher_ker = Pyccelkernel(pusher_kernels.push_pc_GXu)
        else:
            accum_ker = Pyccelkernel(accum_kernels.pc_lin_mhd_6d_full)
            pusher_ker = Pyccelkernel(pusher_kernels.push_pc_GXu_full)

        self._coupling_mat = coupling_params["Ah"] / coupling_params["Ab"]
        self._coupling_vec = coupling_params["Ah"] / coupling_params["Ab"]
        self._scale_push = 1

        self._boundary_cut_e1 = boundary_cut["e1"]

        self._ACC = Accumulator(
            particles,
            "Hcurl",
            accum_ker,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="pressure",
            filter_params=filter,
        )

        self._tmp_g1 = self._G.codomain.zeros()
        self._tmp_g2 = self._G.codomain.zeros()
        self._tmp_g3 = self._G.codomain.zeros()

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._tmp_g1[0]._data,
            self._tmp_g1[1]._data,
            self._tmp_g1[2]._data,
            self._tmp_g2[0]._data,
            self._tmp_g2[1]._data,
            self._tmp_g2[2]._data,
            self._tmp_g3[0]._data,
            self._tmp_g3[1]._data,
            self._tmp_g3[2]._data,
            self._boundary_cut_e1,
        )

        self._pusher = Pusher(
            particles,
            pusher_ker,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        # Define operators
        self._A = getattr(self.mass_ops, id_Mn)
        self._X = getattr(self.basis_ops, id_X)
        self._XT = self._X.transpose()

        # Instantiate schur solver with dummy BC
        self._schur_solver = SchurSolver(
            self._A,
            self._XT @ self._X,
            solver["type"][0],
            pc=pc,
            tol=solver["tol"],
            maxiter=solver["maxiter"],
            verbose=solver["verbose"],
            recycle=solver["recycle"],
        )

        self.u_temp = u.space.zeros()
        self.u_temp2 = u.space.zeros()
        self._tmp = self._X.codomain.zeros()
        self._BV = u.space.zeros()

        self._MAT = [
            [self._ACC.operators[0], self._ACC.operators[1], self._ACC.operators[2]],
            [self._ACC.operators[1], self._ACC.operators[3], self._ACC.operators[4]],
            [self._ACC.operators[2], self._ACC.operators[4], self._ACC.operators[5]],
        ]

        self._GT_VEC = BlockVector(self.derham.Vh["v"])

    def __call__(self, dt):
        # current u
        un = self.feec_vars[0]
        un.update_ghost_regions()

        # acuumulate MAT and VEC
        self._ACC(self._coupling_mat, self._coupling_vec, self._boundary_cut_e1)

        # update GT_VEC
        for i in range(3):
            self._GT_VEC[i] = self._GT.dot(self._ACC.vectors[i])

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        self._schur_solver.BC = -1 / 4 * self._XT @ self.GT_MAT_G(self.derham, self._MAT) @ self._X

        self._BV = self._XT.dot(self._GT_VEC) * (-1 / 2)

        # update u (no tmps created here)
        un1, info = self._schur_solver(un, self._BV, dt, out=self.u_temp)

        _u = un.copy(out=self.u_temp2)
        _u += un1

        # calculate GXu
        Xu = self._X.dot(_u, out=self._tmp)

        GXu_1 = self._G.dot(Xu[0], out=self._tmp_g1)
        GXu_2 = self._G.dot(Xu[1], out=self._tmp_g2)
        GXu_3 = self._G.dot(Xu[2], out=self._tmp_g3)

        GXu_1.update_ghost_regions()
        GXu_2.update_ghost_regions()
        GXu_3.update_ghost_regions()

        # push particles
        self._pusher(dt)

        # write new coeffs into Propagator.variables
        (max_du,) = self.feec_vars_update(un1)

        # update weights in case of control variate
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

        if self._info and self._rank == 0:
            print("Status     for StepPressurecoupling:", info["success"])
            print("Iterations for StepPressurecoupling:", info["niter"])
            print("Maxdiff u1 for StepPressurecoupling:", max_du)
            print()

    class GT_MAT_G(LinOpWithTransp):
        r"""
        Class for defining LinearOperator corresponding to :math:`G^\top (\text{MAT}) G \in \mathbb{R}^{3N^0 \times 3N^0}`
        where :math:`\text{MAT} = V^\top (\bar {\mathbf \Lambda}^1)^\top \bar{DF}^{-1} \bar{W} \bar{DF}^{-\top} \bar{\mathbf \Lambda}^1 V \in \mathbb{R}^{3N^1 \times 3N^1}`.

        Parameters
        ----------
            derham : struphy.feec.psydac_derham.Derham
                Discrete de Rham sequence on the logical unit cube.

            MAT : List of StencilMatrices
                List with six of accumulated pressure terms
        """

        def __init__(self, derham, MAT, transposed=False):
            self._derham = derham
            self._G = derham.grad
            self._GT = derham.grad.transpose()

            self._domain = derham.Vh["v"]
            self._codomain = derham.Vh["v"]
            self._MAT = MAT

            self._vector = BlockVector(derham.Vh["v"])
            self._temp = BlockVector(derham.Vh["1"])

        @property
        def domain(self):
            return self._domain

        @property
        def codomain(self):
            return self._codomain

        @property
        def dtype(self):
            return self._derham.Vh["v"].dtype

        @property
        def tosparse(self):
            raise NotImplementedError()

        @property
        def toarray(self):
            raise NotImplementedError()

        @property
        def transposed(self):
            return self._transposed

        def transpose(self):
            return self.GT_MAT_G(self._derham, self._MAT, True)

        def dot(self, v, out=None):
            """dot product between GT_MAT_G and v.

            Parameters
            ----------
                v : StencilVector or BlockVector
                    Input FE coefficients from V.coeff_space.

            Returns
            -------
                A StencilVector or BlockVector from W.coeff_space."""

            assert v.space == self.domain

            v.update_ghost_regions()

            for i in range(3):
                for j in range(3):
                    self._temp += self._MAT[i][j].dot(self._G.dot(v[j]))

                self._vector[i] = self._GT.dot(self._temp)
                self._temp *= 0.0

            self._vector.update_ghost_regions()

            if out is not None:
                self._vector.copy(out=out)

            assert self._vector.space == self.codomain

            return self._vector


class CurrentCoupling6DCurrent(Propagator):
    r"""
    :ref:`FEEC <gempic>` discretization of the following equations:
    find :math:`\tilde{\mathbf{U}}  \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and :math:`f` such that

    .. math::

        \int_\Omega \rho_0 &\frac{\partial \tilde{\mathbf{U}}}{\partial t} \cdot \mathbf V \,\textrm d \mathbf x = - \frac{A_\textnormal{h}}{A_\textnormal{b}} \frac{1}{\varepsilon}  \int_\Omega n_\textnormal{h}\mathbf{u}_\textnormal{h} \times(\mathbf{B}_0+\tilde{\mathbf{B}}) \cdot \mathbf V \,\textrm d \mathbf x
        \qquad \forall \, \mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
        \\[2mm]
        &\frac{\partial f_\textnormal{h}}{\partial t} + \frac{1}{\varepsilon} \Big[(\mathbf{B}_0+\tilde{\mathbf{B}})\times\tilde{\mathbf{U}} \Big] \cdot \frac{\partial f_\textnormal{h}}{\partial \mathbf{v}} =0\,,
        \\[2mm]
        &n_\textnormal{h}\mathbf{u}_\textnormal{h}=\int_{\mathbb{R}^3}f_\textnormal{h}\mathbf{v}\,\textnormal{d}^3 \mathbf v\,.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`.
    """

    @staticmethod
    def options(default=False):
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
        dct["filter"] = {
            "use_filter": None,
            "modes": (1),
            "repeat": 1,
            "alpha": 0.5,
        }
        dct["boundary_cut"] = {
            "e1": 0.0,
            "e2": 0.0,
            "e3": 0.0,
        }
        dct["turn_off"] = False

        if default:
            dct = descend_options_dict(dct, [])

        return dct

    def __init__(
        self,
        particles: Particles6D,
        u: BlockVector,
        *,
        u_space: str,
        b_eq: BlockVector | PolarVector,
        b_tilde: BlockVector | PolarVector,
        Ab: int = 1,
        Ah: int = 1,
        epsilon: float = 1.0,
        solver: dict = options(default=True)["solver"],
        filter: dict = options(default=True)["filter"],
        boundary_cut: dict = options(default=True)["boundary_cut"],
    ):
        super().__init__(particles, u)

        if u_space == "H1vec":
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[u_space],
            )

        self._b_eq = b_eq
        self._b_tilde = b_tilde

        self._info = solver["info"]

        if self.derham.comm is None:
            self._rank = 0
        else:
            self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = Ah / Ab / epsilon**2
        self._coupling_vec = Ah / Ab / epsilon
        self._scale_push = 1.0 / epsilon

        self._boundary_cut_e1 = boundary_cut["e1"]

        # load accumulator
        self._accumulator = Accumulator(
            particles,
            u_space,
            Pyccelkernel(accum_kernels.cc_lin_mhd_6d_2),
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
            filter_params=filter,
        )

        # if self.particles[0].control_variate:

        #     # control variate method is only valid with Maxwellian distributions
        #     assert isinstance(self.particles[0].f0, Maxwellian)

        #     self._accumulator.init_control_variate(self.mass_ops)

        # evaluate and save nh0 (0-form) * uh0 (2-form if H1vec or vector if Hdiv) at quadrature points for control variate
        # quad_pts = [
        #     quad_grid[nquad].points.flatten()
        #     for quad_grid, nquad in zip(self.derham.get_quad_grids(self.derham.Vh_fem['0']), self.derham.nquads)
        # ]

        #     uh0_cart = self.particles[0].f0.u

        #     self._nuh0_at_quad = self.domain.pull(
        #         uh0_cart, *quad_pts, kind='v', squeeze_out=False, coordinates='logical')

        #     self._nuh0_at_quad[0] *= self.domain.pull(
        #         self.particles[0].f0.n, *quad_pts, kind='0', squeeze_out=False, coordinates='logical')
        #     self._nuh0_at_quad[1] *= self.domain.pull(
        #         self.particles[0].f0.n, *quad_pts, kind='0', squeeze_out=False, coordinates='logical')
        #     self._nuh0_at_quad[2] *= self.domain.pull(
        #         self.particles[0].f0.n, *quad_pts, kind='0', squeeze_out=False, coordinates='logical')

        #     # memory allocation for magnetic field at quadrature points
        #     self._b_quad1 = xp.zeros_like(self._nuh0_at_quad[0])
        #     self._b_quad2 = xp.zeros_like(self._nuh0_at_quad[0])
        #     self._b_quad3 = xp.zeros_like(self._nuh0_at_quad[0])

        #     # memory allocation for (self._b_quad x self._nuh0_at_quad) * self._coupling_vec
        #     self._vec1 = xp.zeros_like(self._nuh0_at_quad[0])
        #     self._vec2 = xp.zeros_like(self._nuh0_at_quad[0])
        #     self._vec3 = xp.zeros_like(self._nuh0_at_quad[0])

        # FEM spaces and basis extraction operators for u and b
        u_id = self.derham.space_to_form[u_space]
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._EbT = self.derham.extraction_ops["2"].transpose()

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._EbT.codomain.zeros()

        self._u_new = u.space.zeros()

        self._u_avg1 = u.space.zeros()
        self._u_avg2 = self._EuT.codomain.zeros()

        # load particle pusher kernel
        if u_space == "Hcurl":
            kernel = Pyccelkernel(pusher_kernels.push_bxu_Hcurl)
        elif u_space == "Hdiv":
            kernel = Pyccelkernel(pusher_kernels.push_bxu_Hdiv)
        elif u_space == "H1vec":
            kernel = Pyccelkernel(pusher_kernels.push_bxu_H1vec)
        else:
            raise ValueError(
                f'{u_space = } not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
            )

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._b_full2[0]._data,
            self._b_full2[1]._data,
            self._b_full2[2]._data,
            self._u_avg2[0]._data,
            self._u_avg2[1]._data,
            self._u_avg2[2]._data,
            self._boundary_cut_e1,
        )

        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(self.mass_ops, "M" + u_id + "n")

        # preconditioner
        if solver["type"][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver["type"][1])
            pc = pc_class(_A)

        _BC = -1 / 4 * self._accumulator.operators[0]

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            solver["type"][0],
            pc=pc,
            tol=solver["tol"],
            maxiter=solver["maxiter"],
            verbose=solver["verbose"],
            recycle=solver["recycle"],
        )

    def __call__(self, dt):
        # pointer to old coefficients
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        self._b_eq.copy(out=self._b_full1)

        if self._b_tilde is not None:
            self._b_full1 += self._b_tilde

        # extract coefficients to tensor product space (in-place)
        self._EbT.dot(self._b_full1, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        self._b_full2.update_ghost_regions()

        # # perform accumulation (either with or without control variate)
        # if self.particles[0].control_variate:

        #     # evaluate magnetic field at quadrature points (in-place)
        #     WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._b_full2,
        #                                    out=[self._b_quad1, self._b_quad2, self._b_quad3])

        #     self._vec1[:, :, :] = self._coupling_vec * \
        #         (self._b_quad2 *
        #          self._nuh0_at_quad[2] - self._b_quad3*self._nuh0_at_quad[1])
        #     self._vec2[:, :, :] = self._coupling_vec * \
        #         (self._b_quad3 *
        #          self._nuh0_at_quad[0] - self._b_quad1*self._nuh0_at_quad[2])
        #     self._vec3[:, :, :] = self._coupling_vec * \
        #         (self._b_quad1 *
        #          self._nuh0_at_quad[1] - self._b_quad2*self._nuh0_at_quad[0])

        #     self._accumulator(self._b_full2[0]._data,
        #                       self._b_full2[1]._data,
        #                       self._b_full2[2]._data,
        #                       self._space_key_int,
        #                       self._coupling_mat,
        #                       self._coupling_vec,
        #                       control_vec=[self._vec1, self._vec2, self._vec3])
        # else:
        self._accumulator(
            self._b_full2[0]._data,
            self._b_full2[1]._data,
            self._b_full2[2]._data,
            self._space_key_int,
            self._coupling_mat,
            self._coupling_vec,
            self._boundary_cut_e1,
        )

        # solve linear system for updated u coefficients (in-place)
        un1, info = self._schur_solver(
            un,
            -self._accumulator.vectors[0] / 2,
            dt,
            out=self._u_new,
        )

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg1)
        _u += un1
        _u *= 0.5

        _Eu = self._EuT.dot(_u, out=self._u_avg2)

        _Eu.update_ghost_regions()

        # push particles
        self._pusher(self._scale_push * dt)

        # write new coeffs into Propagator.variables
        max_du = self.feec_vars_update(un1)

        # update weights in case of control variate
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

        if self._info and self._rank == 0:
            print("Status     for CurrentCoupling6DCurrent:", info["success"])
            print("Iterations for CurrentCoupling6DCurrent:", info["niter"])
            print("Maxdiff up for CurrentCoupling6DCurrent:", max_du)
            print()


class CurrentCoupling5DCurlb(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \left\{ 
            \begin{aligned} 
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V\, \textnormal{d} \mathbf x = - \frac{A_\textnormal{h}}{A_b} \iint \frac{f^\text{vol}}{B^*_\parallel} v_\parallel^2 (\nabla \times \mathbf b_0)  \times \mathbf B \cdot \mathbf V \, \textnormal{d} \mathbf x \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
                \\
                &\frac{\partial f}{\partial t} = - \frac{1}{B^*_\parallel} v_\parallel (\nabla \times \mathbf b_0) \cdot (\mathbf B \times \tilde{\mathbf U}) \nabla_{v_\parallel}f \,.
            \end{aligned}
        \right.

    :ref:`time_discret`: Crank-Nicolson (implicit mid-point). System size reduction via :class:`~struphy.linear_algebra.schur_solver.SchurSolver`:

    .. math::

        \begin{bmatrix} 
            \mathbf u^{n+1} - \mathbf u^n \\ V_\parallel^{n+1} - V_\parallel^n
        \end{bmatrix} 
        = \frac{\Delta t}{2} 
        \begin{bmatrix} 
            0 & - (\mathbb{M}^{2,n})^{-1} \left\{ \mathbb{L}^2 \frac{1}{\bar{\sqrt{g}}} \right\}\cdot_\text{vector} \left\{\bar{b}^{\nabla \times}_0 (\bar{B}^\times_f)^\top \bar{V}_\parallel \frac{1}{\bar{\sqrt{g}}}\right\} \frac{1}{\bar B^{*0}_\parallel})
            \\  
            \frac{1}{\bar B^{*0}_\parallel} \left\{\bar{b}^{\nabla \times}_0 (\bar{B}^\times_f)^\top \bar{V}_\parallel \frac{1}{\bar{\sqrt{g}}}\right\}\, \cdot_\text{vector} \left\{\frac{1}{\bar{\sqrt{g}}}(\mathbb{L}²)^\top\right\} (\mathbb{M}^{2,n})^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            (\mathbb{M}^{2,n})^{-1} (\mathbf u^{n+1} + \mathbf u^n)
            \\
            \frac{A_\textnormal{h}}{A_b} W (V_\parallel^{n+1} + V_\parallel^n)
        \end{bmatrix} \,.

    For the detail explanation of the notations, see `2022_DriftKineticCurrentCoupling <https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2022_DriftKineticCurrentCoupling.md?ref_type=heads>`_.
    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None
            self._energetic_ions: PICVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

        @property
        def energetic_ions(self) -> PICVariable:
            return self._energetic_ions

        @energetic_ions.setter
        def energetic_ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._energetic_ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # propagator options
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
        _A = getattr(self.mass_ops, id_M)

        # Preconditioner
        if self.options.precond is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, self.options.precond)
            pc = pc_class(getattr(self.mass_ops, id_M))

        # magnetic equilibrium field
        unit_b1 = self.projected_equil.unit_b1
        curl_unit_b1 = self.projected_equil.curl_unit_b1
        self._b2 = self.projected_equil.b2

        # magnetic field
        self._b_tilde = self.options.b_tilde.spline.vector

        # scaling factor
        epsilon = self.variables.energetic_ions.species.equation_params.epsilon

        # temporary vectors to avoid memory allocation
        self._b_full = self._b2.space.zeros()
        self._u_new = self.variables.u.spline.vector.space.zeros()
        self._u_avg = self.variables.u.spline.vector.space.zeros()

        # define Accumulator and arguments
        self._ACC = Accumulator(
            self.variables.energetic_ions.particles,
            self.options.u_space,
            accum_kernels_gc.cc_lin_mhd_5d_curlb,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
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

        # define Pusher
        if self.options.u_space == "Hcurl":
            pusher_kernel = pusher_kernels_gc.push_gc_cc_J1_Hcurl
        elif self.options.u_space == "Hdiv":
            pusher_kernel = pusher_kernels_gc.push_gc_cc_J1_Hdiv
        elif self.options.u_space == "H1vec":
            pusher_kernel = pusher_kernels_gc.push_gc_cc_J1_H1vec
        else:
            raise ValueError(
                f'{self.options.u_space  = } not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
            )

        args_pusher_kernel = (
            self.derham.args_derham,
            epsilon,
            self._b_full[0]._data,
            self._b_full[1]._data,
            self._b_full[2]._data,
            unit_b1[0]._data,
            unit_b1[1]._data,
            unit_b1[2]._data,
            curl_unit_b1[0]._data,
            curl_unit_b1[1]._data,
            curl_unit_b1[2]._data,
            self._u_avg[0]._data,
            self._u_avg[1]._data,
            self._u_avg[2]._data,
        )

        self._pusher = Pusher(
            self.variables.energetic_ions.particles,
            pusher_kernel,
            args_pusher_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        _BC = -1 / 4 * self._ACC.operators[0]

        self._schur_solver = SchurSolver(
            _A,
            _BC,
            self.options.solver,
            precond=pc,
            solver_params=self.options.solver_params,
        )

    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b2.copy(out=self._b_full)

        b_full += self._b_tilde
        b_full.update_ghost_regions()

        self._ACC(
            *self._args_accum_kernel,
        )

        # solve
        un1, info = self._schur_solver(
            un,
            -self._ACC.vectors[0] / 2,
            dt,
            out=self._u_new,
        )

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg)
        _u += un1
        _u *= 0.5

        _u.update_ghost_regions()

        self._pusher(dt)

        # update u coefficients
        diffs = self.update_feec_variables(u=un1)

        # update_weights
        if self.variables.energetic_ions.species.weights_params.control_variate:
            self.variables.energetic_ions.particles.update_weights()

        if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
            print("Status     for CurrentCoupling5DCurlb:", info["success"])
            print("Iterations for CurrentCoupling5DCurlb:", info["niter"])
            print("Maxdiff up for CurrentCoupling5DCurlb:", diffs["u"])
            print()


class CurrentCoupling5DGradB(Propagator):
    r""":ref:`FEEC <gempic>` discretization of the following equations: 
    find :math:`\mathbf U \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}` and  :math:`\mathbf B \in H(\textnormal{div})` such that

    .. math::

        \left\{ 
            \begin{aligned} 
                \int \rho_0 &\frac{\partial \tilde{\mathbf U}}{\partial t} \cdot \mathbf V \, \textnormal{d} \mathbf x = - \frac{A_\textnormal{h}}{A_b} \iint \mu \frac{f^\text{vol}}{B^*_\parallel} (\mathbf b_0 \times \nabla B_\parallel) \times \mathbf B \cdot \mathbf V \,\textnormal{d} \mathbf x \textnormal{d} v_\parallel \textnormal{d} \mu \quad \forall \,\mathbf V \in \{H(\textnormal{curl}), H(\textnormal{div}), (H^1)^3\}\,,
                \\
                &\frac{\partial f}{\partial t} = \frac{1}{B^*_\parallel} \left[ \mathbf b_0 \times (\tilde{\mathbf U} \times \mathbf B) \right] \cdot \nabla f\,.
            \end{aligned}
        \right.

    :ref:`time_discret`: Explicit ('rk4', 'forward_euler', 'heun2', 'rk2', 'heun3').

    .. math::

        \begin{bmatrix} 
            \dot{\mathbf u}\\ \dot{\mathbf H}
        \end{bmatrix} 
        =
        \begin{bmatrix} 
            0 & (\mathbb{M}^{2,n})^{-1} \mathbb{L}² \frac{1}{\bar{\sqrt{g}}} \frac{1}{\bar B^{*0}_\parallel}\bar{B}^\times_f \bar{G}^{-1} \bar{b}^\times_0 \bar{G}^{-1} 
            \\  
            -\bar{G}^{-1} \bar{b}^\times_0 \bar{G}^{-1}  \bar{B}^\times_f \frac{1}{\bar B^{*0}_\parallel} \frac{1}{\bar{\sqrt{g}}} (\mathbb{L}²)^\top (\mathbb{M}^{2,n})^{-1} & 0 
        \end{bmatrix} 
        \begin{bmatrix}
            \mathbb M^{2,n} \mathbf u
            \\
            \frac{A_\textnormal{h}}{A_b} \bar M \bar W \overline{\nabla B}_\parallel 
        \end{bmatrix} \,.

    For the detail explanation of the notations, see `2022_DriftKineticCurrentCoupling <https://gitlab.mpcdf.mpg.de/struphy/struphy-projects/-/blob/main/running-projects/2022_DriftKineticCurrentCoupling.md?ref_type=heads>`_.
    """

    class Variables:
        def __init__(self):
            self._u: FEECVariable = None
            self._energetic_ions: PICVariable = None

        @property
        def u(self) -> FEECVariable:
            return self._u

        @u.setter
        def u(self, new):
            assert isinstance(new, FEECVariable)
            assert new.space in ("Hcurl", "Hdiv", "H1vec")
            self._u = new

        @property
        def energetic_ions(self) -> PICVariable:
            return self._energetic_ions

        @energetic_ions.setter
        def energetic_ions(self, new):
            assert isinstance(new, PICVariable)
            assert new.space == "Particles5D"
            self._energetic_ions = new

    def __init__(self):
        self.variables = self.Variables()

    @dataclass
    class Options:
        # specific literals
        OptsAlgo = Literal[
            "discrete_gradient",
            "explicit",
        ]
        # propagator options
        b_tilde: FEECVariable = None
        ep_scale: float = 1.0
        algo: OptsAlgo = "explicit"
        butcher: ButcherTableau = None
        u_space: OptsVecSpace = "Hdiv"
        solver: OptsSymmSolver = "pcg"
        precond: OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        dg_solver_params: DiscreteGradientSolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)
            check_option(self.u_space, OptsVecSpace)
            check_option(self.solver, OptsSymmSolver)
            check_option(self.precond, OptsMassPrecond)
            assert isinstance(self.b_tilde, FEECVariable)
            assert isinstance(self.ep_scale, float)

            # defaults
            if self.algo == "explicit" and self.butcher is None:
                self.butcher = ButcherTableau()

            if self.algo == "discrete_gradient" and self.dg_solver_params is None:
                self.dg_solver_params = DiscreteGradientSolverParameters()

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
        self._PB = getattr(self.basis_ops, "PB")

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
        # magnetic equilibrium field
        unit_b1 = self.projected_equil.unit_b1
        curl_unit_b1 = self.projected_equil.curl_unit_b1
        self._b2 = self.projected_equil.b2
        gradB1 = self.projected_equil.gradB1
        absB0 = self.projected_equil.absB0

        # magnetic field
        self._b_tilde = self.options.b_tilde.spline.vector

        # scaling factor
        epsilon = self.variables.energetic_ions.species.equation_params.epsilon

        if self.options.algo == "explicit":
            # temporary vectors to avoid memory allocation
            self._b_full = self._b2.space.zeros()
            self._u_new = self.variables.u.spline.vector.space.zeros()
            self._u_temp = self.variables.u.spline.vector.space.zeros()
            self._ku = self.variables.u.spline.vector.space.zeros()
            self._PB_b = self._PB.codomain.zeros()
            self._grad_PB_b = self.derham.grad.codomain.zeros()

            # define Accumulator and arguments
            self._ACC = Accumulator(
                self.variables.energetic_ions.particles,
                self.options.u_space,
                accum_kernels_gc.cc_lin_mhd_5d_gradB,
                self.mass_ops,
                self.domain.args_domain,
                add_vector=True,
                symmetry="symm",
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
                self._grad_PB_b[0]._data,
                self._grad_PB_b[1]._data,
                self._grad_PB_b[2]._data,
                self._u_form_int,
            )

            # define Pusher
            if self.options.u_space == "Hdiv":
                self._pusher_kernel = pusher_kernels_gc.push_gc_cc_J2_stage_Hdiv
            elif self.options.u_space == "H1vec":
                self._pusher_kernel = pusher_kernels_gc.push_gc_cc_J2_stage_H1vec
            else:
                raise ValueError(
                    f'{self.options.u_space  = } not valid, choose from "Hdiv" or "H1vec.',
                )

            # temp fix due to refactoring of ButcherTableau:
            butcher = self.options.butcher
            import numpy as np

            butcher._a = xp.diag(butcher.a, k=-1)
            butcher._a = xp.array(list(butcher.a) + [0.0])

            self._args_pusher_kernel = (
                self.domain.args_domain,
                self.derham.args_derham,
                epsilon,
                self._b_full[0]._data,
                self._b_full[1]._data,
                self._b_full[2]._data,
                unit_b1[0]._data,
                unit_b1[1]._data,
                unit_b1[2]._data,
                curl_unit_b1[0]._data,
                curl_unit_b1[1]._data,
                curl_unit_b1[2]._data,
                self._u_temp[0]._data,
                self._u_temp[1]._data,
                self._u_temp[2]._data,
                self.options.butcher.a,
                self.options.butcher.b,
                self.options.butcher.c,
            )

        else:
            # temporary vectors to avoid memory allocation
            self._b_full = self._b2.space.zeros()
            self._PB_b = self._PB.codomain.zeros()
            self._grad_PB_b = self.derham.grad.codomain.zeros()
            self._u_old = self.variables.u.spline.vector.space.zeros()
            self._u_new = self.variables.u.spline.vector.space.zeros()
            self._u_diff = self.variables.u.spline.vector.space.zeros()
            self._u_mid = self.variables.u.spline.vector.space.zeros()
            self._M2n_dot_u = self.variables.u.spline.vector.space.zeros()
            self._ku = self.variables.u.spline.vector.space.zeros()
            self._u_temp = self.variables.u.spline.vector.space.zeros()

            # Call the accumulation and Pusher class
            accum_kernel_init = accum_kernels_gc.cc_lin_mhd_5d_gradB_dg_init
            accum_kernel = accum_kernels_gc.cc_lin_mhd_5d_gradB_dg
            self._accum_kernel_en_fB_mid = utilities_kernels.eval_gradB_ediff

            self._args_accum_kernel = (
                epsilon,
                self.options.ep_scale,
                self._b_tilde[0]._data,
                self._b_tilde[1]._data,
                self._b_tilde[2]._data,
                self._b2[0]._data,
                self._b2[1]._data,
                self._b2[2]._data,
                unit_b1[0]._data,
                unit_b1[1]._data,
                unit_b1[2]._data,
                curl_unit_b1[0]._data,
                curl_unit_b1[1]._data,
                curl_unit_b1[2]._data,
                self._grad_PB_b[0]._data,
                self._grad_PB_b[1]._data,
                self._grad_PB_b[2]._data,
                gradB1[0]._data,
                gradB1[1]._data,
                gradB1[2]._data,
                self._u_form_int,
            )

            self._args_accum_kernel_en_fB_mid = (
                self.domain.args_domain,
                self.derham.args_derham,
                gradB1[0]._data,
                gradB1[1]._data,
                gradB1[2]._data,
                self._grad_PB_b[0]._data,
                self._grad_PB_b[1]._data,
                self._grad_PB_b[2]._data,
            )

            self._ACC_init = AccumulatorVector(
                self.variables.energetic_ions.particles,
                self.options.u_space,
                accum_kernel_init,
                self.mass_ops,
                self.domain.args_domain,
                filter_params=self.options.filter_params,
            )

            self._ACC = AccumulatorVector(
                self.variables.energetic_ions.particles,
                self.options.u_space,
                accum_kernel,
                self.mass_ops,
                self.domain.args_domain,
                filter_params=self.options.filter_params,
            )

            self._args_pusher_kernel_init = (
                self.domain.args_domain,
                self.derham.args_derham,
                epsilon,
                self._b_full[0]._data,
                self._b_full[1]._data,
                self._b_full[2]._data,
                unit_b1[0]._data,
                unit_b1[1]._data,
                unit_b1[2]._data,
                curl_unit_b1[0]._data,
                curl_unit_b1[1]._data,
                curl_unit_b1[2]._data,
                self.variables.u.spline.vector[0]._data,
                self.variables.u.spline.vector[1]._data,
                self.variables.u.spline.vector[2]._data,
            )

            self._args_pusher_kernel = (
                self.domain.args_domain,
                self.derham.args_derham,
                epsilon,
                self._b_full[0]._data,
                self._b_full[1]._data,
                self._b_full[2]._data,
                unit_b1[0]._data,
                unit_b1[1]._data,
                unit_b1[2]._data,
                curl_unit_b1[0]._data,
                curl_unit_b1[1]._data,
                curl_unit_b1[2]._data,
                self._u_mid[0]._data,
                self._u_mid[1]._data,
                self._u_mid[2]._data,
                self._u_temp[0]._data,
                self._u_temp[1]._data,
                self._u_temp[2]._data,
            )

            self._pusher_kernel_init = pusher_kernels_gc.push_gc_cc_J2_dg_init_Hdiv
            self._pusher_kernel = pusher_kernels_gc.push_gc_cc_J2_dg_Hdiv

    def __call__(self, dt):
        # current FE coeffs
        un = self.variables.u.spline.vector

        # particle markers and idx
        particles = self.variables.energetic_ions.particles
        holes = particles.holes
        args_markers = particles.args_markers
        markers = args_markers.markers
        first_init_idx = args_markers.first_init_idx
        first_free_idx = args_markers.first_free_idx

        # clear buffer
        markers[:, first_init_idx:-2] = 0.0

        # save old marker positions
        markers[:, first_init_idx : first_init_idx + 3] = markers[:, :3]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b2.copy(out=self._b_full)

        b_full += self._b_tilde
        b_full.update_ghost_regions()

        if self.options.algo == "explicit":
            PB_b = self._PB.dot(b_full, out=self._PB_b)
            grad_PB_b = self.derham.grad.dot(PB_b, out=self._grad_PB_b)
            grad_PB_b.update_ghost_regions()

            # save old u
            u_new = un.copy(out=self._u_new)

            for stage in range(self.options.butcher.n_stages):
                # accumulate
                self._ACC(
                    *self._args_accum_kernel,
                )

                # push particles
                self._pusher_kernel(
                    dt,
                    stage,
                    args_markers,
                    *self._args_pusher_kernel,
                )

                if particles.mpi_comm is not None:
                    particles.mpi_sort_markers()
                else:
                    particles.apply_kinetic_bc()

                # solve linear system for updating u coefficients
                ku = self._A_inv.dot(self._ACC.vectors[0], out=self._ku)
                info = self._A_inv._info

                # calculate u^{n+1}_k
                u_temp = un.copy(out=self._u_temp)
                u_temp += ku * dt * self.options.butcher.a[stage]

                u_temp.update_ghost_regions()

                # calculate u^{n+1}
                u_new += ku * dt * self.options.butcher.b[stage]

                if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
                    print("Stage: ", stage)
                    print("Status     for CurrentCoupling5DGradB:", info["success"])
                    print("Iterations for CurrentCoupling5DGradB:", info["niter"])
                    print()

            # update u coefficients
            diffs = self.update_feec_variables(u=u_new)

            # clear the buffer
            markers[:, first_init_idx:-2] = 0.0

            # update_weights
            if self.variables.energetic_ions.species.weights_params.control_variate:
                particles.update_weights()

            if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
                print("Maxdiff up for CurrentCoupling5DGradB:", diffs["u"])
                print()

        else:
            # total number of markers
            n_mks_tot = particles.Np

            # relaxation factor
            alpha = self.options.dg_solver_params.relaxation_factor

            # eval parallel tilde b and its gradient
            PB_b = self._PB.dot(self._b_tilde, out=self._PB_b)
            PB_b.update_ghost_regions()
            grad_PB_b = self.derham.grad.dot(PB_b, out=self._grad_PB_b)
            grad_PB_b.update_ghost_regions()

            # save old u
            u_old = un.copy(out=self._u_old)
            u_new = un.copy(out=self._u_new)

            # save en_U_old
            self._A.dot(un, out=self._M2n_dot_u)
            en_U_old = un.inner(self._M2n_dot_u) / 2.0

            # save en_fB_old
            particles.save_magnetic_energy(PB_b)
            en_fB_old = xp.sum(markers[~holes, 8].dot(markers[~holes, 5])) * self.options.ep_scale
            en_fB_old /= n_mks_tot

            buffer_array = xp.array([en_fB_old])

            if particles.mpi_comm is not None:
                particles.mpi_comm.Allreduce(
                    MPI.IN_PLACE,
                    buffer_array,
                    op=MPI.SUM,
                )

            if particles.clone_config is not None:
                particles.clone_config.inter_comm.Allreduce(
                    MPI.IN_PLACE,
                    buffer_array,
                    op=MPI.SUM,
                )

            en_fB_old = buffer_array[0]
            en_tot_old = en_U_old + en_fB_old

            # initial guess
            self._ACC_init(*self._args_accum_kernel)

            ku = self._A_inv.dot(self._ACC_init.vectors[0], out=self._ku)
            u_new += ku * dt

            u_new.update_ghost_regions()

            # save en_U_new
            self._A.dot(u_new, out=self._M2n_dot_u)
            en_U_new = u_new.inner(self._M2n_dot_u) / 2.0

            # push eta
            self._pusher_kernel_init(
                dt,
                args_markers,
                *self._args_pusher_kernel_init,
            )

            if particles.mpi_comm is not None:
                particles.mpi_sort_markers(apply_bc=False)

            # save en_fB_new
            particles.save_magnetic_energy(PB_b)
            en_fB_new = xp.sum(markers[~holes, 8].dot(markers[~holes, 5])) * self.options.ep_scale
            en_fB_new /= n_mks_tot

            buffer_array = xp.array([en_fB_new])

            if particles.mpi_comm is not None:
                particles.mpi_comm.Allreduce(
                    MPI.IN_PLACE,
                    buffer_array,
                    op=MPI.SUM,
                )

            if particles.clone_config is not None:
                particles.clone_config.inter_comm.Allreduce(
                    MPI.IN_PLACE,
                    buffer_array,
                    op=MPI.SUM,
                )

            en_fB_new = buffer_array[0]

            # fixed-point iterations
            iter_num = 0

            while True:
                iter_num += 1

                if self.options.dg_solver_params.verbose and MPI.COMM_WORLD.Get_rank() == 0:
                    print("# of iteration: ", iter_num)

                # calculate discrete gradient
                # save u^{n+1, k}
                u_old = u_new.copy(out=self._u_old)

                u_diff = u_old.copy(out=self._u_diff)
                u_diff -= un
                u_diff.update_ghost_regions()

                u_mid = u_old.copy(out=self._u_mid)
                u_mid += un
                u_mid /= 2.0
                u_mid.update_ghost_regions()

                # save H^{n+1, k}
                markers[~holes, first_free_idx : first_free_idx + 3] = markers[~holes, 0:3]

                # calculate denominator ||z^{n+1, k} - z^n||^2
                sum_u_diff_loc = xp.sum((u_diff.toarray() ** 2))

                sum_H_diff_loc = xp.sum(
                    (markers[~holes, :3] - markers[~holes, first_init_idx : first_init_idx + 3]) ** 2
                )

                buffer_array = xp.array([sum_u_diff_loc])

                if particles.mpi_comm is not None:
                    particles.mpi_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                denominator = buffer_array[0]

                buffer_array = xp.array([sum_H_diff_loc])

                if particles.mpi_comm is not None:
                    particles.mpi_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                if particles.clone_config is not None:
                    particles.clone_config.inter_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                denominator += buffer_array[0]

                # sorting markers at mid-point
                if particles.mpi_comm is not None:
                    particles.mpi_sort_markers(apply_bc=False, alpha=0.5)

                self._accum_kernel_en_fB_mid(
                    args_markers,
                    *self._args_accum_kernel_en_fB_mid,
                    first_free_idx + 3,
                )
                en_fB_mid = xp.sum(markers[~holes, first_free_idx + 3].dot(markers[~holes, 5])) * self.options.ep_scale

                en_fB_mid /= n_mks_tot

                buffer_array = xp.array([en_fB_mid])

                if particles.mpi_comm is not None:
                    particles.mpi_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                if particles.clone_config is not None:
                    particles.clone_config.inter_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                en_fB_mid = buffer_array[0]

                if denominator == 0.0:
                    const = 0.0
                else:
                    const = (en_fB_new - en_fB_old - en_fB_mid) / denominator

                # update u^{n+1, k}
                self._ACC(*self._args_accum_kernel, const)

                ku = self._A_inv.dot(self._ACC.vectors[0], out=self._ku)

                u_new = un.copy(out=self._u_new)
                u_new += ku * dt
                u_new *= alpha
                u_new += u_old * (1.0 - alpha)

                u_new.update_ghost_regions()

                # update en_U_new
                self._A.dot(u_new, out=self._M2n_dot_u)
                en_U_new = u_new.inner(self._M2n_dot_u) / 2.0

                # update H^{n+1, k}
                self._pusher_kernel(
                    dt,
                    args_markers,
                    *self._args_pusher_kernel,
                    const,
                    alpha,
                )

                sum_H_diff_loc = xp.sum(
                    xp.abs(markers[~holes, 0:3] - markers[~holes, first_free_idx : first_free_idx + 3])
                )

                if particles.mpi_comm is not None:
                    particles.mpi_sort_markers(apply_bc=False)

                # update en_fB_new
                particles.save_magnetic_energy(PB_b)
                en_fB_new = xp.sum(markers[~holes, 8].dot(markers[~holes, 5])) * self.options.ep_scale
                en_fB_new /= n_mks_tot

                buffer_array = xp.array([en_fB_new])

                if particles.mpi_comm is not None:
                    particles.mpi_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                if particles.clone_config is not None:
                    particles.clone_config.inter_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                en_fB_new = buffer_array[0]

                # calculate total energy difference
                e_diff = xp.abs(en_U_new + en_fB_new - en_tot_old)

                # calculate ||z^{n+1, k} - z^{n+1, k-1||
                sum_u_diff_loc = xp.sum(xp.abs(u_new.toarray() - u_old.toarray()))

                buffer_array = xp.array([sum_u_diff_loc])

                if particles.mpi_comm is not None:
                    particles.mpi_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                diff = buffer_array[0]

                buffer_array = xp.array([sum_H_diff_loc])

                if particles.mpi_comm is not None:
                    particles.mpi_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                if particles.clone_config is not None:
                    particles.clone_config.inter_comm.Allreduce(
                        MPI.IN_PLACE,
                        buffer_array,
                        op=MPI.SUM,
                    )

                diff += buffer_array[0]

                # check convergence
                if diff < self.options.dg_solver_params.tol:
                    if self.options.dg_solver_params.verbose and MPI.COMM_WORLD.Get_rank() == 0:
                        print("converged diff: ", diff)
                        print("converged e_diff: ", e_diff)

                    if particles.mpi_comm is not None:
                        particles.mpi_comm.Barrier()
                    break

                else:
                    if self.options.dg_solver_params.verbose and MPI.COMM_WORLD.Get_rank() == 0:
                        print("not converged diff: ", diff)
                        print("not converged e_diff: ", e_diff)

                if iter_num == self.options.dg_solver_params.maxiter:
                    if self.options.dg_solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
                        print(
                            f"{iter_num = }, maxiter={self.options.dg_solver_params.maxiter} reached! diff: {diff}, e_diff: {e_diff}",
                        )
                    if particles.mpi_comm is not None:
                        particles.mpi_comm.Barrier()
                    break

            # sorting markers
            if particles.mpi_comm is not None:
                particles.mpi_sort_markers()
            else:
                particles.apply_kinetic_bc()

            # update u coefficients
            diffs = self.update_feec_variables(u=u_new)

            # clear the buffer
            markers[:, first_init_idx:-2] = 0.0

            # update_weights
            if self.variables.energetic_ions.species.weights_params.control_variate:
                particles.update_weights()

            if self.options.dg_solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
                print("Maxdiff up for CurrentCoupling5DGradB:", diffs["u"])
                print()
