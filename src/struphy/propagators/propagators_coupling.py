"Particle and FEEC variables are updated."

from dataclasses import dataclass
from typing import Literal

import numpy as np
from line_profiler import profile
from mpi4py import MPI
from psydac.linalg.block import BlockVector
from psydac.linalg.stencil import StencilVector

from struphy.feec import preconditioner
from struphy.feec.linear_operators import LinOpWithTransp
from struphy.io.options import OptsGenSolver, OptsMassPrecond, OptsSymmSolver, OptsVecSpace, check_option
from struphy.io.setup import descend_options_dict
from struphy.kinetic_background.base import Maxwellian
from struphy.kinetic_background.maxwellians import Maxwellian3D
from struphy.linear_algebra.schur_solver import SchurSolver
from struphy.linear_algebra.solver import SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.pic.accumulation import accum_kernels, accum_kernels_gc
from struphy.pic.accumulation.particles_to_grid import Accumulator
from struphy.pic.particles import Particles5D, Particles6D
from struphy.pic.pushing import pusher_kernels, pusher_kernels_gc
from struphy.pic.pushing.pusher import Pusher
from struphy.polar.basic import PolarVector
from struphy.propagators.base import Propagator


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
        accum_kernel = accum_kernels.vlasov_maxwell

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
            pusher_kernels.push_v_with_efield,
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
            max_diff = np.max(
                np.abs(
                    np.sqrt(
                        particles.markers_wo_holes[:, 3] ** 2
                        + particles.markers_wo_holes[:, 4] ** 2
                        + particles.markers_wo_holes[:, 5] ** 2,
                    )
                    - np.sqrt(
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
            accum_kernels.linear_vlasov_ampere,
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
        self._f0_values = np.zeros(particles.markers.shape[0], dtype=float)
        self._old_weights = np.empty(particles.markers.shape[0], dtype=float)

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
            pusher_kernels.push_weights_with_efield_lin_va,
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
            max_diff = np.max(
                np.abs(
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
            accum_ker = accum_kernels.pc_lin_mhd_6d
            pusher_ker = pusher_kernels.push_pc_GXu
        else:
            accum_ker = accum_kernels.pc_lin_mhd_6d_full
            pusher_ker = pusher_kernels.push_pc_GXu_full

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
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = Ah / Ab / epsilon**2
        self._coupling_vec = Ah / Ab / epsilon
        self._scale_push = 1.0 / epsilon

        self._boundary_cut_e1 = boundary_cut["e1"]

        # load accumulator
        self._accumulator = Accumulator(
            particles,
            u_space,
            accum_kernels.cc_lin_mhd_6d_2,
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
        #     self._b_quad1 = np.zeros_like(self._nuh0_at_quad[0])
        #     self._b_quad2 = np.zeros_like(self._nuh0_at_quad[0])
        #     self._b_quad3 = np.zeros_like(self._nuh0_at_quad[0])

        #     # memory allocation for (self._b_quad x self._nuh0_at_quad) * self._coupling_vec
        #     self._vec1 = np.zeros_like(self._nuh0_at_quad[0])
        #     self._vec2 = np.zeros_like(self._nuh0_at_quad[0])
        #     self._vec3 = np.zeros_like(self._nuh0_at_quad[0])

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
            kernel = pusher_kernels.push_bxu_Hcurl
        elif u_space == "Hdiv":
            kernel = pusher_kernels.push_bxu_Hdiv
        elif u_space == "H1vec":
            kernel = pusher_kernels.push_bxu_H1vec
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
        particles: Particles5D,
        u: BlockVector,
        *,
        b: BlockVector,
        b_eq: BlockVector,
        unit_b1: BlockVector,
        absB0: StencilVector,
        gradB1: BlockVector,
        curl_unit_b2: BlockVector,
        u_space: str,
        solver: dict = options(default=True)["solver"],
        filter: dict = options(default=True)["filter"],
        coupling_params: dict,
        epsilon: float = 1.0,
        boundary_cut: dict = options(default=True)["boundary_cut"],
    ):
        super().__init__(particles, u)

        assert u_space in {"Hcurl", "Hdiv", "H1vec"}

        if u_space == "H1vec":
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[u_space],
            )

        self._epsilon = epsilon
        self._b = b
        self._b_eq = b_eq
        self._unit_b1 = unit_b1
        self._absB0 = absB0
        self._gradB1 = gradB1
        self._curl_norm_b = curl_unit_b2

        self._info = solver["info"]
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = coupling_params["Ah"] / coupling_params["Ab"]
        self._coupling_vec = coupling_params["Ah"] / coupling_params["Ab"]
        self._scale_push = 1

        self._boundary_cut_e1 = boundary_cut["e1"]

        u_id = self.derham.space_to_form[u_space]
        self._E0T = self.derham.extraction_ops["0"].transpose()
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._E2T = self.derham.extraction_ops["2"].transpose()
        self._E1T = self.derham.extraction_ops["1"].transpose()

        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)
        self._curl_norm_b.update_ghost_regions()
        self._absB0 = self._E0T.dot(self._absB0)

        # define system [[A B], [C I]] [u_new, v_new] = [[A -B], [-C I]] [u_old, v_old] (without time step size dt)
        _A = getattr(self.mass_ops, "M" + u_id + "n")

        # preconditioner
        if solver["type"][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver["type"][1])
            pc = pc_class(_A)

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()
        self._u_new = u.space.zeros()
        self._u_avg1 = u.space.zeros()
        self._u_avg2 = self._EuT.codomain.zeros()

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(
            particles,
            u_space,
            accum_kernels_gc.cc_lin_mhd_5d_J1,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
            filter_params=filter,
        )

        if u_space == "Hcurl":
            kernel = pusher_kernels_gc.push_gc_cc_J1_Hcurl
        elif u_space == "Hdiv":
            kernel = pusher_kernels_gc.push_gc_cc_J1_Hdiv
        elif u_space == "H1vec":
            kernel = pusher_kernels_gc.push_gc_cc_J1_H1vec
        else:
            raise ValueError(
                f'{u_space = } not valid, choose from "Hcurl", "Hdiv" or "H1vec.',
            )

        # instantiate Pusher
        args_kernel = (
            self.derham.args_derham,
            self._epsilon,
            self._b_full2[0]._data,
            self._b_full2[1]._data,
            self._b_full2[2]._data,
            self._unit_b1[0]._data,
            self._unit_b1[1]._data,
            self._unit_b1[2]._data,
            self._curl_norm_b[0]._data,
            self._curl_norm_b[1]._data,
            self._curl_norm_b[2]._data,
            self._u_avg2[0]._data,
            self._u_avg2[1]._data,
            self._u_avg2[2]._data,
            0.0,
        )

        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        # define BC and B dot V of the Schur block matrix [[A, B], [C, I]]
        _BC = -1 / 4 * self._ACC.operators[0]

        # call SchurSolver class
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
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b_eq.copy(out=self._b_full1)

        if self._b is not None:
            self._b_full1 += self._b

        # extract coefficients to tensor product space (in-place)
        Eb_full = self._E2T.dot(b_full, out=self._b_full2)

        # update ghost regions because of non-local access in accumulation kernel!
        Eb_full.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        # if self.particles[0].control_variate:

        #     # evaluate magnetic field at quadrature points (in-place)
        #     WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._b_full2,
        #                                    out=[self._b_at_quad[0], self._b_at_quad[1], self._b_at_quad[2]])

        #     # evaluate B_parallel
        #     self._B_para_at_quad = np.sum(
        #         p * q for p, q in zip(self._unit_b1_at_quad, self._b_at_quad))
        #     self._B_para_at_quad += self._unit_b1_dot_curl_norm_b_at_quad

        #     # assemble (B x)(curl norm_b)(curl norm_b)(B x) / B_star_para² / det_df³ * (f0.u_para² + f0.vth_para²) * f0.n
        #     self._mat11[:, :, :] = (self._b_at_quad[1]*self._curl_norm_b_at_quad[2] -
        #                             self._b_at_quad[2]*self._curl_norm_b_at_quad[1])**2 * \
        #         self._control_const * self._coupling_mat / \
        #         self._det_df_at_quad**3 / self._B_para_at_quad**2
        #     self._mat12[:, :, :] = (self._b_at_quad[1]*self._curl_norm_b_at_quad[2] -
        #                             self._b_at_quad[2]*self._curl_norm_b_at_quad[1]) * \
        #         (self._b_at_quad[2]*self._curl_norm_b_at_quad[0] -
        #          self._b_at_quad[0]*self._curl_norm_b_at_quad[2]) * \
        #         self._control_const * self._coupling_mat / \
        #         self._det_df_at_quad**3 / self._B_para_at_quad**2
        #     self._mat13[:, :, :] = (self._b_at_quad[1]*self._curl_norm_b_at_quad[2] -
        #                             self._b_at_quad[2]*self._curl_norm_b_at_quad[1]) * \
        #         (self._b_at_quad[0]*self._curl_norm_b_at_quad[1] -
        #          self._b_at_quad[1]*self._curl_norm_b_at_quad[0]) * \
        #         self._control_const * self._coupling_mat / \
        #         self._det_df_at_quad**3 / self._B_para_at_quad**2
        #     self._mat22[:, :, :] = (self._b_at_quad[2]*self._curl_norm_b_at_quad[0] -
        #                             self._b_at_quad[0]*self._curl_norm_b_at_quad[2])**2 * \
        #         self._control_const * self._coupling_mat / \
        #         self._det_df_at_quad**3 / self._B_para_at_quad**2
        #     self._mat23[:, :, :] = (self._b_at_quad[2]*self._curl_norm_b_at_quad[0] -
        #                             self._b_at_quad[0]*self._curl_norm_b_at_quad[2]) * \
        #         (self._b_at_quad[0]*self._curl_norm_b_at_quad[1] -
        #          self._b_at_quad[1]*self._curl_norm_b_at_quad[0]) * \
        #         self._control_const * self._coupling_mat / \
        #         self._det_df_at_quad**3 / self._B_para_at_quad**2
        #     self._mat33[:, :, :] = (self._b_at_quad[0]*self._curl_norm_b_at_quad[1] -
        #                             self._b_at_quad[1]*self._curl_norm_b_at_quad[0])**2 * \
        #         self._control_const * self._coupling_mat / \
        #         self._det_df_at_quad**3 / self._B_para_at_quad**2

        #     self._mat21[:, :, :] = -self._mat12
        #     self._mat31[:, :, :] = -self._mat13
        #     self._mat32[:, :, :] = -self._mat23

        #     # assemble (B x)(curl norm_b) / B_star_para / det_df * (f0.u_para² + f0.vth_para²) * f0.n
        #     self._vec1[:, :, :] = (self._b_at_quad[1]*self._curl_norm_b_at_quad[2] -
        #                            self._b_at_quad[2]*self._curl_norm_b_at_quad[1]) * \
        #         self._control_const * self._coupling_vec / \
        #         self._det_df_at_quad / self._B_para_at_quad
        #     self._vec2[:, :, :] = (self._b_at_quad[2]*self._curl_norm_b_at_quad[0] -
        #                            self._b_at_quad[0]*self._curl_norm_b_at_quad[2]) * \
        #         self._control_const * self._coupling_vec / \
        #         self._det_df_at_quad / self._B_para_at_quad
        #     self._vec3[:, :, :] = (self._b_at_quad[0]*self._curl_norm_b_at_quad[1] -
        #                            self._b_at_quad[1]*self._curl_norm_b_at_quad[0]) * \
        #         self._control_const * self._coupling_vec / \
        #         self._det_df_at_quad / self._B_para_at_quad

        #     self._ACC.accumulate(self.particles[0], self._epsilon,
        #                          Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
        #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                          self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
        #                          self._space_key_int, self._coupling_mat, self._coupling_vec, 0.1,
        #                          control_mat=[[None, self._mat12, self._mat13],
        #                                       [self._mat21, None, self._mat23],
        #                                       [self._mat31, self._mat32, None]],
        #                          control_vec=[self._vec1, self._vec2, self._vec3])
        # else:
        #     self._ACC.accumulate(self.particles[0], self._epsilon,
        #                          Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
        #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
        #                          self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
        #                          self._space_key_int, self._coupling_mat, self._coupling_vec, 0.1)

        self._ACC(
            self._epsilon,
            Eb_full[0]._data,
            Eb_full[1]._data,
            Eb_full[2]._data,
            self._unit_b1[0]._data,
            self._unit_b1[1]._data,
            self._unit_b1[2]._data,
            self._curl_norm_b[0]._data,
            self._curl_norm_b[1]._data,
            self._curl_norm_b[2]._data,
            self._space_key_int,
            self._coupling_mat,
            self._coupling_vec,
            self._boundary_cut_e1,
        )

        # update u coefficients
        un1, info = self._schur_solver(
            un,
            -self._ACC.vectors[0] / 2,
            dt,
            out=self._u_new,
        )

        # call pusher kernel with average field (u_new + u_old)/2 and update ghost regions because of non-local access in kernel
        _u = un.copy(out=self._u_avg1)
        _u += un1
        _u *= 0.5

        _Eu = self._EuT.dot(_u, out=self._u_avg2)

        _Eu.update_ghost_regions()

        self._pusher(self._scale_push * dt)

        # write new coeffs into Propagator.variables
        (max_du,) = self.feec_vars_update(un1)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

        if self._info and self._rank == 0:
            print("Status     for CurrentCoupling5DCurlb:", info["success"])
            print("Iterations for CurrentCoupling5DCurlb:", info["niter"])
            print("Maxdiff up for CurrentCoupling5DCurlb:", max_du)
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
        dct["algo"] = ["rk4", "forward_euler", "heun2", "rk2", "heun3"]
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
        u: BlockVector,
        *,
        b: BlockVector,
        b_eq: BlockVector,
        unit_b1: BlockVector,
        unit_b2: BlockVector,
        absB0: StencilVector,
        gradB1: BlockVector,
        curl_unit_b2: BlockVector,
        u_space: str,
        solver: dict = options(default=True)["solver"],
        algo: dict = options(default=True)["algo"],
        filter: dict = options(default=True)["filter"],
        coupling_params: dict,
        epsilon: float = 1.0,
        boundary_cut: dict = options(default=True)["boundary_cut"],
    ):
        from psydac.linalg.solvers import inverse

        from struphy.ode.utils import ButcherTableau

        super().__init__(particles, u)

        assert u_space in {"Hcurl", "Hdiv", "H1vec"}

        if u_space == "H1vec":
            self._space_key_int = 0
        else:
            self._space_key_int = int(
                self.derham.space_to_form[u_space],
            )

        self._epsilon = epsilon
        self._b = b
        self._b_eq = b_eq
        self._unit_b1 = unit_b1
        self._unit_b2 = unit_b2
        self._absB0 = absB0
        self._gradB1 = gradB1
        self._curl_norm_b = curl_unit_b2

        self._info = solver["info"]
        self._rank = self.derham.comm.Get_rank()

        self._coupling_mat = coupling_params["Ah"] / coupling_params["Ab"]
        self._coupling_vec = coupling_params["Ah"] / coupling_params["Ab"]
        self._scale_push = 1

        self._boundary_cut_e1 = boundary_cut["e1"]

        u_id = self.derham.space_to_form[u_space]
        self._E0T = self.derham.extraction_ops["0"].transpose()
        self._EuT = self.derham.extraction_ops[u_id].transpose()
        self._E1T = self.derham.extraction_ops["1"].transpose()
        self._E2T = self.derham.extraction_ops["2"].transpose()

        self._PB = getattr(self.basis_ops, "PB")

        self._unit_b1 = self._E1T.dot(self._unit_b1)
        self._unit_b2 = self._E2T.dot(self._unit_b2)
        self._curl_norm_b = self._E2T.dot(self._curl_norm_b)
        self._absB0 = self._E0T.dot(self._absB0)

        _A = getattr(self.mass_ops, "M" + u_id + "n")

        # preconditioner
        if solver["type"][1] is None:
            pc = None
        else:
            pc_class = getattr(preconditioner, solver["type"][1])
            pc = pc_class(_A)

        self._solver = inverse(
            _A,
            solver["type"][0],
            pc=pc,
            tol=solver["tol"],
            maxiter=solver["maxiter"],
            verbose=solver["verbose"],
            recycle=solver["recycle"],
        )

        # Call the accumulation and Pusher class
        self._ACC = Accumulator(
            particles,
            u_space,
            accum_kernels_gc.cc_lin_mhd_5d_J2,
            self.mass_ops,
            self.domain.args_domain,
            add_vector=True,
            symmetry="symm",
            filter_params=filter,
        )

        # if self.particles[0].control_variate:

        #     # control variate method is only valid with Maxwellian distributions
        #     assert isinstance(self.particles[0].f0, Maxwellian)
        #     assert params['u_space'] == 'Hdiv'

        #     self._ACC.init_control_variate(self.mass_ops)

        #     # evaluate and save n0 at quadrature points
        #     quad_pts = [quad_grid[nquad].points.flatten()
        #                 for quad_grid, nquad in zip(self.derham.get_quad_grids(self.derham.Vh_fem['0']), self.derham.nquads)]

        #     self._n0_at_quad = self.domain.push(
        #         self.particles[0].f0.n, *quad_pts, kind='0', squeeze_out=False)

        #     # evaluate unit_b1 (1form) dot epsilon * u0_parallel * curl_norm_b/|det(DF)| at quadrature points
        #     quad_pts_array = self.domain.prepare_eval_pts(*quad_pts)[:3]

        #     u0_parallel_at_quad = self.particles[0].f0.u(
        #         *quad_pts_array)[0]

        #     vth_perp = self.particles[0].f0.vth(*quad_pts_array)[1]

        #     absB0_at_quad = WeightedMassOperator.eval_quad(
        #         self.derham.Vh_fem['0'], self._absB0)

        #     self._det_df_at_quad = self.domain.jacobian_det(
        #         *quad_pts, squeeze_out=False)

        #     self._unit_b1_at_quad = WeightedMassOperator.eval_quad(
        #         self.derham.Vh_fem['1'], self._unit_b1)

        #     curl_norm_b_at_quad = WeightedMassOperator.eval_quad(
        #         self.derham.Vh_fem['2'], self._curl_norm_b)

        #     self._unit_b1_dot_curl_norm_b_at_quad = np.sum(
        #         p * q for p, q in zip(self._unit_b1_at_quad, curl_norm_b_at_quad))

        #     self._unit_b1_dot_curl_norm_b_at_quad /= self._det_df_at_quad
        #     self._unit_b1_dot_curl_norm_b_at_quad *= self._epsilon
        #     self._unit_b1_dot_curl_norm_b_at_quad *= u0_parallel_at_quad

        #     # precalculate constant 2 * f0.vth_perp² / B0 * f0.n for control MAT and VEC
        #     self._control_const = vth_perp**2 / absB0_at_quad * self._n0_at_quad

        #     # assemble the matrix (G_inv)(unit_b1 x)(G_inv)
        #     G_inv_at_quad = self.domain.metric_inv(
        #         *quad_pts, squeeze_out=False)

        #     self._G_inv_bx_G_inv_at_quad = [[np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad)],
        #                                     [np.zeros_like(self._n0_at_quad), np.zeros_like(
        #                                         self._n0_at_quad), np.zeros_like(self._n0_at_quad)],
        #                                     [np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad)]]

        #     for j in range(3):
        #         temp = (-self._unit_b1_at_quad[2]*G_inv_at_quad[1, j] + self._unit_b1_at_quad[1]*G_inv_at_quad[2, j],
        #                 self._unit_b1_at_quad[2]*G_inv_at_quad[0, j] -
        #                 self._unit_b1_at_quad[0]*G_inv_at_quad[2, j],
        #                 -self._unit_b1_at_quad[1]*G_inv_at_quad[0, j] + self._unit_b1_at_quad[0]*G_inv_at_quad[1, j])

        #         for i in range(3):
        #             self._G_inv_bx_G_inv_at_quad[i][j] = np.sum(
        #                 p * q for p, q in zip(G_inv_at_quad[i], temp[:]))

        #     # memory allocation of magnetic field at quadrature points
        #     self._b_at_quad = [np.zeros_like(self._n0_at_quad),
        #                        np.zeros_like(self._n0_at_quad),
        #                        np.zeros_like(self._n0_at_quad)]

        #     # memory allocation of parallel magnetic field at quadrature points
        #     self._B_para_at_quad = np.zeros_like(self._n0_at_quad)

        #     # memory allocation of gradient of parallel magnetic field at quadrature points
        #     self._grad_PBb_at_quad = (np.zeros_like(self._n0_at_quad),
        #                               np.zeros_like(self._n0_at_quad),
        #                               np.zeros_like(self._n0_at_quad))
        #     # memory allocation for temporary matrix
        #     self._temp = [[np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad)],
        #                   [np.zeros_like(self._n0_at_quad), np.zeros_like(
        #                       self._n0_at_quad), np.zeros_like(self._n0_at_quad)],
        #                   [np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad), np.zeros_like(self._n0_at_quad)]]

        #     # memory allocation for control VEC
        #     self._vec1 = np.zeros_like(self._n0_at_quad)
        #     self._vec2 = np.zeros_like(self._n0_at_quad)
        #     self._vec3 = np.zeros_like(self._n0_at_quad)

        # choose algorithm
        self._butcher = ButcherTableau(algo)
        # temp fix due to refactoring of ButcherTableau:
        self._butcher._a = np.diag(self._butcher.a, k=-1)
        self._butcher._a = np.array(list(self._butcher.a) + [0.0])

        # instantiate Pusher
        if u_space == "Hdiv":
            kernel = pusher_kernels_gc.push_gc_cc_J2_stage_Hdiv
        elif u_space == "H1vec":
            kernel = pusher_kernels_gc.push_gc_cc_J2_stage_H1vec
        else:
            raise ValueError(
                f'{u_space = } not valid, choose from "Hdiv" or "H1vec.',
            )

        args_kernel = (self.derham.args_derham,)

        self._pusher = Pusher(
            particles,
            kernel,
            args_kernel,
            self.domain.args_domain,
            alpha_in_kernel=1.0,
        )

        # temporary vectors to avoid memory allocation
        self._b_full1 = self._b_eq.space.zeros()
        self._b_full2 = self._E2T.codomain.zeros()
        self._u_new = u.space.zeros()
        self._Eu_new = self._EuT.codomain.zeros()
        self._u_temp1 = u.space.zeros()
        self._u_temp2 = u.space.zeros()
        self._Eu_temp = self._EuT.codomain.zeros()
        self._tmp1 = self._E0T.codomain.zeros()
        self._tmp2 = self._gradB1.space.zeros()
        self._tmp3 = self._E1T.codomain.zeros()

    def __call__(self, dt):
        un = self.feec_vars[0]

        # sum up total magnetic field b_full1 = b_eq + b_tilde (in-place)
        b_full = self._b_eq.copy(out=self._b_full1)

        if self._b is not None:
            self._b_full1 += self._b

        PBb = self._PB.dot(self._b, out=self._tmp1)
        grad_PBb = self.derham.grad.dot(PBb, out=self._tmp2)
        grad_PBb += self._gradB1

        Eb_full = self._E2T.dot(b_full, out=self._b_full2)
        Eb_full.update_ghost_regions()

        Egrad_PBb = self._E1T.dot(grad_PBb, out=self._tmp3)
        Egrad_PBb.update_ghost_regions()

        # perform accumulation (either with or without control variate)
        # if self.particles[0].control_variate:

        #     # evaluate magnetic field at quadrature points (in-place)
        #     WeightedMassOperator.eval_quad(self.derham.Vh_fem['2'], self._b_full2,
        #                                    out=[self._b_at_quad[0], self._b_at_quad[1], self._b_at_quad[2]])

        #     # evaluate B_parallel
        #     self._B_para_at_quad = np.sum(
        #         p * q for p, q in zip(self._unit_b1_at_quad, self._b_at_quad))
        #     self._B_para_at_quad += self._unit_b1_dot_curl_norm_b_at_quad

        #     # evaluate grad B_parallel
        #     WeightedMassOperator.eval_quad(self.derham.Vh_fem['1'], self._tmp3,
        #                                    out=[self._grad_PBb_at_quad[0], self._grad_PBb_at_quad[1], self._grad_PBb_at_quad[2]])

        #     # assemble temp = (B x)(G_inv)(unit_b1 x)(G_inv)
        #     for i in range(3):
        #         self._temp[0][i] = -self._b_at_quad[2]*self._G_inv_bx_G_inv_at_quad[1][i] + \
        #             self._b_at_quad[1]*self._G_inv_bx_G_inv_at_quad[2][i]
        #         self._temp[1][i] = +self._b_at_quad[2]*self._G_inv_bx_G_inv_at_quad[0][i] - \
        #             self._b_at_quad[0]*self._G_inv_bx_G_inv_at_quad[2][i]
        #         self._temp[2][i] = -self._b_at_quad[1]*self._G_inv_bx_G_inv_at_quad[0][i] + \
        #             self._b_at_quad[0]*self._G_inv_bx_G_inv_at_quad[1][i]

        #     # assemble (temp)(grad B_parallel) / B_star_para * 2 * f0.vth_perp² / B0 * f0.n
        #     self._vec1[:, :, :] = np.sum(p * q for p, q in zip(self._temp[0][:], self._grad_PBb_at_quad)) * \
        #         self._control_const * self._coupling_vec / self._B_para_at_quad
        #     self._vec2[:, :, :] = np.sum(p * q for p, q in zip(self._temp[1][:], self._grad_PBb_at_quad)) * \
        #         self._control_const * self._coupling_vec / self._B_para_at_quad
        #     self._vec3[:, :, :] = np.sum(p * q for p, q in zip(self._temp[2][:], self._grad_PBb_at_quad)) * \
        #         self._control_const * self._coupling_vec / self._B_para_at_quad

        # save old u
        _u_new = un.copy(out=self._u_new)
        _u_temp = un.copy(out=self._u_temp1)

        # save old marker positions
        self.particles[0].markers[
            ~self.particles[0].holes,
            11:14,
        ] = self.particles[0].markers[~self.particles[0].holes, 0:3]

        for stage in range(self._butcher.n_stages):
            # accumulate RHS
            # if self.particles[0].control_variate:
            #     self._ACC.accumulate(self.particles[0], self._epsilon,
            #                          Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
            #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
            #                          self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
            #                          self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
            #                          Egrad_PBb[0]._data, Egrad_PBb[1]._data, Egrad_PBb[2]._data,
            #                          self._space_key_int, self._coupling_mat, self._coupling_vec, 0.,
            #                          control_vec=[self._vec1, self._vec2, self._vec3])
            # else:
            #     self._ACC.accumulate(self.particles[0], self._epsilon,
            #                          Eb_full[0]._data, Eb_full[1]._data, Eb_full[2]._data,
            #                          self._unit_b1[0]._data, self._unit_b1[1]._data, self._unit_b1[2]._data,
            #                          self._unit_b2[0]._data, self._unit_b2[1]._data, self._unit_b2[2]._data,
            #                          self._curl_norm_b[0]._data, self._curl_norm_b[1]._data, self._curl_norm_b[2]._data,
            #                          Egrad_PBb[0]._data, Egrad_PBb[1]._data, Egrad_PBb[2]._data,
            #                          self._space_key_int, self._coupling_mat, self._coupling_vec, 0.)

            self._ACC(
                self._epsilon,
                Eb_full[0]._data,
                Eb_full[1]._data,
                Eb_full[2]._data,
                self._unit_b1[0]._data,
                self._unit_b1[1]._data,
                self._unit_b1[2]._data,
                self._unit_b2[0]._data,
                self._unit_b2[1]._data,
                self._unit_b2[2]._data,
                self._curl_norm_b[0]._data,
                self._curl_norm_b[1]._data,
                self._curl_norm_b[2]._data,
                Egrad_PBb[0]._data,
                Egrad_PBb[1]._data,
                Egrad_PBb[2]._data,
                self._space_key_int,
                self._coupling_mat,
                self._coupling_vec,
                self._boundary_cut_e1,
            )

            # push particles
            Eu = self._EuT.dot(_u_temp, out=self._Eu_temp)
            Eu.update_ghost_regions()

            self._pusher.kernel(
                dt,
                stage,
                self.particles[0].args_markers,
                self.domain.args_domain,
                self.derham.args_derham,
                self._epsilon,
                Eb_full[0]._data,
                Eb_full[1]._data,
                Eb_full[2]._data,
                self._unit_b1[0]._data,
                self._unit_b1[1]._data,
                self._unit_b1[2]._data,
                self._unit_b2[0]._data,
                self._unit_b2[1]._data,
                self._unit_b2[2]._data,
                self._curl_norm_b[0]._data,
                self._curl_norm_b[1]._data,
                self._curl_norm_b[2]._data,
                Eu[0]._data,
                Eu[1]._data,
                Eu[2]._data,
                self._butcher.a,
                self._butcher.b,
                self._butcher.c,
                self._boundary_cut_e1,
            )

            self.particles[0].mpi_sort_markers()

            # solve linear system for updated u coefficients
            _ku = self._solver.dot(self._ACC.vectors[0], out=self._u_temp2)

            # calculate u^{n+1}_k
            _u_temp = un.copy(out=self._u_temp1)
            _u_temp += _ku * dt * self._butcher.a[stage]

            # calculate u^{n+1}
            _u_new += _ku * dt * self._butcher.b[stage]

            if self._info and self._rank == 0:
                print("Stage:", stage)
                print(
                    "Status     for CurrentCoupling5DGradB:",
                    self._solver._info["success"],
                )
                print(
                    "Iterations for CurrentCoupling5DGradB:",
                    self._solver._info["niter"],
                )

            # clear the buffer
            if stage == self._butcher.n_stages - 1:
                self.particles[0].markers[
                    ~self.particles[0].holes,
                    11:-1,
                ] = 0.0

        # write new coeffs into Propagator.variables
        (max_du,) = self.feec_vars_update(_u_new)

        # update_weights
        if self.particles[0].control_variate:
            self.particles[0].update_weights()

        if self._info and self._rank == 0:
            print("Maxdiff up for CurrentCoupling5DGradB:", max_du)
            print()
