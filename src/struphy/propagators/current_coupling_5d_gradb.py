"Particle and FEEC variables are updated."

import logging
from dataclasses import dataclass
from typing import Literal

import cunumpy as xp
from feectools.ddm.mpi import mpi as MPI
from feectools.linalg.solvers import inverse
from line_profiler import profile

from struphy.feec import preconditioner
from struphy.io.options import LiteralOptions, OptionsBase
from struphy.linear_algebra.solver import DiscreteGradientSolverParameters, SolverParameters
from struphy.models.variables import FEECVariable, PICVariable
from struphy.ode.utils import ButcherTableau
from struphy.pic import utilities_kernels
from struphy.pic.accumulation import accum_kernels_gc
from struphy.pic.accumulation.filter import FilterParameters
from struphy.pic.accumulation.particles_to_grid import Accumulator, AccumulatorVector
from struphy.pic.pushing import pusher_kernels_gc
from struphy.propagators.base import Propagator
from struphy.utils.pyccel import Pyccelkernel
from struphy.utils.utils import check_option

logger = logging.getLogger("struphy")


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
        """Container for variables advanced by :class:`CurrentCoupling5DGradB`.

        Attributes
        ----------
        u : FEECVariable
            Fluid-like FEEC variable in one of ``"Hcurl"``, ``"Hdiv"``, or
            ``"H1vec"``.
        energetic_ions : PICVariable
            Energetic-ion particle variable in ``"Particles5D"`` space.
        """

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

    def __init__(self, b_tilde: FEECVariable = None):
        """
        Parameters
        ----------
        b_tilde : FEECVariable, default=None
            Magnetic perturbation 1-form (``"Hcurl"`` space) entering the grad-B
            coupling term. If ``None``, only the equilibrium field is used.
        """
        self.variables = self.Variables()
        self.b_tilde = b_tilde

    @dataclass(repr=False)
    class Options(OptionsBase):
        """Configuration options for :class:`CurrentCoupling5DGradB`.

        Parameters
        ----------
        ep_scale : float, default=1.0
            Scaling factor applied in particle accumulation.

        algo : {"discrete_gradient", "explicit"}, default="explicit"
            Time integration algorithm.

            - ``"explicit"``: explicit Runge-Kutta update with ``butcher``.
            - ``"discrete_gradient"``: nonlinear discrete-gradient update.

        butcher : ButcherTableau, default=None
            Butcher tableau used in explicit mode. If ``None`` and
            ``algo="explicit"``, defaults to ``ButcherTableau()``.

        u_space : LiteralOptions.OptsVecSpace, default="Hdiv"
            FEEC space used for the unknown ``u`` variable.

        solver : LiteralOptions.OptsSymmSolver, default="pcg"
            Symmetric iterative solver used for mass-matrix inversions.

        precond : LiteralOptions.OptsMassPrecond, default="MassMatrixPreconditioner"
            Preconditioner used with the mass matrix.

        solver_params : SolverParameters, default=None
            Iterative-solver controls. If ``None``, defaults to
            ``SolverParameters()``.

        filter_params : FilterParameters, default=None
            Particle-to-grid filtering parameters used by the accumulator.
            If ``None``, defaults to ``FilterParameters()``.

        dg_solver_params : DiscreteGradientSolverParameters, default=None
            Nonlinear solver controls for discrete-gradient mode. If ``None``
            and ``algo="discrete_gradient"``, defaults to
            ``DiscreteGradientSolverParameters()``.
        """

        # specific literals
        OptsAlgo = Literal[
            "discrete_gradient",
            "explicit",
        ]
        # propagator options
        ep_scale: float = 1.0
        algo: OptsAlgo = "explicit"
        butcher: ButcherTableau = None
        u_space: LiteralOptions.OptsVecSpace = "Hdiv"
        solver: LiteralOptions.OptsSymmSolver = "pcg"
        precond: LiteralOptions.OptsMassPrecond = "MassMatrixPreconditioner"
        solver_params: SolverParameters = None
        filter_params: FilterParameters = None
        dg_solver_params: DiscreteGradientSolverParameters = None

        def __post_init__(self):
            # checks
            check_option(self.algo, self.OptsAlgo)
            check_option(self.u_space, LiteralOptions.OptsVecSpace)
            check_option(self.solver, LiteralOptions.OptsSymmSolver)
            check_option(self.precond, LiteralOptions.OptsMassPrecond)
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
        self._options = new
        logger.info(f"\nNew options for propagator '{self.__class__.__name__}':\n{self._options}")

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
            recycle=self.options.solver_params.recycle,
        )
        # magnetic equilibrium field
        unit_b1 = self.projected_equil.unit_b1
        curl_unit_b2 = self.projected_equil.curl_unit_b2
        self._b2 = self.projected_equil.b2
        gradB1 = self.projected_equil.gradB1
        absB0 = self.projected_equil.absB0

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
                Pyccelkernel(accum_kernels_gc.cc_lin_mhd_5d_gradB),
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
                curl_unit_b2[0]._data,
                curl_unit_b2[1]._data,
                curl_unit_b2[2]._data,
                self._grad_PB_b[0]._data,
                self._grad_PB_b[1]._data,
                self._grad_PB_b[2]._data,
                gradB1[0]._data,
                gradB1[1]._data,
                gradB1[2]._data,
                self._u_form_int,
            )

            # define Pusher
            if self.options.u_space == "Hdiv":
                self._pusher_kernel = pusher_kernels_gc.push_gc_cc_J2_stage_Hdiv
            elif self.options.u_space == "H1vec":
                self._pusher_kernel = pusher_kernels_gc.push_gc_cc_J2_stage_H1vec
            else:
                raise ValueError(
                    f'{self.options.u_space  =} not valid, choose from "Hdiv" or "H1vec.',
                )

            # temp fix due to refactoring of ButcherTableau:

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
                curl_unit_b2[0]._data,
                curl_unit_b2[1]._data,
                curl_unit_b2[2]._data,
                self._u_temp[0]._data,
                self._u_temp[1]._data,
                self._u_temp[2]._data,
                self.options.butcher.a_stage,
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
                self.b_tilde.spline.vector[0]._data,
                self.b_tilde.spline.vector[1]._data,
                self.b_tilde.spline.vector[2]._data,
                self._b2[0]._data,
                self._b2[1]._data,
                self._b2[2]._data,
                unit_b1[0]._data,
                unit_b1[1]._data,
                unit_b1[2]._data,
                curl_unit_b2[0]._data,
                curl_unit_b2[1]._data,
                curl_unit_b2[2]._data,
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
                curl_unit_b2[0]._data,
                curl_unit_b2[1]._data,
                curl_unit_b2[2]._data,
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
                curl_unit_b2[0]._data,
                curl_unit_b2[1]._data,
                curl_unit_b2[2]._data,
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

        b_full += self.b_tilde.spline.vector
        b_full.update_ghost_regions()

        if self.options.algo == "explicit":
            PB_b = self._PB.dot(self.b_tilde.spline.vector, out=self._PB_b)
            grad_PB_b = self.derham.grad.dot(PB_b, out=self._grad_PB_b)
            grad_PB_b.update_ghost_regions()

            # save old u
            u_new = un.copy(out=self._u_new)
            u_temp = un.copy(out=self._u_temp)

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
                u_temp += ku * dt * self.options.butcher.a_stage[stage]

                u_temp.update_ghost_regions()

                # calculate u^{n+1}
                u_new += ku * dt * self.options.butcher.b[stage]
                
                u_new.update_ghost_regions()

                if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
                    logger.info(f"Stage: {stage}")
                    logger.info(f"Status     for CurrentCoupling5DGradB: {info['success']}")
                    logger.info(f"Iterations for CurrentCoupling5DGradB: {info['niter']}")
                    logger.info("")

            # update u coefficients
            diffs = self.update_feec_variables(u=u_new)

            # clear the buffer
            markers[:, first_init_idx:-2] = 0.0

            # update_weights
            if self.variables.energetic_ions.species.weights_params.control_variate:
                particles.update_weights()

            if self.options.solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
                logger.info(f"Maxdiff up for CurrentCoupling5DGradB: {diffs['u']}")
                logger.info("")

        else:
            # total number of markers
            n_mks_tot = particles.Np

            # relaxation factor
            alpha = self.options.dg_solver_params.relaxation_factor

            # eval parallel tilde b and its gradient
            PB_b = self._PB.dot(self.b_tilde.spline.vector, out=self._PB_b)
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
                    logger.info(f"# of iteration: {iter_num}")

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
                    (markers[~holes, :3] - markers[~holes, first_init_idx : first_init_idx + 3]) ** 2,
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
                    xp.abs(markers[~holes, 0:3] - markers[~holes, first_free_idx : first_free_idx + 3]),
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
                        logger.info(f"converged diff: {diff}")
                        logger.info(f"converged e_diff: {e_diff}")

                    if particles.mpi_comm is not None:
                        particles.mpi_comm.Barrier()
                    break

                else:
                    if self.options.dg_solver_params.verbose and MPI.COMM_WORLD.Get_rank() == 0:
                        logger.info(f"not converged diff: {diff}")
                        logger.info(f"not converged e_diff: {e_diff}")

                if iter_num == self.options.dg_solver_params.maxiter:
                    if self.options.dg_solver_params.info and MPI.COMM_WORLD.Get_rank() == 0:
                        logger.info(
                            f"{iter_num =}, maxiter={self.options.dg_solver_params.maxiter} reached! diff: {diff}, e_diff: {e_diff}",
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
                logger.info(f"Maxdiff up for CurrentCoupling5DGradB: {diffs['u']}")
                logger.info("")
