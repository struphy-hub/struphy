"Accelerated particle pushing."

import logging

import cunumpy
import numpy as np
from cunumpy import PyccelKernel
from feectools.ddm.mpi import mpi as MPI
from line_profiler import profile
from scope_profiler import ProfileManager

from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.pic.base import Particles
from struphy.pic.pushing.pusher_kernels_cuda import (
    SUPPORTED_GENERAL_KIND_MAPS,
    push_bxu_H1vec_general_gpu,
    push_bxu_Hcurl_general_gpu,
    push_bxu_Hdiv_general_gpu,
    push_deterministic_diffusion_stage_general_gpu,
    push_eta_rk_periodic_gpu,
    push_eta_stage_cuboid_gpu,
    push_eta_stage_general_gpu,
    push_pc_eta_stage_H1vec_general_gpu,
    push_pc_eta_stage_Hcurl_general_gpu,
    push_pc_eta_stage_Hdiv_general_gpu,
    push_pc_GXu_full_general_gpu,
    push_pc_GXu_general_gpu,
    push_random_diffusion_stage_gpu,
    push_v_with_efield_cuboid_gpu,
    push_v_with_efield_general_gpu,
    push_vxb_analytic_general_gpu,
    push_vxb_implicit_general_gpu,
    push_weights_with_efield_lin_va_general_gpu,
)
from struphy.pic.pushing.eval_kernels_gc_cuda import driftkinetic_hamiltonian_gpu
from struphy.pic.pushing.pusher_kernels_sph_cuda import (
    push_v_sph_pressure_gpu,
    push_v_sph_pressure_ideal_gas_gpu,
    push_v_viscosity_gpu,
)
from struphy.pic.pushing.pusher_kernels_gc_cuda import (
    push_gc_bxEstar_discrete_gradient_1st_order_gpu,
    push_gc_bxEstar_explicit_multistage_general_gpu,
    push_gc_Bstar_discrete_gradient_1st_order_gpu,
    push_gc_Bstar_explicit_multistage_general_gpu,
)

logger = logging.getLogger("struphy")


def _kernel_name(kernel) -> str:
    """Name of a pyccelized kernel, which can be a bare pyccel function or a PyccelKernel."""
    return getattr(kernel, "name", None) or getattr(kernel, "__name__", type(kernel).__name__)


class Pusher:
    r"""
    Class for solving particle ODEs

    .. math::

        \dot{\mathbf Z}_p(t) = \mathbf U(t, \mathbf Z_p(t))\,,

    for each marker :math:`p` in :class:`~struphy.pic.base.Particles` class,
    where :math:`\mathbf Z_p` are the marker coordinates and
    the vector field :math:`\mathbf U` can contain discrete :class:`~struphy.feec.psydac_derham.Derham` splines
    and metric coefficients from accelerated :mod:`~struphy.geometry.evaluation_kernels`.

    The solve is MPI distributed and can handle multi-stage Runge-Kutta methods
    for any :class:`~struphy.ode.utils.ButcherTableau`
    as well as iterative nonlinear methods.

    The particle push is performed via accelerated :mod:`~struphy.pic.pushing.pusher_kernels`
    or :mod:`~struphy.pic.pushing.pusher_kernels_gc` for guiding-center models.

    Notes
    -----

    For iterative methods with iteration index :math:`k`, spline evaluations at positions
    :math:`\alpha_i \eta_{p,i}^{n+1,k} + (1 - \alpha_i) \eta_{p,i}^n`
    for :math:`i=1, 2, 3` and different :math:`\alpha_i \in [0,1]`
    need particle MPI sorting in between.
    This requires calling dedicated ``eval_kernels`` during the iteration. Here are some
    rules to follow for iterative solvers:

    * Spline/geometry evaluations at :math:`\boldsymbol \eta^n_p` can be be done via ``init_kernels``.
    * Pusher ``kernel`` and ``eval_kernels`` can perform evaluations at arbitrary weighted averages :math:`\eta_{p,i} = \alpha_i \eta_{p,i}^{n+1,k} + (1 - \alpha_i) \eta_{p,i}^n`, for :math:`i=1,2,3`.
    * MPI sorting is done automatically before kernel calls according to the specified values :math:`\alpha_i` for each kernel.

    Parameters
    ----------
    particles : Particles
        Particles object holding the markers to push.

    kernel : pyccelized function
        The pusher kernel.

    args_kernel : tuple
        Optional arguments passed to the kernel.

    args_domain : DomainArguments
        Mapping infos.

    alpha_in_kernel: float | int | tuple | list
        For i=0,1,2, the spline/geometry evaluations in kernel are at
        alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, buffer_idx + i].
        If float or int or then alpha = (alpha, alpha, alpha).
        alpha must be between 0 and 1.
        alpha[i]=0 means that evaluation is at the initial positions (time n),
        stored at markers[:, buffer_idx + i].

    init_kernels : dict
        Keys: initialization kernels for spline/ SPH evaluations at time n (initial state).
        Values: optional arguments.

    eval_kernels : dict
        Keys: evaluation kernels for splines before the pusher kernel is called.
        Values: optional arguments and weighting parameters alpha for
        sorting (before evaluation), according to
        alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, buffer_idx + i] for i=0,1,2.
        alpha must be between 0 and 1, see :meth:`~struphy.pic.base.Particles.mpi_sort_markers`.

    n_stages : int
        Number of stages of the pusher (e.g. 4 for RK4)

    maxiter : int
        Maximum number of iterations (=1 for explicit pushers).

    tol : float
        Iteration terminates when residual<tol.

    mpi_sort : str
        When to do MPI sorting:
        * None : no sorting at all.
        * each : sort markers after each stage.
        * last : sort markers after last stage.
    """

    def __init__(
        self,
        particles: Particles,
        kernel: PyccelKernel,
        args_kernel: tuple,
        args_domain: DomainArguments,
        *,
        alpha_in_kernel: float | int | tuple | list,
        init_kernels: list = [],
        eval_kernels: list = [],
        n_stages: int = 1,
        maxiter: int = 1,
        tol: float = 1.0e-8,
        mpi_sort: str = None,
    ):
        self._particles = particles
        assert isinstance(kernel, PyccelKernel), f"{kernel} is not of type PyccelKernel"
        self._kernel = kernel
        self._newton = "newton" in kernel.name
        self._args_kernel = args_kernel
        self._args_domain = args_domain

        # hand-written CUDA replacement for push_eta_stage on a Cuboid domain
        # (constant, diagonal Jacobian -> no spline evaluation needed at all)
        self._gpu_eta_cuboid = cunumpy.cupy_backend and kernel.name == "push_eta_stage" and args_domain.kind_map == 10
        if self._gpu_eta_cuboid:
            l1, r1, l2, r2, l3, r3 = (float(p) for p in args_domain.params[:6])
            self._gpu_eta_cuboid_scale = (1.0 / (r1 - l1), 1.0 / (r2 - l2), 1.0 / (r3 - l3))

        # general (non-Cuboid) CUDA replacement for push_eta_stage: evaluates
        # DF(eta) per marker instead of assuming it's constant, so it covers
        # any domain in SUPPORTED_GENERAL_KIND_MAPS -- currently Cuboid and
        # Colella, see pusher_kernels_cuda.py. Only used when the more
        # specialized _gpu_eta_cuboid path above doesn't already apply.
        self._gpu_eta_general = (
            cunumpy.cupy_backend
            and kernel.name == "push_eta_stage"
            and not self._gpu_eta_cuboid
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_eta_general:
            import cupy as cp

            self._gpu_eta_general_kind_map = int(args_domain.kind_map)
            self._gpu_eta_general_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)

        # determines the evaluation points for kernel
        self._alpha_in_kernel = alpha_in_kernel
        self._n_stages = n_stages
        self._maxiter = maxiter
        self._tol = tol
        self._mpi_sort = mpi_sort

        # prepare and check init_kernels
        for ker_args in init_kernels:
            assert len(ker_args) == 4
            column_nr = ker_args[1]
            comps = ker_args[2]

            # check marker array column number
            assert isinstance(comps, np.ndarray)
            assert column_nr + comps.size < particles.n_cols, (
                f"{column_nr + comps.size} not smaller than {particles.n_cols =}; not enough columns in marker array !!"
            )

        # prepare and check eval_kernels
        for ker_args in eval_kernels:
            assert len(ker_args) == 5
            column_nr = ker_args[2]
            comps = ker_args[3]

            # check marker array column number
            assert isinstance(comps, np.ndarray)
            assert column_nr + comps.size < particles.n_cols, (
                f"{column_nr + comps.size} not smaller than {particles.n_cols =}; not enough columns in marker array !!"
            )

        self._init_kernels = init_kernels
        self._eval_kernels = eval_kernels

        # profiling region names (cached, they are looked up on every call)
        self._region_name = "pusher: " + self.kernel.name
        self._kernel_region_names = {}

        # marker-row-indexed, so they live on the same backend as the markers
        # (device under CuPy) -- see Particles._allocate_marker_array
        self._residuals = cunumpy.zeros(self.particles.markers.shape[0])
        self._converged_loc = self._residuals == 1.0
        self._not_converged_loc = self._residuals == 0.0

        if self.particles.sorting_boxes is not None:
            self._box_comm = self.particles.sorting_boxes.communicate
        else:
            self._box_comm = False

        # whole-push GPU-resident fast path: on top of _gpu_eta_cuboid, also
        # requires an all-periodic bc (so apply_kinetic_bc reduces to wrap +
        # shift bookkeeping, which push_eta_rk_periodic_gpu fuses in) and no
        # MPI / iterative-solver / eval-kernel machinery (none of which
        # push_eta uses, but other Pusher users might).
        self._gpu_eta_cuboid_periodic = (
            self._gpu_eta_cuboid
            and all(b == "periodic" for b in self.particles.bc)
            and not self._init_kernels
            and not self._eval_kernels
            and self.particles.mpi_comm is None
            and self._maxiter == 1
            and not self._newton
        )

        # hand-written CUDA replacement for push_v_with_efield's per-marker
        # math on a Cuboid domain. Unlike the whole-push fast path below, this
        # is unconditional on MPI/bc/maxiter -- it only swaps out the inner
        # kernel call (see the "push markers" branch in _push(), mirroring
        # how _gpu_eta_cuboid is used there), so it stays correct alongside
        # unmodified apply_kinetic_bc/mpi_sort_markers/update_holes for
        # multi-rank runs, exactly like _gpu_eta_cuboid already does for
        # push_eta_stage.
        self._gpu_v_efield_cuboid = (
            cunumpy.cupy_backend and kernel.name == "push_v_with_efield" and args_domain.kind_map == 10
        )
        if self._gpu_v_efield_cuboid:
            import cupy as cp

            l1, r1, l2, r2, l3, r3 = (float(p) for p in args_domain.params[:6])
            self._gpu_v_efield_scale = (1.0 / (r1 - l1), 1.0 / (r2 - l2), 1.0 / (r3 - l3))

            args_derham, e1_1, e1_2, e1_3, const = args_kernel
            self._gpu_v_efield_const = float(const)
            self._gpu_v_efield_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_v_efield_starts = tuple(int(s) for s in args_derham.starts)
            # knot vectors are tiny host arrays; cache them on the device once
            self._gpu_v_efield_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_v_efield_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_v_efield_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            # FE coefficients are already device-resident CuPy arrays under the
            # CuPy backend (StencilVector allocates via cunumpy's xp) and are
            # never reassigned after PushVinEfield.allocate() builds them, so
            # these references stay valid and need no per-call transfer.
            self._gpu_v_efield_e1_1 = e1_1
            self._gpu_v_efield_e1_2 = e1_2
            self._gpu_v_efield_e1_3 = e1_3

        # general (non-Cuboid) CUDA replacement for push_v_with_efield: same
        # B-spline evaluation as _gpu_v_efield_cuboid, but with DF(eta)
        # evaluated per marker instead of assumed constant-diagonal -- see
        # _gpu_eta_general above.
        self._gpu_v_efield_general = (
            cunumpy.cupy_backend
            and kernel.name == "push_v_with_efield"
            and not self._gpu_v_efield_cuboid
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_v_efield_general:
            import cupy as cp

            self._gpu_v_efield_general_kind_map = int(args_domain.kind_map)
            self._gpu_v_efield_general_params = cp.asarray(
                np.asarray(args_domain.params, dtype=float), dtype=cp.float64
            )

            args_derham, e1_1, e1_2, e1_3, const = args_kernel
            self._gpu_v_efield_general_const = float(const)
            self._gpu_v_efield_general_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_v_efield_general_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_v_efield_general_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_v_efield_general_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_v_efield_general_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            # FE coefficients are already device-resident under CuPy, see
            # _gpu_v_efield_cuboid above.
            self._gpu_v_efield_general_e1_1 = e1_1
            self._gpu_v_efield_general_e1_2 = e1_2
            self._gpu_v_efield_general_e1_3 = e1_3

        # whole-push GPU-resident fast path: on top of _gpu_v_efield_cuboid,
        # additionally bypasses the per-call reset/apply_kinetic_bc/
        # update_holes machinery entirely (this kernel never touches position
        # or holes/ghost columns, so that machinery is a no-op for it -- but
        # only provably so under the same conditions as
        # _gpu_eta_cuboid_periodic: no MPI, since mpi_sort_markers does real
        # host-side communication we can't just skip).
        self._gpu_v_efield_cuboid_wholepush = (
            self._gpu_v_efield_cuboid
            and all(b == "periodic" for b in self.particles.bc)
            and not init_kernels
            and not eval_kernels
            and self.particles.mpi_comm is None
            and maxiter == 1
            and not self._newton
            and n_stages == 1
        )

        # general (non-Cuboid) CUDA replacement for push_vxb_analytic /
        # push_vxb_implicit, sharing the same B-spline/geometry evaluation as
        # _gpu_v_efield_general above (2-form instead of 1-form field).
        self._gpu_vxb_general = (
            cunumpy.cupy_backend
            and kernel.name
            in (
                "push_vxb_analytic",
                "push_vxb_implicit",
            )
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_vxb_general:
            import cupy as cp

            self._gpu_vxb_general_analytic = kernel.name == "push_vxb_analytic"
            self._gpu_vxb_general_kind_map = int(args_domain.kind_map)
            self._gpu_vxb_general_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)

            args_derham, b2_1, b2_2, b2_3 = args_kernel
            self._gpu_vxb_general_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_vxb_general_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_vxb_general_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_vxb_general_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_vxb_general_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            # FE coefficients are already device-resident under CuPy, see
            # _gpu_v_efield_cuboid above. Unlike push_v_with_efield's e1_*,
            # b2_* here can be *reassigned* between calls (PushVxB.allocate()
            # rebuilds self._b_full = b2_0 (+ b2_var) once per allocate(), but
            # __call__ does not touch the underlying StencilVector objects
            # again after that), so caching the references once here is
            # still valid for the propagator's lifetime.
            self._gpu_vxb_general_b2_1 = b2_1
            self._gpu_vxb_general_b2_2 = b2_2
            self._gpu_vxb_general_b2_3 = b2_3

        # general (non-Cuboid) CUDA replacement for push_bxu_{Hdiv,Hcurl,H1vec},
        # sharing the same B-field (2-form) evaluation as _gpu_vxb_general;
        # only the U-field's FEEC space (and therefore its evaluation/metric
        # handling) differs between the three.
        self._gpu_bxu_general = (
            cunumpy.cupy_backend
            and kernel.name
            in (
                "push_bxu_Hdiv",
                "push_bxu_Hcurl",
                "push_bxu_H1vec",
            )
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_bxu_general:
            import cupy as cp

            self._gpu_bxu_general_variant = kernel.name
            self._gpu_bxu_general_kind_map = int(args_domain.kind_map)
            self._gpu_bxu_general_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)

            args_derham, b2_1, b2_2, b2_3, u_1, u_2, u_3, boundary_cut = args_kernel
            self._gpu_bxu_general_boundary_cut = float(boundary_cut)
            self._gpu_bxu_general_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_bxu_general_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_bxu_general_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_bxu_general_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_bxu_general_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            # FE coefficients are device-resident under CuPy, see
            # _gpu_v_efield_cuboid above.
            self._gpu_bxu_general_b2_1 = b2_1
            self._gpu_bxu_general_b2_2 = b2_2
            self._gpu_bxu_general_b2_3 = b2_3
            self._gpu_bxu_general_u_1 = u_1
            self._gpu_bxu_general_u_2 = u_2
            self._gpu_bxu_general_u_3 = u_3

        # general (non-Cuboid) CUDA replacement for push_pc_GXu_full /
        # push_pc_GXu: the propagator (PressureCoupling6D) always builds the
        # full 9-array args_kernel regardless of which of the two it uses
        # (push_pc_GXu's CPU kernel also takes all 9, only 6 are read) -- so
        # both branches cache the same 9 g_ij arrays and the *_full variant
        # is picked purely by kernel.name.
        self._gpu_pc_gxu_general = (
            cunumpy.cupy_backend
            and kernel.name
            in (
                "push_pc_GXu_full",
                "push_pc_GXu",
            )
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_pc_gxu_general:
            import cupy as cp

            self._gpu_pc_gxu_general_full = kernel.name == "push_pc_GXu_full"
            self._gpu_pc_gxu_general_kind_map = int(args_domain.kind_map)
            self._gpu_pc_gxu_general_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)

            args_derham, g11, g12, g13, g21, g22, g23, g31, g32, g33 = args_kernel
            self._gpu_pc_gxu_general_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_pc_gxu_general_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_pc_gxu_general_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_pc_gxu_general_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_pc_gxu_general_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            self._gpu_pc_gxu_general_g = (g11, g12, g13, g21, g22, g23, g31, g32, g33)

        # general (non-Cuboid) CUDA replacement for
        # push_pc_eta_stage_{Hcurl,Hdiv,H1vec}: a variant of _gpu_eta_general
        # with an extra U-field vector contribution added to the eta rate.
        self._gpu_pc_eta_general = (
            cunumpy.cupy_backend
            and kernel.name
            in (
                "push_pc_eta_stage_Hcurl",
                "push_pc_eta_stage_Hdiv",
                "push_pc_eta_stage_H1vec",
            )
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_pc_eta_general:
            import cupy as cp

            self._gpu_pc_eta_general_variant = kernel.name
            self._gpu_pc_eta_general_kind_map = int(args_domain.kind_map)
            self._gpu_pc_eta_general_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)

            args_derham, u_1, u_2, u_3, use_perp_model = args_kernel[:5]
            self._gpu_pc_eta_general_use_perp_model = bool(use_perp_model)
            self._gpu_pc_eta_general_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_pc_eta_general_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_pc_eta_general_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_pc_eta_general_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_pc_eta_general_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            self._gpu_pc_eta_general_u_1 = u_1
            self._gpu_pc_eta_general_u_2 = u_2
            self._gpu_pc_eta_general_u_3 = u_3

        # general (non-Cuboid) CUDA replacement for
        # push_weights_with_efield_lin_va. Unlike the FE-coefficient
        # arguments cached above, f0_values is recomputed by the caller
        # every step, in place (self._f0_values[:] = ...) -- so the
        # reference can be cached once here like the other FE-coefficient
        # arguments (see push_weights_with_efield_lin_va_general_gpu's
        # docstring).
        self._gpu_weights_efield_general = (
            cunumpy.cupy_backend
            and kernel.name == "push_weights_with_efield_lin_va"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_weights_efield_general:
            import cupy as cp

            self._gpu_weights_efield_general_kind_map = int(args_domain.kind_map)
            self._gpu_weights_efield_general_params = cp.asarray(
                np.asarray(args_domain.params, dtype=float), dtype=cp.float64
            )

            args_derham, e1_1, e1_2, e1_3, f0_values, kappa, vth = args_kernel
            self._gpu_weights_efield_general_kappa = float(kappa)
            self._gpu_weights_efield_general_vth = float(vth)
            self._gpu_weights_efield_general_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_weights_efield_general_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_weights_efield_general_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_weights_efield_general_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_weights_efield_general_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            self._gpu_weights_efield_general_e1_1 = e1_1
            self._gpu_weights_efield_general_e1_2 = e1_2
            self._gpu_weights_efield_general_e1_3 = e1_3
            self._gpu_weights_efield_general_f0_values = f0_values

        # general (non-Cuboid) CUDA replacement for
        # push_deterministic_diffusion_stage.
        self._gpu_det_diffusion_general = (
            cunumpy.cupy_backend
            and kernel.name == "push_deterministic_diffusion_stage"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_det_diffusion_general:
            import cupy as cp

            self._gpu_det_diffusion_general_kind_map = int(args_domain.kind_map)
            self._gpu_det_diffusion_general_params = cp.asarray(
                np.asarray(args_domain.params, dtype=float), dtype=cp.float64
            )

            args_derham, pi_u, pi_grad_u1, pi_grad_u2, pi_grad_u3, diffusion_coeff = args_kernel[:6]
            self._gpu_det_diffusion_general_coeff = float(diffusion_coeff)
            self._gpu_det_diffusion_general_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_det_diffusion_general_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_det_diffusion_general_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_det_diffusion_general_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_det_diffusion_general_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            self._gpu_det_diffusion_general_pi_u = pi_u
            self._gpu_det_diffusion_general_pi_grad_u1 = pi_grad_u1
            self._gpu_det_diffusion_general_pi_grad_u2 = pi_grad_u2
            self._gpu_det_diffusion_general_pi_grad_u3 = pi_grad_u3

        # CUDA replacement for push_random_diffusion_stage: domain-independent
        # (pure additive noise, no geometry), so no kind_map restriction.
        self._gpu_random_diffusion = cunumpy.cupy_backend and kernel.name == "push_random_diffusion_stage"
        if self._gpu_random_diffusion:
            noise, diffusion_coeff = args_kernel[0], args_kernel[1]
            self._gpu_random_diffusion_coeff = float(diffusion_coeff)
            self._gpu_random_diffusion_noise = noise

        # general (non-Cuboid) CUDA replacement for
        # push_gc_bxEstar_explicit_multistage (5D guiding-center pusher).
        self._gpu_gc_bxestar_general = (
            cunumpy.cupy_backend
            and kernel.name == "push_gc_bxEstar_explicit_multistage"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_gc_bxestar_general:
            import cupy as cp

            self._gpu_gc_bxestar_kind_map = int(args_domain.kind_map)
            self._gpu_gc_bxestar_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            (
                args_derham,
                epsilon,
                unit_b1_1,
                unit_b1_2,
                unit_b1_3,
                grad_b_full_1,
                grad_b_full_2,
                grad_b_full_3,
                B_dot_b_coeffs,
                curl_unit_b_dot_b0,
                e_field_1,
                e_field_2,
                e_field_3,
                evaluate_e_field,
            ) = args_kernel[:14]
            self._gpu_gc_bxestar_epsilon = float(epsilon)
            self._gpu_gc_bxestar_evaluate_e_field = bool(evaluate_e_field)
            self._gpu_gc_bxestar_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_gc_bxestar_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_gc_bxestar_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_gc_bxestar_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_gc_bxestar_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            self._gpu_gc_bxestar_unit_b1 = (unit_b1_1, unit_b1_2, unit_b1_3)
            self._gpu_gc_bxestar_grad_b_full = (grad_b_full_1, grad_b_full_2, grad_b_full_3)
            self._gpu_gc_bxestar_B_dot_b_coeffs = B_dot_b_coeffs
            self._gpu_gc_bxestar_curl_unit_b_dot_b0 = curl_unit_b_dot_b0
            self._gpu_gc_bxestar_e_field = (e_field_1, e_field_2, e_field_3)
            self._gpu_gc_bxestar_mu_idx = int(particles.mu_idx)

        # general (non-Cuboid) CUDA replacement for
        # push_gc_Bstar_explicit_multistage (5D guiding-center pusher).
        self._gpu_gc_bstar_general = (
            cunumpy.cupy_backend
            and kernel.name == "push_gc_Bstar_explicit_multistage"
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_gc_bstar_general:
            import cupy as cp

            self._gpu_gc_bstar_kind_map = int(args_domain.kind_map)
            self._gpu_gc_bstar_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            (
                args_derham,
                epsilon,
                grad_b_full_1,
                grad_b_full_2,
                grad_b_full_3,
                b2_1,
                b2_2,
                b2_3,
                curl_unit_b2_1,
                curl_unit_b2_2,
                curl_unit_b2_3,
                B_dot_b_coeffs,
                curl_unit_b_dot_b0,
                e_field_1,
                e_field_2,
                e_field_3,
                evaluate_e_field,
            ) = args_kernel[:17]
            self._gpu_gc_bstar_epsilon = float(epsilon)
            self._gpu_gc_bstar_evaluate_e_field = bool(evaluate_e_field)
            self._gpu_gc_bstar_pn = tuple(int(p) for p in args_derham.pn)
            self._gpu_gc_bstar_starts = tuple(int(s) for s in args_derham.starts)
            self._gpu_gc_bstar_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_gc_bstar_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_gc_bstar_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            self._gpu_gc_bstar_grad_b_full = (grad_b_full_1, grad_b_full_2, grad_b_full_3)
            self._gpu_gc_bstar_b2 = (b2_1, b2_2, b2_3)
            self._gpu_gc_bstar_curl_unit_b2 = (curl_unit_b2_1, curl_unit_b2_2, curl_unit_b2_3)
            self._gpu_gc_bstar_B_dot_b_coeffs = B_dot_b_coeffs
            self._gpu_gc_bstar_curl_unit_b_dot_b0 = curl_unit_b_dot_b0
            self._gpu_gc_bstar_e_field = (e_field_1, e_field_2, e_field_3)
            self._gpu_gc_bstar_mu_idx = int(particles.mu_idx)

        # CUDA replacements for the three SPH velocity pushers. Their inner
        # work is the box-neighbourhood SPH sum (see
        # pusher_kernels_sph_cuda.box_based_kernel_dev); boxes/neighbours are
        # host-owned by SortingBoxes and uploaded per call, everything else is
        # already device-resident.
        self._gpu_sph_pusher = (
            cunumpy.cupy_backend
            and kernel.name in ("push_v_sph_pressure", "push_v_sph_pressure_ideal_gas", "push_v_viscosity")
            and args_domain.kind_map in SUPPORTED_GENERAL_KIND_MAPS
        )
        if self._gpu_sph_pusher:
            import cupy as cp

            self._gpu_sph_name = kernel.name
            self._gpu_sph_kind_map = int(args_domain.kind_map)
            self._gpu_sph_params = cp.asarray(np.asarray(args_domain.params, dtype=float), dtype=cp.float64)
            if kernel.name == "push_v_viscosity":
                (boxes, neighbours, holes, per1, per2, per3, kernel_nr, h1, h2, h3) = args_kernel
                self._gpu_sph_gravity = None
                self._gpu_sph_kappa = None
            else:
                (
                    boxes, neighbours, holes, per1, per2, per3,
                    kernel_nr, h1, h2, h3, gravity, kappa,
                ) = args_kernel
                self._gpu_sph_gravity = cp.asarray(np.asarray(gravity, dtype=float), dtype=cp.float64)
                self._gpu_sph_kappa = float(kappa)
            self._gpu_sph_boxes = boxes
            self._gpu_sph_neighbours = neighbours
            self._gpu_sph_holes = holes
            self._gpu_sph_periodic = (bool(per1), bool(per2), bool(per3))
            self._gpu_sph_kernel_nr = int(kernel_nr)
            self._gpu_sph_h = (float(h1), float(h2), float(h3))

        # CUDA replacements for the 1st-order discrete-gradient guiding-centre
        # pushers. Each call is ONE Picard iteration (the fixed-point loop is
        # the `while` in _push below), so they are per-marker parallel like the
        # explicit pushers; no domain Jacobian is needed, the Poisson-matrix
        # pieces come from marker columns filled by the init/eval kernels.
        self._gpu_gc_dg1 = cunumpy.cupy_backend and kernel.name in (
            "push_gc_bxEstar_discrete_gradient_1st_order",
            "push_gc_Bstar_discrete_gradient_1st_order",
        )
        if self._gpu_gc_dg1:
            import cupy as cp

            self._gpu_gc_dg1_name = kernel.name
            (
                args_derham,
                epsilon,
                gb1, gb2, gb3,
                ef1, ef2, ef3,
                evaluate_e_field,
            ) = args_kernel[:9]
            self._gpu_gc_dg1_epsilon = float(epsilon)
            self._gpu_gc_dg1_eval_e = bool(evaluate_e_field)
            self._gpu_gc_dg1_pn = tuple(int(x) for x in args_derham.pn)
            self._gpu_gc_dg1_starts = tuple(int(x) for x in args_derham.starts)
            self._gpu_gc_dg1_tn1 = cp.asarray(args_derham.tn1, dtype=cp.float64)
            self._gpu_gc_dg1_tn2 = cp.asarray(args_derham.tn2, dtype=cp.float64)
            self._gpu_gc_dg1_tn3 = cp.asarray(args_derham.tn3, dtype=cp.float64)
            self._gpu_gc_dg1_gb = (gb1, gb2, gb3)
            self._gpu_gc_dg1_ef = (ef1, ef2, ef3)
            self._gpu_gc_dg1_mu_idx = int(particles.mu_idx)

    @profile
    def __call__(self, dt: float):
        """
        Applies the chosen pusher kernel by a time step dt,
        applies kinetic boundary conditions and performs MPI sorting.
        """
        with ProfileManager.profile_region(self._region_name):
            if self._gpu_eta_cuboid_periodic:
                self._push_eta_cuboid_periodic_gpu(dt)
            elif self._gpu_v_efield_cuboid_wholepush:
                self._push_v_efield_cuboid_gpu(dt)
            else:
                self._push(dt)

    def _push_eta_cuboid_periodic_gpu(self, dt: float):
        """Whole-push GPU-resident fast path, see :func:`push_eta_rk_periodic_gpu`."""
        particles = self.particles
        a, b, _c = self._args_kernel
        push_eta_rk_periodic_gpu(
            particles.markers,
            particles.n_cols,
            particles.vdim,
            particles.first_pusher_idx,
            particles.first_shift_idx,
            particles.first_free_idx,
            self._gpu_eta_cuboid_scale,
            dt,
            a,
            b,
            self.n_stages,
        )

    def _push_v_efield_cuboid_gpu(self, dt: float):
        """Whole-push GPU-resident fast path, see
        :func:`~struphy.pic.pushing.pusher_kernels_cuda.push_v_with_efield_cuboid_gpu`.

        Only the velocity columns are touched (positions and holes/ghost
        status are untouched), so unlike :meth:`_push`, there is no marker
        buffer reset and no ``apply_kinetic_bc``/``update_holes`` call to
        replicate here: both would be no-ops given this kernel never moves a
        marker.
        """
        particles = self.particles
        push_v_with_efield_cuboid_gpu(
            particles.markers,
            particles.n_cols,
            self._gpu_v_efield_pn,
            self._gpu_v_efield_tn1,
            self._gpu_v_efield_tn2,
            self._gpu_v_efield_tn3,
            self._gpu_v_efield_starts,
            self._gpu_v_efield_e1_1,
            self._gpu_v_efield_e1_2,
            self._gpu_v_efield_e1_3,
            self._gpu_v_efield_scale,
            dt * self._gpu_v_efield_const,
        )

    def _kernel_region(self, kernel) -> str:
        """Cached name of the profiling region of an init/eval kernel."""
        name = self._kernel_region_names.get(id(kernel))
        if name is None:
            name = "kernel: " + _kernel_name(kernel)
            self._kernel_region_names[id(kernel)] = name
        return name


    def _run_marker_column_kernel(self, ker, alpha, column_nr, comps, add_args):
        """Run one init/eval kernel (they write a marker column in place).

        Dispatches to a CUDA port when one exists for this kernel and the
        CuPy backend is active; otherwise falls back to the compiled
        host-only kernel via the marker host mirror.
        """
        name = _kernel_name(ker)
        if cunumpy.cupy_backend and name == "driftkinetic_hamiltonian":
            args_derham, epsilon, B_dot_b, phi, evaluate_e_field = add_args[:5]
            with ProfileManager.profile_region(self._kernel_region(ker) + " [cuda]"):
                driftkinetic_hamiltonian_gpu(
                    self.particles.markers,
                    alpha,
                    column_nr,
                    self.particles.first_pusher_idx,
                    self.particles.first_shift_idx,
                    self.particles.mu_idx,
                    args_derham,
                    epsilon,
                    B_dot_b,
                    phi,
                    evaluate_e_field,
                )
            return

        with (
            ProfileManager.profile_region(self._kernel_region(ker)),
            self.particles.host_markers(write=True) as args_markers,
        ):
            ker(alpha, column_nr, comps, args_markers, self._args_domain, *add_args)

    def _push(self, dt: float):
        """Body of :meth:`__call__`, see there."""

        # some idx and slice
        markers = self.particles.markers
        vdim = self.particles.vdim
        first_pusher_idx = self.particles.first_pusher_idx
        first_shift_idx = self.particles.first_shift_idx
        residual_idx = self.particles.residual_idx

        logger.debug(f"{first_pusher_idx =}")
        logger.debug(f"{first_shift_idx =}")
        logger.debug(f"{residual_idx =}")
        logger.debug(f"{self.particles.n_cols =}")

        init_slice = slice(first_pusher_idx, first_shift_idx)
        shift_slice = slice(first_shift_idx, residual_idx)

        # Runs in place on whichever backend the markers live on -- device
        # under CuPy, with no transfer (see Particles._allocate_marker_array).
        # save initial phase space coordinates
        markers[:, init_slice] = markers[:, : 3 + vdim]

        # set boundary shifts to zero
        markers[:, shift_slice] = 0.0

        # clear buffer columns starting from residual index, dont clear ID (last column) and loc_box
        markers[:, residual_idx:-2] = 0.0

        rank = self.particles.mpi_rank
        logger.debug(f"rank {rank}: starting {self.kernel} ...")

        # if init_kernels is not empty, do evaluations at initial positions 0:3
        for ker_args in self.init_kernels:
            ker = ker_args[0]
            column_nr = ker_args[1]
            comps = ker_args[2]
            add_args = ker_args[3]

            self._run_marker_column_kernel(
                ker,
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                column_nr,
                comps,
                add_args,
            )

            # update boxes
            if self._box_comm:
                self.particles.put_particles_in_boxes()

        # start stages (e.g. n_stages=4 for RK4)
        for stage in range(self.n_stages):
            # start iteration (maxiter=1 for explicit schemes)
            n_not_converged = np.empty(1, dtype=int)
            n_not_converged[0] = self.particles.n_mks_loc
            k = 0

            if self.maxiter > 1:
                max_res = 1.0
                logger.debug(
                    f"rank {rank}: {k =}, tol: {self._tol}, {n_not_converged[0] =}, {max_res =}",
                )

            n_not_converged[0] = self.particles.Np
            while True:
                k += 1

                # if eval_kernels is not empty, do spline evaluations
                for ker_args in self.eval_kernels:
                    ker = ker_args[0]
                    alpha = ker_args[1]
                    column_nr = ker_args[2]
                    comps = ker_args[3]
                    add_args = ker_args[4]

                    # sort according to alpha-weighted average
                    if self.particles.mpi_comm is not None:
                        self.particles.mpi_sort_markers(
                            apply_bc=False,
                            alpha=alpha[:3],
                            remove_ghost=False,
                        )

                    # evaluate
                    self._run_marker_column_kernel(ker, alpha, column_nr, comps, add_args)

                    # update boxes
                    if self._box_comm:
                        self.particles.put_particles_in_boxes()

                # sort according to alpha-weighted average
                if self.particles.mpi_comm is not None:
                    self.particles.mpi_sort_markers(
                        apply_bc=False,
                        alpha=self._alpha_in_kernel,
                        remove_ghost=False,
                    )

                # push markers
                if self._gpu_eta_cuboid:
                    a, b, _c = self._args_kernel
                    last = 1.0 if stage == self.n_stages - 1 else 0.0
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                        push_eta_stage_cuboid_gpu(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self.particles.first_free_idx,
                            self._gpu_eta_cuboid_scale,
                            dt * float(a[stage]),
                            dt * float(b[stage]),
                            last,
                        )
                elif self._gpu_v_efield_cuboid:
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                        push_v_with_efield_cuboid_gpu(
                            markers,
                            self.particles.n_cols,
                            self._gpu_v_efield_pn,
                            self._gpu_v_efield_tn1,
                            self._gpu_v_efield_tn2,
                            self._gpu_v_efield_tn3,
                            self._gpu_v_efield_starts,
                            self._gpu_v_efield_e1_1,
                            self._gpu_v_efield_e1_2,
                            self._gpu_v_efield_e1_3,
                            self._gpu_v_efield_scale,
                            dt * self._gpu_v_efield_const,
                        )
                elif self._gpu_eta_general:
                    a, b, _c = self._args_kernel
                    last = 1.0 if stage == self.n_stages - 1 else 0.0
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        push_eta_stage_general_gpu(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self.particles.first_free_idx,
                            self._gpu_eta_general_kind_map,
                            self._gpu_eta_general_params,
                            dt * float(a[stage]),
                            dt * float(b[stage]),
                            last,
                        )
                elif self._gpu_pc_eta_general:
                    a, b = self._args_kernel[-3], self._args_kernel[-2]
                    last = 1.0 if stage == self.n_stages - 1 else 0.0
                    gpu_pc_eta_fn = {
                        "push_pc_eta_stage_Hcurl": push_pc_eta_stage_Hcurl_general_gpu,
                        "push_pc_eta_stage_Hdiv": push_pc_eta_stage_Hdiv_general_gpu,
                        "push_pc_eta_stage_H1vec": push_pc_eta_stage_H1vec_general_gpu,
                    }[self._gpu_pc_eta_general_variant]
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        gpu_pc_eta_fn(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self.particles.first_free_idx,
                            self._gpu_pc_eta_general_pn,
                            self._gpu_pc_eta_general_tn1,
                            self._gpu_pc_eta_general_tn2,
                            self._gpu_pc_eta_general_tn3,
                            self._gpu_pc_eta_general_starts,
                            self._gpu_pc_eta_general_u_1,
                            self._gpu_pc_eta_general_u_2,
                            self._gpu_pc_eta_general_u_3,
                            self._gpu_pc_eta_general_use_perp_model,
                            self._gpu_pc_eta_general_kind_map,
                            self._gpu_pc_eta_general_params,
                            dt * float(a[stage]),
                            dt * float(b[stage]),
                            last,
                        )
                elif self._gpu_det_diffusion_general:
                    a, b, _c = self._args_kernel[-3:]
                    last = 1.0 if stage == self.n_stages - 1 else 0.0
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        push_deterministic_diffusion_stage_general_gpu(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self.particles.first_free_idx,
                            self._gpu_det_diffusion_general_pn,
                            self._gpu_det_diffusion_general_tn1,
                            self._gpu_det_diffusion_general_tn2,
                            self._gpu_det_diffusion_general_tn3,
                            self._gpu_det_diffusion_general_starts,
                            self._gpu_det_diffusion_general_pi_u,
                            self._gpu_det_diffusion_general_pi_grad_u1,
                            self._gpu_det_diffusion_general_pi_grad_u2,
                            self._gpu_det_diffusion_general_pi_grad_u3,
                            self._gpu_det_diffusion_general_coeff,
                            self._gpu_det_diffusion_general_kind_map,
                            self._gpu_det_diffusion_general_params,
                            dt * float(a[stage]),
                            dt * float(b[stage]),
                            last,
                        )
                elif self._gpu_random_diffusion:
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        push_random_diffusion_stage_gpu(
                            markers,
                            self.particles.n_cols,
                            self._gpu_random_diffusion_noise,
                            self._gpu_random_diffusion_coeff,
                            dt,
                        )
                elif self._gpu_gc_bxestar_general:
                    a, b, _c = self._args_kernel[-3:]
                    last = 1.0 if stage == self.n_stages - 1 else 0.0
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        push_gc_bxEstar_explicit_multistage_general_gpu(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self.particles.first_free_idx,
                            self._gpu_gc_bxestar_mu_idx,
                            self._gpu_gc_bxestar_kind_map,
                            self._gpu_gc_bxestar_params,
                            self._gpu_gc_bxestar_epsilon,
                            self._gpu_gc_bxestar_pn,
                            self._gpu_gc_bxestar_tn1,
                            self._gpu_gc_bxestar_tn2,
                            self._gpu_gc_bxestar_tn3,
                            self._gpu_gc_bxestar_starts,
                            *self._gpu_gc_bxestar_unit_b1,
                            *self._gpu_gc_bxestar_grad_b_full,
                            self._gpu_gc_bxestar_B_dot_b_coeffs,
                            self._gpu_gc_bxestar_curl_unit_b_dot_b0,
                            *self._gpu_gc_bxestar_e_field,
                            self._gpu_gc_bxestar_evaluate_e_field,
                            dt * float(a[stage]),
                            dt * float(b[stage]),
                            last,
                        )
                elif self._gpu_gc_bstar_general:
                    a, b, _c = self._args_kernel[-3:]
                    last = 1.0 if stage == self.n_stages - 1 else 0.0
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        push_gc_Bstar_explicit_multistage_general_gpu(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self.particles.first_free_idx,
                            self._gpu_gc_bstar_mu_idx,
                            self._gpu_gc_bstar_kind_map,
                            self._gpu_gc_bstar_params,
                            self._gpu_gc_bstar_epsilon,
                            self._gpu_gc_bstar_pn,
                            self._gpu_gc_bstar_tn1,
                            self._gpu_gc_bstar_tn2,
                            self._gpu_gc_bstar_tn3,
                            self._gpu_gc_bstar_starts,
                            *self._gpu_gc_bstar_grad_b_full,
                            *self._gpu_gc_bstar_b2,
                            *self._gpu_gc_bstar_curl_unit_b2,
                            self._gpu_gc_bstar_B_dot_b_coeffs,
                            self._gpu_gc_bstar_curl_unit_b_dot_b0,
                            *self._gpu_gc_bstar_e_field,
                            self._gpu_gc_bstar_evaluate_e_field,
                            dt * float(a[stage]),
                            dt * float(b[stage]),
                            last,
                        )
                elif self._gpu_v_efield_general:
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        push_v_with_efield_general_gpu(
                            markers,
                            self.particles.n_cols,
                            self._gpu_v_efield_general_pn,
                            self._gpu_v_efield_general_tn1,
                            self._gpu_v_efield_general_tn2,
                            self._gpu_v_efield_general_tn3,
                            self._gpu_v_efield_general_starts,
                            self._gpu_v_efield_general_e1_1,
                            self._gpu_v_efield_general_e1_2,
                            self._gpu_v_efield_general_e1_3,
                            self._gpu_v_efield_general_kind_map,
                            self._gpu_v_efield_general_params,
                            dt * self._gpu_v_efield_general_const,
                        )
                elif self._gpu_vxb_general:
                    gpu_vxb_fn = (
                        push_vxb_analytic_general_gpu
                        if self._gpu_vxb_general_analytic
                        else push_vxb_implicit_general_gpu
                    )
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        gpu_vxb_fn(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self._gpu_vxb_general_pn,
                            self._gpu_vxb_general_tn1,
                            self._gpu_vxb_general_tn2,
                            self._gpu_vxb_general_tn3,
                            self._gpu_vxb_general_starts,
                            self._gpu_vxb_general_b2_1,
                            self._gpu_vxb_general_b2_2,
                            self._gpu_vxb_general_b2_3,
                            self._gpu_vxb_general_kind_map,
                            self._gpu_vxb_general_params,
                            dt,
                        )
                elif self._gpu_bxu_general:
                    gpu_bxu_fn = {
                        "push_bxu_Hdiv": push_bxu_Hdiv_general_gpu,
                        "push_bxu_Hcurl": push_bxu_Hcurl_general_gpu,
                        "push_bxu_H1vec": push_bxu_H1vec_general_gpu,
                    }[self._gpu_bxu_general_variant]
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        gpu_bxu_fn(
                            markers,
                            self.particles.n_cols,
                            self._gpu_bxu_general_pn,
                            self._gpu_bxu_general_tn1,
                            self._gpu_bxu_general_tn2,
                            self._gpu_bxu_general_tn3,
                            self._gpu_bxu_general_starts,
                            self._gpu_bxu_general_b2_1,
                            self._gpu_bxu_general_b2_2,
                            self._gpu_bxu_general_b2_3,
                            self._gpu_bxu_general_u_1,
                            self._gpu_bxu_general_u_2,
                            self._gpu_bxu_general_u_3,
                            self._gpu_bxu_general_kind_map,
                            self._gpu_bxu_general_params,
                            self._gpu_bxu_general_boundary_cut,
                            dt,
                        )
                elif self._gpu_pc_gxu_general:
                    g11, g12, g13, g21, g22, g23, g31, g32, g33 = self._gpu_pc_gxu_general_g
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        if self._gpu_pc_gxu_general_full:
                            push_pc_GXu_full_general_gpu(
                                markers,
                                self.particles.n_cols,
                                self._gpu_pc_gxu_general_pn,
                                self._gpu_pc_gxu_general_tn1,
                                self._gpu_pc_gxu_general_tn2,
                                self._gpu_pc_gxu_general_tn3,
                                self._gpu_pc_gxu_general_starts,
                                g11,
                                g12,
                                g13,
                                g21,
                                g22,
                                g23,
                                g31,
                                g32,
                                g33,
                                self._gpu_pc_gxu_general_kind_map,
                                self._gpu_pc_gxu_general_params,
                                dt,
                            )
                        else:
                            push_pc_GXu_general_gpu(
                                markers,
                                self.particles.n_cols,
                                self._gpu_pc_gxu_general_pn,
                                self._gpu_pc_gxu_general_tn1,
                                self._gpu_pc_gxu_general_tn2,
                                self._gpu_pc_gxu_general_tn3,
                                self._gpu_pc_gxu_general_starts,
                                g11,
                                g12,
                                g13,
                                g21,
                                g22,
                                g23,
                                self._gpu_pc_gxu_general_kind_map,
                                self._gpu_pc_gxu_general_params,
                                dt,
                            )
                elif self._gpu_weights_efield_general:
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda general]"):
                        push_weights_with_efield_lin_va_general_gpu(
                            markers,
                            self.particles.n_cols,
                            self._gpu_weights_efield_general_pn,
                            self._gpu_weights_efield_general_tn1,
                            self._gpu_weights_efield_general_tn2,
                            self._gpu_weights_efield_general_tn3,
                            self._gpu_weights_efield_general_starts,
                            self._gpu_weights_efield_general_e1_1,
                            self._gpu_weights_efield_general_e1_2,
                            self._gpu_weights_efield_general_e1_3,
                            self._gpu_weights_efield_general_f0_values,
                            self._gpu_weights_efield_general_kappa,
                            self._gpu_weights_efield_general_vth,
                            self._gpu_weights_efield_general_kind_map,
                            self._gpu_weights_efield_general_params,
                            dt,
                        )
                elif self._gpu_gc_dg1:
                    fn = (
                        push_gc_bxEstar_discrete_gradient_1st_order_gpu
                        if self._gpu_gc_dg1_name == "push_gc_bxEstar_discrete_gradient_1st_order"
                        else push_gc_Bstar_discrete_gradient_1st_order_gpu
                    )
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                        fn(
                            markers,
                            self.particles.n_cols,
                            first_pusher_idx,
                            self.particles.first_shift_idx,
                            self.particles.residual_idx,
                            self.particles.first_free_idx,
                            self._gpu_gc_dg1_mu_idx,
                            self._gpu_gc_dg1_epsilon,
                            self._gpu_gc_dg1_pn,
                            self._gpu_gc_dg1_tn1,
                            self._gpu_gc_dg1_tn2,
                            self._gpu_gc_dg1_tn3,
                            self._gpu_gc_dg1_starts,
                            self._gpu_gc_dg1_gb,
                            self._gpu_gc_dg1_ef,
                            self._gpu_gc_dg1_eval_e,
                            dt,
                        )
                elif self._gpu_sph_pusher:
                    with ProfileManager.profile_region("kernel: " + self.kernel.name + " [cuda]"):
                        common = dict(
                            boxes=self.particles.sorting_boxes.boxes,
                            neighbours=self.particles.sorting_boxes.neighbours,
                            holes=self.particles.holes,
                            periodic=self._gpu_sph_periodic,
                            kernel_type=self._gpu_sph_kernel_nr,
                            h=self._gpu_sph_h,
                            kind_map=self._gpu_sph_kind_map,
                            params_dev=self._gpu_sph_params,
                            dt=dt,
                        )
                        if self._gpu_sph_name == "push_v_viscosity":
                            push_v_viscosity_gpu(
                                markers,
                                self.particles.valid_mks,
                                self.particles.first_free_idx,
                                **common,
                            )
                        else:
                            fn = (
                                push_v_sph_pressure_gpu
                                if self._gpu_sph_name == "push_v_sph_pressure"
                                else push_v_sph_pressure_ideal_gas_gpu
                            )
                            fn(
                                markers,
                                self.particles.valid_mks,
                                self.particles.index["weights"],
                                self.particles.first_free_idx,
                                gravity=self._gpu_sph_gravity,
                                kappa=self._gpu_sph_kappa,
                                **common,
                            )
                else:
                    # no CUDA port for this kernel: fall back to the compiled
                    # host-only one, which pushes markers in place
                    with (
                        ProfileManager.profile_region("kernel: " + self.kernel.name),
                        self.particles.host_markers(write=True) as args_markers,
                    ):
                        self.kernel(
                            dt,
                            stage,
                            args_markers,
                            self._args_domain,
                            *self._args_kernel,
                        )

                self.particles.apply_kinetic_bc(newton=self._newton)
                self.particles.update_holes()

                # update boxes
                if self._box_comm:
                    self.particles.put_particles_in_boxes()

                # compute number of non-converged particles (maxiter=1 for explicit schemes)
                if self.maxiter > 1:
                    self._residuals[:] = markers[:, residual_idx]
                    max_res = float(cunumpy.max(self._residuals))
                    if max_res < 0.0:
                        max_res = None
                    self._converged_loc[:] = self._residuals < self._tol
                    self._not_converged_loc[:] = ~self._converged_loc
                    # n_not_converged is a host buffer: it is passed straight
                    # into an mpi4py Allreduce below.
                    n_not_converged[0] = int(
                        cunumpy.count_nonzero(self._not_converged_loc),
                    )

                    logger.debug(
                        f"rank {rank}: {k =}, tol: {self._tol}, {n_not_converged[0] =}, {max_res =}",
                    )

                    if self.particles.mpi_comm is not None:
                        self.particles.mpi_comm.Allreduce(
                            MPI.IN_PLACE,
                            n_not_converged,
                            op=MPI.SUM,
                        )

                    # take converged markers out of the loop
                    markers[self._converged_loc, first_pusher_idx] = -1.0

                # maxiter=1 for explicit schemes
                if k == self.maxiter:
                    if self.maxiter > 1:
                        rank = self.particles.mpi_rank
                        logger.info(
                            f"rank {rank}: {k =}, maxiter={self.maxiter} reached! tol: {self._tol}, {n_not_converged[0] =}, {max_res =}",
                        )
                    # sort markers according to domain decomposition
                    if self.mpi_sort == "each":
                        if self.particles.mpi_comm is not None:
                            self.particles.mpi_sort_markers()
                        else:
                            self.particles.apply_kinetic_bc()
                    break

                # check for convergence
                if n_not_converged[0] == 0:
                    # sort markers according to domain decomposition
                    if self.mpi_sort == "each":
                        if self.particles.mpi_comm is not None:
                            self.particles.mpi_sort_markers()
                        else:
                            self.particles.apply_kinetic_bc()

                    break

            # print stage info
            logger.debug(
                f"rank {rank}: stage {stage + 1} of {self.n_stages} done.",
            )

        # sort markers according to domain decomposition
        if self.mpi_sort == "last":
            if self.particles.mpi_comm is not None:
                self.particles.mpi_sort_markers(do_test=True)
            else:
                self.particles.apply_kinetic_bc()

    @property
    def particles(self):
        """Particle object."""
        return self._particles

    @property
    def kernel(self):
        """The pyccelized pusher kernel."""
        return self._kernel

    @property
    def init_kernels(self):
        """A dict of kernels for initial spline evaluation before iteration."""
        return self._init_kernels

    @property
    def eval_kernels(self):
        """A dict of kernels for spline evaluation before execution of kernel during iteration."""
        return self._eval_kernels

    @property
    def args_kernel(self):
        """Optional arguments for kernel."""
        return self._args_kernel

    @property
    def args_domain(self):
        """Mandatory Domain arguments."""
        return self._args_domain

    @property
    def n_stages(self):
        """Number of stages of the pusher."""
        return self._n_stages

    @property
    def maxiter(self):
        """Maximum number of iterations (=1 for explicit pushers)."""
        return self._maxiter

    @property
    def tol(self):
        """Iteration terminates when residual<tol."""
        return self._tol

    @property
    def mpi_sort(self):
        """When to do MPI sorting:
        * None : no sorting at all.
        * each : sort markers after each stage.
        * last : sort markers after last stage.
        """
        return self._mpi_sort
