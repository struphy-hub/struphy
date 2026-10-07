"Accelerated particle pushing."

import logging

import cunumpy as xp
from cunumpy.kernels import Kernel, PyccelKernel
from line_profiler import profile
from maybempi import MPI
from scope_profiler import ProfileManager

from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.pic.base import Particles
from struphy.pic.pushing.kernel_setup import KernelSetup
from struphy.utils.cuda_arguments import check_mapping_on_device

logger = logging.getLogger("struphy")


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

    The particle push is performed via an accelerated kernel from :mod:`struphy.pic.pushing.kernels`,
    e.g. ``push_eta_stage`` from :mod:`struphy.pic.pushing.kernels.push_eta_stage`.

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

    MPI sorting is skipped whenever it is known to be a no-op: markers are assumed to be sorted
    according to the domain decomposition on entry, and they stay sorted for any :math:`\alpha`
    until the pusher kernel has moved them. Pushers with ``pushes_eta=False`` hence never sort.
    Pushers with ``pushes_eta=True`` always sort at least once after the last stage,
    such that the above assumption holds for the next pusher.

    Parameters
    ----------
    particles : Particles
        Particles object holding the markers to push.

    kernel : PyccelKernel | Kernel
        The pusher kernel. A :class:`~cunumpy.kernels.Kernel` also holds its CUDA version;
        on the CuPy backend, a kernel without CUDA version raises NotImplementedError.

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

    init_kernels : tuple[KernelSetup, ...]
        Evaluations at the initial state, executed once per push in tuple order.
        Each setup specifies the kernel, arguments, and output marker indices.

    eval_kernels : tuple[KernelSetup, ...]
        Evaluations before each pusher stage/iteration. Each setup's alpha
        weights determine the evaluation state and preceding MPI sort.

    n_stages : int
        Number of stages of the pusher (e.g. 4 for RK4)

    maxiter : int
        Maximum number of iterations (=1 for explicit pushers).

    tol : float
        Iteration terminates when residual<tol.

    mpi_sort : str
        When to do MPI sorting:
        * None : no sorting at all (only allowed for ``pushes_eta=False``, becomes "last" otherwise).
        * each : sort markers after each stage.
        * last : sort markers after last stage.

    pushes_eta : bool
        Whether the kernel updates the marker positions :math:`\boldsymbol \eta_p`.
        If False, no MPI sorting is performed at all.

    local_eval_only : bool
        Set to True if the kernel does not evaluate distributed splines, i.e. it only calls
        metric coefficients or equilibrium quantities, which are available on every process.
        Markers then need not be on the right process during the stages;
        they are sorted only once after the last stage (mpi_sort="last").
    """

    def __init__(
        self,
        particles: Particles,
        kernel: PyccelKernel | Kernel,
        args_kernel: tuple,
        args_domain: DomainArguments,
        pushes_eta: bool,
        *,
        alpha_in_kernel: float | int | tuple | list,
        init_kernels: tuple[KernelSetup, ...] = (),
        eval_kernels: tuple[KernelSetup, ...] = (),
        n_stages: int = 1,
        maxiter: int = 1,
        tol: float = 1.0e-8,
        mpi_sort: str = None,
        local_eval_only: bool = False,
    ):
        # on the CuPy backend a kernel without CUDA version (yet) raises when called, see CUDA_STRATEGY.md
        self._kernel = kernel if isinstance(kernel, Kernel) else Kernel(kernel)
        self._cuda = xp.cupy_backend

        self._particles = particles
        self._newton = "newton" in kernel.name
        self._args_kernel = args_kernel
        self._args_domain = args_domain

        # determines the evaluation points for kernel
        self._alpha_in_kernel = alpha_in_kernel
        self._n_stages = n_stages
        self._maxiter = maxiter
        self._tol = tol
        self._pushes_eta = pushes_eta
        self._local_eval_only = local_eval_only

        if not pushes_eta:
            assert mpi_sort is None, f"{mpi_sort =} makes no sense for a kernel that does not push eta."
        elif local_eval_only or mpi_sort is None:
            mpi_sort = "last"
        self._mpi_sort = mpi_sort

        if local_eval_only:
            assert len(eval_kernels) == 0, "eval_kernels evaluate splines, not compatible with local_eval_only=True."

        self._init_kernels = tuple(init_kernels)
        self._eval_kernels = tuple(eval_kernels)
        for setup in self._init_kernels + self._eval_kernels:
            if not isinstance(setup, KernelSetup):
                raise TypeError("init_kernels and eval_kernels must contain KernelSetup instances")
            setup.validate_outputs(particles.n_cols)
        if any(any(setup.alpha) for setup in self._init_kernels):
            raise ValueError("init kernels must evaluate the initial state (alpha=0)")

        # profiling region names (cached, they are looked up on every call)
        self._region_name = "pusher: " + self.kernel.name

        self._residuals = xp.zeros(self.particles.markers.shape[0])
        self._converged_loc = self._residuals == 1.0
        self._not_converged_loc = self._residuals == 0.0

        if self.particles.sorting_boxes is not None:
            self._box_comm = self.particles.sorting_boxes.communicate
        else:
            self._box_comm = False

    @profile
    def __call__(self, dt: float):
        """
        Applies the chosen pusher kernel by a time step dt,
        applies kinetic boundary conditions and performs MPI sorting.
        """
        with ProfileManager.profile_region(self._region_name):
            self._push(dt)

    def _evaluate(self, setup: KernelSetup):
        """Run a configured marker evaluation and communicate its outputs."""
        with ProfileManager.profile_region("kernel: " + setup.name):
            setup.evaluate(self.particles.args_markers, self.args_domain)
        if self._box_comm:
            self.particles.put_particles_in_boxes()

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

        # save initial phase space coordinates
        markers[:, init_slice] = markers[:, : 3 + vdim]

        # set boundary shifts to zero
        markers[:, shift_slice] = 0.0

        # clear buffer columns starting from residual index, dont clear ID (last column) and loc_box
        markers[:, residual_idx:-2] = 0.0

        rank = self.particles.mpi_rank
        logger.debug(f"rank {rank}: starting {self.kernel} ...")

        # Evaluate the initial state once, before any stage or iteration.
        for setup in self.init_kernels:
            self._evaluate(setup)

        # markers are sorted on entry and initial positions equal current positions,
        # hence they are sorted for any alpha until the kernel moves them
        self._sorted_for = "any"

        # start stages (e.g. n_stages=4 for RK4)
        for stage in range(self.n_stages):
            # start iteration (maxiter=1 for explicit schemes)
            if self.maxiter > 1:
                n_not_converged = xp.empty(1, dtype=int)
                n_not_converged[0] = self.particles.n_mks_loc
            k = 0

            if self.maxiter > 1:
                max_res = 1.0
                logger.debug(
                    f"rank {rank}: {k =}, tol: {self._tol}, {n_not_converged[0] =}, {max_res =}",
                )

            if self.maxiter > 1:
                n_not_converged[0] = self.particles.Np
            while True:
                k += 1

                for setup in self.eval_kernels:
                    # sort according to alpha-weighted average
                    self._sort_for_alpha(setup.sorting_alpha)
                    self._evaluate(setup)

                # sort according to alpha-weighted average
                self._sort_for_alpha(self._alpha_in_kernel)

                # push markers
                with ProfileManager.profile_region("kernel: " + self.kernel.name):
                    self.kernel(
                        dt,
                        stage,
                        self.particles.args_markers,
                        self._args_domain,
                        *self._args_kernel,
                    )

                # markers have moved
                if self.pushes_eta:
                    self._sorted_for = None

                # kinetic boundary conditions are applied per marker inside the kernel
                self.particles.finish_kernel_bc(newton=self._newton)

                # update boxes
                if self._box_comm:
                    self.particles.put_particles_in_boxes()

                # compute number of non-converged particles (maxiter=1 for explicit schemes)
                if self.maxiter > 1:
                    self._residuals[:] = markers[:, residual_idx]
                    max_res = xp.max(self._residuals)
                    if max_res < 0.0:
                        max_res = None
                    self._converged_loc[:] = self._residuals < self._tol
                    self._not_converged_loc[:] = ~self._converged_loc
                    n_not_converged[0] = xp.count_nonzero(
                        self._not_converged_loc,
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
                        self._sort_for_alpha(1.0, remove_ghost=True)
                    break

                # check for convergence
                if n_not_converged[0] == 0:
                    # sort markers according to domain decomposition
                    if self.mpi_sort == "each":
                        self._sort_for_alpha(1.0, remove_ghost=True)

                    break

            # print stage info
            logger.debug(
                f"rank {rank}: stage {stage + 1} of {self.n_stages} done.",
            )

        # sort markers according to domain decomposition
        if self.mpi_sort == "last":
            if self.particles.mpi_comm is not None and self.particles.mpi_size > 1:
                self.particles.mpi_sort_markers(apply_bc=False, do_test=not self._cuda)

    def _sort_for_alpha(self, alpha: float | int | tuple | list, remove_ghost: bool = False):
        """MPI sort markers according to the alpha-weighted average of positions,
        unless they are already sorted accordingly or the kernel does not need it."""
        if self.particles.mpi_comm is None or self.local_eval_only:
            return

        if xp.ndim(alpha) == 0:
            alpha = (alpha, alpha, alpha)
        alpha = tuple(float(a) for a in alpha)

        if self._sorted_for == "any" or self._sorted_for == alpha:
            return

        self.particles.mpi_sort_markers(
            apply_bc=False,
            alpha=alpha,
            remove_ghost=remove_ghost,
        )
        self._sorted_for = alpha

    @property
    def particles(self):
        """Particle object."""
        return self._particles

    @property
    def kernel(self) -> Kernel:
        """The pusher kernel for the active backend (pyccel or CUDA)."""
        return self._kernel

    @property
    def init_kernels(self) -> tuple[KernelSetup, ...]:
        """Ordered setups for evaluations at the initial state."""
        return self._init_kernels

    @property
    def eval_kernels(self) -> tuple[KernelSetup, ...]:
        """Ordered setups for evaluations before each pusher stage/iteration."""
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
    def pushes_eta(self):
        """Whether the kernel updates the marker positions."""
        return self._pushes_eta

    @property
    def local_eval_only(self):
        """Whether the kernel needs no distributed spline evaluations (sorting only after last stage)."""
        return self._local_eval_only

    @property
    def mpi_sort(self):
        """When to do MPI sorting:
        * None : no sorting at all (only for ``pushes_eta=False``).
        * each : sort markers after each stage.
        * last : sort markers after last stage.
        """
        return self._mpi_sort
