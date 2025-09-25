"Accelerated particle pushing."

import numpy as np
from line_profiler import profile
from mpi4py.MPI import IN_PLACE, SUM

from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments
from struphy.pic.base import Particles
from struphy.profiling.profiling import ProfileManager


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

    verbose : bool
        Whether to print some info or not.
    """

    def __init__(
        self,
        particles: Particles,
        kernel: Pyccelkernel,
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
        verbose: bool = False,
    ):
        self._particles = particles
        assert isinstance(kernel, Pyccelkernel), f"{kernel} is not of type Pyccelkernel"
        self._kernel = kernel
        self._newton = "newton" in kernel.name
        self._args_kernel = args_kernel
        self._args_domain = args_domain

        # determines the evaluation points for kernel
        self._alpha_in_kernel = alpha_in_kernel
        self._n_stages = n_stages
        self._maxiter = maxiter
        self._tol = tol
        self._mpi_sort = mpi_sort
        self._verbose = verbose

        # prepare and check init_kernels
        for ker_args in init_kernels:
            assert len(ker_args) == 4
            column_nr = ker_args[1]
            comps = ker_args[2]

            # check marker array column number
            assert isinstance(comps, np.ndarray)
            assert column_nr + comps.size < particles.n_cols, (
                f"{column_nr + comps.size} not smaller than {particles.n_cols = }; not enough columns in marker array !!"
            )

        # prepare and check eval_kernels
        for ker_args in eval_kernels:
            assert len(ker_args) == 5
            column_nr = ker_args[2]
            comps = ker_args[3]

            # check marker array column number
            assert isinstance(comps, np.ndarray)
            assert column_nr + comps.size < particles.n_cols, (
                f"{column_nr + comps.size} not smaller than {particles.n_cols = }; not enough columns in marker array !!"
            )

        self._init_kernels = init_kernels
        self._eval_kernels = eval_kernels

        self._residuals = np.zeros(self.particles.markers.shape[0])
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

        # some idx and slice
        markers = self.particles.markers
        vdim = self.particles.vdim
        first_pusher_idx = self.particles.first_pusher_idx
        first_shift_idx = self.particles.first_shift_idx
        residual_idx = self.particles.residual_idx

        if self.verbose:
            print(f"{first_pusher_idx = }")
            print(f"{first_shift_idx = }")
            print(f"{residual_idx = }")
            print(f"{self.particles.n_cols = }")

        init_slice = slice(first_pusher_idx, first_shift_idx)
        shift_slice = slice(first_shift_idx, residual_idx)

        # save initial phase space coordinates
        markers[:, init_slice] = markers[:, : 3 + vdim]

        # set boundary shifts to zero
        markers[:, shift_slice] = 0.0

        # clear buffer columns starting from residual index, dont clear ID (last column) and loc_box
        markers[:, residual_idx:-2] = 0.0

        if self.verbose:
            rank = self.particles.mpi_rank
            print(f"rank {rank}: starting {self.kernel} ...")
            if self.particles.mpi_comm is not None:
                self.particles.mpi_comm.Barrier()

        # if init_kernels is not empty, do evaluations at initial positions 0:3
        for ker_args in self.init_kernels:
            ker = ker_args[0]
            column_nr = ker_args[1]
            comps = ker_args[2]
            add_args = ker_args[3]

            ker(
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                column_nr,
                comps,
                self.particles.args_markers,
                self._args_domain,
                *add_args,
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

            if self.verbose and self.maxiter > 1:
                max_res = 1.0
                print(
                    f"rank {rank}: {k = }, tol: {self._tol}, {n_not_converged[0] = }, {max_res = }",
                )
                if self.particles.mpi_comm is not None:
                    self.particles.mpi_comm.Barrier()

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
                    ker(
                        alpha,
                        column_nr,
                        comps,
                        self.particles.args_markers,
                        self._args_domain,
                        *add_args,
                    )

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
                with ProfileManager.profile_region("kernel: " + self.kernel.__name__):
                    self.kernel(
                        dt,
                        stage,
                        self.particles.args_markers,
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
                    max_res = np.max(self._residuals)
                    if max_res < 0.0:
                        max_res = None
                    self._converged_loc[:] = self._residuals < self._tol
                    self._not_converged_loc[:] = ~self._converged_loc
                    n_not_converged[0] = np.count_nonzero(
                        self._not_converged_loc,
                    )

                    if self.verbose:
                        print(
                            f"rank {rank}: {k = }, tol: {self._tol}, {n_not_converged[0] = }, {max_res = }",
                        )
                        if self.particles.mpi_comm is not None:
                            self.particles.mpi_comm.Barrier()

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
                        print(
                            f"rank {rank}: {k = }, maxiter={self.maxiter} reached! tol: {self._tol}, {n_not_converged[0] = }, {max_res = }",
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
            if self.verbose:
                print(
                    f"rank {rank}: stage {stage + 1} of {self.n_stages} done.",
                )
                if self.particles.mpi_comm is not None:
                    self.particles.mpi_comm.Barrier()

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

    @property
    def verbose(self):
        """Print more info."""
        return self._verbose
