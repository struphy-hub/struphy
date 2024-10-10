'Accelerated particle pushing.'


from struphy.pic.base import Particles
from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments

import numpy as np
from mpi4py.MPI import SUM, IN_PLACE


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
    for any :class:`~struphy.pic.pushing.pusher.ButcherTableau`
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
    * Pusher ``kernel`` and ``eval_kernels`` can perform evaluations at arbitrary
    weighted averages :math:`\eta_{p,i} = \alpha_i \eta_{p,i}^{n+1,k} + (1 - \alpha_i) \eta_{p,i}^n`,
    for :math:`i=1,2,3`.
    * MPI sorting is done automatically before kernel calls according to the specified 
    values :math:`\alpha_i` for each kernel.

    Parameters
    ----------
    particles : Particles
        Particles object holding the markers to push.

    kernel : pyccelized function
        The pusher kernel.

    args_kernel : tuple
        Optional arguments passed to the kernel.

    args_derham : DerhamArguments
        Discrete FE space infos.

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
        Keys: initialization kernels for spline evaluations at time n (initial state).
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

    def __init__(self,
                 particles: Particles,
                 kernel,
                 args_kernel: tuple,
                 args_derham: DerhamArguments,
                 args_domain: DomainArguments,
                 *,
                 alpha_in_kernel: float | int | tuple | list,
                 init_kernels: dict = {},
                 eval_kernels: dict = {},
                 n_stages: int = 1,
                 maxiter: int = 1,
                 tol: float = 1.e-8,
                 mpi_sort: str = None,
                 verbose: bool = False):

        self._particles = particles
        self._kernel = kernel
        self._newton = 'newton' in kernel.__name__
        self._args_kernel = args_kernel
        self._args_derham = args_derham
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
            assert column_nr + \
                comps.size < particles.n_cols, f'{column_nr + comps.size} not smaller than {particles.n_cols = }; not enough columns in marker array !!'

        # prepare and check eval_kernels
        for ker_args in eval_kernels:
            assert len(ker_args) == 5
            column_nr = ker_args[2]
            comps = ker_args[3]

            # check marker array column number
            assert isinstance(comps, np.ndarray)
            assert column_nr + \
                comps.size < particles.n_cols, f'{column_nr + comps.size} not smaller than {particles.n_cols = }; not enough columns in marker array !!'

        self._init_kernels = init_kernels
        self._eval_kernels = eval_kernels

        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        self._residuals = np.zeros(self.particles.markers.shape[0])
        self._converged_loc = self._residuals == 1.
        self._not_converged_loc = self._residuals == 0.

    def __call__(self, dt: float):
        """
        Applies the chosen pusher kernel by a time step dt, 
        applies kinetic boundary conditions and performs MPI sorting.
        """

        # some pointers
        markers = self.particles.markers
        buffer_idx = self.particles.bufferindex
        residual_idx = self.particles.args_markers.residual_idx
        vdim = self.particles.vdim

        # save initial phase space coordinates
        markers[:, buffer_idx:buffer_idx + 3 + vdim] = markers[:, :3 + vdim]

        # set boundary shifts to zero
        markers[:, buffer_idx + 3 + vdim: buffer_idx + 3 + vdim + 3] = 0.

        # clear buffer columns starting from residual index, dont clear ID (last column)
        markers[:,  residual_idx:-1] = 0.

        if self.verbose:
            rank = self.particles.derham.comm.Get_rank()
            print(f'rank {rank}: starting {self.kernel} ...')
            self.particles.derham.comm.Barrier()

        # if init_kernels is not empty, do spline evaluations at initial positions 0:3
        for ker_args in self.init_kernels:
            ker = ker_args[0]
            column_nr = ker_args[1]
            comps = ker_args[2]
            add_args = ker_args[3]

            ker(np.array([0., 0., 0., 0., 0., 0.]),
                column_nr,
                comps,
                self.particles.args_markers,
                self._args_derham,
                self._args_domain,
                *add_args)

        # start stages (e.g. n_stages=4 for RK4)
        for stage in range(self.n_stages):

            # start iteration (maxiter=1 for explicit schemes)
            n_not_converged = np.empty(1, dtype=int)
            n_not_converged[0] = self.particles.n_mks_loc
            k = 0

            if self.verbose and self.maxiter > 1:
                max_res = 1.
                print(
                    f'rank {rank}: {k = }, tol: {self._tol}, {n_not_converged[0] = }, {max_res = }')
                self.particles.derham.comm.Barrier()

            n_not_converged[0] = self.particles.n_mks
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
                    self.particles.mpi_sort_markers(apply_bc=False,
                                                    alpha=alpha[:3])
                    # evaluate
                    ker(alpha,
                        column_nr,
                        comps,
                        self.particles.args_markers,
                        self._args_derham,
                        self._args_domain,
                        *add_args)

                # sort according to alpha-weighted average
                self.particles.mpi_sort_markers(apply_bc=False,
                                                alpha=self._alpha_in_kernel)

                # push markers
                self.kernel(dt,
                            stage,
                            self.particles.args_markers,
                            self._args_derham,
                            self._args_domain,
                            *self._args_kernel)

                self.particles.apply_kinetic_bc(newton=self._newton)

                # compute number of non-converged particles (maxiter=1 for explicit schemes)
                if self.maxiter > 1:
                    self._residuals[:] = markers[:, residual_idx]
                    max_res = np.max(self._residuals)
                    if max_res < 0.:
                        max_res = None
                    self._converged_loc[:] = self._residuals < self._tol
                    self._not_converged_loc[:] = ~self._converged_loc
                    n_not_converged[0] = np.count_nonzero(
                        self._not_converged_loc)

                    if self.verbose:
                        print(
                            f'rank {rank}: {k = }, tol: {self._tol}, {n_not_converged[0] = }, {max_res = }')
                        self.particles.derham.comm.Barrier()

                    self.particles.derham.comm.Allreduce(
                        self._mpi_in_place, n_not_converged, op=self._mpi_sum)

                    # take converged markers out of the loop
                    markers[self._converged_loc, buffer_idx] = -1.

                # maxiter=1 for explicit schemes
                if k == self.maxiter:
                    if self.maxiter > 1:
                        rank = self.particles.derham.comm.Get_rank()
                        print(
                            f'rank {rank}: {k = }, maxiter={self.maxiter} reached! tol: {self._tol}, {n_not_converged[0] = }, {max_res = }')
                    # sort markers according to domain decomposition
                    if self.mpi_sort == 'each':
                        self.particles.mpi_sort_markers()

                    break

                # check for convergence
                if n_not_converged[0] == 0:
                    # sort markers according to domain decomposition
                    if self.mpi_sort == 'each':
                        self.particles.mpi_sort_markers()

                    break

            # print stage info
            if self.verbose:
                print(
                    f'rank {rank}: stage {stage + 1} of {self.n_stages} done.')
                self.particles.derham.comm.Barrier()

        # sort markers according to domain decomposition
        if self.mpi_sort == 'last':
            self.particles.mpi_sort_markers(do_test=True)

    @property
    def particles(self):
        """ Particle object.
        """
        return self._particles

    @property
    def kernel(self):
        """ The pyccelized pusher kernel.
        """
        return self._kernel

    @property
    def init_kernels(self):
        """ A dict of kernels for initial spline evaluation before iteration.
        """
        return self._init_kernels

    @property
    def eval_kernels(self):
        """ A dict of kernels for spline evaluation before execution of kernel during iteration. 
        """
        return self._eval_kernels

    @property
    def args_kernel(self):
        """ Optional arguments for kernel.
        """
        return self._args_kernel

    @property
    def args_derham(self):
        """ Mandatory Derham arguments.
        """
        return self._args_derham

    @property
    def args_domain(self):
        """ Mandatory Domain arguments.
        """
        return self._args_domain

    @property
    def n_stages(self):
        """ Number of stages of the pusher.
        """
        return self._n_stages

    @property
    def maxiter(self):
        """ Maximum number of iterations (=1 for explicit pushers).
        """
        return self._maxiter

    @property
    def tol(self):
        """ Iteration terminates when residual<tol.
        """
        return self._tol

    @property
    def mpi_sort(self):
        """ When to do MPI sorting:
        * None : no sorting at all.
        * each : sort markers after each stage.
        * last : sort markers after last stage.
        """
        return self._mpi_sort

    @property
    def verbose(self):
        """ Print more info.
        """
        return self._verbose


class ButcherTableau:
    r"""
    Butcher tableau for explicit s-stage Runge-Kutta methods. 

    The Butcher tableau has the form

    .. image:: ../../pics/butcher_tableau.png
        :align: center
        :scale: 70%

    Parameters
    ----------
        algo : str
            Name of the RK method.
    """

    @staticmethod
    def available_methods():
        meth_avail = ['rk4',
                      'forward_euler',
                      'heun2',
                      'rk2',
                      'heun3',
                      ]
        return meth_avail

    def __init__(self, algo: str = 'rk4'):

        # choose algorithm
        if algo == 'forward_euler':
            a = []
            b = [1.]
            c = [0.]
        elif algo == 'heun2':
            a = [1.]
            b = [1/2, 1/2]
            c = [0., 1.]
        elif algo == 'rk2':
            a = [1/2]
            b = [0., 1.]
            c = [0., 1/2]
        elif algo == 'heun3':
            a = [1/3, 2/3]
            b = [1/4, 0., 3/4]
            c = [0., 1/3, 2/3]
        elif algo == 'rk4':
            a = [1/2, 1/2, 1.]
            b = [1/6, 1/3, 1/3, 1/6]
            c = [0., 1/2, 1/2, 1.]
        else:
            raise NotImplementedError('Chosen algorithm is not implemented.')

        self._b = np.array(b)
        self._c = np.array(c)
        assert self._b.size == self._c.size

        self._n_stages = self._b.size

        self._a = np.array(a)

        # size is the number of elements in the lower triangular part of A
        assert self._a.size == self._n_stages - 1

        # add zero for last stage
        self._a = np.array(list(self._a) + [0.])

    __available_methods__ = available_methods()

    @property
    def a(self):
        """ Characteristic coefficients of the method (see tableau in class docstring).
        """
        return self._a

    @property
    def b(self):
        """ Characteristic coefficients of the method (see tableau in class docstring).
        """
        return self._b

    @property
    def c(self):
        """ Characteristic coefficients of the method (see tableau in class docstring).
        """
        return self._c

    @property
    def n_stages(self):
        """ Number of stages of the s-stage Runge-Kutta method.
        """
        return self._n_stages
