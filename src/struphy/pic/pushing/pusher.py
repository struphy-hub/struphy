'Syntactic sugar for calling pusher kernels.'


from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing import pusher_kernels_gc
from struphy.pic.pushing import eval_kernels_gc

import numpy as np
from mpi4py.MPI import SUM, IN_PLACE

class Pusher:
    """
    Syntactic sugar for particle pusher kernels. 

    It retrieves the correct pusher kernels and prepares the FEM arguments passed to them.

    Parameters
    ----------
    derham : struphy.feec.psydac_derham.Derham
        Discrete de Rham sequence on the logical unit cube.

    domain : struphy.geometry.domains
        All things mapping.

    kernel_name : str
        The name of the pusher kernel. Must start with "push_".

    init_kernel : bool
        Whether there is an initialization kernel used; 
        the init_kernel name is the kernel_name with "push_" replaced by "init_".

    eval_kernels_names : list[str]
        The names of possible kernels used for evaluation during iteration.

    n_stages : int
        Number of stages of the pusher (e.g. 4 for RK4)

    maxiter : int
            Maximum number of iterations (=1 for explicit pushers).

    tol : float
        Iteration terminates when residual<tol.
    """

    def __init__(self, derham, domain, kernel_name, init_kernel=False, eval_kernels_names=None, n_stages=1, maxiter=1, tol=1.e-8):

        self._derham = derham
        self._domain = domain
        self._n_stages = n_stages
        self._maxiter = maxiter
        self._tol = tol
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

        # get FEM information
        self._args_fem = (np.array(derham.p),
                          derham.Vh_fem['0'].knots[0], 
                          derham.Vh_fem['0'].knots[1], 
                          derham.Vh_fem['0'].knots[2],
                          np.array(derham.Vh['0'].starts))

        # select pusher kernel
        assert kernel_name[:5] == 'push_'
        self._kernel_name = kernel_name
        self._kernel = None

        objs = [pusher_kernels, pusher_kernels_gc]
        for obj in objs:
            try:
                self._kernel = getattr(obj, self.kernel_name)
            except AttributeError:
                pass
        assert self.kernel is not None

        # select initialization kernel
        self._init_kernel = None

        if init_kernel:
            objs = [eval_kernels_gc]

            name = self.kernel_name
            name = name.replace('push_', 'init_')
            for obj in objs:
                try:
                    self._init_kernel = getattr(obj, name)
                except AttributeError:
                    pass
            assert self.init_kernel is not None

        # select evaluation kernels
        self._eval_kernels_names = eval_kernels_names
        self._eval_kernels = []

        if eval_kernels_names is not None:
            objs = [eval_kernels_gc]

            for name in eval_kernels_names:
                for obj in objs:
                    try:
                        self._eval_kernels += [getattr(obj, name)]
                    except AttributeError:
                        pass
            assert not self.eval_kernels == []

    def __call__(self, particles, dt, *args_opt, mpi_sort=None, verbose=False):
        """
        Applies the chosen pusher kernel by a time step dt, 
        applies kinetic boundary conditions and performs MPI sorting.

        Parameters
        ----------
        particles : struphy.pic.particles.Particles6D
            Particles object holding the markers to push.

        dt : float
            Time step.

        args_opt : tuple
            Optional arguments needed for the pushing (typically spline coefficients for field evaluation).

        mpi_sort : str
            When to do MPI sorting:
                * None : no sorting at all.
                * each : sort markers after each stage.
                * last : sort markers after last stage.

        verbose : bool
            Whether to print some info or not.
        """
        
        # save initial phase space coordinates
        particles.markers[~particles.holes,
                          9:12] = particles.markers[~particles.holes, 0:3]

        # prepare the iteration:
        if self.init_kernel is not None:
            self.init_kernel(particles.markers, dt, *self.args_fem,
                             *self.domain.args_map, *args_opt)
            particles.mpi_sort_markers()

        # start stages (e.g. n_stages=4 for RK4)
        for stage in range(self.n_stages):

            # start iteration (maxiter=1 for explicit schemes)
            n_not_converged = np.empty(1, dtype=int)
            n_not_converged[0] = particles.n_mks
            k = 0
            
            while n_not_converged[0] > 0:
                k += 1

                # do evaluations if eval_kernels is not empty
                for eval_ker in self.eval_kernels:
                    eval_ker(particles.markers, dt,
                             *self.args_fem, *self.domain.args_map, *args_opt)
                    particles.mpi_sort_markers()

                # push markers
                self.kernel(particles.markers, dt, stage,
                            *self.args_fem, *self.domain.args_map, *args_opt)

                # sort markers according to domain decomposition
                if mpi_sort == 'each':
                    particles.mpi_sort_markers()
                else:
                    particles.apply_kinetic_bc()

                # compute number of non coverged particles
                not_converged_loc = np.logical_not(particles.markers[:, 9] == -1.)
                n_not_converged[0] = np.count_nonzero(not_converged_loc)

                self.derham.comm.Allreduce(
                    self._mpi_in_place, n_not_converged, op=self._mpi_sum)

                if k == self.maxiter:
                    if verbose:
                        print(
                            f'maxiter={self.maxiter} reached for kernel "{self.kernel_name}" !')
                    break

            # print stage info
            if self.derham.comm.Get_rank() == 0 and verbose:
                print(self.kernel_name, ' done. (stage: ', stage + 1, ')')

        # sort markers according to domain decomposition
        if mpi_sort == 'last':
            particles.mpi_sort_markers(do_test=True)

        # clear buffer columns
        particles.markers[~particles.holes, 9:-1] = 0.

    @property
    def derham(self):
        """ Discrete derham sequence.
        """
        return self._derham

    @property
    def domain(self):
        """ Mapping from logical unit cube to physical domain.
        """
        return self._domain

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
    def kernel_name(self):
        """ The name of the pyccelized pusher kernel.
        """
        return self._kernel_name

    @property
    def kernel(self):
        """ The pyccelized pusher kernel.
        """
        return self._kernel

    @property
    def init_kernel(self):
        """ A kernel for initializing the iteration.
        """
        return self._init_kernel

    @property
    def eval_kernels_names(self):
        """ A list of of names of the evaluation kernels.
        """
        return self._eval_kernels_names

    @property
    def eval_kernels(self):
        """ A list of kernels for evaluation during iteration.
        """
        return self._eval_kernels

    @property
    def args_fem(self):
        """ FEM and MPI related arguments taken by all pushers.
        """
        return self._args_fem


class ButcherTableau:
    r"""
    Butcher tableau for explicit s-stage Runge-Kutta methods. 

    The Butcher tableau has the form

    .. image:: ../pics/butcher_tableau.png
        :align: center
        :scale: 70%

    Parameters
    ----------
        a : array-like
            Characteristic coefficients of the method (see tableau above). Only first lower diagonal is non-zero.

        b : array-like
            Characteristic coefficients of the method (see tableau above).

        c : array-like
            Characteristic coefficients of the method (see tableau above).
    """

    def __init__(self, a, b, c):

        self._b = np.array(b)
        self._c = np.array(c)
        assert self._b.size == self._c.size

        self._n_stages = self._b.size

        self._a = np.array(a)

        # size is the number of elements in the lower triangular part of A
        assert self._a.size == self._n_stages - 1

        # add zero for last stage
        self._a = np.array(list(self._a) + [0.])

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
