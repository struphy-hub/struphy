'Syntactic sugar for calling pusher kernels.'


from struphy.geometry.base import Domain

from struphy.pic.base import Particles
from struphy.pic.pushing import pusher_kernels
from struphy.pic.pushing import pusher_kernels_gc
from struphy.pic.pushing import eval_kernels_gc
from struphy.pic.pushing.pusher_args_kernels import DerhamArguments, DomainArguments

import numpy as np
from mpi4py.MPI import SUM, IN_PLACE


class Pusher:
    """
    Syntactic sugar for particle pusher kernels. 

    It retrieves the correct pusher kernels and prepares the FEM arguments passed to them.

    Parameters
    ----------
    particles : Particles
        Particles object holding the markers to push.
    
    kernel : pyccelized function
        The pusher kernel.
        
    args_derham : DerhamArguments
        Discrete FE space infos.

    args_domain : DomainArguments
        Mapping infos.

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
                 args_derham: DerhamArguments, 
                 args_domain: DomainArguments,
                 *,
                 init_kernel: list = None, 
                 eval_kernels: list = [], 
                 n_stages: int = 1, 
                 maxiter: int = 1, 
                 tol: float = 1.e-8,
                 mpi_sort: str = None, 
                 verbose: bool = False):

        self._particles = particles
        self._kernel = kernel
        self._args_derham = args_derham
        self._args_domain = args_domain
        
        self._init_kernel = init_kernel
        self._eval_kernels = eval_kernels
        self._n_stages = n_stages
        self._maxiter = maxiter
        self._tol = tol
        self._mpi_sort = mpi_sort
        self._verbose = verbose
        
        self._mpi_sum = SUM
        self._mpi_in_place = IN_PLACE

    def __call__(self, 
                 dt: float, 
                 *optional_args):
        """
        Applies the chosen pusher kernel by a time step dt, 
        applies kinetic boundary conditions and performs MPI sorting.

        Parameters
        ----------
        dt : float
            Time step.

        optional_args : any
            Optional arguments needed for the pushing (typically spline coefficients for field evaluation).
        """

        # save initial phase space coordinates
        self.particles.markers[~self.particles.holes,
                          self.particles.bufferindex:self.particles.bufferindex+3] = self.particles.markers[~self.particles.holes, :3]

        # prepare the iteration:
        if self.init_kernel is not None:
            self.init_kernel(self.particles.markers,
                             dt,
                             self._args_derham, 
                             self._args_domain,
                             *optional_args)
            self.particles.mpi_sort_markers()

        # start stages (e.g. n_stages=4 for RK4)
        for stage in range(self.n_stages):

            # start iteration (maxiter=1 for explicit schemes)
            n_not_converged = np.empty(1, dtype=int)
            n_not_converged[0] = self.particles.n_mks
            k = 0

            while n_not_converged[0] > 0:
                k += 1

                # do evaluations if eval_kernels is not empty
                for eval_ker in self.eval_kernels:
                    eval_ker(self.particles.markers,
                             dt,
                             self._args_derham, 
                             self._args_domain,
                             *optional_args)
                    self.particles.mpi_sort_markers()

                # push markers
                self.kernel(self.particles.markers,
                            dt,
                            stage,
                            self._args_derham, 
                            self._args_domain,
                            *optional_args)

                # sort markers according to domain decomposition
                if self.mpi_sort == 'each':
                    self.particles.mpi_sort_markers()
                else:
                    self.particles.apply_kinetic_bc()

                # compute number of non coverged particles
                not_converged_loc = np.logical_not(
                    self.particles.markers[:, self.particles.bufferindex] == -1.)
                n_not_converged[0] = np.count_nonzero(not_converged_loc)

                self.particles.derham.comm.Allreduce(
                    self._mpi_in_place, n_not_converged, op=self._mpi_sum)

                if k == self.maxiter:
                    if self.verbose:
                        print(
                            f'maxiter={self.maxiter} reached for kernel "{self.kernel}" !')
                    break

            # print stage info
            if self.particles.derham.comm.Get_rank() == 0 and self.verbose:
                print(self.kernel, ' done. (stage: ', stage + 1, ')')

        # sort markers according to domain decomposition
        if self.mpi_sort == 'last':
            self.particles.mpi_sort_markers(do_test=True)

        # clear buffer columns
        self.particles.markers[~self.particles.holes,  self.particles.bufferindex:-1] = 0.

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
    def init_kernel(self):
        """ A kernel for initializing the iteration.
        """
        return self._init_kernel

    @property
    def eval_kernels(self):
        """ A list of kernels for evaluation during iteration.
        """
        return self._eval_kernels
    
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
