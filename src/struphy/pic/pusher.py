'Syntactic sugar for calling pusher kernels.'


import struphy.pic.pusher_kernels as pushers
import struphy.pic.pusher_kernels_gc as pushers_gc
import struphy.pic.utilities_kernels as utilities

import numpy as np


class Pusher:
    """
    Wrapper class for particle pushing. 

    It retrieves the correct pusher kernel and prepares the FEM arguments passed to the pusher kernel.

    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.

        kernel_name : str
            The name of the pusher kernel in the file struphy.pic.pusher_kernels.

        n_stages : int
            Number of stages of the pusher (e.g. 4 for RK4)
    """

    def __init__(self, derham, domain, kernel_name, n_stages=1):

        self._derham = derham
        self._domain = domain
        self._n_stages = n_stages

        # get FEM information
        self._args_fem = (np.array(derham.p),
                          derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                          np.array(derham.Vh['0'].starts),
                          np.array(derham.Vh['1'].starts),
                          np.array(derham.Vh['2'].starts),
                          np.array(derham.Vh['3'].starts))

        # select pusher kernel
        self._kernel_name = kernel_name
        self._kernel = None

        objs = [pushers, pushers_gc]
        for obj in objs:
            try:
                self._kernel = getattr(obj, self.kernel_name)
            except AttributeError:
                pass
        assert self.kernel is not None

    def __call__(self, particles, dt, *args_opt, mpi_sort=None, verbose=False):
        """
        Applies the chosen pusher kernel by a time step dt, applies kinetic boundary conditions and performs MPI sorting.

        Parameters
        ----------
            particles : struphy.pic.particles.Particles6D
                The particles object holding the markers of shape (Np, 16) to push.

            dt : float
                The time step.

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
        # save initial etas in columns 9-11
        particles.markers[~particles.holes,
                          9:12] = particles.markers[~particles.holes, 0:3]

        if particles.kinds == 'Particles5D':
            particles.markers[~particles.holes,
                              12] = particles.markers[~particles.holes, 3]

        for stage in range(self._n_stages):
            self.kernel(particles.markers, dt, stage, *
                        self.args_fem, *self.domain.args_map, *args_opt)

            # sort markers according to domain decomposition
            if mpi_sort == 'each':
                particles.mpi_sort_markers()

            else:
                particles.apply_kinetic_bc()

            # print stage info
            if self._derham.comm.Get_rank() == 0 and verbose:
                print(self.kernel_name, 'done. (stage :', stage + 1, ')')

        # sort markers according to domain decomposition
        if mpi_sort == 'last':
            particles.mpi_sort_markers(do_test=True)

        # clear buffer columns 9-14 for multi-stage pushers
        particles.markers[~particles.holes, 9:15] = 0.

        if particles.kinds == 'Particles5D':
            particles.markers[~particles.holes, 9:25] = 0.

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
    def args_fem(self):
        """ FEM and MPI related arguments taken by all pushers.
        """
        return self._args_fem

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


class ButcherTableau:
    """
    Butcher tableau for explicit s-stage Runge-Kutta methods. 

    A Butcher tableau has the form

      c_0   | 
      c_1   | a_10
      c_2   |   0  a_21
      c_3   |   0    0  a_32
       .    |   .    .    .
       .    |   .    .    .
    c_(n-1) |   0   ...   0  a_(n-1,n-2)
    --------------------------------------------
            |  b_0  b_1  b_2    ...       b_(n-1)

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
        self._a = np.array(list(self._a) + [0])

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


class Pusher_iteration_Gonzalez:
    """
    Wrapper class for particle pushing with discrete_gradient scheme (Gonzalez, mid-point).

    It retrieves the correct pusher kernel(s) and prepares the FEM arguments passed to the pusher kernel.

    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.

        kernel_name : str
            The name of the pusher kernel in the file struphy.pic.pusher_kernels.

        n_stages : int
            Number of stages of the pusher (e.g. 4 for RK4)
    """

    def __init__(self, derham, domain, kernel_name, maxiter=10, tol=1.e-12):

        self._derham = derham
        self._domain = domain
        self._maxiter = maxiter
        self._tol = tol

        self._args_fem = (derham.domain_array[derham.comm.Get_rank(), :],
                          np.array(derham.p),
                          derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                          np.array(derham.Vh['0'].starts),
                          np.array(derham.Vh['1'].starts),
                          np.array(derham.Vh['2'].starts),
                          np.array(derham.Vh['3'].starts))

        # select kernels
        self._kernel_name = kernel_name
        self._kernel = None

        objs = [pushers, pushers_gc]
        for obj in objs:
            try:
                self._kernel = getattr(obj, self.kernel_name)
            except AttributeError:
                pass
        assert self.kernel is not None

        self._kernel_prepare = getattr(
            utilities, self.kernel_name + '_prepare')
        self._kernel_eval_gradI = getattr(
            utilities, self.kernel_name + '_eval_gradI')

    def __call__(self, particles, dt, *args_opt, mpi_sort=None, verbose=False):
        """
        Applies the chosen pusher kernel by a time step dt, applies kinetic boundary conditions and performs MPI sorting.

        Parameters
        ----------
            particles : struphy.pic.particles.Particles6D
                The particles object holding the markers of shape (Np, 16) to push.

            dt : float
                The time step.

            args_opt : tuple
                Optional arguments needed for the pushing (typically spline coefficients for field evaluation).

            verbose : bool
                Whether to print some info or not.
        """
        # save initial etas and v_parallel in columns 9:13
        particles.markers[~particles.holes, 9:13] \
            = particles.markers[~particles.holes, 0:4]

        # prepare the iteration:
        self.kernel_prepare(particles.markers, dt, *self.args_fem,
                            *self.domain.args_map, *args_opt)
        particles.mpi_sort_markers()

        # eval gradI
        self.kernel_eval_gradI(particles.markers, dt, *self.args_fem,
                               *self.domain.args_map, *args_opt)
        particles.mpi_sort_markers()

        # start iteration
        for stage in range(self._maxiter):

            self.kernel(particles.markers, dt, stage, self._tol,
                        *self.args_fem, *self.domain.args_map, *args_opt)
            particles.mpi_sort_markers()

            self.kernel_eval_gradI(particles.markers, dt,
                                   *self.args_fem, *self.domain.args_map, *args_opt)
            particles.mpi_sort_markers()

            if stage == self._maxiter-1 and verbose:
                not_converged = np.logical_not(particles.markers[:, 21] == -1.)
                print('Number of not converged particles:',
                      np.count_nonzero(not_converged))
                print('Non converged partices:',
                      particles.markers[not_converged, 9:13])
                print('Number of iterations', np.average(
                      particles.markers[~particles.holes, 20])+1)
                print('Number of lost markers:',
                      particles.n_lost_markers)
                print()

        # clear buffer columns 9-21 for multi-stage pushers
        particles.markers[~particles.holes, 9:22] = 0.

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
    def args_fem(self):
        """ FEM and MPI related arguments taken by all pushers.
        """
        return self._args_fem

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
    def kernel_prepare(self):
        """ A preparation kernel.
        """
        return self._kernel_prepare

    @property
    def kernel_eval_gradI(self):
        """ A preparation kernel.
        """
        return self._kernel_eval_gradI


class Pusher_iteration_Itoh:
    """
    Wrapper class for particle pushing with discrete_gradient scheme (Itoh_Newton).

    It retrieves the correct pusher kernel(s) and prepares the FEM arguments passed to the pusher kernel.

    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.

        kernel_name : str
            The name of the pusher kernel in the file struphy.pic.pusher_kernels.

        n_stages : int
            Number of stages of the pusher (e.g. 4 for RK4)
    """

    def __init__(self, derham, domain, kernel_name, maxiter=10, tol=1.e-12):

        self._derham = derham
        self._domain = domain
        self._maxiter = maxiter
        self._tol = tol

        self._args_fem = (derham.domain_array[derham.comm.Get_rank(), :],
                          np.array(derham.p),
                          derham.Vh_fem['0'].knots[0], derham.Vh_fem['0'].knots[1], derham.Vh_fem['0'].knots[2],
                          np.array(derham.Vh['0'].starts),
                          np.array(derham.Vh['1'].starts),
                          np.array(derham.Vh['2'].starts),
                          np.array(derham.Vh['3'].starts))

        # select kernels
        self._kernel_name = kernel_name
        self._kernel = None

        objs = [pushers, pushers_gc]
        for obj in objs:
            try:
                self._kernel = getattr(obj, self.kernel_name)
            except AttributeError:
                pass
        assert self.kernel is not None

        self._kernel_prepare = getattr(
            utilities, self.kernel_name + '_prepare')
        self._kernel_prepare1 = getattr(
            utilities, self.kernel_name + '_prepare1')
        self._kernel_prepare2 = getattr(
            utilities, self.kernel_name + '_prepare2')

    def __call__(self, particles, dt, *args_opt, mpi_sort=None, verbose=False):
        """
        Applies the chosen pusher kernel by a time step dt, applies kinetic boundary conditions and performs MPI sorting.

        Parameters
        ----------
            particles : struphy.pic.particles.Particles6D
                The particles object holding the markers of shape (Np, 16) to push.

            dt : float
                The time step.

            args_opt : tuple
                Optional arguments needed for the pushing (typically spline coefficients for field evaluation).

            verbose : bool
                Whether to print some info or not.
        """
        # save initial etas and v_parallel in columns 9:13
        particles.markers[~particles.holes, 9:13] \
            = particles.markers[~particles.holes, 0:4]

        # prepare the iteration:
        self.kernel_prepare(particles.markers, dt, *self.args_fem,
                            *self.domain.args_map, *args_opt)
        particles.mpi_sort_markers()

        # start iteration
        for stage in range(self._maxiter):

            self.kernel_prepare1(particles.markers, dt, *self.args_fem,
                                 *self.domain.args_map, *args_opt)
            particles.mpi_sort_markers()

            self.kernel_prepare2(particles.markers, dt, *self.args_fem,
                                 *self.domain.args_map, *args_opt)
            particles.mpi_sort_markers()

            self.kernel(particles.markers, dt, stage, self._maxiter, self._tol,
                        *self.args_fem, *self.domain.args_map, *args_opt)
            particles.mpi_sort_markers()

            if stage == self._maxiter-1 and verbose:
                not_converged = np.logical_not(particles.markers[:, 23] == -1.)
                print('Number of not converged particles:',
                      np.count_nonzero(not_converged))
                print('Non converged partices:',
                      particles.markers[not_converged, 9:13])
                print('Number of iterations', np.average(
                      particles.markers[~particles.holes, 14])+1)
                print('Number of lost markers:',
                      particles.n_lost_markers)
                print()

        # clear buffer columns 9-24 for multi-stage pushers
        particles.markers[~particles.holes, 9:25] = 0.

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
    def args_fem(self):
        """ FEM and MPI related arguments taken by all pushers.
        """
        return self._args_fem

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
    def kernel_prepare(self):
        """ A preparation kernel.
        """
        return self._kernel_prepare

    @property
    def kernel_prepare1(self):
        """ A preparation kernel.
        """
        return self._kernel_prepare1

    @property
    def kernel_prepare2(self):
        """ A preparation kernel.
        """
        return self._kernel_prepare2
