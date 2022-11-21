import struphy.pic.pusher_kernels as pushers
import struphy.pic.utilities_kernels as utilities
from struphy.pic.particles import apply_kinetic_bc

import numpy as np

class Pusher:
    """
    Wrapper class for particle pushing.
    
    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.
            
        pusher_name : str
            The name of the pusher in the file struphy.pic.pusher_kernels.
            
        n_stages : int
            Number of stages of the pusher (e.g. 4 for RK4)
    """
    
    def __init__(self, derham, domain, pusher_name, n_stages=1):
        
        self._derham = derham
        self._domain = domain
        self._n_stages = n_stages
        
        # get FEM information
        self._args_fem = (np.array(derham.p), 
                          derham.V0.knots[0], derham.V0.knots[1], derham.V0.knots[2],
                          np.array(derham.V0.vector_space.starts), 
                          np.array(derham.V1.vector_space.starts),
                          np.array(derham.V2.vector_space.starts),
                          np.array(derham.V3.vector_space.starts))
        
        # select pusher kernel
        self._pusher_name = pusher_name
        self._pusher = getattr(pushers, self._pusher_name) 
          
        
    def __call__(self, particles, dt, *args_opt, bc=None, mpi_sort=None, verbose=False):
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

            bc : list[str]
                Kinetic boundary conditions in each direction (periodic, reflect or remove).
                
            mpi_sort : str
                When to do MPI sorting:
                    * None : no sorting at all.
                    * each : sort markers after each stage.
                    * last : sort markers after last stage.
                
            verbose : bool
                Whether to print some info or not.
        """
        # save initial etas in columns 9-11
        particles.markers[~particles.holes, 9:12] = particles.markers[~particles.holes, 0:3]

        # in case of Particles5D, save initial v in columns 12
        if particles.kinds == 'Particles5D':
            particles.markers[~particles.holes, 12] = particles.markers[~particles.holes, 3]

        for stage in range(self._n_stages):
            self._pusher(particles.markers, dt, stage, *self.args_fem, *self.domain.args_map, *args_opt)

            # apply boundary conditions to markers
            if bc is not None: 
                apply_kinetic_bc(particles.markers, particles.holes, self.domain, bc)

            # sort markers according to domain decomposition
            if mpi_sort == 'each': 
                particles.mpi_sort_markers()

            # print stage info
            if self._derham.comm.Get_rank() == 0 and verbose: 
                print(self._pusher_name, 'done. (stage :', stage + 1, ')')

        # sort markers according to domain decomposition
        if mpi_sort == 'last': 
            particles.mpi_sort_markers()
                
        # clear buffer columns 9-14 for multi-stage pushers
        particles.markers[~particles.holes, 9:15] = 0.

        if particles.kinds == 'Particles5D':
            particles.markers[~particles.holes, 9:17] = 0.
        
        
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
    def pusher_name(self):
        """ The name of the pyccelized pusher kernel.
        """
        return self._pusher_name
    
    

class ButcherTableau:
    """
    Butcher tableau for explicit s-stage Runge-Kutta methods of the form

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


class Pusher_iteration:
    """
    Wrapper class for particle pushing with iterative solver.
    
    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.
            
        pusher_name : str
            The name of the pusher in the file struphy.pic.pusher_kernels.
            
        n_stages : int
            Number of stages of the pusher (e.g. 4 for RK4)
    """
    def __init__(self, derham, domain, pusher_name, maxiter=100, tol=1.e-8):

        
        self._derham = derham
        self._domain = domain
        self._maxiter = maxiter
        self._tol = tol
        
        # get FEM information
        self._args_fem = (derham.domain_array[derham.comm.Get_rank(),:],
                          np.array(derham.p), 
                          derham.V0.knots[0], derham.V0.knots[1], derham.V0.knots[2],
                          np.array(derham.V0.vector_space.starts), 
                          np.array(derham.V1.vector_space.starts),
                          np.array(derham.V2.vector_space.starts),
                          np.array(derham.V3.vector_space.starts))
        
        # select kernels
        self._pusher_name = pusher_name
        self._pusher = getattr(pushers, self._pusher_name)
        self._eval_S_and_I = getattr(utilities, self._pusher_name + '_eval_S_and_I')
        self._initial_guess = getattr(utilities, self._pusher_name + '_initial_guess')
          
        
    def __call__(self, particles, dt, *args_opt, bc=None, mpi_sort=None, verbose=False):
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

            bc : list[str]
                Kinetic boundary conditions in each direction (periodic, reflect or remove).
                
            mpi_sort : str
                When to do MPI sorting:
                    * None : no sorting at all.
                    * each : sort markers after each stage.
                    * last : sort markers after last stage.
                
            verbose : bool
                Whether to print some info or not.
        """
        # TODO: only applicable to Particle5D case! if we have any iterative solver with Particle6D then we should generalize
        # TODO: maybe we can modulize ... discrete gradient method, fixed-point iterative solver ...

        # is_converged = False

        # save initial etas in columns 9:12
        particles.markers[~particles.holes, 9:13] = particles.markers[~particles.holes, 0:4]

        # before iteration:
        # TODO: we can unify eval_S_I and initial_guess
        # save S(z_n) at 13:19 and I(z_n) at 19
        self._eval_S_and_I(particles.markers, *self.args_fem, *self.domain.args_map, *args_opt)
        
        # save initial guess in columns 0:3
        self._initial_guess(particles.markers, dt, *self.args_fem, *self.domain.args_map, *args_opt)

        # start iteration
        for stage in range(self._maxiter):

            self._pusher(particles.markers, dt, stage, self._tol, *self.args_fem, *self.domain.args_map, *args_opt)

            # apply boundary conditions to markers
            if bc is not None: 
                apply_kinetic_bc(particles.markers, particles.holes, self.domain, bc)

            # sort markers according to domain decomposition
            if mpi_sort == 'each': 
                particles.mpi_sort_markers()

            # print stage info
            if self._derham.comm.Get_rank() == 0 and verbose: 
                print(self._pusher_name, 'done. (stage :', stage + 1, ')')

        # sort markers according to domain decomposition
        particles.mpi_sort_markers()
                
        # clear buffer columns 9-14 for multi-stage pushers
        particles.markers[~particles.holes, 9:23] = 0.
        
        
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
    def pusher_name(self):
        """ The name of the pyccelized pusher kernel.
        """
        return self._pusher_name