import struphy.pic.pusher_kernels as pushers

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
    """
    
    def __init__(self, derham, domain, pusher_name, stage_num = 1):
        
        self._derham = derham
        self._domain = domain
        self._stage_num = stage_num
        self._rank = derham.comm.Get_rank()
        
        # get FEM information
        self._args_fem = (np.array(derham.p), 
                          derham.V0.knots[0], derham.V0.knots[1], derham.V0.knots[2],
                          np.array(derham.V0.vector_space.starts), 
                          np.array(derham.V1.vector_space.starts),
                          np.array(derham.V2.vector_space.starts),
                          np.array(derham.V3.vector_space.starts))
        
        # select pusher
        self._pusher_name = pusher_name
        self._pusher = getattr(pushers, self._pusher_name) 
          
        
    def __call__(self, particles, dt, *args_opt, do_mpi_sort=False, verbose=False):
        """
        Applies the chosen particle pusher by a time step dt.
        
        Parameters
        ----------
            particles : struphy.pic.particles.Particles6D
                The particles object holding the markers of shape (Np, 16) to push.
                
            dt : float
                The time step.
                
            args_opt : tuple
                Optional arguments needed for the pushing (typically spline coefficients for field evaluation).
                
            do_mpi_sort : bool
                Whether to do a marker sorting according to the MPI decomposition (needed when marker positions change during push).
        """
        # save eta
        if self._stage_num > 1:
            particles.markers[~particles.holes, 9:12] = particles.markers[~particles.holes, 0:3]

        for step in range(self._stage_num):
            self._pusher(particles.markers, dt, step, *self.args_fem, *self.domain.args_map, *args_opt)

            if do_mpi_sort: 
                self._derham.comm.Barrier()
                particles.send_recv_markers()
                self._derham.comm.Barrier()
                
            if self._rank == 0 and verbose: print(self._pusher_name, 'done. (stage :', step+1, ')')
        
        if self._rank == 0 and verbose: print()

        # clear the markers
        if self._stage_num > 1:
            particles.markers[~particles.holes, 9:15] = 0.
        
        
    @property
    def derham(self):
        return self._derham
    
    @property
    def domain(self):
        return self._domain
    
    @property
    def pusher_name(self):
        return self._pusher_name
    
    @property
    def args_fem(self):
        return self._args_fem