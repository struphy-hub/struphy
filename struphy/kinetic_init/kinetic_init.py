import numpy as np
import h5py
import scipy.special as sp

from struphy.kinetic_equil.analytical import gaussian

from struphy.pic import sampling
from struphy.pic import sobol_seq 

class Initialize_markers:
    '''
    Initialize markers with a Gaussian 3-form (s3) and compute weights from 0-forms, w0 = f0/s0. No importance sampling.

    Parameters
    ----------
    '''

    def __init__(self, DOMAIN, EQ_KINETIC_L, general_init, params_init, params_markers, mpi_comm):
        
        self.DOMAIN = DOMAIN
        self.EQ     = EQ_KINETIC_L

        self.init_type   = general_init['type']
        self.init_coords = general_init['coords']

        self.mpi_comm       = mpi_comm
        self.Np             = params_markers['Np'] # number of markers
        self.Np_loc         = int(self.Np / mpi_comm.Get_size())
        self.params_markers = params_markers

        if self.init_type == 'modes_k':
            self.modes_k  = [ params_init['kx'], 
                              params_init['ky'], 
                              params_init['kz'] ]
            self.amp      =   params_init['amp']
            assert np.all(len(self.amp) == len(self.modes_k[0]) == len(self.modes_k[1]) == len(self.modes_k[2]))

        elif self.init_type == 'modes_mn':
            self.modes_mn = [ params_init['modes_m'],
                              params_init['modes_n'] ]
            self.amp      =   params_init['amp']

        elif self.init_type == 'noise':
            self.plane    =   params_init['plane']

        else:
            raise ValueError('Kinetic initial condition not supported.')

        mpi_size = self.mpi_comm.Get_size()
        mpi_rank = self.mpi_comm.Get_rank()
        
        # initialize particle arrays
        self.particles_loc = np.empty((7, self.Np_loc), dtype=float)    # particles of each process
        self.w0_loc        = np.empty(    self.Np_loc , dtype=float)    # weights for each process: hat_f_ini(eta_0, v_0)/hat_s_ini(eta_0, v_0)
        self.s0_loc        = np.empty(    self.Np_loc , dtype=float)    # initial sampling density: hat_s_ini(eta_0, v_0) for each process
        f0_loc        = np.empty(    self.Np_loc , dtype=float)

        if mpi_rank == 0:
            self.particles_recv = np.empty((7, self.Np_loc), dtype=float)
            self.w0_recv        = np.empty(    self.Np_loc , dtype=float)    
            self.s0_recv        = np.empty(    self.Np_loc , dtype=float)
        else:
            self.particles_recv = None
            self.w0_recv        = None    
            self.s0_recv        = None

        
        # Step 1: create numbers in (0, 1)
        # pseudo-random
        if self.params_markers['loading'] == 'pseudo_random':
            
            np.random.seed(self.params_markers['seed'])

            for i in range(mpi_size):
                temp = np.random.rand(self.Np_loc, 6)

                if i == mpi_rank:
                    self.particles_loc[:6] = temp.T
                    break

            del temp
        # plain sobol numbers (skip first 1000 numbers)
        elif self.params_markers['loading'] == 'sobol_standard':
            self.particles_loc[:6] = sobol_seq.i4_sobol_generate(6, self.Np_loc, 1000 + self.Np_loc*mpi_rank).T 
        # symmetric sobol numbers in all 6 dimensions (skip first 1000 numbers) 
        elif self.params_markers['loading'] == 'sobol_antithetic':
            sampling.set_particles_symmetric(sobol_seq.i4_sobol_generate(6, 
                                             self.Np_loc//64, 1000 + self.Np_loc//64*mpi_rank), self.particles_loc, self.Np_loc)  
        # load numbers from an external files
        elif self.params_markers['loading'] == 'external':

            if mpi_rank == 0:
                file = h5py.File(self.params_markers['dir_particles'], 'r')

                self.particles_loc[:, :] = file['particles'][0, :6, :self.Np_loc]

                for i in range(1, mpi_size):
                    self.mpi_comm.Send(file['particles'][0, :6, i*self.Np_loc:(i + 1)*self.Np_loc], dest=i, tag=11)         
            else:
                self.mpi_comm.Recv(self.particles_loc, source=0, tag=11)
        else:
            raise ValueError('Specified particle loading method does not exist!')
        
        # Step 2: inversion of Gaussian in velocity space (sh3_ini from above)
        if self.params_markers['loading'] != 'external':
            self.particles_loc[3] = sp.erfinv(2*self.particles_loc[3] - 1)*self.EQ.KINETC_P.vth[0] + self.EQ.KINETC_P.shifts[0]
            self.particles_loc[4] = sp.erfinv(2*self.particles_loc[4] - 1)*self.EQ.KINETC_P.vth[1] + self.EQ.KINETC_P.shifts[1]
            self.particles_loc[5] = sp.erfinv(2*self.particles_loc[5] - 1)*self.EQ.KINETC_P.vth[2] + self.EQ.KINETC_P.shifts[2]

        # check if all particle positions are in [0, 1]^3
        if np.any(self.particles_loc[:3]  >= 1.0) or np.any(self.particles_loc[:3] <= 0.0):
            raise ValueError('There are particles outside of the logical domain - aborting...')

        # Step 3: initial weights
        self.s0_loc[:] = self.sh0_ini(self.particles_loc[0], self.particles_loc[1], self.particles_loc[2], self.particles_loc[3], self.particles_loc[4], self.particles_loc[5])
        f0_loc[:] = self.fh0_ini(self.particles_loc[0], self.particles_loc[1], self.particles_loc[2], self.particles_loc[3], self.particles_loc[4], self.particles_loc[5])
        self.w0_loc[:] = f0_loc / self.s0_loc

        if self.params_markers['control']:
            self.particles_loc[6] = self.w0_loc - 1./self.s0_loc * self.EQ.fh0_eq(self.particles_loc[0],
                                                                                  self.particles_loc[1], 
                                                                                  self.particles_loc[2], 
                                                                                  self.particles_loc[3], 
                                                                                  self.particles_loc[4], 
                                                                                  self.particles_loc[5])
        else:
            self.particles_loc[6] = self.w0_loc

    # density for modes_k
    def nh_tot(self, x, y, z):
        '''Total normalized EP density nh/nh_eq = 1 + sum_i amp_i sin(k_i*x).'''

        value = 1.*x

        if self.init_type == 'modes_k' :
            for i in range(len(self.amp)):
                value += self.amp[i]*np.sin(self.modes_k[0][i]*x + self.modes_k[1][i]*y + self.modes_k[2][i]*z)

        return value
        
    # initial distribution function on logical domain (with possible density perturbation)
    def fh0_ini(self, eta1, eta2, eta3, vx, vy, vz):

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape

        if self.init_coords == 'logical':
            value = self.nh_tot(eta1, eta2, eta3) * self.EQ.fh0_eq(eta1, eta2, eta3, vx, vy, vz)

        elif self.init_coords == 'physical':

            # must do evaluation here, because pull needs an array as input (not a 6d callable)
            X = self.DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
            Y = self.DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
            Z = self.DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
            fun = self.nh_tot(X, Y, Z) * self.EQ.KINETC_P.fh_eq_phys(X, Y, Z, vx, vy, vz)

            value = self.DOMAIN.pull(fun, eta1, eta2, eta3, '0_form', flat_eval=True)

        else:
            raise ValueError('Coordinates for fh0_ini not supported.')

        return value
        
    # sampling distribution s3 (3-form) on logical domain, always a Gaussian (no importance sampling)
    def sh3_ini(self, eta1, eta2, eta3, vx, vy, vz):
        '''Gaussian velocity distribution for sampling markers. 
        Parameters are such that Gaussian is close to kinetic equilibirum.'''

        distr = gaussian.Gaussian_3d({'vth_x': self.EQ.KINETC_P.vth[0],
                                      'vth_y': self.EQ.KINETC_P.vth[1],
                                      'vth_z': self.EQ.KINETC_P.vth[2],
                                      'v0_x' : self.EQ.KINETC_P.shifts[0],
                                      'v0_y' : self.EQ.KINETC_P.shifts[1],
                                      'v0_z' : self.EQ.KINETC_P.shifts[2],
                                      })

        return distr.velocity_distribution(eta1, eta2, eta3, vx, vy, vz)

    # sampling distribution s0 (0-form) to compute weights
    def sh0_ini(self, eta1, eta2, eta3, vx, vy, vz):
        '''Sampling distribution as 0-form.'''
        fun = self.sh3_ini(eta1, eta2, eta3, vx, vy, vz) 
        return self.DOMAIN.transformation(fun, eta1, eta2, eta3, '3_to_0', flat_eval=True)

    # when using control variate
    def update_weights(self):
        self.particles_loc[6] = self.w0_loc - 1./self.s0_loc * self.EQ.fh0_eq(self.particles_loc[0], 
                                                                              self.particles_loc[1], 
                                                                              self.particles_loc[2], 
                                                                              self.particles_loc[3], 
                                                                              self.particles_loc[4], 
                                                                              self.particles_loc[5])    
        
        