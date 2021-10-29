import numpy as np
import scipy.special as sp

<<<<<<< HEAD
import hylife.utilitis_PIC.sobol_seq as sobol
import hylife.utilitis_PIC.sampling as pic_sample
=======
import struphy.geometry.mappings_3d as mapping
import struphy.geometry.pullback_3d as pull
>>>>>>> Renamed hylife -> struphy, utilitis_FEEC -> feec, utilitis_PIC -> pic

import h5py

from mpi4py import MPI


class initial_pic:
    
    def __init__(self, domain, nh0=1., v0x=0., v0y=0., v0z=1., vth=1.):
        
        # geometric parameters
        self.domain = domain
        
        # parameters for distribution function
        self.nh0 = nh0
        self.v0x = v0x
        self.v0y = v0y
        self.v0z = v0z
        self.vth = vth
        
    # -----------------------------------------------
    # distribution function on logical domain used as an initial condition
    # -----------------------------------------------
    def fh0_ini(self, eta1, eta2, eta3, vx, vy, vz):
        
        nh = self.nh0 - 0*eta1
        
        fh_out = nh/(np.pi**(3/2)*self.vth**3)*np.exp(-(vx - self.v0x)**2/self.vth**2 - (vy - self.v0y)**2/self.vth**2 - (vz - self.v0z)**2/self.vth**2)
        
        return fh_out
    
    # -----------------------------------------------
    # sampling distribution on logical domain (0-form)
    # -----------------------------------------------
    def sh0_ini(self, eta1, eta2, eta3, vx, vy, vz):

        det_df = self.domain.evaluate(eta1, eta2, eta3, 'det_df', 'flat')

        sh_out = 1/(np.pi**(3/2)*self.vth**3*det_df)*np.exp(-(vx - self.v0x)**2/self.vth**2 - (vy - self.v0y)**2/self.vth**2 - (vz - self.v0z)**2/self.vth**2)

        return sh_out
    
    # -----------------------------------------------
    # load particles
    # -----------------------------------------------
    def load(self, particles_loc, mpi_comm, seed, loading, dir_particles):
        
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
        
        Np_loc = particles_loc.shape[1]
        Np     = Np_loc*mpi_size
        
        # ------------------------- numbers in (0, 1) ------------------------------
        
        # pseudo-random
        if loading == 'pseudo_random':
            
            np.random.seed(seed)

            for i in range(mpi_size):
                temp = np.random.rand(Np_loc, 6)

                if i == mpi_rank:
                    particles_loc[:6] = temp.T
                    break

            del temp

        # plain sobol numbers (skip first 1000 numbers)
        elif loading == 'sobol_standard':
            particles_loc[:6] = sobol.i4_sobol_generate(6, Np_loc, 1000 + Np_loc*mpi_rank).T 

        # symmetric sobol numbers in all 6 dimensions (skip first 1000 numbers) 
        elif loading == 'sobol_antithetic':
            pic_sample.set_particles_symmetric(sobol.i4_sobol_generate(6, Np_loc//64, 1000 + Np_loc//64*mpi_rank), particles_loc, Np_loc)  

        # load numbers from an external files
        elif loading == 'external':

            if mpi_rank == 0:
                file = h5py.File(dir_particles, 'r')

                particles_loc[:, :] = file['particles'][0, :6, :Np_loc]

                for i in range(1, mpi_size):
                    mpi_comm.Send(file['particles'][0, :6, i*Np_loc:(i + 1)*Np_loc], dest=i, tag=11)         
            else:
                mpi_comm.Recv(particles_loc, source=0, tag=11)

        else:
            raise ValueError('Specified particle loading method does not exist!')
            
        # -----------------------------------------------------------------------------
        
        
        # inversion of cumulative distribution function in velocity space
        if loading != 'external':
            particles_loc[3] = sp.erfinv(2*particles_loc[3] - 1)*self.vth + self.v0x
            particles_loc[4] = sp.erfinv(2*particles_loc[4] - 1)*self.vth + self.v0y
            particles_loc[5] = sp.erfinv(2*particles_loc[5] - 1)*self.vth + self.v0z