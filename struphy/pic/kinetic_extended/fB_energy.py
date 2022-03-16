# import pyccel decorators
from pyccel.decorators import types

import struphy.pic.kinetic_extended.fB_energy_kernels as ker

import numpy as np

from mpi4py import MPI

# ============================================================


class Energy:
    '''energy of several parts
    
    Parameters
    ----------
        temperature : double 
            electron temperature
            
        DOMAIN : obj
            Domain object from geometry/domain_3d.

        SPACES : obj
            FEEC self.SPACES.

        Np : int
            Number of all particles.
        
        GATHER : obj
            Object to store grid points from particles
        
        KIN : obj
            Object to store information of particles

        MHD : obj
            Object to store information of density and magnetic field

        MPI_COMM : obj 
            Environment of MPI
    '''

    def __init__(self, temperature, DOMAIN, SPACES, GATHER, KIN, MHD, TEMP, MPI_COMM):

        self.DOMAIN     = DOMAIN
        self.SPACES     = SPACES
        self.Np_loc     = KIN.Np_loc
        self.Np         = KIN.Np
        self.GATHER     = GATHER
        self.MHD        = MHD
        self.mpi_rank   = MPI_COMM.Get_rank()
        self.Nel        = SPACES.Nel
        self.n_quad     = SPACES.n_quad
        self.wts        = SPACES.wts
        self.temperature= temperature # electron temperature
        self.TEMP       = TEMP
    
        if self.mpi_rank == 0:
            self.magnetic    = np.zeros(1, dtype=float)
            self.kinetic_loc = np.zeros(1, dtype=float)
            self.kinetic     = np.zeros(1, dtype=float)
            self.thermal     = np.zeros(1, dtype=float)
            self.total       = np.zeros(1, dtype=float)
        else:
            self.magnetic    = None
            self.kinetic_loc = np.zeros(1, dtype=float)
            self.kinetic     = None
            self.thermal     = None
            self.total       = None

    def cal_total(self, KIN, MPI_COMM):
        ker.kinetic(self.kinetic_loc, KIN.particles_loc, self.Np_loc)
        MPI_COMM.Reduce(self.kinetic_loc/self.Np, self.kinetic, op=MPI.SUM, root=0)
        MPI_COMM.Barrier()
        if self.mpi_rank == 0:
            self.magnetic[0] = 1/2*np.concatenate((self.MHD.b1.flatten(), self.MHD.b2.flatten(), self.MHD.b3.flatten())).dot(self.SPACES.M1.dot(np.concatenate((self.MHD.b1.flatten(), self.MHD.b2.flatten(), self.MHD.b3.flatten()))))
            ker.thermal(self.SPACES.Nel, self.n_quad[0], self.n_quad[1], self.n_quad[2], self.thermal, self.wts[0], self.wts[1], self.wts[2], self.GATHER.gather_quadrature, self.GATHER.quadrature_log, self.TEMP.df_det)
            self.total[0] =  self.kinetic[0] + self.magnetic[0] #+ self.temperature * self.thermal[0] 







