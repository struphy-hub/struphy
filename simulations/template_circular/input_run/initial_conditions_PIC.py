import numpy as np
import scipy.special as sp
from scipy.integrate import quad

from mpi4py import MPI
import struphy.geometry.mappings_3d as mapping
import struphy.geometry.pullback_3d as pull


class initial_pic:
    
    def __init__(self, domain, alpha0, delta, vth):
        
        # geometric parameters
        self.domain = domain
        
        # parameters for anisotropic pitch-angle distribution function
        self.alpha0 = alpha0
        self.delta  = delta
        self.vth    = vth
        
        self.A = 1/quad(lambda eta1 : self.nh_ini(eta1)*self.J(eta1), 0., 1.)[0]
        self.D = sp.erf((1 - np.cos(self.alpha0))/self.delta)
        self.C = self.D + sp.erf((1 + np.cos(self.alpha0))/self.delta)
        
        # find maximum of p1 for acceptance-rejection method
        pts = np.linspace(0., 1., 501)
        self.c = max(self.A*self.nh_ini(pts)*pts) + 0.05
        
        
        
    # -----------------------------------------------
    # anisotropy function
    # -----------------------------------------------
    def theta(self, alpha):
        
        if self.delta == np.inf:
            out = 1. - 0*alpha
        else:
            out = 4/(self.delta*np.sqrt(np.pi)*self.C)*np.exp(-(np.cos(alpha) - np.cos(self.alpha0))**2/self.delta**2)
        
        return out
    
    # -----------------------------------------------
    # number density on logical domain
    # -----------------------------------------------
    def nh_ini(self, eta1):
        
        nh_out = 0.521298*np.exp(-0.198739/0.298228*np.tanh((eta1 - 0.49123)/0.198739))
        
        return nh_out
    
    # -----------------------------------------------
    # thermal velocity on logical domain
    # -----------------------------------------------
    def vth_ini(self, eta1):
        
        vth_out = self.vth - 0*eta1
        
        return vth_out
    
    # -----------------------------------------------
    # distribution function on logical domain
    # -----------------------------------------------
    def fh0_ini(self, eta1, eta2, eta3, vx, vy, vz):
        
        out = self.nh_ini(eta1)/(np.pi**(3/2)*self.vth_ini(eta1)**3)*np.exp(-(vx**2 + vy**2 + vz**2)/self.vth_ini(eta1)**2)
        
        return out
    
    # -----------------------------------------------
    # approximate Jacobian in eta_1-direction 
    # -----------------------------------------------
    def J(self, eta1):
        
        out = 1*eta1
        
        return out
    
    # -----------------------------------------------
    # sampling distribution on logical domain
    # -----------------------------------------------
    def sh0_ini(self, eta1, eta2, eta3, vx, vy, vz):
        
        det_df = self.domain.evaluate(eta1, eta2, eta3, 'det_df', 'flat')
        
        out = self.A*self.nh_ini(eta1)*self.J(eta1)/(det_df*np.pi**(3/2)*self.vth_ini(eta1)**3)*np.exp(-(vx**2 + vy**2 + vz**2)/self.vth_ini(eta1)**2)

        return out
    
    
    # -----------------------------------------------
    # cumulative distribution function P2(eta1, v) and its derivative
    # -----------------------------------------------
    def P2(self, eta1, v, d):
        
        if   d == 0:
            out = sp.erf(v/self.vth_ini(eta1)) - 1/(np.sqrt(np.pi)*self.vth_ini(eta1))*2*v*np.exp(-v**2/self.vth_ini(eta1)**2)
        
        elif d == 1:
            out = 4*v**2*np.exp(-v**2/self.vth_ini(eta1)**2)/(np.sqrt(np.pi)*self.vth_ini(eta1)**3)
    
        return out
    
    
    # -----------------------------------------------
    # load particles in alpha, v, theta - space
    # -----------------------------------------------
    def load(self, particles_loc, mpi_comm, seed, tol, max_it):
        
        mpi_size = mpi_comm.Get_size()
        mpi_rank = mpi_comm.Get_rank()
        
        Np_loc = particles_loc.shape[1]
        Np     = Np_loc*mpi_size
        
        # set seed for all MPI processes
        np.random.seed(seed)
        
        # ------ get eta1 with acceptance-rejection method (only MPI rank 0) ------
        counter_all = np.array([0])
        
        if mpi_rank == 0:
            
            eta1 = np.zeros(Np, dtype=float)
            
            counter = 0
            
            while counter < Np:

                u = np.random.rand()
                e = np.random.rand()

                if u < self.A*self.nh_ini(e)*self.J(e)/self.c:
                    eta1[counter] = e
                    counter += 1
                
                counter_all += 1
                
            particles_loc[0, :] = eta1[:Np_loc]
            
            for i in range(1, mpi_size):
                mpi_comm.Send(eta1[i*Np_loc:(i + 1)*Np_loc], dest=i, tag=1607)
                mpi_comm.Send(counter_all, dest=i, tag=1608)
                
        else:
            mpi_comm.Recv(particles_loc[0, :], source=0, tag=1607)
            mpi_comm.Recv(counter_all, source=0, tag=1608)
            
            temp = np.random.rand(2*counter_all[0])
            
            del temp
        # ---------------------------------------------------------------------------
        
        
        # --- get eta2, eta3, v, alpha and theta with inverse transform sampling ----
        
        # create random numbers in (0, 1)
        for i in range(mpi_size):
            temp = np.random.rand(Np_loc, 5)

            if i == mpi_rank:
                particles_loc[1:6] = temp.T
                break

        del temp
            
            
        # get v with Newton method
        for i in range(Np_loc):
            
            # function of which we want to find the root
            F = lambda v : self.P2(particles_loc[0, i], v, 0) - particles_loc[3, i]
            
            # inner Newton loop
            v_i = 1.
            
            counter = 0
            
            while True:
                
                if abs(F(v_i)) < tol or counter > max_it:
                    break
                
                v_i = v_i - F(v_i)/self.P2(particles_loc[0, i], v_i, 1)
                counter += 1
 
            particles_loc[3, i] = v_i
            
            
        # get alpha with inversion of cumulative distribution function
        if self.delta == np.inf:
            particles_loc[4, :] = np.arccos(1 - 2*particles_loc[4])
        else:
            particles_loc[4, :] = np.arccos(np.cos(self.alpha0) - self.delta*sp.erfinv(self.C*particles_loc[4] - self.D))
        
        
        # get theta with inversion of cumulative distribution function
        particles_loc[5, :] = 2*np.pi*particles_loc[5]
        # -----------------------------------------------------------------------------