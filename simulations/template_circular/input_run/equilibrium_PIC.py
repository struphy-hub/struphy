import numpy as np
import scipy.special as sp


class equilibrium_pic:
    
    def __init__(self, domain, alpha0, delta, vth):
        
        # geometric parameters
        self.domain = domain
        
        # parameters for anisotropic pitch-angle distribution function
        self.alpha0 = alpha0
        self.delta  = delta
        self.vth    = vth
        
        self.D = sp.erf((1 - np.cos(self.alpha0))/self.delta)
        self.C = self.D + sp.erf((1 + np.cos(self.alpha0))/self.delta)
    
    
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
    def nh_eq(self, eta1):
        
        nh_out = 0.521298*np.exp(-0.198739/0.298228*np.tanh((eta1 - 0.49123)/0.198739))
        
        return nh_out
    
    # -----------------------------------------------
    # thermal velocity on logical domain
    # -----------------------------------------------
    def vth_eq(self, eta1):
        
        vth_out = self.vth - 0*eta1
        
        return vth_out
    
    # -----------------------------------------------
    # distribution function on logical domain used for control variate
    # -----------------------------------------------
    def fh0_eq(self, eta1, eta2, eta3, vx, vy, vz):
        
        out = self.nh_eq(eta1)/(np.pi**(3/2)*self.vth_eq(eta1)**3)*np.exp(-(vx**2 + vy**2 + vz**2)/self.vth_eq(eta1)**2)
        
        return out
    
    # -----------------------------------------------
    # 1-st moment of distribution function
    # -----------------------------------------------
    def jh_eq_x(self, x, y, z):
        
        jh_x_out = 0*x
        
        return jh_x_out
    
    def jh_eq_y(self, x, y, z):
        
        jh_y_out = 0*y
        
        return jh_y_out
    
    def jh_eq_z(self, x, y, z):
        
        jh_z_out = 0*z
        
        return jh_z_out