import numpy as np


class equilibrium_pic:
    
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
    # distribution function on logical domain used for control variate
    # -----------------------------------------------
    def fh0_eq(self, eta1, eta2, eta3, vx, vy, vz):
        
        nh = self.nh0 - 0*eta1
        
        fh_out = nh/(np.pi**(3/2)*self.vth**3)*np.exp(-(vx - self.v0x)**2/self.vth**2 - (vy - self.v0y)**2/self.vth**2 - (vz - self.v0z)**2/self.vth**2)
        
        return fh_out
    
    # -----------------------------------------------
    # 0-th moment of distribution function
    # -----------------------------------------------
    def nh_eq(self, eta1):
        
        nh_out = self.nh0 - 0*eta1
        
        return nh_out
    
    # -----------------------------------------------
    # 1-st moment of distribution function
    # -----------------------------------------------
    def jh_eq_x(self, x, y, z):
        
        jh_x_out = self.nh0*self.v0x - 0*x
        
        return jh_x_out
    
    def jh_eq_y(self, x, y, z):
        
        jh_y_out = self.nh0*self.v0y - 0*x
        
        return jh_y_out
    
    def jh_eq_z(self, x, y, z):
        
        jh_z_out = self.nh0*self.v0z - 0*x
        
        return jh_z_out
    
    # -----------------------------------------------
    # total energy of distribution function
    # -----------------------------------------------
    def eh_eq(self):
        
        if self.domain.kind_map == 10:
            
            Lx = self.domain.params_map[0]
            Ly = self.domain.params_map[1]
            Lz = self.domain.params_map[2]
            
            eh_out = self.nh0/2 * Lx*Ly*Lz * (self.v0x**2 + self.v0y**2 + self.v0z**2 + 3*self.vth**2/2)
            
        elif self.domain.kind_map == 12 or self.domain.kind_map == 13:
            
            Lx = self.domain.params_map[0]
            Ly = self.domain.params_map[1]
            Lz = self.domain.params_map[3]
            
            eh_out = self.nh0/2 * Lx*Ly*Lz * (self.v0x**2 + self.v0y**2 + self.v0z**2 + 3*self.vth**2/2)
            
        else:
            
            eh_out = 0.
            
        return eh_out