import numpy as np
import sympy as sy


class mappings:
    
    def __init__(self, mapping):
        
        if mapping[0] == 'hollow cylinder':
            
            self.R1 = mapping[1]
            self.R2 = mapping[2]
            self.dR = self.R2 - self.R1
            
            self.Lz = mapping[3]
            
            Fx = lambda r, phi, z : (r*self.dR + self.R1)*np.cos(2*np.pi*phi)
            Fy = lambda r, phi, z : (r*self.dR + self.R1)*np.sin(2*np.pi*phi)
            Fz = lambda r, phi, z : self.Lz*z
            
            self.F = [Fx, Fy, Fz]
            
            DF11 = lambda r, phi, z : self.dR*np.cos(2*np.pi*phi)
            DF12 = lambda r, phi, z : -2*np.pi*(r*self.dR + self.R1)*np.sin(2*np.pi*phi)
            DF13 = lambda r, phi, z : 0*r
            
            DF21 = lambda r, phi, z : self.dR*np.sin(2*np.pi*phi)
            DF22 = lambda r, phi, z : 2*np.pi*(r*self.dR + self.R1)*np.cos(2*np.pi*phi)
            DF23 = lambda r, phi, z : 0*r
            
            DF31 = lambda r, phi, z : 0*r
            DF32 = lambda r, phi, z : 0*r
            DF33 = lambda r, phi, z : self.Lz*np.ones(r.shape)
            
            self.DF = [[DF11, DF12, DF13], [DF21, DF22, DF23], [DF31, DF32, DF33]]
            
            
            G11 = lambda r, phi, z : self.dR**2*np.ones(r.shape)
            G12 = lambda r, phi, z : 0*r
            G13 = lambda r, phi, z : 0*r
            
            G21 = lambda r, phi, z : 0*r
            G22 = lambda r, phi, z : (2*np.pi)**2*(r*self.dR + self.R1)**2
            G23 = lambda r, phi, z : 0*r
            
            G31 = lambda r, phi, z : 0*r
            G32 = lambda r, phi, z : 0*r
            G33 = lambda r, phi, z : self.Lz**2*np.ones(r.shape)
            
            self.G = [[G11, G12, G13], [G21, G22, G23], [G31, G32, G33]]
            
            
            Ginv11 = lambda r, phi, z : 1/self.dR**2*np.ones(r.shape)
            Ginv12 = lambda r, phi, z : 0*r
            Ginv13 = lambda r, phi, z : 0*r
            
            Ginv21 = lambda r, phi, z : 1/((2*np.pi)**2*(r*self.dR + self.R1)**2)
            Ginv22 = lambda r, phi, z : 0*r
            Ginv23 = lambda r, phi, z : 0*r
            
            Ginv31 = lambda r, phi, z : 0*r
            Ginv32 = lambda r, phi, z : 0*r
            Ginv33 = lambda r, phi, z : 1/self.Lz**2*np.ones(r.shape)
            
            self.Ginv = [[Ginv11, Ginv12, Ginv13], [Ginv21, Ginv22, Ginv23], [Ginv31, Ginv32, Ginv33]]
            
            
            self.g      = lambda r, phi, z : self.dR**2*self.Lz**2*(2*np.pi)**2*(r*self.dR + self.R1)**2
            self.g_sqrt = lambda r, phi, z : self.dR*self.Lz*2*np.pi*(r*self.dR + self.R1)
            
            
        if mapping[0] == 'slab':
            
            self.Lx = mapping[1]
            self.Ly = mapping[2]
            self.Lz = mapping[3]
            
            Fx = lambda x, y, z : self.Lx*x
            Fy = lambda x, y, z : self.Ly*y
            Fz = lambda x, y, z : self.Lz*z
            
            self.F = [Fx, Fy, Fz]
            
            DF11 = lambda x, y, z : self.Lx*np.ones(x.shape)
            DF12 = lambda x, y, z : 0*x
            DF13 = lambda x, y, z : 0*x
            
            DF21 = lambda x, y, z : 0*x
            DF22 = lambda x, y, z : self.Ly*np.ones(x.shape)
            DF23 = lambda x, y, z : 0*x
            
            DF31 = lambda x, y, z : 0*x
            DF32 = lambda x, y, z : 0*x
            DF33 = lambda x, y, z : self.Lz*np.ones(x.shape)
            
            self.DF = [[DF11, DF12, DF13], [DF21, DF22, DF23], [DF31, DF32, DF33]]
            
            
            DF11inv = lambda x, y, z : 1/self.Lx*np.ones(x.shape)
            DF12inv = lambda x, y, z : 0*x
            DF13inv = lambda x, y, z : 0*x
            
            DF21inv = lambda x, y, z : 0*x
            DF22inv = lambda x, y, z : 1/self.Ly*np.ones(x.shape)
            DF23inv = lambda x, y, z : 0*x
            
            DF31inv = lambda x, y, z : 0*x
            DF32inv = lambda x, y, z : 0*x
            DF33inv = lambda x, y, z : 1/self.Lz*np.ones(x.shape)
            
            self.DFinv = [[DF11inv, DF12inv, DF13inv], [DF21inv, DF22inv, DF23inv], [DF31inv, DF32inv, DF33inv]]
            
            
            G11 = lambda x, y, z : self.Lx**2*np.ones(x.shape)
            G12 = lambda x, y, z : 0*x
            G13 = lambda x, y, z : 0*x
            
            G21 = lambda x, y, z : 0*x
            G22 = lambda x, y, z : self.Ly**2*np.ones(x.shape)
            G23 = lambda x, y, z : 0*x
            
            G31 = lambda x, y, z : 0*x
            G32 = lambda x, y, z : 0*x
            G33 = lambda x, y, z : self.Lz**2*np.ones(x.shape)
            
            self.G = [[G11, G12, G13], [G21, G22, G23], [G31, G32, G33]]
            
            
            Ginv11 = lambda x, y, z : 1/self.Lx**2*np.ones(x.shape)
            Ginv12 = lambda x, y, z : 0*x
            Ginv13 = lambda x, y, z : 0*x
            
            Ginv21 = lambda x, y, z : 0*x
            Ginv22 = lambda x, y, z : 1/self.Ly**2*np.ones(x.shape)
            Ginv23 = lambda x, y, z : 0*x
            
            Ginv31 = lambda x, y, z : 0*x
            Ginv32 = lambda x, y, z : 0*x
            Ginv33 = lambda x, y, z : 1/self.Lz**2*np.ones(x.shape)
            
            self.Ginv = [[Ginv11, Ginv12, Ginv13], [Ginv21, Ginv22, Ginv23], [Ginv31, Ginv32, Ginv33]]
            
            
            self.g      = lambda x, y, z : self.Lx**2*self.Ly**2*self.Lz**2*np.ones(x.shape)
            self.g_sqrt = lambda x, y, z : self.Lx*self.Ly*self.Lz*np.ones(x.shape)
            
            
        if mapping[0] == 'torus':
            
            r, theta, phi = sy.symbols('r, theta, phi')
            q             = sy.Matrix([r, theta, phi])
            
            
            self.R0 = mapping[1]
            self.R1 = mapping[2]
            self.R2 = mapping[3]
            self.dR = self.R2 - self.R1
            
            Fx = (self.R0 + (r*self.dR + self.R1)*sy.cos(2*sy.pi*theta))*sy.cos(2*sy.pi*phi)
            Fy = (self.R0 + (r*self.dR + self.R1)*sy.cos(2*sy.pi*theta))*sy.sin(2*sy.pi*phi)
            Fz = (r*self.dR + self.R1)*sy.sin(2*sy.pi*theta)


            F      = sy.Matrix([Fx, Fy, Fz])

            DF     = F.jacobian(q)

            G      = sy.simplify(DF.transpose()*DF)
            Ginv   = G.inverse()

            g      = sy.simplify(G.det())
            g_sqrt = sy.sqrt(g)

            
            self.F = [sy.lambdify(q, F[0]), sy.lambdify(q, F[1]), sy.lambdify(q, F[2])]
            
            
            G11 = sy.lambdify(q, G[0, 0])
            G12 = sy.lambdify(q, G[0, 1])
            G13 = sy.lambdify(q, G[0, 2])
            
            G21 = sy.lambdify(q, G[1, 0])
            G22 = sy.lambdify(q, G[1, 1])
            G23 = sy.lambdify(q, G[1, 2])
            
            G31 = sy.lambdify(q, G[2, 0])
            G32 = sy.lambdify(q, G[2, 1])
            G33 = sy.lambdify(q, G[2, 2])
            
            self.G = [[G11, G12, G13], [G21, G22, G23], [G31, G32, G33]]
            
            
            Ginv11 = sy.lambdify(q, Ginv[0, 0])
            Ginv12 = sy.lambdify(q, Ginv[0, 1])
            Ginv13 = sy.lambdify(q, Ginv[0, 2])
            
            Ginv21 = sy.lambdify(q, Ginv[1, 0])
            Ginv22 = sy.lambdify(q, Ginv[1, 1])
            Ginv23 = sy.lambdify(q, Ginv[1, 2])
            
            Ginv31 = sy.lambdify(q, Ginv[2, 0])
            Ginv32 = sy.lambdify(q, Ginv[2, 1])
            Ginv33 = sy.lambdify(q, Ginv[2, 2])
            
            self.Ginv = [[Ginv11, Ginv12, Ginv13], [Ginv21, Ginv22, Ginv23], [Ginv31, Ginv32, Ginv33]]
            
            
            self.g      = sy.lambdify(q, g)
            self.g_sqrt = sy.lambdify(q, g_sqrt)