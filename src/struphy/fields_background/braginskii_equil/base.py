'Base class for equilibria of the ion Braginskii equations with adiabatic electrons.'


from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from pyevtk.hl import gridToVTK


class BraginskiiEquilibrium(metaclass=ABCMeta):
    r"""
    Base class for analytical Braginskii ion-fluid equilibria in a given magnetic background.
    
    The equilibria are solutions of:
    
    .. math::
    
        \begin{align}
        \mathbf u \cdot \nabla n + n \nabla \cdot \mathbf u &= 0\,,
        \\[2mm]
        mn \mathbf u \cdot \nabla \mathbf u + \nabla p \color{red}+ \nabla \cdot \mathbb P_\wedge &= q n \mathbf u \times \mathbf B_0\,,
        \\[2mm]
        \mathbf u \cdot \nabla p + \gamma p \nabla \cdot \mathbf u &= \color{red} - (\gamma - 1) \nabla \cdot \left(  \mathbf q_\wedge + \mathbb P_\wedge \cdot \mathbf u \right)\,,
        \end{align}
        
    where the red terms denote the gyro-viscous stress tensor :math:`\mathbb P_\wedge` and heat flux :math:`\mathbf q_\wedge`. 

    The base class provides transformations of callables to different representations or coordinates.   
    """    

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        assert hasattr(self, '_domain'), 'Domain for Cartesian MHD equilibrium not set. Only b_xyz, j_xyz, p_xyz and n_xyz available at this stage. Please do obj.domain = ... to have access to all transformations (1-form, 2-form, etc.)'
        return self._domain

    @domain.setter
    def domain(self, domain):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        self._domain = domain

    # abstract methods
    @abstractmethod
    def b_xyz(self, x, y, z):
        """ Cartesian equilibrium magnetic field in physical space. Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def u_xyz(self, x, y, z):
        """ Cartesian equilibrium ion velocity in physical space. Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def p_xyz(self, x, y, z):
        """ Equilibrium pressure in physical space.
        """
        pass

    @abstractmethod
    def n_xyz(self, x, y, z):
        """ Equilibrium number density in physical space.
        """
        pass

    @abstractmethod
    def gradB_xyz(self, x, y, z):
        """ Cartesian gradient of equilibrium magnetic field in physical space. Must return the components as a tuple.
        """
        pass

    # class methods
    def absB0(self, *etas, squeeze_out=False):
        """ 0-form absolute value of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        return np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    
    def b1(self, *etas, squeeze_out=False):
        """ 1-form components of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.b_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='1', squeeze_out=squeeze_out)

    def b2(self, *etas, squeeze_out=False):
        """ 2-form components of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.b_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='2', squeeze_out=squeeze_out)

    def bv(self, *etas, squeeze_out=False):
        """ Contra-variant components of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.b_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='v', squeeze_out=squeeze_out)

    def b_cart(self, *etas, squeeze_out=False):
        """ Cartesian components of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        xyz = self.domain(*etas, squeeze_out=False)
        b_out = self.b_xyz(xyz[0], xyz[1], xyz[2])
        return b_out, self.domain(*etas, squeeze_out=squeeze_out)

    def unit_b1(self, *etas, squeeze_out=False):
        """ Unit vector components of equilibrium magnetic field (1-form) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='1', squeeze_out=squeeze_out)

    def unit_b2(self, *etas, squeeze_out=False):
        """ Unit vector components of equilibrium magnetic field (2-form) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='2', squeeze_out=squeeze_out)
    
    def unit_bv(self, *etas, squeeze_out=False):
        """ Unit vector components of  equilibrium magnetic field (contra-variant) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='v', squeeze_out=squeeze_out)
    
    def unit_b_cart(self, *etas, squeeze_out=False):
        """ Unit vector Cartesian components of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        absB = self.absB0(*etas, squeeze_out=squeeze_out)
        out = np.array([b[0]/absB, b[1]/absB, b[2]/absB], dtype=float)
        return out, xyz

    def u1(self, *etas, squeeze_out=False):
        """ 1-form components of equilibrium ion velocity on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.u_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='1', squeeze_out=squeeze_out)

    def u2(self, *etas, squeeze_out=False):
        """ 2-form components of equilibrium ion velocity on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.u_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='2', squeeze_out=squeeze_out)

    def uv(self, *etas, squeeze_out=False):
        """ Contra-variant components of equilibrium ion velocity on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.u_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='v', squeeze_out=squeeze_out)

    def u_cart(self, *etas, squeeze_out=False):
        """ Cartesian components of equilibrium ion velocity evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        xyz = self.domain(*etas, squeeze_out=False)
        u_out = self.u_xyz(xyz[0], xyz[1], xyz[2])
        return u_out, self.domain(*etas, squeeze_out=squeeze_out)

    def gradB1(self, *etas, squeeze_out=False):
        """ 1-form components of gradient of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.gradB_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='1', squeeze_out=squeeze_out)

    def gradB2(self, *etas, squeeze_out=False):
        """ 2-form components of gradient of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        return self.domain.transform(self.gradB1(*etas, squeeze_out=False), *etas, kind='1_to_2', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def gradBv(self, *etas, squeeze_out=False):
        """ Contra-variant components of gradient of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        return self.domain.transform(self.gradB1(*etas, squeeze_out=False), *etas, kind='1_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def gradB_cart(self, *etas, squeeze_out=False):
        """ Cartesian components of gradient of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        gradB_out = self.domain.push(self.gradB1(*etas, squeeze_out=False), *etas, kind='1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)
        return gradB_out, self.domain(*etas)

    def p0(self, *etas, squeeze_out=False):
        """ 0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.p_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='0', squeeze_out=squeeze_out)

    def p3(self, *etas, squeeze_out=False):
        """ 3-form equilibrium pressure on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.p0(*etas, squeeze_out=False), *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def n0(self, *etas, squeeze_out=False):
        """ 0-form equilibrium number density on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.n_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='0', squeeze_out=squeeze_out)

    def n3(self, *etas, squeeze_out=False):
        """ 3-form equilibrium number density on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.n0(*etas, squeeze_out=False), *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def s0(self, *etas, squeeze_out=False):
        """ 0-form equilibrium entropy density on logical cube [0, 1]^3.
            Hard coded assumption : gamma = 5/3 (monoatomic perfect gaz)
        """
        xyz = self.domain(*etas, squeeze_out=False)
        p = self.p_xyz(xyz[0], xyz[1], xyz[2])
        n = self.n_xyz(xyz[0], xyz[1], xyz[2])
        s = n * np.log(p/(2/3*np.power(n, 5/3)))
        return self.domain.pull(s, *etas, kind='0', squeeze_out=squeeze_out)
    
    def s3(self, *etas, squeeze_out=False):
        """ 3-form equilibrium entropy density on logical cube [0, 1]^3.
            Hard coded assumption : gamma = 5/3 (monoatomic perfect gaz)
        """
        return self.domain.transform(self.s0(*etas, squeeze_out=False), *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    ###################
    # Single components
    ###################

    def b1_1(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[0]

    def b1_2(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[1]

    def b1_3(self, *etas, squeeze_out=False):
        return self.b1(*etas, squeeze_out=squeeze_out)[2]

    def b2_1(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[0]

    def b2_2(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[1]

    def b2_3(self, *etas, squeeze_out=False):
        return self.b2(*etas, squeeze_out=squeeze_out)[2]
    
    def bv_1(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[0]

    def bv_2(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[1]

    def bv_3(self, *etas, squeeze_out=False):
        return self.bv(*etas, squeeze_out=squeeze_out)[2]

    def unit_b1_1(self, *etas, squeeze_out=False):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[0]

    def unit_b1_2(self, *etas, squeeze_out=False):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[1]

    def unit_b1_3(self, *etas, squeeze_out=False):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[2]

    def unit_b2_1(self, *etas, squeeze_out=False):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[0]

    def unit_b2_2(self, *etas, squeeze_out=False):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[1]

    def unit_b2_3(self, *etas, squeeze_out=False):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[2]

    def b_cart_1(self, *etas, squeeze_out=False):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][0]

    def b_cart_2(self, *etas, squeeze_out=False):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][1]

    def b_cart_3(self, *etas, squeeze_out=False):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][2]

    def u1_1(self, *etas, squeeze_out=False):
        return self.u1(*etas, squeeze_out=squeeze_out)[0]

    def u1_2(self, *etas, squeeze_out=False):
        return self.u1(*etas, squeeze_out=squeeze_out)[1]
    
    def u1_3(self, *etas, squeeze_out=False):
        return self.u1(*etas, squeeze_out=squeeze_out)[2]

    def u2_1(self, *etas, squeeze_out=False):
        return self.u2(*etas, squeeze_out=squeeze_out)[0]

    def u2_2(self, *etas, squeeze_out=False):
        return self.u2(*etas, squeeze_out=squeeze_out)[1]
    
    def u2_3(self, *etas, squeeze_out=False):
        return self.u2(*etas, squeeze_out=squeeze_out)[2]

    def gradB1_1(self, *etas, squeeze_out=False):
        return self.gradB1(*etas, squeeze_out=squeeze_out)[0]

    def gradB1_2(self, *etas, squeeze_out=False):
        return self.gradB1(*etas, squeeze_out=squeeze_out)[1]

    def gradB1_3(self, *etas, squeeze_out=False):
        return self.gradB1(*etas, squeeze_out=squeeze_out)[2]
    
    ##########
    # Plotting
    ##########

    def show(self, n1=16, n2=33, n3=21, n_planes=5):
        '''Generate vtk files of equilibirum and do some 2d plots with matplotlib.
        
        Parameters
        ----------
        n1, n2, n3 : int
            Evaluation points of mapping in each direcion.
            
        n_planes : int
            Number of planes to show perpendicular to eta3.'''

        import struphy 

        torus_mappings = ('Tokamak', 'GVECunit', 'DESCunit', 'IGAPolarTorus', 'HollowTorus')

        e1 = np.linspace(0.0001, 1, n1)
        e2 = np.linspace(0, 1, n2)
        e3 = np.linspace(0, 1, n3)

        if self.domain.__class__.__name__ in ('GVECunit', 'DESCunit'):
            if n_planes > 1:
                jump = (n3 - 1)/(n_planes - 1)
            else:
                jump = 0
        else:
            n_planes = 1
            jump = 0

        x, y, z = self.domain(e1, e2, e3)
        print('Evaluation of mapping done.')
        det_df  = self.domain.jacobian_det(e1, e2, e3)
        p = self.p0(e1, e2, e3)
        print('Computation of pressure done.')
        absB = self.absB0(e1, e2, e3)
        print('Computation of abs(B) done.')
        u_cart, xyz = self.u_cart(e1, e2, e3)
        print('Computation of ion velocity done.')
        absu = np.sqrt(u_cart[0]**2 + u_cart[1]**2 + u_cart[2]**2)

        _path = struphy.__path__[0] + '/fields_background/mhd_equil/gvec/output/'
        gridToVTK(_path + 'vtk/gvec_equil', x, y, z, pointData = {'det_df': det_df, 'pressure': p, 'absB': absB})
        print('Generation of vtk files done.')

        # show params
        print('\nEquilibrium parameters:')
        for key, val in self.params.items():
            print(key, ': ', val)

        print('\nMapping parameters:')
        for key, val in self.domain.params_map.items():
            if key not in {'cx', 'cy', 'cz'}:
                print(key, ': ', val)

        # poloidal plane grid
        fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):
            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'
            
            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            for i in range(pc1.shape[0]):
                for j in range(pc1.shape[1] - 1):
                    if i < pc1.shape[0] - 1:
                        ax.plot([pc1[i, j], pc1[i + 1, j]], [pc2[i, j], pc2[i + 1, j]], 'b', linewidth=.6)
                    if j < pc1.shape[1] - 1:
                        ax.plot([pc1[i, j], pc1[i, j + 1]], [pc2[i, j], pc2[i, j + 1]], 'b', linewidth=.6)
                    
            ax.scatter(pc1[0, 0], pc2[0, 0], 20, 'red', zorder=10)    
            #ax.scatter(pc1[0, 32], pc2[0, 32], 20, 'red', zorder=10)
                        
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title('Poloidal plane at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))

        # top view
        e1 = np.linspace(0, 1, n1) # radial coordinate in [0, 1]
        e2 = np.linspace(0, 1, 3) # poloidal angle in [0, 1]
        e3 = np.linspace(0, 1, n3) # toroidal angle in [0, 1]

        xt, yt, zt = self.domain(e1, e2, e3)

        fig = plt.figure(figsize=(13, 2 * 6.5))
        ax = fig.add_subplot()
        for m in range(2):

            xp = xt[:, m, :].squeeze()
            yp = yt[:, m, :].squeeze()
            zp = zt[:, m, :].squeeze()
            
            if self.domain.__class__.__name__ in torus_mappings:
                tc1 = xp
                tc2 = yp
                l1 = 'x'
                l2 = 'y'
            else:
                tc1 = xp
                tc2 = zp
                l1 = 'x'
                l2 = 'z'

            for i in range(tc1.shape[0]):
                for j in range(tc1.shape[1] - 1):
                    if i < tc1.shape[0] - 1:
                        ax.plot([tc1[i, j], tc1[i + 1, j]], [tc2[i, j], tc2[i + 1, j]], 'b', linewidth=.6)
                    if j < tc1.shape[1] - 1:
                        if i == 0:
                            ax.plot([tc1[i, j], tc1[i, j + 1]], [tc2[i, j], tc2[i, j + 1]], 'r', linewidth=1)
                        else:
                            ax.plot([tc1[i, j], tc1[i, j + 1]], [tc2[i, j], tc2[i, j + 1]], 'b', linewidth=.6)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title('Device top view')

        # Jacobian determinant
        fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            detp = det_df[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, detp, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title('Jacobian determinant at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')

        # pressure
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            pp = p[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, pp, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title('Pressure at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')

        # magnetic field strength
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            ab = absB[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, ab, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title('Magnetic field strength at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')

        # ion velocity
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            if self.domain.__class__.__name__ in torus_mappings:
                pc1 = np.sqrt(xp**2 + yp**2)
                pc2 = zp
                l1 = 'R'
                l2 = 'Z'
            else:
                pc1 = xp
                pc2 = yp
                l1 = 'x'
                l2 = 'y'

            ab = absu[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(pc1, pc2, ab, 30)
            ax.set_xlabel(l1)
            ax.set_ylabel(l2)
            ax.axis('equal')
            ax.set_title('Ion velocity (abs) at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')

