from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from pyevtk.hl import gridToVTK


class MHDequilibrium(metaclass=ABCMeta):
    """
    Base class for Struphy MHD equilibria.
    The callables B, J, p and n have to be provided through the child classes `CartesianMHDequilibrium` or `LogicalMHDequilibrium`.
    The base class provides transformations of callables to different representations or coordinates.
    For logical equilibria, the methods b2, j2, p0 and n0 are overidden by the child class.   
    """    

    def absB0(self, *etas, squeeze_out=True):
        """ 0-form absolute value of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        return np.sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    
    def b1(self, *etas, squeeze_out=True):
        """ 1-form components of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.b2(*etas, squeeze_out=False), *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def b2(self, *etas, squeeze_out=True):
        """ 2-form components of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.b_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='2_form', squeeze_out=squeeze_out)

    def bv(self, *etas, squeeze_out=True):
        """ Contra-variant components of equilibrium magnetic field on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.b2(*etas, squeeze_out=False), *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def b_cart(self, *etas, squeeze_out=True):
        """ Cartesian components of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        b_out = self.domain.push(self.b2(*etas, squeeze_out=False), *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)
        return b_out, self.domain(*etas, squeeze_out=squeeze_out)

    def unit_b1(self, *etas, squeeze_out=True):
        """ Unit vector components of equilibrium magnetic field (1-form) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='1_form', squeeze_out=squeeze_out)

    def unit_b2(self, *etas, squeeze_out=True):
        """ Unit vector components of equilibrium magnetic field (2-form) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='2_form', squeeze_out=squeeze_out)
    
    def unit_bv(self, *etas, squeeze_out=False):
        """ Unit vector components of  equilibrium magnetic field (contra-variant) on logical cube [0, 1]^3.
        """
        return self.domain.pull(self.unit_b_cart(*etas, squeeze_out=False)[0], *etas, kind='vector', squeeze_out=squeeze_out)
    
    def unit_b_cart(self, *etas, squeeze_out=True):
        """ Unit vector Cartesian components of equilibrium magnetic field evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        b, xyz = self.b_cart(*etas, squeeze_out=squeeze_out)
        absB = self.absB0(*etas, squeeze_out=squeeze_out)
        out = np.array([b[0]/absB, b[1]/absB, b[2]/absB], dtype=float)
        return out, xyz

    def j1(self, *etas, squeeze_out=True):
        """ 1-form components of equilibrium current on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.j2(*etas, squeeze_out=False), *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def j2(self, *etas, squeeze_out=True):
        """ 2-form components of equilibrium current on logical cube [0, 1]^3.
        """
        xyz = self.domain(*etas, squeeze_out=False)
        return self.domain.pull(self.j_xyz(xyz[0], xyz[1], xyz[2]), *etas, kind='2_form', squeeze_out=squeeze_out)

    def jv(self, *etas, squeeze_out=True):
        """ Contra-variant components of equilibrium current on logical cube [0, 1]^3.
        """
        return self.domain.transform(self.j2(*etas, squeeze_out=False), *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def j_cart(self, *etas, squeeze_out=True):
        """ Cartesian components of equilibrium current evaluated on logical cube [0, 1]^3. Returns also (x,y,z).
        """
        j_out = self.domain.push(self.j2(*etas, squeeze_out=False), *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)
        return j_out, self.domain(*etas)

    def p0(self, *etas, squeeze_out=True):
        """ 0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        return self.domain.pull([self.p_xyz], *etas, kind='0_form', squeeze_out=squeeze_out)

    def p3(self, *etas, squeeze_out=True):
        """ 3-form equilibrium pressure on logical cube [0, 1]^3.
        """
        return self.domain.transform([self.p0], *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def n0(self, *etas, squeeze_out=True):
        """ 0-form equilibrium number density on logical cube [0, 1]^3.
        """
        return self.domain.pull([self.n_xyz], *etas, kind='0_form', squeeze_out=squeeze_out)

    def n3(self, *etas, squeeze_out=True):
        """ 3-form equilibrium number density on logical cube [0, 1]^3.
        """
        return self.domain.transform([self.n0], *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    ###################
    # Single components
    ###################

    def b1_1(self, *etas, squeeze_out=True):
        return self.b1(*etas, squeeze_out=squeeze_out)[0]

    def b1_2(self, *etas, squeeze_out=True):
        return self.b1(*etas, squeeze_out=squeeze_out)[1]

    def b1_3(self, *etas, squeeze_out=True):
        return self.b1(*etas, squeeze_out=squeeze_out)[2]

    def b2_1(self, *etas, squeeze_out=True):
        return self.b2(*etas, squeeze_out=squeeze_out)[0]

    def b2_2(self, *etas, squeeze_out=True):
        return self.b2(*etas, squeeze_out=squeeze_out)[1]

    def b2_3(self, *etas, squeeze_out=True):
        return self.b2(*etas, squeeze_out=squeeze_out)[2]

    def unit_b1_1(self, *etas, squeeze_out=True):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[0]

    def unit_b1_2(self, *etas, squeeze_out=True):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[1]

    def unit_b1_3(self, *etas, squeeze_out=True):
        return self.unit_b1(*etas, squeeze_out=squeeze_out)[2]

    def unit_b2_1(self, *etas, squeeze_out=True):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[0]

    def unit_b2_2(self, *etas, squeeze_out=True):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[1]

    def unit_b2_3(self, *etas, squeeze_out=True):
        return self.unit_b2(*etas, squeeze_out=squeeze_out)[2]

    def b_cart_1(self, *etas, squeeze_out=True):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][0]

    def b_cart_2(self, *etas, squeeze_out=True):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][1]

    def b_cart_3(self, *etas, squeeze_out=True):
        return self.b_cart(*etas, squeeze_out=squeeze_out)[0][2]

    def j1_1(self, *etas, squeeze_out=True):
        return self.j1(*etas, squeeze_out=squeeze_out)[0]

    def j1_2(self, *etas, squeeze_out=True):
        return self.j1(*etas, squeeze_out=squeeze_out)[1]
    
    def j1_3(self, *etas, squeeze_out=True):
        return self.j1(*etas, squeeze_out=squeeze_out)[2]

    def j2_1(self, *etas, squeeze_out=True):
        return self.j2(*etas, squeeze_out=squeeze_out)[0]

    def j2_2(self, *etas, squeeze_out=True):
        return self.j2(*etas, squeeze_out=squeeze_out)[1]
    
    def j2_3(self, *etas, squeeze_out=True):
        return self.j2(*etas, squeeze_out=squeeze_out)[2]

    ##########
    # Plotting
    ##########

    def show(self, n1=16, n2=32, n3=21, n_planes=5):
        '''Generate vtk files of equilibirum and do some 2d plots with matplotlib.
        
        Parameters
        ----------
        n1, n2, n3 : int
            Evaluation points of mapping in each direcion.
            
        n_planes : int
            Number of planes to show perpendicular to eta3.'''

        import struphy 

        e1 = np.linspace(0.0001, 1, n1)
        e2 = np.linspace(0, 1, n2)
        e3 = np.linspace(0, 1, n3)

        if n_planes > 1:
            jump = (n3 - 1)/(n_planes - 1)
        else:
            jump = 0

        x, y, z = self.domain(e1, e2, e3)
        det_df  = self.domain.jacobian_det(e1, e2, e3)
        p = self.p0(e1, e2, e3)
        absB = self.absB0(e1, e2, e3)
        j_cart, xyz = self.j_cart(e1, e2, e3)
        absJ = np.sqrt(j_cart[0]**2 + j_cart[1]**2 + j_cart[2]**2)

        _path = struphy.__path__[0] + '/fields_background/mhd_equil/gvec/output/'
        gridToVTK(_path + 'vtk/gvec_equil', x, y, z, pointData = {'det_df': det_df, 'pressure': p, 'absB': absB})

        # show params
        print('Equilibrium parameters:')
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

            rp = np.sqrt(xp**2 + yp**2)
            
            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            for i in range(rp.shape[0]):
                for j in range(rp.shape[1] - 1):
                    if i < rp.shape[0] - 1:
                        ax.plot([rp[i, j], rp[i + 1, j]], [zp[i, j], zp[i + 1, j]], 'b', linewidth=.6)
                    if j < rp.shape[1] - 1:
                        ax.plot([rp[i, j], rp[i, j + 1]], [zp[i, j], zp[i, j + 1]], 'b', linewidth=.6)
            ax.set_xlabel('r')
            ax.set_ylabel('z')
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

            for i in range(xp.shape[0]):
                for j in range(xp.shape[1] - 1):
                    if i < xp.shape[0] - 1:
                        ax.plot([xp[i, j], xp[i + 1, j]], [yp[i, j], yp[i + 1, j]], 'b', linewidth=.6)
                    if j < xp.shape[1] - 1:
                        if i == 0:
                            ax.plot([xp[i, j], xp[i, j + 1]], [yp[i, j], yp[i, j + 1]], 'r', linewidth=1)
                        else:
                            ax.plot([xp[i, j], xp[i, j + 1]], [yp[i, j], yp[i, j + 1]], 'b', linewidth=.6)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('equal')
            ax.set_title('Device top view')

        # Jacobian determinant
        fig = plt.figure(figsize=(13, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            rp = np.sqrt(xp**2 + yp**2)

            detp = det_df[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(rp, zp, detp, 30)
            ax.set_xlabel('r')
            ax.set_ylabel('z')
            ax.axis('equal')
            ax.set_title('Jacobian determinant at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')

        # pressure
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            rp = np.sqrt(xp**2 + yp**2)

            pp = p[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(rp, zp, pp, 30)
            ax.set_xlabel('r')
            ax.set_ylabel('z')
            ax.axis('equal')
            ax.set_title('Pressure at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')

        # magnetic field strength
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            rp = np.sqrt(xp**2 + yp**2)

            ab = absB[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(rp, zp, ab, 30)
            ax.set_xlabel('r')
            ax.set_ylabel('z')
            ax.axis('equal')
            ax.set_title('Magnetic field strength at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')

        # current density
        fig = plt.figure(figsize=(15, np.ceil(n_planes/2) * 6.5))
        for n in range(n_planes):

            xp = x[:, :, int(n*jump)].squeeze()
            yp = y[:, :, int(n*jump)].squeeze()
            zp = z[:, :, int(n*jump)].squeeze()

            rp = np.sqrt(xp**2 + yp**2)

            ab = absJ[:, :, int(n*jump)].squeeze()

            ax = fig.add_subplot(int(np.ceil(n_planes/2)), 2, n + 1)
            map = ax.contourf(rp, zp, ab, 30)
            ax.set_xlabel('r')
            ax.set_ylabel('z')
            ax.axis('equal')
            ax.set_title('Current density (abs) at $\eta_3$={0:4.3f}'.format(e3[int(n*jump)]))
            fig.colorbar(map, ax=ax, location='right')


class CartesianMHDequilibrium(MHDequilibrium):
    """
    Base class for MHD equilibria where B, J, n and p are specified in Cartesian coordinates.  
    The domain must be set using the setter method.    
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

    @abstractmethod
    def b_xyz(self, x, y, z):
        """ Cartesian equilibrium magnetic field in physical space. Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def j_xyz(self, x, y, z):
        """ Cartesian equilibrium current (curl of equilibrium magnetic field) in physical space. Must return the components as a tuple.
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


class LogicalMHDequilibrium(MHDequilibrium):
    """
    Base class for MHD equilibria where B, J, p and n are specified on the logical cube [0, 1]^3. 
    B and J as 2-forms, p and n as a 0-forms.      
    """

    @property
    @abstractmethod
    def domain(self):
        """ Domain object that characterizes the mapping from the logical cube [0, 1]^3 to the physical domain.
        """
        pass
    
    @abstractmethod
    def b2(self, *etas, squeeze_out=True):
        """2-form magnetic field on logical cube [0, 1]^3. Must return the components as a tuple.
        """
        pass
    
    @abstractmethod
    def j2(self, *etas, squeeze_out=True):
        """2-form current density (=curl B) on logical cube [0, 1]^3. Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def p0(self, *etas, squeeze_out=True):
        """0-form equilibrium pressure on logical cube [0, 1]^3. Must return the components as a tuple.
        """
        pass

    @abstractmethod
    def n0(self, *etas, squeeze_out=True):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        pass


class AxisymmMHDequilibrium(CartesianMHDequilibrium):
    """
    Base class for ideal axisymmetric MHD equilibria based on a poloidal flux function psi = psi(R, Z) and a toroidal field function g_tor = g_tor(R, Z) in a cylindrical coordinate system (R, phi, Z).
    
    The magnetic field and current density are then given by
        
        * B = grad(psi) x grad(phi) + g_tor * grad(phi),
        * j = curl(B).
        
    The pressure and density profiles need to be implemented by child classes.
    """
    
    @abstractmethod
    def psi(self, R, Z, dR=0, dZ=0):
        """ Poloidal flux function per radian. First AND second derivatives dR=0,1,2 and dZ=0,1,2 must be implemented.
        """
        pass
    
    @abstractmethod
    def g_tor(self, R, Z, dR=0, dZ=0):
        """ Toroidal field function. First derivatives dR=0,1 and dZ=0,1 must be implemented.
        """
        pass
    
    @property
    @abstractmethod
    def psi_range(self):
        """ Psi on-axis and at plasma boundary returned as list [psi_axis, psi_boundary].
        """
        pass
    
    @property
    @abstractmethod
    def psi_axis_RZ(self):
        """ Location of magnetic axis in R-Z-coordinates returned as list [psi_axis_R, psi_axis_Z].
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
    
    def b_xyz(self, x, y, z):
        """ Cartesian B-field components calculated as BR = -(dpsi/dZ)/R, BPhi = g_tor/R, BZ = (dpsi/dR)/R.
        """
        
        from struphy.geometry.base import Domain
        
        # check for point-wise evaluation and broadcast input to 3d numpy arrays.
        is_float = all(isinstance(v, (int, float)) for v in [x, y, z])
        
        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)
        
        R, Phi, Z = self.inverse_map(x, y, z)
        
        # at phi = 0°
        BR = -self.psi(R, Z, dZ=1)/R
        BP = self.g_tor(R, Z)/R
        BZ =  self.psi(R, Z, dR=1)/R
        
        # push-forward to Cartesian components
        Bx = BR*np.cos(Phi) - BP*np.sin(Phi)
        By = BR*np.sin(Phi) + BP*np.cos(Phi)
        Bz = 1*BZ
        
        # remove all "dimensions" for point-wise evaluation
        if is_float:
            assert Bx.ndim == 3
            assert By.ndim == 3
            assert Bz.ndim == 3
            Bx = Bx.item()
            By = By.item()
            Bz = Bz.item()

        return Bx, By, Bz

    def j_xyz(self, x, y, z):
        """ Cartesian current density components calculated as curl(B).
        """
        
        from struphy.geometry.base import Domain
        
        # check for point-wise evaluation and broadcast input to 3d numpy arrays.
        is_float = all(isinstance(v, (int, float)) for v in [x, y, z])
        
        x, y, z, is_sparse_meshgrid = Domain.prepare_eval_pts(x, y, z)
        
        R, Phi, Z = self.inverse_map(x, y, z)
        
        # at phi = 0° (j = curl(B))
        jR = -self.g_tor(R, Z, dZ=1)/R
        jP = -self.psi(R, Z, dZ=2)/R + self.psi(R, Z, dR=1)/R**2 - self.psi(R, Z, dR=2)/R
        jZ =  self.g_tor(R, Z, dR=1)/R
        
        # push-forward to Cartesian components
        jx = jR*np.cos(Phi) - jP*np.sin(Phi)
        jy = jR*np.sin(Phi) + jP*np.cos(Phi)
        jz = 1*jZ

        # remove all "dimensions" for point-wise evaluation
        if is_float:
            assert jx.ndim == 3
            assert jy.ndim == 3
            assert jz.ndim == 3
            jx = jx.item()
            jy = jy.item()
            jz = jz.item()

        return jx, jy, jz
    
    @staticmethod
    def inverse_map(x, y, z):
        """ Inverse cylindrical mapping.
        """
        
        R = np.sqrt(x**2 + y**2)
        P = np.arctan2(y, x)
        Z = 1*z
        
        return R, P, Z