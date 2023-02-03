from abc import ABCMeta, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
from pyevtk.hl import gridToVTK


class MHDequilibrium(metaclass=ABCMeta):
    """
    Base class for Struphy MHD equilibria, analytical or numerical.
    The callables B, J, p etc. have to be provided through the child classes `AnalyticMHDequilibrium` or `NumericalMHDequilibrium`.
    The base class provides transformations of callables to different representations or coordinates.
    For numerical equilibria, the methods absB0, bv, unit_bv, j2, p0 and n0 are overidden by the child class.   
    """

    @property
    @abstractmethod
    def domain(self):
        """ Domain object that characterizes the mapping from the logical cube [0, 1]^3 to the physical domain.
        """
        pass

    def absB0(self, *etas, squeeze_out=True):
        """ 0-form absolute value of equilibrium magnetic field in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.absB], *etas, kind='0_form', squeeze_out=squeeze_out)
    
    def b2_1(self, *etas, squeeze_out=True):
        """ 2-form equilibrium magnetic field (eta1-component) in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form', squeeze_out=squeeze_out)[0]

    def b2_2(self, *etas, squeeze_out=True):
        """ 2-form equilibrium magnetic field (eta2-component) in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form', squeeze_out=squeeze_out)[1]

    def b2_3(self, *etas, squeeze_out=True):
        """ 2-form equilibrium magnetic field (eta3-component) in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.b_x, self.b_y, self.b_z], *etas, kind='2_form', squeeze_out=squeeze_out)[2]

    def b_cart_1(self, *etas, squeeze_out=True):
        """ Cartesian equilibrium magnetic field (x-component) in physical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.push([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0], self.domain(*etas)
    
    def b_cart_2(self, *etas, squeeze_out=True):
        """ Cartesian equilibrium magnetic field (y-component) in physical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.push([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1], self.domain(*etas)

    def b_cart_3(self, *etas, squeeze_out=True):
        """ Cartesian equilibrium magnetic field (z-component) in physical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.push([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2], self.domain(*etas)

    def b1_1(self, *etas, squeeze_out=True):
        """ 1-form equilibrium magnetic field (eta1-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0]

    def b1_2(self, *etas, squeeze_out=True):
        """ 1-form equilibrium magnetic field (eta2-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1]

    def b1_3(self, *etas, squeeze_out=True):
        """ 1-form equilibrium magnetic field (eta3-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2]
    
    def bv_1(self, *etas, squeeze_out=True):
        """ First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0]

    def bv_2(self, *etas, squeeze_out=True):
        """ Second contra-variant component (eta2) of magnetic field on logical cube [0, 1]^3.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1]

    def bv_3(self, *etas, squeeze_out=True):
        """ Third contra-variant component (eta3) of magnetic field on logical cube [0, 1]^3.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.b2_1, self.b2_2, self.b2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2]
    
    def unit_b2_1(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta1-component, 2-form) in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.unit_b_x, self.unit_b_y, self.unit_b_z], *etas, kind='2_form', squeeze_out=squeeze_out)[0]

    def unit_b2_2(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta2-component, 2-form) in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.unit_b_x, self.unit_b_y, self.unit_b_z], *etas, kind='2_form', squeeze_out=squeeze_out)[1]

    def unit_b2_3(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta3-component, 2-form) in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.unit_b_x, self.unit_b_y, self.unit_b_z], *etas, kind='2_form', squeeze_out=squeeze_out)[2]
    
    def unit_b1_1(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta1-component, 1-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_b2_1, self.unit_b2_2, self.unit_b2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0]

    def unit_b1_2(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta2-component, 1-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_b2_1, self.unit_b2_2, self.unit_b2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1]

    def unit_b1_3(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta3-component, 1-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_b2_1, self.unit_b2_2, self.unit_b2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2]
    
    def unit_bv_1(self, *etas, squeeze_out=False):
        """ Unit vector equilibrium magnetic field (eta1-component, contra-variant) on logical cube [0, 1]^3.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_b2_1, self.unit_b2_2, self.unit_b2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0]

    def unit_bv_2(self, *etas, squeeze_out=False):
        """ Unit vector equilibrium magnetic field (eta2-component, contra-variant) on logical cube [0, 1]^3.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_b2_1, self.unit_b2_2, self.unit_b2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1]

    def unit_bv_3(self, *etas, squeeze_out=False):
        """ Unit vector equilibrium magnetic field (eta3-component, contra-variant) on logical cube [0, 1]^3.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_b2_1, self.unit_b2_2, self.unit_b2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2]
    
    def j2_1(self, *etas, squeeze_out=True):
        """ First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.j_x, self.j_y, self.j_z], *etas, kind='2_form', squeeze_out=squeeze_out)[0]

    def j2_2(self, *etas, squeeze_out=True):
        """ Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.j_x, self.j_y, self.j_z], *etas, kind='2_form', squeeze_out=squeeze_out)[1]

    def j2_3(self, *etas, squeeze_out=True):
        """ Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.j_x, self.j_y, self.j_z], *etas, kind='2_form', squeeze_out=squeeze_out)[2]

    def j1_1(self, *etas, squeeze_out=True):
        """ 1-form equilibrium current (eta1-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0]

    def j1_2(self, *etas, squeeze_out=True):
        """ 1-form equilibrium current (eta2-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1]

    def j1_3(self, *etas, squeeze_out=True):
        """ 1-form equilibrium current (eta3-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_to_1', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2]

    def jv_1(self, *etas, squeeze_out=True):
        """ Vector-field equilibrium current (eta1-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0]

    def jv_2(self, *etas, squeeze_out=True):
        """ Vector-field equilibrium current (eta2-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1]

    def jv_3(self, *etas, squeeze_out=True):
        """ Vector-field equilibrium current (eta3-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_to_v', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2]

    def j_cart_1(self, *etas, squeeze_out=True):
        """ Cartesian equilibrium current (x-component) in physical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.push([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[0], self.domain(*etas)
    
    def j_cart_2(self, *etas, squeeze_out=True):
        """ Cartesian equilibrium current (y-component) in physical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.push([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[1], self.domain(*etas)

    def j_cart_3(self, *etas, squeeze_out=True):
        """ Cartesian equilibrium current (z-component) in physical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.push([self.j2_1, self.j2_2, self.j2_3], *etas, kind='2_form', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)[2], self.domain(*etas)

    def p0(self, *etas, squeeze_out=True):
        """ 0-form equilibrium pressure in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.p], *etas, kind='0_form', squeeze_out=squeeze_out)

    def p3(self, *etas, squeeze_out=True):
        """ 3-form equilibrium pressure in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.p0], *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def n0(self, *etas, squeeze_out=True):
        """ 0-form equilibrium number density in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.n], *etas, kind='0_form', squeeze_out=squeeze_out)

    def n3(self, *etas, squeeze_out=True):
        """ 3-form equilibrium number density in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.n0], *etas, kind='0_to_3', a_kwargs={'squeeze_out' : False}, squeeze_out=squeeze_out)

    def show(self, n1=16, n2=32, n3=21, n_planes=5):
        '''Generate vtk files of equilibirum and do some 2d plots with matplotlib.
        
        Parameters
        ----------
        n1, n2, n3 : int
            Evaluation points of mapping in each direcion.
            
        n_planes : int
            Number of planes to show perpendicular to eta3.'''

        import struphy as _

        e1 = np.linspace(0.0001, 1, n1)
        e2 = np.linspace(0, 1, n2)
        e3 = np.linspace(0, 1, n3)

        jump = (n3 - 1)/(n_planes - 1)

        x, y, z = self.domain(e1, e2, e3)
        det_df  = self.domain.jacobian_det(e1, e2, e3)
        p = self.p0(e1, e2, e3)
        absB = self.absB0(e1, e2, e3)

        _path = _.__path__[0] + '/fields_background/mhd_equil/gvec/output/'
        gridToVTK(_path + 'vtk/gvec_equil', x, y, z, pointData = {'det_df': det_df, 'pressure': p, 'absB': absB})

        # show params
        print('Equilibrium parameters:')
        for key, val in self.params.items():
            print(key, ': ', val)

        print('\nMapping parameters:')
        for key, val in self.domain.params_map.items():
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


class AnalyticalMHDequilibrium(MHDequilibrium):
    """
    Base class for analytical MHD equilibria. B, J, n and p have to be specified in Cartesian coordinates.  
    The domain must be set using the setter method.     
    """

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        assert hasattr(self, '_domain'), 'Domain for analytical MHD equilibrium not set. Please do obj.domain = ...'
        return self._domain

    @domain.setter
    def domain(self, domain):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        self._domain = domain

    @abstractmethod
    def b_x(self, x, y, z):
        """ Equilibrium magnetic field (x-component) in physical space.
        """
        pass

    @abstractmethod
    def b_y(self, x, y, z):
        """ Equilibrium magnetic field (y-component) in physical space.
        """
        pass

    @abstractmethod
    def b_z(self, x, y, z):
        """ Equilibrium magnetic field (z-component) in physical space.
        """
        pass

    @abstractmethod
    def j_x(self, x, y, z):
        """ Equilibrium current (x-component, curl of equilibrium magnetic field) in physical space.
        """
        pass

    @abstractmethod
    def j_y(self, x, y, z):
        """ Equilibrium current (y-component, curl of equilibrium magnetic field) in physical space.
        """
        pass

    @abstractmethod
    def j_z(self, x, y, z):
        """ Equilibrium current (z-component, curl of equilibrium magnetic field) in physical space.
        """
        pass

    @abstractmethod
    def p(self, x, y, z):
        """ Equilibrium pressure in physical space.
        """
        pass

    @abstractmethod
    def n(self, x, y, z):
        """ Equilibrium number density in physical space.
        """
        pass

    def absB(self, x, y, z):
        """ Equilibrium magnetic field (absolute value).
        """
        bx = self.b_x(x, y, z)
        by = self.b_y(x, y, z)
        bz = self.b_z(x, y, z)

        return np.sqrt(bx**2 + by**2 + bz**2)

    def unit_b_x(self, x, y, z):
        """ Unit vector equilibrium magnetic field (x-component) in physical space.
        """
        return self.b_x(x, y, z) / self.absB(x, y, z)

    def unit_b_y(self, x, y, z):
        """ Unit vector equilibrium magnetic field (y-component) in physical space.
        """
        return self.b_y(x, y, z) / self.absB(x, y, z)

    def unit_b_z(self, x, y, z):
        """ Unit vector equilibrium magnetic field (z-component) in physical space.
        """
        return self.b_z(x, y, z) / self.absB(x, y, z)


class NumericalMHDequilibrium(MHDequilibrium):
    """
    Base class for numerical MHD equilibria. 
    B, J, p and n must be specified on the logical cube [0, 1]^3. 
    B in contra-variant coordinates (i.e. as a vector-field), J as a 2-form, p and n as a 0-form.       
    """

    @abstractmethod
    def b2_1(self, *etas, squeeze_out=True):
        """First 2-form component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def b2_2(self, *etas, squeeze_out=True):
        """Second 2-form component (eta2) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def b2_3(self, *etas, squeeze_out=True):
        """Third 2-form component (eta3) of magnetic field on logical cube [0, 1]^3.
        """
        pass
    
    @abstractmethod
    def j2_1(self, *etas, squeeze_out=True):
        """First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def j2_2(self, *etas, squeeze_out=True):
        """Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def j2_3(self, *etas, squeeze_out=True):
        """Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def p0(self, *etas, squeeze_out=True):
        """0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def n0(self, *etas, squeeze_out=True):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        pass

    def absB0(self, *etas, squeeze_out=True):
        """ 0-form absolute value of equilibrium magnetic field on logical cube [0, 1]^3.
        """

        tmp1 = self.b2_1(*etas, squeeze_out=False)
        tmp2 = self.b2_2(*etas, squeeze_out=False)
        tmp3 = self.b2_3(*etas, squeeze_out=False)

        b = self.domain.push(
            [tmp1, tmp2, tmp3], *etas, kind='2_form', squeeze_out=squeeze_out)
        bx = b[0]
        by = b[1]
        bz = b[2]

        return np.sqrt(bx**2 + by**2 + bz**2)
    
    def unit_b2_1(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta1-component, 2-form) on logical cube [0, 1]^3.
        """
        return self.b2_1(*etas, squeeze_out=squeeze_out) / self.absB0(*etas, squeeze_out=squeeze_out)

    def unit_b2_2(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta2-component, 2-form) on logical cube [0, 1]^3.
        """
        return self.b2_2(*etas, squeeze_out=squeeze_out) / self.absB0(*etas, squeeze_out=squeeze_out)

    def unit_b2_3(self, *etas, squeeze_out=True):
        """ Unit vector equilibrium magnetic field (eta3-component, 2-form) on logical cube [0, 1]^3.
        """
        return self.b2_3(*etas, squeeze_out=squeeze_out) / self.absB0(*etas, squeeze_out=squeeze_out)
