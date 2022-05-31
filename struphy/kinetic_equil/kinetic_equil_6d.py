from abc import ABCMeta, abstractmethod

from struphy.geometry import domain_3d
from struphy.kinetic_equil.analytical import moments, gaussian


class EquilibriumKinetic6D(metaclass=ABCMeta):
    """
    The base class for kinetic equilibria for 6d Vlasov-type models

    Parameters:
    -----------
        params : dict
            contains the relevant parameters of the geometry

        DOMAIN: Domain obj (optional)
            From struphy.geometry.domain_3d.Domain.        
    """

    def __init__(self, params, DOMAIN=None):

        # set parameters
        self._params = params

        # set domain object
        self.DOMAIN = DOMAIN

    @property
    def params(self):
        """Dictionary of parameters defining the kinetic equilibrium."""
        return self._params

    @property
    def DOMAIN(self):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        return self._DOMAIN

    @DOMAIN.setter
    def DOMAIN(self, domain):
        """Domain object that characterizes the mapping from the logical to the physical domain."""
        self._DOMAIN = domain

    @abstractmethod
    def fh_eq_phys(self, x, y, z, vx, vy, vz):
        """Hot equilibrium distribution function (normalized to bulk density). Flat evaluation."""
        return

    @abstractmethod
    def massdens_eq_phys(self, x, y, z, flat_eval=False):
        """Hot equilibrium mass density (normalized to bulk density). Flat evaluation."""
        return

    @abstractmethod
    def jh_x_eq_phys(self, x, y, z):
        """Hot equilibrium current density in x-direction."""
        return

    @abstractmethod
    def jh_y_eq_phys(self, x, y, z):
        """Hot equilibrium current density in y-direction."""
        return

    @abstractmethod
    def jh_z_eq_phys(self, x, y, z):
        """Hot equilibrium current density in z-direction."""
        return

    def fh0_eq(self, eta1, eta2, eta3, vx, vy, vz):
        """EP equilibrium distribution as 0-form. Args must by np.arrays of same shape."""

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'

        # must do evaluation here, because pull needs an array as input (not a 6d callable)
        X = self.DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
        Y = self.DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
        Z = self.DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
        temp = self.fh_eq_phys(X, Y, Z, vx, vy, vz)

        return self.DOMAIN.pull(temp, eta1, eta2, eta3, '0_form', flat_eval=True)

    def fh3_eq(self, eta1, eta2, eta3, vx, vy, vz):
        """EP equilibrium distribution as 3-form. Args must by np.arrays of same shape."""

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'

        # must do evaluation here, because pull needs an array as input (not a 6d callable)
        X = self.DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
        Y = self.DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
        Z = self.DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
        temp = self.fh_eq_phys(X, Y, Z, vx, vy, vz)

        return self.DOMAIN.pull(temp, eta1, eta2, eta3, '3_form', flat_eval=True)

    def nh0_eq(self, eta1, eta2, eta3):
        """EP equilibrium density as 0-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp = self.massdens_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull(temp, E1, E2, E3, '0_form')

    def nh3_eq(self, eta1, eta2, eta3):
        """EP equilibrium density as 3-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp = self.massdens_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull(temp, E1, E2, E3, '3_form')

    def jh1_eq_1(self, eta1, eta2, eta3):
        """First component of EP equilibrium current density as 1-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp_x = self.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_1')

    def jh1_eq_2(self, eta1, eta2, eta3):
        """Second component of EP equilibrium current density as 1-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp_x = self.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_2')

    def jh1_eq_3(self, eta1, eta2, eta3):
        """Third component of EP equilibrium current density as 1-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp_x = self.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_3')

    def jh2_eq_1(self, eta1, eta2, eta3):
        """First component of EP equilibrium current density as 2-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp_x = self.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_1')

    def jh2_eq_2(self, eta1, eta2, eta3):
        """Second component of EP equilibrium current density as 2-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp_x = self.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_2')

    def jh2_eq_3(self, eta1, eta2, eta3):
        """Third component of EP equilibrium current density as 2-form."""
        assert hasattr(self, 'DOMAIN'), 'Domain object has not been set yet!'
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(
            eta1, eta2, eta3, flat_eval=False)
        temp_x = self.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_3')


class MaxwellHomogenSlab(EquilibriumKinetic6D):
    """
    Maxwellian distribution, homogeneous in space:

    f = f(vx,vy,vz) = exp( -(vx - v0_x)^2/vth_x^2 ) / sqrt(pi) / vth_x
                    * exp( -(vy - v0_y)^2/vth_y^2 ) / sqrt(pi) / vth_y
                    * exp( -(vz - v0_z)^2/vth_z^2 ) / sqrt(pi) / vth_z

    Parameters:
    -----------
        params : dict
            contains the relevant parameters of the geometry
    """

    def __init__(self, params, DOMAIN=None):
        super().__init__(params, DOMAIN)

        self.nuh = params['nuh']
        self.p_mass = params['particle_mass']
        self.p_charge = params['particle_charge']

        self._MOMENTS = moments.Kinetic_homogen_slab(
            params['params_' + params['type']])
        self.EQ = gaussian.Gaussian_3d(self._MOMENTS)

    def fh_eq_phys(self, x, y, z, vx, vy, vz):
        """Hot equilibrium distribution function (normalized to bulk density). Flat evaluation."""
        return self.nuh * self._MOMENTS.nh_eq(x, y, z) * self.EQ.velocity_distribution(x, y, z, vx, vy, vz)

    def massdens_eq_phys(self, x, y, z, flat_eval=False):
        """Hot equilibrium mass density (normalized to bulk density). Flat evaluation."""
        return self.p_mass * self.nuh * self._MOMENTS.nh_eq(x, y, z)

    def jh_x_eq_phys(self, x, y, z):
        """Hot equilibrium current density in x-direction."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self._MOMENTS.nh_eq(E1, E2, E3) * self.EQ.uh_eq_x(E1, E2, E3)

    def jh_y_eq_phys(self, x, y, z):
        """Hot equilibrium current density in y-direction."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self._MOMENTS.nh_eq(E1, E2, E3) * self.EQ.uh_eq_y(E1, E2, E3)

    def jh_z_eq_phys(self, x, y, z):
        """Hot equilibrium current density in z-direction."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self._MOMENTS.nh_eq(E1, E2, E3) * self.EQ.uh_eq_z(E1, E2, E3)

    def Ph_xx_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_xx."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * \
            self._MOMENTS.nh_eq(E1, E2, E3) * self.EQ.sig_x(E1, E2, E3)**2/.2
        return value

    def Ph_yy_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_yy."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * \
            self._MOMENTS.nh_eq(E1, E2, E3) * self.EQ.sig_y(E1, E2, E3)**2/.2
        return value

    def Ph_zz_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_zz."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * \
            self._MOMENTS.nh_eq(E1, E2, E3) * self.EQ.sig_z(E1, E2, E3)**2/.2
        return value
