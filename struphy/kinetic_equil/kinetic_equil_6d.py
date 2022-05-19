from abc import ABCMeta, abstractmethod

from struphy.geometry                   import domain_3d
from struphy.kinetic_equil.analytical   import moments
from struphy.kinetic_equil.analytical   import gaussian  

class EquilibriumKinetic6D( metaclass=ABCMeta ):
    """
    The base class for kinetic equilibria for 6d Vlasov-type models

    Parameters:
    -----------
        params : dict
            contains the relevant parameters of the geometry
    """

    def __init__(self, params):
        self._params = params

    @property
    def params(self):
        """Dictionary of parameters defining the kinetic equilibrium."""
        return self._params

    def enable_pullbacks(self, DOMAIN):
        """Function that sets the DOMAIN object to enable pullbacks of fields."""
        self._DOMAIN = DOMAIN

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

    @abstractmethod
    def Ph_xx_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_xx."""
        return

    @abstractmethod
    def Ph_yy_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_yy."""
        return

    @abstractmethod
    def Ph_zz_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_zz."""
        return
    
    @abstractmethod
    def fh0_eq(self, eta1, eta2, eta3, vx, vy, vz):
        """EP equilibrium distribution as 0-form. Args must by np.arrays of same shape."""
        return

    @abstractmethod
    def fh3_eq(self, eta1, eta2, eta3, vx, vy, vz):
        """EP equilibrium distribution as 3-form. Args must by np.arrays of same shape."""
        return

    @abstractmethod
    def nh0_eq(self, eta1, eta2, eta3):
        """EP equilibrium density as 0-form."""
        return

    @abstractmethod
    def nh3_eq(self, eta1, eta2, eta3):
        """EP equilibrium density as 3-form."""
        return

    @abstractmethod
    def jh1_eq_1(self, eta1, eta2, eta3):
        """First component of EP equilibrium current density as 1-form."""
        return

    @abstractmethod
    def jh1_eq_2(self, eta1, eta2, eta3):
        """Second component of EP equilibrium current density as 1-form."""
        return

    @abstractmethod
    def jh1_eq_3(self, eta1, eta2, eta3):
        """Third component of EP equilibrium current density as 1-form."""
        return

    @abstractmethod
    def jh2_eq_1(self, eta1, eta2, eta3):
        """First component of EP equilibrium current density as 2-form."""
        return

    @abstractmethod
    def jh2_eq_2(self, eta1, eta2, eta3):
        """Second component of EP equilibrium current density as 2-form."""
        return

    @abstractmethod
    def jh2_eq_3(self, eta1, eta2, eta3):
        """Third component of EP equilibrium current density as 2-form."""
        return



class MaxwellHomogenSlab( EquilibriumKinetic6D ):
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

    def __init__(self, params):
        super().__init__(params)

        self.MOMENTS = moments.Kinetic_homogen_slab(params)
        self.EQ      = gaussian.Gaussian_3d(self.MOMENTS)
        self.vth     = [params['vth_x'], params['vth_y'], params['vth_z']]
        self.shifts  = [params['v0_x'], params['v0_y'], params['v0_z']] 


    def fh_eq_phys(self, x, y, z, vx, vy, vz):
        """Hot equilibrium distribution function (normalized to bulk density). Flat evaluation."""
        return self.nuh * self.MOMENTS.nh_eq(x, y, z) * self.EQ.velocity_distribution(x, y, z, vx, vy, vz)

    def massdens_eq_phys(self, x, y, z, flat_eval=False):
        """Hot equilibrium mass density (normalized to bulk density). Flat evaluation."""
        return self.p_mass * self.nuh * self.MOMENTS.nh_eq(x, y, z)

    def jh_x_eq_phys(self, x, y, z):
        """Hot equilibrium current density in x-direction."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self.MOMENTS.nh_eq(E1, E2, E3) * self.EQ.uh_eq_x(E1, E2, E3)

    def jh_y_eq_phys(self, x, y, z):
        """Hot equilibrium current density in y-direction."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self.MOMENTS.nh_eq(E1, E2, E3) * self.EQ.uh_eq_y(E1, E2, E3)

    def jh_z_eq_phys(self, x, y, z):
        """Hot equilibrium current density in z-direction."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self.MOMENTS.nh_eq(E1, E2, E3) * self.EQ.uh_eq_z(E1, E2, E3)

    def Ph_xx_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_xx."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * self.MOMENTS.nh_eq(E1, E2, E3) * self.EQ.sig_x(E1, E2, E3)**2/.2                                                                                            
        return value

    def Ph_yy_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_yy."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * self.MOMENTS.nh_eq(E1, E2, E3) * self.EQ.sig_y(E1, E2, E3)**2/.2                                                                                            
        return value

    def Ph_zz_eq_phys(self, x, y, z):
        """Hot equilibrium pressure tensor, component P_zz."""
        E1, E2, E3, = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * self.MOMENTS.nh_eq(E1, E2, E3) * self.EQ.sig_z(E1, E2, E3)**2/.2                                                                                            
        return value
    
    def fh0_eq(self, eta1, eta2, eta3, vx, vy, vz):
        """EP equilibrium distribution as 0-form. Args must by np.arrays of same shape."""

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape

        # must do evaluation here, because pull needs an array as input (not a 6d callable)
        X = self._DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
        Y = self._DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
        Z = self._DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
        temp = self.KINETC_P.fh_eq_phys(X, Y, Z, vx, vy, vz)

        return self._DOMAIN.pull(temp, eta1, eta2, eta3, '0_form', flat_eval=True)

    def fh3_eq(self, eta1, eta2, eta3, vx, vy, vz):
        """EP equilibrium distribution as 3-form. Args must by np.arrays of same shape."""

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape

        # must do evaluation here, because pull needs an array as input (not a 6d callable)
        X = self._DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
        Y = self._DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
        Z = self._DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
        temp = self.KINETC_P.fh_eq_phys(X, Y, Z, vx, vy, vz)

        return self._DOMAIN.pull(temp, eta1, eta2, eta3, '3_form', flat_eval=True)

    def nh0_eq(self, eta1, eta2, eta3):
        """EP equilibrium density as 0-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp = self.KINETC_P.massdens_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull(temp, E1, E2, E3, '0_form')

    def nh3_eq(self, eta1, eta2, eta3):
        """EP equilibrium density as 3-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp = self.KINETC_P.massdens_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull(temp, E1, E2, E3, '3_form')

    def jh1_eq_1(self, eta1, eta2, eta3):
        """First component of EP equilibrium current density as 1-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_1')

    def jh1_eq_2(self, eta1, eta2, eta3):
        """Second component of EP equilibrium current density as 1-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_2')

    def jh1_eq_3(self, eta1, eta2, eta3):
        """Third component of EP equilibrium current density as 1-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_3')

    def jh2_eq_1(self, eta1, eta2, eta3):
        """First component of EP equilibrium current density as 2-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_1')

    def jh2_eq_2(self, eta1, eta2, eta3):
        """Second component of EP equilibrium current density as 2-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_2')

    def jh2_eq_3(self, eta1, eta2, eta3):
        """Third component of EP equilibrium current density as 2-form."""
        E1, E2, E3, = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self._DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_3')
