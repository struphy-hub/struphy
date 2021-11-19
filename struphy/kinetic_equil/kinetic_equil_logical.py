from struphy.geometry import domain_3d


class Equilibrium_kinetic_logical:
    """
    Pullbacks of functions in kinetic_equil_physical.py (so far point-wise evaluation).

    Parameters
    ----------
    DOMAIN : object
        Domain object from domain_3d.py
    
    KINETIC_P : object
        Kinetic equilibirum in (x, y, z)-coordinates from kinetic_equil_physical.py
    """
    
    def __init__(self, DOMAIN, KINETIC_P):

        self.DOMAIN   = DOMAIN
        self.KINETC_P = KINETIC_P

    def fh0_eq(self, eta1, eta2, eta3, vx, vy, vz):
        '''EP equilibrium distribution as 0-form. Args must by np.arrays of same shape.'''

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape

        # must do evaluation here, because pull needs an array as input (not a 6d callable)
        X = self.DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
        Y = self.DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
        Z = self.DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
        temp = self.KINETC_P.fh_eq_phys(X, Y, Z, vx, vy, vz)

        return self.DOMAIN.pull(temp, eta1, eta2, eta3, '0_form', flat_eval=True)

    def fh3_eq(self, eta1, eta2, eta3, vx, vy, vz):
        '''EP equilibrium distribution as 3-form. Args must by np.arrays of same shape.'''

        assert eta1.shape == eta2.shape == eta3.shape == vx.shape == vy.shape == vz.shape

        # must do evaluation here, because pull needs an array as input (not a 6d callable)
        X = self.DOMAIN.evaluate(eta1, eta2, eta3, 'x', flat_eval=True)
        Y = self.DOMAIN.evaluate(eta1, eta2, eta3, 'y', flat_eval=True)
        Z = self.DOMAIN.evaluate(eta1, eta2, eta3, 'z', flat_eval=True)
        temp = self.KINETC_P.fh_eq_phys(X, Y, Z, vx, vy, vz)

        return self.DOMAIN.pull(temp, eta1, eta2, eta3, '3_form', flat_eval=True)

    def nh0_eq(self, eta1, eta2, eta3):
        '''EP equilibrium density as 0-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp = self.KINETC_P.massdens_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull(temp, E1, E2, E3, '0_form')

    def nh3_eq(self, eta1, eta2, eta3):
        '''EP equilibrium density as 3-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp = self.KINETC_P.massdens_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull(temp, E1, E2, E3, '3_form')

    def jh1_eq_1(self, eta1, eta2, eta3):
        '''First component of EP equilibrium current density as 1-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_1')

    def jh1_eq_2(self, eta1, eta2, eta3):
        '''Second component of EP equilibrium current density as 1-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_2')

    def jh1_eq_3(self, eta1, eta2, eta3):
        '''Third component of EP equilibrium current density as 1-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '1_form_3')

    def jh2_eq_1(self, eta1, eta2, eta3):
        '''First component of EP equilibrium current density as 2-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_1')

    def jh2_eq_2(self, eta1, eta2, eta3):
        '''Second component of EP equilibrium current density as 2-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_2')

    def jh2_eq_3(self, eta1, eta2, eta3):
        '''Third component of EP equilibrium current density as 2-form.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(eta1, eta2, eta3, flat_eval=False)
        temp_x = self.KINETC_P.jh_x_eq_phys(E1, E2, E3)
        temp_y = self.KINETC_P.jh_y_eq_phys(E1, E2, E3)
        temp_z = self.KINETC_P.jh_z_eq_phys(E1, E2, E3)
        return self.DOMAIN.pull([temp_x, temp_y, temp_z], E1, E2, E3, '2_form_3')