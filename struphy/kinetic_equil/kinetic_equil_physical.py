import numpy as np

from struphy.kinetic_equil.analytical import gaussian        
from struphy.kinetic_equil.analytical import maxwell_pitchangle 

from struphy.geometry import domain_3d

class Equilibrium_kinetic_physical:
    """
    Specifies an analytical kinetic equilibrium in (x, y, z)-coordinates.

    Parameters
    ----------
    general_kin : dict
        Keys are "type", "nuh", "particle_mass", "particle_charge" and "alpha" (see parameters.yml)

    params_kin : dict
        The parameters needed to define the kinetic equilibrium (see Notes).

    Attributes
    ----------
    kin_type : str
        Equilibrium type: "Maxwell_xyz", "Maxwell_pitchangle"

    nuh : float
        Ratio of EP to bulk number density, nuh = nh/N

    p_mass : float
        Particle mass in units of Proton mass

    p_charge : float
        Particle charge in units of elementary charge

    alpha : float
        Coupling parameter alpha = omega_ci/omega_A

    EQ : obj
        Analytic equilibirum from kinetic_eqil/analytical/

    vth : list
        Thermal velocities for drawing Gaussian markers

    shifts : list
        Velocity shifts for drawing Gaussian markers

    Notes
    -----
    Normalized velocity moments of Maxwellian:
        * massdens = nuh*mh*nh
        * jh_x     = nuh*qh*nh*uh_x
        * jh_y     = nuh*qh*nh*uh_y
        * jh_z     = nuh*qh*nh*uh_z
        * Ph_xx    = nuh*mh*nh*vth_x**2/2
        * Ph_yy    = nuh*mh*nh*vth_y**2/2
        * Ph_zz    = nuh*mh*nh*vth_z**2/2
        * Ph_xy    = ? add later
        * Ph_xz    = ? add later
        * Ph_yz    = ? add later

    """
    
    def __init__(self, general_kin, params_kin):
          
        self.kin_type = general_kin['type']
        self.nuh      = general_kin['nuh']
        self.p_mass   = general_kin['particle_mass']
        self.p_charge = general_kin['particle_charge']
        self.alpha    = general_kin['alpha']
        
        # create equilibrium object of given type
        if self.kin_type == 'Maxwell_xyz':
            self.EQ     = gaussian.Gaussian_3d(params_kin)
            self.vth    = [params_kin['vth_x'], params_kin['vth_y'], params_kin['vth_z']]
            self.shifts = [params_kin['v0_x'], params_kin['v0_y'], params_kin['v0_z']] 
        elif self.kin_type == 'Maxwell_pitchangle':
            self.EQ     = maxwell_pitchangle.Maxwell_pitchangle(params_kin)
            self.vth    = [params_kin['vth'], params_kin['vth'], params_kin['vth']]
            self.shifts = [params_kin['alpha0'], params_kin['alpha0'], params_kin['alpha0']]
        else:
            raise ValueError('Unknown kinetic equilibrium specified.')

    def fh_eq_phys(self, x, y, z, vx, vy, vz):
        '''Hot equilibrium distribution function (normalized to bulk density). Flat evaluation.'''
        return self.massdens_eq_phys(x, y, z) * self.EQ.velocity_distribution(x, y, z, vx, vy, vz)

    def massdens_eq_phys(self, x, y, z, flat_eval=False):
        '''Hot equilibrium mass density (normalized to bulk density). Flat evaluation.'''
        return self.nuh * self.p_mass + 0.*x

    def jh_x_eq_phys(self, x, y, z):
        '''Hot equilibrium current density in x-direction.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self.EQ.nh(E1, E2, E3) * self.EQ.uh_x(E1, E2, E3)

    def jh_y_eq_phys(self, x, y, z):
        '''Hot equilibrium current density in y-direction.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self.EQ.nh(E1, E2, E3) * self.EQ.uh_y(E1, E2, E3)

    def jh_z_eq_phys(self, x, y, z):
        '''Hot equilibrium current density in z-direction.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z, flat_eval=False)
        return self.nuh * self.p_charge * self.EQ.nh(E1, E2, E3) * self.EQ.uh_z(E1, E2, E3)

    def Ph_xx_eq_phys(self, x, y, z):
        '''Hot equilibrium pressure tensor, component P_xx.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * self.EQ.nh(E1, E2, E3) * self.EQ.sig_x(E1, E2, E3)**2/.2                                                                                            
        return value

    def Ph_yy_eq_phys(self, x, y, z):
        '''Hot equilibrium pressure tensor, component P_yy.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * self.EQ.nh(E1, E2, E3) * self.EQ.sig_y(E1, E2, E3)**2/.2                                                                                            
        return value

    def Ph_zz_eq_phys(self, x, y, z):
        '''Hot equilibrium pressure tensor, component P_zz.'''
        E1, E2, E3, is_sparse_meshgrid = domain_3d.prepare_args(x, y, z, flat_eval=False)
        value = self.nuh * self.p_mass * self.EQ.nh(E1, E2, E3) * self.EQ.sig_z(E1, E2, E3)**2/.2                                                                                            
        return value

    # TODO: off-diagonal terms of Ph

    