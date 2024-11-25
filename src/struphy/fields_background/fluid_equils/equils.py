'Pure fluid equilibria'

import numpy as np
import warnings

from struphy.fields_background.fluid_equils.base import FluidEquilibrium
from struphy.fields_background.mhd_equil.equils import set_defaults

class ConstantVelocity(FluidEquilibrium):
    r""" Base class for a constant distribution function on the unit cube. 
    The Background does not depend on the velocity

    """

    def __init__(self, **params):

        params_default = {'ux': 1.,
                          'uy': 1.,
                          'uz': 1.,
                          'n0': 1.,
                          'n1': 0.,
                          'density_profile' : 'affine',
                          'p0': 1.}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """ Parameters dictionary.
        """
        return self._params

    # equilibrium ion velocity 
    def u_xyz(self, x, y, z):
        """ Ion velocity.
        """
        ux = 0*x + self.params['ux']
        uy = 0*x + self.params['uy']
        uz = 0*x + self.params['uz']

        return ux, uy, uz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Plasma pressure.
        """
        pp = 0*x + self.params['p0']

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        if self.params['density_profile']=='constant':
            return self.params['n0'] + 0 * x
        elif self.params['density_profile']=='affine':
            return self.params['n0'] + self.params['n1'] * x