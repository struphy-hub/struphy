#!/usr/bin/env python3

import yaml
import numpy as np

from scipy.integrate import quad
from scipy.integrate import fixed_quad

import scipy.special as sp

import struphy.feec.spline_space as spl
import struphy.feec.basics.mass_matrices_1d  as mass
import struphy.feec.basics.inner_products_1d as inner

from struphy.geometry import domain_3d


class Equilibrium_mhd:
    """
    Specifies an ideal MHD equilibrium, either analytic or from data. 

    Parameters
    ----------

        DOMAIN : DOMAIN object from struphy.geometry.domain_3d.py

        general : dict
            {'type': str_eq, 'particle_mass': float, 'rho0': float, 'beta_s': float}
            Available equilibira (choices for "str_eq") are
                "slab"              : specify p, B, J, rho on (x,y,z)-physical DOMAIN, then pulled back
                "analytic circular" :

        params_eq : dict
            Equilibrium parameters

    Methods
    -------

    """
    
    def __init__(self, DOMAIN, general, params_eq):
          
        self.DOMAIN    = DOMAIN
        self.general   = general
        self.params_eq = params_eq

    # pressure as 0-form:
    def p0_eq(self, eta1, eta2, eta3=0.):

        if self.general['type'] == 'slab':
            values = self.DOMAIN.pull(self.p_eq, eta1, eta2, eta3, '0_form')

        return values

    # equilibrium bulk pressure (physical domain, slab only)
    def p_eq(self, x, y, z=0.):

        arg_x, arg_y, arg_z = domain_3d.prepare_args(x, y, z)

        if self.general['type'] == 'slab':
            values = self.general['beta_s']/200. - 0*arg_x
        else:
            values = None

        return values

    