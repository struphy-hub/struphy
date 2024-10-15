'Braginskii equilibria.'


import numpy as np
import warnings

from struphy.fields_background.braginskii_equil.base import BraginskiiEquilibrium
from struphy.fields_background.mhd_equil.equils import set_defaults


class HomogenSlabITG(BraginskiiEquilibrium):
    r"""
    Homogenous slab equilibrium with temperature gradient in x, B-field in z:

    .. math::

        \mathbf B &= B_{0z}\,\mathbf e_z = const.\,, \qquad n &= n_0 = const.

        p &= p_0*(1 - \frac{x}{L_x} ) + p_\textrm{min}\,,

        \mathbf u &= - \epsilon \frac{p_0}{L_x} \mathbf e_y\,.

    Units are those defned in the parameter file (:code:`struphy units -h`).

    Parameters
    ----------
    B0z : float  
        z-component of magnetic field (default: 1.).
    Lx : float
        Domain length in x; 1/Lx is the temperature scale length.
    p0 : float 
        Constant pressure coefficient (default: 1.).
    pmin : float
        Minimum pressure at x = Lx.
    n0 : float 
        Ion number density (default: 1.).
    eps : float
        The unit factor :math:`1/(\hat\Omega_i \hat t)`.

    Note
    ----
    In the parameter .yml, use the following in the section `mhd_equilibrium`::

        braginskii_equilibrium :
            type : HomogenSlabITG
            HomogenSlabITG :
                B0z  : 1. 
                Lx   : 1. 
                p0   : 1.
                pmin : .1
                n0   : 1. 
                eps  : .1
    """

    def __init__(self, **params):

        params_default = {'B0z': 1.,
                          'Lx': 1.,
                          'p0': 1.,
                          'pmin': .1,
                          'n0': 1.,
                          'eps': .1}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """ Parameters dictionary.
        """
        return self._params

    # ===============================================================
    #                  profiles on physical domain
    # ===============================================================

    # equilibrium magnetic field (curl of equilibrium vector potential)
    def b_xyz(self, x, y, z):
        """ Magnetic field.
        """
        bx = 0*x
        by = 0*x
        bz = self.params['B0z'] - 0*x

        return bx, by, bz

    # equilibrium ion velocity 
    def u_xyz(self, x, y, z):
        """ Ion velocity.
        """
        ux = 0*x
        uy = - self.params['eps']*self.params['p0']/self.params['Lx'] - 0*x
        uz = 0*x

        return ux, uy, uz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """ Plasma pressure.
        """
        pp = self.params['p0']*(1. - x/self.params['Lx']) + self.params['pmin']

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """ Number density.
        """
        nn = self.params['n0'] - 0*x

        return nn

    # equilibrium current (curl of equilibrium magnetic field)
    def gradB_xyz(self, x, y, z):
        """ Field strength gradient.
        """
        gradBx = 0*x
        gradBy = 0*x
        gradBz = 0*x

        return gradBx, gradBy, gradBz