"Pure fluid equilibria"

import warnings

import numpy as np

from struphy.fields_background.fluid_equils.base import FluidEquilibrium
from struphy.fields_background.mhd_equil.equils import set_defaults


class ConstantVelocity(FluidEquilibrium):
    r"""Base class for a constant distribution function on the unit cube.
    The Background does not depend on the velocity

    """

    def __init__(self, **params):
        params_default = {"ux": 1.0, "uy": 1.0, "uz": 1.0, "n": 1.0, "n1": 0.0, "density_profile": "affine", "p0": 1.0}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """Parameters dictionary."""
        return self._params

    # equilibrium ion velocity
    def u_xyz(self, x, y, z):
        """Ion velocity."""
        ux = 0 * x + self.params["ux"]
        uy = 0 * x + self.params["uy"]
        uz = 0 * x + self.params["uz"]

        return ux, uy, uz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """Plasma pressure."""
        pp = 0 * x + self.params["p0"]

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """Number density."""
        if self.params["density_profile"] == "constant":
            return self.params["n"] + 0 * x
        elif self.params["density_profile"] == "affine":
            return self.params["n"] + self.params["n1"] * x



class AnalyticSolutionRestelli(FluidEquilibrium):
    r"""Base class for a constant distribution function on the unit cube.

    """

    def __init__(self, **params):
        params_default = {"R0": 2.0, "a": 1.0, "B0": 10, "Bp": 12.5, "alpha": 0.1, "beta": 1.0}

        self._params = set_defaults(params, params_default)

    @property
    def params(self):
        """Parameters dictionary."""
        return self._params

    # equilibrium ion velocity
    def u_xyz(self, x, y, z):
        """Velocity of ions and electrons."""
        ux = self.params["alpha"]*np.sqrt(x**2+y**2)/(self.params["a"]*self.params["R0"])*(-z) + self.params["beta"]*self.params["Bp"]*self.params["R0"]/(self.params["B0"]*self.params["a"]*np.sqrt(x**2+y**2))*z
        uy = self.params["alpha"]*np.sqrt(x**2+y**2)/(self.params["a"]*self.params["R0"])*(np.sqrt(x**2+y**2)-self.params["R0"]) + self.params["beta"]*self.params["Bp"]*self.params["R0"]/(self.params["B0"]*self.params["a"]*np.sqrt(x**2+y**2))*(-(np.sqrt(x**2+y**2)-self.params["R0"]))
        uz = self.params["beta"]*self.params["Bp"]*self.params["R0"]/(self.params["B0"]*self.params["a"]*np.sqrt(x**2+y**2))*self.params["B0"]*self.params["a"]/self.params["Bp"]

        return ux, uy, uz

    # equilibrium pressure
    def p_xyz(self, x, y, z):
        """Plasma pressure."""
        pp = 0.5*self.params["a"]*self.params["B0"]*self.params["alpha"]*(((np.sqrt(x**2+y**2)-self.params["R0"])**2 + z**2)/self.params["a"]**2-2/3)

        return pp

    # equilibrium number density
    def n_xyz(self, x, y, z):
        """Number density."""
        
        return 0 * x
