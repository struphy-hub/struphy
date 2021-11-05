import numpy as np

from scipy.integrate import quad

import struphy.feec.spline_space as spl


class equilibrium_mhd_torus:
    """
    TODO
    """
    
    def __init__(self, params):
        
         
        # minor and major radius
        self.a  = params['params_cylinder']['a']
        self.r0 = params['params_cylinder']['R0']