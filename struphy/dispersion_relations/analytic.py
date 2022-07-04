import numpy as np
import scipy.special as sp

from struphy.dispersion_relations.base import DispersionRelations1D


class Maxwell1D(DispersionRelations1D):
    '''Dispersion relation for Maxwell's equation in vacuum in Struphy units (see ``Maxwell`` in :ref:`models`):
    
    .. math::
    
        \omega^2 = k^2 \,.
    '''

    def __init__(self):
        super().__init__('light wave', c=1.) 

    def spectrum(self, kvec, kperp=None):

        # One complex array for each branch
        tmps = []
        for n in range(self.nbranches):
            tmps += [np.zeros_like(kvec, dtype=complex)]

        ########### Model specific part ##############################
        # first branch
        tmps[0][:] = self.params['c'] * kvec
        ##############################################################

        # fill output dictionary
        dict = {}
        for name, tmp in zip(self.branches, tmps):
            dict[name] = tmp

        return dict