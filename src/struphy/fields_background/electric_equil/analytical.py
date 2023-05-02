from struphy.fields_background.electric_equil.base import EquilibriumElectric


class HomogenSlab(EquilibriumElectric):
    """
    TODO
    """

    def __init__(self, params=None, domain=None):

        # set default parameters
        if params is None:
            params_default = {'phi0': 1.}
            super().__init__(params_default, domain)

        # or check if given parameter dictionary is complete
        else:
            assert 'phi0' in params
            super().__init__(params, domain)

    def phi(self, x, y, z):
        """ Equilibrium electric potential on physical domain.
        """
        return self.params['phi0'] - 0*x
