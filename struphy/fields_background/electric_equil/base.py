from abc import ABCMeta, abstractmethod


class EquilibriumElectric(metaclass=ABCMeta):
    """
    Base class for electric field equilibria in Struphy.

    Parameters
    ----------
        params: dictionary
            Parameters that characterize the electric field equilibrium.

        domain: struphy.geometry.domains
            Enables pull-backs if set.
    """

    def __init__(self, params, domain=None):

        # set parameters
        self._params = params

        # set domain object
        if domain is not None:
            self._domain = domain

    @property
    def params(self):
        """ Parameters that characterize the electric field equilibrium.
        """
        return self._params

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    @domain.setter
    def domain(self, domain):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        self._domain = domain

    @abstractmethod
    def phi(self, x, y, z):
        """ Equilibrium electric potential in physical domain.
        """
        return

    def phi0(self, *etas):
        """ Equilibrium electric potential as 0-form, evaluated on logical unit cube.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.phi, *etas, kind='0_form')

    def phi3(self, *etas):
        """ Equilibrium electric potential as 3-form, evaluated on logical unit cube.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.phi, *etas, kind='3_form')
