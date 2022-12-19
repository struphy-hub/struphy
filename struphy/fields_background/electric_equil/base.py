from abc import ABCMeta, abstractmethod

from struphy.psydac_api.fields import Field


class EquilibriumElectric(metaclass=ABCMeta):
    """
    Base class for electric field equilibria in Struphy.

    Parameters
    ----------
        params: dictionary
            Parameters that characterize the electric field equilibrium.

        domain: struphy.geometry.domains
            Enables pull-backs if set.

        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete Derham complex.
    """

    def __init__(self, params, domain=None, derham=None):

        # set parameters
        self._params = params

        # set domain object
        if domain is not None:
            self._domain = domain

        # set derham object
        if derham is not None:
            self._derham = derham

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

    @property
    def derham(self):
        """ derham object that contains the discrete derham complex.
        """
        return self._derham

    @domain.setter
    def domain(self, domain):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        self._domain = domain

    @derham.setter
    def derham(self, derham):
        """ derham object that contains the discrete derham complex.
        """
        self._derham = derham

    @abstractmethod
    def phi(self, x, y, z):
        """ Equilibrium electric potential in physical domain.
        """
        return

    def phi0(self, eta1, eta2, eta3):
        """ Equilibrium electric potential as 0-form, evaluated on logical unit cube.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.phi, eta1, eta2, eta3, '0_form')

    @property
    def phi0_vector(self):
        """ Coefficient vectors for equilibrium electric potential as 0-form.
        """
        assert hasattr(self, 'derham')
        return self.derham.P['0'](self.phi0)

    def phi3(self, eta1, eta2, eta3):
        """ Equilibrium electric potential as 3-form, evaluated on logical unit cube.
        """
        assert hasattr(self, 'domain')
        return self.domain.pull(self.phi, eta1, eta2, eta3, '3_form')

    @property
    def e1_vector(self):
        """ Stencil vector for equilibrium electric field as 1-form.
        """
        assert hasattr(self, 'derham')
        return self.derham.grad.dot(self.phi0_vector)
