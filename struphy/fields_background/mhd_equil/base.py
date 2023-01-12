from abc import ABCMeta, abstractmethod
import numpy as np


class MHDequilibrium(metaclass=ABCMeta):
    """
    Base class for Struphy MHD equilibria, analytical or numerical.
    The callables B, J, p etc. have to be provided through the child classes `AnalyticMHDequilibrium` or `NumericalMHDequilibrium`.
    The base class provides transformations of callables to different representations or coordinates.
    For numerical equilibria, the methods absB0, bv, unit_bv, j2, p0 and n0 are overidden by the child class.   
    """

    @property
    @abstractmethod
    def domain(self):
        """ Domain object that characterizes the mapping from the logical cube [0, 1]^3 to the physical domain.
        """
        pass

    def absB0(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ 0-form absolute value of equilibrium magnetic field in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.absB], eta1, eta2, eta3, '0_form', flat_eval, squeeze_output)

    def bv_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.b_x, self.b_y, self.b_z], eta1, eta2, eta3, 'vector_1', flat_eval, squeeze_output)

    def bv_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ Second contra-variant component (eta2) of magnetic field on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.b_x, self.b_y, self.b_z], eta1, eta2, eta3, 'vector_2', flat_eval, squeeze_output)

    def bv_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ Third contra-variant component (eta3) of magnetic field on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.b_x, self.b_y, self.b_z], eta1, eta2, eta3, 'vector_3', flat_eval, squeeze_output)

    def b1_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (eta1-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.bv_1, self.bv_2, self.bv_3], eta1, eta2, eta3, 'v_to_1_1', flat_eval, squeeze_output)

    def b1_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (eta2-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.bv_1, self.bv_2, self.bv_3], eta1, eta2, eta3, 'v_to_1_2', flat_eval, squeeze_output)

    def b1_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium magnetic field (eta3-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.bv_1, self.bv_2, self.bv_3], eta1, eta2, eta3, 'v_to_1_3', flat_eval, squeeze_output)

    def b2_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (eta1-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.bv_1, self.bv_2, self.bv_3], eta1, eta2, eta3, 'v_to_2_1', flat_eval, squeeze_output)

    def b2_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (eta2-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.bv_1, self.bv_2, self.bv_3], eta1, eta2, eta3, 'v_to_2_2', flat_eval, squeeze_output)

    def b2_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 2-form equilibrium magnetic field (eta3-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.bv_1, self.bv_2, self.bv_3], eta1, eta2, eta3, 'v_to_2_3', flat_eval, squeeze_output)

    def unit_bv_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ Unit vector equilibrium magnetic field (eta1-component, contra-variant) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.unit_b_x, self.unit_b_y, self.unit_b_z], eta1, eta2, eta3, 'vector_1', flat_eval, squeeze_output)

    def unit_bv_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ Unit vector equilibrium magnetic field (eta2-component, contra-variant) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.unit_b_x, self.unit_b_y, self.unit_b_z], eta1, eta2, eta3, 'vector_2', flat_eval, squeeze_output)

    def unit_bv_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ Unit vector equilibrium magnetic field (eta3-component, contra-variant) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.unit_b_x, self.unit_b_y, self.unit_b_z], eta1, eta2, eta3, 'vector_3', flat_eval, squeeze_output)

    def unit_b1_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Unit vector equilibrium magnetic field (eta1-component, 1-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_bv_1, self.unit_bv_2, self.unit_bv_3], eta1, eta2, eta3, 'v_to_1_1', flat_eval, squeeze_output)

    def unit_b1_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Unit vector equilibrium magnetic field (eta2-component, 1-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_bv_1, self.unit_bv_2, self.unit_bv_3], eta1, eta2, eta3, 'v_to_1_2', flat_eval, squeeze_output)

    def unit_b1_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Unit vector equilibrium magnetic field (eta3-component, 1-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_bv_1, self.unit_bv_2, self.unit_bv_3], eta1, eta2, eta3, 'v_to_1_3', flat_eval, squeeze_output)

    def unit_b2_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Unit vector equilibrium magnetic field (eta1-component, 2-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_bv_1, self.unit_bv_2, self.unit_bv_3], eta1, eta2, eta3, 'v_to_2_1', flat_eval, squeeze_output)

    def unit_b2_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Unit vector equilibrium magnetic field (eta2-component, 2-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_bv_1, self.unit_bv_2, self.unit_bv_3], eta1, eta2, eta3, 'v_to_2_2', flat_eval, squeeze_output)

    def unit_b2_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Unit vector equilibrium magnetic field (eta3-component, 2-form) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.unit_bv_1, self.unit_bv_2, self.unit_bv_3], eta1, eta2, eta3, 'v_to_2_3', flat_eval, squeeze_output)

    def j2_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.j_x, self.j_y, self.j_z], eta1, eta2, eta3, '2_form_1', flat_eval, squeeze_output)

    def j2_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.j_x, self.j_y, self.j_z], eta1, eta2, eta3, '2_form_2', flat_eval, squeeze_output)

    def j2_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.j_x, self.j_y, self.j_z], eta1, eta2, eta3, '2_form_3', flat_eval, squeeze_output)

    def j1_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium current (eta1-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], eta1, eta2, eta3, '2_to_1_1', flat_eval, squeeze_output)

    def j1_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium current (eta2-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], eta1, eta2, eta3, '2_to_1_2', flat_eval, squeeze_output)

    def j1_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 1-form equilibrium current (eta3-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], eta1, eta2, eta3, '2_to_1_3', flat_eval, squeeze_output)

    def jv_1(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Vector-field equilibrium current (eta1-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], eta1, eta2, eta3, '2_to_v_1', flat_eval, squeeze_output)

    def jv_2(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Vector-field equilibrium current (eta2-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], eta1, eta2, eta3, '2_to_v_2', flat_eval, squeeze_output)

    def jv_3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ Vector-field equilibrium current (eta3-component) in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.j2_1, self.j2_2, self.j2_3], eta1, eta2, eta3, '2_to_v_3', flat_eval, squeeze_output)

    def p0(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ 0-form equilibrium pressure in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.p], eta1, eta2, eta3, '0_form', flat_eval, squeeze_output)

    def p3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 3-form equilibrium pressure in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.p0], eta1, eta2, eta3, '0_to_3', flat_eval, squeeze_output)

    def n0(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=False):
        """ 0-form equilibrium number density in logical space.
        This method is overidden by NumericalMHDequilibrium.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.pull([self.n], eta1, eta2, eta3, '0_form', flat_eval, squeeze_output)

    def n3(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """ 3-form equilibrium number density in logical space.
        """
        assert self.domain is not None, 'Domain not set, use obj.domain=...'
        return self.domain.transform([self.n0], eta1, eta2, eta3, '0_to_3', flat_eval, squeeze_output)


class AnalyticalMHDequilibrium(MHDequilibrium):
    """
    Base class for analytical MHD equilibria. B, J, n and p have to be specified in Cartesian coordinates.  
    The domain must be set using the setter method.     
    """

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        assert hasattr(self, '_domain'), 'Domain for analytical MHD equilibrium not set. Please do obj.domain = ...'
        return self._domain

    @domain.setter
    def domain(self, domain):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        self._domain = domain

    @abstractmethod
    def b_x(self, x, y, z):
        """ Equilibrium magnetic field (x-component) in physical space.
        """
        pass

    @abstractmethod
    def b_y(self, x, y, z):
        """ Equilibrium magnetic field (y-component) in physical space.
        """
        pass

    @abstractmethod
    def b_z(self, x, y, z):
        """ Equilibrium magnetic field (z-component) in physical space.
        """
        pass

    @abstractmethod
    def j_x(self, x, y, z):
        """ Equilibrium current (x-component, curl of equilibrium magnetic field) in physical space.
        """
        pass

    @abstractmethod
    def j_y(self, x, y, z):
        """ Equilibrium current (y-component, curl of equilibrium magnetic field) in physical space.
        """
        pass

    @abstractmethod
    def j_z(self, x, y, z):
        """ Equilibrium current (z-component, curl of equilibrium magnetic field) in physical space.
        """
        pass

    @abstractmethod
    def p(self, x, y, z):
        """ Equilibrium pressure in physical space.
        """
        pass

    @abstractmethod
    def n(self, x, y, z):
        """ Equilibrium number density in physical space.
        """
        pass

    def absB(self, x, y, z):
        """ Equilibrium magnetic field (absolute value).
        """
        bx = self.b_x(x, y, z)
        by = self.b_y(x, y, z)
        bz = self.b_z(x, y, z)

        return np.sqrt(bx**2 + by**2 + bz**2)

    def unit_b_x(self, x, y, z):
        """ Unit vector equilibrium magnetic field (x-component) in physical space.
        """
        return self.b_x(x, y, z) / self.absB(x, y, z)

    def unit_b_y(self, x, y, z):
        """ Unit vector equilibrium magnetic field (y-component) in physical space.
        """
        return self.b_y(x, y, z) / self.absB(x, y, z)

    def unit_b_z(self, x, y, z):
        """ Unit vector equilibrium magnetic field (z-component) in physical space.
        """
        return self.b_z(x, y, z) / self.absB(x, y, z)


class NumericalMHDequilibrium(MHDequilibrium):
    """
    Base class for numerical MHD equilibria. 
    B, J, p and n must be specified on the logical cube [0, 1]^3. 
    B in contra-variant coordinates (i.e. as a vector-field), J as a 2-form, p and n as a 0-form.       
    """

    @abstractmethod
    def bv_1(self, eta1, eta2, eta3):
        """First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def bv_2(self, eta1, eta2, eta3):
        """Second contra-variant component (eta2) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def bv_3(self, eta1, eta2, eta3):
        """Third contra-variant component (eta3) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def j2_1(self, eta1, eta2, eta3):
        """First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def j2_2(self, eta1, eta2, eta3):
        """Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def j2_3(self, eta1, eta2, eta3):
        """Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def p0(self, eta1, eta2, eta3):
        """0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        pass

    @abstractmethod
    def n0(self, eta1, eta2, eta3):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        pass

    def absB0(self, eta1, eta2, eta3):
        """ 0-form absolute value of equilibrium magnetic field on logical cube [0, 1]^3.
        """

        tmp1 = self.bv_1(eta1, eta2, eta3)
        tmp2 = self.bv_2(eta1, eta2, eta3)
        tmp3 = self.bv_3(eta1, eta2, eta3)

        bx = self.domain.push(
            [tmp1, tmp2, tmp3], eta1, eta2, eta3, 'vector_1')
        by = self.domain.push(
            [tmp1, tmp2, tmp3], eta1, eta2, eta3, 'vector_2')
        bz = self.domain.push(
            [tmp1, tmp2, tmp3], eta1, eta2, eta3, 'vector_3')

        return np.sqrt(bx**2 + by**2 + bz**2)

    def unit_bv_1(self, eta1, eta2, eta3):
        """ Unit vector equilibrium magnetic field (eta1-component, contra-variant) on logical cube [0, 1]^3.
        """
        return self.bv_1(eta1, eta2, eta3) / self.absB0(eta1, eta2, eta3)

    def unit_bv_2(self, eta1, eta2, eta3):
        """ Unit vector equilibrium magnetic field (eta2-component, contra-variant) on logical cube [0, 1]^3.
        """
        return self.bv_2(eta1, eta2, eta3) / self.absB0(eta1, eta2, eta3)

    def unit_bv_3(self, eta1, eta2, eta3):
        """ Unit vector equilibrium magnetic field (eta3-component, contra-variant) on logical cube [0, 1]^3.
        """
        return self.bv_3(eta1, eta2, eta3) / self.absB0(eta1, eta2, eta3)
