import numpy as np

from struphy.fields_background.mhd_equil.base import NumericalMHDequilibrium


class EQDSKequilibrium(NumericalMHDequilibrium):

    def __init__(self, eqdsk_file):

        self._domain = 99

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    def bv_1(self, eta1, eta2, eta3):
        """First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        return 'GVECequilibrium'

    def bv_2(self, eta1, eta2, eta3):
        """First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        return 'GVECequilibrium'

    def bv_3(self, eta1, eta2, eta3):
        """First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        return 'GVECequilibrium'

    def j2_1(self, eta1, eta2, eta3):
        """First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def j2_2(self, eta1, eta2, eta3):
        """Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def j2_3(self, eta1, eta2, eta3):
        """Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def p0(self, eta1, eta2, eta3):
        """0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        pass

    def n0(self, eta1, eta2, eta3):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        pass


class GVECequilibrium(NumericalMHDequilibrium):

    def __init__(self, gvec_data_file):
        
        self._domain = 99

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    def bv_1(self, eta1, eta2, eta3):
        """First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        return 'GVECequilibrium'

    def bv_2(self, eta1, eta2, eta3):
        """First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        return 'GVECequilibrium'

    def bv_3(self, eta1, eta2, eta3):
        """First contra-variant component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        return 'GVECequilibrium'

    def j2_1(self, eta1, eta2, eta3):
        """First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def j2_2(self, eta1, eta2, eta3):
        """Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def j2_3(self, eta1, eta2, eta3):
        """Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def p0(self, eta1, eta2, eta3):
        """0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        pass

    def n0(self, eta1, eta2, eta3):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        pass


