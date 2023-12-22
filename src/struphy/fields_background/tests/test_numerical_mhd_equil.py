import pytest
import numpy as np
from struphy.fields_background.mhd_equil.base import LogicalMHDequilibrium


@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}],
    ['HollowTorus', {
        'a1': 1., 'a2': 2., 'R0': 3., 'tor_period': 1}],
    ['ShafranovDshapedCylinder', {
        'R0': 60., 'Lz': 100., 'delta_x': 0.06, 'delta_y': 0.07, 'delta_gs': 0.08, 'epsilon_gs': 9., 'kappa_gs': 10.}],
])
@pytest.mark.parametrize('mhd_equil', ['HomogenSlab', 'ShearedSlab', 'ScrewPinch'])
def test_transformations(mapping, mhd_equil):
    '''Test whether the class LogicalMHDequilibrium yields the same function values as CartesianMHDequilibrium.
    For this we construct an artificial numerical equilibrium from an analytical proxy.'''

    from struphy.geometry import domains
    from struphy.fields_background.mhd_equil import equils

    # domain (mapping from logical unit cube to physical domain)
    dom_type = mapping[0]
    dom_params = mapping[1]
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # analytical mhd equilibrium
    mhd_equil_class = getattr(equils, mhd_equil)
    ana_equil = mhd_equil_class()  # use default parameters

    # set mapping for analytical case
    ana_equil.domain = domain

    # numerical mhd equilibrium
    proxy = mhd_equil_class()  # proxy class with default parameters
    proxy.domain = domain
    num_equil = NumEqTest(domain, proxy)

    # compare values:
    eta1 = np.random.rand(4)
    eta2 = np.random.rand(5)
    eta3 = np.random.rand(6)

    assert np.allclose(ana_equil.absB0(eta1, eta2, eta3),
                       num_equil.absB0(eta1, eta2, eta3))

    assert np.allclose(ana_equil.bv(eta1, eta2, eta3)[0],
                       num_equil.bv(eta1, eta2, eta3)[0])
    assert np.allclose(ana_equil.bv(eta1, eta2, eta3)[1],
                       num_equil.bv(eta1, eta2, eta3)[1])
    assert np.allclose(ana_equil.bv(eta1, eta2, eta3)[2],
                       num_equil.bv(eta1, eta2, eta3)[2])

    assert np.allclose(ana_equil.b1_1(eta1, eta2, eta3),
                       num_equil.b1_1(eta1, eta2, eta3))
    assert np.allclose(ana_equil.b1_2(eta1, eta2, eta3),
                       num_equil.b1_2(eta1, eta2, eta3))
    assert np.allclose(ana_equil.b1_3(eta1, eta2, eta3),
                       num_equil.b1_3(eta1, eta2, eta3))

    assert np.allclose(ana_equil.b2_1(eta1, eta2, eta3),
                       num_equil.b2_1(eta1, eta2, eta3))
    assert np.allclose(ana_equil.b2_2(eta1, eta2, eta3),
                       num_equil.b2_2(eta1, eta2, eta3))
    assert np.allclose(ana_equil.b2_3(eta1, eta2, eta3),
                       num_equil.b2_3(eta1, eta2, eta3))

    assert np.allclose(ana_equil.unit_bv(eta1, eta2, eta3)[0],
                       num_equil.unit_bv(eta1, eta2, eta3)[0])
    assert np.allclose(ana_equil.unit_bv(eta1, eta2, eta3)[1],
                       num_equil.unit_bv(eta1, eta2, eta3)[1])
    assert np.allclose(ana_equil.unit_bv(eta1, eta2, eta3)[2],
                       num_equil.unit_bv(eta1, eta2, eta3)[2])

    assert np.allclose(ana_equil.unit_b1_1(eta1, eta2, eta3),
                       num_equil.unit_b1_1(eta1, eta2, eta3))
    assert np.allclose(ana_equil.unit_b1_2(eta1, eta2, eta3),
                       num_equil.unit_b1_2(eta1, eta2, eta3))
    assert np.allclose(ana_equil.unit_b1_3(eta1, eta2, eta3),
                       num_equil.unit_b1_3(eta1, eta2, eta3))

    assert np.allclose(ana_equil.unit_b2_1(eta1, eta2, eta3),
                       num_equil.unit_b2_1(eta1, eta2, eta3))
    assert np.allclose(ana_equil.unit_b2_2(eta1, eta2, eta3),
                       num_equil.unit_b2_2(eta1, eta2, eta3))
    assert np.allclose(ana_equil.unit_b2_3(eta1, eta2, eta3),
                       num_equil.unit_b2_3(eta1, eta2, eta3))

    assert np.allclose(ana_equil.jv(eta1, eta2, eta3)[0],
                       num_equil.jv(eta1, eta2, eta3)[0])
    assert np.allclose(ana_equil.jv(eta1, eta2, eta3)[1],
                       num_equil.jv(eta1, eta2, eta3)[1])
    assert np.allclose(ana_equil.jv(eta1, eta2, eta3)[2],
                       num_equil.jv(eta1, eta2, eta3)[2])

    assert np.allclose(ana_equil.j1_1(eta1, eta2, eta3),
                       num_equil.j1_1(eta1, eta2, eta3))
    assert np.allclose(ana_equil.j1_2(eta1, eta2, eta3),
                       num_equil.j1_2(eta1, eta2, eta3))
    assert np.allclose(ana_equil.j1_3(eta1, eta2, eta3),
                       num_equil.j1_3(eta1, eta2, eta3))

    assert np.allclose(ana_equil.j2_1(eta1, eta2, eta3),
                       num_equil.j2_1(eta1, eta2, eta3))
    assert np.allclose(ana_equil.j2_2(eta1, eta2, eta3),
                       num_equil.j2_2(eta1, eta2, eta3))
    assert np.allclose(ana_equil.j2_3(eta1, eta2, eta3),
                       num_equil.j2_3(eta1, eta2, eta3))

    assert np.allclose(ana_equil.p0(eta1, eta2, eta3),
                       num_equil.p0(eta1, eta2, eta3))
    assert np.allclose(ana_equil.p3(eta1, eta2, eta3),
                       num_equil.p3(eta1, eta2, eta3))

    assert np.allclose(ana_equil.n0(eta1, eta2, eta3),
                       num_equil.n0(eta1, eta2, eta3))
    assert np.allclose(ana_equil.n3(eta1, eta2, eta3),
                       num_equil.n3(eta1, eta2, eta3))


class NumEqTest(LogicalMHDequilibrium):

    def __init__(self, analytic_domain, analytic_mhd_equil):

        self._domain = analytic_domain
        self._equil = analytic_mhd_equil

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    def b2(self, *etas, squeeze_out=True):
        return self._equil.b2(*etas, squeeze_out=squeeze_out)

    def j2(self, *etas, squeeze_out=True):
        return self._equil.j2(*etas, squeeze_out=squeeze_out)

    def p0(self, *etas, squeeze_out=True):
        return self._equil.p0(*etas, squeeze_out=squeeze_out)

    def n0(self, *etas, squeeze_out=True):
        return self._equil.n0(*etas, squeeze_out=squeeze_out)

    def gradB1(self, *etas, squeeze_out=True):
        return self._equil.gradB1(*etas, squeeze_out=squeeze_out)

if __name__ == '__main__':
    test_transformations(
        ['Colella', {'Lx': 1., 'Ly': 2., 'alpha': .5, 'Lz': 3.}], 'HomogenSlab')
