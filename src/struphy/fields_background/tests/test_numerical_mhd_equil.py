import pytest

from struphy.fields_background.base import FluidEquilibrium, LogicalMHDequilibrium
from struphy.utils.arrays import xp


@pytest.mark.parametrize(
    "mapping",
    [
        ["Cuboid", {"l1": 1.0, "r1": 2.0, "l2": 10.0, "r2": 20.0, "l3": 100.0, "r3": 200.0}],
        ["HollowTorus", {"a1": 1.0, "a2": 2.0, "R0": 3.0, "tor_period": 1}],
        [
            "ShafranovDshapedCylinder",
            {
                "R0": 60.0,
                "Lz": 100.0,
                "delta_x": 0.06,
                "delta_y": 0.07,
                "delta_gs": 0.08,
                "epsilon_gs": 9.0,
                "kappa_gs": 10.0,
            },
        ],
    ],
)
@pytest.mark.parametrize("mhd_equil", ["HomogenSlab", "ShearedSlab", "ScrewPinch"])
def test_transformations(mapping, mhd_equil):
    """Test whether the class LogicalMHDequilibrium yields the same function values as CartesianMHDequilibrium.
    For this we construct an artificial numerical equilibrium from an analytical proxy."""

    from struphy.fields_background import equils
    from struphy.geometry import domains

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
    eta1 = xp.random.rand(4)
    eta2 = xp.random.rand(5)
    eta3 = xp.random.rand(6)

    assert xp.allclose(ana_equil.absB0(eta1, eta2, eta3), num_equil.absB0(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.bv(eta1, eta2, eta3)[0], num_equil.bv(eta1, eta2, eta3)[0])
    assert xp.allclose(ana_equil.bv(eta1, eta2, eta3)[1], num_equil.bv(eta1, eta2, eta3)[1])
    assert xp.allclose(ana_equil.bv(eta1, eta2, eta3)[2], num_equil.bv(eta1, eta2, eta3)[2])

    assert xp.allclose(ana_equil.b1_1(eta1, eta2, eta3), num_equil.b1_1(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.b1_2(eta1, eta2, eta3), num_equil.b1_2(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.b1_3(eta1, eta2, eta3), num_equil.b1_3(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.b2_1(eta1, eta2, eta3), num_equil.b2_1(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.b2_2(eta1, eta2, eta3), num_equil.b2_2(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.b2_3(eta1, eta2, eta3), num_equil.b2_3(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.unit_bv(eta1, eta2, eta3)[0], num_equil.unit_bv(eta1, eta2, eta3)[0])
    assert xp.allclose(ana_equil.unit_bv(eta1, eta2, eta3)[1], num_equil.unit_bv(eta1, eta2, eta3)[1])
    assert xp.allclose(ana_equil.unit_bv(eta1, eta2, eta3)[2], num_equil.unit_bv(eta1, eta2, eta3)[2])

    assert xp.allclose(ana_equil.unit_b1_1(eta1, eta2, eta3), num_equil.unit_b1_1(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.unit_b1_2(eta1, eta2, eta3), num_equil.unit_b1_2(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.unit_b1_3(eta1, eta2, eta3), num_equil.unit_b1_3(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.unit_b2_1(eta1, eta2, eta3), num_equil.unit_b2_1(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.unit_b2_2(eta1, eta2, eta3), num_equil.unit_b2_2(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.unit_b2_3(eta1, eta2, eta3), num_equil.unit_b2_3(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.jv(eta1, eta2, eta3)[0], num_equil.jv(eta1, eta2, eta3)[0])
    assert xp.allclose(ana_equil.jv(eta1, eta2, eta3)[1], num_equil.jv(eta1, eta2, eta3)[1])
    assert xp.allclose(ana_equil.jv(eta1, eta2, eta3)[2], num_equil.jv(eta1, eta2, eta3)[2])

    assert xp.allclose(ana_equil.j1_1(eta1, eta2, eta3), num_equil.j1_1(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.j1_2(eta1, eta2, eta3), num_equil.j1_2(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.j1_3(eta1, eta2, eta3), num_equil.j1_3(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.j2_1(eta1, eta2, eta3), num_equil.j2_1(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.j2_2(eta1, eta2, eta3), num_equil.j2_2(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.j2_3(eta1, eta2, eta3), num_equil.j2_3(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.p0(eta1, eta2, eta3), num_equil.p0(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.p3(eta1, eta2, eta3), num_equil.p3(eta1, eta2, eta3))

    assert xp.allclose(ana_equil.n0(eta1, eta2, eta3), num_equil.n0(eta1, eta2, eta3))
    assert xp.allclose(ana_equil.n3(eta1, eta2, eta3), num_equil.n3(eta1, eta2, eta3))


class NumEqTest(LogicalMHDequilibrium):
    def __init__(self, analytic_domain, analytic_mhd_equil):
        # use domain setter
        self.domain = analytic_domain

        # expose equilibrium
        self._equil = analytic_mhd_equil

    @LogicalMHDequilibrium.domain.setter
    def domain(self, new_domain):
        super(NumEqTest, type(self)).domain.fset(self, new_domain)

    def bv(self, *etas, squeeze_out=True):
        return self._equil.bv(*etas, squeeze_out=squeeze_out)

    def jv(self, *etas, squeeze_out=True):
        return self._equil.jv(*etas, squeeze_out=squeeze_out)

    def p0(self, *etas, squeeze_out=True):
        return self._equil.p0(*etas, squeeze_out=squeeze_out)

    def n0(self, *etas, squeeze_out=True):
        return self._equil.n0(*etas, squeeze_out=squeeze_out)

    def gradB1(self, *etas, squeeze_out=True):
        return self._equil.gradB1(*etas, squeeze_out=squeeze_out)


if __name__ == "__main__":
    test_transformations(["Colella", {"Lx": 1.0, "Ly": 2.0, "alpha": 0.5, "Lz": 3.0}], "HomogenSlab")
