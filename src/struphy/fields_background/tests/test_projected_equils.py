"""Projected equilibria: coefficients in the FEEC space of the Derham (polar with polar splines) and their
tensor-product coefficients (projected_equil.tp)."""

import numpy as np
import pytest
from feectools.linalg.block import BlockVector
from feectools.linalg.stencil import StencilVector
from maybempi import MPI

NAMES = {"0": ("absB0", "p0", "n0"), "1": ("b1", "unit_b1", "gradB1"), "2": ("b2", "unit_b2"), "3": ("p3", "n3")}


def _setup(polar: bool):
    from struphy import domains, equils
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.psydac_derham import Derham
    from struphy.fields_background.projected_equils import ProjectedMHDequilibrium
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    num_elements, degree = (4, 6, 4), (2, 2, 2)
    if polar:
        domain = domains.IGAPolarCylinder(num_elements=num_elements[:2], degree=degree[:2], a=1.0, Lz=3.0)
    else:
        domain = domains.HollowCylinder(a1=0.1, a2=1.0, Lz=3.0)
    equil = equils.ScrewPinch(a=1.0, R0=3.0 / (2 * np.pi), B0=1.0)
    equil.domain = domain
    derham_opts = DerhamOptions(degree=degree, bcs=(("free", "dirichlet"), None, None), polar_splines=polar)
    derham = Derham(TensorProductGrid(num_elements=num_elements), derham_opts, comm=MPI.COMM_WORLD, domain=domain)
    return derham, WeightedMassOperators(derham, domain, eq_mhd=equil), ProjectedMHDequilibrium(equil, derham)


@pytest.mark.parametrize("polar", [False, True])
def test_feec_space_and_tensor_product_coefficients(polar):
    """The projections are in the space of the mass matrices of the Derham (PolarVectors with polar splines), their
    tensor-product coefficients are E^T of them (the same vectors without polar splines), and the energy of the
    equilibrium field is the same from the 2-form and the 1-form."""
    from struphy.polar.basic import PolarVector

    derham, mass_ops, peq = _setup(polar)
    spaces = {"0": mass_ops.M0.domain, "1": mass_ops.M1.domain, "2": mass_ops.M2.domain, "3": mass_ops.M3.domain}
    for form, names in NAMES.items():
        E = derham.extraction_ops[form]
        for name in names:
            v = getattr(peq, name)
            assert v.space == spaces[form], name
            assert isinstance(v, PolarVector) == polar, name
            t = getattr(peq.tp, name)
            assert isinstance(t, (StencilVector, BlockVector)), name
            assert t.space == E.domain if polar else t.space == v.space, name
            ref = E.transpose().dot(v)
            d = t - ref
            assert np.sqrt(d.inner(d)) <= 1e-14 * max(np.sqrt(ref.inner(ref)), 1.0), name

    # 1/2 int |B0|^2 from the 2-form and from the 1-form: the same up to the discretization error
    b1, b2 = peq.b1, peq.b2
    en_2 = 0.5 * b2.inner(mass_ops.M2.dot(b2, apply_bc=False))
    en_1 = 0.5 * b1.inner(mass_ops.M1.dot(b1, apply_bc=False))
    assert en_2 > 0.0
    assert abs(en_1 - en_2) < 1e-2 * en_2

    # tp is read-only access to the properties
    assert "b2" in dir(peq.tp)
    with pytest.raises(AttributeError):
        peq.tp._private
    with pytest.raises(AttributeError):
        peq.tp.derham
