"""Tests of the div-div regularized kinetic metric of the variational fluid models."""

import numpy as np
import pytest
from maybempi import MPI

from struphy import DerhamOptions, domains, grids
from struphy.feec.basis_projection_ops import BasisProjectionOperators
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.feec.variational_utilities import kinetic_metric


def _setup(L1, L2, L3, rho_phys):
    domain = domains.Cuboid(r1=L1, r2=L2, r3=L3)
    derham = Derham(
        grids.TensorProductGrid(num_elements=(16, 4, 4)),
        DerhamOptions(degree=(3, 2, 2)),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    mass_ops = WeightedMassOperators(derham, domain)
    basis_ops = BasisProjectionOperators(derham, domain)
    # density 3-form rho^3 = rho * sqrt(g)
    J = L1 * L2 * L3
    mass_ops.WMMnew.spline_functions["l2_field"].vector = derham.projectors["3"](
        lambda e1, e2, e3: rho_phys(L1 * e1) * J + 0 * e2
    )
    mass_ops.WMMnew.assemble()
    mass_ops.WMM3new.assemble()
    return derham, mass_ops, basis_ops


def _velocity(derham, L1, L2, L3, ux, uy):
    """Coefficients of u = (ux(x), uy(x), 0) in H1vec (logical components u_hat = DF^{-1} u)."""
    return derham.projectors["v"](
        [
            lambda e1, e2, e3: ux(L1 * e1) / L1 + 0 * e2,
            lambda e1, e2, e3: uy(L1 * e1) / L2 + 0 * e2,
            lambda e1, e2, e3: 0 * e1,
        ]
    )


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_kinetic_metric_divdiv(alpha):
    L1, L2, L3 = 2.0, 3.0, 1.5
    k = 2 * np.pi / L1
    rho_phys = lambda x: 1.5 + 0.5 * np.sin(k * x)
    derham, mass_ops, basis_ops = _setup(L1, L2, L3, rho_phys)

    A = kinetic_metric(derham, mass_ops, basis_ops, divdiv_alpha=alpha)
    M = mass_ops.WMMnew
    assert kinetic_metric(derham, mass_ops, basis_ops, divdiv_alpha=0.0) is M

    # compressive field u = (sin(kx), 0, 0): div u = k cos(kx)
    u = _velocity(derham, L1, L2, L3, lambda x: np.sin(k * x), lambda x: 0 * x)
    divdiv = (A.dot_inner(u, u) - M.dot_inner(u, u)) / (2 * alpha)
    # int rho (div u)^2 dx = L2 L3 int_0^L1 (1.5 + 0.5 sin(kx)) k^2 cos^2(kx) dx = L2 L3 k^2 1.5 L1 / 2
    exact = L2 * L3 * k**2 * 1.5 * L1 / 2
    assert abs(divdiv - exact) / exact < 1e-2

    # divergence-free field u = (0, sin(kx), 0): D Uv u vanishes (commuting projectors)
    w = _velocity(derham, L1, L2, L3, lambda x: 0 * x, lambda x: np.sin(k * x))
    assert abs(A.dot_inner(w, w) - M.dot_inner(w, w)) < 1e-10 * M.dot_inner(w, w)

    # symmetry
    assert np.isclose(A.dot_inner(u, w), A.dot_inner(w, u), rtol=1e-12, atol=1e-12)


def test_kinetic_metric_negative_alpha():
    derham, mass_ops, basis_ops = _setup(1.0, 1.0, 1.0, lambda x: 1.0 + 0 * x)
    with pytest.raises(AssertionError):
        kinetic_metric(derham, mass_ops, basis_ops, divdiv_alpha=-1.0)
