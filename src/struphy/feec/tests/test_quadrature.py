"""Tests of :class:`~struphy.feec.quadrature.QuadratureIntegrator`."""

import cunumpy as xp
import numpy as np
import pytest
from maybempi import MPI

from struphy import domains
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.feec.quadrature import QuadratureIntegrator
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid


def _fun(k):
    return lambda e1, e2, e3: xp.sin(2 * xp.pi * e1 + k) * xp.cos(2 * xp.pi * e2) + (k + 1) * e3**2


@pytest.mark.parametrize(
    "space_id, domain",
    [
        ("H1", domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0)),
        ("L2", domains.Colella(Lx=1.0, Ly=1.0, alpha=0.1, Lz=1.0)),
        # the 3x3 geometric weight of the 2-form mass matrix is diagonal on a Cuboid
        ("Hdiv", domains.Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=1.5)),
    ],
)
def test_quadrature_integrator(space_id, domain):
    """The integrals equal the degrees of freedom of the L2 projector (``L2Projector.get_dofs``) of the values divided
    by the geometric weight of the mass matrix; with ``clear=False`` the integrals are accumulated."""
    derham = Derham(
        TensorProductGrid(num_elements=(8, 6, 4)),
        DerhamOptions(degree=(3, 2, 2), bcs=(("dirichlet", "dirichlet"), None, None)),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    mass_ops = WeightedMassOperators(derham, domain)
    proj = L2Projector(space_id, mass_ops)
    integ = QuadratureIntegrator(derham, space_id)

    if space_id in ("H1", "L2"):
        meshes, weights = [proj.quad_grid_mesh], [proj.geom_weights]
    else:
        meshes = proj.quad_grid_mesh
        weights = [proj.geom_weights[k][k] for k in range(3)]
        for k in range(3):
            for j in range(3):
                if j != k:
                    assert xp.allclose(proj.geom_weights[k][j], 0.0)
    vals_a = [_fun(k)(*mesh) for k, mesh in enumerate(meshes)]
    vals_b = [_fun(k + 3)(*mesh) for k, mesh in enumerate(meshes)]

    res = integ.integrate(vals_a)
    div = [v / w for v, w in zip(vals_a, weights)]
    ref = proj.get_dofs(div[0] if space_id in ("H1", "L2") else div, apply_bc=False)
    assert np.allclose(res.toarray(), ref.toarray(), rtol=1e-12, atol=1e-14 * np.max(np.abs(ref.toarray())))

    # accumulation, also after update_ghost_regions of the output
    acc = integ.integrate(vals_a)
    integ.integrate(vals_b, out=acc, clear=False)
    tot = integ.integrate([a + b for a, b in zip(vals_a, vals_b)])
    assert np.allclose(acc.toarray(), tot.toarray(), rtol=1e-12, atol=1e-14 * np.max(np.abs(tot.toarray())))


if __name__ == "__main__":
    pytest.main([__file__])
