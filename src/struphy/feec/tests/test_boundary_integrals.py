import logging
from typing import Callable

import cunumpy as xp
import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy import domains
from struphy.feec.boundary_mass import BoundaryIntegralOperators
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("num_elements", [[8, 9, 10]],)
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [(None, None, None),
                                 (("free", "free"), None, None),
                                 (None, ("free", "free"), None),
                                 (None, None, ("free", "free")),
                                 (("free", "free"), ("free", "free"), None),
                                 (("free", "free"), ("free", "free"),  ("free", "free")),])
def test_boundary_mass_unit_cube_constant(num_elements, degree, bcs):
    """
    Tests the boundary mass operator for alpha = 1 on the unit cube.
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    face_value = 2.1
    alpha = lambda e1, e2, e3: xp.ones_like(e1) * face_value

    num_faces = 0
    for face_tuple in bcs:
        if face_tuple is None:
            continue
        if face_tuple[0] == "free":
            num_faces += 1
        if face_tuple[1] == "free":
            num_faces += 1

    exact = num_faces * face_value

    P = L2Projector("H1", mass_ops)
    alpha_h = P(alpha)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    v = bnd_ops.S0.dot(alpha_h)

    numerical = xp.sum(v.toarray())

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 1e-3


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [(("dirichlet", "free"), ("free", "free"), ("free", "free")),
                                (("free", "dirichlet"), ("free", "free"), ("free", "free")),
                                (("dirichlet", "dirichlet"), ("free", "free"), ("free", "free"))])
def test_boundary_mass_unit_cube_nonconstant(num_elements, degree, bcs):
    """
    Tests the boundary mass operator for alpha = eta1 + eta2 + eta3 on the unit cube.
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    if bcs[0] == ("dirichlet", "free"):
        alpha = lambda e1, e2, e3: e1 + 0 * e2 + 0 * e3
        exact = 3.0
    elif bcs[0] == ("free", "dirichlet"):
        alpha = lambda e1, e2, e3: 1.0 - e1 + 0 * e2 + 0 * e3
        exact = 3.0
    else:
        assert bcs[0] == ("dirichlet", "dirichlet")
        alpha = lambda e1, e2, e3: e1 * (1.0 - e1) + 0 * e2 + 0 * e3
        exact = 2.0 / 3.0

    P = L2Projector("H1", mass_ops)
    alpha_h = P(alpha, apply_bc=True)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    v = bnd_ops.S0.dot(alpha_h)

    numerical = xp.sum(v.toarray())

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 2e-2


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
def test_boundary_mass_cuboid_nontrivial(num_elements, degree, bcs):
    """
    Tests the boundary mass operator for alpha = eta1 + eta2 + eta3
    on a non-unit cuboid [-1,1] x [-1,3] x [0,3].
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.Cuboid(l1=-1.0, r1=1.0, l2=-1.0, r2=3.0, l3=0.0, r3=3.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: e1 + e2 + e3
    exact = 78.0

    P = L2Projector("H1", mass_ops)
    alpha_h = P(alpha)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    v = bnd_ops.S0.dot(alpha_h)

    numerical = xp.sum(v.toarray())

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 1e-3


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [(("free", "free"), None, ("free", "free"))])
def test_boundary_mass_hollow_cylinder_nonconstant(num_elements, degree, bcs):
    """
    Tests the boundary mass operator for alpha = exp(eta3) on a HollowCylinder.
    """
    import math
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    a1 = 0.2
    a2 = 1.0
    Lz = 4.0

    domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: xp.exp(e3)
    e = math.e
    exact = xp.pi * (
        2 * a1 * Lz * (e - 1)
        + 2 * a2 * Lz * (e - 1)
        + (a2**2 - a1**2) * (1 + e)
    )

    P = L2Projector("H1", mass_ops)
    alpha_h = P(alpha)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    v = bnd_ops.S0.dot(alpha_h)

    numerical = xp.sum(v.toarray())

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 1e-2


if __name__ == "__main__":
    from struphy import set_logging_level
    set_logging_level(logging.INFO)

    test_boundary_mass_unit_cube_constant(
        [8, 9, 10],
        [1, 2, 3],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )
    
    test_boundary_mass_unit_cube_nonconstant(
        [8, 9, 10],
        [1, 2, 3],
        (("dirichlet", "free"), ("free", "free"), ("free", "free")),
    )
    
    test_boundary_mass_cuboid_nontrivial(
        [8, 9, 10],
        [1, 2, 3],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )
    test_boundary_mass_hollow_cylinder_nonconstant(
        [8, 9, 10],
        [1, 2, 3],
        (("free", "free"), None, ("free", "free")),
    )