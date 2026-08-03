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
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
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

    alpha = lambda e1, e2, e3: e1 + e2 + e3
    exact = 9.0

    P = L2Projector("H1", mass_ops)
    alpha_h = P(alpha)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    v = bnd_ops.S0.dot(alpha_h)

    numerical = xp.sum(v.toarray())

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 1e-3


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
def test_boundary_mass_cuboid_nontrivial(num_elements, degree, bcs):
    """
    Tests the boundary mass operator for alpha = eta1 + eta2 + eta3
    on a non-unit cuboid [0,2]^3.
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=2.0, l3=0.0, r3=2.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: e1 + e2 + e3
    exact = 36.0

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
def test_boundary_mass_hollow_cylinder(num_elements, degree, bcs):
    """
    Tests the boundary mass operator for alpha = 1 on a HollowCylinder.
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.HollowCylinder(a1=0.2, a2=1.0, Lz=4.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: xp.ones_like(e1)
    exact = 11.52 * xp.pi

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
        [8, 8, 8],
        [2, 2, 2],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )
    test_boundary_mass_cuboid_nontrivial(
        [8, 8, 8],
        [2, 2, 2],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )
    test_boundary_mass_hollow_cylinder(
        [8, 8, 8],
        [2, 2, 2],
        (("free", "free"), None, ("free", "free")),
    )