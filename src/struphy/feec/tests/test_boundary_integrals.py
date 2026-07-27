import logging
from typing import Callable

import cunumpy as xp
import pytest
from feectools.ddm.mpi import mpi as MPI

from struphy import domains
from struphy.feec.boundary_integrals import BoundaryIntegralOperator
from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")


@pytest.mark.parametrize("num_elements", [[8, 8, 8]])
@pytest.mark.parametrize("degree", [[2, 2, 2]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
def test_boundary_integral_callable(num_elements, degree, bcs):
    """
    Tests the boundary integral operator for a callable function alpha on the
    unit cube (Cuboid domain, identity mapping).
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: xp.ones_like(e1)
    exact = 6.0

    bnd_op = BoundaryIntegralOperator(mass_ops)
    v = bnd_op.assemble_callable(alpha)

    numerical = xp.sum(v.toarray())
    
    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    
    assert xp.abs(numerical - exact) < 1e-10


@pytest.mark.parametrize("num_elements", [[8, 8, 8]])
@pytest.mark.parametrize("degree", [[2, 2, 2]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
def test_boundary_integral_callable_nonconstant(num_elements, degree, bcs):
    """
    Tests the boundary integral operator for a non-constant callable alpha
    on the unit cube (Cuboid domain, identity mapping).
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: e1 + e2 + e3
    exact = 9.0

    bnd_op = BoundaryIntegralOperator(mass_ops)
    v = bnd_op.assemble_callable(alpha)

    pads = v.space.pads
    numerical = xp.sum(v._data[pads[0]:-pads[0], pads[1]:-pads[1], pads[2]:-pads[2]])

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 1e-10


@pytest.mark.parametrize("num_elements", [[8, 8, 8]])
@pytest.mark.parametrize("degree", [[2, 2, 2], [3, 3, 3]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
def test_boundary_integral_callable_cuboid_nontrivial(num_elements, degree, bcs):
    """
    Tests the boundary integral operator for a non-constant callable alpha
    on a non-cubic cuboid [0,1] x [0,2] x [0,3].
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=2.0, l3=0.0, r3=3.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: e1 + e2 + e3
    exact = 33.0

    bnd_op = BoundaryIntegralOperator(mass_ops)
    v = bnd_op.assemble_callable(alpha)

    numerical = xp.sum(v.toarray())

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 1e-10


@pytest.mark.parametrize("num_elements", [[8, 8, 8]])
@pytest.mark.parametrize("degree", [[2, 2, 2], [3, 3, 3]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
def test_boundary_integral_callable_hollow_cylinder(num_elements, degree, bcs):
    """
    Tests the boundary integral operator for alpha = 1 on a HollowCylinder.
    """
    comm = MPI.COMM_WORLD

    grid = TensorProductGrid(num_elements=num_elements)
    derham_opts = DerhamOptions(degree=degree, bcs=bcs)
    derham = Derham(grid, derham_opts, comm=comm)

    domain = domains.HollowCylinder(a1=0.2, a2=1.0, Lz=4.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha = lambda e1, e2, e3: xp.ones_like(e1)
    exact = 11.52 * xp.pi

    bnd_op = BoundaryIntegralOperator(mass_ops)
    v = bnd_op.assemble_callable(alpha)

    numerical = xp.sum(v.toarray())

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")

    assert xp.abs(numerical - exact) < 1e-3


if __name__ == "__main__":
    from struphy import set_logging_level
    set_logging_level(logging.INFO)

    test_boundary_integral_callable(
        [8, 8, 8],
        [2, 2, 2],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )
    test_boundary_integral_callable_nonconstant(
        [8, 8, 8],
        [2, 2, 2],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )
    test_boundary_integral_callable_cuboid_nontrivial(
        [8, 8, 8],
        [2, 2, 2],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )
    test_boundary_integral_callable_hollow_cylinder(
        [8, 8, 8],
        [2, 2, 2],
        (("free", "free"), ("free", "free"), ("free", "free")),
    )