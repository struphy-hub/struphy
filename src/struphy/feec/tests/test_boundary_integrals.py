import logging

import cunumpy as xp
import pytest
from feectools.ddm.mpi import MockComm
from feectools.ddm.mpi import mpi as MPI

from struphy import domains
from struphy.feec.boundary_mass import BoundaryIntegralOperators
from struphy.feec.mass import L2Projector, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.io.options import DerhamOptions
from struphy.topology.grids import TensorProductGrid

logger = logging.getLogger("struphy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reduce(comm, arr):
    if isinstance(comm, MockComm):
        return arr
    out = xp.zeros_like(arr)
    comm.Allreduce(arr, out, op=MPI.SUM)
    return out


def _sum_coeffs(comm, v):
    return xp.sum(_reduce(comm, v.toarray()))


# ---------------------------------------------------------------------------
# ScalarBoundaryMass tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (None, None, None),
        (("free", "free"), None, None),
        (None, ("free", "free"), None),
        (None, None, ("free", "free")),
        (("free", "free"), ("free", "free"), None),
        (("free", "free"), ("free", "free"), ("free", "free")),
    ],
)
def test_scalar_unit_cube_constant(num_elements, degree, bcs):
    """ScalarBoundaryMass: alpha = 1 on the unit cube."""
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    face_value = 2.1
    num_faces = sum(
        (1 if ft[0] == "free" else 0) + (1 if ft[1] == "free" else 0)
        for ft in bcs if ft is not None
    )
    exact = num_faces * face_value

    alpha_h = L2Projector("H1", mass_ops)(lambda e1, e2, e3: xp.ones_like(e1) * face_value)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    numerical = _sum_coeffs(comm, bnd_ops.scalar().dot(alpha_h))

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    assert xp.abs(numerical - exact) < 1e-3


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize(
    "bcs",
    [
        (("dirichlet", "free"), ("free", "free"), ("free", "free")),
        (("free", "dirichlet"), ("free", "free"), ("free", "free")),
        (("dirichlet", "dirichlet"), ("free", "free"), ("free", "free")),
    ],
)
def test_scalar_unit_cube_nonconstant(num_elements, degree, bcs):
    """ScalarBoundaryMass: nonconstant alpha on the unit cube."""
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    if bcs[0] == ("dirichlet", "free"):
        alpha = lambda e1, e2, e3: e1 + 0 * e2 + 0 * e3
        exact = 3.0
    elif bcs[0] == ("free", "dirichlet"):
        alpha = lambda e1, e2, e3: 1.0 - e1 + 0 * e2 + 0 * e3
        exact = 3.0
    else:
        alpha = lambda e1, e2, e3: e1 * (1.0 - e1) + 0 * e2 + 0 * e3
        exact = 2.0 / 3.0

    alpha_h = L2Projector("H1", mass_ops)(alpha, apply_bc=True)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    numerical = _sum_coeffs(comm, bnd_ops.scalar().dot(alpha_h))

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    assert xp.abs(numerical - exact) < 2e-2


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
def test_scalar_cuboid_nontrivial(num_elements, degree, bcs):
    """ScalarBoundaryMass: alpha = eta1 + eta2 + eta3 on [-1,1] x [-1,3] x [0,3]."""
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    domain = domains.Cuboid(l1=-1.0, r1=1.0, l2=-1.0, r2=3.0, l3=0.0, r3=3.0)
    mass_ops = WeightedMassOperators(derham, domain)

    alpha_h = L2Projector("H1", mass_ops)(lambda e1, e2, e3: e1 + e2 + e3)

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    numerical = _sum_coeffs(comm, bnd_ops.scalar().dot(alpha_h))

    logger.info(f"numerical = {numerical}, exact = 78.0, error = {xp.abs(numerical - 78.0)}")
    assert xp.abs(numerical - 78.0) < 1e-3


@pytest.mark.parametrize("num_elements", [[8, 9, 10]])
@pytest.mark.parametrize("degree", [[1, 2, 3]])
@pytest.mark.parametrize("bcs", [(("free", "free"), None, ("free", "free"))])
def test_scalar_hollow_cylinder(num_elements, degree, bcs):
    """ScalarBoundaryMass: alpha = exp(eta3) on a HollowCylinder."""
    import math
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    a1, a2, Lz = 0.2, 1.0, 4.0
    domain = domains.HollowCylinder(a1=a1, a2=a2, Lz=Lz)
    mass_ops = WeightedMassOperators(derham, domain)

    e = math.e
    exact = xp.pi * (2 * a1 * Lz * (e - 1) + 2 * a2 * Lz * (e - 1) + (a2**2 - a1**2) * (1 + e))

    alpha_h = L2Projector("H1", mass_ops)(lambda e1, e2, e3: xp.exp(e3))

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    numerical = _sum_coeffs(comm, bnd_ops.scalar().dot(alpha_h))

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    assert xp.abs(numerical - exact) < 1e-2


# ---------------------------------------------------------------------------
# TangentialBoundaryMass tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_elements", [[10, 10, 10]])
@pytest.mark.parametrize("degree", [[2, 2, 2]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
@pytest.mark.parametrize(
    "active_faces, u_idx, v_idx, exact",
    [
        ([True, False, False, False, False, False], 1, 2, 1.0),
        ([False, True, False, False, False, False], 2, 0, 1.0),
        ([False, False, True, False, False, False], 0, 1, 1.0),
        ([False, False, False, True, False, False], 1, 2, -1.0),
        ([False, False, False, False, True, False], 2, 0, -1.0),
        ([False, False, False, False, False, True], 0, 1, -1.0),
    ],
)
def test_tangential_unit_cube_per_face(num_elements, degree, bcs, active_faces, u_idx, v_idx, exact):
    """TangentialBoundaryMass: unit vector fields on the unit cube, one face at a time."""
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    P = L2Projector("Hcurl", mass_ops)
    u_h = P([lambda e1, e2, e3, i=i: xp.ones_like(e1) if i == u_idx else xp.zeros_like(e1) for i in range(3)])
    v_h = P([lambda e1, e2, e3, i=i: xp.ones_like(e1) if i == v_idx else xp.zeros_like(e1) for i in range(3)])

    bnd_ops = BoundaryIntegralOperators(mass_ops, active_faces=active_faces)
    numerical = bnd_ops.tangential().dot_inner(u_h, v_h)

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    assert xp.abs(numerical - exact) < 1e-1


@pytest.mark.parametrize("num_elements", [[10, 10, 10]])
@pytest.mark.parametrize("degree", [[2, 2, 2]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
@pytest.mark.parametrize(
    "active_faces, u_idx, v_idx, exact",
    [
        ([True, False, False, False, False, False], 1, 2, 12.0),
        ([False, True, False, False, False, False], 2, 0, 6.0),
        ([False, False, True, False, False, False], 0, 1, 8.0),
        ([False, False, False, True, False, False], 1, 2, -12.0),
        ([False, False, False, False, True, False], 2, 0, -6.0),
        ([False, False, False, False, False, True], 0, 1, -8.0),
    ],
)
def test_tangential_cuboid_nontrivial(num_elements, degree, bcs, active_faces, u_idx, v_idx, exact):
    """TangentialBoundaryMass: unit vector fields on [-1,1] x [-1,3] x [0,3]."""
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    domain = domains.Cuboid(l1=-1.0, r1=1.0, l2=-1.0, r2=3.0, l3=0.0, r3=3.0)
    mass_ops = WeightedMassOperators(derham, domain)

    def make_pulled(idx):
        phys_funs = [
            lambda x, y, z, i=i: xp.ones_like(x) if i == idx else xp.zeros_like(x)
            for i in range(3)
        ]
        def pulled(*etas):
            return domain.pull(phys_funs, *etas, kind="1")
        return [lambda *etas, p=pulled, c=c: p(*etas)[c] for c in range(3)]

    P = L2Projector("Hcurl", mass_ops)
    u_h = P(make_pulled(u_idx))
    v_h = P(make_pulled(v_idx))

    bnd_ops = BoundaryIntegralOperators(mass_ops, active_faces=active_faces)
    numerical = bnd_ops.tangential().dot_inner(u_h, v_h)

    logger.info(f"numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    assert xp.abs(numerical - exact) < 1


# ---------------------------------------------------------------------------
# NormalBoundaryMass tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_elements", [[20, 20, 20]])
@pytest.mark.parametrize("degree", [[2, 2, 2]])
def test_normal_linear_unit_cube(num_elements, degree):
    """
    NormalBoundaryMass: u = (-1 + 2*x) e_0, all faces active.

    On face 0 (x=0), outward normal n = -e_0:  (u.n) = -(-1) = +1, area = 1  -> +1
    On face 3 (x=1), outward normal n = +e_0:  (u.n) =  (+1) = +1, area = 1  -> +1
    Faces 1,2,4,5: u has no e_1 or e_2 component -> (u.n) = 0.

    Total: int_{dOmega} (u.n) dS = 2.
    """
    comm = MPI.COMM_WORLD
    bcs = (("free", "free"), ("free", "free"), ("free", "free"))
    derham = Derham(
        TensorProductGrid(num_elements=num_elements),
        DerhamOptions(degree=degree, bcs=bcs),
        comm=comm,
    )
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    P_vec = L2Projector("Hdiv", mass_ops)
    u_h = P_vec([
        lambda e1, e2, e3: -1.0 + 2.0 * e1,
        lambda e1, e2, e3: xp.zeros_like(e1),
        lambda e1, e2, e3: xp.zeros_like(e1),
    ])

    bnd_ops = BoundaryIntegralOperators(mass_ops)
    Su = bnd_ops.normal().dot(u_h)
    numerical = _sum_coeffs(comm, Su)

    exact = 2.0
    logger.info(f"numerical={numerical:.6f}, exact={exact}, error={xp.abs(numerical - exact):.2e}")
    assert xp.abs(numerical - exact) < 1e-1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from struphy import set_logging_level
    set_logging_level(logging.INFO)

    test_scalar_unit_cube_constant([8, 9, 10], [1, 2, 3], (("free", "free"), ("free", "free"), ("free", "free")))
    test_scalar_unit_cube_nonconstant([8, 9, 10], [1, 2, 3], (("dirichlet", "free"), ("free", "free"), ("free", "free")))
    test_scalar_cuboid_nontrivial([8, 9, 10], [1, 2, 3], (("free", "free"), ("free", "free"), ("free", "free")))
    test_scalar_hollow_cylinder([8, 9, 10], [1, 2, 3], (("free", "free"), None, ("free", "free")))

    test_tangential_unit_cube_per_face(
        [10, 10, 10], [2, 2, 2], (("free", "free"), ("free", "free"), ("free", "free")),
        [True, False, False, False, False, False], 1, 2, 1.0,
    )
    test_tangential_cuboid_nontrivial(
        [10, 10, 10], [1, 2, 3], (("free", "free"), ("free", "free"), ("free", "free")),
        [True, False, False, False, False, False], 1, 2, 12.0,
    )

    test_normal_linear_unit_cube([20, 20, 20], [2, 2, 2])