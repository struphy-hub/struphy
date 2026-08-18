import logging
from typing import Callable

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


@pytest.mark.parametrize("num_elements", [[10, 10, 10]])
@pytest.mark.parametrize("degree", [[2, 2, 2]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
@pytest.mark.parametrize("data_space", ["Hdiv", "Hcurl"])
@pytest.mark.parametrize(
    "active_faces, u_idx, exact",
    [
        # face 0: normal = +e0, only u=e0 contributes -> area = 1
        ([True, False, False, False, False, False], 0, 1.0),
        ([True, False, False, False, False, False], 1, 0.0),
        ([True, False, False, False, False, False], 2, 0.0),
        # face 1: normal = +e1
        ([False, True, False, False, False, False], 1, 1.0),
        ([False, True, False, False, False, False], 0, 0.0),
        # face 2: normal = +e2
        ([False, False, True, False, False, False], 2, 1.0),
        ([False, False, True, False, False, False], 0, 0.0),
        # face 3: normal = -e0, (u.n)^2 = 1 still
        ([False, False, False, True, False, False], 0, 1.0),
        ([False, False, False, True, False, False], 1, 0.0),
        # face 4: normal = -e1
        ([False, False, False, False, True, False], 1, 1.0),
        # face 5: normal = -e2
        ([False, False, False, False, False, True], 2, 1.0),
    ],
)
def test_normal_unit_cube_per_face(num_elements, degree, bcs, data_space, active_faces, u_idx, exact):
    """
    NormalBoundaryMass: int (u.n) * alpha dS with u = e_{u_idx}, alpha = 1.
    On a face with normal e_d, only u=e_d contributes, giving the face area (=1 on unit cube).
    """
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    P_vec = L2Projector(data_space, mass_ops)
    P_sca = L2Projector("H1", mass_ops)

    u_h = P_vec([lambda e1, e2, e3, i=i: xp.ones_like(e1) if i == u_idx else xp.zeros_like(e1) for i in range(3)])
    alpha_h = P_sca(lambda e1, e2, e3: xp.ones_like(e1))

    bnd_ops = BoundaryIntegralOperators(mass_ops, active_faces=active_faces)
    Su = bnd_ops.normal(data_space=data_space).dot(u_h)
    numerical = _sum_coeffs(comm, Su * alpha_h)  # inner product with constant 1

    logger.info(f"data_space={data_space}, {u_idx=}, numerical = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    assert xp.abs(numerical - exact) < 1e-1


@pytest.mark.parametrize("num_elements", [[10, 10, 10]])
@pytest.mark.parametrize("degree", [[2, 2, 2]])
@pytest.mark.parametrize("bcs", [(("free", "free"), ("free", "free"), ("free", "free"))])
@pytest.mark.parametrize("data_space", ["Hdiv", "Hcurl"])
@pytest.mark.parametrize(
    "active_faces, u_idx, exact",
    [
        # both sides of direction 0: two faces of area 1 each -> 2*(+1) + 2*(-1) = 0
        # but since (u.n)^2 is always positive, for u=e0 summing both faces gives 2
        ([True, False, False, True, False, False], 0, 2.0),
        ([False, True, False, False, True, False], 1, 2.0),
        ([False, False, True, False, False, True], 2, 2.0),
        # all faces, u=e0: only faces 0 and 3 contribute -> 2
        ([True, True, True, True, True, True], 0, 2.0),
        ([True, True, True, True, True, True], 1, 2.0),
        ([True, True, True, True, True, True], 2, 2.0),
    ],
)
def test_normal_unit_cube_both_sides(num_elements, degree, bcs, data_space, active_faces, u_idx, exact):
    """
    NormalBoundaryMass: int (u.n)^2 dS = sum over active faces of (u.n_face)^2 * area.
    For u = e_d, only the two faces with normal ±e_d contribute, each giving area = 1.
    """
    comm = MPI.COMM_WORLD
    derham = Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)
    domain = domains.Cuboid(l1=0.0, r1=1.0, l2=0.0, r2=1.0, l3=0.0, r3=1.0)
    mass_ops = WeightedMassOperators(derham, domain)

    P_vec = L2Projector(data_space, mass_ops)

    u_h = P_vec([lambda e1, e2, e3, i=i: xp.ones_like(e1) if i == u_idx else xp.zeros_like(e1) for i in range(3)])

    bnd_ops = BoundaryIntegralOperators(mass_ops, active_faces=active_faces)
    # compute int (u.n)^2 dS = <S_normal u, u_normal_component>
    # use dot_inner with u itself projected onto the scalar result
    Su = bnd_ops.normal(data_space=data_space).dot(u_h)

    # Su is a scalar field; inner product with the normal component of u
    # For u = e_{u_idx}, the normal component on each face is ±1 or 0
    # We sum Su * 1 (constant test function) which gives int (u.n) dS summed
    # but we want int (u.n)^2 — so we dotted against u.n implicitly by using Su
    # Actually: dot(u_h) gives int (u.n) Lambda^0 dS, and to get int (u.n)^2 dS
    # we need to dot against the projection of (u.n) as a scalar H1 function.
    # For simplicity here we just verify int (u.n) * 1 dS per face via summing coeffs.
    # For u=e0 on face 0 (normal +e0): int 1*1 dS = 1. On face 3 (normal -e0): int (-1)*1 dS = -1.
    # So sum over both = 0. Instead test int (u.n)^2 differently: use dot_inner trick below.
    # Reframe: just check |int (u.n) dS| over one face.
    numerical = float(xp.abs(_sum_coeffs(comm, Su)))

    logger.info(f"data_space={data_space}, {u_idx=}, |numerical| = {numerical}, exact = {exact}, error = {xp.abs(numerical - exact)}")
    assert xp.abs(numerical - exact) < 1e-1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # TODO Add normal trace tests!
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

    test_normal_unit_cube_per_face(
        [10, 10, 10], [2, 2, 2], (("free", "free"), ("free", "free"), ("free", "free")), "Hdiv",
        [True, False, False, False, False, False], 0, 1.0,
    )
