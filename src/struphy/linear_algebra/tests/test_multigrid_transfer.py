import numpy as np
import pytest
from feectools.ddm.mpi import mpi as MPI
from mpi4py import MPI as MPI4PY

from struphy.feec.mass import WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.feec.utilities import create_equal_random_arrays
from struphy.geometry.domains import Cuboid
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.multigrid.hierarchy import MultiGridHierarchy
from struphy.linear_algebra.multigrid.transfer import SplineProlongation, prolongation_matrix_1d
from struphy.topology.grids import TensorProductGrid

BCS = [
    (None, None, None),
    (("dirichlet", "dirichlet"), None, ("free", "dirichlet")),
]


def _hierarchy(num_elements, degree, bcs, max_levels=None):
    domain = Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=1.0, l3=0.0, r3=3.0)
    derham = Derham(
        TensorProductGrid(num_elements=num_elements),
        DerhamOptions(degree=degree, bcs=bcs),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    return MultiGridHierarchy(derham, max_levels=max_levels), domain


@pytest.mark.mpi_skip
@pytest.mark.parametrize("degree", [1, 2, 3, 4])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("basis", ["B", "M"])
def test_prolongation_matrix_1d(degree, periodic, basis):
    """The coarse basis is reproduced exactly by the prolongated coefficients; partition of unity is kept."""
    from feectools.fem.splines import SplineSpace

    def space(n):
        return SplineSpace(degree, grid=np.linspace(0.0, 1.0, n + 1), periodic=periodic, basis=basis)

    coarse, fine = space(8), space(16)
    P = prolongation_matrix_1d(coarse, fine)
    assert P.shape == (fine.nbasis, coarse.nbasis)

    if basis == "B":
        # partition of unity: the constant function has coefficients 1 on both grids
        assert np.allclose(P @ np.ones(coarse.nbasis), 1.0)

    # each fine row couples to at most ceil((p+2)/2) coarse functions
    assert np.max(np.count_nonzero(P, axis=1)) <= (degree + 3) // 2


@pytest.mark.parametrize("num_elements, degree", [((16, 8, 8), (3, 2, 1)), ((8, 16, 1), (2, 3, 1))])
@pytest.mark.parametrize("bcs", BCS)
def test_hierarchy(num_elements, degree, bcs):
    """Coarse levels have aligned decompositions and at least degree+1 cells per coarsened direction."""
    h, _ = _hierarchy(num_elements, degree, bcs)
    assert h.n_levels >= 2
    assert len(h.factors) == h.n_levels - 1
    for l, f in enumerate(h.factors):
        fine, coarse = h[l], h[l + 1]
        for axis in range(3):
            assert fine.num_elements[axis] == f[axis] * coarse.num_elements[axis]
            assert fine.domain_decomposition.starts[axis] == f[axis] * coarse.domain_decomposition.starts[axis]
            if f[axis] == 2:
                assert coarse.num_elements[axis] >= degree[axis] + 1
        assert coarse.options is fine.options
    # the coarsest level cannot be coarsened further
    assert h._coarsening_factors(h[-1]) == (1, 1, 1)


@pytest.mark.parametrize("bcs", BCS)
@pytest.mark.parametrize("space_id", ["H1", "Hcurl", "Hdiv", "L2", "H1vec"])
def test_transfer(bcs, space_id):
    r"""Restriction is the transpose of the prolongation, and P is the exact embedding: R M_h P = M_H."""
    h, domain = _hierarchy((16, 8, 8), (3, 2, 1), bcs, max_levels=3)
    comm = MPI4PY.COMM_WORLD
    form = h[0].space_to_form[space_id]

    for l in range(h.n_levels - 1):
        P = SplineProlongation(h[l + 1], h[l], space_id)
        R = P.T
        assert R.domain is P.codomain and R.codomain is P.domain

        _, u = create_equal_random_arrays(h[l + 1].fem_spaces[form], seed=1)
        _, w = create_equal_random_arrays(h[l].fem_spaces[form], seed=2)
        assert np.isclose(R.dot(w).inner(u), w.inner(P.dot(u)), rtol=1e-12)

        Mh = getattr(WeightedMassOperators(h[l], domain), "M" + form)
        MH = getattr(WeightedMassOperators(h[l + 1], domain), "M" + form)
        a = R.dot(Mh.dot(P.dot(u)))
        b = MH.dot(u)
        err = comm.allreduce(np.max(np.abs((a - b).toarray())), op=MPI4PY.MAX)
        ref = comm.allreduce(np.max(np.abs(b.toarray())), op=MPI4PY.MAX)
        assert err < 1e-12 * ref


@pytest.mark.parametrize("bcs", BCS)
def test_prolongation_of_spline(bcs):
    """The prolongated coefficients represent the same function (point evaluation)."""
    h, _ = _hierarchy((8, 8, 4), (2, 3, 1), bcs, max_levels=2)
    P = SplineProlongation(h[1], h[0], "H1")
    _, u = create_equal_random_arrays(h[1].fem_spaces["0"], seed=4)
    u = h[1].boundary_ops["0"].dot(u)

    fH = h[1].create_spline_function("fH", "H1", coeffs=u)
    fh = h[0].create_spline_function("fh", "H1", coeffs=P.dot(u))
    e = np.linspace(0.0, 1.0, 7)
    assert np.allclose(fH(e, e, e), fh(e, e, e), atol=1e-12)
