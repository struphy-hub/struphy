import json

import numpy as np
import pytest
from feectools.ddm.mpi import mpi as MPI
from mpi4py import MPI as MPI4PY

from struphy.feec.mass import WeightedMassOperator, WeightedMassOperators
from struphy.feec.psydac_derham import Derham
from struphy.feec.utilities import create_equal_random_arrays
from struphy.geometry.domains import Cuboid
from struphy.io.options import DerhamOptions
from struphy.linear_algebra.multigrid.coarsen import OperatorCoarsener
from struphy.linear_algebra.multigrid.hierarchy import MultiGridHierarchy
from struphy.linear_algebra.multigrid.transfer import SplineProlongation
from struphy.topology.grids import TensorProductGrid

BCS = [
    (None, None, None),
    (("dirichlet", "dirichlet"), None, ("free", "dirichlet")),
]


def _derham(num_elements, degree, bcs):
    domain = Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=1.0, l3=0.0, r3=3.0)
    derham = Derham(
        TensorProductGrid(num_elements=num_elements),
        DerhamOptions(degree=degree, bcs=bcs),
        comm=MPI.COMM_WORLD,
        domain=domain,
    )
    return derham, domain


def _max_diff(a, b):
    comm = MPI4PY.COMM_WORLD
    err = comm.allreduce(np.max(np.abs((a - b).toarray())), op=MPI4PY.MAX)
    ref = comm.allreduce(np.max(np.abs(b.toarray())), op=MPI4PY.MAX)
    return err / ref


@pytest.mark.parametrize("bcs", BCS)
def test_mass_to_dict(bcs):
    derham, domain = _derham((8, 6, 4), (2, 2, 1), bcs)
    mass_ops = WeightedMassOperators(derham, domain)
    _, u = create_equal_random_arrays(derham.fem_spaces["1"], seed=1)

    # predefined operator with string weights: JSON serializable
    M1 = mass_ops.M1
    dct = M1.to_dict()
    assert dct["type"] == "WeightedMassOperator"
    assert dct["params"]["weights"] == ["Ginv", "sqrt_g"]
    json.dumps(dct)
    assert _max_diff(WeightedMassOperator.from_dict(dct, mass_ops).dot(u), M1.dot(u)) < 1e-14

    # callable weights and transposes
    Mc = mass_ops.create_weighted_mass(
        "Hcurl", "Hdiv", weights=("Ginv", lambda e1, e2, e3: 1.0 + e1 * e2), name="Mc", assemble=True
    )
    McT = Mc.T
    assert McT.to_dict()["params"]["is_transpose"]
    _, w = create_equal_random_arrays(derham.fem_spaces["2"], seed=2)
    assert _max_diff(WeightedMassOperator.from_dict(McT.to_dict(), mass_ops).dot(w), McT.dot(w)) < 1e-14

    # modified data cannot be re-created
    M = mass_ops.create_weighted_mass("H1", "H1", weights=("sqrt_g",), assemble=True)
    assert M.is_reconstructible
    M *= 2.0
    assert not M.is_reconstructible
    with pytest.raises(ValueError):
        M.to_dict()


def test_basis_projection_to_dict():
    from struphy.feec.basis_projection_ops import BasisProjectionOperator

    derham, domain = _derham((8, 6, 4), (2, 2, 1), BCS[0])
    fun = [[lambda e1, e2, e3: 1.0 + e1 * e3]]
    K = BasisProjectionOperator(
        derham.projectors["L2"],
        derham.fem_spaces["H1"],
        fun,
        V_extraction_op=derham.extraction_ops["H1"],
        V_boundary_op=derham.boundary_ops["H1"],
    )
    for op, V in [(K, "0"), (K.T, "3")]:
        _, u = create_equal_random_arrays(derham.fem_spaces[V], seed=3)
        op2 = BasisProjectionOperator.from_dict(op.to_dict(), derham)
        assert _max_diff(op2.dot(u), op.dot(u)) < 1e-14


@pytest.mark.parametrize("bcs", BCS)
@pytest.mark.parametrize("degree", [(2, 3, 1), (3, 1, 2)])
def test_coarsen_poisson(bcs, degree):
    r"""Re-discretization of :math:`\sigma M_0 + G^\top M_1 G` equals the Galerkin product :math:`R A P` (Cuboid, exact quadrature)."""
    derham, domain = _derham((8, 8, 4), degree, bcs)
    h = MultiGridHierarchy(derham, max_levels=2)
    mass_ops = WeightedMassOperators(derham, domain)
    A = 0.7 * mass_ops.M0 + derham.grad.T @ mass_ops.M1 @ derham.grad

    C = OperatorCoarsener(h[0], h[1], domain)
    Ac = C(A)
    assert Ac.domain is h[1].coeff_spaces["0"] and Ac.codomain is h[1].coeff_spaces["0"]

    # same as building it directly on the coarse level
    mass_c = WeightedMassOperators(h[1], domain)
    Ad = 0.7 * mass_c.M0 + h[1].grad.T @ mass_c.M1 @ h[1].grad
    _, u = create_equal_random_arrays(h[1].fem_spaces["0"], seed=5)
    assert _max_diff(Ac.dot(u), Ad.dot(u)) < 1e-13

    # Galerkin property
    P = SplineProlongation(h[1], h[0], "H1")
    assert _max_diff(P.T.dot(A.dot(P.dot(u))), Ac.dot(u)) < 1e-12

    # leaves are cached: a new scalar re-assembles nothing
    A2 = 2.0 * mass_ops.M0 + derham.grad.T @ mass_ops.M1 @ derham.grad
    Ac2 = C(A2)
    leaves = lambda op: [a for a in op.addends]
    assert leaves(Ac2)[0].operator is leaves(Ac)[0].operator


@pytest.mark.parametrize("bcs", BCS)
def test_coarsen_curl_curl(bcs):
    r"""Re-discretization of :math:`C^\top M_2 C + M_1` equals the Galerkin product on 1-forms."""
    derham, domain = _derham((8, 8, 4), (2, 2, 1), bcs)
    h = MultiGridHierarchy(derham, max_levels=2)
    mass_ops = WeightedMassOperators(derham, domain)
    A = derham.curl.T @ mass_ops.M2 @ derham.curl + mass_ops.M1

    Ac = OperatorCoarsener(h[0], h[1], domain)(A)
    P = SplineProlongation(h[1], h[0], "Hcurl")
    _, u = create_equal_random_arrays(h[1].fem_spaces["1"], seed=6)
    assert _max_diff(P.T.dot(A.dot(P.dot(u))), Ac.dot(u)) < 1e-12


def test_coarsen_unknown_leaf():
    from feectools.linalg.stencil import StencilMatrix

    derham, domain = _derham((8, 8, 4), (2, 2, 1), BCS[0])
    h = MultiGridHierarchy(derham, max_levels=2)
    S = StencilMatrix(derham.coeff_spaces["0"], derham.coeff_spaces["0"])
    with pytest.raises(NotImplementedError):
        OperatorCoarsener(h[0], h[1], domain)(S)
