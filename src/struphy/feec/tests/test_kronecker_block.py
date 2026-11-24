import pytest


def _derham(num_elements, degree, bcs, comm=None):
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    return Derham(TensorProductGrid(num_elements=num_elements), DerhamOptions(degree=degree, bcs=bcs), comm=comm)


def _dense(op, V):
    """Dense global matrix of ``op`` (serial) by applying it to unit vectors; components stacked, C-ordered."""
    import numpy as np
    from feectools.linalg.block import BlockVector

    x = V.zeros()
    blocks = x.blocks if isinstance(x, BlockVector) else (x,)
    sizes = [int(np.prod(b.space.npts)) for b in blocks]
    cols = []
    for c, b in enumerate(blocks):
        for j in range(sizes[c]):
            for bb in blocks:
                bb._data[:] = 0.0
            idx = np.unravel_index(j, b.space.npts)
            b._data[tuple(p + i for p, i in zip(b.space.pads, idx))] = 1.0
            for bb in blocks:
                bb.ghost_regions_in_sync = False
            y = op.dot(x)
            cols.append(np.asarray(y.toarray()))
    return np.array(cols).T


@pytest.mark.mpi_skip
@pytest.mark.parametrize("derivative", ["grad", "curl", "div"])
@pytest.mark.parametrize("bcs", [(None, None, None), (("dirichlet", "dirichlet"), None, ("free", "dirichlet"))])
def test_derivative_operator(derivative, bcs):
    """The Kronecker derivatives equal the tensor-product derivatives of the de Rham sequence."""
    import numpy as np

    from struphy.feec.kronecker_block import derivative_operator

    derham = _derham([4, 5, 3], [2, 3, 1], bcs)
    D = derivative_operator(derham, derivative)
    space = {"grad": "0", "curl": "1", "div": "2"}[derivative]
    D_ex = _dense(getattr(derham, f"{derivative}_bcfree"), derham.coeff_spaces[space])
    assert np.allclose(D.toarray(), D_ex, rtol=0.0, atol=1e-14)

    # composition and transposition: div curl = 0, curl grad = 0
    if derivative != "grad":
        prev = derivative_operator(derham, "grad" if derivative == "curl" else "curl")
        assert np.abs((D @ prev).toarray()).max() < 1e-13
    assert np.allclose(D.T.toarray(), D_ex.T)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("spaces", [("Hdiv", "Hcurl"), ("Hcurl", "Hdiv"), ("H1", "Hdiv")])
@pytest.mark.parametrize("bcs", [(None, None, None), (("dirichlet", "dirichlet"), None, None)])
def test_basis_projection_operator_constant_weights(spaces, bcs):
    """For constant weights on the cube, the Kronecker basis projection operator is exact (incl. histopolation with
    split cells for even degree and essential boundary conditions)."""
    import numpy as np

    from struphy import domains
    from struphy.feec.basis_projection_ops import BasisProjectionOperators
    from struphy.feec.kronecker_block import basis_projection_operator

    V_id, W_id = spaces
    derham = _derham([4, 5, 3], [2, 3, 1], bcs)
    basis_ops = BasisProjectionOperators(derham, domains.Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=1.0))
    rng = np.random.default_rng(3)
    nrows = 1 if W_id == "H1" else 3
    ncols = 1 if V_id == "H1" else 3
    values = rng.standard_normal((nrows, ncols))
    values[0, 0] = 0.0  # a zero block
    fun = [[(lambda e1, e2, e3, a=values[m, n]: a + 0.0 * e1) for n in range(ncols)] for m in range(nrows)]
    op = basis_ops.create_basis_op(fun, V_id, W_id, assemble=True, name="test")

    T = basis_projection_operator(op, derham)
    T_ex = _dense(op, derham.coeff_spaces[derham.space_to_form[V_id]])
    assert np.abs(T.toarray() - T_ex).max() < 1e-12 * np.abs(T_ex).max()
    assert (0, 0) not in T.blocks


@pytest.mark.mpi_skip
def test_separable_weight():
    """The separable approximation is exact for products of 1d functions."""
    import numpy as np

    from struphy.feec.kronecker_block import separable_weight

    def fun(e1, e2, e3):
        return (1.0 + e1**2) * np.cos(e2) * (2.0 + e3)

    mean, shapes = separable_weight(fun)
    e = np.linspace(0.0, 1.0, 7)
    E = np.meshgrid(e, e, e, indexing="ij")
    approx = mean * shapes[0](e)[:, None, None] * shapes[1](e)[None, :, None] * shapes[2](e)[None, None, :]
    assert np.allclose(approx, fun(*E), rtol=1e-12)


@pytest.mark.parametrize("derivative", ["grad", "curl", "div"])
@pytest.mark.parametrize("sigma", [0.0, 0.7])
def test_general_stiffness_preconditioner(derivative, sigma):
    """With the Kronecker derivative as general operator, StiffnessPreconditioner gives the block Jacobi
    preconditioner of the named derivative: on the unit cube both are the exact inverse of the same approximation."""
    import cunumpy as xp
    from maybempi import MPI

    from struphy import domains
    from struphy.feec.kronecker_block import derivative_operator
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import StiffnessPreconditioner
    from struphy.feec.utilities import create_equal_random_arrays

    comm = MPI.COMM_WORLD
    derham = _derham([6, 5, 4], [2, 3, 2], (None, None, None), comm=comm)
    mass_ops = WeightedMassOperators(derham, domains.Cuboid())
    kw = dict(sigma=sigma, apply_bc=False, diagonal_scaling=False, kernel_correction=False, weights="unit")
    P = StiffnessPreconditioner(mass_ops, derivative, **kw)
    P_gen = StiffnessPreconditioner(mass_ops, derivative_operator(derham, derivative), **kw)
    assert P_gen.diagonal_scaling is False and P_gen.kernel_correction is None

    form = {"grad": "0", "curl": "1", "div": "2"}[derivative]
    x = create_equal_random_arrays(derham.fem_spaces[form], seed=1)[1]
    y, y_gen = P.matrix.dot(x), P_gen.matrix.dot(x)
    diff = (y - y_gen).toarray()
    assert xp.abs(diff).max() < 1e-12 * xp.abs(y.toarray()).max()
    if sigma > 0.0:
        z = P_gen.solver.dot(y_gen) - x
        assert xp.abs(z.toarray()).max() < 1e-10

    # a list of operators: L^T M L + L^T (2 M) L = 3 L^T M L
    mid = mass_ops.M1 if derivative == "grad" else (mass_ops.M2 if derivative == "curl" else mass_ops.M3)
    L = derivative_operator(derham, derivative)
    P_list = StiffnessPreconditioner(mass_ops, [L, L], mid=[mid, 2.0 * mid], **{**kw, "sigma": 3 * sigma})
    diff = (P_list.matrix.dot(x) - 3.0 * y).toarray()
    assert xp.abs(diff).max() < 1e-11 * xp.abs(y.toarray()).max()


def test_general_stiffness_preconditioner_options():
    """Options that the general operator does not support."""
    from maybempi import MPI

    from struphy import domains
    from struphy.feec.kronecker_block import derivative_operator
    from struphy.feec.mass import WeightedMassOperators
    from struphy.feec.preconditioner import StiffnessPreconditioner

    derham = _derham([6, 5, 4], [2, 3, 2], (None, None, None), comm=MPI.COMM_WORLD)
    mass_ops = WeightedMassOperators(derham, domains.Cuboid())
    G = derivative_operator(derham, "grad")
    with pytest.raises(NotImplementedError):
        StiffnessPreconditioner(mass_ops, G, sigma=1.0, stab_average="eta3")
    P = StiffnessPreconditioner(mass_ops, G, sigma=1.0)
    with pytest.raises(NotImplementedError):
        P.core_operator
