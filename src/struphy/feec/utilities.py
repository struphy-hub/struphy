from psydac.api.essential_bc import apply_essential_bc_stencil
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import VectorFemSpace
from psydac.linalg.block import BlockLinearOperator, BlockVector
from psydac.linalg.stencil import StencilMatrix, StencilVector
from psydac.linalg.basic import Vector

import struphy.feec.utilities_kernels as kernels
from struphy.feec import banded_to_stencil_kernels as bts
from struphy.polar.basic import PolarVector
from struphy.utils.arrays import xp as np


class RotationMatrix:
    """For a given vector-valued function a(e1, e2, e3), creates the callable matrix R(e1, e2, e3)
    that represents the local rotation Rv = a x v at (e1, e2, e3) for any vector v in R^3.

    When called, R(e1, e2, e3) is a five-dimensional array, with the 3x3 matrix in the last two indices.

    Parameters
    ----------
    *vec_fun : list
        Three callables that represent the vector-valued function a.
    """

    def __init__(self, *vec_fun):
        assert len(vec_fun) == 3
        assert all([callable(fun) for fun in vec_fun])

        self._cross_mask = [
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0],
        ]

        self._funs = [
            [lambda e1, e2, e3: 0 * e1, vec_fun[2], vec_fun[1]],
            [vec_fun[2], lambda e1, e2, e3: 0 * e2, vec_fun[0]],
            [vec_fun[1], vec_fun[0], lambda e1, e2, e3: 0 * e3],
        ]

    def __call__(self, e1, e2, e3):
        # array from 2d list gives 3x3 array is in the first two indices
        tmp = np.array(
            [
                [self._cross_mask[m][n] * fun(e1, e2, e3) for n, fun in enumerate(row)]
                for m, row in enumerate(self._funs)
            ]
        )

        # numpy operates on the last two indices with @
        return np.transpose(tmp, axes=(2, 3, 4, 0, 1))


def create_equal_random_arrays(V, seed=123, flattened=False):
    """Creates two equal random arrays, where one array is a numpy array and the other one a distributed psydac array.

    Parameters
    ----------
        V : TensorFemSpace or VectorFemSpace
            The FEM space to which the arrays belong to.

        seed : int
            The seed used in the radom number generator.

        flattened : bool
            Whether to return flattened arrays or 3d arrays.

    Returns
    -------
        arr : list or array (if flattened)
            The 3d numpy arrays for each component of the space.

        arr_psy : StencilVector of BlockVector
            The distributed psydac array.
    """

    assert isinstance(V, (TensorFemSpace, VectorFemSpace))

    np.random.seed(seed)

    arr = []

    if hasattr(V.symbolic_space, "name"):
        V_name = V.symbolic_space.name
    elif isinstance(V.symbolic_space, str):
        V_name = V.symbolic_space
    else:
        V_name = "H1vec"

    if V_name in {"H1", "L2"}:
        arr_psy = StencilVector(V.coeff_space)

        dims = V.coeff_space.npts

        arr += [np.random.rand(*dims)]

        s = arr_psy.starts
        e = arr_psy.ends

        arr_psy[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1] = arr[-1][
            s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1
        ]

        if flattened:
            arr = arr[-1].flatten()

    else:
        arr_psy = BlockVector(V.coeff_space)

        for d, block in enumerate(arr_psy.blocks):
            dims = V.spaces[d].coeff_space.npts

            arr += [np.random.rand(*dims)]

            s = block.starts
            e = block.ends

            arr_psy[d][s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1] = arr[-1][
                s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1
            ]

        if flattened:
            arr = np.concatenate(
                (
                    arr[0].flatten(),
                    arr[1].flatten(),
                    arr[2].flatten(),
                )
            )

    arr_psy.update_ghost_regions()

    return arr, arr_psy


def compare_arrays(arr_psy, arr, rank, atol=1e-14, verbose=False):
    """Assert equality of distributed psydac array and corresponding fraction of cloned numpy array.
    Arrays can be block-structured as nested lists/tuples.

    Parameters
    ----------
        arr_psy : psydac object
            Stencil/Block Vector/Matrix.

        arr : array
            Numpy array with same tuple/list structure as arr_psy. If arr is a matrix it can be stencil or band format.

        rank : int
            Rank of mpi process

        atol : float
            Absolute tolerance used in numpy.allclose()
    """

    if isinstance(arr_psy, StencilVector):
        s = arr_psy.starts
        e = arr_psy.ends

        tmp1 = arr_psy[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1]

        if arr.ndim == 3:
            tmp2 = arr[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1]
        else:
            tmp2 = arr.reshape(
                arr_psy.space.npts[0],
                arr_psy.space.npts[1],
                arr_psy.space.npts[2],
            )[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1]

        assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, BlockVector):
        if not (isinstance(arr, tuple) or isinstance(arr, list)):
            arrs = np.split(
                arr,
                [
                    arr_psy.blocks[0].shape[0],
                    arr_psy.blocks[0].shape[0] + arr_psy.blocks[1].shape[0],
                ],
            )
        else:
            arrs = arr

        for vec_psy, vec in zip(arr_psy.blocks, arrs):
            s = vec_psy.starts
            e = vec_psy.ends

            tmp1 = vec_psy[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1]

            if vec.ndim == 3:
                tmp2 = vec[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1]
            else:
                tmp2 = vec.reshape(vec_psy.space.npts[0], vec_psy.space.npts[1], vec_psy.space.npts[2])[
                    s[0] : e[0] + 1,
                    s[1] : e[1] + 1,
                    s[2] : e[2] + 1,
                ]

            assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, StencilMatrix):
        s = arr_psy.codomain.starts
        e = arr_psy.codomain.ends
        p = arr_psy.pads
        tmp_arr = arr[s[0] : e[0] + 1, s[1] : e[1] + 1, s[2] : e[2] + 1, :, :, :]
        tmp1 = arr_psy[
            s[0] : e[0] + 1,
            s[1] : e[1] + 1,
            s[2] : e[2] + 1,
            -p[0] : p[0] + 1,
            -p[1] : p[1] + 1,
            -p[2] : p[2] + 1,
        ]

        if tmp_arr.shape == tmp1.shape:
            tmp2 = tmp_arr
        else:
            tmp2 = np.zeros(
                (
                    e[0] + 1 - s[0],
                    e[1] + 1 - s[1],
                    e[2] + 1 - s[2],
                    2 * p[0] + 1,
                    2 * p[1] + 1,
                    2 * p[2] + 1,
                ),
                dtype=float,
            )
            bts.band_to_stencil_3d(tmp_arr, tmp2)

        assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, BlockLinearOperator):
        for row_psy, row in zip(arr_psy.blocks, arr):
            for mat_psy, mat in zip(row_psy, row):
                if mat_psy == None:
                    continue

                s = mat_psy.codomain.starts
                e = mat_psy.codomain.ends
                p = mat_psy.pads
                tmp_mat = mat[
                    s[0] : e[0] + 1,
                    s[1] : e[1] + 1,
                    s[2] : e[2] + 1,
                    :,
                    :,
                    :,
                ]
                tmp1 = mat_psy[
                    s[0] : e[0] + 1,
                    s[1] : e[1] + 1,
                    s[2] : e[2] + 1,
                    -p[0] : p[0] + 1,
                    -p[1] : p[1] + 1,
                    -p[2] : p[2] + 1,
                ]

                if tmp_mat.shape == tmp1.shape:
                    tmp2 = tmp_mat
                else:
                    tmp2 = np.zeros(
                        (
                            e[0] + 1 - s[0],
                            e[1] + 1 - s[1],
                            e[2] + 1 - s[2],
                            2 * p[0] + 1,
                            2 * p[1] + 1,
                            2 * p[2] + 1,
                        ),
                        dtype=float,
                    )
                    bts.band_to_stencil_3d(tmp_mat, tmp2)

                assert np.allclose(tmp1, tmp2, atol=atol)

    else:
        raise AssertionError("Wrong input type.")

    if verbose:
        print(
            f"Rank {rank}: Assertion for array comparison passed with atol={atol}.",
        )


def apply_essential_bc_to_array(space_id: str, vector: Vector, bc: tuple):
    """
    Sets entries corresponding to boundary B-splines to zero.

    Parameters
    ----------
        space_id : str
            The name of the continuous functions space the given vector belongs to (H1, Hcurl, Hdiv, L2 or H1vec).

        vector : Vector
            The vector whose boundary values shall be set to zero.

        bc : tuple[tuple[bool]]
            Whether to apply homogeneous Dirichlet boundary conditions (at left or right boundary in each direction).
    """

    assert isinstance(vector, (StencilVector, BlockVector, PolarVector))
    assert isinstance(bc, tuple)
    assert len(bc) == 3

    if isinstance(vector, PolarVector):
        vec_tp = vector.tp
    else:
        vec_tp = vector

    if space_id == "H1":
        assert isinstance(vec_tp, StencilVector)

        # eta1-direction
        if bc[0][0]:
            apply_essential_bc_stencil(vec_tp, axis=0, ext=-1, order=0)
        if bc[0][1]:
            apply_essential_bc_stencil(vec_tp, axis=0, ext=+1, order=0)

        # eta2-direction
        if bc[1][0]:
            apply_essential_bc_stencil(vec_tp, axis=1, ext=-1, order=0)
        if bc[1][1]:
            apply_essential_bc_stencil(vec_tp, axis=1, ext=+1, order=0)

        # eta3-direction
        if bc[2][0]:
            apply_essential_bc_stencil(vec_tp, axis=2, ext=-1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[0][:, 0] = 0.0

        if bc[2][1]:
            apply_essential_bc_stencil(vec_tp, axis=2, ext=+1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[0][:, -1] = 0.0

    elif space_id == "Hcurl":
        assert isinstance(vec_tp, BlockVector)

        # eta1-direction
        if bc[0][0]:
            apply_essential_bc_stencil(vec_tp[1], axis=0, ext=-1, order=0)
            apply_essential_bc_stencil(vec_tp[2], axis=0, ext=-1, order=0)
        if bc[0][1]:
            apply_essential_bc_stencil(vec_tp[1], axis=0, ext=+1, order=0)
            apply_essential_bc_stencil(vec_tp[2], axis=0, ext=+1, order=0)

        # eta2-direction
        if bc[1][0]:
            apply_essential_bc_stencil(vec_tp[0], axis=1, ext=-1, order=0)
            apply_essential_bc_stencil(vec_tp[2], axis=1, ext=-1, order=0)
        if bc[1][1]:
            apply_essential_bc_stencil(vec_tp[0], axis=1, ext=+1, order=0)
            apply_essential_bc_stencil(vec_tp[2], axis=1, ext=+1, order=0)

        # eta3-direction
        if bc[2][0]:
            apply_essential_bc_stencil(vec_tp[0], axis=2, ext=-1, order=0)
            apply_essential_bc_stencil(vec_tp[1], axis=2, ext=-1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[0][:, 0] = 0.0
                vector.pol[1][:, 0] = 0.0

        if bc[2][1]:
            apply_essential_bc_stencil(vec_tp[0], axis=2, ext=+1, order=0)
            apply_essential_bc_stencil(vec_tp[1], axis=2, ext=+1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[0][:, -1] = 0.0
                vector.pol[1][:, -1] = 0.0

    elif space_id == "Hdiv":
        assert isinstance(vec_tp, BlockVector)

        # eta1-direction
        if bc[0][0]:
            apply_essential_bc_stencil(vec_tp[0], axis=0, ext=-1, order=0)
        if bc[0][1]:
            apply_essential_bc_stencil(vec_tp[0], axis=0, ext=+1, order=0)

        # eta2-direction
        if bc[1][0]:
            apply_essential_bc_stencil(vec_tp[1], axis=1, ext=-1, order=0)
        if bc[1][1]:
            apply_essential_bc_stencil(vec_tp[1], axis=1, ext=+1, order=0)

        # eta3-direction
        if bc[2][0]:
            apply_essential_bc_stencil(vec_tp[2], axis=2, ext=-1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[2][:, 0] = 0.0

        if bc[2][1]:
            apply_essential_bc_stencil(vec_tp[2], axis=2, ext=+1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[2][:, -1] = 0.0

    elif space_id == "H1vec":
        assert isinstance(vec_tp, BlockVector)

        # eta1-direction
        if bc[0][0]:
            apply_essential_bc_stencil(vec_tp[0], axis=0, ext=-1, order=0)
        if bc[0][1]:
            apply_essential_bc_stencil(vec_tp[0], axis=0, ext=+1, order=0)

        # eta2-direction
        if bc[1][0]:
            apply_essential_bc_stencil(vec_tp[1], axis=1, ext=-1, order=0)
        if bc[1][1]:
            apply_essential_bc_stencil(vec_tp[1], axis=1, ext=+1, order=0)

        # eta3-direction
        if bc[2][0]:
            apply_essential_bc_stencil(vec_tp[2], axis=2, ext=-1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[2][:, 0] = 0.0

        if bc[2][1]:
            apply_essential_bc_stencil(vec_tp[2], axis=2, ext=+1, order=0)

            if isinstance(vector, PolarVector):
                vector.pol[2][:, -1] = 0.0


def create_weight_weightedmatrix_hybrid(b, weight_pre, derham, accum_density, domain):
    """Creates weights needed for asembling matrix of hybrid model with kinetic ions and massless electrons

    Parameters
    ----------
        b : Stencilvector
            finite element coefficients of magnetic fields.

        self._weight_pre : list of 3D arrays storing weights

        derham : derham class

        accum_density : StencilMatrix
            the density obtained by deposition of particles

        domain : domain class

    Returns
    -------
        self._weight_pre : list of 3D arrays storing weights

    """

    nqs = [
        quad_grid[nquad].num_quad_pts
        for quad_grid, nquad in zip(
            derham.get_quad_grids(derham.Vh_fem["0"]),
            derham.nquads,
        )
    ]

    for aa, wspace in enumerate(derham.Vh_fem["2"].spaces):
        # knot span indices of elements of local domain
        spans_out = [
            quad_grid[nquad].spans for quad_grid, nquad in zip(self.derham.get_quad_grids(wspace), derham.nquads)
        ]
        # global start spline index on process
        starts_out = [int(start) for start in wspace.coeff_space.starts]

        # Iniitialize hybrid linear operators
        # global quadrature points (flattened) and weights in format (local element, local weight)
        pts = [quad_grid[nquad].points for quad_grid, nquad in zip(self.derham.get_quad_grids(wspace), derham.nquads)]
        wts = [quad_grid[nquad].weights for quad_grid, nquad in zip(self.derham.get_quad_grids(wspace), derham.nquads)]

        p = wspace.degree

        # evaluated basis functions at quadrature points of the space
        basis_o = [
            quad_grid[nquad].basis for quad_grid, nquad in zip(self.derham.get_quad_grids(wspace), derham.nquads)
        ]

        pads_out = wspace.coeff_space.pads

        kernels.hybrid_curlA(
            *starts_out,
            *spans_out,
            p[0],
            p[1],
            p[2],
            nqs[0],
            nqs[1],
            nqs[2],
            *basis_o,
            weight_pre[aa],
            b[aa]._data,
        )
    # generate the weight for generating the matrix
    kernels.hybrid_weight(
        *pads_out,
        *pts,
        *spans_out,
        nqs[0],
        nqs[1],
        nqs[2],
        wts[0],
        wts[1],
        wts[2],
        weight_pre[0],
        weight_pre[1],
        weight_pre[2],
        accum_density._operators[0].matrix._data,
        *domain.args_map,
    )
