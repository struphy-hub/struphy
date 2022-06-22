from psydac.linalg.stencil import StencilVector, StencilMatrix
from psydac.linalg.block import BlockVector, BlockMatrix

from struphy.psydac_api import banded_to_stencil_kernels as bts

import numpy as np


def compare_arrays(arr_psy, arr, rank, atol=1e-14):
    '''Assert equality of distributed psydac array and corresponding fraction of cloned numpy array.
    Arrays can be block-structured as nested lists/tuples.

    Parameters
    ----------
        arr_psy : psydac object
            Stencil/Block Vector/Matrix.

        arr : array
            Numpy array with arr.size = arr_psy.toarray().size and same tuple/list structure as arr_psy.

        rank : int
            Rank of mpi process

        atol : float
            Absolute tolerance used in numpy.allclose()
    '''

    if isinstance(arr_psy, StencilVector):

        s = arr_psy.starts
        e = arr_psy.ends
        tmp1 = arr_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]
        tmp2 = arr[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]
        assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, BlockVector):

        for vec_psy, vec in zip(arr_psy, arr):
            s = vec_psy.starts
            e = vec_psy.ends
            tmp1 = vec_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]
            tmp2 = vec[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]
            assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, StencilMatrix):

        s = arr_psy.codomain.starts
        e = arr_psy.codomain.ends
        p = arr_psy.pads
        tmp1 = arr_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, -p[0]: p[0], -p[1]: p[1], -p[2]: p[2]]
        tmp2 = np.zeros((e[0] + 1 - s[0], e[1] + 1 - s[1], e[2] + 1 - s[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
        bts.band_to_stencil_3d(
            arr[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, :, :, :], tmp2)
        assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, BlockMatrix):

        for row_psy, row in zip(arr_psy, arr):
            for mat_psy, mat in zip(row_psy, row):
                s = mat_psy.codomain.starts
                e = mat_psy.codomain.ends
                p = mat_psy.pads
                tmp1 = mat_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, -p[0]: p[0], -p[1]: p[1], -p[2]: p[2]]
                tmp2 = np.zeros((e[0] + 1 - s[0], e[1] + 1 - s[1], e[2] + 1 - s[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
                bts.band_to_stencil_3d(
                    mat[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, :, :, :], tmp2)
                assert np.allclose(tmp1, tmp2, atol=atol)

    else:
        raise AssertionError('Wrong input type.')

    print(f'Rank {rank}: Assertion for array comparison passed with atol={atol}.')

