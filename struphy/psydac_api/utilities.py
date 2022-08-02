from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix

from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.essential_bc import apply_essential_bc_stencil

from struphy.psydac_api import banded_to_stencil_kernels as bts

import numpy as np


def create_equal_random_arrays(V, seed=123, flattened=False):
    '''Creates two equal random arrays, where one array is a numpy array and the other one a distributed psydac array.

    Parameters
    ----------
        V : TensorFemSpace or ProductFemSpace
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
    '''

    np.random.seed(seed)

    arr = []

    if isinstance(V, TensorFemSpace):

        dims = V.vector_space.npts
        
        arr += [np.random.rand(*dims)]

        arr_psy = StencilVector(V.vector_space)

        s = arr_psy.starts
        e = arr_psy.ends

        arr_psy[s[0]:e[0] + 1, s[1]:e[1] + 1, s[2]:e[2] + 1] = arr[-1][s[0]:e[0] + 1, s[1]:e[1] + 1, s[2]:e[2] + 1]

        arr_psy.update_ghost_regions()

        if flattened:
            arr = arr[-1].flatten()

    elif isinstance(V, ProductFemSpace):

        arr_psy = BlockVector(V.vector_space)

        for d, block in enumerate(arr_psy.blocks):

            dims = V.spaces[d].vector_space.npts
        
            arr += [np.random.rand(*dims)]

            s = block.starts
            e = block.ends

            arr_psy[d][s[0]:e[0] + 1, s[1]:e[1] + 1, s[2]:e[2] + 1] = arr[-1][s[0]:e[0] + 1, s[1]:e[1] + 1, s[2]:e[2] + 1]

            arr_psy[d].update_ghost_regions()

            if flattened:
                arr[-1] = arr[-1].flatten()

        if flattened:
            arr = np.concatenate((arr[0], arr[1], arr[2]))

    return arr, arr_psy


def compare_arrays(arr_psy, arr, rank, atol=1e-14, verbose=False):
    '''Assert equality of distributed psydac array and corresponding fraction of cloned numpy array.
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
    '''

    if isinstance(arr_psy, StencilVector):

        s = arr_psy.starts
        e = arr_psy.ends

        tmp1 = arr_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]

        if arr.ndim == 3:
            tmp2 = arr[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]
        else:
            tmp2 = arr.reshape(arr_psy.space.npts[0], arr_psy.space.npts[1], arr_psy.space.npts[2])[
                s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]

        assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, BlockVector):

        if not (isinstance(arr, tuple) or isinstance(arr, list)):
            arrs = np.split(arr, [arr_psy.blocks[0].shape[0],
                            arr_psy.blocks[0].shape[0] + arr_psy.blocks[1].shape[0]])
        else:
            arrs = arr

        for vec_psy, vec in zip(arr_psy.blocks, arrs):
            s = vec_psy.starts
            e = vec_psy.ends

            tmp1 = vec_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]

            if vec.ndim == 3:
                tmp2 = vec[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]
            else:
                tmp2 = vec.reshape(vec_psy.space.npts[0], vec_psy.space.npts[1], vec_psy.space.npts[2])[
                    s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1]

            assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, StencilMatrix):

        s = arr_psy.codomain.starts
        e = arr_psy.codomain.ends
        p = arr_psy.pads
        tmp_arr = arr[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, :, :, :]
        tmp1 = arr_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, -p[0]: p[0] + 1, -p[1]: p[1] + 1, -p[2]: p[2] + 1]

        if tmp_arr.shape == tmp1.shape:
            tmp2 = tmp_arr
        else:
            tmp2 = np.zeros((e[0] + 1 - s[0], e[1] + 1 - s[1], e[2] +
                            1 - s[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
            bts.band_to_stencil_3d(tmp_arr, tmp2)

        assert np.allclose(tmp1, tmp2, atol=atol)

    elif isinstance(arr_psy, BlockMatrix):

        for row_psy, row in zip(arr_psy.blocks, arr):
            for mat_psy, mat in zip(row_psy, row):
                if mat_psy == None:
                    continue

                s = mat_psy.codomain.starts
                e = mat_psy.codomain.ends
                p = mat_psy.pads
                tmp_mat = mat[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, :, :, :]
                tmp1 = mat_psy[s[0]: e[0] + 1, s[1]: e[1] + 1, s[2]: e[2] + 1, -p[0]: p[0] + 1, -p[1]: p[1] + 1, -p[2]: p[2] + 1]

                if tmp_mat.shape == tmp1.shape:
                    tmp2 = tmp_mat
                else:
                    tmp2 = np.zeros((e[0] + 1 - s[0], e[1] + 1 - s[1], e[2] +
                                    1 - s[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
                    bts.band_to_stencil_3d(tmp_mat, tmp2)

                assert np.allclose(tmp1, tmp2, atol=atol)

    else:
        raise AssertionError('Wrong input type.')

    if verbose:
        print(f'Rank {rank}: Assertion for array comparison passed with atol={atol}.')
        
        
        
def apply_essential_bc_to_array(space_id, vector, bc):
    
    if space_id == 'H1':
        
        assert isinstance(vector, StencilVector)
        
        # eta1-direction
        if bc[0][0] == 'd': 
            apply_essential_bc_stencil(vector, axis=0, ext=-1, order=0)
        if bc[0][1] == 'd': 
            apply_essential_bc_stencil(vector, axis=0, ext=+1, order=0)
            
        # eta2-direction
        if bc[1][0] == 'd': 
            apply_essential_bc_stencil(vector, axis=1, ext=-1, order=0)
        if bc[1][1] == 'd': 
            apply_essential_bc_stencil(vector, axis=1, ext=+1, order=0)
            
        # eta3-direction
        if bc[2][0] == 'd': 
            apply_essential_bc_stencil(vector, axis=2, ext=-1, order=0)
        if bc[2][1] == 'd': 
            apply_essential_bc_stencil(vector, axis=2, ext=+1, order=0)

    elif space_id == 'Hcurl':
        
        assert isinstance(vector, BlockVector)
        
        # eta1-direction
        if bc[0][0] == 'd': 
            apply_essential_bc_stencil(vector[1], axis=0, ext=-1, order=0)
            apply_essential_bc_stencil(vector[2], axis=0, ext=-1, order=0)
        if bc[0][1] == 'd': 
            apply_essential_bc_stencil(vector[1], axis=0, ext=+1, order=0)
            apply_essential_bc_stencil(vector[2], axis=0, ext=+1, order=0)
            
        # eta2-direction
        if bc[1][0] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=1, ext=-1, order=0)
            apply_essential_bc_stencil(vector[2], axis=1, ext=-1, order=0)
        if bc[1][1] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=1, ext=+1, order=0)
            apply_essential_bc_stencil(vector[2], axis=1, ext=+1, order=0)
            
        # eta3-direction
        if bc[2][0] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=2, ext=-1, order=0)
            apply_essential_bc_stencil(vector[1], axis=2, ext=-1, order=0)
        if bc[2][1] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=2, ext=+1, order=0)
            apply_essential_bc_stencil(vector[1], axis=2, ext=+1, order=0)

    elif space_id == 'Hdiv':
        
        assert isinstance(vector, BlockVector)
        
        # eta1-direction
        if bc[0][0] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=0, ext=-1, order=0)
        if bc[0][1] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=0, ext=+1, order=0)
            
        # eta2-direction
        if bc[1][0] == 'd': 
            apply_essential_bc_stencil(vector[1], axis=1, ext=-1, order=0)
        if bc[1][1] == 'd': 
            apply_essential_bc_stencil(vector[1], axis=1, ext=+1, order=0)
            
        # eta3-direction
        if bc[2][0] == 'd': 
            apply_essential_bc_stencil(vector[2], axis=2, ext=-1, order=0)
        if bc[2][1] == 'd': 
            apply_essential_bc_stencil(vector[2], axis=2, ext=+1, order=0)

    elif space_id == 'H1vec':
        
        assert isinstance(vector, BlockVector)
        
        # eta1-direction
        if bc[0][0] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=0, ext=-1, order=0)
        if bc[0][1] == 'd': 
            apply_essential_bc_stencil(vector[0], axis=0, ext=+1, order=0)
            
        # eta2-direction
        if bc[1][0] == 'd': 
            apply_essential_bc_stencil(vector[1], axis=1, ext=-1, order=0)
        if bc[1][1] == 'd': 
            apply_essential_bc_stencil(vector[1], axis=1, ext=+1, order=0)
            
        # eta3-direction
        if bc[2][0] == 'd': 
            apply_essential_bc_stencil(vector[2], axis=2, ext=-1, order=0)
        if bc[2][1] == 'd': 
            apply_essential_bc_stencil(vector[2], axis=2, ext=+1, order=0)
