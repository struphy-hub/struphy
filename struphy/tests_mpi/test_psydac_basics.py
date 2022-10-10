import pytest

from mpi4py import MPI
import numpy as np
from time import sleep


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 8, 12]])
@pytest.mark.parametrize('p', [[1, 2, 3]])
@pytest.mark.parametrize('spl_kind', [[False, False, True]])
@pytest.mark.parametrize('mapping', [
    ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ])
def test_psydac_basics(Nel, p, spl_kind, mapping):
    '''Show attributes of basic psydac objects.'''

    from struphy.psydac_api.psydac_derham import Derham

    from psydac.fem.basic import FemField
    from psydac.linalg.stencil import StencilVector, StencilMatrix
    from psydac.linalg.block import BlockVector, BlockMatrix
    from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    DR = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print(
            f'\nNel={DR.Nel}, p={DR.p}, spl_kind={DR.spl_kind}, mpi_size={comm.size}.')
        print(f'\nDiscrete Derham DR set. All print commands are from rank 0.\n')
        print('This test contains the following screen outputs:')
        print('- DR attributes')
        print('- Derham space types')
        print('- DR.V0 attributes')
        print('- DR.V1 attributes')
        print('- DR.V0.vector_space (StencilVectorSpace) attributes')
        print('- DR.V1.vector_space (BlockVectorSpace) attributes')
        print(
            '- DR.V1.spaces[n].vector_space (StencilVectorSpace) attribute comparison')
        print('- DR.V0.spaces[0] (SplineSpace) attributes')
        print('- DR.V0.spaces[1] (SplineSpace) attributes')
        print('- DR.V0.spaces[2] (SplineSpace) attributes')
        print('- DR.V1.spaces[n].spaces (SplineSpaces) attribute comparison')
        print('- Attributes of FemField f0')
        print('- Attributes of FemField f1')
        print('- Attributes of StencilVector x0')
        print('- Attributes of BlockVector x1')
        print('- Data of Stencil Vector x0')
        print('- Attributes of StencilMatrix A0')
        print('- Attributes of BlockMatrix A1')
        print('- Data of StencilMatrix A0')
        print('- StencilVector and StencilMatrix local output before/after update_ghost_regions.')

        print('\n###### DR attributes ######')
        for k in dir(DR):
            if k[0] != '_' and 'assemble' not in k:
                print(k, getattr(DR, k))
        # for k, v in DR.__dict__.items():
        #     print(k, v)
        print('\n###### Derham space types ######')
        print(f'type(DR.V0): {type(DR.V0)}')
        print(f'type(DR.V1): {type(DR.V1)}')
        print(f'type(DR.V2): {type(DR.V2)}')
        print(f'type(DR.V3): {type(DR.V3)}\n')
        print('In what follows we look only at V0 (scalar space) and V1 (vector-valued space).\n')
        print('###### DR.V0 attributes ######')
        for k in dir(DR.V0):
            if k[0] != '_' and 'preprocess' not in k and 'reduce_' not in k and 'eval_' not in k and 'init_' not in k:
                print(k, getattr(DR.V0, k))
        print('\n###### DR.V1 attributes ######')
        for k in dir(DR.V1):
            if k[0] != '_' and k != 'comm' and 'eval_' not in k:
                print(k, getattr(DR.V1, k))
        print('\nThe .spaces attribute of V0 contains three 1d SplineSpace objects, whereas for V1 it contains TensorFemSpace objects (V0 is a TensorFemSpace).')
        print('\n###### DR.V0.vector_space (StencilVectorSpace) attributes (rank 0) ######')
        for k in dir(DR.V0.vector_space):
            if k[0] != '_' and 'reduce_' not in k:
                print(k, getattr(DR.V0.vector_space, k))
        print('\n###### DR.V1.vector_space (BlockVectorSpace) attributes (rank 0) ######')
        for k in dir(DR.V1.vector_space):
            if k[0] != '_' and 'reduce_' not in k:
                print(k, getattr(DR.V1.vector_space, k))
        print(
            '\n###### DR.V1.spaces[n].vector_space (StencilVectorSpace) attribute comparison (rank 0) ######')
        for n, space in enumerate(DR.V1.spaces):
            print(
                f'V1_{n}.vector_space.starts  in eta1: {space.vector_space.starts}')
            print(
                f'V1_{n}.vector_space.ends    in eta1: {space.vector_space.ends}')
            print(
                f'V1_{n}.vector_space.pads    in eta1: {space.vector_space.pads}')
            print(
                f'V1_{n}.vector_space.periods in eta1: {space.vector_space.periods}')
            print(
                f'V1_{n}.vector_space.npts    in eta1: {space.vector_space.npts}\n')
        print('The three components of V1 can have different starts and ends in some direction.')
        print(
            '\n###### DR.V0.spaces[0] (SplineSpace) attributes (rank 0) ######')
        for k in dir(DR.V0.spaces[0]):
            if k[0] != '_' and 'compute_' not in k and 'eval_' not in k and 'init_' not in k:
                print(k, getattr(DR.V0.spaces[0], k))
        print(
            '\n###### DR.V0.spaces[1] (SplineSpace) attributes (rank 0) ######')
        for k in dir(DR.V0.spaces[1]):
            if k[0] != '_' and 'compute_' not in k and 'eval_' not in k and 'init_' not in k:
                print(k, getattr(DR.V0.spaces[1], k))
        print(
            '\n###### DR.V0.spaces[2] (SplineSpace) attributes (rank 0) ######')
        for k in dir(DR.V0.spaces[2]):
            if k[0] != '_' and 'compute_' not in k and 'eval_' not in k and 'init_' not in k:
                print(k, getattr(DR.V0.spaces[2], k))
        print(
            '\n###### DR.V1.spaces[n].spaces (SplineSpaces) attribute comparison (rank 0) ######')
        for n, space in enumerate(DR.V1.spaces):
            print(f'V1_{n}.degree    in eta1: {space.spaces[0].degree}')
            print(f'V1_{n}.periodic  in eta1: {space.spaces[0].periodic}')
            print(f'V1_{n}.dirichlet in eta1: {space.spaces[0].dirichlet}')
            print(f'V1_{n}.basis     in eta1: {space.spaces[0].basis}')
            print(f'V1_{n}.nbasis    in eta1: {space.spaces[0].nbasis}\n')
            print(f'V1_{n}.degree    in eta2: {space.spaces[1].degree}')
            print(f'V1_{n}.periodic  in eta2: {space.spaces[1].periodic}')
            print(f'V1_{n}.dirichlet in eta2: {space.spaces[1].dirichlet}')
            print(f'V1_{n}.basis     in eta2: {space.spaces[1].basis}')
            print(f'V1_{n}.nbasis    in eta2: {space.spaces[1].nbasis}\n')
            print(f'V1_{n}.degree    in eta3: {space.spaces[2].degree}')
            print(f'V1_{n}.periodic  in eta3: {space.spaces[2].periodic}')
            print(f'V1_{n}.dirichlet in eta3: {space.spaces[2].dirichlet}')
            print(f'V1_{n}.basis     in eta3: {space.spaces[2].basis}')
            print(f'V1_{n}.nbasis    in eta3: {space.spaces[2].nbasis}\n')

    # FemFields (distributed)
    f0 = FemField(DR.V0)
    f1 = FemField(DR.V1)

    # only for M1 Mac users
    PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'

    # Stencil objects (distributed)
    x0 = StencilVector(DR.V0.vector_space)
    A0 = StencilMatrix(DR.V0.vector_space, DR.V0.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)

    x1 = BlockVector(DR.V1.vector_space)
    A1 = BlockMatrix(DR.V1.vector_space, DR.V1.vector_space)

    starts = DR.V0.vector_space.starts
    ends = DR.V0.vector_space.ends
    pads = DR.V0.vector_space.pads

    if rank == 0:
        print(
            f'Nel={DR.Nel}, p={DR.p}, spl_kind={DR.spl_kind}, mpi_size={comm.size}.')

        print('\n###### Attributes of FemField f0 (rank 0) ######')
        for k in dir(f0):
            if k[0] != '_' or k == '_data':
                if k == '_data':
                    v = f'array of shape {getattr(f0, k).shape}'
                else:
                    v = getattr(f0, k)
                print(k, v)
        print('\n###### Attributes of FemField f1 (rank 0) ######')
        for k in dir(f1):
            if k[0] != '_' or k == '_data':
                if k == '_data':
                    v = f'array of shape {getattr(f1, k).shape}'
                else:
                    v = getattr(f1, k)
                print(k, v)

        print('\nFemfiels have the attribute .coeffs which holds the corresponding Stencil- or BlockVector.')

        print('\n###### Attributes of StencilVector x0 (rank 0) ######')
        for k in dir(x0):
            if k[0] != '_' or k == '_data':
                if k == '_data':
                    v = f'array of shape {getattr(x0, k).shape}'
                else:
                    v = getattr(x0, k)
                print(k, v)
        print('\n###### Attributes of BlockVector x1 (rank 0) ######')
        for k in dir(x1):
            if k[0] != '_' or k == '_data':
                if k == '_data':
                    v = f'array of shape {getattr(x1, k).shape}'
                else:
                    v = getattr(x1, k)
                print(k, v)

        print('\nBlockVectors have the attribute "blocks", which is a tuple of StencilVectors. You can loop through a BlockVector like "for vec in x1: print(vec)"')

        print('\n###### Data of Stencil Vector x0 (rank 0) ######')
        print(f'type(x0)={type(x0)}')
        print(f'type(x0[:, :, :])={type(x0[:, :, :])}')
        print(f'type(x0[:])={type(x0[:])}')
        print(f'type(x0._data)={type(x0._data)}')
        print(f'type(x0.toarray())={type(x0.toarray())}')
        print(f'type(x0.toarray_local())={type(x0.toarray_local())}')
        print(f'x0.shape={x0.shape}')
        print(f'x0[:, :, :].shape={x0[:, :, :].shape}')
        print(f'x0[:].shape={x0[:].shape}')
        print(f'x0._data.shape={x0._data.shape}')
        print(f'x0.toarray().shape={x0.toarray().shape}')
        print(f'x0.toarray_local().shape={x0.toarray_local().shape}')

        print('\n###### Attributes of StencilMatrix A0 (rank 0) ######')
        for k in dir(A0):
            if (k[0] != '_' or k == '_data'):
                if k == '_data':
                    v = f'array of shape {getattr(A0, k).shape}'
                elif k == 'T':
                    v = 'transpose matrix'
                else:
                    v = getattr(A0, k)
                print(k, v)
        print('\n###### Attributes of BlockMatrix A1 (rank 0) ######')
        for k in dir(A1):
            if (k[0] != '_' or k == '_data'):
                if k == '_data':
                    v = f'array of shape {getattr(A1, k).shape}'
                elif k == 'T':
                    v = 'transpose matrix'
                else:
                    v = getattr(A1, k)
                print(k, v)

        print('\nBlockMatrices have the attribute "blocks", which at creation is a nested tuple of None, but can be filled with StencilMatrices.')

        print('\n###### Data of Stencil Matrix A0 (rank 0) ######')
        print(f'type(A0)={type(A0)}')
        print(f'type(A0[:, :, :, :, :, :])={type(A0[:, :, :, :, :, :])}')
        print(f'type(A0[:, :])={type(A0[:, :])}')
        print(f'type(A0._data)={type(A0._data)}')
        print(f'type(A0.toarray())={type(A0.toarray())}')
        print(f'type(A0.toarray_local()) does not exist.')
        print(f'A0.shape={A0.shape}')
        print(f'A0[:, :, :, :, :, :].shape={A0[:, :, :, :, :, :].shape}')
        print(f'A0[:, :].shape={A0[:, :].shape}')
        print(f'A0._data.shape={A0._data.shape}')
        print(f'A0.toarray().shape={A0.toarray().shape}')
        print(f'A0.toarray_local().shape does not exist.\n')

    # Assign values directly to Stencil Vector (no padding + global indices !!)
    # -------------------------------------------------------------------------
    x0[:] = 99 - rank*11
    y0 = x0.copy()
    x0[starts[0]: ends[0] + 1, starts[1]        : ends[1] + 1, starts[2]: ends[2] + 1] = rank

    # Assign values to _data attribute (=numpy array) (padding + local indices !!)
    # ----------------------------------------------------------------------------
    y0._data[pads[0]: -pads[0], pads[1]: -pads[1], pads[2]: -pads[2]] = rank

    assert np.allclose(x0[:], y0[:])

    comm.Barrier()
    sleep(.1*(rank + 1))
    print(
        f'Rank: {rank}, x0[starts[0], starts[1], :]={x0[starts[0], starts[1], :]}')
    comm.Barrier()

    if rank == 0:
        print('\n Update ghost regions...\n')
    x0.update_ghost_regions()
    comm.Barrier()

    sleep(.1*(rank + 1))
    print(
        f'Rank: {rank}, x0[starts[0], starts[1], :]={x0[starts[0], starts[1], :]}')
    comm.Barrier()

    # Assign values directly to Stencil Matrix
    # ----------------------------------------
    # rows:   no padding + global indices
    # colums: no padding + from -p to p, diagonal is at index 0 !!)
    A0[:, :] = 99 - rank*11
    B0 = A0.copy()
    for n in range(2*p[2] + 1):
        A0[starts[0]: ends[0] + 1, starts[1]: ends[1] + 1, starts[2]
            : ends[2] + 1, :, :, n - p[2]] = (n - p[2])*10**rank

    # Assign values to _data attribute (=numpy array)
    # -----------------------------------------------
    # rows:   padding + local indices
    # colums: padding + from -p to p, diagonal is at index 0 !!
    for n in range(2*p[2] + 1):
        B0._data[pads[0]: -pads[0], pads[1]: -pads[1], pads[2]: -
                 pads[2], :, :, pads[2] + n - p[2]] = (n - p[2])*10**rank

    assert np.allclose(A0[:, :], B0[:, :])

    comm.Barrier()
    sleep(.1*(rank + 1))
    print(
        f'Rank: {rank}, A0[starts[0], starts[1], :, 0, 0, :]=\n{A0[starts[0], starts[1], :, 0, 0, :]}')
    comm.Barrier()

    if rank == 0:
        print('\n Update ghost regions...\n')
    A0.update_ghost_regions()
    comm.Barrier()

    sleep(.1*(rank + 1))
    print(
        f'Rank: {rank}, A0[starts[0], starts[1], :, 0, 0, :]=\n{A0[starts[0], starts[1], :, 0, 0, :]}')
    comm.Barrier()


if __name__ == '__main__':
    test_psydac_basics([8, 8, 12], [2, 3, 4], [False, False, True], ['Cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}])
