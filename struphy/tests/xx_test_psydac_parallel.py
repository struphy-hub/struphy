def test_psydac_stencil_objects():

    from struphy.geometry.domain_3d import Domain

    from sympde.topology import Cube, Derham

    from psydac.api.discretization import discretize
    from psydac.fem.tensor import TensorFemSpace 
    from psydac.fem.vector import ProductFemSpace
    from psydac.linalg.stencil import StencilVectorSpace, StencilVector, StencilMatrix
    from psydac.linalg.block import BlockVectorSpace, BlockVector, BlockMatrix

    from mpi4py import MPI
    import numpy as np

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # Domain object
    map = 'cuboid'
    params_map = {'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}

    DOMAIN = Domain(map, params_map)

    # Psydac mapping
    Mapping_psydac = DOMAIN.Psydac_mapping('F', **params_map)
    
    # Psydac symbolic domain
    DOMAIN_PSYDAC_LOGICAL = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    DOMAIN_symb = Mapping_psydac(DOMAIN_PSYDAC_LOGICAL)

    # Psydac symbolic Derham
    DERHAM_symb = Derham(DOMAIN_symb)

    # grid parameters
    pow = 4
    Nel      = [2**pow, 2**pow, 2]
    p        = [2, 2, 2]
    spl_kind = [True, True, True] 
    n_quad   = [4, 4, 4]
    bc       = [['f', 'f'], None, None]

    # Psydac discrete De Rham
    DOMAIN_PSY  = discretize(DOMAIN_symb, ncells=Nel, comm=MPI.COMM_WORLD) # The parallelism is initiated here.
    DERHAM_PSY  = discretize(DERHAM_symb, DOMAIN_PSY, degree=p, periodic=spl_kind)

    # Spline spaces
    V0 = DERHAM_PSY.V0
    V1 = DERHAM_PSY.V1
    V2 = DERHAM_PSY.V2
    V3 = DERHAM_PSY.V3

    assert isinstance(V0, TensorFemSpace)
    assert isinstance(V1, ProductFemSpace)
    assert isinstance(V2, ProductFemSpace)
    assert isinstance(V3, TensorFemSpace)

    assert isinstance(V0.vector_space, StencilVectorSpace)
    assert isinstance(V1.vector_space, BlockVectorSpace)
    assert isinstance(V2.vector_space, BlockVectorSpace)
    assert isinstance(V3.vector_space, StencilVectorSpace)

    #####################################
    ### Stencil objects (distributed) ###
    #####################################
    # Stencil vector
    u0 = StencilVector(V0.vector_space)

    if mpi_rank == 0:
        print('\nRank:', mpi_rank, 'Global grid size, Nel =',Nel, '\n')
    MPI_COMM.Barrier()

    print('Rank:', mpi_rank, 'Local indices: u0.starts =', u0.starts, 'u0.ends =', u0.ends, 'u0.pads = ', u0.pads)
    MPI_COMM.Barrier()

    print('Rank:', mpi_rank, 'u0[:].shape =', u0[:].shape, 'u0.shape =', u0.shape)
    MPI_COMM.Barrier()

    # Assign some values to local fields
    if mpi_rank == 0:
        u0[u0.starts[0] : u0.ends[0] + 1, u0.starts[1] : u0.ends[1] + 1, :] = 1.
        print('Ghost regions in sync:', u0.ghost_regions_in_sync)
    elif mpi_rank == 1:
        u0[u0.starts[0] : u0.ends[0] + 1, u0.starts[1] : u0.ends[1] + 1, :] = 2.
    elif mpi_rank == 2:
        u0[u0.starts[0] : u0.ends[0] + 1, u0.starts[1] : u0.ends[1] + 1, :] = 3.
    elif mpi_rank == 3:
        u0[u0.starts[0] : u0.ends[0] + 1, u0.starts[1] : u0.ends[1] + 1, :] = 4.

    print('u0 updated.')
    MPI_COMM.Barrier()

    u0.update_ghost_regions()
    print('Rank:', mpi_rank, 'Ghost regions in sync:', u0.ghost_regions_in_sync)

    print('Rank:', mpi_rank, 'u0[:, :, 0] =', u0[:, :, 0], '\n')
    MPI_COMM.Barrier()

    # Difference in assignment when operating with _data array
    v0 = u0.copy()
    if mpi_rank == 1:
        print('Rank:', mpi_rank, 'v0[:, :, 0] =', v0[:, :, 0], '\n')

        # Assign to stencil vector
        shift_i = 1
        shift_j = 2
        u0[u0.starts[0] + shift_i, u0.starts[1] + shift_j, :] = 999.

        # Assign to ._data attribute (np.array) of stencil vector (needs padding and is local)
        v0._data[u0.pads[0] + shift_i, u0.pads[1] + shift_j, :] = 999.
        print('Set u0 only on rank 1.\n')

    assert np.all(u0.toarray() == v0.toarray())

    print('Rank:', mpi_rank, 'u0[:, :, 0] =', u0[:, :, 0], '\n')
    MPI_COMM.Barrier()
    print('Rank:', mpi_rank, 'v0[:, :, 0] =', v0[:, :, 0], '\n')
    MPI_COMM.Barrier()

    # Numpy arrays
    print('Rank:', mpi_rank, '(global array): ', u0.toarray(), '\n')
    MPI_COMM.Barrier()
    
    print('Rank:', mpi_rank, '(local array): ', u0.toarray_local(), '\n')
    MPI_COMM.Barrier()

    # Assign some new values to local fields
    if mpi_rank == 0:
        u0[u0.starts[0] + 2 : u0.ends[0] + 1, u0.starts[1] + 2 : u0.ends[1] + 3, :] = 0.
        print('Ghost regions in sync:', u0.ghost_regions_in_sync)
    elif mpi_rank == 1:
        u0[:, :, :] = 2.
        u0[u0.starts[0] + 2 : u0.ends[0] + 1, u0.starts[1] + 2 : u0.ends[1] + 1, :] = 0.
    elif mpi_rank == 2:
        u0[:, :, :] = 3.
        u0[u0.starts[0] + 2 : u0.ends[0] + 1, u0.starts[1] + 2 : u0.ends[1] + 1, :] = 0.
    elif mpi_rank == 3:
        u0[:, :, :] = 4.
        u0[u0.starts[0] + 2 : u0.ends[0] + 1, u0.starts[1] + 2 : u0.ends[1] + 1, :] = 0.

    print('u0 updated again.')
    MPI_COMM.Barrier()

    u0.update_ghost_regions()
    print('Rank:', mpi_rank, 'Ghost regions in sync:', u0.ghost_regions_in_sync)

    print('Rank:', mpi_rank, 'u0[:, :, 0] =', u0[:, :, 0], '\n')
    MPI_COMM.Barrier()

    # Stencil Matrix
    A00 = StencilMatrix(V0.vector_space, V0.vector_space)

    if mpi_rank == 0:
        print('\nRank:', mpi_rank, 'Global grid size, Nel =',Nel, '\n')
    MPI_COMM.Barrier()

    print('Rank:', mpi_rank, 'domain.starts =', A00.domain.starts, 'domain.ends =', A00.domain.ends, 'domain.pads = ', A00.domain.pads)
    MPI_COMM.Barrier()

    print('Rank:', mpi_rank, 'codomain.starts =', A00.codomain.starts, 'codomain.ends =', A00.codomain.ends, 'codomain.pads = ', A00.codomain.pads)
    MPI_COMM.Barrier()

    print('Rank:', mpi_rank, 'A00[:, :].shape =', A00[:, :].shape, 'A00.shape =', A00.shape)
    MPI_COMM.Barrier()
    
    if mpi_rank == 0:
        # _data is local and includes paddings
        print('\nRank:', mpi_rank, '_data shape:', A00._data.shape)

    # Identity matrix (columns are indexed j - i)
    A00[A00.codomain.starts[0] : A00.codomain.ends[0] + 1, 
        A00.codomain.starts[1] : A00.codomain.ends[1] + 1,
        A00.codomain.starts[2] : A00.codomain.ends[2] + 1, 0, 0, 0] = 1.
    MPI_COMM.Barrier()

    # Distributed matrix-vector product 
    v0 = A00.dot(u0)

    assert np.all(v0.toarray() == u0.toarray())


if __name__ == '__main__':
    test_psydac_stencil_objects()
