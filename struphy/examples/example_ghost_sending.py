from time import time
from struphy.psydac_api.psydac_derham import DerhamBuild
from struphy.geometry.domain_3d import Domain

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

from numpy import empty



def main():
    """To test a specific region set its spl_kind to True and all other to false. In this way
    Psydac splits in this direction and the processes have neighbours that they send the ghost regions to.
    """

    from mpi4py import MPI
    from psydac.linalg.stencil import StencilVector, StencilMatrix
    from psydac.linalg.block import BlockMatrix
    from time import sleep

    mpi_comm = MPI.COMM_WORLD
    rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # parameters
    Nel = [16, 16, 16]
    p = [2, 2, 2]
    spl_kind = [False, False, True]

    # DOMAIN object
    dom_type = 'cuboid'
    dom_params = {'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 20.}

    DOMAIN = Domain(dom_type, dom_params)

    F_psy = DOMAIN.Psydac_mapping('F', **dom_params)

    Np_per_cell = 8

    DR = DerhamBuild(Nel, p, spl_kind, F=F_psy, comm=mpi_comm)

    # particle loading
    np.random.seed(1607)

    index_array = DR.index_array_N
    domain_array = DR.domain_array
    ind_array = DR.index_array_N[rank, :]
    dom_array = DR.domain_array[rank, :]
    neighbours = DR.neighbours

    start1 = dom_array[0]
    end1 = dom_array[1]
    start2 = dom_array[2]
    end2 = dom_array[3]
    start3 = dom_array[4]
    end3 = dom_array[5]

    if rank == 0:
        print(f'\n Nel : \n {Nel}')
        print(f'\n degrees : \n {p}')
        print(f'\n spl_kind : \n {spl_kind}')
        print(f'\n dom_array : \n {domain_array}')
        print(f'\n ind_array : \n {index_array} \n')
    mpi_comm.Barrier()

    n_cells = np.prod(ind_array[1::2] - ind_array[::2] + 1)

    particles_loc = np.empty((n_cells*Np_per_cell, 3), dtype=float)
    particles_loc[:, 0] = start1 + \
        np.random.rand(n_cells*Np_per_cell)*(end1 - start1)
    particles_loc[:, 1] = start2 + \
        np.random.rand(n_cells*Np_per_cell)*(end2 - start2)
    particles_loc[:, 2] = start3 + \
        np.random.rand(n_cells*Np_per_cell)*(end3 - start3)




    V = DR.V1.vector_space.spaces[0]

    from struphy.pic.particles_to_grid import Accumulator

    acc = Accumulator(DOMAIN, DR, 'H1', 'linear_vlasov_maxwell')

    starts_acc = acc._args_space[0]
    ends_acc = acc._args_space[1]
    pads_acc = acc._args_space[2]


    # ====================================
    # ======== Test StencilMatrix ========
    # ====================================

    if rank == 0:
        print('\n ======= testing in eta1-direction =========')
        acc._args_data[0][:pads_acc[0], pads_acc[1]:-pads_acc[1], pads_acc[2]:-pads_acc[2], :, :, :] = 88.
        acc._args_data[0][-pads_acc[0]:, pads_acc[1]:-pads_acc[1], pads_acc[2]:-pads_acc[2], :, :, :] = 99.
    mpi_comm.Barrier()

    print(f'\n rank {rank} has accum matrix \n {acc._args_data[0][:, pads_acc[1], pads_acc[2], :, pads_acc[1], pads_acc[2]]}')
    mpi_comm.Barrier()

    if rank == 0:
        print('\n ========== Now comes the GHOSTING ========= \n')
    mpi_comm.Barrier()

    acc._send_ghost_regions()
    mpi_comm.Barrier()

    print(f'rank {rank} has accum matrix \n {acc._args_data[0][:, pads_acc[1], pads_acc[2], :, pads_acc[1], pads_acc[2]]}')

    acc._args_data[0][:,:,:,:,:,:] = 0.
    mpi_comm.Barrier()


    if rank == 0:
        print('\n\n ======= testing in eta2-direction =========')
        acc._args_data[0][pads_acc[0]:-pads_acc[0], :pads_acc[1], pads_acc[2]:-pads_acc[2], :, :, :] = 88.
        acc._args_data[0][pads_acc[0]:-pads_acc[0], -pads_acc[1]:, pads_acc[2]:-pads_acc[2], :, :, :] = 99.
    mpi_comm.Barrier()

    print(f'\n rank {rank} has accum matrix \n {acc._args_data[0][pads_acc[0], :, pads_acc[2], pads_acc[0], :, pads_acc[2]]}')
    mpi_comm.Barrier()

    if rank == 0:
        print('\n ========== Now comes the GHOSTING ========= \n')
    mpi_comm.Barrier()

    acc._send_ghost_regions()
    mpi_comm.Barrier()

    print(f'rank {rank} has accum matrix \n {acc._args_data[0][pads_acc[0], :, pads_acc[2], pads_acc[0], :, pads_acc[2]]}')

    acc._args_data[0][:,:,:,:,:,:] = 0.
    mpi_comm.Barrier()


    if rank == 0:
        print('\n\n ======= testing in eta3-direction =========')
        acc._args_data[0][pads_acc[0]:-pads_acc[0], pads_acc[1]:-pads_acc[1], :pads_acc[2], :, :, :] = 88.
        acc._args_data[0][pads_acc[0]:-pads_acc[0], pads_acc[1]:-pads_acc[1], -pads_acc[2]:, :, :, :] = 99.
    mpi_comm.Barrier()

    print(f'\n rank {rank} has accum matrix \n {acc._args_data[0][pads_acc[0], pads_acc[1], :, pads_acc[0], pads_acc[1], :]}')
    mpi_comm.Barrier()

    if rank == 0:
        print('\n ========== Now comes the GHOSTING ========= \n')
    mpi_comm.Barrier()

    acc._send_ghost_regions()
    mpi_comm.Barrier()

    print(f'rank {rank} has accum matrix \n {acc._args_data[0][pads_acc[0], pads_acc[1], :, pads_acc[0], pads_acc[1], :]}')

    acc._args_data[0][:,:,:,:,:,:] = 0.



    # ====================================
    # ======== Test StencilMatrix ========
    # ====================================

    if rank == 0:
        print('\n ======= testing in eta1-direction =========')
        acc._args_data[1][:pads_acc[0], pads_acc[1]:-pads_acc[1], pads_acc[2]:-pads_acc[2]] = 88.
        acc._args_data[1][-pads_acc[0]:, pads_acc[1]:-pads_acc[1], pads_acc[2]:-pads_acc[2]] = 99.
    mpi_comm.Barrier()

    print(f'\n rank {rank} has accum vector \n {acc._args_data[0][:, :, pads_acc[2]]}')
    mpi_comm.Barrier()

    if rank == 0:
        print('\n ========== Now comes the GHOSTING ========= \n')
    mpi_comm.Barrier()

    acc._send_ghost_regions()
    mpi_comm.Barrier()

    print(f'rank {rank} has accum vector \n {acc._args_data[1][:, :, pads_acc[2]]}')

    acc._args_data[1][:,:,:] = 0.
    mpi_comm.Barrier()


    if rank == 0:
        print('\n\n ======= testing in eta2-direction =========')
        acc._args_data[1][pads_acc[0]:-pads_acc[0], :pads_acc[1], pads_acc[2]:-pads_acc[2]] = 88.
        acc._args_data[1][pads_acc[0]:-pads_acc[0], -pads_acc[1]:, pads_acc[2]:-pads_acc[2]] = 99.
    mpi_comm.Barrier()

    print(f'\n rank {rank} has accum matrix \n {acc._args_data[1][:, :, pads_acc[2]]}')
    mpi_comm.Barrier()

    if rank == 0:
        print('\n ========== Now comes the GHOSTING ========= \n')
    mpi_comm.Barrier()

    acc._send_ghost_regions()
    mpi_comm.Barrier()

    print(f'rank {rank} has accum matrix \n {acc._args_data[1][:, :, pads_acc[2]]}')

    acc._args_data[1][:,:,:] = 0.
    mpi_comm.Barrier()


    if rank == 0:
        print('\n\n ======= testing in eta3-direction =========')
        acc._args_data[1][pads_acc[0]:-pads_acc[0], pads_acc[1]:-pads_acc[1], :pads_acc[2]] = 88.
        acc._args_data[1][pads_acc[0]:-pads_acc[0], pads_acc[1]:-pads_acc[1], -pads_acc[2]:] = 99.
    mpi_comm.Barrier()

    print(f'\n rank {rank} has accum matrix \n {acc._args_data[1][:, pads_acc[1], :]}')
    mpi_comm.Barrier()

    if rank == 0:
        print('\n ========== Now comes the GHOSTING ========= \n')
    mpi_comm.Barrier()

    acc._send_ghost_regions()
    mpi_comm.Barrier()

    print(f'rank {rank} has accum matrix \n {acc._args_data[1][:, pads_acc[1], :]}')

    acc._args_data[1][:,:,:] = 0.
    mpi_comm.Barrier()


if __name__ == '__main__':
    main()
