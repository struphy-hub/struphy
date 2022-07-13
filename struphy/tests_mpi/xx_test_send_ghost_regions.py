import pytest

from mpi4py import MPI
import numpy as np


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[8, 9, 10]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ])
def test_send_ghost_regions(Nel, p, spl_kind, mapping, verbose=False):
    '''
    Tests for all 4 discrete spaces and all 3 directions if the sending to the left and to the right works correctly.
    There are two cases when the ghost regions have to be sent: If the process has neighbours (from struphy.psydac_api.psydac_derham.Derham.neighbours)
    or if it is a single process in this direction but there are periodic boundary conditions.
    '''

    from struphy.pic.particles_to_grid import Accumulator

    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import Derham

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Domain object
    map = mapping[0]
    params_map = mapping[1]

    DOMAIN = Domain(map, params_map)

    # Psydac discrete Derham sequence
    DR = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print(f'rank {rank} | domain_array:\n {DR.domain_array}\nrank {rank} | spl_kind: {spl_kind}')

    comm.Barrier()
    print(f'rank {rank} | neighbours: \n {DR.neighbours}')

    return

    # just a placeholder for the Accumulator object to be initialized later
    accum_name = 'cc_lin_mhd_6d_1'

    if rank == 0:
        print(f'\nspl_kind now : {spl_kind}')
        print(f'domain array : \n {DR.domain_array}\n')

    for space_id in ['H1', 'Hcurl', 'Hdiv', 'L2']:
        if rank == 0:
            print(f'space {space_id}')
        
        Acc = Accumulator(DOMAIN, DR, space_id, accum_name, do_vector=False)

        pads = DR.V1.vector_space.pads

        dirs = np.nonzero(DR.neighbours[::2] + 1)[0]
        dirs = np.append(dirs, np.nonzero(DR.neighbours[1::2] + 1)[0])
        # Get direction in which the domain is periodic since ghost_sending also has to be done in this direction even if this direction
        # is not split amogn the processes
        dirs_per = np.nonzero(spl_kind)[0]

        dirs = np.append(dirs, dirs_per)

        dirs = set(dirs)

        for dat in Acc.args_data:
            for di in dirs:
                if rank == 0 and verbose == True:
                    print('\n================================')
                    print(f'========= Direction {di} ==========')
                    print('================================')

                comm.Barrier()

                fill = 3*(di+1)

                # test sending to the left
                if rank == 0 and verbose == True:
                    print(f'\n===== Testing to the left ======')

                if len(dat.shape) == 6:
                    # indices for testing
                    inds = [pads[0], pads[1], pads[2],
                            pads[0], pads[1], pads[2]]
                    inds[di] = slice(None)
                    inds[3+di] = slice(None)

                    # indices for reading
                    inds_r = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]
                    inds_r[di] = slice(pads[di], 2*pads[di])

                    # indices for writing
                    inds_w = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]
                    inds_w[di] = slice(-pads[di], None)

                    inds = tuple(inds)
                    inds_w = tuple(inds_w)
                    inds_r = tuple(inds_r)

                    dat[inds_w] = fill

                    if verbose == True:
                        print(f'\nrank {rank} has matrix before : \n {dat[inds]}')
                        print(f'\nrank {rank} has matrix before : \n {dat[inds_r]}')

                    Acc._send_ghost_regions()

                    Acc.update_ghost_regions()

                    if verbose == True:
                        print(f'\nrank {rank} has matrix after : \n {dat[inds]}')
                        print(f'\nrank {rank} has matrix after : \n {dat[inds_r]}')

                elif len(dat.shape) == 3:
                    # indices for reading
                    inds_r = [pads[0], pads[1], pads[2]]
                    inds_r[di] = slice(pads[di], 2*pads[di])

                    # indices for writing
                    inds_w = [pads[0], pads[1], pads[2],]
                    inds_w[di] = slice(-pads[di], None)

                    inds_w = tuple(inds_w)
                    inds_r = tuple(inds_r)

                    dat[inds_w] = fill

                    Acc._send_ghost_regions()

                    Acc.update_ghost_regions()

                if DR.neighbours[2*di] != -1 or spl_kind[di] == True:
                    if verbose == True:
                        print(f'Testing rank {rank} to the left')
                    assert np.any(dat[inds_r] == fill)

                comm.Barrier()

                if len(dat.shape) == 6:
                    dat[:, :, :, :, :, :] = 0.
                elif len(dat.shape) == 3:
                    dat[:, :, :] = 0.

                comm.Barrier()

                # test sending to the right
                if rank == 0 and verbose == True:
                    print(f'\n===== Testing to the right ======')

                if len(dat.shape) == 6:
                    # indices for testing
                    inds = [pads[0], pads[1], pads[2],
                            pads[0], pads[1], pads[2]]
                    inds[di] = slice(None)
                    inds[3+di] = slice(None)

                    # indices for reading
                    inds_r = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]
                    inds_r[di] = slice(-2*pads[di], -pads[di])

                    # indices for writing
                    inds_w = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]
                    inds_w[di] = slice(0, pads[di])

                    inds = tuple(inds)
                    inds_w = tuple(inds_w)
                    inds_r = tuple(inds_r)

                    dat[inds_w] = fill

                    if verbose == True:
                        print(f'\nrank {rank} has matrix before : \n {dat[inds]}')
                        print(f'\nrank {rank} has matrix before : \n {dat[inds_r]}')

                    Acc._send_ghost_regions()

                    Acc.update_ghost_regions()

                    if verbose == True:
                        print(f'\nrank {rank} has matrix after : \n {dat[inds]}')
                        print(f'\nrank {rank} has matrix after : \n {dat[inds_r]}')

                elif len(dat.shape) == 3:
                    # indices for reading
                    inds_r = [pads[0], pads[1], pads[2]]
                    inds_r[di] = slice(-2*pads[di], -pads[di])

                    # indices for writing
                    inds_w = [pads[0], pads[1], pads[2]]
                    inds_w[di] = slice(0, pads[di])

                    inds_w = tuple(inds_w)
                    inds_r = tuple(inds_r)

                    dat[inds_w] = fill

                    Acc._send_ghost_regions()

                    Acc.update_ghost_regions()

                if DR.neighbours[2*di+1] != -1 or spl_kind[di] == True:
                    if verbose == True and verbose == True:
                        print(f'Testing rank {rank} to the right')
                    assert np.any(dat[inds_r] == fill)

                comm.Barrier()
        
        if rank == 0:
            print('Test passed!\n')


if __name__ == '__main__':

    # run with 1 and 2 processes:
    # for spl_kind in [[False, False, True], 
    #                  [False, True, False], 
    #                  [True, False, False],
    #                  [False, True, True],
    #                  [True, False, True],
    #                  [True, True, False],
    #                  [True, True, True]]:
    #     test_send_ghost_regions([8, 8, 8], [2, 2, 2], spl_kind, ['cuboid', {
    #         'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], verbose=False)

    # run with 4 processes:
    test_send_ghost_regions([4, 4, 16], [2, 2, 2], [False, False, True], ['cuboid', {
            'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], verbose=False)

    test_send_ghost_regions([4, 16, 4], [2, 2, 2], [False, True, False], ['cuboid', {
            'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], verbose=False)

    test_send_ghost_regions([16, 4, 4], [2, 2, 2], [True, False, False], ['cuboid', {
            'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], verbose=False)

    test_send_ghost_regions([4, 16, 16], [2, 2, 2], [False, True, True], ['cuboid', {
            'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], verbose=False)
