import pytest

from mpi4py import MPI
import numpy as np


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize('Nel', [[10, 16, 20]])
@pytest.mark.parametrize('p', [[2, 3, 4]])
@pytest.mark.parametrize('spl_kind', [[False, False, True], [False, True, False], [True, False, False], [True, True, False], [True, False, True], [False, True, True], [True, True, True]])
@pytest.mark.parametrize('mapping', [
    ['cuboid', {
        'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], ])
def test_send_ghost_regions(Nel, p, spl_kind, mapping, verbose=False):
    """
    Tests for all 4 discrete spaces and all 26 direction components describing the 26 geometric neighbours of a 3x3x3 cube.
    It is tested if the sending to the component and inverted component works. The test runs through all direction components
    and sends both to the component and the inverse component but assertion statement is only tested if the neighbour is not -1.

    For this test to work it is important that the nuber of elements for one process is bigger than 2*p for every direction!
    Otherwise the writing of the ghost regions will overlap and the test cannot handle this case (it is still correct).
    Now, only a single element is being written and then tested; you can test it also with the whole ghost region, but pay attention
    to the above warning.
    """
    from struphy.pic.particles_to_grid import Accumulator

    from struphy.geometry.domain_3d import Domain
    from struphy.psydac_api.psydac_derham import Derham

    from itertools import product

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Domain object
    map = mapping[0]
    params_map = mapping[1]

    DOMAIN = Domain(map, params_map)

    # Psydac discrete Derham sequence
    DR = Derham(Nel, p, spl_kind, comm=comm)

    # just a placeholder for the Accumulator object to be initialized later
    accum_name = 'cc_lin_mhd_6d_1'

    neighbours = DR.neighbours

    if rank == 0:
        print(f'\nspl_kind now : {spl_kind}')
        print(f'domain array : \n {DR.domain_array}\n')
    if verbose == True:
        print(f'rank {rank} has neighbours : \n {neighbours}\n')

    for space_id in ['H1', 'Hcurl', 'Hdiv', 'L2']:
        if rank == 0:
            if space_id in {'H1', 'L2'}:
                print('\n================================')
                print(f'========== Space {space_id} ============')
                print('================================')
            elif space_id in {'Hcurl', 'Hdiv'}:
                print('\n================================')
                print(f'========== Space {space_id} =========')
                print('================================')
        
        Acc = Accumulator(DOMAIN, DR, space_id, accum_name, do_vector=False)

        pads = DR.V0.vector_space.pads

        for dat in Acc.args_data:
            for comp in product([0,1,2], repeat=3):

                if rank == 0 and verbose == True:
                    print('\n================================')
                    print(f'====== Direction ({comp[0]},{comp[1]},{comp[2]}) =======')
                    print('================================')

                comm.Barrier()

                inv_comp = Acc._invert_component(comp)

                fill = 3*np.sum(comp) + 1

                if len(dat.shape) == 6:
                    # Test component
                    # indices for writing
                    inds_w_co = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]
                    for k,di in enumerate(comp):
                        if di == 1:
                            continue
                        elif di == 0:
                            # inds_w_co[k] = slice(None, pads[k])
                            inds_w_co[k] = 1
                        elif di == 2:
                            # inds_w_co[k] = slice(-pads[k], None)
                            inds_w_co[k] = -pads[k] + 1
                    inds_w_co = tuple(inds_w_co)
                    
                    # indices for reading
                    inds_r_co = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]

                    for k,di in enumerate(inv_comp):
                        if di == 1:
                            continue
                        elif di == 0:
                            # inds_r_co[k] = slice(pads[k], 2*pads[k])
                            inds_r_co[k] = pads[k] + 1
                        elif di == 2:
                            # inds_r_co[k] = slice(-2*pads[k], -pads[k])
                            inds_r_co[k] = -2*pads[k] + 1
                    inds_r_co = tuple(inds_r_co)

                    dat[inds_w_co] = fill

                    # Test inverse component
                    # indices for writing
                    inds_w_in_co = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]
                    for k,di in enumerate(inv_comp):
                        if di == 1:
                            continue
                        elif di == 0:
                            # inds_w_in_co[k] = slice(None, pads[k])
                            inds_w_in_co[k] = 1
                        elif di == 2:
                            # inds_w_in_co[k] = slice(-pads[k], None)
                            inds_w_in_co[k] = -pads[k] + 1
                    inds_w_in_co = tuple(inds_w_in_co)
                    
                    # indices for reading
                    inds_r_in_co = [pads[0], pads[1], pads[2],
                              pads[0], pads[1], pads[2]]

                    for k,di in enumerate(comp):
                        if di == 1:
                            continue
                        elif di == 0:
                            # inds_r_in_co[k] = slice(pads[k], 2*pads[k])
                            inds_r_in_co[k] = pads[k] + 1
                        elif di == 2:
                            # inds_r_in_co[k] = slice(-2*pads[k], -pads[k])
                            inds_r_in_co[k] = -2*pads[k] + 1
                    inds_r_in_co = tuple(inds_r_in_co)

                    dat[inds_w_in_co] = fill

                    # print before sending ghost regions
                    if verbose == True:
                        print(f'\nrank {rank} read component ({comp[0]},{comp[1]},{comp[2]}) before : \n {dat[inds_r_co]}')
                        print(f'\nrank {rank} read inverse component ({inv_comp[0]},{inv_comp[1]},{inv_comp[2]}) before : \n {dat[inds_r_in_co]}')
                        print(f'\nrank {rank} write component ({comp[0]},{comp[1]},{comp[2]}) before : \n {dat[inds_w_co]}')
                        print(f'\nrank {rank} write inverse component ({inv_comp[0]},{inv_comp[1]},{inv_comp[2]}) before : \n {dat[inds_w_in_co]}')

                    Acc._send_ghost_regions()

                    Acc.update_ghost_regions()

                    # print after sending ghost regions
                    if verbose == True:
                        print(f'\nrank {rank} read component ({comp[0]},{comp[1]},{comp[2]}) after : \n {dat[inds_r_co]}')
                        print(f'\nrank {rank} read inverse component ({inv_comp[0]},{inv_comp[1]},{inv_comp[2]}) after : \n {dat[inds_r_in_co]}')
                        print(f'\nrank {rank} write component ({comp[0]},{comp[1]},{comp[2]}) after : \n {dat[inds_w_co]}')
                        print(f'\nrank {rank} write inverse component ({inv_comp[0]},{inv_comp[1]},{inv_comp[2]}) after : \n {dat[inds_w_in_co]}')
                
                else:
                    raise NotImplementedError('Unknown shape of data object!')

                if neighbours[comp] != -1 and neighbours[inv_comp] == -1:
                    assert (dat[inds_r_in_co] == fill).all(), f'rank {rank} component ({inv_comp[0]},{inv_comp[1]},{inv_comp[2]}) neighbour {neighbours[inv_comp]}'
                
                elif neighbours[comp] == -1 and neighbours[inv_comp] != -1:
                    assert (dat[inds_r_co] == fill).all(), f'rank {rank} component ({comp[0]},{comp[1]},{comp[2]}) neighbour {neighbours[comp]}'
                
                elif neighbours[comp] != -1 and neighbours[inv_comp] != -1:
                    assert (dat[inds_r_co] == fill).all(), f'rank {rank} component ({comp[0]},{comp[1]},{comp[2]}) neighbour {neighbours[comp]}'
                    assert (dat[inds_r_in_co] == fill).all(), f'rank {rank} component ({inv_comp[0]},{inv_comp[1]},{inv_comp[2]}) neighbour {neighbours[inv_comp]}'

                elif neighbours[comp] == -1 and neighbours[inv_comp] == -1:
                    continue

                else:
                    raise NotImplementedError('Something weird happened')

                comm.Barrier()

                if len(dat.shape) == 6:
                    dat[:, :, :, :, :, :] = 0.
                elif len(dat.shape) == 3:
                    dat[:, :, :] = 0.

                comm.Barrier()
        
        if rank == 0:
            print('Test passed!\n')

if __name__ == '__main__':
    import itertools
    for spl_kind in itertools.product([True, False], repeat=3):
        if (np.array(spl_kind) == False).all():
            continue
        else:
            test_send_ghost_regions([12, 14, 16], [2, 3, 4], spl_kind, ['cuboid', {
                'l1': 1., 'r1': 2., 'l2': 10., 'r2': 20., 'l3': 100., 'r3': 200.}], verbose=False)
