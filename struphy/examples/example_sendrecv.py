from mpi4py import MPI

import numpy as np
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
comm_size = comm.Get_size()

assert comm_size == 4

# This test must be run with 4 mpi processes.
# 15 + rank particles are drawn randomly in [0,1]^3 per process.
# The domain is decomposed according to
# process 0: [0 .5 0 .5 0 1.]
# process 1: [.5 1 0 .5 0 1.]
# process 2: [0 .5 .5 1 0 1.]
# process 3: [.5 1 .5 1 0 1.]
# Particles that are not on the process sub-domain are sent to the correct process.
# Array "particles_loc" is allocated once; holes are created/filled in place.
# For testing: after receive, the particle data is set to the sending process number and filled in the holes on the current process.


# ========================
# Define helper functions
# ========================


def sendrecv_determine_ptbs(particles_loc, dom_arr):
    '''Determine which particles have to be sent from current process, put them in a new array 
    and set their position in particles_loc to -1.
    This can be done purely with numpy functions (fast, vectorized).

    Parameters
    ----------
        particles_loc : np.array
            Local particles array of shape (n_rows, 7). 
            The first three columns hold the positions (eta1, eta2, eta3) of each particle.

        dom_arr : np.array
            2d array of shape (comm_size, 6) defining the domain of each process.

    Returns
    -------
        An array of particles to be sent and particles_loc with corresponding holes.'''

    # Check which particles are in a certain interval (e.g. the process domain)
    conds = np.logical_and(
        particles_loc[:, :3] > dom_arr[rank, ::2], particles_loc[:, :3] < dom_arr[rank, 1::2])
    conds_m1 = particles_loc[:, 0] == -1

    # To stay on the current process, all three columns must be True
    stay = np.all(conds, axis=1)

    # True values can stay on the process, False must be sent, already empty rows (-1) cannot be sent
    holes = np.nonzero(~stay)[0]
    send_inds = np.nonzero(~stay[~conds_m1])[0]

    sleep(.1*(rank + 1))
    print(f'rank {rank} | send_inds={send_inds}')
    comm.Barrier()

    # New array for sending particles. TODO: do not create new array, but just return send_inds? Careful: just particles_loc[send_ids] already creates a new array in memory
    particles_to_be_sent = particles_loc[send_inds]

    # set holes to -1
    #particles_loc[send_inds] = -1

    return particles_to_be_sent, holes  # , particles_loc


def sendrecv_get_destinations(ptbs, dom_arr):
    '''Determine to which process particles have to be sent.

    Parameters
    ----------
        ptbs: np.array
            Particles to be sent of shape (send_inds.size, 7).

        dom_arr : np.array
            2d array of shape (comm_size, 6) defining the domain of each process.'''

    # One entry for each process
    send_info = np.zeros(comm_size, dtype=int)
    send_list = []

    # TODO: do not loop over all processes, start with neighbours and work outwards (using while)
    for i in range(comm_size):

        conds = np.logical_and(
            ptbs[:, :3] > dom_arr[i, ::2], ptbs[:, :3] < dom_arr[i, 1::2])
        send_to_i = np.nonzero(np.all(conds, axis=1))
        send_info[i] = send_to_i[0].size
        send_list += [ptbs[send_to_i]]

    return send_info, send_list


def sendrecv_all_to_all(send_info):
    '''Distribute info on how many partciels will be sent/received to/from each process via all-to-all.'''

    recvbuf = np.zeros(comm_size, dtype=int)

    comm.Alltoall(send_info, recvbuf)

    return recvbuf


def sendrecv_particles(send_list, recv_info, holes, particles_loc):
    '''Use non-blocking communication.

    Parameters
    ----------
        send_list : list
            Holds one particle array for each process to send to.

        recv_info : array[int]
            i-th entry holds the number of particles to be received from process i.

        holes : array[int]
            Indices of holes in particles_loc that can be filled.'''

    # i-th entry holds the number (not the index) of the first hole to be filled by data from process i
    first_hole = np.cumsum(recv_info) - recv_info

    if rank == 0:
        print(f'first_hole: {first_hole}')

    # Initialize send and receive commands
    reqs = []
    recvbufs = []
    for i, (mail, N_recv) in enumerate(zip(send_list, list(recv_info))):
        if i == rank:
            reqs += [None]
            recvbufs += [None]
        else:
            comm.Isend(mail, dest=i, tag=rank)
            recvbufs += [np.zeros((N_recv, 7), dtype=float)]
            reqs += [comm.Irecv(recvbufs[-1], source=i, tag=i)]

    # Wait for buffer, then put particles into holes
    test_reqs = [False] * (recv_info.size - 1)
    while len(test_reqs) > 0:
        # loop over all receive requests
        for i, req in enumerate(reqs):
            if req is None:
                continue
            else:
                # check if data has been received
                if req.Test():
                    print(f'rank {rank} | Data received from process {i}.')
                    #if rank == 0: print(f'rank {rank} | i={i}, recvbuf: \n {recvbufs[i]}')
                    # for testing, set received data to sending process number
                    recvbufs[i][:] = i
                    for n in range(recv_info[i]):
                        # fill a hole with particle received from process i
                        particles_loc[holes[first_hole[i] + n]
                                      ] = recvbufs[i][n]
                    test_reqs.pop()
                    reqs[i] = None
                    print(f'rank {rank} | test_reqs: {test_reqs}')
                # else:
                #     print(f'rank {rank} | Data from process {i} not yet reveived.')


def main():
    """
    TODO
    """
    # set up local particles
    Np_loc = 15 + rank
    eps = .4  # high eps (40 %) because of low particle number in this test
    n_rows = round(Np_loc*(1 + 1/np.sqrt(Np_loc) + eps))
    particles_loc = -1 * np.ones((n_rows, 7), dtype=float)
    sleep(.1*(rank + 1))
    print(f'rank {rank} | Np_loc={Np_loc}')
    comm.Barrier()

    particles_loc[:Np_loc] = np.random.rand(Np_loc, 7)
    sleep(.1*(rank + 1))
    print(f'rank {rank} | particles_loc.shape={particles_loc.shape}')
    comm.Barrier()
    if rank == 0:
        print(f'rank {rank} | particles_loc:\n {particles_loc}')

    # Set up domain array for this example
    domain_array = np.zeros((comm_size, 6), dtype=float)
    domain_array[:, 1] = 1.
    domain_array[:, 3] = 1.
    domain_array[:, 5] = 1.

    domain_array[0, 1] = .5
    domain_array[0, 3] = .5
    domain_array[1, 0] = .5
    domain_array[1, 3] = .5
    domain_array[2, 1] = .5
    domain_array[2, 2] = .5
    domain_array[3, 0] = .5
    domain_array[3, 2] = .5

    if rank == 0:
        print(f'rank {rank} | dom_array: \n {domain_array}')

    # create new send particles array and make corresponding holes in particles array
    particles_to_be_sent, holes = sendrecv_determine_ptbs(
        particles_loc, domain_array)

    if rank == 0:
        print('')
    sleep(.1*(rank + 1))
    print(f'rank {rank} | holes:\n {holes}')
    comm.Barrier()

    if rank == 0:
        print('')
    if rank == 0:
        print(f'rank {rank} | particles_to_be_sent:\n {particles_to_be_sent}')

    # where to send particles
    send_info, send_list = sendrecv_get_destinations(
        particles_to_be_sent, domain_array)

    sleep(.1*(rank + 1))
    print(
        f'rank {rank} will send the following amounts (to i-th process): {send_info}')
    comm.Barrier()

    if rank == 0:
        print('')
    if rank == 0:
        print(f'rank {rank} | send_list:\n {send_list}')
    comm.Barrier()

    # transpose send_info
    recv_info = sendrecv_all_to_all(send_info)

    if rank == 0:
        print('')

    sleep(.1*(rank + 1))
    print(
        f'rank {rank} will receive the following amounts (from i-th process): {recv_info}')
    comm.Barrier()

    # send/receive partciels
    sendrecv_particles(send_list, recv_info, holes, particles_loc)

    if rank == 0:
        print('')
    if rank == 0:
        print(f'rank {rank} | particles_loc after receive:\n {particles_loc}')


if __name__ == '__main__':
    main()
