import numpy as np
import matplotlib.pyplot as plt

from struphy.geometry import domains

from struphy.psydac_api.psydac_derham import Derham

from struphy.pic.particles import Particles6D

from mpi4py import MPI

comm= MPI.COMM_WORLD
mpi_size = comm.Get_size()
rank = comm.Get_rank()

# parameters
Nel = [8, 16, 4]
p = [2, 2, 2]
spl_kind = [False, True, True]


loading_params = {'type': 'pseudo_random', 'seed': 1234, 'dir_particles': 'dir',
                  'moms_params': [1., 0., 0., 0., 1., 1., 1.]}

marker_params = {'type': 'fullorbit', 'ppc': 10,
                 'loading': loading_params, 'n_bins': [32, 32], 'v_max': 5.}

# create domain
dom_type = 'ShafranovShiftCylinder'
domain_class = getattr(domains, dom_type)
domain = domain_class()

# create de rham object
derham = Derham(Nel, p, spl_kind, comm=comm)

if rank == 0:
    print()
    print('Domain decomposition according to : ')
    print(derham.domain_array)

# create particles
particles = Particles6D('energetic_ions', marker_params,
                        domain, derham.domain_array, comm)

comm.Barrier()
print('Number of particles w/wo holes on each process before sorting : ')
print('Rank', rank, ':', particles.n_mks_loc, particles.n_mks_loc_with_holes)

particles.show_physical()

# sort particles according to domain decomposition
comm.Barrier()
particles.send_recv_markers()

comm.Barrier()
print('Number of particles w/wo holes on each process after sorting : ')
print('Rank', rank, ':', particles.n_mks_loc, particles.n_mks_loc_with_holes)

particles.show_physical()

# are all markers in the correct domain?
conds = np.logical_and(particles.markers[:, :3] > derham.domain_array[rank, ::3], particles.markers[:, :3] < derham.domain_array[rank, 1::3])
holes = particles.markers[:, 0] == -1.
stay = np.all(conds, axis=1)

error_mks = particles.markers[np.logical_and(~stay, ~holes)]

print(f'rank {rank} | markers mot on correct process: {np.nonzero(np.logical_and(~stay, ~holes))} \n corresponding positions:\n {error_mks[:, :3]}')

assert error_mks.size == 0
