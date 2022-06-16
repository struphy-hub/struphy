import numpy as np
import matplotlib.pyplot as plt

from struphy.geometry.domain_3d import Domain

from struphy.psydac_api.psydac_derham import Derham

from struphy.pic.particles import Particles6D

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

# parameters
Nel = [8, 16, 2]
p = [2, 2, 2]
spl_kind = [False, True, True]


loading_params = {'type' : 'pseudo_random', 'seed' : 1234, 'dir_particles' : 'dir', 'vth_x' : 1., 'vth_y': 1., 'vth_z' : 1., 'v0_x' : 0., 'v0_y' : 0., 'v0_z' : 0.}

marker_params = {'type' : 'fullorbit', 'ppc' : 10, 'loading' : loading_params, 'n_bins' : [32, 32], 'v_max' : 5.}

# create domain
domain = Domain('shafranov_shift')

dummy_params = {'x0' : 0.5, 'y0' : 0.5, 'z0' : 0.5}
F_psy = domain.Psydac_mapping('F', **dummy_params)

# create de rham object
derham = Derham(Nel, p, spl_kind, F=F_psy, comm=mpi_comm)

if mpi_rank == 0:
    print()
    print('Domain decomposition according to : ')
    print(derham.domain_array)

# create particles
particles = Particles6D('energetic_ions', marker_params, domain, derham.domain_array, mpi_comm)

mpi_comm.Barrier()
print('Number of particles w/wo holes on each process before sorting : ')
print('Rank', mpi_rank, ':', particles.n_k_loc, particles.n_k_loc_all)

particles.show_physical()

# sort particles according to domain decomposition
mpi_comm.Barrier()
particles.send_recv_markers()

mpi_comm.Barrier()
print('Number of particles w/wo holes on each process after sorting : ')
print('Rank', mpi_rank, ':', particles.n_k_loc, particles.n_k_loc_all)

particles.show_physical()