import sysconfig
import yaml

import numpy as np
from mpi4py import MPI 

from struphy.geometry                   import domain_3d
from struphy.kinetic_equil              import kinetic_equil_physical 
from struphy.kinetic_equil              import kinetic_equil_logical
from struphy.kinetic_init               import kinetic_init 

# mpi communicator
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
mpi_comm.Barrier()

file_in = sysconfig.get_path("platlib") + '/struphy/io/inp/cc_lin_mhd_6d/parameters.yml'
print(file_in)

# load simulation parameters
with open(file_in) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# Domain:
DOMAIN = domain_3d.Domain(params['geometry']['type'], 
                          params['geometry']['params_' + params['geometry']['type']])
print('Domain object set.')

# kinetic equilibirum (physical)
EQ_KINETIC_P = kinetic_equil_physical.Equilibrium_kinetic_physical(
                    params['kinetic_equilibrium']['general'], 
                    params['kinetic_equilibrium']['params_' + params['kinetic_equilibrium']['general']['type']]
                    )
print('Kinetic equilibrium (physical) set.')
print()

# kinetic equilibrium (logical)
EQ_KINETIC_L = kinetic_equil_logical.Equilibrium_kinetic_logical(DOMAIN, EQ_KINETIC_P)
print('Kinetic equilibrium (logical) set.')
print()

# initialize markers
KIN = kinetic_init.Initialize_markers(DOMAIN, EQ_KINETIC_L, 
                                    params['kinetic_init']['general'],
                                    params['kinetic_init']['params_' + params['kinetic_init']['general']['type']],
                                    params['markers'], mpi_comm
                                    )

print('Markers initialized on rank', mpi_rank)
print()

print(KIN.Np, KIN.Np_loc)
print(KIN.particles_loc.shape)
print(KIN.particles_loc[:3, :])
print(KIN.w0_loc.shape)
print(KIN.w0_loc[:10])