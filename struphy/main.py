#!/usr/bin/env python3

'''
STRUPHY main execution file.
'''

from psydac.api.postprocessing import OutputManager
from psydac.linalg.stencil import StencilVector
from struphy.post_processing.output_handling import DataContainer
from struphy.models import models
from struphy.geometry import domains
from struphy.psydac_api.psydac_derham import Derham
import numpy as np
import time
import pickle
import yaml
import datetime
import sysconfig
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# get arguments
model_name = sys.argv[1]
file_in = sys.argv[2]
path_out = sys.argv[3]
path_batch = sys.argv[4]
file_meta = sys.argv[5]
mode = 'w'  # needs fix
if len(sys.argv) > 6:
    mode = sys.argv[6]

# write meta data
if rank == 0:
    with open(file_meta, 'w') as f:
        f.write('\ndate of simulation: '.ljust(20) +
                str(datetime.datetime.now()) + '\n')
        f.write('platform: '.ljust(20) + sysconfig.get_platform() + '\n')
        f.write('python version: '.ljust(20) +
                sysconfig.get_python_version() + '\n')
        f.write('model_name: '.ljust(20) + model_name + '\n')
        f.write('# processes: '.ljust(20) + str(comm.Get_size()) + '\n')

    print(
        f'\nMPI communicator initialized with {comm.Get_size()} process(es).\n')
    print('Starting model ' + model_name + '...\n')

# load simulation VARIABLES
with open(file_in) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# domain object
dom_type = params['geometry']['type']
dom_params = params['geometry'][dom_type]

domain_class = getattr(domains, dom_type)
domain = domain_class(dom_params)

if rank == 0:
    print(f'domain type: {dom_type}')
    print(f'domain parameters: {dom_params}\n')

# DERHAM object
Nel = params['grid']['Nel']             # Number of grid cells
p = params['grid']['p']                 # spline degrees
spl_kind = params['grid']['spl_kind']   # Spline types (clamped vs. periodic)
bc = params['grid']['bc']               # Boundary conditions (Homogeneous Dirichlet or None)

# Number of quadrature points per grid cell
nq_el = params['grid']['nq_el']
# Number of quadrature points per histopolation cell
nq_pr = params['grid']['nq_pr']

derham = Derham(Nel, p, spl_kind, bc, quad_order=[
                nq_el[0] - 1, nq_el[1] - 1, nq_el[2] - 1], nq_pr=nq_pr, comm=comm)

if rank == 0:
    print('GRID parameters:')
    print(f'Nel     : {Nel}')
    print(f'p       : {p}')
    print(f'spl_kind: {spl_kind}')
    print(f'bc      : {bc}')
    print(f'nq_el   : {nq_el}')
    print(f'nq_pr   : {nq_pr}\n')
    print('Discrete Derham set (polar=' + str(params['grid']['polar']) + ').')
    print()

# STRUPHY model
model_class = getattr(models, model_name)
model = model_class(derham, domain, params)

# Set initial conditions for fields and particles (if they exist)
if 'fields' in params:
    fields_init = params['fields']['init']
else:
    fields_init = None

if 'kinetic' in params:
    particles_init = []
    for key, val in params['kinetic'].items():
        particles_init += [val]
else:
    particles_init = None

model.set_initial_conditions(
    fields_init, particles_init)

model.update_scalar_quantities(0.)

# data object for saving
data = DataContainer(path_out, comm=comm)

# save scalar quantities in group 'scalar/'
for key, val in model.scalar_quantities.items():
    key_scalar = 'scalar/' + key
    data.add_data({key_scalar : val})

# save fields in group 'fields/'
for field in model.fields:
    key_field = 'fields/' + field.name
    
    # save numpy array to be updated each time step.
    if isinstance(field.vector, StencilVector):    
        data.add_data({key_field : field.vector._data})
    else:
        for n in range(3):
            key_component = key_field + '/' + str(n + 1)
            data.add_data({key_component : field.vector[n]._data})
            
    # save field meta data
    data.file[key_field].attrs['space_id'] = field.space_id
    data.file[key_field].attrs['starts'] = field.starts
    data.file[key_field].attrs['ends'] = field.ends
    data.file[key_field].attrs['pads'] = field.pads

# save kinetic data in group 'kinetic/'
# TODO
            

if rank == 0:
    print(f'\nRank: {rank} | Initial time series saved.\n')
    model.print_scalar_quantities()

# define stepping scheme
dt = params['time']['dt']
split_algo = params['time']['split_algo']


def integrate_in_time():

    # First order in time
    if split_algo == 'LieTrotter':

        for propagator in model.propagators:
            propagator(dt)

    # Second order in time
    elif split_algo == 'Strang':

        assert len(model.propagators) > 1

        for propagator in model.propagators:
            propagator(dt/2.)
            
        for propagator in model.propagators[::-1]:
            propagator(dt/2.)

    else:
        raise NotImplementedError(
            f'Splitting scheme {split_algo} not available.')


# start time integration
if rank == 0:
    print('Start time integration: ' + split_algo)
    print()

start_simulation = time.time()

# time loop
time_steps_done = 0
while True:

    # synchronize MPI processes and check if simulation end is reached
    comm.Barrier()

    break_cond_1 = time_steps_done * \
        params['time']['dt'] >= params['time']['Tend']
    break_cond_2 = (time.time() - start_simulation) / \
        60 > params['time']['max_time']

    # stop time loop?
    if break_cond_1 or break_cond_2:
        # close output file and time loop
        data.file.close()
        # om.export_space_info() TODO: Psydac Derham functionaltiy not yet implemented.
        end_simulation = time.time()
        if rank == 0:
            print()
            print('time of simulation [sec]: ',
                  end_simulation - start_simulation)
            print()
        break

    # call time integrator for time stepping
    integrate_in_time()
    time_steps_done += 1

    # update time series
    model.update_scalar_quantities(dt*time_steps_done)

    # save data
    data.save_data()

    # print number of finished time steps and current energies
    if rank == 0 and time_steps_done % 1 == 0:
        total_steps = str(
            int(round(params['time']['Tend'] / params['time']['dt'])))
        str_len = len(total_steps)
        step = str(time_steps_done).zfill(str_len)
        message = 'time steps finished : ' + step + '/' + total_steps
        print('\r', message, end='\n')
        model.print_scalar_quantities()
