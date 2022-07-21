#!/usr/bin/env python3

'''
STRUPHY main execution file.
'''

from psydac.api.postprocessing import OutputManager
from psydac.linalg.stencil import StencilVector
from struphy.diagnostics.data_module import Data_container_psydac as Data_container
from struphy.models import models
from struphy.psydac_api.psydac_derham import Derham
from struphy.geometry.domain_3d import Domain
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
domain = Domain(dom_type, dom_params)

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

# Save model object
if rank == 0:
    with open(path_out + 'MODEL_names.bin', 'wb') as handle:
        pickle.dump([model.names, model.space_ids], handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

# Set initial conditions for fields and particles (if they exist)
if 'fields' in params:
    fields_init = params['fields']['init']
else:
    fields_init = None

if 'kinetic' in params:
    particles_init = []
    particles_params = []
    for key, val in params['kinetic'].items():
        particles_init += [val['perturbations']['type']]
        particles_params += [val['perturbations'][particles_init[-1]]]
else:
    particles_init = None
    particles_params = None

model.set_initial_conditions(
    fields_init, particles_init, particles_params)

model.update_scalar_quantities(0.)

# Output Manager Initialization
# om = OutputManager(path_out + 'FIELD_DATA_spaces.yml', path_out + 'FIELD_DATA_fields.h5')
# om.add_spaces(V0=derham.V0, V1=derham.V1, V2=derham.V2, V3=derham.V3)
# om.export_space_info()

# for patch in om.space_info['patches']:
#     for space in patch['vector_spaces']:
#         for key, val in space.items():
#             print(key)
#             print(val)
#             print('')
#             if isinstance(val, list):
#                 for va in val:
#                     for k, v in va.items():
#                         print(k)
#                         print(v)
#                 print('')

# field_dict = {}
# for field in model.fields:
#     field_dict[field.name] = field.field

# om.add_snapshot(t=0., ts=0)
# om.export_fields(**field_dict)

# data object for saving
data = Data_container(path_out, comm=comm)

for field in model.fields:

    if isinstance(field.vector, StencilVector):
        key = field.name
        # save numpy array to be updated each time step.
        data.add_data({key: field.vector._data})
        data.f[key].attrs['space_id'] = field.space_id
        data.f[key].attrs['starts'] = field.starts
        data.f[key].attrs['ends'] = field.ends
        data.f[key].attrs['pads'] = field.pads
    else:
        for n in range(3):
            key = field.name + '_' + str(n)
            # save numpy array to be updated each time step.
            data.add_data({key: field.vector[n]._data})
            data.f[key].attrs['space_id'] = field.space_id
            data.f[key].attrs['starts'] = field.starts
            data.f[key].attrs['ends'] = field.ends
            data.f[key].attrs['pads'] = field.pads

data.add_data(model.scalar_quantities)

if rank == 0:
    print(f'\nRank: {rank} | Initial time series saved.\n')
    model.print_scalar_quantities()

# Define stepping scheme
dt = params['time']['dt']
split_algo = params['time']['split_algo']


def update():

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


# time integration
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
        data.f.close()
        # om.export_space_info() TODO: Psydac Derham functionaltiy not yet implemented.
        end_simulation = time.time()
        if rank == 0:
            print()
            print('time of simulation [sec]: ',
                  end_simulation - start_simulation)
            print()
        break

    # call update function for time stepping
    update()
    time_steps_done += 1

    # update time series
    model.update_scalar_quantities(dt*time_steps_done)

    # save data:
    # om.add_snapshot(t=dt*time_steps_done, ts=time_steps_done) 
    # om.export_fields(**field_dict)
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
