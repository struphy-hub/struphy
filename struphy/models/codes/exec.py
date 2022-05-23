#!/usr/bin/env python3

'''
Main execution file.
'''

from mpi4py import MPI 

MPI_COMM = MPI.COMM_WORLD
mpi_rank = MPI_COMM.Get_rank()

import sys
import sysconfig
import datetime
import yaml
import pickle
import time
import numpy as np

from struphy.geometry.domain_3d import Domain
from struphy.psydac_api.psydac_derham import Derham_build
from struphy.models.codes import models
from struphy.diagnostics.data_module import Data_container_psydac as Data_container

from psydac.linalg.stencil import StencilVector
from psydac.api.postprocessing import OutputManager

# get arguments
code       = sys.argv[1]
file_in    = sys.argv[2]
path_out   = sys.argv[3] 
path_batch = sys.argv[4]
file_meta  = sys.argv[5]
mode = 'w' # needs fix
if len(sys.argv) > 6: 
    mode = sys.argv[6]

# write meta data
if mpi_rank == 0: 
    with open(file_meta, 'w') as f:
        f.write('\ndate of simulation: '.ljust(20) + str(datetime.datetime.now()) + '\n')
        f.write('platform: '.ljust(20) + sysconfig.get_platform() + '\n')
        f.write('python version: '.ljust(20) + sysconfig.get_python_version() + '\n')
        f.write('code: '.ljust(20) + code + '\n')
        f.write('# processes: '.ljust(20) + str(MPI_COMM.Get_size()) + '\n')

    print(f'\nMPI communicator initialized with {MPI_COMM.Get_size()} process(es).\n')
    print('Starting code ' + code +  '...\n')

# load simulation VARIABLES
with open(file_in) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# DOMAIN object
dom_type   = params['geometry']['type']
dom_params = params['geometry']['params_' + dom_type]

DOMAIN     = Domain(dom_type, dom_params)
F_psy      = DOMAIN.Psydac_mapping('F', **dom_params) # create psydac mapping for mass matrices only

if mpi_rank == 0:
    print(f'domain type: {dom_type}')
    print(f'domain parameters: {dom_params}\n')

# DERHAM object
Nel             = params['grid']['Nel']             # Number of grid cells
p               = params['grid']['p']               # spline degree
spl_kind        = params['grid']['spl_kind']        # Spline type

DR=Derham_build(Nel, p, spl_kind, F = F_psy, comm = MPI_COMM)   

if mpi_rank == 0:
    print('GRID parameters:')
    print(f'Nel     : {Nel}')
    print(f'p       : {p}')
    print(f'spl_kind: {spl_kind}\n')
    print('Discrete Derham set (polar=' + str(params['grid']['polar']) + ').')
    print()

# ======================================= 
# MODEL SPECIFIC: instance of model class
# =======================================
if code=='maxwell':
    MODEL = models.Maxwell(DR, DOMAIN, params['solvers']['pcg_1'])
else:
    raise NotImplementedError(f'Model {code} not implemented.')
    exit()
# ======================================= 
# END OF MODEL SPECIFIC PART
# =======================================

# Save MODEL object
if mpi_rank == 0:
    with open(path_out + 'MODEL_names.bin', 'wb') as handle:
        pickle.dump([MODEL.names, MODEL.space_ids], handle, protocol=pickle.HIGHEST_PROTOCOL) 

# Set initial conditions
init_type   = params['fields']['init']['type']
init_coords = params['fields']['init']['coords']
init_comps  = params['fields']['init']['comps']
init_params = params['fields']['params_' + init_type]

MODEL.set_initial_conditions(init_comps, init_type, init_coords, init_params)

# TODO: Psydac Derham functionaltiy not yet implemented.
# Output Manager Initialization
# FIELD_DATA = OutputManager(path_out + 'FIELD_DATA_spaces.yml', path_out + 'FIELD_DATA_fields.h5')
# FIELD_DATA.add_spaces(V0=DR.V0, V1=DR.V1, V2=DR.V2, V3=DR.V3)
# for patch in FIELD_DATA.space_info['patches']:
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
# for field in MODEL.fields:
#     field_dict[field.name] = field.field

# FIELD_DATA.add_snapshot(t=0., ts=0)
# FIELD_DATA.export_fields(**field_dict)

# DATA object for saving
DATA = Data_container(path_out, comm=MPI_COMM)

for field in MODEL.fields:

    if isinstance(field.vector, StencilVector):
        key = field.name
        DATA.add_data({key: field.vector._data}) # save numpy array to be updated each time step.
        DATA.f[key].attrs['space_id'] = field.space_id
        DATA.f[key].attrs['starts'] = field.starts
        DATA.f[key].attrs['ends'] = field.ends
        DATA.f[key].attrs['pads'] = field.pads
    else:
        for n in range(3):
            key = field.name + '_' + str(n)
            DATA.add_data({key: field.vector[n]._data}) # save numpy array to be updated each time step.
            DATA.f[key].attrs['space_id'] = field.space_id
            DATA.f[key].attrs['starts'] = field.starts
            DATA.f[key].attrs['ends'] = field.ends
            DATA.f[key].attrs['pads'] = field.pads

DATA.add_data(MODEL.scalar_quantities)

MODEL.update_scalar_quantities(0.)

if mpi_rank == 0: 
    print(f'Rank: {mpi_rank} | Initial time series saved.\n')
    MODEL.print_scalar_quantities()

# Define stepping scheme
dt = params['time']['dt']
split_algo = params['time']['split_algo']

def update():

    # First order in time
    if split_algo == 'LieTrotter':
        
        for propagator in MODEL.propagators:
            propagator.push(dt)

    # Second order in time
    elif split_algo == 'Strang':

        assert len(MODEL.propagators) > 1
        
        for propagator in MODEL.propagators[:-1]:
            propagator.push(dt/2.)

        MODEL.propagators[-1].push(dt)

        for propagator in MODEL.propagators[::-1][1:]:
            propagator.push(dt/2.)

    else:
        raise NotImplementedError(f'Splitting scheme {split_algo} not available.') 
        
# time integration 
if mpi_rank == 0:
    print('Start time integration: ' + split_algo)
    print()

start_simulation = time.time()

# time loop
time_steps_done = 0
while True:

    # synchronize MPI processes and check if simulation end is reached
    MPI_COMM.Barrier()

    break_cond_1 = time_steps_done * params['time']['dt'] >= params['time']['Tend']
    break_cond_2 = (time.time() - start_simulation)/60 > params['time']['max_time']

    # stop time loop?
    if break_cond_1 or break_cond_2:
        # close output file and time loop
        DATA.f.close()
        #FIELD_DATA.export_space_info() TODO: Psydac Derham functionaltiy not yet implemented.
        end_simulation = time.time()
        if mpi_rank == 0:
            print()
            print('time of simulation [sec]: ', end_simulation - start_simulation)
            print()
        break
        
    # call update function for time stepping
    update() 
    time_steps_done += 1  

    # update time series
    MODEL.update_scalar_quantities( dt*time_steps_done )

    # save data:
    # FIELD_DATA.add_snapshot(t=dt*time_steps_done, ts=time_steps_done) TODO: Psydac Derham functionaltiy not yet implemented.
    # FIELD_DATA.export_fields(**field_dict)
    DATA.save_data() 

    # print number of finished time steps and current energies
    if mpi_rank == 0 and time_steps_done%1 == 0:
        total_steps    = str(int(round(params['time']['Tend'] / params['time']['dt']))) 
        str_len         = len(total_steps)
        step            = str(time_steps_done).zfill(str_len)
        message         = 'time steps finished : ' + step + '/' + total_steps
        print('\r', message, end='\n')
        MODEL.print_scalar_quantities()


