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
import time
import numpy as np

from struphy.geometry.domain_3d import Domain
from struphy.feec.psydac_derham import Derham_build
from struphy.models.codes import models
from struphy.diagnostics.data_module import Data_container_psydac as Data_container

from psydac.linalg.stencil import StencilVector

# print('Number of arguments:', len(sys.argv), 'arguments.')
# print('Argument List:', sys.argv)

code       = sys.argv[1]
file_in    = sys.argv[2]
path_out   = sys.argv[3] 
path_batch = sys.argv[4]
file_meta  = sys.argv[5]
mode = 'w' # needs fix
if len(sys.argv) > 6: 
    mode = sys.argv[6]

if mpi_rank == 0: 
    with open(file_meta, 'w') as f:
        f.write('\ndate of simulation: '.ljust(20) + str(datetime.datetime.now()) + '\n')
        f.write('platform: '.ljust(20) + sysconfig.get_platform() + '\n')
        f.write('python version: '.ljust(20) + sysconfig.get_python_version() + '\n')
        f.write('code: '.ljust(20) + code + '\n')
        f.write('# processes: '.ljust(20) + str(MPI_COMM.Get_size()) + '\n')

# start simulation:   
if code=='lin_mhd':
    from struphy.models.codes import lin_mhd
    lin_mhd.execute(file_in, path_out, mode=='a') 

elif code=='lin_mhd_MF':
    from struphy.models.codes import lin_mhd_MF
    lin_mhd_MF.execute(file_in, path_out, mode=='a') 

elif code=='lin_mhd_psydac':
    from struphy.models.codes import lin_mhd_psydac
    lin_mhd_psydac.execute(file_in, path_out, mode=='a') 

elif code=='cc_lin_mhd_6d':
    from struphy.models.codes import cc_lin_mhd_6d 
    cc_lin_mhd_6d.execute(file_in, path_out, mode=='a')

elif code=='cc_lin_mhd_6d_MF':
    from struphy.models.codes import cc_lin_mhd_6d_MF 
    cc_lin_mhd_6d_MF.execute(file_in, path_out, mode=='a')

elif code=='pc_lin_mhd_6d_MF_full':
    from struphy.models.codes import pc_lin_mhd_6d_MF_full 
    pc_lin_mhd_6d_MF_full.execute(file_in, path_out, mode=='a')

elif code=='pc_lin_mhd_6d_MF_perp':
    from struphy.models.codes import pc_lin_mhd_6d_MF_perp 
    pc_lin_mhd_6d_MF_perp.execute(file_in, path_out, mode=='a')
    
elif code=='kinetic_extended':
    from struphy.models.codes import kinetic_extended
    kinetic_extended.execute(file_in, path_out, mode=='a')

elif code=='maxwell':
    from struphy.models.codes import maxwell 
    maxwell.execute(file_in, path_out, mode=='a')

elif code=='inverse_mass_test':
    from struphy.models.codes import inverse_mass_test as code_file
    code_file.execute(file_in, path_out, MPI_COMM)

elif code=='cold_plasma':
    from struphy.models.codes import cold_plasma
    cold_plasma.execute(file_in, path_out, mode=='a')

elif code=='lin_Vlasov_Maxwell':
    from struphy.models.codes import lin_Vlasov_Maxwell
    lin_Vlasov_Maxwell.execute(file_in, path_out, mode=='a')
    
elif code=='cc_cold_plasma_6d':
    from struphy.models.codes import cc_cold_palsma_6d
    cc_cold_palsma_6d.execute(file_in, path_out, mode=='a')
    
elif code=='maxwell_psydac':
    
    if mpi_rank == 0:
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

    # FE fields
    names  = params['fields']['general']['names']
    spaces = params['fields']['general']['spaces']
    # ======================================= 
    # MODEL SPECIFIC: instance of model class
    # =======================================
    if code=='maxwell_psydac':
        MODEL = models.Maxwell(names, spaces, DR, DOMAIN, params['solvers']['option_1'])
    else:
        raise NotImplementedError(f'Model {code} not implemented.')
        exit()

    # Set initial conditions
    init_type   = params['fields']['general']['init']
    init_coords = params['fields']['general']['init_coords']
    init_comps  = params['fields']['general']['init_comps']
    init_params = params['fields']['params_' + init_type]

    MODEL.set_initial_conditions(init_comps, init_type, init_coords, init_params)

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

    # TODO: ad hoc, needs fix:
    # Add other variables to be saved
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_E' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    }
 
    e = MODEL.fields[0].vector
    b = MODEL.fields[1].vector

    time_series['time'][0] = 0.
    time_series['en_E'][0] = 1/2*e.dot(DR.M1.dot(e))
    time_series['en_B'][0] = 1/2*b.dot(DR.M2.dot(b))

    DATA.add_data(time_series)

    if mpi_rank == 0: 
        print(f'Rank: {mpi_rank} | Initial time series saved.\n')
        print(f'total energy: {time_series["en_E"][0] + time_series["en_B"][0]}')

    # Define stepping scheme
    dt = params['time']['dt']
    split_algo = params['time']['split_algo']

    # Define update function
    def update():
        if split_algo == 'LieTrotter':
            
            for prop, vars in zip(MODEL.propagators(), MODEL.substep_vars()):
                prop(*vars, dt)

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
        time_series['time'][0] = 0.
        time_series['en_E'][0] = 1/2*e.dot(DR.M1.dot(e))
        time_series['en_B'][0] = 1/2*b.dot(DR.M2.dot(b))

        # save data:
        DATA.save_data() 

        # print number of finished time steps and current energies
        if mpi_rank == 0 and time_steps_done%1 == 0:
            total_stetps    = str(int(round(params['time']['Tend'] / params['time']['dt']))) 
            str_len         = len(total_stetps)
            step            = str(time_steps_done).zfill(str_len)
            message         = 'time steps finished : ' + step + '/' + total_stetps
            print('\r', message, end='\n')
            print(f'total energy: {time_series["en_E"][0] + time_series["en_B"][0]}')


