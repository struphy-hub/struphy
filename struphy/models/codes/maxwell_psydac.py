#!/usr/bin/env python3


def execute(file_in, path_out, comm, restart=False, verbose=False):
    '''Executes the code maxwell_psydac.

    Parameters
    ----------

    file_in : str
        Absolute path to input parameters file (.yml).

    path_out : str
        Absolute path to output folder.

    comm : mpi communicator

    restart : boolean
        Restart ('True') or new simulation ('False').

    verbose : boolean
        Print more solver info.
    '''
     
    import yaml
    import time
    import socket
    import numpy as np

    from struphy.feec.psydac_derham import Derham_build
    from struphy.diagnostics.data_module import Data_container_psydac as Data_container
    from struphy.geometry.domain_3d import Domain
    from struphy.mhd_equil.gvec             import mhd_equil_gvec
    from struphy.feec                       import spline_space
    from struphy.models.substeps.push_maxwell import Push_maxwell_psydac
    from struphy.mhd_init                   import emw_init
    from struphy.psydac_linear_operators.fields import Field_init

    from psydac.linalg.stencil import StencilVector

    # mpi communicator
    MPI_COMM = comm
    mpi_rank = MPI_COMM.Get_rank()
    print("Hello from rank {:0>4d} : {}".format(mpi_rank, socket.gethostname()))
    MPI_COMM.Barrier()

    if mpi_rank == 0:
        print(f'\nMPI communicator initialized with {MPI_COMM.Get_size()} process(es).\n')

    code_name = '"maxwell_psydac"'
    
    if mpi_rank == 0:
        print('Starting code ' + code_name +  '...\n')
        print(f'file_in : {file_in}')
        print(f'path_out: {path_out}\n')
    
    # load simulation VARIABLES
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # ========================================================================================= 
    # DOMAIN object
    # =========================================================================================
    dom_type   = params['geometry']['type']
    dom_params = params['geometry']['params_' + dom_type]

    DOMAIN     = Domain(dom_type, dom_params)
    F_psy      = DOMAIN.Psydac_mapping('F', **dom_params) # create psydac mapping for mass matrices only
    
    if mpi_rank == 0:
        print(f'domain type: {dom_type}')
        print(f'domain parameters: {dom_params}')
        print(f'DOMAIN of type "' + dom_type + '" set.')
        print()
    
    # ========================================================================================= 
    # DERHAM sequence (Psydac)
    # =========================================================================================
    # Grid parameters
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

    # Assemble necessary mass matrices 
    DR.assemble_M1()
    if verbose: print(f'Rank: {mpi_rank} | Assembly of M1 done.')
    DR.assemble_M2()
    if verbose: print(f'Rank: {mpi_rank} | Assembly of M2 done.\n')
    
    # ========================================================================================= 
    # Field variables (with feec)
    # =========================================================================================
    # TODO: restart has to be done here
    f_names  = params['fields']['general']['names']
    f_spaces = params['fields']['general']['spaces']
    f_init   = params['fields']['general']['init']
    f_coords = params['fields']['general']['init_coords']
    f_comps  = params['fields']['general']['init_comps']
    f_params = params['fields']['params_' + f_init]

    fields = []
    for name, space, comps in zip(f_names, f_spaces, f_comps):
        fields += [Field_init(name, space, comps, f_init, f_coords, f_params, DR, DOMAIN)]

        if verbose:
            print(f'Rank: {mpi_rank} | field      : {fields[-1].name}')
            print(f'Rank: {mpi_rank} | space_cont : {fields[-1].space_cont}')
            print(f'Rank: {mpi_rank} | starts     : {fields[-1].starts}')
            print(f'Rank: {mpi_rank} | ends       : {fields[-1].ends}')
            print(f'Rank: {mpi_rank} | pads       : {fields[-1].pads}')

        MPI_COMM.Barrier()

    # Pointers to Stencil-/Blockvectors
    e = fields[0].vector
    b = fields[1].vector
    # print('')

    # ========================================================================================= 
    # DATA object for saving
    # =========================================================================================
    DATA = Data_container(path_out, comm=MPI_COMM)

    for field in fields:

        if isinstance(field.vector, StencilVector):
            key = field.name
            DATA.add_data({key: field.vector._data}) # save numpy array to be updated each time step.
            DATA.f[key].attrs['space_cont'] = field.space_cont
            DATA.f[key].attrs['starts'] = field.starts
            DATA.f[key].attrs['ends'] = field.ends
            DATA.f[key].attrs['pads'] = field.pads
        else:
            for n in range(3):
                key = field.name + '_' + str(n + 1)
                DATA.add_data({key: field.vector[n]._data}) # save numpy array to be updated each time step.
                DATA.f[key].attrs['space_cont'] = field.space_cont
                DATA.f[key].attrs['starts'] = field.starts
                DATA.f[key].attrs['ends'] = field.ends
                DATA.f[key].attrs['pads'] = field.pads

    if mpi_rank == 0: print(f'Rank: {mpi_rank} | Field initial conditions saved.\n')
    
    # Add other variables to be saved
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_E' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    }
 
    time_series['time'][0] = 0.
    time_series['en_E'][0] = 1/2*e.dot(DR.M1.dot(e))
    time_series['en_B'][0] = 1/2*b.dot(DR.M2.dot(b))

    DATA.add_data(time_series)

    if mpi_rank == 0: print(f'Rank: {mpi_rank} | Initial time series saved.\n')

    if verbose:
        if mpi_rank == 0: DATA.info()

    # ========================================================================================= 
    # Initialize time stepping function
    # =========================================================================================
    # Define stepping scheme
    dt = params['time']['dt']
    split_algo = params['time']['split_algo']
    
    if split_algo == 'LieTrotter':
        time_steps     = [dt, dt]

    elif params['time']['split_algo'] == 'Strang':
        time_steps     = [dt, dt/2.]  

    else:
        raise ValueError('Time stepping scheme not available.')

    # Initialize splitting substeps
    update_maxwell = Push_maxwell_psydac(DR, time_steps, params['solvers']['step_maxwell'])  

    # Define update function
    def update():
        if split_algo == 'LieTrotter':
            update_maxwell(e, b)
        elif split_algo == 'Strang':
            update_maxwell(e, b) # No splitting here in Maxwell equations.
        else:
            raise NotImplementedError('Only Lie-Trotter and Strang splitting available.')   

    if mpi_rank == 0:
        print('Update function set.')
        print()
        
    # =========================================================================================    
    # time integration 
    # ========================================================================================= 
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
  

if __name__ == '__main__':
    # do "pip install -e ." to use these paths
    execute('struphy/io/inp/maxwell_psydac/parameters.yml', 
            'struphy/io/out/sim_1/', restart=False)
