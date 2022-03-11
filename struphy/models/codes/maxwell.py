#!/usr/bin/env python3

from mpi4py import MPI      
import yaml
import time
import numpy as np

from struphy.diagnostics                import data_module
from struphy.geometry                   import domain_3d
from struphy.mhd_equil.gvec             import mhd_equil_gvec
from struphy.feec                       import spline_space
from struphy.models.substeps            import push_maxwell
from struphy.mhd_init                   import emw_init




def execute(file_in, path_out, restart):
    '''Executes the code maxwell.

    Parameters
    ----------

    file_in : str
        Absolute path to input parameters file (.yml).
    path_out : str
        Absolute path to output folder.
    restart : boolean
        Restart ('True') or new simulation ('False').
    '''

    code_name = '"maxwell"'
    
    print()
    print('Starting code' + code_name +  '...')
    print()

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()
    

    # load simulation VARIABLES
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # ========================================================================================= 
    # DOMAIN object
    # =========================================================================================
    domain_type = params['geometry']['type']
    DOMAIN      = domain_3d.Domain(domain_type, params['geometry']['params_' + domain_type])
    
    print('Domain object of type "' + domain_type + '" set.')
    print()
    
    # ========================================================================================= 
    # FEEC SPACES object
    # =========================================================================================
    Nel             = params['grid']['Nel']             # Number of grid cells
    p               = params['grid']['p']               # spline degree
    spl_kind        = params['grid']['spl_kind']        # Spline type
    spaces_FEM_1    = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0], params['grid']['nq_el'][0], params['grid']['bc']) 
    spaces_FEM_2    = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1], params['grid']['nq_el'][1])
    spaces_FEM_3    = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2], params['grid']['nq_el'][2])

    spaces_FEM_1.set_projectors(params['grid']['nq_pr'][0]) 
    spaces_FEM_2.set_projectors(params['grid']['nq_pr'][1])
    spaces_FEM_3.set_projectors(params['grid']['nq_pr'][2])

    if params['grid']['polar']:
        SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3], ck=1, cx=DOMAIN.cx, cy=DOMAIN.cy)
    else:
        SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    SPACES.set_projectors('general')
    print('FEEC spaces and projectors set (polar=' + str(params['grid']['polar']) + ').')
    print()

    # assemble mass matrices 
    SPACES.assemble_M1(DOMAIN)
    SPACES.assemble_M2(DOMAIN)
    print('Assembly of mass matrices done.')
    print()

    # preconditioner
    if params['solvers']['PRE'] == 'ILU':
        SPACES.projectors.assemble_approx_inv(params['solvers']['tol_inv'])

    
    # ========================================================================================= 
    # MHD variables object
    # =========================================================================================
    # TODO: restart has to be done here
    init_type       = params['initialization']['general']['type']
    init_general    = params['initialization']['general']
    init_params     = params['initialization']['params_' + init_type]
    VARIABLES       = emw_init.Initialize_emw(DOMAIN, SPACES, init_general, init_params)

    print('Variables of type "' + init_type + '" initialized')
    print()

    # ========================================================================================= 
    # DATA object for saving
    # =========================================================================================
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_E' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    }
 
    time_series['time'][0] = 0.
    time_series['en_E'][0] = 1/2*VARIABLES.e1.dot(SPACES.M1.dot(VARIABLES.e1))
    time_series['en_B'][0] = 1/2*VARIABLES.b2.dot(SPACES.M2.dot(VARIABLES.b2))

     # create object for data saving (only rank 0)
    if mpi_rank == 0:
        DATA = data_module.Data_container(path_out)
        print('Data object initialized.')
        print()
        # add other mhd variabels to data object (flattened) for saving
        DATA.add_data({'electric field': VARIABLES.e1})
        DATA.add_data({'magnetic field': VARIABLES.b2})
        DATA.add_data(time_series)
        print()


    # ========================================================================================= 
    # Initialize time stepping function
    # =========================================================================================
    
    # Define stepping scheme
    split_algo = params['time']['split_algo']
    
    if split_algo == 'LieTrotter':
        time_steps     = [params['time']['dt'], params['time']['dt']]

    elif params['time']['split_algo'] == 'Strang':
        time_steps     = [params['time']['dt'], params['time']['dt']/2.]  

    else:
        raise ValueError('Time stepping scheme not available.')
    
    # Initialize Update function
    UPDATE = push_maxwell.Push_maxwell(DOMAIN, SPACES, time_steps,  params)  

    def update():
        if split_algo == 'LieTrotter':
            UPDATE.step_maxwell(VARIABLES.e1, VARIABLES.b2, print_info=params['solvers']['show_info'])
        elif split_algo == 'Strang':
            UPDATE.step_maxwell(VARIABLES.e1, VARIABLES.b2, print_info=params['solvers']['show_info'])

        else:
            raise NotImplementedError('Only Lie-Trotter and Strang splitting available.')   

    print('Update function set.')
    print() 
        
    # =========================================================================================    
    # time integration 
    # ========================================================================================= 
    if mpi_rank == 0:
        print('Start time integration: ' + split_algo)
        print()

    if mpi_rank == 0: 
        if restart:
            time_steps_done = DATA.file['time'].size
        else:
            time_steps_done = 0

    start_simulation = time.time()

    # time loop
    while True:

        # synchronize MPI processes and check if simulation end is reached
        MPI_COMM.Barrier()

        break_cond_1 = time_steps_done * params['time']['dt'] >= params['time']['Tend']
        break_cond_2 = (time.time() - start_simulation)/60 > params['time']['max_time']

        # stop time loop?
        if break_cond_1 or break_cond_2:
            # close output file and time loop
            if mpi_rank == 0:
                DATA.file.close()
                end_simulation = time.time()
                print()
                print('time of simulation [sec]: ', end_simulation - start_simulation)
                print()

            break
            
        # call update function for time stepping
        update() 
        time_steps_done += 1  

        # update time series
        time_series['time'][0] = 0.
        time_series['en_E'][0] = 1/2*VARIABLES.e1.dot(SPACES.M1.dot(VARIABLES.e1))
        time_series['en_B'][0] = 1/2*VARIABLES.b2.dot(SPACES.M2.dot(VARIABLES.b2))

        # save data:
        if mpi_rank == 0: DATA.save_data() 

        # print number of finished time steps and current energies
        if mpi_rank == 0 and time_steps_done%1 == 0:
            total_stetps    = str(int(round(params['time']['Tend'] / params['time']['dt']))) 
            str_len         = len(total_stetps)
            step            = str(time_steps_done).zfill(str_len)
            message         = 'time steps finished : ' + step + '/' + total_stetps
            print('\r', message, end='')
  
