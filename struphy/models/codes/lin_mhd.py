#!/usr/bin/env python3

from mpi4py import MPI      
import yaml
import time
import numpy as np

from struphy.diagnostics                import data_module
from struphy.geometry                   import domain_3d
from struphy.mhd_equil                  import mhd_equil_physical 
from struphy.mhd_equil                  import mhd_equil_logical 
from struphy.mhd_equil.gvec             import mhd_equil_gvec
from struphy.feec                       import spline_space
from struphy.mhd_init                   import mhd_init 
from struphy.feec.projectors.pro_global import mhd_operators_cc_lin_6d as mhd_ops  
from struphy.models.substeps            import push_linear_mhd

def execute(file_in, path_out, restart):
    '''Executes the code lin_mhd.

    Parameters
    ----------

    file_in : str
        Absolute path to input parameters file (.yml).
    path_out : str
        Absolute path to output folder.
    restart : boolean
        Restart ('True') or new simulation ('False').
    '''
    print()
    print('Starting code "lin_mhd" ...')
    print()

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # load simulation parameters
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # ========================================================================================= 
    # DOMAIN object
    # =========================================================================================
    domain_type = params['geometry']['type']
    DOMAIN   = domain_3d.Domain(domain_type, params['geometry']['params_' + domain_type])
    print('Domain object of type "' + domain_type + '" set.')
    print()
    
    # ========================================================================================= 
    # FEEC SPACES object
    # =========================================================================================
    Nel         = params['grid']['Nel']
    p           = params['grid']['p']
    spl_kind    = params['grid']['spl_kind']
    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0], params['grid']['nq_el'][0], params['grid']['bc']) 
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1], params['grid']['nq_el'][1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2], params['grid']['nq_el'][2])

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
    SPACES.assemble_M2(DOMAIN)
    SPACES.assemble_M3(DOMAIN)
    print('Assembly of mass matrices done.')
    print()

    # preconditioner
    if params['solvers']['PRE'] == 'ILU':
        SPACES.projectors.assemble_approx_inv(params['solvers']['tol_inv'])

    # ========================================================================================= 
    # MHD EQUILIBRIUM object
    # =========================================================================================
    # MHD equilibirum (physical)
    mhd_equil_type = params['mhd_equilibrium']['general']['type']
    EQ_MHD_P = mhd_equil_physical.Equilibrium_mhd_physical(mhd_equil_type, params['mhd_equilibrium']['params_' + mhd_equil_type])
    
    # MHD equilibrium (logical)
    if mhd_equil_type == 'gvec':
        EQ_MHD_L = mhd_equil_gvec.Equilibrium_mhd_gvec(params['mhd_equilibrium']['params_' + mhd_equil_type], DOMAIN, EQ_MHD_P, SPACES, SOURCE_DOMAIN=None)
    else:
        EQ_MHD_L = mhd_equil_logical.Equilibrium_mhd_logical(DOMAIN, EQ_MHD_P)
        
    print('MHD equilibrium of type "' + mhd_equil_type + '" set.')
    print()

    # ========================================================================================= 
    # MHD variables object
    # =========================================================================================
    # TODO: restart has to be done here
    mhd_init_type = params['mhd_init']['general']['type']
    MHD = mhd_init.Initialize_mhd(DOMAIN, SPACES, params['mhd_init']['general'],
                                                  params['mhd_init']['params_' + mhd_init_type])
    print('MHD variables of type "' + mhd_init_type + '" initialized.')
    print()

    # ========================================================================================= 
    # MHD PROJECTION OPERATORS object
    # =========================================================================================
    # TODO: add matrix-free version 
    MHD_OPS = mhd_ops.MHD_operators(SPACES, EQ_MHD_L, DOMAIN, params['mhd_init']['general']['basis_u']) 
        
    # assemble mass matrix weighted with 0-form density
    MHD_OPS.assemble_MR()
    MHD_OPS.assemble_MJ()
    print('Assembly of weighted mass matrices done.')
    print()

    # assemble right-hand sides of projection matrices
    MHD_OPS.assemble_dofs_EF()
    MHD_OPS.assemble_dofs_FL('m')
    MHD_OPS.assemble_dofs_FL('p')
    MHD_OPS.assemble_dofs_PR() 
    
    # create liner MHD operators as scipy.sparse.linalg.LinearOperator
    MHD_OPS.setOperators(params['time']['dt'], params['time']['dt'], 'step_2')
    print('Assembly of MHD projection operators done.')
    print()

    MHD_OPS.setInverseA()
    MHD_OPS.setPreconditionerS2(params['solvers']['PRE'])
    MHD_OPS.setPreconditionerS6(params['solvers']['PRE'])
    print('MHD preconditioners of type "' + params['solvers']['PRE'] + '" set.')
    print()

    # ========================================================================================= 
    # DATA object for saving
    # =========================================================================================
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_U' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    'en_p' : np.empty(1, dtype=float), 
                    #'divB' : np.empty(1, dtype=float),
                    #'momentum' : np.empty(1, dtype=float),
                    #'helicity' : np.empty(1, dtype=float),
                    #'cr_helicity' : np.empty(1, dtype=float),
                    }
 
    time_series['time'][0] = 0.
    time_series['en_U'][0] = 1/2*MHD.up.dot(MHD_OPS.A(MHD.up))
    time_series['en_B'][0] = 1/2*MHD.b2.dot(SPACES.M2.dot(MHD.b2))
    time_series['en_p'][0] = 1/(5./3. - 1.)*sum(MHD.pp.flatten())
        
    # create object for data saving (only rank 0)
    if mpi_rank == 0:
        DATA = data_module.Data_container(path_out)
        print('Data object initialized.')
        print()
        # add other mhd variabels to data object (flattened) for saving
        DATA.add_data({'density': MHD.r3})
        DATA.add_data({'mhd velocity': MHD.up})
        DATA.add_data({'magnetic field': MHD.b2})
        DATA.add_data({'pressure': MHD.pp})
        DATA.add_data(time_series)
        print()

    # ========================================================================================= 
    # Initialize time stepping function
    # =========================================================================================
    if   params['time']['split_algo'] == 'LieTrotter':

        # set time steps for Lie-Trotter splitting
        dts_mhd     = [params['time']['dt'],
                       params['time']['dt']]

    elif params['time']['split_algo'] == 'Strang':

        # set time steps for Strang splitting
        dts_mhd     = [params['time']['dt'],
                       params['time']['dt']/2.]  

    else:
        raise ValueError('Time stepping scheme not available.')

    UPDATE_MHD = push_linear_mhd.Linear_mhd(dts_mhd, SPACES, MHD_OPS, params['solvers'], 
                                                                 params['mhd_init']['general']['basis_u'], 
                                                                 params['mhd_init']['general']['basis_p'])  
    print('MHD time stepping available.')  
    print() 

    def update():

        if params['time']['split_algo'] == 'LieTrotter':

            # substeps (Lie-Trotter splitting):
            UPDATE_MHD.step_alfven(MHD.up, MHD.b2, print_info=params['solvers']['show_info_alfven'])
            UPDATE_MHD.step_magnetosonic(MHD.r3, MHD.up, MHD.b2, MHD.pp, print_info=params['solvers']['show_info_sonic'])
  
        elif params['time']['split_algo'] == 'Strang':

            # substeps (Strang splitting):
            UPDATE_MHD.step_magnetosonic(MHD.r3, MHD.up, MHD.b2, MHD.pp, print_info=params['solvers']['show_info_sonic'])
            UPDATE_MHD.step_alfven(MHD.up, MHD.b2, print_info=params['solvers']['show_info_alfven'])
            UPDATE_MHD.step_magnetosonic(MHD.r3, MHD.up, MHD.b2, MHD.pp, print_info=params['solvers']['show_info_sonic'])

        else:
            raise NotImplementedError('Only Lie-Trotter and Strang splitting available.')   

    print('Update function set.')
    print() 
        
    # =========================================================================================    
    # time integration 
    # ========================================================================================= 
    if mpi_rank == 0:
        print('Start time integration: ' + params['time']['split_algo'])
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

        break_cond_1 = time_steps_done * params['time']['dt'] > params['time']['Tend']
        break_cond_2 = (time.time() - start_simulation)/60 > params['time']['max_time']

        # stop time loop?
        if break_cond_1 or break_cond_2:
            # close output file and time loop
            if mpi_rank == 0:
                DATA.file.close()
                end_simulation = time.time()
                print('time of simulation [sec]: ', end_simulation - start_simulation)

            break
            
        # call update function for time stepping
        update() 
        time_steps_done += 1  

        # update time series
        time_series['time'][0] = 0.
        time_series['en_U'][0] = 1/2*MHD.up.dot(MHD_OPS.A(MHD.up))
        time_series['en_B'][0] = 1/2*MHD.b2.dot(SPACES.M2.dot(MHD.b2))
        time_series['en_p'][0] = 1/(5./3. - 1.)*sum(MHD.pp.flatten())

        # save data:
        if mpi_rank == 0: DATA.save_data() 

        # print number of finished time steps and current energies
        if mpi_rank == 0 and time_steps_done%1 == 0:
            print('time steps finished : ' + str(time_steps_done) + '/'
                                           + str(int(round(params['time']['Tend'] / params['time']['dt'])))) 
            print()     