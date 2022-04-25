#!/usr/bin/env python3

from mpi4py import MPI      
import yaml
import time
import numpy as np

from struphy.diagnostics                    import data_module 
from struphy.geometry                       import domain_3d 
from struphy.feec                           import spline_space 
from struphy.fields_equil                   import fields_equil_physical 
from struphy.fields_equil                   import fields_equil_logical 
from struphy.kinetic_equil                  import kinetic_equil_physical 
from struphy.kinetic_equil                  import kinetic_equil_logical 
from struphy.fields_init                    import fields_init 
from struphy.kinetic_init                   import kinetic_init 

from struphy.pic.lin_Vlasov_Maxwell         import accumulation 

from struphy.models.substeps                import push_maxwell
from struphy.models.substeps                import push_markers

from struphy.models.substeps                import push_lin_VM


def execute(file_in, path_out, restart):
    '''Executes the code lin_Vlasov_Maxwell.

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
    print('Starting code "lin_Vlasov_Maxwell" ...')
    print()

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_size = MPI_COMM.Get_size()
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # load simulation parameters
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    v_shift = np.array([params['kinetic_equilibrium']['params_Maxwell_homogen_slab']['v0_x'],
                        params['kinetic_equilibrium']['params_Maxwell_homogen_slab']['v0_y'],
                        params['kinetic_equilibrium']['params_Maxwell_homogen_slab']['v0_z']])

    v_th    = np.array([params['kinetic_equilibrium']['params_Maxwell_homogen_slab']['vth_x'],
                        params['kinetic_equilibrium']['params_Maxwell_homogen_slab']['vth_y'],
                        params['kinetic_equilibrium']['params_Maxwell_homogen_slab']['vth_z']])

    n0      = params['kinetic_equilibrium']['params_Maxwell_homogen_slab']['nh0']



    # ========================================================================================= 
    # DOMAIN Object
    # =========================================================================================
    domain_type = params['geometry']['type']
    DOMAIN   = domain_3d.Domain(domain_type, params['geometry']['params_' + domain_type])
    print('Domain object of type "' + domain_type + '" set.')
    print()
    


    # ========================================================================================= 
    # FEEC SPACES Object
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
    SPACES.assemble_M1(DOMAIN)
    SPACES.assemble_M2(DOMAIN)
    print('Assembly of mass matrices done.')
    print()

    # preconditioner
    if params['solvers']['PRE'] == 'ILU':
        SPACES.projectors.assemble_approx_inv(params['solvers']['tol_inv'])



    # ========================================================================================= 
    # FIELDS EQUILIBRIUM Object
    # =========================================================================================
    
    # FIELDS equilibirum (physical)
    fields_equil_type = params['fields_equilibrium']['general']['type']
    EQ_FIELDS_P = fields_equil_physical.Equilibrium_fields_physical(fields_equil_type, params['fields_equilibrium']['params_' + fields_equil_type])
    
    # FIELDS equilibrium (logical)
    EQ_FIELDS_L = fields_equil_logical.Equilibrium_fields_logical(DOMAIN, EQ_FIELDS_P)
        
    print('FIELDS equilibrium of type "' + fields_equil_type + '" set.')
    print()

    B2_eq = [EQ_FIELDS_L.b2_eq_1, EQ_FIELDS_L.b2_eq_2, EQ_FIELDS_L.b2_eq_3]
    b2_eq = SPACES.projectors.pi_2(B2_eq, include_bc=True, eval_kind='tensor_product', interp=True)



    # ========================================================================================= 
    # FIELDS Variables Object
    # =========================================================================================
    
    # Initialize variables for the field objects
    fields_init_type = params['fields_init']['general']['type']
    FIELDS = fields_init.Initialize_fields(DOMAIN, SPACES, params['fields_init']['general'],
                                                  params['fields_init']['params_' + fields_init_type])
    print('FIELDS variables of type "' + fields_init_type + '" initialized.')
    print()



    # ========================================================================================= 
    # KINETIC EQUILIBRIUM Object
    # =========================================================================================
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

    

    # ========================================================================================= 
    # MARKER and ACCUMULATION Objects
    # =========================================================================================
    # TODO: restart has to be done here
    KIN = kinetic_init.Initialize_markers(  DOMAIN, EQ_KINETIC_L, 
                                            params['kinetic_init']['general'],
                                            params['kinetic_init']['params_' + params['kinetic_init']['general']['type']],
                                            params['markers'],
                                            MPI_COMM
                                            )
    print(KIN.Np_loc, 'markers initialized on rank', mpi_rank)
    print()

    # create particle accumulator (all processes)
    ACCUM = accumulation.Accumulation(SPACES, DOMAIN, MPI_COMM)

    print('Accumulator initialized on rank', mpi_rank)
    print()



    # ========================================================================================= 
    # DATA Object for Saving
    # =========================================================================================
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_E' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    # 'en_W' : np.empty(1, dtype=float), 
                    # 'divB' : np.empty(1, dtype=float),
                    }
 
    time_series['time'][0] = 0.
    time_series['en_E'][0] = 1/2*FIELDS.e1.dot(SPACES.M1.dot(FIELDS.e1))
    time_series['en_B'][0] = 1/2*FIELDS.b2.dot(SPACES.M2.dot(FIELDS.b2))
    # time_series['en_W'][0] = 1/2*KIN
    
    # create object for data saving (only rank 0)
    if mpi_rank == 0:
        DATA = data_module.Data_container(path_out)
        print('Data object initialized.')
        print()
        # add other fields variabels to data object (flattened) for saving
        DATA.add_data({'electric field': FIELDS.e1})
        DATA.add_data({'magnetic field': FIELDS.b2})
        # DATA.add_data({'weights': FIELDS.w})
        DATA.add_data(time_series)
        print()



    # ========================================================================================= 
    # Initialize Time Stepping Function
    # =========================================================================================
    if   params['time']['split_algo'] == 'LieTrotter':

        # set time steps for Lie-Trotter splitting
        dts_fields      = [ params['time']['dt'],
                            params['time']['dt']]
        dts_markers     = [ params['time']['dt'],
                            params['time']['dt']]
        dts_coupling    = [ params['time']['dt'],
                            params['time']['dt']]
        

    elif params['time']['split_algo'] == 'Strang':

        # set time steps for Strang splitting
        dts_markers     = [ params['time']['dt'],
                            params['time']['dt']/2] 
        dts_coupling    = [ params['time']['dt'],
                            params['time']['dt']/2.]
        dts_fields      = [ params['time']['dt'],
                            params['time']['dt']]  

    else:
        raise ValueError('Time stepping scheme not available.')



    UPDATE_FIELDS = push_maxwell.Push_maxwell(DOMAIN, SPACES, dts_fields, params)  
    print('Fields time stepping available.')  
    print() 

    UPDATE_MARKERS = push_markers.Push( dts_markers, DOMAIN, SPACES, KIN.Np_loc,
                                        params['kinetic_equilibrium']['general']
                                        )
    print('Marker time stepping available.')
    print()

    UPDATE_E_W = push_lin_VM.Push_lVM(  KIN.particles_loc,
                                        dts_coupling,
                                        DOMAIN, SPACES, KIN.Np_loc,
                                        ACCUM, MPI_COMM,
                                        v_shift, v_th, n0,
                                        params
                                        )
    print('Coupling time stepping available.')
    print()

    # set accuracy for particles in constant electric field
    accuracy = [1e-10, 1e-10]


    # get accuracy for particles in constant electric field
    accuracy = [params['solvers']['tol_x'],
                params['solvers']['tol_x']]

    maxiter  = params['solvers']['maxiter']



    def update():

        if params['time']['split_algo'] == 'LieTrotter':

            # substeps (Lie-Trotter splitting):

            # substep 1 for \fJ_1 of X-V subsystem;  
            UPDATE_MARKERS.step_in_const_efield(KIN.particles_loc, FIELDS.e1, accuracy, maxiter, print_info=True)

            # substep 2 for \fJ_2 of X-V subsystem
            UPDATE_MARKERS.step_v_cyclotron_ana(KIN.particles_loc, b2_eq, 0.*b2_eq, print_info=True)

            # W-e-b subsystem, step for \fJ_3 where bfield is constant
            UPDATE_E_W.step_e_W(KIN.particles_loc, FIELDS.e1, print_info=True)
            MPI_COMM.Bcast(FIELDS.e1, root=0)

            # W-e-b subsystem, step for \fJ_4 where weights are constant
            UPDATE_FIELDS.step_maxwell(FIELDS.e1, FIELDS.b2, print_info=True)
            MPI_COMM.Bcast(FIELDS.e1, root=0)
            MPI_COMM.Bcast(FIELDS.b2, root=0)

        elif params['time']['split_algo'] == 'Strang':

            # substeps (Strang splitting):

            # substep 1 for \fJ_1 of X-V subsystem; with half-size time-step  
            UPDATE_MARKERS.step_in_const_efield(KIN.particles_loc, FIELDS.e1, accuracy, maxiter, print_info=True)

            # substep 2 for \fJ_2 of X-V subsystem;  with half-size time-step
            UPDATE_MARKERS.step_v_cyclotron_ana(KIN.particles_loc, b2_eq, 0.*b2_eq, print_info=True)

            # W-e-b subsystem, step for \fJ_3 where bfield is constant;  with half-size time-step
            UPDATE_E_W.step_e_W(KIN.particles_loc, FIELDS.e1, print_info=True)
            MPI_COMM.Bcast(FIELDS.e1, root=0)

            # W-e-b subsystem, step for \fJ_4 where weights are constant;  with full-size time-step
            UPDATE_FIELDS.step_maxwell(FIELDS.e1, FIELDS.b2, print_info=True)
            MPI_COMM.Bcast(FIELDS.e1, root=0)
            MPI_COMM.Bcast(FIELDS.b2, root=0)

            # W-e-b subsystem, step for \fJ_3 where bfield is constant;  with half-size time-step
            UPDATE_E_W.step_e_W(KIN.particles_loc, FIELDS.e1, print_info=True)

            # substep 2 for \fJ_2 of X-V subsystem;  with half-size time-step
            UPDATE_MARKERS.step_v_cyclotron_ana(KIN.particles_loc, b2_eq, 0.*b2_eq, print_info=True)

            # substep 1 for \fJ_1 of X-V subsystem; with half-size time-step
            UPDATE_MARKERS.step_in_const_efield(KIN.particles_loc, FIELDS.e1, accuracy, maxiter, print_info=True)

        else:
            raise NotImplementedError('Only Lie-Trotter and Strang splitting available.')   

    print('Update function set.')
    print() 

        
    # =========================================================================================    
    # Time Integration 
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
        time_series['en_E'][0] = 1/2*FIELDS.e1.dot(SPACES.M1.dot(FIELDS.e1))
        time_series['en_B'][0] = 1/2*FIELDS.b2.dot(SPACES.M2.dot(FIELDS.b2))
        # time_series['en_W'][0] = 

        # save data:
        if mpi_rank == 0: DATA.save_data() 

        # print number of finished time steps and current energies
        if mpi_rank == 0 and time_steps_done%1 == 0:
            print('time steps finished : ' + str(time_steps_done) + '/'
                                           + str(int(round(params['time']['Tend'] / params['time']['dt'])))) 
            print()     
