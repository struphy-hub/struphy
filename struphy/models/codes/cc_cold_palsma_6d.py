#!/usr/bin/env python3

from multiprocessing import dummy
from traceback import print_tb
from mpi4py import MPI      
import yaml
import time
import numpy as np

from struphy.diagnostics                import data_module
from struphy.geometry                   import domain_3d
from struphy.mhd_equil                  import mhd_equil_physical 
from struphy.mhd_equil                  import mhd_equil_logical 
from struphy.mhd_equil.gvec             import mhd_equil_gvec
from struphy.kinetic_equil              import kinetic_equil_physical 
from struphy.kinetic_equil              import kinetic_equil_logical
from struphy.feec                       import spline_space
from struphy.mhd_init                   import emw_init 
from struphy.kinetic_init               import kinetic_init 
from struphy.feec.projectors.pro_global import emw_operators as emw_ops  
from struphy.pic.cc_cold_plasma_6d      import accumulation   
from struphy.models.substeps            import push_cold_plasma
from struphy.models.substeps            import push_markers
from struphy.models.substeps            import push_cc_cold_plasma


def execute(file_in, path_out, restart):
    '''Executes the code cc_lin_mhd_6d.

    Parameters
    ----------

    file_in : str
        Absolute path to input parameters file (.yml).
    path_out : str
        Absolute path to output folder.
    restart : boolean
        Restart ('True') or new simulation ('False').
    '''

    code_name = "cc_cold_plasma_6d"

    print()
    print('Starting code "' + code_name + '" ...')
    print()
    
    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_size = MPI_COMM.Get_size()
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # load simulation parameters
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
    # MHD EQUILIBRIUM object
    # =========================================================================================
    
    # COLD_PLASMA equilibirum (physical)
    equil_type      = params['equilibrium']['general']['type']
    equil_params    = params['equilibrium']['params_' + equil_type]
    EQ_FLUID_P      = mhd_equil_physical.Equilibrium_mhd_physical(equil_type, equil_params) 
    
    # MHD equilibrium (logical)
    if equil_type == 'gvec':
        EQ_FLUID_L      = mhd_equil_logical.Equilibrium_mhd_logical(DOMAIN, EQ_FLUID_P)   
    else:
        EQ_FLUID_L      = mhd_equil_logical.Equilibrium_mhd_logical(DOMAIN, EQ_FLUID_P)
        
    print('MHD equilibrium of type "' + equil_type + '" set.')
    print()

    # projection of magentic equilibrium
    B2_eq = [EQ_FLUID_L.b2_eq_1, EQ_FLUID_L.b2_eq_2, EQ_FLUID_L.b2_eq_3]
    b2_eq = SPACES.projectors.pi_2(B2_eq, include_bc=True, eval_kind='tensor_product', interp=True)

    # ========================================================================================= 
    # FLUID variables object
    # =========================================================================================
    # TODO: restart has to be done here
    init_type       = params['initialization']['general']['type']
    init_general    = params['initialization']['general']
    init_params     = params['initialization']['params_' + init_type]
    FLUID           = emw_init.Initialize_emw(DOMAIN, SPACES, init_general, init_params)
    print('FLUID variables of type "' + init_type + '" initialized.')
    print()
    
    # ========================================================================================= 
    # MHD PROJECTION OPERATORS object
    # =========================================================================================
    # TODO: add matrix-free version 
    FLUID_OPS = emw_ops.EMW_operators(DOMAIN, SPACES, EQ_FLUID_L) 
    
    FLUID_OPS.set_Operators()
    print('Assembly of FLUID projection operators done.')
    
    FLUID_OPS.set_Preconditioners(params['solvers']['PRE'])
    print('FLUID preconditioners of type "' + params['solvers']['PRE'] + '" set.')
    print()
    
    # ========================================================================================= 
    # KINETIC EQUILIBRIUM object
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
    # MARKER and ACCUMULATION objects
    # =========================================================================================
    # TODO: restart has to be done here
    KIN = kinetic_init.Initialize_markers(DOMAIN, EQ_KINETIC_L, 
                                        params['kinetic_init']['general'],
                                        params['kinetic_init']['params_' + params['kinetic_init']['general']['type']],
                                        params['markers'], MPI_COMM
                                        )
    print(KIN.Np_loc, 'markers initialized on rank', mpi_rank)
    print()

    # create particle accumulator (all processes)
    ACCUM = accumulation.Accumulation(SPACES, DOMAIN, params['initialization']['general']['basis_u'], 
                                            MPI_COMM, params['markers']['control'], EQ_KINETIC_L)
    print('Accumulator initialized on rank', mpi_rank)
    print()

    # ========================================================================================= 
    # DATA object for saving
    # =========================================================================================
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_E' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    'en_J' : np.empty(1, dtype=float), 
                    'en_f' : np.empty(1, dtype=float),
                    #'divB' : np.empty(1, dtype=float),
                    #'momentum' : np.empty(1, dtype=float),
                    #'helicity' : np.empty(1, dtype=float),
                    #'cr_helicity' : np.empty(1, dtype=float),
                    }
 
    time_series['time'][0] = 0.
    time_series['en_E'][0] = 1/2*FLUID.e1.dot(SPACES.M1.dot(FLUID.e1))
    time_series['en_B'][0] = 1/2*FLUID.b2.dot(SPACES.M2.dot(FLUID.b2))
    time_series['en_J'][0] = 1/2*FLUID.j1.dot(SPACES.M1.dot(FLUID.j1))

    energies_loc = {'en_f' : np.empty(1, dtype=float)}
    energies_loc['en_f'][0] = KIN.particles_loc[6].dot(KIN.particles_loc[3]**2 
                                                     + KIN.particles_loc[4]**2 
                                                     + KIN.particles_loc[5]**2
                                                     ) / (2.*KIN.Np)
    MPI_COMM.Reduce(energies_loc['en_f'], time_series['en_f'], op=MPI.SUM, root=0)
        
    # snapshots of distribution function via 2d particle binning
    n_bins_x  = params['markers']['n_bins'][0]
    n_bins_v  = params['markers']['n_bins'][1]
    bin_edges = [np.linspace(0., 1., n_bins_x + 1), np.linspace(-params['markers']['v_max'], params['markers']['v_max'], n_bins_v + 1)]
    dbin      = [bin_edges[0][1] - bin_edges[0][0], bin_edges[1][1] - bin_edges[1][0]]     
    fh_loc    = np.empty((n_bins_x, n_bins_v), dtype=float)
    fh        = fh_loc.copy()

    # initial distribution function in (eta1, vx)-plane
    fh_loc[:, :] = np.histogram2d(KIN.particles_loc[0], KIN.particles_loc[3], bins=bin_edges,
                                  weights=KIN.particles_loc[6], normed=False)[0] / (KIN.Np*dbin[0]*dbin[1]
                                  )
    MPI_COMM.Reduce(fh_loc, fh, op=MPI.SUM, root=0)

    # create object for data saving (only rank 0)
    if mpi_rank == 0:
        DATA = data_module.Data_container(path_out)
        print('Data object initialized.')
        print()
        # add other mhd variabels to data object (flattened) for saving
        DATA.add_data({'electric field': FLUID.e1})
        DATA.add_data({'magnetic field': FLUID.b2})
        DATA.add_data({'cold current': FLUID.j1})
        DATA.add_data({'fh_binned': fh})
        DATA.add_data(time_series)
        print()

        # print initial time_series to screen
        DATA.print_data_to_screen(['en_E', 'en_B', 'en_J', 'en_f'])
        print()

    # array of all particles for saving:
    if mpi_rank == 0:    
        particles = np.empty((7, KIN.Np), dtype=float)
        
        for i in range(1, mpi_size):
            MPI_COMM.Recv(KIN.particles_recv, source=i, tag=11)
            # MPI_COMM.Recv(KIN.w0_recv       , source=i, tag=12)
            # MPI_COMM.Recv(KIN.s0_recv       , source=i, tag=13)
            
            particles[:, i*KIN.Np_loc : (i + 1)*KIN.Np_loc] = KIN.particles_recv 

        DATA.add_data({'particles': particles})
        print()
    else:
        MPI_COMM.Send(KIN.particles_loc, dest=0, tag=11)

    # ========================================================================================= 
    # Initialize time stepping function
    # =========================================================================================
    if   params['time']['split_algo'] == 'LieTrotter':

        # set time steps for Lie-Trotter splitting
        dts_fluid   = [params['time']['dt'],
                       params['time']['dt']]
        dts_markers = [params['time']['dt'],
                       params['time']['dt']]
        dts_cc      = [params['time']['dt'],
                       params['time']['dt']]

    elif params['time']['split_algo'] == 'Strang':

        # set time steps for Strang splitting
        dts_fluid   = [params['time']['dt']/2.,
                       params['time']['dt']/2.]
        dts_markers = [params['time']['dt']/2.,
                       params['time']['dt']/2.]
        dts_cc      = [params['time']['dt'],
                       params['time']['dt']/2.]  

    else:
        raise ValueError('Time stepping scheme not available.')
        
        
    print(params['initialization']['general']['basis_u'], params['initialization']['general']['basis_p'])
    

    UPDATE_FLUID = push_cold_plasma.Push_cold_plasma(DOMAIN, SPACES, FLUID_OPS, dts_fluid,  params)  
    print('FIELDS time stepping available.')  
    print() 

    UPDATE_MARKERS = push_markers.Push(dts_markers, DOMAIN, SPACES, KIN.Np_loc, params['kinetic_equilibrium']['general'])
    print('Marker time stepping available.')
    print()

    UPDATE_CC = push_cc_cold_plasma.Current_coupling(dts_cc, DOMAIN, SPACES, FLUID_OPS, ACCUM, MPI_COMM, KIN.Np, KIN.Np_loc, 
                                                                 params['solvers'], params['kinetic_equilibrium']['general'],
                                                                 params['initialization']['general']['basis_u'])
    print('Current coupling time stepping available.')
    print()

    def update():

        if params['time']['split_algo'] == 'LieTrotter':

            UPDATE_FLUID.step_maxwell(FLUID.e1, FLUID.b2, print_info=params['solvers']['show_info'])
            UPDATE_FLUID.step_analytic(FLUID.e1, FLUID.j1, print_info=params['solvers']['show_info'])
            UPDATE_FLUID.step_rotation(FLUID.j1, print_info=params['solvers']['show_info'])
            MPI_COMM.Bcast(FLUID.e1, root=0)
            MPI_COMM.Bcast(FLUID.b2, root=0)
            MPI_COMM.Bcast(FLUID.j1, root=0)
            MPI_COMM.Barrier()

            # dummy = np.sum(KIN.particles_loc)
            # print('particels: step_0,  mpi_rank:', mpi_rank, 'nans:', np.isnan(dummy))
            UPDATE_MARKERS.step_v_cyclotron_ana(KIN.particles_loc, b2_eq, FLUID.b2, print_info=True)
            UPDATE_MARKERS.step_eta_RK4(KIN.particles_loc, print_info=True)
            UPDATE_CC.step_jh(FLUID.j1, KIN.particles_loc, b2_eq, FLUID.b2, print_info=True)
            MPI_COMM.Bcast(FLUID.e1, root=0)
            if params['markers']['control']: KIN.update_weights()
            
        else:
            raise NotImplementedError('Only Lie-Trotter and Strang splitting available.')   
    
        # update binned distribution function
        fh_loc[:, :] = np.histogram2d(KIN.particles_loc[0], KIN.particles_loc[3], bins=bin_edges,
                                    weights=KIN.particles_loc[6], normed=False)[0] / (KIN.Np*dbin[0]*dbin[1])
        MPI_COMM.Reduce(fh_loc, fh, op=MPI.SUM, root=0)

    print('Update function set.', mpi_rank)
    print() 
        
    # =========================================================================================    
    # time integration 
    # =========================================================================================
    if mpi_rank == 0:
        print('Start time integration: ' + params['time']['split_algo'])
        print()

    # if mpi_rank == 0: 
    #     if restart:
    #         time_steps_done = DATA.file['time'].size
    #     else:
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
                print('time of simulation [sec]: ', end_simulation - start_simulation)

            break
            
        # call update function for time stepping
        update()
        time_steps_done += 1 
        if mpi_rank == 0:
            print('E_e  :', time_series['en_E'][0])
            print('E_b  :', time_series['en_B'][0])
            print('E_j  :', time_series['en_J'][0])
            print('E_f  :', time_series['en_f'], 'rank')
            print('E_ges:', time_series['en_E'][0] + time_series['en_B'][0] + time_series['en_J'][0] + time_series['en_f'][0])

        # update time series
        time_series['time'][0]  = 0.
        time_series['en_E'][0]  = 1/2*FLUID.e1.dot(SPACES.M1.dot(FLUID.e1))
        time_series['en_B'][0]  = 1/2*FLUID.b2.dot(SPACES.M2.dot(FLUID.b2))
        time_series['en_J'][0]  = 1/2*FLUID.j1.dot(SPACES.M1.dot(FLUID.j1))

        energies_loc['en_f'][0] = KIN.particles_loc[6].dot(KIN.particles_loc[3]**2 
                                                         + KIN.particles_loc[4]**2 
                                                         + KIN.particles_loc[5]**2
                                                          ) / (2.*KIN.Np)
        MPI_COMM.Reduce(energies_loc['en_f'], time_series['en_f'], op=MPI.SUM, root=0)


        # gather particles for saving
        if mpi_rank == 0:    
            for i in range(1, mpi_size):
                MPI_COMM.Recv(KIN.particles_recv, source=i, tag=11)
                # MPI_COMM.Recv(KIN.w0_recv       , source=i, tag=12)
                # MPI_COMM.Recv(KIN.s0_recv       , source=i, tag=13)
                
                particles[:, i*KIN.Np_loc : (i + 1)*KIN.Np_loc] = KIN.particles_recv 
        else:
            MPI_COMM.Send(KIN.particles_loc, dest=0, tag=11)

        # save data:
        if mpi_rank == 0: DATA.save_data() 

        # print number of finished time steps and current energies
        if mpi_rank == 0 and time_steps_done%1 == 0:
            print('time steps finished : ' + str(time_steps_done) + '/'
                                           + str(int(round(params['time']['Tend'] / params['time']['dt'])))) 
            print() 

             