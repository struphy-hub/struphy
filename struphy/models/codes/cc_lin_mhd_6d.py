#!/usr/bin/env python3

from mpi4py import MPI      
import yaml
import time
import numpy as np

from struphy.diagnostics                import data_module
from struphy.geometry                   import domain_3d
from struphy.mhd_equil                  import mhd_equil_physical 
from struphy.mhd_equil                  import mhd_equil_logical 
from struphy.kinetic_equil              import kinetic_equil_physical 
from struphy.kinetic_equil              import kinetic_equil_logical
from struphy.feec                       import spline_space
from struphy.mhd_init                   import mhd_init 
from struphy.kinetic_init               import kinetic_init 
from struphy.feec.projectors.pro_global import mhd_operators_cc_lin_6d as mhd_ops
from struphy.pic                        import accumulation   
from struphy.models.substeps            import push_linear_mhd
from struphy.models.substeps            import push_markers
from struphy.models.substeps            import push_cc


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

    print()
    print('Starting code "cc_lin_mhd_6d" ...')
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
    DOMAIN   = domain_3d.Domain(domain_type, params['geometry']['params_' + domain_type])
    print('Domain object of type "' + domain_type + '" set.')
    print()

    # ========================================================================================= 
    # MHD EQUILIBRIUM object
    # =========================================================================================
    # MHD equilibirum (physical)
    mhd_equil_type = params['mhd_equilibrium']['general']['type']
    EQ_MHD_P = mhd_equil_physical.Equilibrium_mhd_physical(mhd_equil_type, params['mhd_equilibrium']['params_' + mhd_equil_type])
    
    # MHD equilibrium (logical)
    EQ_MHD_L = mhd_equil_logical.Equilibrium_mhd_logical(DOMAIN, EQ_MHD_P)
    print('MHD equilibrium of type "' + mhd_equil_type + '" set.')
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

    # Replace `EQ_MHD_L` if a GVEC equilibrium is requested, depending on params.
    if params['mhd_equilibrium']['general']['type'] == 'gvec':
        EQ_MHD_L = mhd_equil_gvec.Equilibrium_mhd_gvec(params, SPACES, DOMAIN)
        print('GVEC MHD equilibrium (logical) set.')

    # ======= reserve memory for FEM cofficients (all MPI processes) ========
    NbaseN  = SPACES.NbaseN
    NbaseD  = SPACES.NbaseD

    N_0form = SPACES.Nbase_0form
    N_1form = SPACES.Nbase_1form
    N_2form = SPACES.Nbase_2form
    N_3form = SPACES.Nbase_3form

    # N_dof_all_0form = SPACES.E0_all.shape[0]
    # N_dof_all_1form = SPACES.E1_all.shape[0]
    # N_dof_all_2form = SPACES.E2_all.shape[0]
    # N_dof_all_3form = SPACES.E3_all.shape[0]

    N_dof_0form = SPACES.E0.shape[0]
    N_dof_1form = SPACES.E1.shape[0]
    N_dof_2form = SPACES.E2.shape[0]
    N_dof_3form = SPACES.E3.shape[0]

    r3 = np.zeros(N_dof_3form, dtype=float)
    b2 = np.zeros(N_dof_2form, dtype=float)

    if   params['forms']['basis_p'] == 0:
        pp = np.zeros(N_dof_0form, dtype=float)
    elif params['forms']['basis_p'] == 3:
        pp = np.zeros(N_dof_3form, dtype=float)

    if params['forms']['basis_u'] == 1:
        up     = np.zeros(N_dof_1form, dtype=float)
        up_old = np.zeros(N_dof_1form, dtype=float)
    # elif   params['forms']['basis_u'] == 0:
    #     up     = np.zeros(N_dof_0form + 2*N_dof_all_0form, dtype=float)
    #     up_old = np.zeros(N_dof_0form + 2*N_dof_all_0form, dtype=float)
    elif params['forms']['basis_u'] == 2:
        up     = np.zeros(N_dof_2form, dtype=float)
        up_old = np.zeros(N_dof_2form, dtype=float)
    # =======================================================================

    # initialize mhd variables 
    MHD_ini = mhd_init.Initialize_mhd(DOMAIN, SPACES, file_in)
    MHD_ini.initialize(r3, pp, b2, up) 
    # equilibrium magn. field and current
    J2_eq = [EQ_MHD_L.j2_eq_1, EQ_MHD_L.j2_eq_2, EQ_MHD_L.j2_eq_3]
    B2_eq = [EQ_MHD_L.b2_eq_1, EQ_MHD_L.b2_eq_2, EQ_MHD_L.b2_eq_3]
    # assemble mass matrices 
    SPACES.assemble_M2(DOMAIN)
    SPACES.assemble_M3(DOMAIN)
    print('Assembly of mass matrices done.')
    print()

    # preconditioner 
    if params['solvers']['PRE'] == 'ILU':
        SPACES.projectors.assemble_approx_inv(params['solvers']['tol_inv'])

    # projection of magentic equilibrium
    B2_eq = [EQ_MHD_L.b2_eq_1, EQ_MHD_L.b2_eq_2, EQ_MHD_L.b2_eq_3]
    b2_eq = SPACES.projectors.pi_2(B2_eq, include_bc=True, eval_kind='tensor_product', interp=True)

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

    # pre conditioners
    MHD_OPS.setInverseA()
    MHD_OPS.setPreconditionerS2(params['solvers']['PRE'])
    MHD_OPS.setPreconditionerS6(params['solvers']['PRE'])
    print('MHD preconditioners of type "' + params['solvers']['PRE'] + '" set.')
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
    ACCUM = accumulation.Accumulation(SPACES, DOMAIN, params['mhd_init']['general']['basis_u'], 
                                            MPI_COMM, params['markers']['control'], EQ_KINETIC_L)
    print('Accumulator initialized on rank', mpi_rank)
    print()

    # ========================================================================================= 
    # DATA object for saving
    # =========================================================================================
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_U' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    'en_p' : np.empty(1, dtype=float), 
                    'en_f' : np.empty(1, dtype=float),
                    #'divB' : np.empty(1, dtype=float),
                    #'momentum' : np.empty(1, dtype=float),
                    #'helicity' : np.empty(1, dtype=float),
                    #'cr_helicity' : np.empty(1, dtype=float),
                    }
 
    time_series['time'][0] = 0.
    time_series['en_U'][0] = 1/2*MHD.up.dot(MHD_OPS.A(MHD.up))
    time_series['en_B'][0] = 1/2*MHD.b2.dot(SPACES.M2.dot(MHD.b2))
    time_series['en_p'][0] = 1/(5./3. - 1.)*sum(MHD.pp.flatten())

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
        DATA.add_data({'density': MHD.r3})
        DATA.add_data({'mhd velocity': MHD.up})
        DATA.add_data({'magnetic field': MHD.b2})
        DATA.add_data({'pressure': MHD.pp})
        DATA.add_data({'fh_binned': fh})
        DATA.add_data(time_series)
        print()

        # print initial time_series to screen
        DATA.print_data_to_screen(['en_U', 'en_B', 'en_p', 'en_f'])
        print()

    # array of all particles for saving:
    if mpi_rank == 0:    
        particles = np.empty((7, KIN.Np), dtype=float)
        
        for i in range(1, mpi_size):
            MPI_COMM.Recv(KIN.particles_recv, source=i, tag=11)
            MPI_COMM.Recv(KIN.w0_recv       , source=i, tag=12)
            MPI_COMM.Recv(KIN.s0_recv       , source=i, tag=13)
            
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
        dts_mhd     = [params['time']['dt'],
                       params['time']['dt']]
        dts_markers = [params['time']['dt'],
                       params['time']['dt']]
        dts_cc      = [params['time']['dt'],
                       params['time']['dt']]

    elif params['time']['split_algo'] == 'Strang':

        # set time steps for Strang splitting
        dts_mhd     = [params['time']['dt']/2.,
                       params['time']['dt']/2.]
        dts_markers = [params['time']['dt']/2.,
                       params['time']['dt']/2.]
        dts_cc      = [params['time']['dt'],
                       params['time']['dt']/2.]  

    else:
        raise ValueError('Time stepping scheme not available.')

    UPDATE_MHD = push_linear_mhd.Linear_mhd(dts_mhd, SPACES, MHD_OPS, params['solvers'], 
                                                                 params['mhd_init']['general']['basis_u'], 
                                                                 params['mhd_init']['general']['basis_p'])  
    print('MHD time stepping available.')  
    print() 

    UPDATE_MARKERS = push_markers.Push(dts_markers, DOMAIN, SPACES, KIN.Np_loc, params['kinetic_equilibrium']['general'])
    print('Marker time stepping available.')
    print()

    UPDATE_CC = push_cc.Current_coupling(dts_cc, DOMAIN, SPACES, MHD_OPS, ACCUM, MPI_COMM, KIN.Np, KIN.Np_loc, 
                                                                 params['solvers'], params['kinetic_equilibrium']['general'],
                                                                 params['mhd_init']['general']['basis_u'])
    print('Current coupling time stepping available.')
    print()

    def update():

        if params['time']['split_algo'] == 'LieTrotter':

            # substeps (Lie-Trotter splitting):
            UPDATE_CC.step_rhoh(MHD.up, KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0) # TODO: put Bcast in step_ function

            UPDATE_MHD.step_alfven(MHD.up, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0)
            MPI_COMM.Bcast(MHD.b2, root=0)

            UPDATE_CC.step_jh(MHD.up, KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0)

            UPDATE_MARKERS.step_eta_RK4(KIN.particles_loc, print_info=True)

            UPDATE_MHD.step_magnetosonic(MHD.r3, MHD.up, MHD.b2, MHD.pp, print_info=True)
            MPI_COMM.Bcast(MHD.r3, root=0)
            MPI_COMM.Bcast(MHD.up, root=0)
            MPI_COMM.Bcast(MHD.up, root=0)
            
            UPDATE_MARKERS.step_v_cyclotron_ana(KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            if params['markers']['control']: KIN.update_weights()

        elif params['time']['split_algo'] == 'Strang':

            # substeps (Strang splitting):
            UPDATE_MHD.step_alfven(MHD.up, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0)
            MPI_COMM.Bcast(MHD.b2, root=0)

            UPDATE_MHD.step_magnetosonic(MHD.r3, MHD.up, MHD.b2, MHD.pp, print_info=True)
            MPI_COMM.Bcast(MHD.r3, root=0)
            MPI_COMM.Bcast(MHD.up, root=0)
            MPI_COMM.Bcast(MHD.up, root=0)

            UPDATE_MARKERS.step_eta_RK4(KIN.particles_loc, print_info=True)

            UPDATE_MARKERS.step_v_cyclotron_ana(KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            if params['markers']['control']: KIN.update_weights()

            UPDATE_CC.step_rhoh(MHD.up, KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0) # TODO: put Bcast in step_ function?

            UPDATE_CC.step_jh(MHD.up, KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0)
            if params['markers']['control']: KIN.update_weights()

            UPDATE_CC.step_rhoh(MHD.up, KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0) # TODO: put Bcast in step_ function?

            UPDATE_MARKERS.step_v_cyclotron_ana(KIN.particles_loc, b2_eq, MHD.b2, print_info=True)
            if params['markers']['control']: KIN.update_weights()

            UPDATE_MARKERS.step_eta_RK4(KIN.particles_loc, print_info=True)

            UPDATE_MHD.step_magnetosonic(MHD.r3, MHD.up, MHD.b2, MHD.pp, print_info=True)
            MPI_COMM.Bcast(MHD.r3, root=0)
            MPI_COMM.Bcast(MHD.up, root=0)
            MPI_COMM.Bcast(MHD.up, root=0)

            UPDATE_MHD.step_alfven(MHD.up, MHD.b2, print_info=True)
            MPI_COMM.Bcast(MHD.up, root=0)
            MPI_COMM.Bcast(MHD.b2, root=0)

        else:
            raise NotImplementedError('Only Lie-Trotter and Strang splitting available.')   
    
        # update binned distribution function
        fh_loc[:, :] = np.histogram2d(KIN.particles_loc[0], KIN.particles_loc[3], bins=bin_edges,
                                    weights=KIN.particles_loc[6], normed=False)[0] / (KIN.Np*dbin[0]*dbin[1])
        MPI_COMM.Reduce(fh_loc, fh, op=MPI.SUM, root=0)

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

        # update time series
        time_series['time'][0] = 0.
        time_series['en_U'][0] = 1/2*MHD.up.dot(MHD_OPS.A(MHD.up))
        time_series['en_B'][0] = 1/2*MHD.b2.dot(SPACES.M2.dot(MHD.b2))
        time_series['en_p'][0] = 1/(5./3. - 1.)*sum(MHD.pp.flatten())

        energies_loc['en_f'][0] = KIN.particles_loc[6].dot(KIN.particles_loc[3]**2 
                                                         + KIN.particles_loc[4]**2 
                                                         + KIN.particles_loc[5]**2
                                                          ) / (2.*KIN.Np)
        MPI_COMM.Reduce(energies_loc['en_f'], time_series['en_f'], op=MPI.SUM, root=0)

        # gather particles for saving
        if mpi_rank == 0:    
            for i in range(1, mpi_size):
                MPI_COMM.Recv(KIN.particles_recv, source=i, tag=11)
                MPI_COMM.Recv(KIN.w0_recv       , source=i, tag=12)
                MPI_COMM.Recv(KIN.s0_recv       , source=i, tag=13)
                
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

             