#!/usr/bin/env python3

from mpi4py import MPI      
import yaml
import time
import numpy as np

from struphy.diagnostics                         import data_module
from struphy.geometry                            import domain_3d

from struphy.mhd_equil                           import mhd_equil_physical 
from struphy.mhd_equil                           import mhd_equil_logical 
from struphy.mhd_equil.gvec                      import mhd_equil_gvec
from struphy.kinetic_equil                       import kinetic_equil_physical 
from struphy.kinetic_equil                       import kinetic_equil_logical

from struphy.feec                                import spline_space

from struphy.mhd_init.kinetic_extended           import mhd_init 
from struphy.kinetic_init.kinetic_extended       import kinetic_init


from struphy.pic.kinetic_extended                import fB_massless_accumulation   
from struphy.pic.kinetic_extended                import fB_particle_gather
from struphy.pic.kinetic_extended                import fB_shape
from struphy.pic.kinetic_extended                import fB_energy

from struphy.models.substeps.kinetic_extended    import fB_bb
from struphy.models.substeps.kinetic_extended    import fB_bv
from struphy.models.substeps.kinetic_extended    import fB_vv
from struphy.models.substeps.kinetic_extended    import fB_v_rot
from struphy.models.substeps.kinetic_extended    import fB_xv

from struphy.feec.massless_operators             import fB_arrays
from struphy.feec.massless_operators             import fB_massless_linear_operators
from struphy.feec.basics                         import mass_matrices_3d_pre




# =========== 
def execute(file_in, path_out, restart):
    '''Executes the code kinetic_extended.

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
    print('Starting code "kinetic_extended" ...')
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
    # FEEC SPACES object
    # =========================================================================================
    Nel         = params['grid']['Nel']
    p           = params['grid']['p']
    spl_kind    = params['grid']['spl_kind']
    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0], params['grid']['nq_el'][0]) 
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
    SPACES.assemble_M0(DOMAIN)
    SPACES.assemble_M1(DOMAIN)
    SPACES.assemble_M2(DOMAIN)
    SPACES.assemble_M3(DOMAIN)
    print('Assembly of mass matrices done.')
    print()

    # preconditioner 
    if params['solvers']['PRE'] == 'ILU':
        SPACES.projectors.assemble_approx_inv(params['solvers']['tol_inv'])
    elif params['solvers']['PRE'] == 'FFT':
        M0_PRE = mass_matrices_3d_pre.get_M0_PRE(SPACES, DOMAIN)
        M1_PRE = mass_matrices_3d_pre.get_M1_PRE(SPACES, DOMAIN)
        M2_PRE = mass_matrices_3d_pre.get_M2_PRE(SPACES, DOMAIN)
        M3_PRE = mass_matrices_3d_pre.get_M3_PRE(SPACES, DOMAIN)


    # ========================================================================================= 
    # MHD EQUILIBRIUM object
    # =========================================================================================
    # MHD equilibirum (physical)
    mhd_equil_type = params['mhd_equilibrium']['general']['type']
    EQ_MHD_P = mhd_equil_physical.Equilibrium_mhd_physical(mhd_equil_type, params['mhd_equilibrium']['params_' + mhd_equil_type])
    
    # MHD equilibrium (logical)
    if mhd_equil_type == 'gvec':
        EQ_MHD_L = mhd_equil_gvec.Equilibrium_mhd_gvec(params, SPACES, DOMAIN)
    else:
        EQ_MHD_L = mhd_equil_logical.Equilibrium_mhd_logical(DOMAIN, EQ_MHD_P)
        
    print('MHD equilibrium of type "' + mhd_equil_type + '" set.')
    print()

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
    # ========================================================================================= 
    # INITIALIZATION of particle ACCUMULATOR (all processes) and LINEAR OPERATORS object
    # =========================================================================================
    GATHER   = fB_particle_gather.gather(SPACES, DOMAIN, KIN.Np_loc, params['markers']['p_shape'], params['markers']['p_size'], params['markers']['tol_gather'], MPI_COMM, params['markers']['control'])
    ACCUM4   = fB_massless_accumulation.accumulation(SPACES, DOMAIN, 4, KIN.Np_loc, MPI_COMM, params['markers']['control'], params['markers']['p_shape'], params['markers']['p_size'])
    ACCUMVV  = fB_massless_accumulation.accumulation(SPACES, DOMAIN, 2, KIN.Np_loc, MPI_COMM, params['markers']['control'], params['markers']['p_shape'], params['markers']['p_size'])
    SHAPE    = fB_shape.Shape(MPI_COMM, params['markers']['p_shape'], params['markers']['p_size'])
    TEMP     = fB_arrays.Temp_arrays(SPACES, DOMAIN, params['markers']['control'], MPI_COMM)
    ENERGY   = fB_energy.Energy(params['temperature'], DOMAIN, SPACES, GATHER, KIN, MHD, TEMP, MPI_COMM)
    LINEAR_OPERATORS = fB_massless_linear_operators.Massless_linear_operators(SPACES, DOMAIN, KIN)
    print('Accumulator and Linear Operators initialized on rank', mpi_rank)
    print()


    # DATA object for saving
    # =========================================================================================
    time_series  = {'time' : np.empty(1, dtype=float),
                    'en_T' : np.empty(1, dtype=float), 
                    'en_B' : np.empty(1, dtype=float), 
                    'en_f' : np.empty(1, dtype=float),
                    #'divB' : np.empty(1, dtype=float),
                    #'momentum' : np.empty(1, dtype=float),
                    #'helicity' : np.empty(1, dtype=float),
                    #'cr_helicity' : np.empty(1, dtype=float),
                    }
    # initial energy 
    ENERGY.cal_total(KIN, MPI_COMM)
    time_series['time'][0] = 0.
    time_series['en_T'][0] = ENERGY.temperature * ENERGY.thermal[0] 
    time_series['en_B'][0] = ENERGY.magnetic[0]
    time_series['en_f'][0] = ENERGY.kinetic[0]
        
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
        DATA.add_data({'magnetic field first component':  MHD.b1})
        DATA.add_data({'magnetic field second component': MHD.b2})
        DATA.add_data({'magnetic field third component':  MHD.b3})
        DATA.add_data({'density': MHD.n0})
        DATA.add_data({'fh_binned': fh})
        DATA.add_data(time_series)
        print()

        # print initial time_series to screen
        DATA.print_data_to_screen(['en_T', 'en_B', 'en_f'])
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
    SUBSTEP1 = fB_xv.Substep_1(DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, params['temperature'], SHAPE, TEMP)
    SUBSTEP2 = fB_vv.Substep_2(LINEAR_OPERATORS, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, params['temperature'], SHAPE, TEMP, ACCUMVV, params['markers']['control'], M2_PRE, M1_PRE)
    SUBSTEP3 = fB_bb.Substep_3(LINEAR_OPERATORS, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, params['temperature'], SHAPE, TEMP, params['markers']['control'])
    SUBSTEP4 = fB_bv.Substep_4(LINEAR_OPERATORS, DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM, params['temperature'], SHAPE, TEMP, params['markers']['control'], ACCUM4)
    SUBSTEP5 = fB_v_rot.Substep_5(DOMAIN, SPACES, GATHER, KIN, MHD, MPI_COMM)
    
    print('Substeps available.') 
    print()  


    def update():

        if params['time']['split_algo'] == 'LieTrotter':

            # substeps (Lie-Trotter splitting):
            #SUBSTEP1.push(params['time']['dt'], ENERGY, KIN, MPI_COMM)

            #SUBSTEP5.push(params['time']['dt'], ENERGY, KIN, MPI_COMM)
            SUBSTEP2.push_proj_RK2(params['time']['dt'], ENERGY, KIN, params['markers']['tol_gather'], MPI_COMM)

            #SUBSTEP3.update(params['time']['dt'], ENERGY, KIN, M1_PRE, MPI_COMM)

            #SUBSTEP4.update(params['time']['dt'], M1_PRE, ENERGY, KIN, MPI_COMM)

            

            if params['markers']['control']: KIN.update_weights()

        else:
            raise NotImplementedError('Only Lie-Trotter splitting available.')   
    
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
            
        # call update energy
        update()
        ENERGY.cal_total(KIN, MPI_COMM)
        time_steps_done += 1 

        # update time series
        time_series['time'][0] = 0.
        time_series['en_T'][0] = ENERGY.temperature * ENERGY.thermal[0] 
        time_series['en_B'][0] = ENERGY.magnetic[0]
        time_series['en_f'][0] = ENERGY.kinetic[0]

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

             