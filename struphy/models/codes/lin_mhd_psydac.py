#!/usr/bin/env python3

from mpi4py import MPI      
import yaml
import time
import numpy as np

from struphy.geometry.domain_3d            import Domain
from struphy.feec.spline_space             import Spline_space_1d 
from struphy.mhd_equil.mhd_equil_physical  import Equilibrium_mhd_physical
from struphy.mhd_equil.mhd_equil_logical   import Equilibrium_mhd_logical
from struphy.mhd_equil.gvec.mhd_equil_gvec import Equilibrium_mhd_gvec
from struphy.mhd_init.mhd_init             import Initialize_mhd_psydac

from struphy.diagnostics                   import data_module

from struphy.psydac_api.mhd_ops         import MHD_ops  
from struphy.models.substeps            import push_linear_mhd

from sympde.topology import Cube, Derham
from sympde.topology import elements_of
from sympde.expr     import BilinearForm, integral
from sympde.calculus import dot

from psydac.api.discretization import discretize
from psydac.api.settings       import PSYDAC_BACKEND_PYTHON, PSYDAC_BACKEND_GPYCCEL


def execute(file_in, path_out, restart):
    '''Executes the code lin_mhd_psydac.

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
    print('Starting code "lin_mhd_psydac" ...')
    print()

    # mpi communicator
    MPI_COMM = MPI.COMM_WORLD
    mpi_rank = MPI_COMM.Get_rank()
    MPI_COMM.Barrier()

    # load simulation parameters
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    #####################
    ### DOMAIN object ###
    #####################
    map_type    = params['geometry']['type']
    map_params  = params['geometry']['params_' + map_type]

    # Struphy domain object
    DOMAIN = Domain(map_type, map_params)

    Mapping_psydac = DOMAIN.Psydac_mapping('F', **map_params)
    print(Mapping_psydac._expressions, '\n')

    # Psydac mapping
    F_PSY = Mapping_psydac.get_callable_mapping()

    # Psydac domain object
    DOMAIN_PSYDAC_LOGICAL = Cube('C', bounds1=(0, 1), bounds2=(0, 1), bounds3=(0, 1))
    DOMAIN_symb = Mapping_psydac(DOMAIN_PSYDAC_LOGICAL)

    print('Domain object of type "' + map_type + '" set.')
    print()
    
    ##########################
    ### FEEC SPACES object ###
    ##########################
    Nel         = params['grid']['Nel']
    p           = params['grid']['p']
    spl_kind    = params['grid']['spl_kind']
    nq_el      = params['grid']['nq_el']
    nq_pr      = params['grid']['nq_pr']
    bc          = params['grid']['bc']

    # Psydac De Rham
    DERHAM_symb = Derham(DOMAIN_symb)
    DOMAIN_PSY  = discretize(DOMAIN_symb, ncells=Nel, comm=MPI_COMM)
    DERHAM  = discretize(DERHAM_symb, DOMAIN_PSY, degree=p, periodic=spl_kind)

    # Psydac derivative operators
    grad, curl, div = DERHAM.derivatives_as_matrices

    # Psydac mass matrices
    u0, v0 = elements_of(DERHAM_symb.V0, names='u0, v0')
    u1, v1 = elements_of(DERHAM_symb.V1, names='u1, v1')
    u2, v2 = elements_of(DERHAM_symb.V2, names='u2, v2')
    u3, v3 = elements_of(DERHAM_symb.V3, names='u3, v3')

    a0 = BilinearForm((u0, v0), integral(DOMAIN_symb, u0*v0))
    a1 = BilinearForm((u1, v1), integral(DOMAIN_symb, dot(u1, v1)))
    a2 = BilinearForm((u2, v2), integral(DOMAIN_symb, dot(u2, v2)))
    a3 = BilinearForm((u3, v3), integral(DOMAIN_symb, u3*v3))

    a0_h = discretize(a0, DOMAIN_PSY, (DERHAM.V0, DERHAM.V0), backend=PSYDAC_BACKEND_GPYCCEL)
    a1_h = discretize(a1, DOMAIN_PSY, (DERHAM.V1, DERHAM.V1), backend=PSYDAC_BACKEND_GPYCCEL)
    a2_h = discretize(a2, DOMAIN_PSY, (DERHAM.V2, DERHAM.V2), backend=PSYDAC_BACKEND_GPYCCEL)
    a3_h = discretize(a3, DOMAIN_PSY, (DERHAM.V3, DERHAM.V3), backend=PSYDAC_BACKEND_GPYCCEL)

    M0 = a0_h.assemble()
    M1 = a1_h.assemble()
    M2 = a2_h.assemble()
    M3 = a3_h.assemble()

    # Psydac projection operators
    P0, P1, P2, P3 = DERHAM.projectors(nquads=nq_pr)

    print('Psydac Derham set.\n')

    # Struphy spline spaces (for 1d projectors)
    space_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0]) 
    space_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1])
    space_3 = Spline_space_1d(Nel[2], p[2], spl_kind[2], nq_el[2])

    space_1.set_projectors(nq_pr[0]) 
    space_2.set_projectors(nq_pr[1])
    space_3.set_projectors(nq_pr[2])

    projectors_1d = (space_1.projectors, space_2.projectors, space_3.projectors)

    ##############################
    ### MHD EQUILIBRIUM object ###
    ##############################
    # MHD equilibirum (physical)
    mhd_equil_type   = params['mhd_equilibrium']['general']['type']
    mhd_equil_params = params['mhd_equilibrium']['params_' + mhd_equil_type]

    EQ_MHD_P = Equilibrium_mhd_physical(mhd_equil_type, mhd_equil_params)
    
    # MHD equilibrium (logical)
    if mhd_equil_type == 'gvec':
        EQ_MHD_L = Equilibrium_mhd_gvec(params['mhd_equilibrium']['params_' + mhd_equil_type], DOMAIN, EQ_MHD_P, DERHAM, SOURCE_DOMAIN=None)
    else:
        EQ_MHD_L = Equilibrium_mhd_logical(DOMAIN, EQ_MHD_P)
        
    print('MHD equilibrium of type "' + mhd_equil_type + '" set.')
    print()

    ############################ 
    ### MHD variables object ###
    ############################
    # TODO: restart has to be done here
    mhd_init_general = params['mhd_init']['general']
    mhd_init_params  = params['mhd_init']['params_' + mhd_init_general['type']]

    MHD = Initialize_mhd_psydac(DOMAIN, DERHAM, nq_pr, mhd_init_general, mhd_init_params)

    print('MHD variables of type "' + mhd_init_general['type'] + '" initialized.')
    print()

    # ========================================================================================= 
    # MHD PROJECTION OPERATORS object
    # =========================================================================================
    MHD_OPS = MHD_ops(DERHAM, nq_pr, EQ_MHD_L, F_PSY, projectors_1d) 

    # MHD_OPS.setInverseA()
    # MHD_OPS.setPreconditionerS2(params['solvers']['PRE'])
    # MHD_OPS.setPreconditionerS6(params['solvers']['PRE'])
    # print('MHD preconditioners of type "' + params['solvers']['PRE'] + '" set.')
    # print()

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
    time_series['en_B'][0] = 1/2*MHD.b2.dot(M2.dot(MHD.b2))
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