#!/usr/bin/env python3

from mpi4py import MPI      
import yaml
import numpy as np

from struphy.geometry         import domain_3d
from struphy.mhd_equil        import mhd_equil_physical 
from struphy.mhd_equil        import mhd_equil_logical 
from struphy.feec             import spline_space
from struphy.io.inp           import mhd_init 


def execute(file_in, path_out, mode):
    '''Executes the code cc_lin_mhd_6d.

    Parameters
    ----------

    file_in : str
        Absolute path to input parameters file (.yml).
    path_out : str
        Absolute path to output folder.
    mode : boolean
        Restart ('True') or new simulation ('False').
    '''

    print('')
    print('Starting code "cc_lin_mhd_6d" ...')
    #print('- parameters from "' + file_in + '"')
    #print('- ouput in folder "' + path_out + '"')
    #print('- restart:', mode)
    print('')

    # mpi communicator
    mpi_comm = MPI.COMM_WORLD
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()
    mpi_comm.Barrier()

    # load simulation parameters
    with open(file_in) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    # create particle arrays
    Np = params['markers']['Np']
    Np_loc        = int(Np/mpi_size)                      # number of particles per MPI process
    particles_loc = np.empty((7, Np_loc), dtype=float)    # particles of each process
    w0_loc        = np.empty(    Np_loc , dtype=float)    # weights for each process: hat_f_ini(eta_0, v_0)/hat_s_ini(eta_0, v_0)
    s0_loc        = np.empty(    Np_loc , dtype=float)    # initial sampling density: hat_s_ini(eta_0, v_0) for each process

    if mpi_rank == 0:
        particles_recv = np.empty((7, Np_loc), dtype=float)
        w0_recv        = np.empty(    Np_loc , dtype=float)    
        s0_recv        = np.empty(    Np_loc , dtype=float)

    # time series diagnostics
    energies_loc = {'U' : np.empty(1, dtype=float), 
                    'B' : np.empty(1, dtype=float), 
                    'p' : np.empty(1, dtype=float), 
                    'f' : np.empty(1, dtype=float)}
    energies     = energies_loc.copy()
    divB         = np.empty(1, dtype=float)
    momentum     = np.empty(1, dtype=float)
    helicity     = np.empty(1, dtype=float)
    cr_helicity  = np.empty(1, dtype=float)

    # snapshots of distribution function via 2d particle binning
    n_bins_x = params['markers']['n_bins'][0]
    n_bins_v = params['markers']['n_bins'][1]
    bin_edges    = [np.linspace(0., 1., n_bins_x + 1),
                    np.linspace(0., params['markers']['v_max'], n_bins_v + 1)]
    dbin         = [bin_edges[0][1] - bin_edges[0][0], bin_edges[1][1] - bin_edges[1][0]]     
    fh_loc       = np.empty((n_bins_x, n_bins_v), dtype=float)
    fh           = fh_loc.copy()
        
    # domain object for metric coefficients
    Nel      = params['grid']['Nel']
    p        = params['grid']['p']
    spl_kind = params['grid']['spl_kind']
    DOMAIN   = domain_3d.Domain(params['geometry']['type'], 
                                params['geometry']['params_' + params['geometry']['type']])
    print('Domain object set.')

    # MHD equilibirum (physical)
    EQ_MHD_P = mhd_equil_physical.Equilibrium_mhd_physical(params['mhd_equilibrium']['general']['type'], 
         params['mhd_equilibrium']['params_' + params['mhd_equilibrium']['general']['type']])
    print('MHD equilibrium (physical) set.')
    
    # MHD equilibrium (logical)
    EQ_MHD_L = mhd_equil_logical.Equilibrium_mhd_logical(DOMAIN, EQ_MHD_P)
    print('MHD equilibrium (logical) set.')
    
    # FEEC spaces
    spaces_FEM_1 = spline_space.Spline_space_1d(Nel[0], p[0], spl_kind[0], params['grid']['nq_el'][0], params['grid']['bc']) 
    spaces_FEM_2 = spline_space.Spline_space_1d(Nel[1], p[1], spl_kind[1], params['grid']['nq_el'][1])
    spaces_FEM_3 = spline_space.Spline_space_1d(Nel[2], p[2], spl_kind[2], params['grid']['nq_el'][2])

    spaces_FEM_1.set_projectors(params['grid']['nq_pr'][0]) 
    spaces_FEM_2.set_projectors(params['grid']['nq_pr'][1])
    spaces_FEM_3.set_projectors(params['grid']['nq_pr'][2])

    SPACES = spline_space.Tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

    if params['grid']['polar']:
        SPACES.set_polar_splines(DOMAIN.cx[:, :, 0], DOMAIN.cy[:, :, 0])

    SPACES.set_projectors('general', params['grid']['nq_pr'])
    print('FEEC spaces set.')

    # initialize mhd variables 
    MHD = mhd_init.Initialize_mhd(DOMAIN, SPACES, file_in) 
    print('MHD variables initialized.')

    # assemble mass matrices
    if mpi_rank == 0: 
        SPACES.assemble_M2(DOMAIN)
        SPACES.assemble_M3(DOMAIN)
        print('Assembly of mass matrices done.')


    exit()


    # ================== PART K: MHD projection operators ================================
    # projection of input equilibrium
    R0_eq =  eq_MHD.r0_eq
    R3_eq =  eq_MHD.r3_eq
    P3_eq =  eq_MHD.p3_eq

    J2_eq = [eq_MHD.j2_eq_1, eq_MHD.j2_eq_2, eq_MHD.j2_eq_3]
    B2_eq = [eq_MHD.b2_eq_1, eq_MHD.b2_eq_2, eq_MHD.b2_eq_3]

    #r3_eq = spaces.projectors.pi_3(R3_eq, include_bc=True, eval_kind='tensor_product', interp=True)
    #p3_eq = spaces.projectors.pi_3(P3_eq, include_bc=True, eval_kind='tensor_product', interp=True)
    #j2_eq = spaces.projectors.pi_2(J2_eq, include_bc=True, eval_kind='tensor_product', interp=True)
    b2_eq = spaces.projectors.pi_2(B2_eq, include_bc=True, eval_kind='tensor_product', interp=True)

    if mpi_rank == 0:  
        # interface for semi-discrete linear MHD operators
        MHD = mhd.operators_mhd(spaces, dt, eq_MHD.gamma, params['loc_j_eq'], basis_u)
        
        # assemble right-hand sides of projection matrices
        MHD.assemble_rhs_EF(domain     , B2_eq)
        MHD.assemble_rhs_F( domain, 'm', R3_eq)
        MHD.assemble_rhs_F( domain, 'p', P3_eq)
        MHD.assemble_rhs_PR(domain     , P3_eq)
        
        if basis_u == 0:
            MHD.assemble_rhs_F(domain, 'j')
            
        print('Assembly of MHD projection operators done!')
            
        # assemble mass matrix weighted with 0-form density
        MHD.assemble_MR(domain, R0_eq)
        
        print('Assembly of weighted mass matrix done (density)!')
        
        # assemble mass matrix weighted with J_eq x
        MHD.assemble_JB_strong(domain, J2_eq)
        
        print('Assembly of weighted mass matrix done (current)!')
        
        # create liner MHD operators as scipy.sparse.linalg.LinearOperator
        MHD.setOperators()
        
        print('Assembly of MHD operators finished!')
    # =======================================================================



    # ======================== PART L: initialize particles =====================
    # particle accumulator (all processes)
    if nuh > 0.:
        acc = pic_accumu.accumulation(spaces, domain, basis_u, mpi_comm, control)

    if loading == 'pseudo-random':
        # pseudo-random numbers between (0, 1)
        np.random.seed(seed)
        
        for i in range(mpi_size):
            temp = np.random.rand(Np_loc, 6)
            
            if i == mpi_rank:
                particles_loc[:6] = temp.T
                break
                
        del temp

    elif loading == 'sobol_standard':
        # plain sobol numbers between (0, 1) (skip first 1000 numbers)
        particles_loc[:6] = sobol.i4_sobol_generate(6, Np_loc, 1000 + Np_loc*mpi_rank).T 

    elif loading == 'sobol_antithetic':
        # symmetric sobol numbers between (0, 1) (skip first 1000 numbers) in all 6 dimensions
        pic_sample.set_particles_symmetric(sobol.i4_sobol_generate(6, int(Np_loc/64), 1000 + int(Np_loc/64)*mpi_rank), particles_loc, Np_loc)  

    elif loading == 'external':
        
        if mpi_rank == 0:
            file = h5py.File(params['dir_particles'], 'a')
            
            particles_loc[:, :] = file['particles'][0, :Np_loc].T
                
            for i in range(1, mpi_size):
                particles_recv[:, :] = file['particles'][0, i*Np_loc:(i + 1)*Np_loc].T
                mpi_comm.Send(particles_recv, dest=i, tag=11)         
        else:
            mpi_comm.Recv(particles_loc, source=0, tag=11)

    else:
        print('particle loading not specified')

    # inversion of cumulative distribution function
    particles_loc[3] = sp.erfinv(2*particles_loc[3] - 1)*vth + v0[0]
    particles_loc[4] = sp.erfinv(2*particles_loc[4] - 1)*vth + v0[1]
    particles_loc[5] = sp.erfinv(2*particles_loc[5] - 1)*vth + v0[2]

    # compute initial weights
    pic_sample.compute_weights_ini(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

    if control == True:
        pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
    else:
        particles_loc[6] = w0_loc

    print(mpi_rank, 'particle initialization done!')
    # ======================================================================================



    # ================ PART M: initial energies and distribution function ===========
    # initial energies
    if mpi_rank == 0:
        energies['U'][0] = 1/2*up.dot(MHD.A(up))
        energies['B'][0] = 1/2*b2.dot(spaces.M2.dot(b2))
        energies['p'][0] = 1/(eq_MHD.gamma - 1)*sum(p3.flatten())

        energies_H['U'][0] = 1/2*up.dot(MHD.A(up))
        energies_H['B'][0] = 1/2*b2.dot(spaces.M2.dot(b2))

    energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
    mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

    # subtract equilibrium hot ion energy for analaytical mappings and full-f
    if domain.kind_map >= 10:
        energies['f'] += (control - 1)*equ_PIC.eh_eq(domain.kind_map, domain.params_map)*nuh
        
    # initial distribution function
    fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
    mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)

    print('initial diagnostics done')
    # ===============================================================================



    # =============== PART N: preconditioners for time integration ==========================
    if mpi_rank == 0:
        
        # assemble approximate inverse interpolation/histopolation matrices
        if params['PRE'] == 'ILU':
            spaces.projectors.assemble_approx_inv(params['tol_inv'])
        
        timea = time.time()
        MHD.setPreconditionerA(domain, R3_eq, params['PRE'], drop_tol_A, fill_fac_A)
        timeb = time.time()
        print('Preconditioner for A  done!', timeb - timea)
        
        timea = time.time()
        MHD.setPreconditionerS2(domain, R3_eq, params['PRE'], drop_tol_S2, fill_fac_S2)
        timeb = time.time()
        print('Preconditioner for S2 done!', timeb - timea)
        
        timea = time.time()
        MHD.setPreconditionerS6(domain, R3_eq, params['PRE'], drop_tol_S6, fill_fac_S6)
        timeb = time.time()
        print('Preconditioner for S6 done!', timeb - timea)
    # ===============================================================================



    # ================ PART O: define time splitting substeps ======================
    def substep_1(dt):
        # MHD: shear Alfven terms 
        # update (u,b)

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        if mpi_rank == 0:
            
            # save coefficients from previous time step
            up_old[:] = up
            
            # RHS of linear system
            RHS = MHD.RHS2(up, b2)
            
            # solve linear system with gmres method and values from last time step as initial guess (weak)
            timea = time.time()
                
            num_iters = 0
            
            if   solver_type_2 == 'gmres':
                up[:], info = spa.linalg.gmres(MHD.S2, RHS, x0=up, tol=tol2, maxiter=maxiter2, M=MHD.S2_PRE, callback=count_iters)
            elif solver_type_2 == 'cg':
                up[:], info = spa.linalg.cg(   MHD.S2, RHS, x0=up, tol=tol2, maxiter=maxiter2, M=MHD.S2_PRE, callback=count_iters)
            elif solver_type_2 == 'cgs':
                up[:], info = spa.linalg.cgs(  MHD.S2, RHS, x0=up, tol=tol2, maxiter=maxiter2, M=MHD.S2_PRE, callback=count_iters)
            else:
                raise ValueError('only gmres and cg solvers available')
                
            #print('linear solver step 2 : ', info, num_iters)
            
            timeb = time.time()
            times_elapsed['update_step2u'] = timeb - timea
            #print('update_step2u : ', timeb - timea)
            
            # update magnetic field (strong)
            timea = time.time()
            b2[:] = b2 - dt*spaces.C.dot(MHD.EF((up + up_old)/2))
            timeb = time.time()
            times_elapsed['update_step2b'] = timeb - timea
            #print('update_step2b : ', timeb - timea)
        
        # broadcast new magnetic FEM coefficients
        mpi_comm.Bcast(b2, root=0)



    def substep_2(dt):
        # MHD: non-Hamiltonian part
        # update (rho, u, p)

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        if mpi_rank == 0:
            
            # save energies after Hamiltonian steps
            energies_H['U'][0] = 1/2*up.dot(MHD.A(up))

            timea = time.time()
            
            # save coefficients from previous time step
            up_old[:] = up
            
            # RHS of linear system
            RHS = MHD.RHS6(up, p3, b2)
            
            # solve linear system with conjugate gradient squared method and values from last time step as initial guess
            timea = time.time()
                
            num_iters = 0
            up[:], info = spa.linalg.gmres(MHD.S6, RHS, x0=up, tol=tol6, maxiter=maxiter6, M=MHD.S6_PRE, callback=count_iters)
            #print('linear solver step 6 : ', info, num_iters)
            
            timeb = time.time()
            times_elapsed['update_step2u'] = timeb - timea
            
            # update pressure
            p3[:] = p3 + dt*MHD.L((up + up_old)/2)
            
            # update density
            r3[:] = r3 - dt*spaces.D.dot(MHD.MF((up + up_old)/2))

            timeb = time.time()
            times_elapsed['update_step6'] = timeb - timea



    def substep_3(dt):
        # Vlasov 6D
        # update H (particle positions)

        timea = time.time()
        pic_pusher.pusher_step4(particles_loc, dt, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
        timeb = time.time()
        times_elapsed['pusher_step4'] = timeb - timea
        #print('pusher_step4 : ', timeb - timea)



    def substep_4(dt):
        # CCS: j_h term
        # update (u,V)

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # current accumulation (all processes) 
        timea = time.time()
        acc.accumulate_step3(particles_loc, b2 + b2_eq, mpi_comm)
        timeb = time.time()
        times_elapsed['accumulation_step3'] = timeb - timea
        
        # set up and solve linear system (only MHD process) 
        if mpi_rank == 0:
            
            # save coefficients from previous time step
            up_old[:] = up
            
            # build global sparse matrix and vector
            timea    = time.time()
            mat, vec = acc.assemble_step3(Np, b2 + b2_eq)
            mat      = nuh*alpha*params['Zh']/params['Ab']*mat
            vec      = nuh*alpha*params['Zh']/params['Ab']*vec
            timeb    = time.time()
            times_elapsed['control_step3'] = timeb - timea
            #print('control_step3 : ', timeb - timea)
            
            # RHS of linear system
            RHS = MHD.A(up) - dt**2/4*mat.dot(up) + dt*vec
            
            # LHS of linear system
            LHS = spa.linalg.LinearOperator(MHD.A.shape, lambda x : MHD.A(x) + dt**2/4*mat.dot(x))
            
            # solve linear system with gmres method and values from last time step as initial guess
            timea = time.time()
            
            num_iters = 0
            
            if   solver_type_3 == 'gmres':
                up[:], info = spa.linalg.gmres(LHS, RHS, x0=up, tol=tol3, maxiter=maxiter3, M=MHD.A_PRE, callback=count_iters)
            elif solver_type_3 == 'cg':
                up[:], info = spa.linalg.cg(   LHS, RHS, x0=up, tol=tol3, maxiter=maxiter3, M=MHD.A_PRE, callback=count_iters)
            elif solver_type_3 == 'cgs':
                up[:], info = spa.linalg.cgs(  LHS, RHS, x0=up, tol=tol3, maxiter=maxiter3, M=MHD.A_PRE, callback=count_iters)
                
            #print('linear solver step 3 : ', info, num_iters)
            
            timeb = time.time()
            times_elapsed['update_step3u'] = timeb - timea
            #print('update_step3u : ', timeb - timea)
        
        # broadcast new FEM coefficients
        mpi_comm.Bcast(up    , root=0)
        mpi_comm.Bcast(up_old, root=0)
        
        # update velocities 
        timea = time.time()
        
        b2_ten_1, b2_ten_2, b2_ten_3 = spaces.extract_2(b2 + b2_eq)
        
        if basis_u == 0:
            up_ten_1, up_ten_2, up_ten_3 = spaces.extract_0((up + up_old)/2)
        else:
            up_ten_1, up_ten_2, up_ten_3 = spaces.extract_2((up + up_old)/2)
        
        pic_pusher.pusher_step3(particles_loc, alpha*params['Zh']/params['Ah']*dt, spaces.T[0], spaces.T[1], spaces.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, np.zeros(N_0form, dtype=float), up_ten_1, up_ten_2, up_ten_3, basis_u, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz, np.zeros(Np_loc, dtype=float))
        
        timeb = time.time()
        times_elapsed['pusher_step3'] = timeb - timea



    def substep_5(dt):
        # Vlasov 6D, dot v = v x B
        # update V (particle velocities)

        # push particles
        timea = time.time()
        
        b2_ten_1, b2_ten_2, b2_ten_3 = spaces.extract_2(b2 + b2_eq)
        
        pic_pusher.pusher_step5_ana(particles_loc, alpha*params['Zh']/params['Ah']*dt, spaces.T[0], spaces.T[1], spaces.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
        
        timeb = time.time()
        times_elapsed['pusher_step5'] = timeb - timea
        #print('pusher_step5 : ', timeb - timea)



    def substep_6(dt):
        # CCS: rho_h term
        # update u

        # counter for number of interation steps in iterative solvers
        num_iters = 0
        def count_iters(xk):
            nonlocal num_iters
            num_iters += 1

        # charge accumulation (all processes) 
        timea = time.time()
        acc.accumulate_step1(particles_loc, b2 + b2_eq, mpi_comm)
        timeb = time.time()
        times_elapsed['accumulation_step1'] = timeb - timea
        
        # set up and solve linear system (only MHD process) 
        if mpi_rank == 0:
            
            # build global sparse matrix 
            timea = time.time()
            mat   = nuh*alpha*params['Zh']/params['Ab']*acc.assemble_step1(Np, b2 + b2_eq)
            timeb = time.time()
            times_elapsed['control_step1'] = timeb - timea
            #print('control_step1 : ', timeb - timea)
            
            # RHS of linear system
            RHS = MHD.A(up) + dt/2*mat.dot(up)
            
            # LHS of linear system
            LHS = spa.linalg.LinearOperator(MHD.A.shape, lambda x : MHD.A(x) - dt/2*mat.dot(x))
                
            # solve linear system with gmres method and values from last time step as initial guess 
            timea = time.time()
            
            num_iters = 0
            up[:], info = spa.linalg.gmres(LHS, RHS, x0=up, tol=tol1, maxiter=maxiter1, M=MHD.A_PRE, callback=count_iters)
            #print('linear solver step 1 : ', info, num_iters)
            
            timeb = time.time()
            times_elapsed['update_step1u'] = timeb - timea
            #print('update_step1u : ', timeb - timea)



    def update_weights():
        # update particle weights in case of delta-f
        timea = time.time()
        pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
        timeb = time.time()
        times_elapsed['control_weights'] = timeb - timea
        #print('control_weights : ', timeb - timea)



    # ==================== PART P: define time integrator ==========================================
    # time series of cpu time of different portions of the code:
    times_elapsed = {'total' : 0., 'accumulation_step1' : 0., 
                    'accumulation_step3' : 0., 'pusher_step3' : 0., 
                    'pusher_step4' : 0., 'pusher_step5' : 0.,
                    'control_step1' : 0., 'control_step3' : 0.,
                    'control_weights' : 0., 'update_step1u' : 0., 
                    'update_step2u' : 0., 'update_step2b' : 0., 
                    'update_step3u' : 0.,'update_step6' : 0.}

    # time stepping function:
    def update():
        
        # define global variables
        global up, up_old, b2, b2_eq, p3, r3, particles_loc
        
        time_tota = time.time()

        if split_algo=='LieTrotter':

            # substeps (Lie-Trotter splitting):
            substep_6(dt)          # CCS: update u
            substep_1(dt)              # MHD shear Alfven:    update (u,b) 
            substep_4(dt)          # CCS: update (u,V)
            substep_3(dt)          # CCS: update H
            substep_2(dt)              # MHD non-Hamiltonian: update (rho,u,p) 
            substep_5(dt)          # CCS: update V
            #if nuh > 0.: 
            
            
            
            if control == True: update_weights()
            

        elif split_algo=='Strang':

            # substeps (Strang splitting):
            if nuh > 0.: 

                substep_1(dt/2.)          # MHD shear Alfven:    update (u,b) 
                substep_2(dt/2.)          # MHD non-Hamiltonian: update (rho,u,p) 
                substep_3(dt/2.)          # CCS: update H
                substep_5(dt/2.)          # CCS: update V
                if control == True: update_weights()
                substep_6(dt/2.)          # CCS: update u
                substep_4(dt)             # CCS: update (u,V)
                if control == True: update_weights()
                substep_6(dt/2.)          # CCS: update u
                substep_5(dt/2.)          # CCS: update V
                substep_3(dt/2.)          # CCS: update H
                substep_2(dt/2.)          # MHD non-Hamiltonian: update (rho,u,p) 
                substep_1(dt/2.)          # MHD shear Alfven:    update (u,b) 

            else:

                substep_2(dt/2.)          # MHD non-Hamiltonian: update (rho,u,p) 
                substep_1(dt)             # MHD shear Alfven:    update (u,b) 
                substep_2(dt/2.)          # MHD non-Hamiltonian: update (rho,u,p) 

        else:
            raise NotImplementedError('Only Lie-Trotter and Strang splitting available!')   
    
        
        # diagnostics:
        # energies
        if mpi_rank == 0:
            energies['U'][0] = 1/2*up.dot(MHD.A(up))
            energies['B'][0] = 1/2*b2.dot(spaces.M2.dot(b2))
            energies['p'][0] = 1/(eq_MHD.gamma - 1)*sum(p3.flatten())

        energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
        mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

        # subtract equilibrium hot ion energy for analaytical mappings and full-f
        if domain.kind_map >= 10:
            energies['f'] += (control - 1)*equ_PIC.eh_eq(domain.kind_map, domain.params_map)*nuh

        # distribution function
        fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
        mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)
        
        # total time taken:
        time_totb = time.time()
        times_elapsed['total'] = time_totb - time_tota


        
    # ========================== PART Q: time integration ================================
    if params['time_int'] == True:
        
        # a new simulation
        if params['restart'] == False:
        
            if mpi_rank == 0:
            
                # create hdf5 file and datasets for simulation output 
                file = h5py.File('results_' + identifier + '.hdf5', 'a')

                # current time
                file.create_dataset('time'                            , (1,), maxshape=(None,), dtype=float, chunks=True)
                
                # energies
                file.create_dataset('energies/bulk_kinetic'           , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('energies/magnetic'               , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('energies/bulk_internal'          , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('energies/energetic'              , (1,), maxshape=(None,), dtype=float, chunks=True)
                
                file.create_dataset('energies/bulk_kinetic_H'         , (1,), maxshape=(None,), dtype=float, chunks=True)

                # elapsed times of different parts of the code
                file.create_dataset('times_elapsed/total'             , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/accumulation_step1', (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/accumulation_step3', (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/pusher_step3'      , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/pusher_step4'      , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/pusher_step5'      , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/control_step1'     , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/control_step3'     , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/control_weights'   , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/update_step1u'     , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/update_step2u'     , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/update_step2b'     , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/update_step3u'     , (1,), maxshape=(None,), dtype=float, chunks=True)
                file.create_dataset('times_elapsed/update_step6'      , (1,), maxshape=(None,), dtype=float, chunks=True)

                # FEM coefficients
                sh_0_ini  = (1   , N_0form[0]   , N_0form[1]   , N_0form[2])
                sh_0_max  = (None, N_0form[0]   , N_0form[1]   , N_0form[2])
                
                sh_11_ini = (1   , N_1form[0][0], N_1form[0][1], N_1form[0][2])
                sh_11_max = (None, N_1form[0][0], N_1form[0][1], N_1form[0][2])
                
                sh_12_ini = (1   , N_1form[1][0], N_1form[1][1], N_1form[1][2])
                sh_12_max = (None, N_1form[1][0], N_1form[1][1], N_1form[1][2])
                
                sh_13_ini = (1   , N_1form[2][0], N_1form[2][1], N_1form[2][2])
                sh_13_max = (None, N_1form[2][0], N_1form[2][1], N_1form[2][2])
                
                sh_21_ini = (1   , N_2form[0][0], N_2form[0][1], N_2form[0][2])
                sh_21_max = (None, N_2form[0][0], N_2form[0][1], N_2form[0][2])
                
                sh_22_ini = (1   , N_2form[1][0], N_2form[1][1], N_2form[1][2])
                sh_22_max = (None, N_2form[1][0], N_2form[1][1], N_2form[1][2])
                
                sh_23_ini = (1   , N_2form[2][0], N_2form[2][1], N_2form[2][2])
                sh_23_max = (None, N_2form[2][0], N_2form[2][1], N_2form[2][2])
                
                sh_3_ini  = (1   , N_3form[0]   , N_3form[1]   , N_3form[2])
                sh_3_max  = (None, N_3form[0]   , N_3form[1]   , N_3form[2])
                
                file.create_dataset('pressure', sh_3_ini, maxshape=sh_3_max, dtype=float, chunks=True)
                file.create_dataset('density' , sh_3_ini, maxshape=sh_3_max, dtype=float, chunks=True)
                
                if basis_u == 0:
                    file.create_dataset('velocity_field/1_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
                    file.create_dataset('velocity_field/2_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
                    file.create_dataset('velocity_field/3_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
                else:
                    file.create_dataset('velocity_field/1_component', sh_21_ini, maxshape=sh_21_max, dtype=float, chunks=True)
                    file.create_dataset('velocity_field/2_component', sh_22_ini, maxshape=sh_22_max, dtype=float, chunks=True)
                    file.create_dataset('velocity_field/3_component', sh_23_ini, maxshape=sh_23_max, dtype=float, chunks=True)
                
                file.create_dataset('magnetic_field/1_component', sh_21_ini, maxshape=sh_21_max, dtype=float, chunks=True)
                file.create_dataset('magnetic_field/2_component', sh_22_ini, maxshape=sh_22_max, dtype=float, chunks=True)
                file.create_dataset('magnetic_field/3_component', sh_23_ini, maxshape=sh_23_max, dtype=float, chunks=True)
                
                # particles
                file.create_dataset('particles', (1, 7, Np), maxshape=(None, 7, Np), dtype=float, chunks=True)
                
                # other diagnostics
                file.create_dataset('bulk_mass', (1,), maxshape=(None,), dtype=float, chunks=True)

                file.create_dataset('magnetic_field/divergence', sh_3_ini , maxshape=sh_3_max , dtype=float, chunks=True)

                file.create_dataset('distribution_function/eta1_vx', (1, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), maxshape=(None, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float, chunks=True)

                # datasets for restart function
                file.create_dataset('restart/time_steps_done', (1,), maxshape=(None,), dtype=int, chunks=True)

                file.create_dataset('restart/p', (1, p3.size), maxshape=(None, p3.size), dtype=float, chunks=True)
                file.create_dataset('restart/r', (1, r3.size), maxshape=(None, p3.size), dtype=float, chunks=True)
                file.create_dataset('restart/u', (1, up.size), maxshape=(None, up.size), dtype=float, chunks=True)
                file.create_dataset('restart/b', (1, b2.size), maxshape=(None, b2.size), dtype=float, chunks=True)

                file.create_dataset('restart/particles', (1, 6, Np), maxshape=(None, 6, Np), dtype=float, chunks=True)
                file.create_dataset('restart/control_w0', (Np,), dtype=float)
                file.create_dataset('restart/control_s0', (Np,), dtype=float)

                # save initial data 
                file['time'][0]                       = 0.

                file['energies/bulk_kinetic'][0]      = energies['U'][0]
                file['energies/magnetic'][0]          = energies['B'][0]
                file['energies/bulk_internal'][0]     = energies['p'][0]
                file['energies/energetic'][0]         = energies['f'][0]
                
                file['energies/bulk_kinetic_H'][0]    = energies_H['U'][0]
                
                file['pressure'][0]                   = spaces.extract_3(p3)
                file['density'][0]                    = spaces.extract_3(r3)
                
                if basis_u == 0:
                    up_ten_1, up_ten_2, up_ten_3 = spaces.extract_0(up)
                else:
                    up_ten_1, up_ten_2, up_ten_3 = spaces.extract_2(up)
                
                file['velocity_field/1_component'][0] = up_ten_1
                file['velocity_field/2_component'][0] = up_ten_2
                file['velocity_field/3_component'][0] = up_ten_3
                
                b2_ten_1, b2_ten_2, b2_ten_3 = spaces.extract_2(b2)
                file['magnetic_field/1_component'][0] = b2_ten_1
                file['magnetic_field/2_component'][0] = b2_ten_2
                file['magnetic_field/3_component'][0] = b2_ten_3
                
                file['magnetic_field/divergence'][0]  = spaces.extract_3(spaces.D.dot(b2))
                file['bulk_mass'][0]                  = sum(r3.flatten())
                
                file['distribution_function/eta1_vx'][0] = fh['eta1_vx']
                
                file['particles'][0, :, :Np_loc]      = particles_loc
                file['restart/control_w0'][:Np_loc]   = w0_loc
                file['restart/control_s0'][:Np_loc]   = s0_loc
                
                for i in range(1, mpi_size):
                    mpi_comm.Recv(particles_recv, source=i, tag=11)
                    mpi_comm.Recv(w0_recv       , source=i, tag=12)
                    mpi_comm.Recv(s0_recv       , source=i, tag=13)
                    
                    file['particles'][0, :, i*Np_loc:(i + 1)*Np_loc]    = particles_recv
                    file['restart/control_w0'][i*Np_loc:(i + 1)*Np_loc] = w0_recv
                    file['restart/control_s0'][i*Np_loc:(i + 1)*Np_loc] = s0_recv
                # =================================   
                
            else:
                mpi_comm.Send(particles_loc, dest=0, tag=11)
                mpi_comm.Send(w0_loc       , dest=0, tag=12)
                mpi_comm.Send(s0_loc       , dest=0, tag=13)

            time_steps_done = 0
        
        # restarting another simulation
        else:
            
            if mpi_rank == 0:

                # open existing hdf5 file
                file = h5py.File('results_' + identifier + '.hdf5', 'a')

                # load restart data from last time step
                num_restart = params['num_restart']
                
                time_steps_done = file['restart/time_steps_done'][num_restart]

                p3[:] = file['restart/p'][num_restart]
                r3[:] = file['restart/r'][num_restart]
                up[:] = file['restart/u'][num_restart]
                b2[:] = file['restart/b'][num_restart]
                
                particles_loc[:6] = file['restart/particles'][num_restart, :, :Np_loc]
                w0_loc[:]         = file['restart/control_w0'][:Np_loc]
                s0_loc[:]         = file['restart/control_s0'][:Np_loc]
                
                for i in range(1, mpi_size):
                    particles_recv[:6] = file['restart/particles'][num_restart, :, i*Np_loc:(i + 1)*Np_loc]
                    w0_recv[:]         = file['restart/control_w0'][i*Np_loc:(i + 1)*Np_loc]
                    s0_recv[:]         = file['restart/control_s0'][i*Np_loc:(i + 1)*Np_loc]
                    
                    mpi_comm.Send(particles_recv, dest=i, tag=11)
                    mpi_comm.Send(w0_recv       , dest=i, tag=12)
                    mpi_comm.Send(s0_recv       , dest=i, tag=13)
                    
            else:
                mpi_comm.Recv(particles_loc, source=0, tag=11)
                mpi_comm.Recv(w0_loc       , source=0, tag=12)
                mpi_comm.Recv(s0_loc       , source=0, tag=13)
                
                time_steps_done = None
            
            # broadcast time_steps_done to all processes
            time_steps_done = mpi_comm.bcast(time_steps_done, root=0)
            
            # broadcast loaded FEM coefficients to all processes
            mpi_comm.Bcast(up, root=0)
            mpi_comm.Bcast(b2, root=0)

            # perform initialization for next time step: compute particle weights
            if control == True:
                pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            else:
                particles_loc[6] = w0_loc

        
        
        # ========================================================================================
        #              time loop 
        # ========================================================================================
        mpi_comm.Barrier()
        if mpi_rank == 0:
            print()
            print('Start time integration: ' + split_algo)
        # ========================================================================================
        time_start = time.time()
        while True:

            # synchronize MPI processes and check if simulation end is reached
            mpi_comm.Barrier()
            if (time_steps_done*dt >= Tend) or ((time.time() - start_simulation)/60 > max_time):
                
                # save data needed for restart
                if params['create_restart']:
                    if mpi_rank == 0:
                        
                        file['restart/time_steps_done'][-1] = time_steps_done
                        
                        file['restart/p'][-1] = p3
                        file['restart/r'][-1] = r3
                        file['restart/u'][-1] = up
                        file['restart/b'][-1] = b2
                        
                        file['restart/particles'][-1, :, :Np_loc]      = particles_loc[:6]

                        for i in range(1, mpi_size):
                            mpi_comm.Recv(particles_recv, source=i, tag=20)
                            file['restart/particles'][-1, :, i*Np_loc:(i + 1)*Np_loc] = particles_recv[:6]

                        file['restart/time_steps_done'].resize(file['restart/time_steps_done'].shape[0] + 1, axis = 0)
                        
                        file['restart/p'].resize(file['restart/p'].shape[0] + 1, axis = 0)
                        file['restart/r'].resize(file['restart/r'].shape[0] + 1, axis = 0)
                        file['restart/u'].resize(file['restart/u'].shape[0] + 1, axis = 0)
                        file['restart/b'].resize(file['restart/b'].shape[0] + 1, axis = 0)

                        file['restart/particles'].resize(file['restart/particles'].shape[0] + 1, axis = 0)

                    else:
                        mpi_comm.Send(particles_loc, dest=0, tag=20)
                
                # close output file and time loop
                if mpi_rank == 0:
                    file.close()
                    time_end = time.time()
                    print('time of simulation [sec]: ', time_end - time_start)

                break

            # print number of finished time steps and current energies
            if mpi_rank == 0 and time_steps_done%50 == 0:
                print('time steps finished : ' + str(time_steps_done) + '/' + str(int(Tend/dt)), 
                    'B-energy:', energies['B'][0])
                #print('energies : ', energies)
                
            

            # ========================= time integration and data output =========================
            timea = time.time()
            update()
            time_steps_done += 1
            
            
            if mpi_rank == 0:
                # == data to save ==========
                
                # current time
                file['time'].resize(file['time'].shape[0] + 1, axis = 0)
                file['time'][-1] = time_steps_done*dt

                # energies
                file['energies/bulk_kinetic'].resize(file['energies/bulk_kinetic'].shape[0] + 1, axis = 0)
                file['energies/magnetic'].resize(file['energies/magnetic'].shape[0] + 1, axis = 0)
                file['energies/bulk_internal'].resize(file['energies/bulk_internal'].shape[0] + 1, axis = 0)
                file['energies/energetic'].resize(file['energies/energetic'].shape[0] + 1, axis = 0)
                file['energies/bulk_kinetic'][-1]  = energies['U'][0]
                file['energies/magnetic'][-1]      = energies['B'][0]
                file['energies/bulk_internal'][-1] = energies['p'][0]
                file['energies/energetic'][-1]     = energies['f'][0]
                
                file['energies/bulk_kinetic_H'].resize(file['energies/bulk_kinetic_H'].shape[0] + 1, axis = 0)

                # elapsed times of different parts of the code
                file['times_elapsed/total'].resize(file['times_elapsed/total'].shape[0] + 1, axis = 0)
                file['times_elapsed/accumulation_step1'].resize(file['times_elapsed/accumulation_step1'].shape[0] + 1, axis = 0)
                file['times_elapsed/accumulation_step3'].resize(file['times_elapsed/accumulation_step3'].shape[0] + 1, axis = 0)
                file['times_elapsed/pusher_step3'].resize(file['times_elapsed/pusher_step3'].shape[0] + 1, axis = 0)
                file['times_elapsed/pusher_step4'].resize(file['times_elapsed/pusher_step4'].shape[0] + 1, axis = 0)
                file['times_elapsed/pusher_step5'].resize(file['times_elapsed/pusher_step5'].shape[0] + 1, axis = 0)
                file['times_elapsed/control_step1'].resize(file['times_elapsed/control_step1'].shape[0] + 1, axis = 0)
                file['times_elapsed/control_step3'].resize(file['times_elapsed/control_step3'].shape[0] + 1, axis = 0)
                file['times_elapsed/control_weights'].resize(file['times_elapsed/control_weights'].shape[0] + 1, axis = 0)
                file['times_elapsed/update_step1u'].resize(file['times_elapsed/update_step1u'].shape[0] + 1, axis = 0)
                file['times_elapsed/update_step2u'].resize(file['times_elapsed/update_step2u'].shape[0] + 1, axis = 0)
                file['times_elapsed/update_step2b'].resize(file['times_elapsed/update_step2b'].shape[0] + 1, axis = 0)
                file['times_elapsed/update_step3u'].resize(file['times_elapsed/update_step3u'].shape[0] + 1, axis = 0)
                file['times_elapsed/update_step6'].resize(file['times_elapsed/update_step6'].shape[0] + 1, axis = 0)
                file['times_elapsed/total'][-1]              = times_elapsed['total']
                file['times_elapsed/accumulation_step1'][-1] = times_elapsed['accumulation_step1']
                file['times_elapsed/accumulation_step3'][-1] = times_elapsed['accumulation_step3']
                file['times_elapsed/pusher_step3'][-1]       = times_elapsed['pusher_step3']
                file['times_elapsed/pusher_step4'][-1]       = times_elapsed['pusher_step4']
                file['times_elapsed/pusher_step5'][-1]       = times_elapsed['pusher_step5']
                file['times_elapsed/control_step1'][-1]      = times_elapsed['control_step1']
                file['times_elapsed/control_step3'][-1]      = times_elapsed['control_step3']
                file['times_elapsed/control_weights'][-1]    = times_elapsed['control_weights']
                file['times_elapsed/update_step1u'][-1]      = times_elapsed['update_step1u']
                file['times_elapsed/update_step2u'][-1]      = times_elapsed['update_step2u']
                file['times_elapsed/update_step2b'][-1]      = times_elapsed['update_step2b']
                file['times_elapsed/update_step3u'][-1]      = times_elapsed['update_step3u']
                file['times_elapsed/update_step6'][-1]       = times_elapsed['update_step6']

                # FEM coefficients
                file['pressure'].resize(file['pressure'].shape[0] + 1, axis = 0)
                file['pressure'][-1] = spaces.extract_3(p3)
                
                file['density'].resize(file['density'].shape[0] + 1, axis = 0)
                file['density'][-1] = spaces.extract_3(r3)
                
                b2_ten_1, b2_ten_2, b2_ten_3 = spaces.extract_2(b2)
                file['magnetic_field/1_component'].resize(file['magnetic_field/1_component'].shape[0] + 1, axis = 0)
                file['magnetic_field/2_component'].resize(file['magnetic_field/2_component'].shape[0] + 1, axis = 0)
                file['magnetic_field/3_component'].resize(file['magnetic_field/3_component'].shape[0] + 1, axis = 0)
                file['magnetic_field/1_component'][-1] = b2_ten_1
                file['magnetic_field/2_component'][-1] = b2_ten_2
                file['magnetic_field/3_component'][-1] = b2_ten_3
                
                if basis_u == 0:
                    up_ten_1, up_ten_2, up_ten_3 = spaces.extract_0(up)
                else:
                    up_ten_1, up_ten_2, up_ten_3 = spaces.extract_2(up)

                file['velocity_field/1_component'].resize(file['velocity_field/1_component'].shape[0] + 1, axis = 0)
                file['velocity_field/2_component'].resize(file['velocity_field/2_component'].shape[0] + 1, axis = 0)
                file['velocity_field/3_component'].resize(file['velocity_field/3_component'].shape[0] + 1, axis = 0)
                file['velocity_field/1_component'][-1] = up_ten_1
                file['velocity_field/2_component'][-1] = up_ten_2
                file['velocity_field/3_component'][-1] = up_ten_3
                
                # other diagnostics
                file['magnetic_field/divergence'].resize(file['magnetic_field/divergence'].shape[0] + 1, axis = 0)
                file['magnetic_field/divergence'][-1] = spaces.extract_3(spaces.D.dot(b2))

                file['bulk_mass'].resize(file['bulk_mass'].shape[0] + 1, axis = 0)
                file['bulk_mass'][-1] = sum(r3.flatten())
                
                #file['distribution_function/eta1_vx'].resize(file['distribution_function/eta1_vx'].shape[0] + 1, axis = 0)
                #file['distribution_function/eta1_vx'][-1] = fh['eta1_vx']
                
                """
                # particles
                if time_steps_done%5 == 0:
                    file['particles'].resize(file['particles'].shape[0] + 1, axis = 0)
                    file['particles'][-1, :, :Np_loc] = particles_loc
                    
                    for i in range(1, mpi_size):
                        mpi_comm.Recv(particles_recv, source=i, tag=11)
                        file['particles'][-1, :, i*Np_loc:(i + 1)*Np_loc] = particles_recv
                # ==========================
            
            else:
                if time_steps_done%5 == 0:
                    mpi_comm.Send(particles_loc, dest=0, tag=11)
                """
                    
            timeb = time.time()
            
            if mpi_rank == 0 and time_steps_done == 1:
                print('time for one time step : ', timeb - timea)        