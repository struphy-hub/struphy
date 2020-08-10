import time
start_simulation = time.time()

import sys
import h5py
import yaml

# load python modules
import numpy         as np
import scipy.sparse  as spa
import scipy.special as sp

# load global hylife modules
import hylife.utilitis_FEEC.bsplines                 as bsp
import hylife.utilitis_FEEC.spline_space             as spl
import hylife.utilitis_FEEC.derivatives              as der
import hylife.utilitis_FEEC.basics.mass_matrices_3d  as mass
import hylife.utilitis_FEEC.basics.inner_products_3d as inner

import hylife.utilitis_PIC.sobol_seq                 as sobol
import hylife.utilitis_PIC.pusher                    as pic_pusher
import hylife.utilitis_PIC.accumulation              as pic_accumu
import hylife.utilitis_PIC.accumulation_kernels      as pic_accumu_ker


# load local source files
import source_run.projectors_local     as proj
import source_run.projectors_local_mhd as mhd
import source_run.control_variate      as cv
import source_run.sampling             as pic_sample
import source_run.fields               as pic_fields


# load local input files
import input_run.equilibrium_PIC as eq_PIC


# load mpi4py and create communicator
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()


timea = time.time()
# ======================== load parameters ============================
identifier = 'sed_replace_run_dir'   

with open('parameters_sed_replace_run_dir.yml') as file:
    params = yaml.load(file)


# mesh generation
Nel            = params['Nel']
bc             = params['bc']
p              = params['p']
nq_el          = params['nq_el']
nq_pr          = params['nq_pr']

# time integration
time_int       = params['time_int']
dt             = params['dt']
Tend           = params['Tend']
max_time       = params['max_time']

# geometry
kind_map       = params['kind_map']
params_map     = params['params_map']

# general
add_pressure   = params['add_pressure']
gamma          = params['gamma']

# particles
add_PIC        = params['add_PIC']
Np             = params['Np']
control        = params['control']
v0             = params['v0']                    
vth            = params['vth']
v0x            = v0[0]
v0y            = v0[1]                       
v0z            = v0[2]
loading        = params['loading']

# restart function
restart        = params['restart']
num_restart    = params['num_restart']
create_restart = params['create_restart']
# ========================================================================


# ================= MPI initialization for particles =====================
Np_loc         = int(Np/mpi_size)                      # number of particles for each process

particles_loc  = np.empty((Np_loc, 7), dtype=float)    # particles of each process
w0_loc         = np.empty( Np_loc    , dtype=float)    # weights for each process: hat_f_ini(eta_0, v_0)/hat_s_ini(eta_0, v_0)
s0_loc         = np.empty( Np_loc    , dtype=float)    # initial sampling density: hat_s_ini(eta_0, v_0) for each process

U_part_loc     = np.empty((Np_loc, 3), dtype=float)    # bulk velocity field (1-form) at particle positions
B_part_loc     = np.empty((Np_loc, 3), dtype=float)    # magnetic field (2-form )at particles positions

en_deltaf_loc  = np.empty( 1         , dtype=float)    # hot ion energy computed from particles on each process (delta-f)
en_deltaf      = np.empty( 1         , dtype=float)    # total hot ion energy (delta-f)

n_bins         = [32, 64]
bin_edges      = [np.linspace(0., 1., n_bins[0] + 1), np.linspace(0., 5., n_bins[1] + 1)]
dbin           = [bin_edges[0][1] - bin_edges[0][0], bin_edges[1][1] - bin_edges[1][0]]

fh_eta1_vx_loc = np.empty((n_bins[0], n_bins[1]), dtype=float)  # hot ion distribution function (in eta1-vx-plane) for process
fh_eta1_vx     = np.empty((n_bins[0], n_bins[1]), dtype=float)  # total hot ion distribution function (in eta1-vx-plane)
# ========================================================================



# ================== basics ==============================================
# element boundaries and spline knot vectors (N and D)
el_b           = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                      
T              = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]
t              = [T[1:-1] for T in T] 
   
# 1d B-spline finite element spaces (save evaluated quadrature points only for MHD process)
if mpi_rank == 0:
    spaces     = [spl.spline_space_1d(T, p, bc, nq_el) for T, p, bc, nq_el in zip(T, p, bc, nq_el)]
else:
    spaces     = [spl.spline_space_1d(T, p, bc) for T, p, bc in zip(T, p, bc)]

# 3d tensor-product B-spline spaces
tensor_space   = spl.tensor_spline_space(spaces)

# number of basis functions in different spaces
NbaseN         = tensor_space.NbaseN
NbaseD         = tensor_space.NbaseD

Nbase_0form    =  [NbaseN[0], NbaseN[1], NbaseN[2]]
Nbase_1form    = [[NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]]]
Nbase_2form    = [[NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]]]
Nbase_3form    =  [NbaseD[0], NbaseD[1], NbaseD[2]]

Ntot_0form     =  NbaseN[0]*NbaseN[1]*NbaseN[2] 
Ntot_1form     = [NbaseD[0]*NbaseN[1]*NbaseN[2], NbaseN[0]*NbaseD[1]*NbaseN[2], NbaseN[0]*NbaseN[1]*NbaseD[2]]
Ntot_2form     = [NbaseN[0]*NbaseN[1]*NbaseD[2], NbaseD[0]*NbaseN[1]*NbaseD[2], NbaseD[0]*NbaseD[1]*NbaseN[2]]  
Ntot_3form     =  NbaseD[0]*NbaseD[1]*NbaseD[2]

if add_PIC == True:

    # delta-f corrections (only MHD process)
    if control == True and mpi_rank == 0:
        cont = cv.terms_control_variate(tensor_space, kind_map, params_map)

    # particle accumulator (all processes)
    acc = pic_accumu.accumulation(tensor_space)
# =======================================================================



# ======= reserve memory for FEM cofficients (all MPI processes) ========
pr     = np.empty(Nbase_0form,    dtype=float)     # bulk pressure FEM coefficients

u1     = np.empty(Nbase_1form[0], dtype=float)     # bulk velocity FEM coefficients (1 - component)
u2     = np.empty(Nbase_1form[1], dtype=float)     # bulk velocity FEM coefficients (2 - component)
u3     = np.empty(Nbase_1form[2], dtype=float)     # bulk velocity FEM coefficients (3 - component)

u1_old = np.empty(Nbase_1form[0], dtype=float)     # bulk velocity FEM coefficients from previous time step (1 - component)
u2_old = np.empty(Nbase_1form[1], dtype=float)     # bulk velocity FEM coefficients from previous time step (2 - component)
u3_old = np.empty(Nbase_1form[2], dtype=float)     # bulk velocity FEM coefficients from previous time step (3 - component)

b1     = np.empty(Nbase_2form[0], dtype=float)     # magnetic field FEM coefficients (1 - component)
b2     = np.empty(Nbase_2form[1], dtype=float)     # magnetic field FEM coefficients (2 - component)
b3     = np.empty(Nbase_2form[2], dtype=float)     # magnetic field FEM coefficients (3 - component)

rh     = np.empty(Nbase_3form,    dtype=float)     # bulk mass density FEM coefficients
# =======================================================================


# ==== reserve memory for implicit particle-coupling sub-steps ==========
mat11_loc = np.empty((NbaseD[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat12_loc = np.empty((NbaseD[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat13_loc = np.empty((NbaseD[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat22_loc = np.empty((NbaseN[0], NbaseD[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat23_loc = np.empty((NbaseN[0], NbaseD[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat33_loc = np.empty((NbaseN[0], NbaseN[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)

vec1_loc  = np.empty((NbaseD[0], NbaseN[1], NbaseN[2]), dtype=float)
vec2_loc  = np.empty((NbaseN[0], NbaseD[1], NbaseN[2]), dtype=float)
vec3_loc  = np.empty((NbaseN[0], NbaseN[1], NbaseD[2]), dtype=float)

if mpi_rank == 0:
    mat11 = np.empty((NbaseD[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat12 = np.empty((NbaseD[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat13 = np.empty((NbaseD[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat22 = np.empty((NbaseN[0], NbaseD[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat23 = np.empty((NbaseN[0], NbaseD[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat33 = np.empty((NbaseN[0], NbaseN[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)

    vec1  = np.empty((NbaseD[0], NbaseN[1], NbaseN[2]), dtype=float)
    vec2  = np.empty((NbaseN[0], NbaseD[1], NbaseN[2]), dtype=float)
    vec3  = np.empty((NbaseN[0], NbaseN[1], NbaseD[2]), dtype=float)
else:
    mat11, mat12, mat13, mat22, mat23, mat33 = None, None, None, None, None, None
    vec1,  vec2,  vec3                       = None, None, None
# =======================================================================


if mpi_rank == 0:
    # =================== some diagnostics ====================================
    
    # energies (bulk kinetic energy, magnetic energy, bulk internal energy, hot ion kinetic + internal energy (delta f))
    energies  = {'en_U' : 0., 'en_B' : 0., 'en_p' : 0., 'en_deltaf' : 0.}

    # snapshots of distribution function via particle binning
    fh        = {'fh_eta1_vx' : np.empty((n_bins[0], n_bins[1]), dtype=float)}
    # =========================================================================


    # ============= projection of initial conditions ==========================
    # create object for projecting initial conditions
    pro = proj.projectors_local_3d(tensor_space, nq_pr)

    pr[:, :, :]                           = pro.pi_0( None,               1,        kind_map, params_map)
    u1[:, :, :], u2[:, :, :], u3[:, :, :] = pro.pi_1([None, None, None], [2, 3, 4], kind_map, params_map) 
    b1[:, :, :], b2[:, :, :], b3[:, :, :] = pro.pi_2([None, None, None], [5, 6, 7], kind_map, params_map)
    rh[:, :, :]                           = pro.pi_3( None,               8,        kind_map, params_map)

    del pro

    """
    amps = np.random.rand(8, pr.shape[0], pr.shape[1])

    for k in range(pr.shape[2]):
        pr[:, :, k] = amps[0]

        u1[:, :, k] = amps[1]
        u2[:, :, k] = amps[2]
        u3[:, :, k] = amps[3]

        b1[:, :, :] = 0.
        b2[:, :, :] = 0.
        b3[:, :, k] = amps[6]

        rh[:, :, k] = amps[7]
    """
    
    print('rank : ', mpi_rank, ' projection of initial conditions done!')
    # ==========================================================================


    # ==================== matrices ============================================
    # create object for projecting MHD matrices
    MHD = mhd.projectors_local_mhd(tensor_space, nq_pr)

    # mass matrices in V0, V1 and V2
    M0 = mass.mass_V0(tensor_space, 0, kind_map, params_map)
    M1 = mass.mass_V1(tensor_space, 0, kind_map, params_map)
    M2 = mass.mass_V2(tensor_space, 0, kind_map, params_map)

    print('rank : ', mpi_rank, ' mass matrices done!')

    # normalization vector in V0 (for bulk thermal energy)
    norm_0form = inner.inner_prod_V0(tensor_space, lambda eta1, xi2, xi3 : np.ones(eta1.shape), 0, kind_map, params_map).flatten()

    # discrete grad, curl and div matrices
    derivatives = der.discrete_derivatives(tensor_space)

    GRAD = derivatives.grad_3d()
    CURL = derivatives.curl_3d()
    DIV  = derivatives.div_3d()

    print('rank : ', mpi_rank, ' discrete derivatives done!')

    # projection matrices
    Q   = MHD.projection_Q(kind_map, params_map)     # pi_2[rho_eq * g_inv * lambda^1]
    W   = MHD.projection_W(kind_map, params_map)     # pi_1[rho_eq/g_sqrt * lambda^1]
    TAU = MHD.projection_T(kind_map, params_map)     # pi_1[b_eq * g_inv * lambda^1]
    S   = MHD.projection_S(kind_map, params_map)     # pi_1[p_eq * lambda^1]
    K   = MHD.projection_K(kind_map, params_map)     # pi_0[p_eq * lambda^0]  
    P   = MHD.projection_P(kind_map, params_map)     # pi_1[curl(b_eq) * lambda^2]

    print('rank : ', mpi_rank, ' projection matrices done!')

    # compute matrix A
    A = 1/2*(M1.dot(W) + W.T.dot(M1)).tocsc()

    del W

    # matrices for step 2
    S2      = (A + dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tocsc()
    STEP2_1 = (A - dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tocsc()
    STEP2_2 = dt*TAU.T.dot(CURL.T.dot(M2)).tocsc()

    S2_ILU  = spa.linalg.spilu(S2)

    # matrices for step 6
    L       = GRAD.T.dot(M1).dot(S) + (gamma - 1)*K.T.dot(GRAD.T).dot(M1)

    del S, K

    S6      = spa.bmat([[A,  dt/2*M1.dot(GRAD)], [-dt/2*L, M0]]).tocsc()
    STEP6   = spa.bmat([[A, -dt/2*M1.dot(GRAD)], [ dt/2*L, M0]]).tocsc()

    S6_ILU  = spa.linalg.spilu(S6)

    del MHD, M0, L, GRAD

    print('rank : ', mpi_rank, ' assembly of constant matrices done!')
    # ==========================================================================
    
    
# ======================== create particles ======================================
if   loading == 'pseudo-random':
    # pseudo-random numbers between (0, 1)
    np.random.seed(38)
    
    for i in range(mpi_size):
        temp = np.random.rand(Np_loc, 6)
        
        if i == mpi_rank:
            particles_loc[:, :6] = temp
            break
           
    #particles_loc[:, :6] = np.random.rand(Np_loc, 6)

elif loading == 'sobol_standard':
    # plain sobol numbers between (0, 1) (skip first 1000 numbers)
    particles_loc[:, :6] = sobol.i4_sobol_generate(6, Np_loc, 1000) 

elif loading == 'sobol_antithetic':
    # symmetric sobol numbers between (0, 1) (skip first 1000 numbers) in all 6 dimensions
    pic_sample.set_particles_symmetric(sobol.i4_sobol_generate(6, int(Np_loc/64), 1000), particles_loc)  

elif loading == 'pr_space_uni_velocity':
    # pseudo-random numbers in space and uniform in velocity space
    particles_loc[:, :3] = np.random.rand(Np_loc, 3)

    dv = 1/Np_loc
    particles_loc[:,  3] = np.linspace(dv, 1 - dv, Np_loc)
    particles_loc[:,  4] = np.linspace(dv, 1 - dv, Np_loc)
    particles_loc[:,  5] = np.linspace(dv, 1 - dv, Np_loc)

elif loading == 'external':
    # load numbers between (0, 1) from an external file
    particles_loc[:, :6] = np.load('initial_particles.npy')

else:
    print('particle loading not specified')


# inversion of cumulative distribution function
particles_loc[:, 3]  = sp.erfinv(2*particles_loc[:, 3] - 1)*vth + v0x
particles_loc[:, 4]  = sp.erfinv(2*particles_loc[:, 4] - 1)*vth + v0y
particles_loc[:, 5]  = sp.erfinv(2*particles_loc[:, 5] - 1)*vth + v0z


# compute initial weights
pic_sample.compute_weights_ini(particles_loc, w0_loc, s0_loc, kind_map, params_map)

if control == True:
    pic_sample.update_weights(particles_loc, w0_loc, s0_loc, kind_map, params_map)
else:
    particles_loc[:, 6] = w0_loc

#print(mpi_rank, 'particle initialization done!')
# ======================================================================================


# ========= compute initial fields at particle positions and initial energies ==========
# broadcast FEM coeffiecients from zeroth rank
mpi_comm.Bcast(u1, root=0)
mpi_comm.Bcast(u2, root=0)
mpi_comm.Bcast(u3, root=0)

mpi_comm.Bcast(b1, root=0)
mpi_comm.Bcast(b2, root=0)
mpi_comm.Bcast(b3, root=0)

# compute fields
if add_PIC == True:
    timea = time.time()
    pic_fields.evaluate_1form(particles_loc[:, 0:3], T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, u1, u2, u3, U_part_loc, kind_map, params_map)
    pic_fields.evaluate_2form(particles_loc[:, 0:3], T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, B_part_loc, kind_map, params_map)
    timeb = time.time()
    #print(mpi_rank, 'initial field computation at particles done. Time : ', timeb-timea)


# initial energies (MHD)
if mpi_rank == 0:
    energies['en_U'] = 1/2*np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())).dot(A.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))))
    energies['en_B'] = 1/2*np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())).dot(M2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))))
    energies['en_p'] = 1/(gamma - 1)*pr.flatten().dot(norm_0form)


# initial energies (hot ions)
en_deltaf_loc[0] = 1/2*particles_loc[:, 6].dot(particles_loc[:, 3]**2 + particles_loc[:, 4]**2 + particles_loc[:, 5]**2)/Np
mpi_comm.Reduce(en_deltaf_loc, en_deltaf, op=MPI.SUM, root=0)

if mpi_rank == 0:
    energies['en_deltaf'] = en_deltaf[0] + (control - 1)*eq_PIC.eh_eq(kind_map, params_map)


# initial distribution function
fh_eta1_vx_loc[:, :] = np.histogram2d(particles_loc[:, 0], particles_loc[:, 3], bins=bin_edges, weights=particles_loc[:, 6], normed=False)[0]/(Np*dbin[0]*dbin[1])
mpi_comm.Reduce(fh_eta1_vx_loc, fh_eta1_vx, op=MPI.SUM, root=0)

if mpi_rank == 0:
    fh['fh_eta1_vx'][:, :] = fh_eta1_vx
# =====================================================================================================================

    

# ==================== time integrator ==========================================
times_elapsed = {'total'               : 0.,
                 'accumulation_step1'  : 0., 
                 'accumulation_step3'  : 0., 
                 'evaluation_1form'    : 0., 
                 'evaluation_2form'    : 0., 
                 'pusher_step3'        : 0., 
                 'pusher_step4'        : 0., 
                 'pusher_step5'        : 0.,
                 'control_step1'       : 0.,
                 'control_step3'       : 0.,
                 'control_weights'     : 0.,
                 'update_step1u'       : 0.,
                 'update_step2u'       : 0.,
                 'update_step2b'       : 0.,
                 'update_step3u'       : 0.,
                 'update_step6'        : 0.}

    
def update():
    
    global u1, u2, u3
    global u1_old, u2_old, u3_old
    global b1, b2, b3
    global pr, rh
    global particles_loc
    global U_part_loc, B_part_loc
    
    time_tota = time.time()
    
    # ================== step 1 (1 : update u) ====================
    if add_PIC == True:
        
        # charge accumulation
        timea   = time.time()
        
        pic_accumu_ker.kernel_step1(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, B_part_loc, kind_map, params_map, mat12_loc, mat13_loc, mat23_loc)
        
        mpi_comm.Reduce(mat12_loc, mat12, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat13_loc, mat13, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat23_loc, mat23, op=MPI.SUM, root=0)
        
        timeb = time.time()
        times_elapsed['accumulation_step1'] = timeb - timea
        
        if mpi_rank == 0:
            
            # build global sparse matrix
            mat = -acc.to_sparse_step1(mat12, mat13, mat23)/Np

            # delta-f correction
            if control == True:
                timea = time.time()
                mat  -= cont.mass_V1_nh_eq([b1, b2, b3])
                timeb = time.time()
                times_elapsed['control_step1'] = timeb - timea
        
            # solve linear system
            timea = time.time()
            temp1, temp2, temp3 = np.split(spa.linalg.spsolve(A - dt*mat/2, (A + dt*mat/2).dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())))), [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
            timeb = time.time()
            times_elapsed['update_step1u'] = timeb - timea

            u1[:, :, :] = temp1.reshape(Nbase_1form[0])
            u2[:, :, :] = temp2.reshape(Nbase_1form[1])
            u3[:, :, :] = temp3.reshape(Nbase_1form[2])
    # ==============================================================                               
    
    
    # ====== step 2 (1 : update u, 2 : update b, 3 : evaluate B-field at particle positions) =============
    if mpi_rank == 0:
        
        u1_old[:, :, :] = u1[:, :, :]
        u2_old[:, :, :] = u2[:, :, :]
        u3_old[:, :, :] = u3[:, :, :]
        
        # solve linear system with conjugate gradient method with an incomplete LU decomposition as preconditioner and values from last time step as initial guess
        timea = time.time()
        temp1, temp2, temp3 = np.split(spa.linalg.cg(S2, STEP2_1.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) + STEP2_2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=1e-8, M=spa.linalg.LinearOperator(S2.shape, lambda x : S2_ILU.solve(x)))[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
        timeb = time.time()
        times_elapsed['update_step2u'] = timeb - timea

        u1[:, :, :] = temp1.reshape(Nbase_1form[0])
        u2[:, :, :] = temp2.reshape(Nbase_1form[1])
        u3[:, :, :] = temp3.reshape(Nbase_1form[2])

        timea = time.time()
        temp1, temp2, temp3 = np.split(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())) - dt/2*CURL.dot(TAU.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())) + np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten())))), [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]]) 
        timeb = time.time()
        times_elapsed['update_step2b'] = timeb - timea

        b1[:, :, :] = temp1.reshape(Nbase_2form[0])
        b2[:, :, :] = temp2.reshape(Nbase_2form[1])
        b3[:, :, :] = temp3.reshape(Nbase_2form[2])
    
    # broadcast new FEM coefficients and evaluate B-field at particle positions
    mpi_comm.Bcast(b1, root=0)
    mpi_comm.Bcast(b2, root=0)
    mpi_comm.Bcast(b3, root=0)
    
    if add_PIC == True:
        pic_fields.evaluate_2form(particles_loc[:, 0:3], T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, B_part_loc, kind_map, params_map)
    # ====================================================================================================
    
    
    # ===== step 3 (1 : update u, 2 : evaluate U-field at particle positions, 3 : update particles velocities (V)) =====
    if add_PIC == True:
        
        # current accumulation
        timea = time.time()
        
        pic_accumu_ker.kernel_step3(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, B_part_loc, kind_map, params_map, mat11_loc, mat12_loc, mat13_loc, mat22_loc, mat23_loc, mat33_loc, vec1_loc, vec2_loc, vec3_loc)
        
        mpi_comm.Reduce(mat11_loc, mat11, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat12_loc, mat12, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat13_loc, mat13, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat22_loc, mat22, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat23_loc, mat23, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat33_loc, mat33, op=MPI.SUM, root=0)

        mpi_comm.Reduce(vec1_loc , vec1 , op=MPI.SUM, root=0)
        mpi_comm.Reduce(vec2_loc , vec2 , op=MPI.SUM, root=0)
        mpi_comm.Reduce(vec3_loc , vec3 , op=MPI.SUM, root=0)
                   
        timeb = time.time()
        times_elapsed['accumulation_step3'] = timeb - timea

        
        if mpi_rank == 0:
            
            u1_old[:, :, :] = u1[:, :, :]
            u2_old[:, :, :] = u2[:, :, :]
            u3_old[:, :, :] = u3[:, :, :]
            
            # build global sparse matrix
            mat = acc.to_sparse_step3(mat11, mat12, mat13, mat22, mat23, mat33)/Np
        
            # delta-f update
            if control == True:
                timea  = time.time()
                vec_cv = cont.inner_prod_V1_jh_eq([b1, b2, b3])
                timeb  = time.time()
                times_elapsed['control_step3'] = timeb - timea

                timea = time.time()
                temp1, temp2, temp3 = np.split(spa.linalg.spsolve(A + dt**2*mat/4, (A - dt**2*mat/4).dot(np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten()))) + dt*np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))/Np + dt*np.concatenate((vec_cv[0].flatten(), vec_cv[1].flatten(), vec_cv[2].flatten()))), [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]]) 
                timeb = time.time()
                times_elapsed['update_step3u'] = timeb - timea

                u1[:, :, :] = temp1.reshape(Nbase_1form[0])
                u2[:, :, :] = temp2.reshape(Nbase_1form[1])
                u3[:, :, :] = temp3.reshape(Nbase_1form[2])

            # full-f update
            else: 
                timea = time.time()
                temp1, temp2, temp3 = np.split(spa.linalg.spsolve(A + dt**2*mat/4, (A - dt**2*mat/4).dot(np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten()))) + dt*np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))/Np), [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]]) 
                timeb = time.time()
                times_elapsed['update_step3u'] = timeb - timea

                u1[:, :, :] = temp1.reshape(Nbase_1form[0])
                u2[:, :, :] = temp2.reshape(Nbase_1form[1])
                u3[:, :, :] = temp3.reshape(Nbase_1form[2])

        
        # broadcast new FEM coefficients
        mpi_comm.Bcast(u1    , root=0)
        mpi_comm.Bcast(u2    , root=0)
        mpi_comm.Bcast(u3    , root=0)
        
        mpi_comm.Bcast(u1_old, root=0)
        mpi_comm.Bcast(u2_old, root=0)
        mpi_comm.Bcast(u3_old, root=0)
        
        timea = time.time()
        pic_fields.evaluate_1form(particles_loc[:, 0:3], T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, (u1 + u1_old)/2, (u2 + u2_old)/2, (u3 + u3_old)/2, U_part_loc, kind_map, params_map)
        timeb = time.time()
        times_elapsed['evaluation_1form'] = timeb - timea
        
        timea = time.time()
        pic_pusher.pusher_step3(particles_loc, dt, B_part_loc, U_part_loc, kind_map, params_map)
        timeb = time.time()
        times_elapsed['pusher_step3'] = timeb - timea
    # ====================================================================================================
    
    
    # ====== step 4 (1 : update particles positions (Xi)) ===============
    if add_PIC == True:
        timea = time.time()
        pic_pusher.pusher_step4(particles_loc, dt, kind_map, params_map)
        timeb = time.time()
        times_elapsed['pusher_step4'] = timeb - timea
    # ===================================================================

    
    # ========= step 5 (1 : update particle veclocities (V), 2 : update particle weights (W)) ===============
    if add_PIC == True:
        timea = time.time()
        pic_fields.evaluate_2form(particles_loc[:, 0:3], T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, B_part_loc, kind_map, params_map)
        timeb = time.time()
        times_elapsed['evaluation_2form'] = timeb - timea
        
        timea = time.time()
        pic_pusher.pusher_step5(particles_loc, dt, B_part_loc, kind_map, params_map)
        timeb = time.time()
        times_elapsed['pusher_step5'] = timeb - timea

        if control == True:
            timea = time.time()
            pic_sample.update_weights(particles, w0, s0, kind_map, params_map)
            timeb = time.time()
            times_elapsed['control_weights'] = timeb - timea
    # =======================================================================================================
    
    
    # ============== step 6 (1 : update rh, u and pr from non - Hamiltonian MHD terms) ======================
    if add_pressure == True and mpi_rank == 0:

        timea = time.time()

        # solve linear system with conjugate gradient method with an incomplete LU decomposition as preconditioner and values from last time step as initial guess
        u1_new, u2_new, u3_new, pr_new = np.split(spa.linalg.cg(S6, STEP6.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten(), pr.flatten())) + np.concatenate((dt*M1.dot(P).dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))), np.zeros(Ntot_0form)))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten(), pr.flatten())), tol=1e-8, M=spa.linalg.LinearOperator(S6.shape, lambda x : S6_ILU.solve(x)))[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1], Ntot_1form[0] + Ntot_1form[1] + Ntot_1form[2]])

        # update density
        rh[:, :, :] = rh - dt/2*(DIV.dot(Q).dot(np.concatenate((u1_new, u2_new, u3_new)) + np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())))).reshape(Nbase_3form)

        # update pressure
        pr[:, :, :]  = pr_new.reshape(Nbase_0form)

        # update velocity
        u1[:, :, :]  = u1_new.reshape(Nbase_1form[0])
        u2[:, :, :]  = u2_new.reshape(Nbase_1form[1])
        u3[:, :, :]  = u3_new.reshape(Nbase_1form[2])

        timeb = time.time()
        times_elapsed['update_step6'] = timeb - timea
    # =======================================================================================================
    
        
    time_totb = time.time()
    times_elapsed['total'] = time_totb - time_tota
    
    # diagnostics: energies (MHD)
    if mpi_rank == 0:
        energies['en_U'] = 1/2*np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())).dot(A.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))))
        energies['en_B'] = 1/2*np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())).dot(M2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))))
        energies['en_p'] = 1/(gamma - 1)*pr.flatten().dot(norm_0form)


    # diagnostics: energies (hot ions)
    en_deltaf_loc[0] = 1/2*particles_loc[:, 6].dot(particles_loc[:, 3]**2 + particles_loc[:, 4]**2 + particles_loc[:, 5]**2)/Np
    mpi_comm.Reduce(en_deltaf_loc, en_deltaf, op=MPI.SUM, root=0)

    if mpi_rank == 0:
        energies['en_deltaf'] = en_deltaf[0] + (control - 1)*eq_PIC.eh_eq(kind_map, params_map)


    # diagnostics: distribution function
    fh_eta1_vx_loc[:, :] = np.histogram2d(particles_loc[:, 0], particles_loc[:, 3], bins=bin_edges, weights=particles_loc[:, 6], normed=False)[0]/(Np*dbin[0]*dbin[1])
    mpi_comm.Reduce(fh_eta1_vx_loc, fh_eta1_vx, op=MPI.SUM, root=0)

    if mpi_rank == 0:
        fh['fh_eta1_vx'][:, :] = fh_eta1_vx
# ============================================================================

mpi_comm.Barrier()
timea = time.time()
for i in range(20):
    #if mpi_rank == 0:
        #print(i, energies)
    update()
timeb = time.time()
print(mpi_rank, (timeb - timea)/20)
sys.exit()


# ========================== time integration ================================
if time_int == True:
    
    if mpi_rank == 0:
        # a new simulation
        if restart == False:

            # create hdf5 file and datasets for simulation output
            file = h5py.File('results_' + identifier + '.hdf5', 'a')

            file.create_dataset('time', (1,),   maxshape=(None,),   dtype=float, chunks=True)

            file.create_dataset('energies/bulk_kinetic',     (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/magnetic',         (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/bulk_internal',    (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/energetic_deltaf', (1,), maxshape=(None,), dtype=float, chunks=True)

            file.create_dataset('times_elapsed/total',              (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/accumulation_step1', (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/accumulation_step3', (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/evaluation_1form',   (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/evaluation_2form',   (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/pusher_step3',       (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/pusher_step4',       (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/pusher_step5',       (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/control_step1',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/control_step3',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/control_weights',    (1,), maxshape=(None,),   dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step1u',      (1,), maxshape=(None,),   dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step2u',      (1,), maxshape=(None,),   dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step2b',      (1,), maxshape=(None,),   dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step3u',      (1,), maxshape=(None,),   dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step6',       (1,), maxshape=(None,),   dtype=float, chunks=True)


            file.create_dataset('pressure',                   (1, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]),    maxshape=(None, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]),    dtype=float, chunks=True)
            file.create_dataset('velocity_field/1_component', (1, Nbase_1form[0][0], Nbase_1form[0][1], Nbase_1form[0][2]), maxshape=(None, Nbase_1form[0][0], Nbase_1form[0][1], Nbase_1form[0][2]), dtype=float, chunks=True)
            file.create_dataset('velocity_field/2_component', (1, Nbase_1form[1][0], Nbase_1form[1][1], Nbase_1form[1][2]), maxshape=(None, Nbase_1form[1][0], Nbase_1form[1][1], Nbase_1form[1][2]), dtype=float, chunks=True)
            file.create_dataset('velocity_field/3_component', (1, Nbase_1form[2][0], Nbase_1form[2][1], Nbase_1form[2][2]), maxshape=(None, Nbase_1form[2][0], Nbase_1form[2][1], Nbase_1form[2][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/1_component', (1, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), maxshape=(None, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/2_component', (1, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), maxshape=(None, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/3_component', (1, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), maxshape=(None, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), dtype=float, chunks=True)
            file.create_dataset('density',                    (1, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    dtype=float, chunks=True)

            file.create_dataset('bulk_mass', (1,), maxshape=(None,), dtype=float, chunks=True)

            file.create_dataset('magnetic_field/divergence',  (1, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    dtype=float, chunks=True)

            file.create_dataset('particles', (1, Np, 7), maxshape=(None, Np, 7), dtype=float, chunks=True)
            file.create_dataset('distribution_function/xi1_vx', (1, n_bins[0], n_bins[1]), maxshape=(None, n_bins[0], n_bins[1]), dtype=float, chunks=True)


            # datasets for restart function
            file.create_dataset('restart/time_steps_done', (1,), maxshape=(None,), dtype=int, chunks=True)

            file.create_dataset('restart/pressure',                   (1, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]),    maxshape=(None, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]),    dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/1_component', (1, Nbase_1form[0][0], Nbase_1form[0][1], Nbase_1form[0][2]), maxshape=(None, Nbase_1form[0][0], Nbase_1form[0][1], Nbase_1form[0][2]), dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/2_component', (1, Nbase_1form[1][0], Nbase_1form[1][1], Nbase_1form[1][2]), maxshape=(None, Nbase_1form[1][0], Nbase_1form[1][1], Nbase_1form[1][2]), dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/3_component', (1, Nbase_1form[2][0], Nbase_1form[2][1], Nbase_1form[2][2]), maxshape=(None, Nbase_1form[2][0], Nbase_1form[2][1], Nbase_1form[2][2]), dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/1_component', (1, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), maxshape=(None, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/2_component', (1, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), maxshape=(None, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/3_component', (1, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), maxshape=(None, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), dtype=float, chunks=True)
            file.create_dataset('restart/density',                    (1, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    dtype=float, chunks=True)

            file.create_dataset('restart/particles', (1, Np, 7), maxshape=(None, Np, 7), dtype=float, chunks=True)

            file.create_dataset('restart/control_w0', (Np,), dtype=float)
            file.create_dataset('restart/control_s0', (Np,), dtype=float)


            # == save initial data ============
            file['time'][0] = 0.

            file['energies/bulk_kinetic'][0]     = energies['en_U']
            file['energies/magnetic'][0]         = energies['en_B']
            file['energies/bulk_internal'][0]    = energies['en_p']
            file['energies/energetic_deltaf'][0] = energies['en_deltaf']

            file['pressure'][0]                   = pr
            file['velocity_field/1_component'][0] = u1
            file['velocity_field/2_component'][0] = u2
            file['velocity_field/3_component'][0] = u3
            file['magnetic_field/1_component'][0] = b1
            file['magnetic_field/2_component'][0] = b2
            file['magnetic_field/3_component'][0] = b3
            file['density'][0]                    = rh

            file['magnetic_field/divergence'][0] = DIV.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))).reshape(Nbase_3form[0], Nbase_3form[1], Nbase_3form[2])

            file['bulk_mass'][0] = sum(rh.flatten())

            #file['particles'][0] = particles
            #file['distribution_function/xi1_vx'][0] = fh['fh_xi1_vx']

            #file['restart/control_w0'][:] = w0
            #file['restart/control_s0'][:] = s0
            # =================================

            #print('initial energies : ', energies)
            #time_steps_done = 0

        # restarting another simulation
        else:

            # open existing hdf5 file
            file = h5py.File('results_' + identifier  + '.hdf5', 'a')

            # load restart data from last time step
            time_steps_done = file['restart/time_steps_done'][num_restart]

            pr[:, :, :]     = file['restart/pressure'][num_restart]
            u1[:, :, :]     = file['restart/velocity_field/1_component'][num_restart]
            u2[:, :, :]     = file['restart/velocity_field/2_component'][num_restart]
            u3[:, :, :]     = file['restart/velocity_field/3_component'][num_restart]
            b1[:, :, :]     = file['restart/magnetic_field/1_component'][num_restart]
            b2[:, :, :]     = file['restart/magnetic_field/2_component'][num_restart]
            b3[:, :, :]     = file['restart/magnetic_field/3_component'][num_restart]
            rh[:, :, :]     = file['restart/density'][num_restart]

            particles[:, :] = file['restart/particles'][num_restart]
            w0[:]           = file['restart/control_w0'][:]
            s0[:]           = file['restart/control_s0'][:]

            # perform initialization for next time step: field evaluation at particle positions
            pic_fields.evaluate_1form(particles[:, 0:3], T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np, u1, u2, u3, U_part, kind_map, params_map)
            pic_fields.evaluate_2form(particles[:, 0:3], T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np, b1, b2, b3, B_part, kind_map, params_map)

            # perform initialization for next time step: compute partice weights
            if control == True:
                pic_sample.update_weights(particles, w0, s0, kind_map, params_map)
            else:
                particles[:, 6] = w0

    time_steps_done = 0
    
    
    
    
    # ========================================================================================
    #              time loop 
    # ========================================================================================
    #print('start time integration! (total number of time steps : ' + str(int(Tend/dt)) + ')')
    # ========================================================================================
    while True:

        if (time_steps_done*dt >= Tend) or ((time.time() - start_simulation)/60 > max_time):
            
            if mpi_rank == 0:
                if create_restart:

                    file['restart/time_steps_done'][-1]            = time_steps_done
                    file['restart/pressure'][-1]                   = pr
                    file['restart/velocity_field/1_component'][-1] = u1
                    file['restart/velocity_field/2_component'][-1] = u2
                    file['restart/velocity_field/3_component'][-1] = u3
                    file['restart/magnetic_field/1_component'][-1] = b1
                    file['restart/magnetic_field/2_component'][-1] = b2
                    file['restart/magnetic_field/3_component'][-1] = b3
                    file['restart/density'][-1]                    = rh
                    file['restart/particles'][-1]                  = particles

                    file['restart/time_steps_done'].resize(file['restart/time_steps_done'].shape[0] + 1, axis = 0)
                    file['restart/pressure'].resize(file['restart/pressure'].shape[0] + 1, axis = 0)
                    file['restart/velocity_field/1_component'].resize(file['restart/velocity_field/1_component'].shape[0] + 1, axis = 0)
                    file['restart/velocity_field/2_component'].resize(file['restart/velocity_field/2_component'].shape[0] + 1, axis = 0)
                    file['restart/velocity_field/3_component'].resize(file['restart/velocity_field/3_component'].shape[0] + 1, axis = 0)
                    file['restart/magnetic_field/1_component'].resize(file['restart/magnetic_field/1_component'].shape[0] + 1, axis = 0)
                    file['restart/magnetic_field/2_component'].resize(file['restart/magnetic_field/2_component'].shape[0] + 1, axis = 0)
                    file['restart/magnetic_field/3_component'].resize(file['restart/magnetic_field/3_component'].shape[0] + 1, axis = 0)
                    file['restart/density'].resize(file['restart/density'].shape[0] + 1, axis = 0)
                    file['restart/particles'].resize(file['restart/particles'].shape[0] + 1, axis = 0)
            
            break

        if mpi_rank == 0:
            if time_steps_done%1 == 0:
                print('time steps finished : ' + str(time_steps_done))
                print('energies : ', energies)

        # === time integration =======
        update()
        time_steps_done += 1
        # ============================
        
        #if time_steps_done == 1:
        if mpi_rank == 0:
            print('time for current time step : ', times_elapsed['total'])

        if mpi_rank == 0:
            # == data to save ==========
            file['time'].resize(file['time'].shape[0] + 1, axis = 0)
            file['time'][-1] = time_steps_done*dt

            file['energies/bulk_kinetic'].resize(file['energies/bulk_kinetic'].shape[0] + 1, axis = 0)
            file['energies/magnetic'].resize(file['energies/magnetic'].shape[0] + 1, axis = 0)
            file['energies/bulk_internal'].resize(file['energies/bulk_internal'].shape[0] + 1, axis = 0)
            file['energies/energetic_deltaf'].resize(file['energies/energetic_deltaf'].shape[0] + 1, axis = 0)
            file['energies/bulk_kinetic'][-1]     = energies['en_U']
            file['energies/magnetic'][-1]         = energies['en_B']
            file['energies/bulk_internal'][-1]    = energies['en_p']
            file['energies/energetic_deltaf'][-1] = energies['en_deltaf']

            file['times_elapsed/total'].resize(file['times_elapsed/total'].shape[0] + 1, axis = 0)
            file['times_elapsed/accumulation_step1'].resize(file['times_elapsed/accumulation_step1'].shape[0] + 1, axis = 0)
            file['times_elapsed/accumulation_step3'].resize(file['times_elapsed/accumulation_step3'].shape[0] + 1, axis = 0)
            file['times_elapsed/evaluation_1form'].resize(file['times_elapsed/evaluation_1form'].shape[0] + 1, axis = 0)
            file['times_elapsed/evaluation_2form'].resize(file['times_elapsed/evaluation_2form'].shape[0] + 1, axis = 0)
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
            file['times_elapsed/evaluation_1form'][-1]   = times_elapsed['evaluation_1form']
            file['times_elapsed/evaluation_2form'][-1]   = times_elapsed['evaluation_2form']
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


            #file['pressure'].resize(file['pressure'].shape[0] + 1, axis = 0)
            #file['pressure'][-1] = pr

            file['magnetic_field/1_component'].resize(file['magnetic_field/1_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/2_component'].resize(file['magnetic_field/2_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/3_component'].resize(file['magnetic_field/3_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/1_component'][-1] = b1
            file['magnetic_field/2_component'][-1] = b2
            file['magnetic_field/3_component'][-1] = b3

            #file['velocity_field/1_component'].resize(file['velocity_field/1_component'].shape[0] + 1, axis = 0)
            #file['velocity_field/2_component'].resize(file['velocity_field/2_component'].shape[0] + 1, axis = 0)
            #file['velocity_field/3_component'].resize(file['velocity_field/3_component'].shape[0] + 1, axis = 0)
            #file['velocity_field/1_component'][-1] = u1
            #file['velocity_field/2_component'][-1] = u2
            #file['velocity_field/3_component'][-1] = u3

            #file['density'].resize(file['density'].shape[0] + 1, axis = 0)
            #file['density'][-1] = rh

            #file['magnetic_field/divergence'].resize(file['magnetic_field/divergence'].shape[0] + 1, axis = 0)
            #file['magnetic_field/divergence'][-1] = DIV.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))).reshape(Nbase_3form[0], Nbase_3form[1], Nbase_3form[2])

            file['bulk_mass'].resize(file['bulk_mass'].shape[0] + 1, axis = 0)
            file['bulk_mass'][-1] = sum(rh.flatten())

            #if time_steps_done%10 == 0:
             #   file['particles'].resize(file['particles'].shape[0] + 1, axis = 0)
              #  file['particles'][-1] = particles

            #file['distribution_function/xi1_vx'].resize(file['distribution_function/xi1_vx'].shape[0] + 1, axis = 0)
            #file['distribution_function/xi1_vx'][-1] = fh['fh_xi1_vx']
            # ==========================
    
    if mpi_rank == 0:
        file.close()
# ============================================================================