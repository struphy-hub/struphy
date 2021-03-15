# load mpi4py and create communicator
from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

# synchronize MPI processes and set start of simulation
import time
mpi_comm.Barrier()
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
import hylife.utilitis_PIC.accumulation              as pic_accumu


# load local source files
import source_run.projectors_local     as proj
import source_run.projectors_local_mhd as mhd
import source_run.control_variate      as cv
import source_run.pusher               as pic_pusher
import source_run.accumulation_kernels as pic_accumu_ker
import source_run.sampling             as pic_sample


# load local input files
import input_run.equilibrium_PIC as eq_PIC


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
mapping        = params['mapping']

kind_map       = params['kind_map']
params_map     = params['params_map']

Nel_F          = params['Nel_F']
bc_F           = params['bc_F']
p_F            = params['p_F']

# general
add_pressure   = params['add_pressure']
gamma          = params['gamma']

# ILUs
drop_tol_S2    = params['drop_tol_S2']
fill_fac_S2    = params['fill_fac_S2']

drop_tol_A     = params['drop_tol_A']
fill_fac_A     = params['fill_fac_A']

drop_tol_M0    = params['drop_tol_M0']
fill_fac_M0    = params['fill_fac_M0']

# tolerances for iterative solvers
tol1           = params['tol1']
tol2           = params['tol2']
tol3           = params['tol3']
tol6           = params['tol6']

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
seed           = params['seed']

# restart function
restart        = params['restart']
num_restart    = params['num_restart']
create_restart = params['create_restart']
# ========================================================================




# ================= MPI initialization for particles =====================
Np_loc         = int(Np/mpi_size)                      # number of particles for each process

particles_loc  = np.empty((7, Np_loc), dtype=float)    # particles of each process
w0_loc         = np.empty(    Np_loc , dtype=float)    # weights for each process: hat_f_ini(eta_0, v_0)/hat_s_ini(eta_0, v_0)
s0_loc         = np.empty(    Np_loc , dtype=float)    # initial sampling density: hat_s_ini(eta_0, v_0) for each process

if mpi_rank == 0:
    particles_recv = np.empty((7, Np_loc), dtype=float)
    w0_recv        = np.empty(    Np_loc , dtype=float)    
    s0_recv        = np.empty(    Np_loc , dtype=float)
# ========================================================================



# =================== some diagnostics ===================================
# energies (bulk kinetic energy, magnetic energy, bulk internal energy, hot ion kinetic + internal energy (delta f))
energies     = {'U' : np.empty(1, dtype=float), 'B' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float), 'f' : np.empty(1, dtype=float)}

energies_loc = {'U' : np.empty(1, dtype=float), 'B' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float), 'f' : np.empty(1, dtype=float)}

# snapshots of distribution function via particle binning
n_bins       = {'eta1_vx' : [32, 64]}
bin_edges    = {'eta1_vx' : [np.linspace(0., 1., n_bins['eta1_vx'][0] + 1), np.linspace(0., 5., n_bins['eta1_vx'][1] + 1)]}
dbin         = {'eta1_vx' : [bin_edges['eta1_vx'][0][1] - bin_edges['eta1_vx'][0][0], bin_edges['eta1_vx'][1][1] - bin_edges['eta1_vx'][1][0]]}
                
fh           = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
                
fh_loc       = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
# ========================================================================



# ================== basics ==============================================
# element boundaries and spline knot vectors for finite elements (N and D)
el_b = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                      
T    = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]
t    = [T[1:-1] for T in T]

# element boundaries and spline knot vectors for spline mapping (N)
el_b_F = [np.linspace(0., 1., Nel_F + 1) for Nel_F in Nel_F]                      
T_F    = [bsp.make_knots(el_b_F, p_F, bc_F) for el_b_F, p_F, bc_F in zip(el_b_F, p_F, bc_F)]

# 1d B-spline finite element spaces (save evaluated quadrature points only for MHD process)
if mpi_rank == 0:
    spaces   = [spl.spline_space_1d(T  , p  , bc  , nq_el) for T,   p  , bc  , nq_el in zip(T  , p  , bc  , nq_el)]
    spaces_F = [spl.spline_space_1d(T_F, p_F, bc_F       ) for T_F, p_F, bc_F        in zip(T_F, p_F, bc_F       )]
    
else:
    spaces   = [spl.spline_space_1d(T  , p  , bc  ) for T  , p  , bc   in zip(T  , p  , bc  )]
    spaces_F = [spl.spline_space_1d(T_F, p_F, bc_F) for T_F, p_F, bc_F in zip(T_F, p_F, bc_F)]


# 3d tensor-product B-spline spaces for finite elements and mapping
tensor_space   = spl.tensor_spline_space(spaces  )
tensor_space_F = spl.tensor_spline_space(spaces_F)

# number of basis functions in different spaces (finite elements)
NbaseN      = tensor_space.NbaseN
NbaseD      = tensor_space.NbaseD

Nbase_0form =  [NbaseN[0], NbaseN[1], NbaseN[2]]
Nbase_1form = [[NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]]]
Nbase_2form = [[NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]]]
Nbase_3form =  [NbaseD[0], NbaseD[1], NbaseD[2]]

Ntot_0form  =  NbaseN[0]*NbaseN[1]*NbaseN[2] 
Ntot_1form  = [NbaseD[0]*NbaseN[1]*NbaseN[2], NbaseN[0]*NbaseD[1]*NbaseN[2], NbaseN[0]*NbaseN[1]*NbaseD[2]]
Ntot_2form  = [NbaseN[0]*NbaseN[1]*NbaseD[2], NbaseD[0]*NbaseN[1]*NbaseD[2], NbaseD[0]*NbaseD[1]*NbaseN[2]]  
Ntot_3form  =  NbaseD[0]*NbaseD[1]*NbaseD[2]

# number of basis functions for spline mapping
NbaseN_F    = tensor_space_F.NbaseN
# =======================================================================


# ========= geometry in case of spline mapping ==========================
# interpolation of mapping on discrete space with local interpolator pi_0 (get controlpoints)
#Fx = lambda eta1, eta2, eta3 : 2*np.pi/0.8*(eta1 + 0.1*np.sin(2*np.pi*eta1)*np.sin(2*np.pi*eta2))
#Fy = lambda eta1, eta2, eta3 : 2*np.pi/0.8*(eta2 + 0.1*np.sin(2*np.pi*eta1)*np.sin(2*np.pi*eta2))
#Fz = lambda eta1, eta2, eta3 : 1*eta3

Fx = lambda eta1, eta2, eta3 : 2*np.pi/0.8*eta1
Fy = lambda eta1, eta2, eta3 : 2*np.pi/0.8*eta2
Fz = lambda eta1, eta2, eta3 : 1*eta3

pro_F = proj.projectors_local_3d(tensor_space_F, [p_F[0] + 1, p_F[1] + 1, p_F[2] + 1])

cx = pro_F.pi_0(Fx)
cy = pro_F.pi_0(Fy)
cz = pro_F.pi_0(Fz)

del pro_F
# =======================================================================



# ======= particle accumulator (and delta-f corrections) ================
if add_PIC == True:

    # particle accumulator (all processes)
    acc = pic_accumu.accumulation(tensor_space)
    
    # delta-f corrections (only MHD process)
    if control == True and mpi_rank == 0:
        cont = cv.terms_control_variate(tensor_space, acc, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
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
    # ============= projection of initial conditions ==========================
    # create object for projecting initial conditions
    
    pro = proj.projectors_local_3d(tensor_space, nq_pr)
    
    if   mapping == 0:

        pr[:, :, :]                           = pro.pi_0( None,              0,  1,        kind_map, params_map)
        u1[:, :, :], u2[:, :, :], u3[:, :, :] = pro.pi_1([None, None, None], 0, [2, 3, 4], kind_map, params_map) 
        b1[:, :, :], b2[:, :, :], b3[:, :, :] = pro.pi_2([None, None, None], 0, [5, 6, 7], kind_map, params_map)
        rh[:, :, :]                           = pro.pi_3( None,              0,  8,        kind_map, params_map)

    elif mapping == 1:
        
        pr[:, :, :]                           = pro.pi_0( None,              1,  1,        T_F, p_F, NbaseN_F, [cx, cy, cz])
        u1[:, :, :], u2[:, :, :], u3[:, :, :] = pro.pi_1([None, None, None], 1, [2, 3, 4], T_F, p_F, NbaseN_F, [cx, cy, cz]) 
        b1[:, :, :], b2[:, :, :], b3[:, :, :] = pro.pi_2([None, None, None], 1, [5, 6, 7], T_F, p_F, NbaseN_F, [cx, cy, cz])
        rh[:, :, :]                           = pro.pi_3( None,              1,  8,        T_F, p_F, NbaseN_F, [cx, cy, cz])
    
    del pro
    
    
    """
    np.random.seed(1234)
    amps = 1e-3*np.random.rand(8, pr.shape[0], pr.shape[1])

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
    
    print('projection of initial conditions done!')
    
    # ==========================================================================


    # ==================== matrices ============================================
    # mass matrices in V0, V1 and V2
    M0 = mass.mass_V0(tensor_space, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
    M1 = mass.mass_V1(tensor_space, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
    M2 = mass.mass_V2(tensor_space, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
    
    print('mass matrices done!')
    
    # normalization vector in V0 (for bulk thermal energy)
    norm_0form = inner.inner_prod_V0(tensor_space, lambda eta1, eta2, eta3 : np.ones(eta1.shape), mapping, kind_map, params_map, tensor_space_F, cx, cy, cz).flatten() 
    
    # discrete grad, curl and div matrices
    derivatives = der.discrete_derivatives(tensor_space)

    GRAD = derivatives.grad_3d()
    CURL = derivatives.curl_3d()
    DIV  = derivatives.div_3d()
    
    del derivatives

    print('discrete derivatives done!')

    
    # projection matrices
    MHD = mhd.projectors_local_mhd(tensor_space, nq_pr)
    
    Q   = MHD.projection_Q(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_2[rho_eq * g_inv * lambda^1]
    W   = MHD.projection_W(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_1[rho_eq/g_sqrt * lambda^1]
    TAU = MHD.projection_T(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_1[b_eq * g_inv * lambda^1]
    S   = MHD.projection_S(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_1[p_eq * lambda^1]
    K   = MHD.projection_K(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_0[p_eq * lambda^0]  
    P   = MHD.projection_P(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_1[curl(b_eq) * lambda^2]
    
    del MHD
    print('projection matrices done!')
    # ==========================================================================

    
    # ========= compute symmetric matrix A and a ILU preconditioner ============
    A     = 1/2*(M1.dot(W) + W.T.dot(M1)).tocsc()
    print('A done')
    
    del W
    
    A_ILU = spa.linalg.spilu(A , drop_tol=drop_tol_A , fill_factor=fill_fac_A)
    print('A_ILU done')
    
    A_PRE = spa.linalg.LinearOperator(A.shape, lambda x : A_ILU.solve(x)) 
    # ==========================================================================

    

    # ================== matrices and preconditioner for step 2 ================
    S2      = (A + dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tocsc()
    print('S2 done')
    
    STEP2_1 = (A - dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tocsc()
    print('STEP2_1 done')
    
    STEP2_2 = dt*TAU.T.dot(CURL.T.dot(M2)).tocsc()
    print('STEP2_2 done')

    # incomplete LU decomposition for preconditioning
    S2_ILU  = spa.linalg.spilu(S2, drop_tol=drop_tol_S2, fill_factor=fill_fac_S2)
    print('S2_ILU done')
    
    S2_PRE  = spa.linalg.LinearOperator(S2.shape, lambda x : S2_ILU.solve(x))
    # ===========================================================================

    
    
    # ================== matrices and preconditioner for step 6 =================
    if add_pressure == True:
    
        L       = GRAD.T.dot(M1).dot(S) + (gamma - 1)*K.T.dot(GRAD.T).dot(M1)
        print('L done')

        del S, K

        # incomplete LU decompositions of M0
        M0_ILU  = spa.linalg.spilu(M0, drop_tol=drop_tol_M0, fill_factor=fill_fac_M0)
        print('M0_ILU done')

        # linear operators
        S6_LHS  = spa.linalg.LinearOperator(A.shape, lambda x : A.dot(x) + dt**2/4*M1.dot(GRAD.dot(M0_ILU.solve(L.dot(x)))))
        S6_RHS  = spa.linalg.LinearOperator(A.shape, lambda x : A.dot(x) - dt**2/4*M1.dot(GRAD.dot(M0_ILU.solve(L.dot(x)))))

        S6_P    = spa.linalg.LinearOperator((M1.shape[0], GRAD.shape[1]), lambda x : M1.dot(GRAD.dot(x)))
        S6_B    = spa.linalg.LinearOperator((M1.shape[0], P.shape[1])   , lambda x : M1.dot(P.dot(x)))
    # ==========================================================================
    
    print('assembly of constant matrices done!')
    

if mpi_rank == 0:
    timea = time.time()
    A_PRE(np.random.rand(A.shape[0]))
    timeb = time.time()
    print('preconditioner of A : ', timeb - timea)
    
    timea = time.time()
    np.split(spa.linalg.cg(S2, STEP2_1.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) + STEP2_2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol2, M=S2_PRE)[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
    timeb = time.time()
    print('linear system of step 2 : ', timeb - timea)

    
if mpi_rank == 0 and add_pressure == True:
    timea = time.time()
    S6_LHS(np.random.rand(A.shape[0]))
    timeb = time.time()
    print('left hand side of step 6 : ', timeb - timea)

    timea = time.time()
    np.split(spa.linalg.cgs(S6_LHS, S6_RHS(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) - dt*S6_P(pr.flatten()) + dt*S6_B(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol6, M=A_PRE)[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
    timeb = time.time()
    print('linear system of step 6 : ', timeb - timea)


    
if control == True and mpi_rank == 0:
    timea = time.time()
    mat   = cont.mass_V1_nh_eq(b1, b2, b3)
    timeb = time.time()
    print('control step 1 : ', timeb - timea)

    timea = time.time()
    vec   = cont.inner_prod_V1_jh_eq(b1, b2, b3)
    timeb = time.time()
    print('control step 3 : ', timeb - timea)

    

# ======================== create particles ======================================
if   loading == 'pseudo-random':
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
particles_loc[3] = sp.erfinv(2*particles_loc[3] - 1)*vth + v0x
particles_loc[4] = sp.erfinv(2*particles_loc[4] - 1)*vth + v0y
particles_loc[5] = sp.erfinv(2*particles_loc[5] - 1)*vth + v0z


# compute initial weights
if mapping == 0:
    pic_sample.compute_weights_ini(particles_loc, Np_loc, w0_loc, s0_loc, kind_map, params_map)
elif mapping == 1:
    pic_sample.compute_weights_ini(particles_loc, Np_loc, w0_loc, s0_loc, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)

if control == True:
    if mapping == 0:
        pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, kind_map, params_map)
    elif mapping == 1:
        pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
else:
    particles_loc[6] = w0_loc

print(mpi_rank, 'particle initialization done!')
# ======================================================================================



# ================ compute initial energies and distribution function ==================
# broadcast FEM coeffiecients from zeroth rank
mpi_comm.Bcast(u1, root=0)
mpi_comm.Bcast(u2, root=0)
mpi_comm.Bcast(u3, root=0)

mpi_comm.Bcast(b1, root=0)
mpi_comm.Bcast(b2, root=0)
mpi_comm.Bcast(b3, root=0)


# initial energies
if mpi_rank == 0:
    energies['U'][0] = 1/2*np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())).dot(A.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))))
    energies['B'][0] = 1/2*np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())).dot(M2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))))
    energies['p'][0] = 1/(gamma - 1)*pr.flatten().dot(norm_0form)

energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

if mapping == 0:
    energies['f'] += (control - 1)*eq_PIC.eh_eq(kind_map, params_map)
    
# initial distribution function
fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)

print('initial diagnostics done')
# ======================================================================================



"""
# ======= test pusher step 3 =========
mpi_comm.Barrier()
timea = time.time()

if mapping == 0:
    pic_pusher.pusher_step3(particles_loc, dt, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, u1, u2, u3, kind_map, params_map)
elif mapping == 1:
    pic_pusher.pusher_step3(particles_loc, dt, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, u1, u2, u3, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz)
    
timeb = time.time()
print(timeb - timea)
# ====================================


# ======= test pusher step 4 =========
mpi_comm.Barrier()
timea = time.time()

if mapping == 0:
    pic_pusher.pusher_step4(particles_loc, dt, Np_loc, kind_map, params_map)
elif mapping == 1:
    pic_pusher.pusher_step4(particles_loc, dt, Np_loc, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz)

timeb = time.time()
print(timeb - timea)
# ====================================


print(particles_loc[0].max())
print(particles_loc[1].max())
print(particles_loc[2].max())
sys.exit()

# ======= test pusher step 5 =========
mpi_comm.Barrier()
timea = time.time()

if mapping == 0:
    pic_pusher.pusher_step5(particles_loc, dt, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, kind_map, params_map)
elif mapping == 1:
    pic_pusher.pusher_step5(particles_loc, dt, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz)

timeb = time.time()
print(timeb - timea)
# ====================================


# ==== test accumulator step 1 =======
mpi_comm.Barrier()
timea = time.time()

if mapping == 0:
    pic_accumu_ker.kernel_step1(particles_loc, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, kind_map, params_map, mat12_loc, mat13_loc, mat23_loc)
elif mapping == 1:
    pic_accumu_ker.kernel_step1(particles_loc, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz, mat12_loc, mat13_loc, mat23_loc)
    
timeb = time.time()
print(timeb - timea)
# ====================================


# ==== test accumulator step 3 =======
mpi_comm.Barrier()
timea = time.time()

if mapping == 0:
    pic_accumu_ker.kernel_step3(particles_loc, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, kind_map, params_map, mat11_loc, mat12_loc, mat13_loc, mat22_loc, mat23_loc, mat33_loc, vec1_loc, vec2_loc, vec3_loc)
elif mapping == 1:
    pic_accumu_ker.kernel_step3(particles_loc, T[0], T[1], T[2], p, Nel, Nbase_0form, Nbase_3form, Np_loc, b1, b2, b3, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz, mat11_loc, mat12_loc, mat13_loc, mat22_loc, mat23_loc, mat33_loc, vec1_loc, vec2_loc, vec3_loc)
    
timeb = time.time()
print(timeb - timea)
# ====================================
    
sys.exit()
"""



# ==================== time integrator ==========================================
times_elapsed = {'total' : 0., 'accumulation_step1' : 0., 'accumulation_step3' : 0., 'pusher_step3' : 0., 'pusher_step4' : 0., 'pusher_step5' : 0., 'control_step1' : 0., 'control_step3' : 0., 'control_weights' : 0., 'update_step1u' : 0., 'update_step2u' : 0., 'update_step2b' : 0., 'update_step3u' : 0.,'update_step6' : 0.}

    
def update():
    
    global u1, u2, u3
    global u1_old, u2_old, u3_old
    global b1, b2, b3
    global pr, rh
    global particles_loc
    
    time_tota = time.time()
    
    # ====================================================================================
    #                           step 1 (1: update u)
    # ====================================================================================
    if add_PIC == True:
        
        # ------- charge accumulation -----------
        timea = time.time()
        
        if mapping == 0:
            pic_accumu_ker.kernel_step1(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, kind_map, params_map, mat12_loc, mat13_loc, mat23_loc)
        elif mapping == 1:
            pic_accumu_ker.kernel_step1(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz, mat12_loc, mat13_loc, mat23_loc)
        
        
        mpi_comm.Reduce(mat12_loc, mat12, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat13_loc, mat13, op=MPI.SUM, root=0)
        mpi_comm.Reduce(mat23_loc, mat23, op=MPI.SUM, root=0)
        
        timeb = time.time()
        times_elapsed['accumulation_step1'] = timeb - timea
        # ----------------------------------------
        
        
        # ----- set up and solve linear system ---
        if mpi_rank == 0:
            
            # build global sparse matrix
            mat = -acc.to_sparse_step1(mat12, mat13, mat23)/Np

            # delta-f correction
            if control == True:
                timea = time.time()
                mat  -= cont.mass_V1_nh_eq(b1, b2, b3)
                timeb = time.time()
                times_elapsed['control_step1'] = timeb - timea
        
            # solve linear system with conjugate gradient squared method with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess 
            timea = time.time()
            temp1, temp2, temp3 = np.split(spa.linalg.cgs(A - dt*mat/2, (A + dt*mat/2).dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol1, M=A_PRE)[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])         
            timeb = time.time()
            times_elapsed['update_step1u'] = timeb - timea

            u1[:, :, :] = temp1.reshape(Nbase_1form[0])
            u2[:, :, :] = temp2.reshape(Nbase_1form[1])
            u3[:, :, :] = temp3.reshape(Nbase_1form[2])
        # -----------------------------------------
            
    # ====================================================================================
    #                           step 1 (1: update u)
    # ====================================================================================
    
    
    
    
    # ====================================================================================
    #                       step 2 (1 : update u, 2 : update b) 
    # ====================================================================================
    if mpi_rank == 0:
        
        # save coefficients from previous time step
        u1_old[:, :, :] = u1[:, :, :]
        u2_old[:, :, :] = u2[:, :, :]
        u3_old[:, :, :] = u3[:, :, :]
        
        # solve linear system with conjugate gradient method (S2 is a symmetric positive definite matrix) with an incomplete LU decomposition as preconditioner and values from last time step as initial guess
        timea = time.time()
        temp1, temp2, temp3 = np.split(spa.linalg.cg(S2, STEP2_1.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) + STEP2_2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol2, M=S2_PRE)[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
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
    
    # broadcast new magnetic FEM coefficients
    mpi_comm.Bcast(b1, root=0)
    mpi_comm.Bcast(b2, root=0)
    mpi_comm.Bcast(b3, root=0)
    # ====================================================================================
    #                       step 2 (1 : update u, 2 : update b) 
    # ====================================================================================
    
    
    
    # ====================================================================================
    #             step 3 (1 : update u,  2 : update particles velocities V)
    # ====================================================================================
    if add_PIC == True:
        
        # ------------ current accumulation ----------------
        timea = time.time()
        
        if mapping == 0:
            pic_accumu_ker.kernel_step3(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, kind_map, params_map, mat11_loc, mat12_loc, mat13_loc, mat22_loc, mat23_loc, mat33_loc, vec1_loc, vec2_loc, vec3_loc)
        elif mapping == 1:
            pic_accumu_ker.kernel_step3(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz, mat11_loc, mat12_loc, mat13_loc, mat22_loc, mat23_loc, mat33_loc, vec1_loc, vec2_loc, vec3_loc)
        
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
        # ---------------------------------------------------
        
        
        # ---------- set up and solve linear system ---------
        if mpi_rank == 0:
            
            # save coefficients from previous time step
            u1_old[:, :, :] = u1[:, :, :]
            u2_old[:, :, :] = u2[:, :, :]
            u3_old[:, :, :] = u3[:, :, :]
            
            # build global sparse matrix and vector
            mat = acc.to_sparse_step3(mat11, mat12, mat13, mat22, mat23, mat33)/Np
            vec = np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))/Np
            
        
            # delta-f correction
            if control == True:
                timea  = time.time()
                vec   += cont.inner_prod_V1_jh_eq(b1, b2, b3)      
                timeb  = time.time()
                times_elapsed['control_step3'] = timeb - timea

            
            # solve linear system with conjugate gradient method (A + dt**2*mat/4 is a symmetric positive definite matrix) with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess
            timea = time.time()
            temp1, temp2, temp3 = np.split(spa.linalg.cg(A + dt**2*mat/4, (A - dt**2*mat/4).dot(np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten()))) + dt*vec, x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol3, M=A_PRE)[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
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
        # ---------------------------------------------------
        
        
        # ---------------- update velocities ----------------
        timea = time.time()
        
        if mapping == 0:
            pic_pusher.pusher_step3(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, (u1 + u1_old)/2, (u2 + u2_old)/2, (u3 + u3_old)/2, kind_map, params_map)
        elif mapping == 1:
            pic_pusher.pusher_step3(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, (u1 + u1_old)/2, (u2 + u2_old)/2, (u3 + u3_old)/2, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz)
        
        timeb = time.time()
        times_elapsed['pusher_step3'] = timeb - timea
        # ---------------------------------------------------
        
    # ====================================================================================
    #             step 3 (1 : update u,  2 : update particles velocities V)
    # ====================================================================================
    
    
    
    # ====================================================================================
    #             step 4 (1 : update particles positions ETA)
    # ====================================================================================
    if add_PIC == True:
        timea = time.time()
        
        if mapping == 0:
            pic_pusher.pusher_step4(particles_loc, dt, Np_loc, kind_map, params_map)
        elif mapping == 1:
            pic_pusher.pusher_step4(particles_loc, dt, Np_loc, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz)
        
        timeb = time.time()
        times_elapsed['pusher_step4'] = timeb - timea
    # ====================================================================================
    #             step 4 (1 : update particles positions ETA)
    # ====================================================================================

    
    
    # ====================================================================================
    #       step 5 (1 : update particle veclocities V, 2 : update particle weights W)
    # ====================================================================================
    if add_PIC == True:
        # push particles
        timea = time.time()
        
        if mapping == 0:
            pic_pusher.pusher_step5(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, kind_map, params_map)
        elif mapping == 1:
            pic_pusher.pusher_step5(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz)
            
        timeb = time.time()
        times_elapsed['pusher_step5'] = timeb - timea

        # update particle weights in case of delta-f
        if control == True:
            timea = time.time()
            
            if mapping == 0:
                pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, kind_map, params_map)
            elif mapping == 1:
                pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, T_F[0], T_F[1], T_F[2], p_F, NbaseN_F, cx, cy, cz)
                 
            timeb = time.time()
            times_elapsed['control_weights'] = timeb - timea
    # ====================================================================================
    #       step 5 (1 : update particle veclocities V, 2 : update particle weights W)
    # ====================================================================================
    
    
    
    # ====================================================================================
    #       step 6 (1 : update rh, u and pr from non - Hamiltonian MHD terms)
    # ====================================================================================
    if add_pressure == True and mpi_rank == 0:

        timea = time.time()

        # solve linear system of u^(n+1) with conjugate gradient squared method with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess
        u1_new, u2_new, u3_new = np.split(spa.linalg.cgs(S6_LHS, S6_RHS(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) - dt*S6_P(pr.flatten()) + dt*S6_B(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol6, M=A_PRE)[0], [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
        
        # solve linear system of pr^(n+1) with ILU of M0
        pr[:, :, :] = M0_ILU.solve(M0.dot(pr.flatten()) + dt/2*L.dot(np.concatenate((u1_new, u2_new, u3_new)) + np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())))).reshape(Nbase_0form)
        
        # update density rh^(n+1)
        rh[:, :, :] = rh - dt/2*(DIV.dot(Q.dot(np.concatenate((u1_new, u2_new, u3_new)) + np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))))).reshape(Nbase_3form)

        # update velocity
        u1[:, :, :] = u1_new.reshape(Nbase_1form[0])
        u2[:, :, :] = u2_new.reshape(Nbase_1form[1])
        u3[:, :, :] = u3_new.reshape(Nbase_1form[2])

        timeb = time.time()
        times_elapsed['update_step6'] = timeb - timea
    # ====================================================================================
    #       step 6 (1 : update rh, u and pr from non - Hamiltonian MHD terms)
    # ====================================================================================
    
        
    # ====================================================================================
    #                                diagnostics
    # ====================================================================================
    # energies
    if mpi_rank == 0:
        energies['U'][0] = 1/2*np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())).dot(A.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))))
        energies['B'][0] = 1/2*np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())).dot(M2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))))
        energies['p'][0] = 1/(gamma - 1)*pr.flatten().dot(norm_0form)

    energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
    mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

    if mapping == 0:
        energies['f'] += (control - 1)*eq_PIC.eh_eq(kind_map, params_map)


    # distribution function
    fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
    mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)
    # ====================================================================================
    #                                diagnostics
    # ====================================================================================
    
    time_totb = time.time()
    times_elapsed['total'] = time_totb - time_tota
    
# ============================================================================


"""
mpi_comm.Barrier()
timea = time.time()
for i in range(10):
    if mpi_rank == 0:
        print(i, energies)
    update()
timeb = time.time()
if mpi_rank == 0:
    print(mpi_rank, (timeb - timea)/10)
sys.exit()
"""


# ========================== time integration ================================
if time_int == True:
    
    # a new simulation
    if restart == False:
    
        if mpi_rank == 0:
        
            # ============= create hdf5 file and datasets for simulation output ======================
            file = h5py.File('results_' + identifier + '.hdf5', 'a')

            # current time
            file.create_dataset('time', (1,),   maxshape=(None,),   dtype=float, chunks=True)
            
            # energies
            file.create_dataset('energies/bulk_kinetic',  (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/magnetic',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/bulk_internal', (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/energetic',     (1,), maxshape=(None,), dtype=float, chunks=True)

            # elapsed times of different parts of the code
            file.create_dataset('times_elapsed/total',              (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/accumulation_step1', (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/accumulation_step3', (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/pusher_step3',       (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/pusher_step4',       (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/pusher_step5',       (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/control_step1',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/control_step3',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/control_weights',    (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step1u',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step2u',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step2b',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step3u',      (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('times_elapsed/update_step6',       (1,), maxshape=(None,), dtype=float, chunks=True)

            # FEM coefficients
            file.create_dataset('pressure',                   (1, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]),    maxshape=(None, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]),    dtype=float, chunks=True)
            file.create_dataset('velocity_field/1_component', (1, Nbase_1form[0][0], Nbase_1form[0][1], Nbase_1form[0][2]), maxshape=(None, Nbase_1form[0][0], Nbase_1form[0][1], Nbase_1form[0][2]), dtype=float, chunks=True)
            file.create_dataset('velocity_field/2_component', (1, Nbase_1form[1][0], Nbase_1form[1][1], Nbase_1form[1][2]), maxshape=(None, Nbase_1form[1][0], Nbase_1form[1][1], Nbase_1form[1][2]), dtype=float, chunks=True)
            file.create_dataset('velocity_field/3_component', (1, Nbase_1form[2][0], Nbase_1form[2][1], Nbase_1form[2][2]), maxshape=(None, Nbase_1form[2][0], Nbase_1form[2][1], Nbase_1form[2][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/1_component', (1, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), maxshape=(None, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/2_component', (1, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), maxshape=(None, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/3_component', (1, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), maxshape=(None, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), dtype=float, chunks=True)
            file.create_dataset('density',                    (1, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    dtype=float, chunks=True)

            # particles
            file.create_dataset('particles', (1, 7, Np), maxshape=(None, 7, Np), dtype=float, chunks=True)
            
            # other diagnostics
            file.create_dataset('bulk_mass', (1,), maxshape=(None,), dtype=float, chunks=True)

            file.create_dataset('magnetic_field/divergence',  (1, Nbase_3form[0], Nbase_3form[1], Nbase_3form[2]), maxshape=(None, Nbase_3form[0], Nbase_3form[1], Nbase_3form[2]), dtype=float, chunks=True)

            file.create_dataset('distribution_function/eta1_vx', (1, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), maxshape=(None, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float, chunks=True)

            
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

            file.create_dataset('restart/particles', (1, 6, Np), maxshape=(None, 6, Np), dtype=float, chunks=True)

            file.create_dataset('restart/control_w0', (Np,), dtype=float)
            file.create_dataset('restart/control_s0', (Np,), dtype=float)


            # ==================== save initial data =======================
            file['time'][0]                       = 0.

            file['energies/bulk_kinetic'][0]      = energies['U'][0]
            file['energies/magnetic'][0]          = energies['B'][0]
            file['energies/bulk_internal'][0]     = energies['p'][0]
            file['energies/energetic'][0]         = energies['f'][0]

            file['pressure'][0]                   = pr
            file['velocity_field/1_component'][0] = u1
            file['velocity_field/2_component'][0] = u2
            file['velocity_field/3_component'][0] = u3
            file['magnetic_field/1_component'][0] = b1
            file['magnetic_field/2_component'][0] = b2
            file['magnetic_field/3_component'][0] = b3
            file['density'][0]                    = rh

            file['magnetic_field/divergence'][0]  = DIV.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))).reshape(Nbase_3form[0], Nbase_3form[1], Nbase_3form[2])

            file['bulk_mass'][0]                  = sum(rh.flatten())
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
            time_steps_done   = file['restart/time_steps_done'][num_restart]

            pr[:, :, :]       = file['restart/pressure'][num_restart]
            u1[:, :, :]       = file['restart/velocity_field/1_component'][num_restart]
            u2[:, :, :]       = file['restart/velocity_field/2_component'][num_restart]
            u3[:, :, :]       = file['restart/velocity_field/3_component'][num_restart]
            b1[:, :, :]       = file['restart/magnetic_field/1_component'][num_restart]
            b2[:, :, :]       = file['restart/magnetic_field/2_component'][num_restart]
            b3[:, :, :]       = file['restart/magnetic_field/3_component'][num_restart]
            rh[:, :, :]       = file['restart/density'][num_restart]
            
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
        mpi_comm.Bcast(u1, root=0)
        mpi_comm.Bcast(u2, root=0)
        mpi_comm.Bcast(u3, root=0)
        
        mpi_comm.Bcast(b1, root=0)
        mpi_comm.Bcast(b2, root=0)
        mpi_comm.Bcast(b3, root=0)
       

        # perform initialization for next time step: compute particle weights
        if control == True:
            pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, kind_map, params_map)
        else:
            particles_loc[6] = w0_loc

    
    
    # ========================================================================================
    #              time loop 
    # ========================================================================================
    mpi_comm.Barrier()
    if mpi_rank == 0:
        print('start time integration! (total number of time steps : ' + str(int(Tend/dt)) + ')')
    # ========================================================================================
    while True:

        # synchronize MPI processes and check if simulation end is reached
        mpi_comm.Barrier()
        if (time_steps_done*dt >= Tend) or ((time.time() - start_simulation)/60 > max_time):
            
            # save data needed for restart
            if create_restart:
                if mpi_rank == 0:
                    file['restart/time_steps_done'][-1]            = time_steps_done
                    file['restart/pressure'][-1]                   = pr
                    file['restart/velocity_field/1_component'][-1] = u1
                    file['restart/velocity_field/2_component'][-1] = u2
                    file['restart/velocity_field/3_component'][-1] = u3
                    file['restart/magnetic_field/1_component'][-1] = b1
                    file['restart/magnetic_field/2_component'][-1] = b2
                    file['restart/magnetic_field/3_component'][-1] = b3
                    file['restart/density'][-1]                    = rh

                    file['restart/particles'][-1, :, :Np_loc]      = particles_loc[:6]

                    for i in range(1, mpi_size):
                        mpi_comm.Recv(particles_recv, source=i, tag=20)
                        file['restart/particles'][-1, :, i*Np_loc:(i + 1)*Np_loc] = particles_recv[:6]

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

                else:
                    mpi_comm.Send(particles_loc, dest=0, tag=20)
            
            # close output file and time loop
            if mpi_rank == 0:
                file.close()

            break

        # print number of finished time steps and current energies
        if mpi_rank == 0 and time_steps_done%1 == 0:
            print('time steps finished : ' + str(time_steps_done))
            print('energies : ', energies)
            
        

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

            
            # other diagnostics
            #file['magnetic_field/divergence'].resize(file['magnetic_field/divergence'].shape[0] + 1, axis = 0)
            #file['magnetic_field/divergence'][-1] = DIV.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))).reshape(Nbase_3form[0], Nbase_3form[1], Nbase_3form[2])

            file['bulk_mass'].resize(file['bulk_mass'].shape[0] + 1, axis = 0)
            file['bulk_mass'][-1] = sum(rh.flatten())
            
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
        # ==============================================================================================
        
# ============================================================================
