# load and initialize PETSc
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

# PETSc communicator
pet_comm = PETSc.COMM_WORLD

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
kind_map       = params['kind_map']
params_map     = params['params_map']

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
tol2           = params['tol2']
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


# ==== functions for creating parallel Petsc matrices and vectors ========
def CSRtoPetsc(A):
    
    Ap = PETSc.Mat().createAIJ(size=A.shape, comm=pet_comm)
    Ap.setFromOptions()
    
    # maximum number of nonzero entries per row
    nnz_max = (A.indptr[1:] - A.indptr[:-1]).max()
    Ap.setPreallocationNNZ((nnz_max, nnz_max))
    
    # fill matrix entries owned by process
    row_start, row_end = Ap.getOwnershipRange()

    for i in range(row_start, row_end):
        Ap.setValues(i, list(A.indices[A.indptr[i]:A.indptr[i + 1]]), A.data[A.indptr[i]:A.indptr[i + 1]])

    Ap.assemblyBegin(Ap.AssemblyType.FINAL)
    Ap.assemblyEnd(Ap.AssemblyType.FINAL)
    
    return Ap

def VecToPetsc(a):
    
    ap = PETSc.Vec().create(comm=pet_comm)
    ap.setSizes(a.size)
    ap.setFromOptions()

    # fill vector entries owned by process
    row_start, row_end = ap.getOwnershipRange()

    for i in range(row_start, row_end):
        ap.setValues(i, a[i])

    ap.assemblyBegin()
    ap.assemblyEnd()
    
    return ap
# ========================================================================    



# ================= MPI initialization for particles =====================
Np_loc        = int(Np/mpi_size)                      # number of particles for each process

particles_loc = np.empty((7, Np_loc), dtype=float)    # particles of each process
w0_loc        = np.empty(    Np_loc , dtype=float)    # weights for each process: hat_f_ini(eta_0, v_0)/hat_s_ini(eta_0, v_0)
s0_loc        = np.empty(    Np_loc , dtype=float)    # initial sampling density: hat_s_ini(eta_0, v_0) for each process

if mpi_rank == 0:
    particles_recv = np.empty((7, Np_loc), dtype=float)
    w0_recv        = np.empty(    Np_loc , dtype=float)    
    s0_recv        = np.empty(    Np_loc , dtype=float)
# ========================================================================



# =================== some diagnostics ===================================
# energies (bulk kinetic energy, magnetic energy, bulk internal energy, hot ion kinetic + internal energy (delta f))
energies     = {'U' : np.empty(1, dtype=float), 'B' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float), 'df' : np.empty(1, dtype=float)}

energies_loc = {'U' : np.empty(1, dtype=float), 'B' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float), 'df' : np.empty(1, dtype=float)}

# snapshots of distribution function via particle binning
n_bins       = {'eta1_vx' : [32, 64]}
bin_edges    = {'eta1_vx' : [np.linspace(0., 1., n_bins['eta1_vx'][0] + 1), np.linspace(0., 5., n_bins['eta1_vx'][1] + 1)]}
dbin         = {'eta1_vx' : [bin_edges['eta1_vx'][0][1] - bin_edges['eta1_vx'][0][0], bin_edges['eta1_vx'][1][1] - bin_edges['eta1_vx'][1][0]]}
                
fh           = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
                
fh_loc       = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
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

Ncum_1form     = [0, Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1], Ntot_1form[0] + Ntot_1form[1] + Ntot_1form[2]]
Ncum_2form     = [0, Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1], Ntot_2form[0] + Ntot_2form[1] + Ntot_2form[2]]

Ntottot_1form  = Ncum_1form[3]
Ntottot_2form  = Ncum_2form[3]



if add_PIC == True:

    # delta-f corrections (only MHD process)
    if control == True and mpi_rank == 0:
        cont = cv.terms_control_variate(tensor_space, kind_map, params_map)

    # particle accumulator (all processes)
    acc = pic_accumu.accumulation(tensor_space)
# =======================================================================



# ======= reserve memory for FEM cofficients (all MPI processes) ========
pr     = np.empty(Ntot_0form   , dtype=float)         # bulk pressure FEM coefficients
ut     = np.empty(Ntottot_1form, dtype=float)         # bulk velocity FEM coefficients
utnew  = np.empty(Ntottot_1form, dtype=float)         # bulk velocity FEM coefficients
bt     = np.empty(Ntottot_2form, dtype=float)         # magnetic field FEM coefficients
rh     = np.empty(Ntot_3form   , dtype=float)         # bulk mass density FEM coefficients
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



# ============= projection of initial conditions ==========================
if mpi_rank == 0:
    
    """
    # create object for projecting initial conditions
    pro   = proj.projectors_local_3d(tensor_space, nq_pr)

    pr[:] = pro.pi_0( None,               1,        kind_map, params_map).flatten()
    
    temp  = pro.pi_1([None, None, None], [2, 3, 4], kind_map, params_map)
    ut[:] = np.concatenate((temp[0].flatten(), temp[1].flatten(), temp[2].flatten()))
    
    temp  = pro.pi_2([None, None, None], [5, 6, 7], kind_map, params_map)
    bt[:] = np.concatenate((temp[0].flatten(), temp[1].flatten(), temp[2].flatten()))
    
    rh[:] = pro.pi_3( None,               8,        kind_map, params_map).flatten()
    
    del pro
    """
    
    
    amps = np.random.rand(8, NbaseN[0], NbaseN[1])
    temp = np.empty(Nbase_0form, dtype=float)

    for k in range(NbaseN[2]):
        temp[:, :, k] = amps[0]
    pr[:] = temp.flatten()
    
    for k in range(NbaseN[2]):
        temp[:, :, k] = amps[1]
    ut[Ncum_1form[0]:Ncum_1form[1]] = temp.flatten()
    
    for k in range(NbaseN[2]):
        temp[:, :, k] = amps[2]
    ut[Ncum_1form[1]:Ncum_1form[2]] = temp.flatten()
    
    for k in range(NbaseN[2]):
        temp[:, :, k] = amps[3]
    ut[Ncum_1form[2]:Ncum_1form[3]] = temp.flatten()

    bt[Ncum_2form[0]:Ncum_2form[1]] = 0.
    bt[Ncum_2form[1]:Ncum_2form[2]] = 0.
    
    for k in range(NbaseN[2]):
        temp[:, :, k] = amps[6]
    bt[Ncum_2form[2]:Ncum_2form[3]] = temp.flatten()
        
    for k in range(NbaseN[2]):
        temp[:, :, k] = amps[3]
    rh[:] = temp.flatten()
    
    
    print('projection of initial conditions done!')
# ==========================================================================
    

    
# ==== broadcast initial coefficients and create petsc parallel vectors ====  
mpi_comm.Bcast(pr, root=0)
mpi_comm.Bcast(ut, root=0)
mpi_comm.Bcast(bt, root=0)
mpi_comm.Bcast(rh, root=0)

pr_pet    = VecToPetsc(pr.flatten())
ut_pet    = VecToPetsc(ut.flatten())
bt_pet    = VecToPetsc(bt.flatten())
rh_pet    = VecToPetsc(rh.flatten())

utnew_pet = ut_pet.copy()

# get row ranges of calling process
p_start, p_end = pr_pet.getOwnershipRange()
u_start, u_end = ut_pet.getOwnershipRange()
b_start, b_end = bt_pet.getOwnershipRange()
r_start, r_end = rh_pet.getOwnershipRange()
# ==========================================================================
 
    
    
# ================= mass matrices in V0, V1 and V2 =========================
if mpi_rank == 0:
    M0  = mass.mass_V0(tensor_space, 0, kind_map, params_map).tocsr()
else:
    M0  = None

M0 = mpi_comm.bcast(M0, root=0)    
M0 = CSRtoPetsc(M0)

if mpi_rank == 0:
    M1s = mass.mass_V1(tensor_space, 0, kind_map, params_map).tocsr()
else:
    M1s = None

M1 = mpi_comm.bcast(M1s, root=0)    
M1 = CSRtoPetsc(M1)

if mpi_rank == 0:
    M2 = mass.mass_V2(tensor_space, 0, kind_map, params_map).tocsr()
else:
    M2 = None

M2 = mpi_comm.bcast(M2, root=0)    
M2 = CSRtoPetsc(M2)

print('mass matrices done!')
# ==========================================================================



# ====== normalization vector in V0 (for bulk thermal energy)===============
if mpi_rank == 0:
    norm_0form = inner.inner_prod_V0(tensor_space, lambda eta1, eta2, eta3 : np.ones(eta1.shape), 0, kind_map, params_map).flatten()
else:
    norm_0form = None
    
norm_0form = mpi_comm.bcast(norm_0form, root=0)    
norm_0form = VecToPetsc(norm_0form)
# ==========================================================================


# ============= discrete grad, curl and div matrices =======================
if mpi_rank == 0:
    derivatives = der.discrete_derivatives(tensor_space)

    GRAD = derivatives.grad_3d().tocsr()
    CURL = derivatives.curl_3d().tocsr()
    DIV  = derivatives.div_3d().tocsr()
    
    del derivatives
else:
    GRAD = None
    CURL = None
    DIV  = None
    
GRAD = mpi_comm.bcast(GRAD, root=0)
CURL = mpi_comm.bcast(CURL, root=0) 
DIV  = mpi_comm.bcast(DIV,  root=0) 

GRAD = CSRtoPetsc(GRAD)
CURL = CSRtoPetsc(CURL)
DIV  = CSRtoPetsc(DIV)

print('discrete derivatives done!')
# ==========================================================================




# =========================== projection matrices ==========================
if mpi_rank == 0:
    # create object for projecting MHD matrices
    MHD = mhd.projectors_local_mhd(tensor_space, nq_pr)
    
    # pi_2[rho_eq * g_inv * lambda^1]
    Q   = MHD.projection_Q(kind_map, params_map).tocsr()     
else:
    Q   = None
    
Q = mpi_comm.bcast(Q, root=0)    
Q = CSRtoPetsc(Q)

if mpi_rank == 0:
    # pi_1[rho_eq/g_sqrt * lambda^1]
    Ws = MHD.projection_W(kind_map, params_map).tocsr() 
else:
    Ws = None
    
W = mpi_comm.bcast(Ws, root=0)    
W = CSRtoPetsc(W)

if mpi_rank == 0:
    # pi_1[b_eq * g_inv * lambda^1]
    TAU = MHD.projection_T(kind_map, params_map).tocsr() 
else:
    TAU = None
    
TAU = mpi_comm.bcast(TAU, root=0)    
TAU = CSRtoPetsc(TAU)

if mpi_rank == 0:
    # pi_1[p_eq * lambda^1]
    S = MHD.projection_S(kind_map, params_map).tocsr()   
else:
    S = None
    
S = mpi_comm.bcast(S, root=0)    
S = CSRtoPetsc(S)

if mpi_rank == 0:
    # pi_0[p_eq * lambda^0]  
    K = MHD.projection_K(kind_map, params_map).tocsr() 
else:
    K = None
    
K = mpi_comm.bcast(K, root=0)    
K = CSRtoPetsc(K)

if mpi_rank == 0:
    # pi_1[curl(b_eq) * lambda^2]
    P = MHD.projection_P(kind_map, params_map).tocsr()       
    del MHD
else:
    P = None
    
P = mpi_comm.bcast(P, root=0)    
P = CSRtoPetsc(P)
    
    
# compute symmetric matrix A
if mpi_rank == 0:
    As = (1/2*(M1s.dot(Ws) + Ws.T.dot(M1s))).tocsc()
    
A = 1/2*(M1.matMult(W) + W.transposeMatMult(M1))
print('A done')

del W    
        
print('projection matrices done!')
# ==========================================================================




# ============= matrices and itertive solver for step 2 ====================
S2      = A + dt**2/4*TAU.transposeMatMult(CURL.transposeMatMult(M2.matMult(CURL.matMult(TAU))))
print('S2 done')

STEP2_1 = A - dt**2/4*TAU.transposeMatMult(CURL.transposeMatMult(M2.matMult(CURL.matMult(TAU))))
print('STEP2_1 done')

STEP2_2 = dt*TAU.transposeMatMult(CURL.transposeMatMult(M2))
print('STEP2_2 done')

# solver for step 2
ksp2 = PETSc.KSP().create(comm=pet_comm)
ksp2.setOperators(S2, S2)
ksp2.setFromOptions()
ksp2.setInitialGuessNonzero(True)
ksp2.setUp()
# ==========================================================================


# ============= matrices and itertive solver for step 6 ====================
if add_pressure == True:
    
    # approximate inverse of A by its reciprocal diagonal
    A_diag = A.getDiagonal()
    A_diag.reciprocal()

    # approximate Schur complement (overwrites temporarily M1)
    M1.diagonalScale(A_diag)
    Schur  = M0 + dt**2/4*(GRAD.transposeMatMult(M1.matMult(S)) + (gamma - 1)*K.transposeMatMult(GRAD.transposeMatMult(M1))).matMult(M1.matMult(GRAD))
    
    # rebuild M1!
    A_diag = A.getDiagonal()
    M1.diagonalScale(A_diag)

    # left- and right-hand side matrices
    S6_LHS = PETSc.Mat().createNest([[A,  dt/2*M1.matMult(GRAD)], [-dt/2*(GRAD.transposeMatMult(M1.matMult(S)) + (gamma - 1)*K.transposeMatMult(GRAD.transposeMatMult(M1))), M0]], comm=pet_comm)
    S6_RHS = PETSc.Mat().createNest([[A, -dt/2*M1.matMult(GRAD)], [ dt/2*(GRAD.transposeMatMult(M1.matMult(S)) + (gamma - 1)*K.transposeMatMult(GRAD.transposeMatMult(M1))), M0]], comm=pet_comm)

    del A_diag, M0

    # linear solver
    ksp6 = PETSc.KSP().create(comm=pet_comm)
    ksp6.setOperators(S6_LHS, S6_LHS)
    ksp6.setInitialGuessNonzero(True)
    ksp6.setFromOptions()
    ksp6.setType('cgs')
    
    
    # set up field split preconditioner
    pc6 = ksp6.getPC()
    pc6.setType('fieldsplit')         # use fieldsplit method for general block systems
    pc6.setFieldSplitType(4)          # use Schur complement method (specifically for 2x2 block systems)
    pc6.setFieldSplitSchurFactType(3) # set type of preconditioning Schur complement

    # set row indices to separate blocks
    pc6.setFieldSplitIS(['u', S6_LHS.getNestISs()[0][0]])
    pc6.setFieldSplitIS(['p', S6_LHS.getNestISs()[0][1]])

    # set preconditioner for Schur complement manually
    pc6.setFieldSplitSchurPreType(3, Schur)
# ==========================================================================    

    
# ======================== create particles ======================================
if   loading == 'pseudo-random':
    # pseudo-random numbers between (0, 1)
    np.random.seed(seed)
    
    for i in range(mpi_size):
        temp = np.random.rand(Np_loc, 6)
        
        if i == mpi_rank:
            particles_loc[:6] = temp.T
            break

elif loading == 'sobol_standard':
    # plain sobol numbers between (0, 1) (skip first 1000 numbers)
    particles_loc[:6] = sobol.i4_sobol_generate(6, Np_loc, 1000).T 

elif loading == 'sobol_antithetic':
    # symmetric sobol numbers between (0, 1) (skip first 1000 numbers) in all 6 dimensions
    pic_sample.set_particles_symmetric(sobol.i4_sobol_generate(6, int(Np_loc/64), 1000), particles_loc)  

elif loading == 'external':
    
    if mpi_rank == 0:
        file = h5py.File(params['dir_particles'], 'r')
        
        particles_loc[:, :] = file['particles'][0, :Np_loc].T
            
        for i in range(1, mpi_size):
            particles_recv[:, :] = file['particles'][0, i*Np_loc:(i + 1)*Np_loc].T
            mpi_comm.Send(particles_recv, dest=i, tag=11) 
            
        file.close()
    else:
        mpi_comm.Recv(particles_loc, source=0, tag=11)

else:
    print('particle loading not specified')


# inversion of cumulative distribution function
particles_loc[3]  = sp.erfinv(2*particles_loc[3] - 1)*vth + v0x
particles_loc[4]  = sp.erfinv(2*particles_loc[4] - 1)*vth + v0y
particles_loc[5]  = sp.erfinv(2*particles_loc[5] - 1)*vth + v0z


# compute initial weights
pic_sample.compute_weights_ini(particles_loc, w0_loc, s0_loc, kind_map, params_map)

if control == True:
    pic_sample.update_weights(particles_loc, w0_loc, s0_loc, kind_map, params_map)
else:
    particles_loc[6] = w0_loc
                    
#print(mpi_rank, 'particle initialization done!')
# ======================================================================================



# ========= compute initial fields at particle positions and initial energies ==========    
# initial energies
energies['U'][0] = 1/2*ut_pet.dot(A(ut_pet))
energies['B'][0] = 1/2*bt_pet.dot(M2(bt_pet))
energies['p'][0] = 1/(gamma - 1)*pr_pet.dot(norm_0form)

energies_loc['df'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
mpi_comm.Allreduce(energies_loc['df'], energies['df'], op=MPI.SUM)

energies['df'] += (control - 1)*eq_PIC.eh_eq(kind_map, params_map)


# initial distribution function
fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
mpi_comm.Allreduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM)

#print(mpi_rank, energies['B'][0])
# ======================================================================================




# ==================== time integrator =================================================
times_elapsed = {'total' : 0., 'accumulation_step1' : 0., 'accumulation_step3' : 0., 'pusher_step3' : 0., 'pusher_step4' : 0., 'pusher_step5' : 0., 'control_step1' : 0., 'control_step3' : 0., 'control_weights' : 0., 'update_step1u' : 0., 'update_step2u' : 0., 'update_step2b' : 0., 'update_step3u' : 0.,'update_step6' : 0.}

def update():
    
    global pr    , ut    , utnew    , bt    , rh
    global pr_pet, ut_pet, utnew_pet, bt_pet, rh_pet
    global particles_loc

    
    time_tota = time.time()
    
    # ====================================================================================
    #                           step 1 (1: update ut)
    # ====================================================================================
    if add_PIC == True:
        
        # charge accumulation
        timea = time.time()
        
        pic_accumu_ker.kernel_step1(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, bt[Ncum_2form[0]:Ncum_2form[1]].reshape(Nbase_2form[0]), bt[Ncum_2form[1]:Ncum_2form[2]].reshape(Nbase_2form[1]), bt[Ncum_2form[2]:Ncum_2form[3]].reshape(Nbase_2form[2]), kind_map, params_map, mat12_loc, mat13_loc, mat23_loc)

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
            ut[:] = spa.linalg.spsolve(As - dt*mat/2, (As + dt*mat/2).dot(ut))
            timeb = time.time()
            times_elapsed['update_step1u'] = timeb - timea
    
    # broadcast to all processes and update parallel vector
    mpi_comm.Bcast(ut, root=0)
    ut_pet[u_start:u_end] = ut[u_start:u_end]
    # ====================================================================================
    #                           step 1 (1: update u)
    # ====================================================================================
    
    
    
    
    # ====================================================================================
    #                       step 2 (1 : update u, 2 : update b) 
    # ====================================================================================
    # solve linear system
    utnew_pet[u_start:u_end] = ut_pet[u_start:u_end]
    timea = time.time()
    ksp2.solve(STEP2_1(ut_pet) + STEP2_2(bt_pet), utnew_pet)
    timeb = time.time()
    times_elapsed['update_step2u'] = timeb - timea

    # update bt_pet
    bt_pet.assemblyBegin()
    bt_pet.assemblyEnd()
    
    ut_pet.assemblyBegin()
    ut_pet.assemblyEnd()
    
    utnew_pet.assemblyBegin()
    utnew_pet.assemblyEnd()
    
    timea = time.time()
    bt_pet[b_start:b_end] = (bt_pet - dt/2*CURL(TAU(ut_pet)) - dt/2*CURL(TAU(utnew_pet)))[b_start:b_end]
    timeb = time.time()
    times_elapsed['update_step2b'] = timeb - timea

    # update ut_pet
    ut_pet[u_start:u_end] = utnew_pet[u_start:u_end]

    # distribute local updates to all processes
    mpi_comm.Allgather(bt_pet[b_start:b_end], bt)
    mpi_comm.Allgather(ut_pet[u_start:u_end], ut)
    # ====================================================================================
    #                       step 2 (1 : update u, 2 : update b) 
    # ====================================================================================
    
    
    # ====================================================================================
    #             step 3 (1 : update u,  2 : update particles velocities V)
    # ====================================================================================
    if add_PIC == True:
        
        # current accumulation
        timea = time.time()
        
        pic_accumu_ker.kernel_step3(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, bt[Ncum_2form[0]:Ncum_2form[1]].reshape(Nbase_2form[0]), bt[Ncum_2form[1]:Ncum_2form[2]].reshape(Nbase_2form[1]), bt[Ncum_2form[2]:Ncum_2form[3]].reshape(Nbase_2form[2]), kind_map, params_map, mat11_loc, mat12_loc, mat13_loc, mat22_loc, mat23_loc, mat33_loc, vec1_loc, vec2_loc, vec3_loc)

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
            
            # build global sparse matrix
            mat = acc.to_sparse_step3(mat11, mat12, mat13, mat22, mat23, mat33)/Np
        
            # delta-f update
            if control == True:
                timea  = time.time()
                vec_cv = cont.inner_prod_V1_jh_eq([b1, b2, b3])
                timeb  = time.time()
                times_elapsed['control_step3'] = timeb - timea

                timea = time.time()
                utnew[:] = spa.linalg.spsolve(As + dt**2*mat/4, (As - dt**2*mat/4).dot(ut) + dt*np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))/Np + dt*np.concatenate((vec_cv[0].flatten(), vec_cv[1].flatten(), vec_cv[2].flatten())))
                timeb = time.time()
                times_elapsed['update_step3u'] = timeb - timea
                

            # full-f update
            else: 
                timea = time.time()
                utnew[:] = spa.linalg.spsolve(As + dt**2*mat/4, (As - dt**2*mat/4).dot(ut) + dt*np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))/Np)
                timeb = time.time()
                times_elapsed['update_step3u'] = timeb - timea

        
        # broadcast to all processes and update parallel vector
        mpi_comm.Bcast(utnew, root=0)
        
        
        # update particle velocities
        timea = time.time()
        pic_pusher.pusher_step3(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, bt[Ncum_2form[0]:Ncum_2form[1]].reshape(Nbase_2form[0]), bt[Ncum_2form[1]:Ncum_2form[2]].reshape(Nbase_2form[1]), bt[Ncum_2form[2]:Ncum_2form[3]].reshape(Nbase_2form[2]), 1/2*(ut[Ncum_1form[0]:Ncum_1form[1]] + utnew[Ncum_1form[0]:Ncum_1form[1]]).reshape(Nbase_1form[0]), 1/2*(ut[Ncum_1form[1]:Ncum_1form[2]] + utnew[Ncum_1form[1]:Ncum_1form[2]]).reshape(Nbase_1form[1]), 1/2*(ut[Ncum_1form[2]:Ncum_1form[3]] + utnew[Ncum_1form[2]:Ncum_1form[3]]).reshape(Nbase_1form[2]), kind_map, params_map)
        timeb = time.time()
        times_elapsed['pusher_step3'] = timeb - timea
        
        # update ut and ut_pet
        ut[:] = utnew
        ut_pet[u_start:u_end] = ut[u_start:u_end]
    # ====================================================================================
    #             step 3 (1 : update u,  2 : update particles velocities V)
    # ====================================================================================


    
    # ====================================================================================
    #             step 4 (1 : update particles positions ETA)
    # ====================================================================================
    if add_PIC == True:
        timea = time.time()
        pic_pusher.pusher_step4(particles_loc, dt, Np_loc, kind_map, params_map)
        timeb = time.time()
        times_elapsed['pusher_step4'] = timeb - timea
    # ====================================================================================
    #             step 4 (1 : update particles positions ETA)
    # ====================================================================================
    
    
    # ====================================================================================
    #       step 5 (1 : update particle veclocities V, 2 : update particle weights W)
    # ====================================================================================
    if add_PIC == True:
        timea = time.time()
        pic_pusher.pusher_step5(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, bt[Ncum_2form[0]:Ncum_2form[1]].reshape(Nbase_2form[0]), bt[Ncum_2form[1]:Ncum_2form[2]].reshape(Nbase_2form[1]), bt[Ncum_2form[2]:Ncum_2form[3]].reshape(Nbase_2form[2]), kind_map, params_map)
        timeb = time.time()
        times_elapsed['pusher_step5'] = timeb - timea

        if control == True:
            timea = time.time()
            pic_sample.update_weights(particles_loc, w0_loc, s0_loc, kind_map, params_map)
            timeb = time.time()
            times_elapsed['control_weights'] = timeb - timea
    # ====================================================================================
    #       step 5 (1 : update particle veclocities V, 2 : update particle weights W)
    # ====================================================================================

    
    # ====================================================================================
    #       step 6 (1 : update rh, u and pr from non - Hamiltonian MHD terms)
    # ====================================================================================
    if add_pressure == True:
        timea = time.time()
        
        # block vectors
        up_old = PETSc.Vec().createNest([ut_pet, pr_pet]                      , comm=pet_comm) # old velocity and pressure
        up_new = PETSc.Vec().createNest([ut_pet, pr_pet]                      , comm=pet_comm) # new velocity and pressure
        bp_old = PETSc.Vec().createNest([dt*M1(P(bt_pet)), pr_pet.duplicate()], comm=pet_comm) # magnetic field contribution
        
        # solve system
        ksp6.solve(S6_RHS(up_old) + bp_old, up_new)
        
        # reassemble block vectors (TODO: check why this is necessary)
        up_old.assemblyBegin()
        up_old.assemblyEnd()

        up_new.assemblyBegin()
        up_new.assemblyEnd()
        
        bp_old.assemblyBegin()
        bp_old.assemblyEnd()

        # update density
        rh_pet[r_start:r_end] = (rh_pet - dt/2*DIV(Q(up_old.getNestSubVecs()[0] + up_new.getNestSubVecs()[0])))[r_start:r_end]

        # update velocity and pressure
        ut_pet[u_start:u_end] = up_new.getNestSubVecs()[0][u_start:u_end]
        pr_pet[p_start:p_end] = up_new.getNestSubVecs()[1][p_start:p_end]
        
        rh_pet.assemblyBegin()
        rh_pet.assemblyEnd()

        ut_pet.assemblyBegin()
        ut_pet.assemblyEnd()
        
        pr_pet.assemblyBegin()
        pr_pet.assemblyEnd()

        # distribute local updates to all processes
        mpi_comm.Allgather(ut_pet[u_start:u_end], ut)
        mpi_comm.Allgather(pr_pet[p_start:p_end], pr)
        mpi_comm.Allgather(rh_pet[r_start:r_end], rh)
        
        timeb = time.time()
        times_elapsed['update_step6']
    # ====================================================================================
    #       step 6 (1 : update rh, u and pr from non - Hamiltonian MHD terms)
    # ====================================================================================
        
    time_totb = time.time()
    times_elapsed['total'] = time_totb - time_tota
    
    

    """
    # step 6 =============================================
    if add_pressure == True:
        up_old.getNestSubVecs()[0][u_start:u_end] = ut_pet[u_start:u_end]             # set old velocity
        up_old.getNestSubVecs()[1][p_start:p_end] = pr_pet[p_start:p_end]             # set old pressure

        up_new.getNestSubVecs()[0][u_start:u_end] = ut_pet[u_start:u_end]             # set initial guess for new velocity
        up_new.getNestSubVecs()[1][p_start:p_end] = pr_pet[p_start:p_end]             # set initial guess for new pressure

        bp_old.getNestSubVecs()[0][u_start:u_end] = dt*M1(P(bt_pet))[u_start:u_end]   # set old magnetic field

        # solve system
        ksp6.solve(S6_RHS(up_old) + bp_old, up_new)

        # update density
        rh_pet[r_start:r_end] = (rh_pet - dt/2*DIV(Q(up_old.getNestSubVecs()[0] + up_new.getNestSubVecs()[0])))[r_start:r_end]

        # update velocity and pressure
        ut_pet[u_start:u_end] = up_new.getNestSubVecs()[0][u_start:u_end]
        pr_pet[p_start:p_end] = up_new.getNestSubVecs()[1][p_start:p_end]

        # distribute local updates to all processes
        mpi_comm.Allgather(ut_pet[u_start:u_end], ut)
    """


    # ====================================================================================
    #                                diagnostics
    # ====================================================================================
    energies['U'][0] = 1/2*ut_pet.dot(A(ut_pet))
    energies['B'][0] = 1/2*bt_pet.dot(M2(bt_pet))
    energies['p'][0] = 1/(gamma - 1)*pr_pet.dot(norm_0form)

    energies_loc['df'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
    mpi_comm.Allreduce(energies_loc['df'], energies['df'], op=MPI.SUM)

    energies['df'] += (control - 1)*eq_PIC.eh_eq(kind_map, params_map)

    # distribution function
    fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
    mpi_comm.Allreduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM)  
    # ====================================================================================
    #                                diagnostics
    # ====================================================================================
    
    time_totb = time.time()
    times_elapsed['total'] = time_totb - time_tota
    
# ======================================================================================    
  
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
            file.create_dataset('energies/bulk_kinetic',     (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/magnetic',         (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/bulk_internal',    (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/energetic_deltaf', (1,), maxshape=(None,), dtype=float, chunks=True)

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

            #file.create_dataset('distribution_function/eta1_vx', (1, n_bins[0], n_bins[1]), maxshape=(None, n_bins[0], n_bins[1]), dtype=float, chunks=True)

            
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
            file['energies/energetic_deltaf'][0]  = energies['df'][0]

            file['pressure'][0]                   = pr.reshape(Nbase_0form)
            file['velocity_field/1_component'][0] = ut[Ncum_1form[0]:Ncum_1form[1]].reshape(Nbase_1form[0])
            file['velocity_field/2_component'][0] = ut[Ncum_1form[1]:Ncum_1form[2]].reshape(Nbase_1form[1])
            file['velocity_field/3_component'][0] = ut[Ncum_1form[2]:Ncum_1form[3]].reshape(Nbase_1form[2])
            file['magnetic_field/1_component'][0] = bt[Ncum_2form[0]:Ncum_2form[1]].reshape(Nbase_2form[0])
            file['magnetic_field/2_component'][0] = bt[Ncum_2form[1]:Ncum_2form[2]].reshape(Nbase_2form[1])
            file['magnetic_field/3_component'][0] = bt[Ncum_2form[2]:Ncum_2form[3]].reshape(Nbase_2form[2])
            file['density'][0]                    = rh.reshape(Nbase_3form)

            #file['magnetic_field/divergence'][0]  = DIV.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))).reshape(Nbase_3form[0], Nbase_3form[1], Nbase_3form[2])

            #file['bulk_mass'][0]                  = sum(rh.flatten())
            #file['distribution_function/eta1_vx'][0] = fh['eta1_vx']
            
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
            pic_sample.update_weights(particles_loc, w0_loc, s0_loc, kind_map, params_map)
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
            file['energies/energetic_deltaf'].resize(file['energies/energetic_deltaf'].shape[0] + 1, axis = 0)
            file['energies/bulk_kinetic'][-1]     = energies['U'][0]
            file['energies/magnetic'][-1]         = energies['B'][0]
            file['energies/bulk_internal'][-1]    = energies['p'][0]
            file['energies/energetic_deltaf'][-1] = energies['df'][0]

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
            file['pressure'][-1] = pr.reshape(Nbase_0form)

            #file['magnetic_field/1_component'].resize(file['magnetic_field/1_component'].shape[0] + 1, axis = 0)
            #file['magnetic_field/2_component'].resize(file['magnetic_field/2_component'].shape[0] + 1, axis = 0)
            #file['magnetic_field/3_component'].resize(file['magnetic_field/3_component'].shape[0] + 1, axis = 0)
            #file['magnetic_field/1_component'][-1] = b1
            #file['magnetic_field/2_component'][-1] = b2
            #file['magnetic_field/3_component'][-1] = b3

            #file['velocity_field/1_component'].resize(file['velocity_field/1_component'].shape[0] + 1, axis = 0)
            #file['velocity_field/2_component'].resize(file['velocity_field/2_component'].shape[0] + 1, axis = 0)
            file['velocity_field/3_component'].resize(file['velocity_field/3_component'].shape[0] + 1, axis = 0)
            #file['velocity_field/1_component'][-1] = u1
            #file['velocity_field/2_component'][-1] = u2
            file['velocity_field/3_component'][-1] = ut[Ncum_1form[2]:Ncum_1form[3]].reshape(Nbase_1form[2])

            #file['density'].resize(file['density'].shape[0] + 1, axis = 0)
            #file['density'][-1] = rh

            
            # other diagnostics
            #file['magnetic_field/divergence'].resize(file['magnetic_field/divergence'].shape[0] + 1, axis = 0)
            #file['magnetic_field/divergence'][-1] = DIV.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))).reshape(Nbase_3form[0], Nbase_3form[1], Nbase_3form[2])

            #file['bulk_mass'].resize(file['bulk_mass'].shape[0] + 1, axis = 0)
            #file['bulk_mass'][-1] = sum(rh.flatten())
            
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
