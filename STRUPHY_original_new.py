# load and initialize PETSc
#import sys
#import petsc4py
#petsc4py.init(sys.argv)
#from petsc4py import PETSc

# PETSc communicator
#pet_comm = PETSc.COMM_WORLD

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
import hylife.geometry.polar_splines as spl_pol
import hylife.geometry.domain        as dom

import hylife.utilitis_FEEC.spline_space as spl

import hylife.utilitis_FEEC.projectors.projectors_local      as proj_local
import hylife.utilitis_FEEC.projectors.projectors_global     as proj_global
import hylife.utilitis_FEEC.projectors.linear_operators_mhd  as mhd
import hylife.utilitis_FEEC.projectors.projectors_local_mhd  as mhd_local
import hylife.utilitis_FEEC.projectors.projectors_global_mhd as mhd_global

import hylife.utilitis_PIC.sobol_seq    as sobol
import hylife.utilitis_PIC.pusher       as pic_pusher
import hylife.utilitis_PIC.accumulation as pic_accumu
import hylife.utilitis_PIC.sampling     as pic_sample

import hylife.linear_algebra.kernels_tensor_product as ker_la

# load local input files
import input_run.equilibrium_PIC as eq_PIC
import input_run.equilibrium_MHD as eq_MHD


# ======================== load parameters =================================================================================
identifier = 'sed_replace_run_dir'   

with open('parameters_sed_replace_run_dir.yml') as file:
    params = yaml.load(file)


# mesh generation
Nel            = params['Nel']
bc             = params['bc']
p              = params['p']
nq_el          = params['nq_el']
nq_pr          = params['nq_pr']

# boundary conditions
bc_u1          = params['bc_u1']
bc_b1          = params['bc_b1']

# representation of MHD bulk velocity
basis_u        = params['basis_u']

# projectors
use_projector      = params['use_projector']
tol_approx_reduced = params['tol_approx_reduced']

# time integration
time_int       = params['time_int']
dt             = params['dt']
Tend           = params['Tend']
max_time       = params['max_time']

# polar splines in poloidal plane?
polar          = params['polar']

# geometry
kind_map       = params['kind_map']
params_map     = params['params_map']

Nel_MAP        = params['Nel_MAP']
bc_MAP         = params['bc_MAP']
p_MAP          = params['p_MAP']

# general
add_pressure   = params['add_pressure']
gamma          = params['gamma']

# ILU preconditioners for linear systems
drop_tol_S2    = params['drop_tol_S2']
fill_fac_S2    = params['fill_fac_S2']

drop_tol_A     = params['drop_tol_A']
fill_fac_A     = params['fill_fac_A']

drop_tol_S6    = params['drop_tol_S6']
fill_fac_S6    = params['fill_fac_S6']

# tolerances for iterative linear solvers
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
# ==========================================================================================================================



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
energies     = {'U' : np.empty(1, dtype=float), 'B' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float), 'f' : np.empty(1, dtype=float)}

energies_loc = {'U' : np.empty(1, dtype=float), 'B' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float), 'f' : np.empty(1, dtype=float)}

# snapshots of distribution function via particle binning
n_bins       = {'eta1_vx' : [32, 64]}
bin_edges    = {'eta1_vx' : [np.linspace(0., 1., n_bins['eta1_vx'][0] + 1), np.linspace(0., 5., n_bins['eta1_vx'][1] + 1)]}
dbin         = {'eta1_vx' : [bin_edges['eta1_vx'][0][1] - bin_edges['eta1_vx'][0][0], bin_edges['eta1_vx'][1][1] - bin_edges['eta1_vx'][1][0]]}
                
fh           = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
                
fh_loc       = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
# ========================================================================



# ================== FEM space ==========================================
# 1d B-spline spline spaces for finite elements
spaces_FEM = [spl.spline_space_1d(Nel, p, bc, nq_el) for Nel, p, bc, nq_el in zip(Nel, p, bc, nq_el)]

# 3d tensor-product B-spline space for finite elements
tensor_space_FEM = spl.tensor_spline_space(spaces_FEM)
# =======================================================================


# ========= geometry in case of spline mapping ==========================
# 1d B-spline spline spaces for mapping
spaces_MAP = [spl.spline_space_1d(Nel_MAP, p_MAP, bc_MAP) for Nel_MAP, p_MAP, bc_MAP in zip(Nel_MAP, p_MAP, bc_MAP)]

# 3d tensor-product B-spline space for mapping
tensor_space_MAP = spl.tensor_spline_space(spaces_MAP)
tensor_space_MAP.set_extraction_operators()

# number of basis functions for spline mapping
NbaseN_MAP = tensor_space_MAP.NbaseN

# interpolation of mapping on discrete space with local interpolator pi_0 (get controlpoints)
Fx = lambda eta1, eta2, eta3 : params_map[1]*eta1*np.cos(2*np.pi*eta2)
Fy = lambda eta1, eta2, eta3 : params_map[1]*eta1*np.sin(2*np.pi*eta2)
Fz = lambda eta1, eta2, eta3 : params_map[2]*eta3

#Fx = lambda eta1, eta2, eta3 : params_map[0]*eta1
#Fy = lambda eta1, eta2, eta3 : params_map[1]*eta2
#Fz = lambda eta1, eta2, eta3 : params_map[2]*eta3

pro_MAP = proj_global.projectors_global_3d(tensor_space_MAP, [p_MAP[0] + 1, p_MAP[1] + 1, p_MAP[2] + 1])

cx = pro_MAP.pi_0(Fx).reshape(tensor_space_MAP.Nbase_0form)
cy = pro_MAP.pi_0(Fy).reshape(tensor_space_MAP.Nbase_0form)
cz = pro_MAP.pi_0(Fz).reshape(tensor_space_MAP.Nbase_0form)

del pro_MAP

# create polar splines in poloidal plane
if polar == True:
    polar_splines = spl_pol.polar_splines(tensor_space_FEM, cx, cy)
else:
    polar_splines = None
    
# create domain object
domain = dom.domain(kind_map, params_map, tensor_space_MAP, cx, cy, cz)
    
# set extraction operators in tensor-product splines space
tensor_space_FEM.set_extraction_operators(polar_splines)
# =======================================================================


# ========= for plotting ================================================
import matplotlib.pyplot as plt
    
etaplot = [np.linspace(0., 1., 100), np.linspace(0., 1., 100), np.linspace(0., 1., 40)]

xplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'x')
yplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'y')
zplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'z')
# =======================================================================


# ======= particle accumulator (all processes) ==========================
if add_PIC == True:
    acc = pic_accumu.accumulation(tensor_space_FEM, domain, basis_u, mpi_comm, control)
# =======================================================================


# ======= reserve memory for FEM cofficients (all MPI processes) ========
# number of basis functions in different spaces (finite elements)
NbaseN     = tensor_space_FEM.NbaseN
NbaseD     = tensor_space_FEM.NbaseD

N_0form    = tensor_space_FEM.Nbase_0form
N_1form    = tensor_space_FEM.Nbase_1form
N_2form    = tensor_space_FEM.Nbase_2form
N_3form    = tensor_space_FEM.Nbase_3form

Ntot_0form = tensor_space_FEM.Ntot_0form
Ntot_1form = tensor_space_FEM.Ntot_1form
Ntot_2form = tensor_space_FEM.Ntot_2form
Ntot_3form = tensor_space_FEM.Ntot_3form

Nall_0form = tensor_space_FEM.E0.shape[0]
Nall_1form = tensor_space_FEM.E1.shape[0]
Nall_2form = tensor_space_FEM.E2.shape[0]
Nall_3form = tensor_space_FEM.E3.shape[0]

p3    = np.empty(Nall_3form, dtype=float)    # bulk pressure FEM coefficients
p0_eq = np.empty(Nall_0form, dtype=float)    # bulk pressure FEM coefficients (static background 0-form)
p3_eq = np.empty(Nall_3form, dtype=float)    # bulk pressure FEM coefficients (static background 3-form)

if   basis_u == 0:
    up     = np.empty(3*Nall_0form, dtype=float) # bulk velocity FEM coefficients (0-form)
    up_old = np.empty(3*Nall_0form, dtype=float) # bulk velocity FEM coefficients (0-form) from previous step 
elif basis_u == 2:
    up     = np.empty(  Nall_2form, dtype=float) # bulk velocity FEM coefficients (2-form)
    up_old = np.empty(  Nall_2form, dtype=float) # bulk velocity FEM coefficients (2-form) from previous step

b2    = np.empty(Nall_2form, dtype=float)       # magnetic field FEM coefficients (2-form)
b2_eq = np.empty(Nall_2form, dtype=float)       # magnetic field FEM coefficients (2-form) static background

r3    = np.empty(Nall_3form, dtype=float)    # bulk mass density FEM coefficients
r0_eq = np.empty(Nall_0form, dtype=float)    # bulk mass density FEM coefficients (static background 0-form)
r3_eq = np.empty(Nall_3form, dtype=float)    # bulk mass density FEM coefficients (static background 3-form)
# =======================================================================


# ============= projection of initial conditions and equilibrium ========
# create 3d projector
pro_3d = proj_global.projectors_global_3d(tensor_space_FEM, nq_pr, polar_splines)

# initial conditions
p3[:] = pro_3d.pi_3(1, domain)

if   basis_u == 0:
    
    up[0*Nall_0form:1*Nall_0form] = pro_3d.pi_0(2, domain)
    up[1*Nall_0form:2*Nall_0form] = pro_3d.pi_0(3, domain)
    up[2*Nall_0form:3*Nall_0form] = pro_3d.pi_0(4, domain)
    
elif basis_u == 2:    
    
    up[:] = pro_3d.pi_2([61, 62, 63], domain)

b2[:] = pro_3d.pi_2([5, 6, 7], domain)
r3[:] = pro_3d.pi_3(8, domain)

# equilibrium
r3_eq[:] = pro_3d.pi_3(11, domain)
r0_eq[:] = pro_3d.pi_0(12, domain)

p3_eq[:] = pro_3d.pi_3(31, domain)
p0_eq[:] = pro_3d.pi_0(41, domain)

b2_eq[:] = pro_3d.pi_2([21, 22, 23], domain)

# apply boundary conditions to u and b in first direction
tensor_space_FEM.apply_bc_2form(up, bc_u1)
tensor_space_FEM.apply_bc_2form(b2, bc_b1)


#u1, u2, u3 = tensor_space.unravel_2form(tensor_space.E2.T.dot(u))

#plt.plot(etaplot[0], tensor_space.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], u1)[:, 25, 0])
#plt.plot(etaplot[0], tensor_space.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], u2)[:,  0, 0])
#plt.show()
#sys.exit()


## initialization with white noise
#np.random.seed(1234)
#amps = 1e-3*np.random.rand(8, pr.shape[0], pr.shape[2])
#
#for k in range(pr.shape[1]):
#    pr[:, k, :] = amps[0]
#
#    u1[:, k, :] = amps[1]
#    u2[:, k, :] = amps[2]
#    u3[:, k, :] = amps[3]
#
#    b1[:, :, :] = 0.
#    b2[:, k, :] = amps[5]
#    b3[:, :, :] = 0.
#
#    rh[:, k, :] = amps[7]


## create parallel petsc vectors ====  
#p3_pet = VecToPetsc(p3)
#up_pet = VecToPetsc(up)
#b2_pet = VecToPetsc(b2)
#r3_pet = VecToPetsc(r3)
#
## get row ranges of calling process
#p3_start, p3_end = p3_pet.getOwnershipRange()
#up_start, up_end = up_pet.getOwnershipRange()
#b2_start, b2_end = b2_pet.getOwnershipRange()
#r3_start, r3_end = r3_pet.getOwnershipRange()

print('projection of initial conditions and equilibrium done!')
# ==========================================================================



if mpi_rank == 0:   
    # ==================== matrices ============================================
    # mass matrices in space of discrete 2-forms (V2) and discrete 3-forms (V3)
    tensor_space_FEM.assemble_M2(domain)
    tensor_space_FEM.assemble_M3(domain)

    if basis_u == 0:
        # mass matrix in space of discrete contravariant vector fields using the V0 basis (NNN)
        tensor_space_FEM.assemble_Mv0(domain)
        
    print('assembly of mass matrices done!')

    # create discrete grad, curl and div matrices
    tensor_space_FEM.set_derivatives(polar_splines)
    print('assembly of discrete derivatives done!')
    # =========================================================================
    
if mpi_rank == 0:    
    # ================== linear MHD operators =================================
    MHD = mhd_global.operators_mhd(pro_3d, bc_u1, bc_b1, dt, gamma)

    MHD.assemble_rhs_F( r3_eq, 'M', domain)
    MHD.assemble_rhs_F( p3_eq, 'P', domain)
    MHD.assemble_rhs_EF(b2_eq     , domain)
    MHD.assemble_rhs_PR(p3_eq)
    MHD.assemble_TF(tensor_space_FEM.CURL.T.dot(tensor_space_FEM.M2.dot(b2_eq)), domain)

    MHD.setOperators()

    # dummy coefficients in sub-step 2
    #g_dummy = pro_3d.apply_IinvT_V1(CURL.T.dot(M2.dot(b2)))
    #e_dummy = pro_3d.solve_V1(MHD.rhs_TAU.dot(up))
    #f_dummy = pro_3d.solve_V2(MHD.rhs_W.dot(up))
    #h_dummy = pro_3d.apply_IinvT_V2(M2.dot(up))

    #LHS2 = spa.bmat([[None       , None                 , -dt/2*MHD.rhs_TAU.T, None      , 1/2*Mu    , 1/2*MHD.rhs_W.T],
    #                 [None       , spa.identity(b2.size), None               , dt/2*CURL , None      , None           ],
    #                 [None       , CURL.T.dot(M2)       , -pro_3d.I1.T       , None      , None      , None           ], 
    #                 [MHD.rhs_TAU, None                 , None               , -pro_3d.I1, None      , None           ],
    #                 [MHD.rhs_W  , None                 , None               , None      , -pro_3d.I2, None           ],
    #                 [Mu         , None                 , None               , None      , None      , -pro_3d.I2.T   ]], format='lil')
#
#
    ## apply boundary conditions
    #if bc[0] == False:
    #    if bc_u1[1] == 'dirichlet':
    #        
    #        print('hello')
    #        
    #        lower = 2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]
    #        upper = 2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2]
#
    #        LHS2[lower:upper,      :     ] = 0.
    #        LHS2[lower:upper, lower:upper] = np.identity(NbaseD[1]*NbaseD[2])
#
#
    #LHS2     = LHS2.tocsr()
    #LHS2_ILU = spa.linalg.spilu(LHS2.tocsc())
    #LHS2_PRE = spa.linalg.LinearOperator(LHS2.shape, lambda x : LHS2_ILU.solve(x))
    
    # define LHS of linear system in step 2
    #def S2(x):
#
    #    u_in = x[:up.size]
    #    b_in = x[up.size:up.size + b2.size]
    #    g_in = x[up.size + b2.size:up.size + b2.size + g_dummy.size]
    #    e_in = x[-e_dummy.size:]
#
    #    u_out = A(u_in) - dt/2*MHD.rhs_TAU.T.dot(g_in)
    #    b_out = b_in + dt/2*CURL.dot(e_in)
    #    g_out = CURL.T.dot(M2.dot(b_in)) - pro_3d.I1.T.dot(g_in)
    #    e_out = MHD.rhs_TAU.dot(u_in) - pro_3d.I1.dot(e_in)
#
    #    # apply boundary conditions
    #    if bc[0] == False:
    #        if bc_u1[1] == 'dirichlet':
    #            u_out[2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]:2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2]] = u_in[2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]:2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2]]
#
    #    return np.concatenate((u_out, b_out, g_out, e_out))
#
    #LHS2_shape = up.size + b2.size + g_dummy.size + e_dummy.size
    #LHS2 = spa.linalg.LinearOperator((LHS2_shape, LHS2_shape), S2)
    

if mpi_rank == 0:
    # ========================= preconditioners ================================
    # assemble approximate inverse interpolation/histopolation matrices
    pro_3d.assemble_approx_inv(tol_approx_reduced)
    #pro_3d.assemble_approx_inv_V2()

    timea = time.time()
    MHD.setPreconditionerA(drop_tol_A, fill_fac_A)
    timeb = time.time()
    print('ILU of A_local done!', timeb - timea)
    
    timea = time.time()
    MHD.setPreconditionerS2(drop_tol_S2, fill_fac_S2)
    timeb = time.time()
    print('ILU of S2_local done!', timeb - timea)
    
    if add_pressure == True:
        timea = time.time()
        MHD.setPreconditionerS6(drop_tol_S6, fill_fac_S6)
        timeb = time.time()
        print('ILU of S6_local done!', timeb - timea)
    
    #LHS2_local = spa.bmat([[A_local    , None                 , -dt/2*MHD.rhs_TAU.T, None      ],
    #                       [None       , spa.identity(b2.size), None               , dt/2*CURL ], 
    #                       [None       , CURL.T.dot(M2)       , -pro_3d.I1.T       , None      ], 
    #                       [MHD.rhs_TAU, None                 , None               , -pro_3d.I1]], format='lil')
    
    
    ## apply boundary conditions
    #if bc[0] == False:
    #    if bc_u1[1] == 'dirichlet':
    #        
    #        print('hello')
    #        
    #        lower = 2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]
    #        upper = 2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2]
#
    #        LHS2_local[lower:upper,      :     ] = 0.
    #        LHS2_local[lower:upper, lower:upper] = np.identity(NbaseD[1]*NbaseD[2])

    #sys.exit()
    
    #LHS2_ILU = spa.linalg.spilu(LHS2_local.tocsc())
    #LHS2_PRE = spa.linalg.LinearOperator(LHS2_local.shape, lambda x : LHS2_ILU.solve(x))
    
    #del LHS2_local
# ===========================================================================
    

           
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
pic_sample.compute_weights_ini(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)

if control == True:
    pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
else:
    particles_loc[6] = w0_loc

print(mpi_rank, 'particle initialization done!')
# ======================================================================================



# ================ compute initial energies and distribution function ==================
# initial energies
if mpi_rank == 0:
    energies['U'][0] = 1/2*up.dot(MHD.A(up))
    energies['B'][0] = 1/2*b2.dot(tensor_space_FEM.M2.dot(b2))
    energies['p'][0] = 1/(gamma - 1)*sum(p3.flatten())

energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

# subtract equilibrium hot ion energy for analaytical mappings and full-f
if kind_map != 0:
    energies['f'] += (control - 1)*eq_PIC.eh_eq(domain.kind_map, domain.params_map)
    
# initial distribution function
fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)

print('initial diagnostics done')
# ======================================================================================


"""
if mpi_rank == 0:
    LHS  = spa.linalg.LinearOperator(A.shape, lambda x : A(x) + dt**2/4*TAU.T(CURL.T.dot(M2.dot(CURL.dot(TAU(x))))))
    
    timea = time.time()
    LHS(np.random.rand(3*Ntot_0form))
    timeb = time.time()
    
    print(timeb - timea)
    
sys.exit()
"""

 
# ==================== time integrator ==========================================
times_elapsed = {'total' : 0., 'accumulation_step1' : 0., 'accumulation_step3' : 0., 'pusher_step3' : 0., 'pusher_step4' : 0., 'pusher_step5' : 0., 'control_step1' : 0., 'control_step3' : 0., 'control_weights' : 0., 'update_step1u' : 0., 'update_step2u' : 0., 'update_step2b' : 0., 'update_step3u' : 0.,'update_step6' : 0.}

    
def update():
    
    global up, up_old, b2, b2_eq, p3, r3, particles_loc
    
    time_tota = time.time()
    
    # ====================================================================================
    #                           step 1 (1: update u)
    # ====================================================================================
    if add_PIC == True:
        
        
        # <<<<<<<<<<<<<<<<<<<<< charge accumulation (all processes) <<<<<<<<<<<<<<<<<
        timea = time.time()
        acc.accumulate_step1(particles_loc, b2 + b2_eq, mpi_comm)
        timeb = time.time()
        times_elapsed['accumulation_step1'] = timeb - timea
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        # <<<<<<<<< set up and solve linear system (only MHD process) <<<<<<<<<<<<<<
        if mpi_rank == 0:
            
            # build global sparse matrix 
            timea = time.time()
            mat   = acc.assemble_step1(Np, b2 + b2_eq)
            timeb = time.time()
            times_elapsed['control_step1'] = timeb - timea
                
            # solve linear system with conjugate gradient squared method and values from last time step as initial guess 
            timea = time.time()
            
            # LHS and RHS of linear system
            LHS = spa.linalg.LinearOperator(MHD.A.shape, lambda x : MHD.A(x) - dt/2*mat.dot(x))
            RHS = MHD.A(up) + dt/2*mat.dot(up)
            
            up[:], info = spa.linalg.gmres(LHS, RHS, x0=up, tol=tol1, M=MHD.A_PRE)
            print('linear solver step 1 : ', info)
            
            times_elapsed['update_step1u'] = timeb - timea
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    # ====================================================================================
    #                           step 1 (1: update u)
    # ====================================================================================
    
    
    ## ====================================================================================
    ##                       step 2 (1 : update u, 2 : update b) 
    ## ====================================================================================
    #if mpi_rank == 0:
    #    
    #    # solve for old dummy coefficients
    #    g_dummy[:] = pro_3d.apply_IinvT_V1(CURL.T.dot(M2.dot(b2)))
    #    e_dummy[:] = pro_3d.solve_V1(MHD.rhs_TAU.dot(up))
    #    #f_dummy[:] = pro_3d.solve_V2(MHD.rhs_W.dot(up))
    #    #h_dummy[:] = pro_3d.apply_IinvT_V2(M2.dot(up))
    #    
    #    # RHS of linear system
    #    #RHS1 = dt/2*MHD.rhs_TAU.T.dot(g_dummy) + 1/2*Mu.dot(f_dummy) + 1/2*MHD.rhs_W.T.dot(h_dummy)
    #    #RHS2 = b2 - dt/2*CURL.dot(e_dummy)
    #    #RHS3 = np.zeros(g_dummy.size + e_dummy.size + f_dummy.size + h_dummy.size, dtype=float)
    #    
    #    RHS1 = A(up) + dt/2*MHD.rhs_TAU.T.dot(g_dummy)
    #    RHS2 = b2 - dt/2*CURL.dot(e_dummy)
    #    RHS3 = np.zeros(g_dummy.size + e_dummy.size, dtype=float)
    #    
    #    RHS  = np.concatenate((RHS1, RHS2, RHS3))
    #    
    #    # apply boundary conditions at r=R to u1
    #    if bc[0] == False:
    #        if bc_u1[1] == 'dirichlet':
    #            RHS[(2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]):(2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2])] = 0.
    #            
    #    # solve linear system with conjugate gradient method (S2 is a symmetric positive definite matrix) and values from last time step as initial guess
    #    timea = time.time()
    #        
    #    temp, info = spa.linalg.gmres(LHS2, RHS, x0=np.concatenate((up, b2, g_dummy, e_dummy)), tol=tol2, maxiter=100, M=LHS2_PRE)
    #    print('linear solver step 2 : ', info)
    #    
    #    timeb = time.time()
    #    times_elapsed['update_step2u'] = timeb - timea
    #    
    #    up[:] = temp[:up.size]
    #    b2[:] = temp[up.size:up.size + b2.size]
    #    
    #    # apply boundary conditions at r=R to u1 and b1
    #    if bc[0] == False:
    #        if bc_u1[1] == 'dirichlet':
    #            up[(2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]):(2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2])] = 0.
    #        if bc_b1[1] == 'dirichlet':
    #            b2[(2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]):(2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2])] = 0.
    #
    ## broadcast new magnetic FEM coefficients
    #mpi_comm.Bcast(b2, root=0)
    ## ====================================================================================
    ##                       step 2 (1 : update u, 2 : update b) 
    ## ====================================================================================
    
    
    ## ====================================================================================
    ##                       step 2 (1 : update u, 2 : update b) 
    ## ====================================================================================
    #
    ## solve for old dummy coefficients
    #g_dummy[:] = pro_3d.apply_IinvT_V1(CURL.T.dot(M2.dot(b2)))
    #e_dummy[:] = pro_3d.solve_V1(MHD.rhs_TAU.dot(up))
    #f_dummy[:] = pro_3d.solve_V2(MHD.rhs_W.dot(up))
    #h_dummy[:] = pro_3d.apply_IinvT_V2(M2.dot(up))    
    #    
    ## RHS of linear system
    #RHS1 = dt/2*MHD.rhs_TAU.T.dot(g_dummy) + 1/2*Mu.dot(f_dummy) + 1/2*MHD.rhs_W.T.dot(h_dummy)
    #RHS2 = b2 - dt/2*CURL.dot(e_dummy)
    #RHS3 = np.zeros(g_dummy.size + e_dummy.size + f_dummy.size + h_dummy.size, dtype=float)
    #
    #RHS  = np.concatenate((RHS1, RHS2, RHS3))
    #
    #
    #
    ## apply boundary conditions at r=R to u1
    #if bc[0] == False:
    #    if bc_u1[1] == 'dirichlet':
    #        RHS[(2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]):(2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2])] = 0.
    #
    #RHS2 = VecToPetsc(RHS)
    #NEW2 = VecToPetsc(RHS)
    #
    #
    #
    #ksp2.solve(RHS2, NEW2)
    #
    ##print(NEW2.min(), NEW2.max())
    ##sys.exit()
    #
    ## distribute local updates to all processes
    #start, end = NEW2.getOwnershipRange()
    #mpi_comm.Allgather(NEW2[start:end], RHS)
    #
    #up[:] = RHS[:up.size]
    #b2[:] = RHS[up.size:up.size + b2.size]
    #
    ## broadcast new magnetic FEM coefficients
    #mpi_comm.Bcast(b2, root=0)
                
    ## solve linear system with conjugate gradient method (S2 is a symmetric positive definite matrix) and values from last time step as initial guess
    #timea = time.time()
#
    #temp = spa.linalg.gmres(BIG, RHS, x0=np.concatenate((u, b2, f_dummy, a_dummy, g_dummy, h_dummy)), tol=1e-6, M=BIG_PRE)[0]
#
    #timeb = time.time()
    #times_elapsed['update_step2u'] = timeb - timea
#
    #u[:]  = temp[:u.size]
    #b2[:] = temp[u.size:u.size + b2.size]
    #    
    ## apply boundary conditions
    #if bc[0] == False:
    #    if bc_u1[1] == 'dirichlet':
    #        u[(2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]):(2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2])] = 0.
#
#
#
#
    ### update magnetic field (strong)
    ##timea = time.time()
    ##
    ##b2[:] = b2 - dt/2*CURL.dot(TAU(u + u_old))
    ##
    ##timeb = time.time()
    ##times_elapsed['update_step2b'] = timeb - timea
    ##
    ## apply boundary conditions
    #if bc[0] == False:
    #    if bc_b1[1] == 'dirichlet':
    #        b2[(2*NbaseD[2] + (NbaseN[0] - 3)*NbaseD[1]*NbaseD[2]):(2*NbaseD[2] + (NbaseN[0] - 2)*NbaseD[1]*NbaseD[2])] = 0.
    #
    ## broadcast new magnetic FEM coefficients
    #mpi_comm.Bcast(b2, root=0)
    ## ====================================================================================
    ##                       step 2 (1 : update u, 2 : update b) 
    ## ====================================================================================
    
    
    # ====================================================================================
    #                       step 2 (1 : update u, 2 : update b) 
    # ====================================================================================
    if mpi_rank == 0:
        
        # save coefficients from previous time step
        up_old[:] = up
        
        # RHS of linear system
        RHS = MHD.RHS2(up, b2)
                
        # solve linear system with gmres method and values from last time step as initial guess (weak)
        timea = time.time()
            
        up[:], info = spa.linalg.gmres(MHD.S2, RHS, x0=up, tol=tol2, maxiter=100, M=MHD.S2_PRE)
        print('linear solver step 2 : ', info)
        
        timeb = time.time()
        times_elapsed['update_step2u'] = timeb - timea
        
        # apply boundary conditions at r=R to u1
        tensor_space_FEM.apply_bc_2form(up, bc_u1)
        
        # update magnetic field (strong)
        timea = time.time()
        b2[:] = b2 - dt*tensor_space_FEM.CURL.dot(MHD.EF((up + up_old)/2))
        timeb = time.time()
        times_elapsed['update_step2b'] = timeb - timea
        
        # apply boundary conditions at r=R to b1
        tensor_space_FEM.apply_bc_2form(b2, bc_b1)
    
    # broadcast new magnetic FEM coefficients
    mpi_comm.Bcast(b2, root=0)
    # ====================================================================================
    #                       step 2 (1 : update u, 2 : update b) 
    # ====================================================================================
    

    
    # ====================================================================================
    #             step 3 (1 : update u,  2 : update particles velocities V)
    # ====================================================================================
    if add_PIC == True:
        
        # <<<<<<<<<<<<<<<<<< current accumulation (all processes) <<<<<<<<<<<<
        timea = time.time()
        acc.accumulate_step3(particles_loc, b2 + b2_eq, mpi_comm)
        timeb = time.time()
        times_elapsed['accumulation_step3'] = timeb - timea
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        # <<<<<<<<<<<<<< set up and solve linear system (only MHD process) <<<
        if mpi_rank == 0:
            
            # save coefficients from previous time step
            up_old[:] = up
            
            # build global sparse matrix and vector
            timea    = time.time()
            mat, vec = acc.assemble_step3(Np, b2 + b2_eq)
            timeb    = time.time()
            times_elapsed['control_step3'] = timeb - timea
            
            # solve linear system with conjugate gradient method (A + dt**2*mat/4 is a symmetric positive definite matrix) with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess
            timea = time.time()
            
            # LHS and RHS of linear system
            LHS = spa.linalg.LinearOperator(MHD.A.shape, lambda x : MHD.A(x) + dt**2/4*mat.dot(x))
            RHS = MHD.A(up) - dt**2/4*mat.dot(up) + dt*vec
            
            up[:], info = spa.linalg.cg(LHS, RHS, x0=up, tol=tol3, M=MHD.A_PRE)
            print('linear solver step 3 : ', info)
            
            timeb = time.time()
            times_elapsed['update_step3u'] = timeb - timea
        
        # broadcast new FEM coefficients
        mpi_comm.Bcast(up    , root=0)
        mpi_comm.Bcast(up_old, root=0)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        # <<<<<<<<<<<<<<<<<<<< update velocities <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        timea = time.time()
        
        b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(b2 + b2_eq))
        up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot((up + up_old)/2))
        
        pic_pusher.pusher_step3(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, np.zeros(N_0form, dtype=float), up_ten_1, up_ten_2, up_ten_3, basis_u, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz, np.zeros(Np_loc, dtype=float))
        
        timeb = time.time()
        times_elapsed['pusher_step3'] = timeb - timea
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    # ====================================================================================
    #             step 3 (1 : update u,  2 : update particles velocities V)
    # ====================================================================================
    
    
    
    # ====================================================================================
    #             step 4 (1 : update particles positions ETA)
    # ====================================================================================
    if add_PIC == True:
        timea = time.time()
        pic_pusher.pusher_step4(particles_loc, dt, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
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
        
        b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(b2 + b2_eq))
        
        pic_pusher.pusher_step5(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            
        timeb = time.time()
        times_elapsed['pusher_step5'] = timeb - timea

        # update particle weights in case of delta-f
        if control == True:
            timea = time.time()
            pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
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
        
        #up[:] = 0.
        
        # save coefficients from previous time step
        up_old[:] = up
        
        # RHS of linear system
        RHS = MHD.RHS6(up, p3, b2)
        
        # solve linear system with conjugate gradient squared method and values from last time step as initial guess
        timea = time.time()
            
        up[:], info = spa.linalg.gmres(MHD.S6, RHS, x0=up, tol=tol6, maxiter=100, M=MHD.S6_PRE)
        print('linear solver step 6 : ', info)
        
        timeb = time.time()
        times_elapsed['update_step2u'] = timeb - timea
        
        # apply boundary conditions
        tensor_space_FEM.apply_bc_2form(up, bc_u1)
        
        # update pressure
        p3[:] = p3 + dt/2*MHD.L((up + up_old)/2)
        
        # update density
        r3[:] = r3 - dt/2*tensor_space_FEM.DIV.dot(MHD.FM((up + up_old)/2))

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
        energies['U'][0] = 1/2*up.dot(MHD.A(up))
        energies['B'][0] = 1/2*b2.dot(tensor_space_FEM.M2.dot(b2))
        energies['p'][0] = 1/(gamma - 1)*sum(p3.flatten())

    energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
    mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

    # subtract equilibrium hot ion energy for analaytical mappings and full-f
    if kind_map != 0:
        energies['f'] += (control - 1)*eq_PIC.eh_eq(domain.kind_map, domain.params_map)

    # distribution function
    fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
    mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)
    # ====================================================================================
    #                                diagnostics
    # ====================================================================================
    
    
    """
    # <<<<< apply Fourier filter in perodic directions <<<<<<<<<<<<
    spec = np.zeros((Nel[0], Nel[1], Nel[2]), dtype=complex)

    spec[ 1, 0, 0] = np.fft.fftn(u1)[ 1, 0, 0]
    spec[-1, 0, 0] = np.fft.fftn(u1)[-1, 0, 0]
    u1[:, :, :]    = np.real(np.fft.ifftn(spec))

    spec[ 1, 0, 0] = np.fft.fftn(u2)[ 1, 0, 0]
    spec[-1, 0, 0] = np.fft.fftn(u2)[-1, 0, 0]
    u2[:, :, :]    = np.real(np.fft.ifftn(spec))

    spec[ 1, 0, 0] = np.fft.fftn(u3)[ 1, 0, 0]
    spec[-1, 0, 0] = np.fft.fftn(u3)[-1, 0, 0]
    u3[:, :, :]    = np.real(np.fft.ifftn(spec))
    
    spec[ 1, 0, 0] = np.fft.fftn(b1)[ 1, 0, 0]
    spec[-1, 0, 0] = np.fft.fftn(b1)[-1, 0, 0]
    b1[:, :, :]    = np.real(np.fft.ifftn(spec))

    spec[ 1, 0, 0] = np.fft.fftn(b2)[ 1, 0, 0]
    spec[-1, 0, 0] = np.fft.fftn(b2)[-1, 0, 0]
    b2[:, :, :]    = np.real(np.fft.ifftn(spec))

    spec[ 1, 0, 0] = np.fft.fftn(b3)[ 1, 0, 0]
    spec[-1, 0, 0] = np.fft.fftn(b3)[-1, 0, 0]
    b3[:, :, :]    = np.real(np.fft.ifftn(spec))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    
    
    """
    # <<<<< apply Fourier filter in perodic directions <<<<<<<<<<<<
    spec = np.zeros((Nel[1], Nel[2]), dtype=complex)
    
    for i in range(u1.shape[0]):
        spec[ 1, -1] = np.fft.fftn(u1[i])[ 1, -1]
        spec[-1,  1] = np.fft.fftn(u1[i])[-1,  1]
        u1[i, :, :]  = np.real(np.fft.ifftn(spec))

    for i in range(u2.shape[0]):
        spec[ 1, -1] = np.fft.fftn(u2[i])[ 1, -1]
        spec[-1,  1] = np.fft.fftn(u2[i])[-1,  1]
        u2[i, :, :]  = np.real(np.fft.ifftn(spec))

    for i in range(u3.shape[0]):
        spec[ 1, -1] = np.fft.fftn(u3[i])[ 1, -1]
        spec[-1,  1] = np.fft.fftn(u3[i])[-1,  1]
        u3[i, :, :]  = np.real(np.fft.ifftn(spec))
    
    for i in range(b1.shape[0]):
        spec[ 1, -1] = np.fft.fftn(b1[i])[ 1, -1]
        spec[-1,  1] = np.fft.fftn(b1[i])[-1,  1]
        b1[i, :, :]  = np.real(np.fft.ifftn(spec))

    for i in range(b2.shape[0]):
        spec[ 1, -1] = np.fft.fftn(b2[i])[ 1, -1]
        spec[-1,  1] = np.fft.fftn(b2[i])[-1,  1]
        b2[i, :, :]  = np.real(np.fft.ifftn(spec))

    for i in range(b3.shape[0]):
        spec[ 1, -1] = np.fft.fftn(b3[i])[ 1, -1]
        spec[-1,  1] = np.fft.fftn(b3[i])[-1,  1]
        b3[i, :, :]  = np.real(np.fft.ifftn(spec))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """
    
    time_totb = time.time()
    times_elapsed['total'] = time_totb - time_tota
    
# ============================================================================



#update()
#
#u_ten_1, u_ten_2, u_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(up))
#b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(b2))
#
#plt.plot(etaplot[0], tensor_space_FEM.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], b2_ten_1)[:,  0, 0])
#plt.plot(etaplot[0], tensor_space_FEM.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], b2_ten_2)[:, 25, 0])
#plt.plot(etaplot[0], tensor_space_FEM.evaluate_DDN(etaplot[0], etaplot[1], etaplot[2], b2_ten_3)[:, 25, 0])
#
#plt.show()
#
#plt.plot(etaplot[0], tensor_space_FEM.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], u_ten_1)[:, 25, 0])
#plt.show()
#
#plt.plot(etaplot[0], tensor_space_FEM.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], u_ten_2)[:,  0, 0])
#plt.show()
#
#plt.plot(etaplot[0], tensor_space_FEM.evaluate_DDN(etaplot[0], etaplot[1], etaplot[2], u_ten_3)[:,  0, 0])
#plt.show()
#
#sys.exit()





# ========================== time integration ================================
if time_int == True:
    
    # a new simulation
    if restart == False:
    
        if mpi_rank == 0:
        
            # ============= create hdf5 file and datasets for simulation output ======================
            file = h5py.File('results_' + identifier + '.hdf5', 'a')

            # current time
            file.create_dataset('time'                            , (1,), maxshape=(None,), dtype=float, chunks=True)
            
            # energies
            file.create_dataset('energies/bulk_kinetic'           , (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/magnetic'               , (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/bulk_internal'          , (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/energetic'              , (1,), maxshape=(None,), dtype=float, chunks=True)

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
            sh_0_ini , sh_0_max  = (1, N_0form[0]   , N_0form[1]   , N_0form[2]   ), (None, N_0form[0]   , N_0form[1]   , N_0form[2])
            
            sh_11_ini, sh_11_max = (1, N_1form[0][0], N_1form[0][1], N_1form[0][2]), (None, N_1form[0][0], N_1form[0][1], N_1form[0][2])
            sh_12_ini, sh_12_max = (1, N_1form[1][0], N_1form[1][1], N_1form[1][2]), (None, N_1form[1][0], N_1form[1][1], N_1form[1][2])
            sh_13_ini, sh_13_max = (1, N_1form[2][0], N_1form[2][1], N_1form[2][2]), (None, N_1form[2][0], N_1form[2][1], N_1form[2][2])
            
            sh_21_ini, sh_21_max = (1, N_2form[0][0], N_2form[0][1], N_2form[0][2]), (None, N_2form[0][0], N_2form[0][1], N_2form[0][2])
            sh_22_ini, sh_22_max = (1, N_2form[1][0], N_2form[1][1], N_2form[1][2]), (None, N_2form[1][0], N_2form[1][1], N_2form[1][2])
            sh_23_ini, sh_23_max = (1, N_2form[2][0], N_2form[2][1], N_2form[2][2]), (None, N_2form[2][0], N_2form[2][1], N_2form[2][2])
            
            sh_3_ini , sh_3_max  = (1, N_3form[0]   , N_3form[1]   , N_3form[2]   ), (None, N_3form[0]   , N_3form[1]   , N_3form[2])
            
            
            file.create_dataset('pressure'                  , sh_3_ini , maxshape=sh_3_max , dtype=float, chunks=True)
            
            #file.create_dataset('velocity_field/1_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
            #file.create_dataset('velocity_field/2_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
            #file.create_dataset('velocity_field/3_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
            
            file.create_dataset('velocity_field/1_component', sh_21_ini, maxshape=sh_21_max , dtype=float, chunks=True)
            file.create_dataset('velocity_field/2_component', sh_22_ini, maxshape=sh_22_max , dtype=float, chunks=True)
            file.create_dataset('velocity_field/3_component', sh_23_ini, maxshape=sh_23_max , dtype=float, chunks=True)
            
            
            file.create_dataset('magnetic_field/1_component', sh_21_ini, maxshape=sh_21_max, dtype=float, chunks=True)
            file.create_dataset('magnetic_field/2_component', sh_22_ini, maxshape=sh_22_max, dtype=float, chunks=True)
            file.create_dataset('magnetic_field/3_component', sh_23_ini, maxshape=sh_23_max, dtype=float, chunks=True)
            file.create_dataset('density'                   , sh_3_ini , maxshape=sh_3_max , dtype=float, chunks=True)

            # particles
            file.create_dataset('particles', (1, 7, Np), maxshape=(None, 7, Np), dtype=float, chunks=True)
            
            # other diagnostics
            file.create_dataset('bulk_mass', (1,), maxshape=(None,), dtype=float, chunks=True)

            file.create_dataset('magnetic_field/divergence', sh_3_ini , maxshape=sh_3_max , dtype=float, chunks=True)

            file.create_dataset('distribution_function/eta1_vx', (1, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), maxshape=(None, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float, chunks=True)

            # datasets for restart function
            file.create_dataset('restart/time_steps_done', (1,), maxshape=(None,), dtype=int, chunks=True)

            file.create_dataset('restart/pressure'                  , sh_3_ini , maxshape=sh_3_max , dtype=float, chunks=True)
            
            #file.create_dataset('restart/velocity_field/1_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
            #file.create_dataset('restart/velocity_field/2_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
            #file.create_dataset('restart/velocity_field/3_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
            
            file.create_dataset('restart/velocity_field/1_component', sh_21_ini , maxshape=sh_21_max , dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/2_component', sh_22_ini , maxshape=sh_22_max , dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/3_component', sh_23_ini , maxshape=sh_23_max , dtype=float, chunks=True)
            
            file.create_dataset('restart/magnetic_field/1_component', sh_21_ini, maxshape=sh_21_max, dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/2_component', sh_22_ini, maxshape=sh_22_max, dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/3_component', sh_23_ini, maxshape=sh_23_max, dtype=float, chunks=True)
            file.create_dataset('restart/density'                   , sh_3_ini , maxshape=sh_3_max , dtype=float, chunks=True)

            file.create_dataset('restart/particles', (1, 6, Np), maxshape=(None, 6, Np), dtype=float, chunks=True)

            file.create_dataset('restart/control_w0', (Np,), dtype=float)
            file.create_dataset('restart/control_s0', (Np,), dtype=float)


            # ==================== save initial data =======================
            file['time'][0]                       = 0.

            file['energies/bulk_kinetic'][0]      = energies['U'][0]
            file['energies/magnetic'][0]          = energies['B'][0]
            file['energies/bulk_internal'][0]     = energies['p'][0]
            file['energies/energetic'][0]         = energies['f'][0]
            
            u_ten_1, u_ten_2, u_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(up))
            b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(b2))

            file['pressure'][0]                   = tensor_space_FEM.E3.T.dot(p3).reshape(N_3form)
            file['velocity_field/1_component'][0] = u_ten_1
            file['velocity_field/2_component'][0] = u_ten_2
            file['velocity_field/3_component'][0] = u_ten_3
            file['magnetic_field/1_component'][0] = b2_ten_1
            file['magnetic_field/2_component'][0] = b2_ten_2
            file['magnetic_field/3_component'][0] = b2_ten_3
            file['density'][0]                    = tensor_space_FEM.E3.T.dot(r3).reshape(N_3form)

            file['magnetic_field/divergence'][0]  = tensor_space_FEM.E3.T.dot(tensor_space_FEM.DIV.dot(b2)).reshape(N_3form)

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
            pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, kind_map, params_map, tensor_space_MAP.T[0], tensor_space_MAP.T[1], tensor_space_MAP.T[2], p_MAP, NbaseN_MAP, cx, cy, cz)
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
                    
                    u_ten_1, u_ten_2, u_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(up))
                    b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(b2))
                    
                    file['restart/time_steps_done'][-1]            = time_steps_done
                    file['restart/pressure'][-1]                   = tensor_space_FEM.E3.T.dot(p3).reshape(N_3form)
                    file['restart/velocity_field/1_component'][-1] = u_ten_1
                    file['restart/velocity_field/2_component'][-1] = u_ten_2
                    file['restart/velocity_field/3_component'][-1] = u_ten_3
                    file['restart/magnetic_field/1_component'][-1] = b2_ten_1
                    file['restart/magnetic_field/2_component'][-1] = b2_ten_2
                    file['restart/magnetic_field/3_component'][-1] = b2_ten_3
                    file['restart/density'][-1]                    = tensor_space_FEM.E3.T.dot(r3).reshape(N_3form)

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
            file['pressure'].resize(file['pressure'].shape[0] + 1, axis = 0)
            file['pressure'][-1] = tensor_space_FEM.E3.T.dot(p3).reshape(N_3form)
            
            u_ten_1, u_ten_2, u_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(up))
            b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.unravel_2form(tensor_space_FEM.E2.T.dot(b2))

            file['magnetic_field/1_component'].resize(file['magnetic_field/1_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/2_component'].resize(file['magnetic_field/2_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/3_component'].resize(file['magnetic_field/3_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/1_component'][-1] = b2_ten_1
            file['magnetic_field/2_component'][-1] = b2_ten_2
            file['magnetic_field/3_component'][-1] = b2_ten_3

            file['velocity_field/1_component'].resize(file['velocity_field/1_component'].shape[0] + 1, axis = 0)
            file['velocity_field/2_component'].resize(file['velocity_field/2_component'].shape[0] + 1, axis = 0)
            file['velocity_field/3_component'].resize(file['velocity_field/3_component'].shape[0] + 1, axis = 0)
            file['velocity_field/1_component'][-1] = u_ten_1
            file['velocity_field/2_component'][-1] = u_ten_2
            file['velocity_field/3_component'][-1] = u_ten_3
            
            #file['density'].resize(file['density'].shape[0] + 1, axis = 0)
            #file['density'][-1] = tensor_space_FEM.E3.T.dot(r3).reshape(N_3form)

            
            # other diagnostics
            #file['magnetic_field/divergence'].resize(file['magnetic_field/divergence'].shape[0] + 1, axis = 0)
            #file['magnetic_field/divergence'][-1] = DIV.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten()))).reshape(N_3form[0], N_3form[1], N_3form[2])

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
        # ==============================================================================================
        
# ============================================================================