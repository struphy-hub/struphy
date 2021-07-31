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
import hylife.geometry.domain_3d as dom

import hylife.utilitis_FEEC.spline_space as spl

import hylife.utilitis_FEEC.projectors.mhd_operators_3d_global as mhd

import hylife.utilitis_PIC.sobol_seq    as sobol
import hylife.utilitis_PIC.pusher       as pic_pusher
import hylife.utilitis_PIC.accumulation as pic_accumu

import hylife.dispersion_relations.MHD_eigenvalues_2D as eig_MHD

# load local input and source files
import input_run.equilibrium_PIC        as equ_PIC
import input_run.equilibrium_MHD        as equ_MHD
import input_run.initial_conditions_MHD as ini_MHD

import source_run.sampling as pic_sample


# ======================= load some parameters =============================
identifier = 'sed_replace_run_dir'   

with open('parameters_sed_replace_run_dir.yml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# -------- mesh parameters -------------------
# number of elements, clamped (False) or periodic (True) splines and spline degrees
Nel        = params['Nel']
spl_kind   = params['spl_kind']
p          = params['p']

# boundary conditions for U2_1 (or Uv_1) and B2_1 in eta_1 direction
bc         = params['bc']

# number of quadrature points per element (nq_el) and histopolation cell (nq_pr)
nq_el      = params['nq_el']
nq_pr      = params['nq_pr']

# polar splines in eta_1-eta_2 (usually poloidal) plane?
polar      = params['polar']

# representation of MHD bulk velocity (0 : vector field with 0-form basis for each component, 2 : 2-form)
basis_u    = params['basis_u']

# geometry
geometry   = params['geometry']
params_map = params['params_map']
# --------------------------------------------

# ---- MHD equilibrium file ------------------
eq_type = params['eq_type']
# --------------------------------------------

# --------- time integration -----------------
dt       = params['dt']
Tend     = params['Tend']
max_time = params['max_time']
# --------------------------------------------

# ---------- linear solvers ------------------
# for ILU preconditioners
drop_tol_A  = params['drop_tol_A']
fill_fac_A  = params['fill_fac_A']

drop_tol_S2 = params['drop_tol_S2']
fill_fac_S2 = params['fill_fac_S2']

drop_tol_S6 = params['drop_tol_S6']
fill_fac_S6 = params['fill_fac_S6']

# parameters for iterative linear solvers
solver_type_2 = params['solver_type_2']
solver_type_3 = params['solver_type_3']

tol1 = params['tol1']
tol2 = params['tol2']
tol3 = params['tol3']
tol6 = params['tol6']

maxiter1 = params['maxiter1']
maxiter2 = params['maxiter2']
maxiter3 = params['maxiter3']
maxiter6 = params['maxiter6']
# --------------------------------------------

# ----------- kinetic parameters -------------
add_PIC = params['add_PIC']
Np      = params['Np']
control = params['control']
v0      = params['v0']                    
vth     = params['vth']
loading = params['loading']
seed    = params['seed']
# --------------------------------------------
# ========================================================================


# ================= MPI initialization for particles =====================
Np_loc        = int(Np/mpi_size)                      # number of particles per MPI process

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

# energies after Hamiltonian steps
energies_H   = {'U' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float)}

# snapshots of distribution function via particle binning
n_bins       = {'eta1_vx' : [32, 64]}
bin_edges    = {'eta1_vx' : [np.linspace(0., 1., n_bins['eta1_vx'][0] + 1), np.linspace(0., 5., n_bins['eta1_vx'][1] + 1)]}
dbin         = {'eta1_vx' : [bin_edges['eta1_vx'][0][1] - bin_edges['eta1_vx'][0][0], bin_edges['eta1_vx'][1][1] - bin_edges['eta1_vx'][1][0]]}
                
fh_loc       = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
fh           = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
# ========================================================================


# ================== create domain object ===============================

# 3D discrete spline mapping
if geometry == 'spline':

    X = lambda eta1, eta2, eta3 : 1.0*eta1
    Y = lambda eta1, eta2, eta3 : 1.0*eta2
    Z = lambda eta1, eta2, eta3 : 1.0*eta3
                               
    cx, cy, cz = dom.interp_mapping(Nel, p, spl_kind, X, Y, Z)
    
    # create domain object
    domain = dom.domain(geometry, params_map, Nel, p, spl_kind, cx, cy, cz)
    
# 2D discrete spline mapping at eta_3 = 0, analytical in eta_3
elif geometry == 'spline cylinder' or geometry == 'spline torus':
    
    X = lambda eta1, eta2 : params['a']*eta1*np.cos(2*np.pi*eta2) + params['R0']
    Y = lambda eta1, eta2 : params['a']*eta1*np.sin(2*np.pi*eta2)
    
    cx, cy = dom.interp_mapping(Nel[:2], p[:2], spl_kind[:2], X, Y)
    
    # create domain object
    domain = dom.domain(geometry, params_map, Nel[:2], p[:2], spl_kind[:2], cx, cy)
          
# 3D analytical mapping
elif geometry == 'cuboid' or geometry == 'colella':
    
    # create domain object
    domain = dom.domain(geometry, params_map)
    
else:
    raise NotImplementedError('The specified geometry is not implemented yet!')
    
    
# for plotting
if params['enable_plotting']:
    
    import matplotlib.pyplot as plt

    etaplot = [np.linspace(0., 1., 100), np.linspace(0., 1., 100), np.linspace(0., 1., 10)]

    xplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'x')
    yplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'y')
    zplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'z')

    E1, E2, E3 = np.meshgrid(etaplot[0], etaplot[1], etaplot[2], indexing='ij')

#plt.scatter(domain.cx[:, :, 0].flatten(), domain.cy[:, :, 0].flatten())
#plt.axis('square')
#plt.show()
#sys.exit()
# =======================================================================


# ============================ load MHD equilibrium =====================
if   eq_type == 'slab':
    eq_MHD = equ_MHD.equilibrium_mhd(domain, params['B0x'], params['B0y'], params['B0z'], params['rho0'], params['beta_s'])
elif eq_type == 'circular':
    eq_MHD = equ_MHD.equilibrium_mhd(domain, params['a'], params['R0'], params['B0'], params['q0'], params['q1'], params['q_add'], params['rl'], params['bmp0'], params['cg0'], params['wg0'], params['bmp1'], params['cg1'], params['wg1'], params['bmp2'], params['cg2'], params['wg2'], params['shafranov'], params['r1'], params['r2'], params['rho_a'], params['beta'], params['p1'], params['p2'])
else:
    raise NotImplementedError('Only equilibra for slab and circular geometry available!')

#plt.plot(etaplot[0], eq_MHD.q(etaplot[0]*params['a']), label='$q$')
#plt.plot(etaplot[0], eq_MHD.p(etaplot[0]*params['a']), label='$p$')
#plt.plot(etaplot[0], eq_MHD.rho(etaplot[0]*params['a']), label=r'$\rho$')
#plt.xlabel('$x$ [m]')
#plt.legend()
#plt.show()
#sys.exit()    
# =======================================================================


# ======================== MHD eigenvalue solver ========================
if params['run_mode'] == 0:
    print('Start of MHD eigenvalue computation on a ' + str(Nel[:2]) + ' grid with splines of degree ' + str(p[:2]))
    
    omega2, U2_eig = eig_MHD.solve_ev_problem_FEEC_2D([Nel[:2], p[:2], spl_kind[:2], nq_el[:2], nq_pr[:2], bc], domain, eq_MHD, params['n_tor'], project_profiles=params['profiles'], return_kind=1, dir_out=params['dir_eig'])
    
    sys.exit()
# =======================================================================


# ================== FEM spaces ==========================================
# 1d B-spline spline spaces for finite elements (with boundary conditions for first space)
spaces_FEM_1 = spl.spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc)
spaces_FEM_2 = spl.spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1])
spaces_FEM_3 = spl.spline_space_1d(Nel[2], p[2], spl_kind[2], nq_el[2])

# 3d tensor-product B-spline space for finite elements
tensor_space_FEM = spl.tensor_spline_space([spaces_FEM_1, spaces_FEM_2, spaces_FEM_3])

if polar == True:
    tensor_space_FEM.set_polar_splines(domain.cx[:, :, 0], domain.cy[:, :, 0])

tensor_space_FEM.set_projectors('general', nq_pr)
# =======================================================================


# ======= reserve memory for FEM cofficients (all MPI processes) ========
# number of basis functions in different spaces (finite elements)
NbaseN  = tensor_space_FEM.NbaseN
NbaseD  = tensor_space_FEM.NbaseD

N_0form = tensor_space_FEM.Nbase_0form
N_1form = tensor_space_FEM.Nbase_1form
N_2form = tensor_space_FEM.Nbase_2form
N_3form = tensor_space_FEM.Nbase_3form

N_dof_all_0form = tensor_space_FEM.E0_all.shape[0]
N_dof_all_1form = tensor_space_FEM.E1_all.shape[0]
N_dof_all_2form = tensor_space_FEM.E2_all.shape[0]
N_dof_all_3form = tensor_space_FEM.E3_all.shape[0]

N_dof_0form = tensor_space_FEM.E0.shape[0]
N_dof_1form = tensor_space_FEM.E1.shape[0]
N_dof_2form = tensor_space_FEM.E2.shape[0]
N_dof_3form = tensor_space_FEM.E3.shape[0]

# initial conditions (density, pressure, magnetic field, velocity)
r3 = np.zeros(N_dof_3form, dtype=float)
p3 = np.zeros(N_dof_3form, dtype=float)
b2 = np.zeros(N_dof_2form, dtype=float)

if   basis_u == 0:
    up     = np.zeros(N_dof_0form + 2*N_dof_all_0form, dtype=float)
    up_old = np.zeros(N_dof_0form + 2*N_dof_all_0form, dtype=float)
elif basis_u == 2:
    up     = np.zeros(N_dof_2form, dtype=float)
    up_old = np.zeros(N_dof_2form, dtype=float)
# =======================================================================


# ==== initialization with 2d MHD eigenfunctions ========================
if params['run_mode'] == 1:
    
    U2_ini = eig_MHD.solve_ev_problem_FEEC_2D([Nel[:2], p[:2], spl_kind[:2], nq_el[:2], nq_pr[:2], bc], domain, eq_MHD, params['n_tor'], project_profiles=params['profiles'], return_kind=params['eig_kind'], om2=params['eig_freq'])
    
    print('Squared eigenfrequency : ', U2_ini[3])

    up[:] = tensor_space_FEM.projectors.pi_2(U2_ini[:3], include_bc=False, eval_kind='tensor_product', interp=True)

    
    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], U2_ini[0](etaplot[0], etaplot[1], np.array([0.03]))[:, :, 0], levels=50, cmap='jet')
    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], U2_ini[1](etaplot[0], etaplot[1], np.array([0.03]))[:, :, 0], levels=50, cmap='jet')
    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], U2_ini[2](etaplot[0], etaplot[1], np.array([0.03]))[:, :, 0], levels=50, cmap='jet')
    
    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], tensor_space_FEM.evaluate_NDD(etaplot[0], etaplot[1], np.array([0.03]), up)[:, :, 0], levels=50, cmap='jet')
    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], tensor_space_FEM.evaluate_DND(etaplot[0], etaplot[1], np.array([0.03]), up)[:, :, 0], levels=50, cmap='jet')
    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], tensor_space_FEM.evaluate_DDN(etaplot[0], etaplot[1], np.array([0.03]), up)[:, :, 0], levels=50, cmap='jet')

    #plt.axis('square')
    #plt.colorbar()
    #plt.show()
    #sys.exit()
# ======================================================================


# ==== initialization with projection of input functions ===============
elif params['run_mode'] == 2:
    
    # load initial conditions
    in_MHD = ini_MHD.initial_mhd(domain)
    
    R3_ini =  in_MHD.r3_ini
    P3_ini =  in_MHD.p3_ini
    B2_ini = [in_MHD.b2_ini_1, in_MHD.b2_ini_2, in_MHD.b2_ini_3]
    
    # perform projections to get initial coefficients
    r3[:] = tensor_space_FEM.projectors.pi_3(R3_ini, include_bc=False, eval_kind='tensor_product', interp=True)
    p3[:] = tensor_space_FEM.projectors.pi_3(P3_ini, include_bc=False, eval_kind='tensor_product', interp=True)
    b2[:] = tensor_space_FEM.projectors.pi_2(B2_ini, include_bc=False, eval_kind='tensor_product', interp=True)
    
    if   basis_u == 0:
        
        Uv_ini = [in_MHD.uv_ini_1, in_MHD.uv_ini_2, in_MHD.uv_ini_3]

        up_1   = tensor_space_FEM.projectors.pi_0(Uv_ini[0], include_bc=False, eval_kind='tensor_product', interp=True)
        up_2   = tensor_space_FEM.projectors.pi_0(Uv_ini[1], include_bc=True , eval_kind='tensor_product', interp=True)
        up_3   = tensor_space_FEM.projectors.pi_0(Uv_ini[2], include_bc=True , eval_kind='tensor_product', interp=True)

        up[:]  = np.concatenate((up_1, up_2, up_3))
        
    elif basis_u == 2:
        
        U2_ini = [in_MHD.u2_ini_1, in_MHD.u2_ini_2, in_MHD.u2_ini_3]

        up[:]  = tensor_space_FEM.projectors.pi_2(U2_ini, include_bc=False, eval_kind='tensor_product', interp=True)
        
# =======================================================================
    
        
# ===== initialization with white noise on periodic domain ==============
elif params['run_mode'] == 3:
    np.random.seed(1607)

    p3_temp = np.empty(N_3form   , dtype=float)

    u1_temp = np.empty(N_0form   , dtype=float)
    u2_temp = np.empty(N_0form   , dtype=float)
    u3_temp = np.empty(N_0form   , dtype=float)

    b1_temp = np.empty(N_2form[0], dtype=float)
    b2_temp = np.empty(N_2form[1], dtype=float)
    b3_temp = np.empty(N_2form[2], dtype=float)

    r3_temp = np.empty(N_3form   , dtype=float)

    # spectrum in xy-plane
    if params['plane'] == 'xy':
        amps = np.random.rand(8, NbaseN[0], NbaseN[1])

        for k in range(NbaseN[2]):
            p3_temp[:, :, k] = amps[0]

            u1_temp[:, :, k] = amps[1]
            u2_temp[:, :, k] = amps[2]
            u3_temp[:, :, k] = amps[3]

            b1_temp[:, :, :] = 0.
            b2_temp[:, :, :] = 0.
            b3_temp[:, :, k] = amps[6]

            r3_temp[:, :, k] = amps[7]

    # for spectrum in yz-plane
    elif params['plane'] == 'yz':
        amps = np.random.rand(8, NbaseN[1], NbaseN[2])

        for k in range(NbaseN[0]):
            p3_temp[k, :, :] = amps[0]

            u1_temp[k, :, :] = amps[1]
            u2_temp[k, :, :] = amps[2]
            u3_temp[k, :, :] = amps[3]

            b1_temp[k, :, :] = amps[4]
            b2_temp[:, :, :] = 0.
            b3_temp[:, :, :] = 0.

            r3_temp[k, :, :] = amps[7]

    # for spectrum in xz-plane
    elif params['plane'] == 'xz':
        amps = np.random.rand(8, NbaseN[0], NbaseN[2])

        for k in range(NbaseN[1]):
            p3_temp[:, k, :] = amps[0]

            u1_temp[:, k, :] = amps[1]
            u2_temp[:, k, :] = amps[2]
            u3_temp[:, k, :] = amps[3]

            b1_temp[:, :, :] = 0.
            b2_temp[:, k, :] = amps[5]
            b3_temp[:, :, :] = 0.

            r3_temp[:, :, :] = amps[7]

    p3[:] = p3_temp.flatten()
    up[:] = np.concatenate((u1_temp.flatten(), u2_temp.flatten(), u3_temp.flatten()))
    b2[:] = np.concatenate((b1_temp.flatten(), b2_temp.flatten(), b3_temp.flatten()))
    r3[:] = r3_temp.flatten()
    
    
else:
    print('Wrong initialization specified!')
    sys.exit()

print('Loading of initial conditions done!')
# ========================================================================



# ================= mass matrices ========================================
if mpi_rank == 0: 
    
    # assemble mass matrices in space of discrete 2-forms (V2) and discrete 3-forms (V3)
    tensor_space_FEM.assemble_M2(domain)
    tensor_space_FEM.assemble_M3(domain)
    
    print('Assembly of mass matrices M2 and M3 done!')

    if basis_u == 0:
        # mass matrix in space of discrete contravariant vector fields using the V0 basis (NNN)
        tensor_space_FEM.assemble_Mv(domain)
        
        print('Assembly of mass matrix Mv done!')
    
# ========================================================================
    

# ================== linear MHD operators ================================
# projection of input equilibrium
R3_eq =  eq_MHD.r3_eq
P3_eq =  eq_MHD.p3_eq

J2_eq = [eq_MHD.j2_eq_1, eq_MHD.j2_eq_2, eq_MHD.j2_eq_3]
B2_eq = [eq_MHD.b2_eq_1, eq_MHD.b2_eq_2, eq_MHD.b2_eq_3]

#r3_eq = tensor_space_FEM.projectors.pi_3(R3_eq, include_bc=True, eval_kind='tensor_product', interp=True)
#p3_eq = tensor_space_FEM.projectors.pi_3(P3_eq, include_bc=True, eval_kind='tensor_product', interp=True)
#j2_eq = tensor_space_FEM.projectors.pi_2(J2_eq, include_bc=True, eval_kind='tensor_product', interp=True)
b2_eq = tensor_space_FEM.projectors.pi_2(B2_eq, include_bc=True, eval_kind='tensor_product', interp=True)

if mpi_rank == 0:  
    # interface for semi-discrete linear MHD operators
    MHD = mhd.operators_mhd(tensor_space_FEM.projectors, dt, eq_MHD.gamma, params['loc_j_eq'], basis_u)
    
    # assemble right-hand sides of projection matrices
    MHD.assemble_rhs_EF(domain     , B2_eq)
    MHD.assemble_rhs_F( domain, 'm', R3_eq)
    MHD.assemble_rhs_F( domain, 'p', P3_eq)
    MHD.assemble_rhs_PR(domain     , P3_eq)
    
    if basis_u == 0:
        MHD.assemble_rhs_W(domain, R3_eq)
        MHD.assemble_rhs_F(domain, 'j')
        
    print('Assembly of MHD projection operators done!')
        
    # assemble mass matrix weighted with 0-form density
    MHD.assemble_MR(domain, R3_eq)
    
    print('Assembly of weighted mass matrix done (density)!')
    
    # assemble mass matrix weighted with J_eq x
    MHD.assemble_JB_strong(domain, J2_eq)
    
    print('Assembly of weighted mass matrix done (current)!')
    
    # create liner MHD operators as scipy.sparse.linalg.LinearOperator
    MHD.setOperators()
    
    print('Assembly of MHD operators finished!')
# =======================================================================

    
# ======================== create particles ======================================
# particle accumulator (all processes)
if add_PIC == True:
    acc = pic_accumu.accumulation(tensor_space_FEM, domain, basis_u, mpi_comm, control)

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

#b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2 + b2_eq)
#        
#if basis_u == 0:
#    up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_0form_vec((up + up_old)/2)
#else:
#    up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_2form((up + up_old)/2)
#
#timea = time.time() 
#pic_pusher.pusher_step3(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, np.zeros(N_0form, dtype=float), up_ten_1, up_ten_2, up_ten_3, basis_u, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz, np.zeros(Np_loc, dtype=float))
#timeb = time.time()
#print('time for pusher step 3 : ', timeb - timea)
#
#timea = time.time() 
#pic_pusher.pusher_step4(particles_loc, dt, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
#timeb = time.time()
#print('time for pusher step 4 : ', timeb - timea)
#
#timea = time.time() 
#pic_pusher.pusher_step5_ana(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
#timeb = time.time()
#print('time for pusher step 5 : ', timeb - timea)
#
#timea = time.time()
#acc.accumulate_step1(particles_loc, b2 + b2_eq, mpi_comm)
#timeb = time.time()
#print('time for accumulation step 1 : ', timeb - timea)
#
#timea = time.time()
#acc.accumulate_step3(particles_loc, b2 + b2_eq, mpi_comm)
#timeb = time.time()
#print('time for accumulation step 3 : ', timeb - timea)
#
#timea = time.time() 
#pic_sample.compute_weights_ini(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
#timeb = time.time()
#print('time for initial weights : ', timeb - timea)
#
#timea = time.time()
#pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
#timeb = time.time()
#print('time for weight update : ', timeb - timea)
#
#sys.exit()


# ================ compute initial energies and distribution function ===========
# initial energies
if mpi_rank == 0:
    energies['U'][0] = 1/2*up.dot(MHD.A(up))
    energies['B'][0] = 1/2*b2.dot(tensor_space_FEM.M2.dot(b2))
    energies['p'][0] = 1/(eq_MHD.gamma - 1)*sum(p3.flatten())
    
    energies_H['U'][0] = energies['U'][0]
    energies_H['p'][0] = energies['p'][0]

energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

# subtract equilibrium hot ion energy for analaytical mappings and full-f
if domain.kind_map >= 10:
    energies['f'] += (control - 1)*equ_PIC.eh_eq(domain.kind_map, domain.params_map)
    
# initial distribution function
fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)

print('initial diagnostics done')
# ===============================================================================


# =============== preconditioners for time integration ==========================
if mpi_rank == 0:
    
    # assemble approximate inverse interpolation/histopolation matrices
    if params['PRE'] == 'ILU':
        tensor_space_FEM.projectors.assemble_approx_inv(params['tol_inv'])
    
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



# ==================== time integrator ==========================================
times_elapsed = {'total' : 0., 'accumulation_step1' : 0., 'accumulation_step3' : 0., 'pusher_step3' : 0., 'pusher_step4' : 0., 'pusher_step5' : 0., 'control_step1' : 0., 'control_step3' : 0., 'control_weights' : 0., 'update_step1u' : 0., 'update_step2u' : 0., 'update_step2b' : 0., 'update_step3u' : 0.,'update_step6' : 0.}

    
def update():
    
    # === counter for number of interation steps in iterative solvers ===
    num_iters = 0
    
    def count_iters(xk):
        nonlocal num_iters
        num_iters += 1
    # ===================================================================
    
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
        #print('accumulation_step1 : ', timeb - timea)
        
        # <<<<<<<<< set up and solve linear system (only MHD process) <<<<<<<<<<<<<<
        if mpi_rank == 0:
            
            # build global sparse matrix 
            timea = time.time()
            mat   = acc.assemble_step1(Np, b2 + b2_eq)
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
            print('linear solver step 1 : ', info, num_iters)
            
            timeb = time.time()
            times_elapsed['update_step1u'] = timeb - timea
            #print('update_step1u : ', timeb - timea)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
    # ====================================================================================
    #                           step 1 (1: update u)
    # ====================================================================================
    
    
    
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
            
        num_iters = 0
        
        if   solver_type_2 == 'gmres':
            up[:], info = spa.linalg.gmres(MHD.S2, RHS, x0=up, tol=tol2, maxiter=maxiter2, M=MHD.S2_PRE, callback=count_iters)
        elif solver_type_2 == 'cg':
            up[:], info = spa.linalg.cg(   MHD.S2, RHS, x0=up, tol=tol2, maxiter=maxiter2, M=MHD.S2_PRE, callback=count_iters)
        elif solver_type_2 == 'cgs':
            up[:], info = spa.linalg.cgs(  MHD.S2, RHS, x0=up, tol=tol2, maxiter=maxiter2, M=MHD.S2_PRE, callback=count_iters)
        else:
            raise ValueError('only gmres and cg solvers available')
            
        print('linear solver step 2 : ', info, num_iters)
        
        timeb = time.time()
        times_elapsed['update_step2u'] = timeb - timea
        #print('update_step2u : ', timeb - timea)
        
        # update magnetic field (strong)
        timea = time.time()
        b2[:] = b2 - dt*tensor_space_FEM.C.dot(MHD.EF((up + up_old)/2))
        timeb = time.time()
        times_elapsed['update_step2b'] = timeb - timea
        #print('update_step2b : ', timeb - timea)
    
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
        #print('accumulation_step3 : ', timeb - timea)
        
        # <<<<<<<<<<<<<< set up and solve linear system (only MHD process) <<<
        if mpi_rank == 0:
            
            # save coefficients from previous time step
            up_old[:] = up
            
            # build global sparse matrix and vector
            timea    = time.time()
            mat, vec = acc.assemble_step3(Np, b2 + b2_eq)
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
                
            print('linear solver step 3 : ', info, num_iters)
            
            timeb = time.time()
            times_elapsed['update_step3u'] = timeb - timea
            #print('update_step3u : ', timeb - timea)
        
        # broadcast new FEM coefficients
        mpi_comm.Bcast(up    , root=0)
        mpi_comm.Bcast(up_old, root=0)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        # <<<<<<<<<<<<<<<<<<<< update velocities <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        timea = time.time()
        
        b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2 + b2_eq)
        
        if basis_u == 0:
            up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_0form_vec((up + up_old)/2)
        else:
            up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_2form((up + up_old)/2)
       
        pic_pusher.pusher_step3(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, np.zeros(N_0form, dtype=float), up_ten_1, up_ten_2, up_ten_3, basis_u, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz, np.zeros(Np_loc, dtype=float))
       
        timeb = time.time()
        times_elapsed['pusher_step3'] = timeb - timea
        #print('pusher_step3 : ', timeb - timea)
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
        #print('pusher_step4 : ', timeb - timea)
    # ====================================================================================
    #             step 4 (1 : update particles positions ETA)
    # ====================================================================================

    
    
    # ====================================================================================
    #       step 5 (1 : update particle veclocities V, 2 : update particle weights W)
    # ====================================================================================
    if add_PIC == True:
        
        # push particles
        timea = time.time()
        
        b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2 + b2_eq)
        
        pic_pusher.pusher_step5_ana(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
        
        timeb = time.time()
        times_elapsed['pusher_step5'] = timeb - timea
        #print('pusher_step5 : ', timeb - timea)

        # update particle weights in case of delta-f
        if control == True:
            timea = time.time()
            pic_sample.update_weights(particles_loc, Np_loc, w0_loc, s0_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.NbaseN, domain.cx, domain.cy, domain.cz)
            timeb = time.time()
            times_elapsed['control_weights'] = timeb - timea
            #print('control_weights : ', timeb - timea)
    # ====================================================================================
    #       step 5 (1 : update particle veclocities V, 2 : update particle weights W)
    # ====================================================================================
    
    
    
    # ====================================================================================
    #       step 6 (1 : update rh, u and pr from non - Hamiltonian MHD terms)
    # ====================================================================================
    if mpi_rank == 0:
        
        # save energies after Hamiltonian steps
        energies_H['U'][0] = 1/2*up.dot(MHD.A(up))
        energies_H['p'][0] = 1/(eq_MHD.gamma - 1)*sum(p3.flatten())

        timea = time.time()
        
        # save coefficients from previous time step
        up_old[:] = up
        
        # RHS of linear system
        RHS = MHD.RHS6(up, p3, b2)
        
        # solve linear system with conjugate gradient squared method and values from last time step as initial guess
        timea = time.time()
            
        num_iters = 0
        up[:], info = spa.linalg.gmres(MHD.S6, RHS, x0=up, tol=tol6, maxiter=maxiter6, M=MHD.S6_PRE, callback=count_iters)
        print('linear solver step 6 : ', info, num_iters)
        
        timeb = time.time()
        times_elapsed['update_step2u'] = timeb - timea
        
        # update pressure
        p3[:] = p3 + dt*MHD.L((up + up_old)/2)
        
        # update density
        r3[:] = r3 - dt*tensor_space_FEM.D.dot(MHD.MF((up + up_old)/2))

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
        energies['p'][0] = 1/(eq_MHD.gamma - 1)*sum(p3.flatten())

    energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
    mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

    # subtract equilibrium hot ion energy for analaytical mappings and full-f
    if domain.kind_map >= 10:
        energies['f'] += (control - 1)*equ_PIC.eh_eq(domain.kind_map, domain.params_map)

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
    

    
# ========================== time integration ================================
if params['time_int'] == True:
    
    # a new simulation
    if params['restart'] == False:
    
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
            
            file.create_dataset('energies/bulk_kinetic_H'         , (1,), maxshape=(None,), dtype=float, chunks=True)
            file.create_dataset('energies/bulk_internal_H'        , (1,), maxshape=(None,), dtype=float, chunks=True)

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


            # ==================== save initial data =======================
            file['time'][0]                       = 0.

            file['energies/bulk_kinetic'][0]      = energies['U'][0]
            file['energies/magnetic'][0]          = energies['B'][0]
            file['energies/bulk_internal'][0]     = energies['p'][0]
            file['energies/energetic'][0]         = energies['f'][0]
            
            file['energies/bulk_kinetic_H'][0]    = energies_H['U'][0]
            file['energies/bulk_internal_H'][0]   = energies_H['p'][0]
            
            file['pressure'][0]                   = tensor_space_FEM.extract_3form(p3)
            file['density'][0]                    = tensor_space_FEM.extract_3form(r3)
            
            if basis_u == 0:
                up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_0form_vec(up)
            else:
                up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_2form(up)
            
            file['velocity_field/1_component'][0] = up_ten_1
            file['velocity_field/2_component'][0] = up_ten_2
            file['velocity_field/3_component'][0] = up_ten_3
            
            b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2)
            file['magnetic_field/1_component'][0] = b2_ten_1
            file['magnetic_field/2_component'][0] = b2_ten_2
            file['magnetic_field/3_component'][0] = b2_ten_3
            
            file['magnetic_field/divergence'][0]  = tensor_space_FEM.extract_3form(tensor_space_FEM.D.dot(b2))
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
        print('start time integration! (total number of time steps : ' + str(int(Tend/dt)) + ')')
    # ========================================================================================
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
            
            file['energies/bulk_kinetic_H'].resize(file['energies/bulk_kinetic_H'].shape[0] + 1, axis = 0)
            file['energies/bulk_internal_H'].resize(file['energies/bulk_internal_H'].shape[0] + 1, axis = 0)
            file['energies/bulk_kinetic_H'][-1]  = energies_H['U'][0]
            file['energies/bulk_internal_H'][-1] = energies_H['p'][0]

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
            file['pressure'][-1] = tensor_space_FEM.extract_3form(p3)
            
            file['density'].resize(file['density'].shape[0] + 1, axis = 0)
            file['density'][-1] = tensor_space_FEM.extract_3form(r3)
            
            b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2)
            file['magnetic_field/1_component'].resize(file['magnetic_field/1_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/2_component'].resize(file['magnetic_field/2_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/3_component'].resize(file['magnetic_field/3_component'].shape[0] + 1, axis = 0)
            file['magnetic_field/1_component'][-1] = b2_ten_1
            file['magnetic_field/2_component'][-1] = b2_ten_2
            file['magnetic_field/3_component'][-1] = b2_ten_3
            
            if basis_u == 0:
                up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_0form_vec(up)
            else:
                up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_2form(up)

            file['velocity_field/1_component'].resize(file['velocity_field/1_component'].shape[0] + 1, axis = 0)
            file['velocity_field/2_component'].resize(file['velocity_field/2_component'].shape[0] + 1, axis = 0)
            file['velocity_field/3_component'].resize(file['velocity_field/3_component'].shape[0] + 1, axis = 0)
            file['velocity_field/1_component'][-1] = up_ten_1
            file['velocity_field/2_component'][-1] = up_ten_2
            file['velocity_field/3_component'][-1] = up_ten_3
            
            # other diagnostics
            file['magnetic_field/divergence'].resize(file['magnetic_field/divergence'].shape[0] + 1, axis = 0)
            file['magnetic_field/divergence'][-1] = tensor_space_FEM.extract_3form(tensor_space_FEM.D.dot(b2))

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
