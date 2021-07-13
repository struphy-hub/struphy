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
import hylife.geometry.domain_3d     as dom
import hylife.geometry.polar_splines as pol_spl

import hylife.utilitis_FEEC.spline_space as spl

import hylife.utilitis_FEEC.projectors.projectors_global       as pro
import hylife.utilitis_FEEC.projectors.mhd_operators_3d_global as mhd

import hylife.utilitis_PIC.sobol_seq    as sobol
import hylife.utilitis_PIC.pusher       as pic_pusher
import hylife.utilitis_PIC.accumulation as pic_accumu
import hylife.utilitis_PIC.sampling     as pic_sample

import hylife.dispersion_relations.MHD_eigenvalues_2D as eig_2d

# load local input files
import input_run.equilibrium_PIC        as equ_PIC
import input_run.equilibrium_MHD        as equ_MHD
import input_run.initial_conditions_MHD as init_MHD

# ======================== load parameters =================================================================================
identifier = 'sed_replace_run_dir'   

with open('parameters_sed_replace_run_dir.yml') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)


# mesh generation
Nel            = params['Nel']
spl_kind       = params['spl_kind']
p              = params['p']
nq_el          = params['nq_el']
nq_pr          = params['nq_pr']

# boundary conditions in eta1 direction
bc             = params['bc']

# representation of MHD bulk velocity
basis_u        = params['basis_u']

# time integration
time_int       = params['time_int']
dt             = params['dt']
Tend           = params['Tend']
max_time       = params['max_time']

# polar splines in poloidal plane?
polar          = params['polar']

# geometry
geometry       = params['geometry']
params_map     = params['params_map']

Nel_MAP        = params['Nel_MAP']
spl_kind_MAP   = params['spl_kind_MAP']
p_MAP          = params['p_MAP']

# general
add_step_6     = params['add_step_6']
loc_jeq        = params['loc_jeq']
gamma          = params['gamma']

# preconditioners for linear systems
PRE            = params['PRE']

tol_inv        = params['tol_inv']

drop_tol_A     = params['drop_tol_A']
fill_fac_A     = params['fill_fac_A']

drop_tol_S2    = params['drop_tol_S2']
fill_fac_S2    = params['fill_fac_S2']

drop_tol_S6    = params['drop_tol_S6']
fill_fac_S6    = params['fill_fac_S6']

# tolerances for iterative linear solvers
tol1           = params['tol1']
tol2           = params['tol2']
tol3           = params['tol3']
tol6           = params['tol6']

maxiter1       = params['maxiter1']
maxiter2       = params['maxiter2']
maxiter3       = params['maxiter3']
maxiter6       = params['maxiter6']

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

# energies after Hamiltonian steps
energies_H   = {'U' : np.empty(1, dtype=float), 'p' : np.empty(1, dtype=float)}

# snapshots of distribution function via particle binning
n_bins       = {'eta1_vx' : [32, 64]}
bin_edges    = {'eta1_vx' : [np.linspace(0., 1., n_bins['eta1_vx'][0] + 1), np.linspace(0., 5., n_bins['eta1_vx'][1] + 1)]}
dbin         = {'eta1_vx' : [bin_edges['eta1_vx'][0][1] - bin_edges['eta1_vx'][0][0], bin_edges['eta1_vx'][1][1] - bin_edges['eta1_vx'][1][0]]}
                
fh_loc       = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
fh           = {'eta1_vx' : np.empty((n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float)}
# ========================================================================


# ================== FEM space ==========================================
# 1d B-spline spline spaces for finite elements
spaces_FEM = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]

# 2d tensor-product B-spline space for polar splines (if used)
tensor_space_pol = spl.tensor_spline_space(spaces_FEM[:2])

# 3d tensor-product B-spline space for finite elements
tensor_space_FEM = spl.tensor_spline_space(spaces_FEM)
# =======================================================================


# ========= geometry in case of spline mapping ==========================
if geometry == 'spline' or polar == True:

    # ========= 3D discrete mapping ===========
    #X = lambda eta1, eta2, eta3 : 1*eta1*np.cos(2*np.pi*eta2)
    #Y = lambda eta1, eta2, eta3 : 1*eta1*np.sin(2*np.pi*eta2)
    #Z = lambda eta1, eta2, eta3 : params_map[0]*eta3
                               
    #cx, cy, cz = dom.interp_mapping(Nel_MAP, p_MAP, spl_kind_MAP, X, Y, Z)
    
    # ========= 2D discrete mapping ===========
    X = lambda eta1, eta2 : 1*eta1*np.cos(2*np.pi*eta2) + 10.0
    Y = lambda eta1, eta2 : 1*eta1*np.sin(2*np.pi*eta2)
    
    cx, cy = dom.interp_mapping(Nel_MAP[:2], p_MAP[:2], spl_kind_MAP[:2], X, Y)
    
    # create domain object
    domain = dom.domain(geometry, params_map, Nel_MAP[:2], p_MAP[:2], spl_kind_MAP[:2], cx, cy)

    # create polar splines in poloidal plane
    if polar == True:
        tensor_space_pol.set_polar_splines(domain.cx[:, :, 0], domain.cy[:, :, 0])
          
else:
    
    # create domain object
    domain = dom.domain(geometry, params_map)
    
    
    
# for plotting
#import matplotlib.pyplot as plt
    
etaplot = [np.linspace(0., 1., 100), np.linspace(0., 1., 100), np.linspace(0., 1., 10)]

xplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'x')
yplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'y')
zplot = domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'z')

E1, E2, E3 = np.meshgrid(etaplot[0], etaplot[1], etaplot[2], indexing='ij')

# load MHD equilibrium
eq_MHD = equ_MHD.equilibrium_mhd(domain)
# =======================================================================


#plt.scatter(domain.cx[:, :, 0].flatten(), domain.cy[:, :, 0].flatten())
#plt.axis('square')
#plt.show()
#sys.exit()

#plt.plot(etaplot[0], eq_MHD.q(etaplot[0]))
#plt.plot(etaplot[0], eq_MHD.r0_eq(etaplot[0], etaplot[1])[:, 50])
#plt.show()
#sys.exit()

# ======= set extraction operators and discrete derivatives ============= 
#tensor_space_FEM.set_extraction_operators(bc, polar_splines)
tensor_space_FEM.set_derivatives()
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
    
# create 3d projector
pro_3d = pro.projectors_global_3d(tensor_space_FEM, nq_pr)
# =======================================================================


# ===== initialization with 2d eigenfunctions ===========================
if False:
    U2_ini = eig_2d.solve_ev_problem_FEEC_2D([Nel[:2], p[:2], spl_kind[:2], nq_el[:2], nq_pr[:2], bc], domain, eq_MHD, gamma, -1, 1, 1482)
    
    print(np.sqrt(U2_ini[3])*10)

    up[:]  = pro_3d.pi_2(U2_ini, include_bc=False, eval_kind='tensor_product', interp=True)

    
    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], tensor_space_FEM.evaluate_NDD(etaplot[0], etaplot[1], np.array([0.03]), up)[:, :, 0], levels=50, cmap='jet')

    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], U2_ini[0](etaplot[0], etaplot[1], np.array([0.03]))[:, :, 0], levels=50, cmap='jet')

    #plt.axis('square')
    #plt.colorbar()
    #plt.show()

    #sys.exit()
# ========================================================================



# ==== initialization with projection of input functions ===============
if True:
    ini_MHD = init_MHD.initial_mhd(domain)
    
    r3[:] = pro_3d.pi_3(ini_MHD.r3_ini, include_bc=False, eval_kind='tensor_product', interp=True)
    p3[:] = pro_3d.pi_3(ini_MHD.p3_ini, include_bc=False, eval_kind='tensor_product', interp=True)

    b2[:] = pro_3d.pi_2([ini_MHD.b2_ini_1, ini_MHD.b2_ini_2, ini_MHD.b2_ini_3], include_bc=False, eval_kind='tensor_product', interp=True)

    if   basis_u == 0:

        up_1  = pro_3d.pi_0(ini_MHD.u_ini_1, include_bc=False, eval_kind='tensor_product', interp=True)
        up_2  = pro_3d.pi_0(ini_MHD.u_ini_2, include_bc=True , eval_kind='tensor_product', interp=True)
        up_3  = pro_3d.pi_0(ini_MHD.u_ini_3, include_bc=True , eval_kind='tensor_product', interp=True)

        up[:] = np.concatenate((up_1, up_2, up_3))

    elif basis_u == 2:    

        up[:] = pro_3d.pi_2([ini_MHD.u2_ini_1, ini_MHD.u2_ini_2, ini_MHD.u2_ini_3], include_bc=False, eval_kind='tensor_product', interp=True)


    #plt.contourf(xplot[:, :, 0], yplot[:, :, 0], tensor_space_FEM.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], up)[:, :, 0], levels=50, cmap='jet')
    #plt.colorbar()
    #plt.show()
    #
    #plt.plot(zplot[10, 10, :], tensor_space_FEM.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], up)[10, 10, :])
    #plt.show()
    #
    #print(abs(tensor_space_FEM.D.dot(up)).max())
    #

    #plt.plot(etaplot[2], tensor_space_FEM.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], b2)[10, 10, :])
    #plt.show()
    #sys.exit()
        
        
# ===== initialization with white noise on periodic domain =============
if False:
    np.random.seed(1607)

    p3_temp = np.empty(N_3form   , dtype=float)

    u1_temp = np.empty(N_0form   , dtype=float)
    u2_temp = np.empty(N_0form   , dtype=float)
    u3_temp = np.empty(N_0form   , dtype=float)

    b1_temp = np.empty(N_2form[0], dtype=float)
    b2_temp = np.empty(N_2form[1], dtype=float)
    b3_temp = np.empty(N_2form[2], dtype=float)

    r3_temp = np.empty(N_3form   , dtype=float)

    plane = 'yz'

    # spectrum in xy-plane
    if plane == 'xy':
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
    if plane == 'yz':
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
    if plane == 'xz':
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
    
    #p3[:] = np.random.rand(p3.size)
    #up[:] = np.random.rand(up.size)
    #b2[:] = 0.
    #r3[:] = 0.

print('projection of initial conditions and equilibrium done!')
# ========================================================================



# ================= mass matrices ========================================
if mpi_rank == 0: 
    
    # assemble mass matrices in space of discrete 2-forms (V2) and discrete 3-forms (V3)
    tensor_space_FEM.assemble_M2(domain)
    tensor_space_FEM.assemble_M3(domain)

    if basis_u == 0:
        # mass matrix in space of discrete contravariant vector fields using the V0 basis (NNN)
        tensor_space_FEM.assemble_Mv(domain, basis_u)
        
    print('assembly of mass matrices done!')
# ========================================================================
    

# ================== linear MHD operators ================================
# projection of input equilibrium
r3_eq = pro_3d.pi_3(eq_MHD.r3_eq, include_bc=True, eval_kind='tensor_product', interp=True)
r0_eq = pro_3d.pi_0(eq_MHD.r0_eq, include_bc=True, eval_kind='tensor_product', interp=True)

p3_eq = pro_3d.pi_3(eq_MHD.p3_eq, include_bc=True, eval_kind='tensor_product', interp=True)
p0_eq = pro_3d.pi_0(eq_MHD.p0_eq, include_bc=True, eval_kind='tensor_product', interp=True)

b2_eq = pro_3d.pi_2([eq_MHD.b2_eq_1, eq_MHD.b2_eq_2, eq_MHD.b2_eq_3], include_bc=True, eval_kind='tensor_product', interp=True)
j2_eq = pro_3d.pi_2([eq_MHD.j2_eq_1, eq_MHD.j2_eq_2, eq_MHD.j2_eq_3], include_bc=True, eval_kind='tensor_product', interp=True)

if mpi_rank == 0:  
    # interface for semi-discrete linear MHD operators
    MHD = mhd.operators_mhd(pro_3d, dt, gamma, loc_jeq, basis_u)
    
    # assemble right-hand sides of projection matrices
    MHD.assemble_rhs_EF(domain     , [eq_MHD.b2_eq_1, eq_MHD.b2_eq_2, eq_MHD.b2_eq_3])
    MHD.assemble_rhs_F( domain, 'm',  eq_MHD.r3_eq)
    MHD.assemble_rhs_F( domain, 'p',  eq_MHD.p3_eq)
    MHD.assemble_rhs_PR(domain     ,  eq_MHD.p3_eq)
    
    if basis_u == 0:
        MHD.assemble_rhs_W(domain, eq_MHD.r3_eq)
        MHD.assemble_rhs_F(domain, 'j')
        
    print('assembly of MHD projection operators done!')
        
    # assemble mass matrix weighted with 0-form density
    MHD.assemble_MR(domain, eq_MHD.r3_eq)    
    
    # assemble mass matrix weighted with Jeq x
    MHD.assemble_JB_strong(domain, [eq_MHD.j2_eq_1, eq_MHD.j2_eq_2, eq_MHD.j2_eq_3])
    
    # create liner MHD operators as scipy.sparse.linalg.LinearOperator
    MHD.setOperators()
    
    print('assembly of all MHD operators done!')
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

#b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2 + b2_eq)
#up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_2form(up)

#print(particles_loc[:, 0:4])

#pic_pusher.pusher_step3(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, np.zeros(N_0form, dtype=float), up_ten_1, up_ten_2, up_ten_3, basis_u, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz, np.zeros(Np_loc, dtype=float))

#pic_pusher.pusher_step4(particles_loc, dt, Np_loc, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

#pic_pusher.pusher_step5(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)

#print(particles_loc[:, 0:4])

#acc.accumulate_step3(particles_loc, b2 + b2_eq, mpi_comm)
#mat, vec   = acc.assemble_step3(Np, b2 + b2_eq)

#print(vec)
#sys.exit()



# ================ compute initial energies and distribution function ===========
# initial energies
if mpi_rank == 0:
    energies['U'][0] = 1/2*up.dot(MHD.A(up))
    energies['B'][0] = 1/2*b2.dot(tensor_space_FEM.M2.dot(b2))
    energies['p'][0] = 1/(gamma - 1)*sum(p3.flatten())
    
    energies_H['U'][0] = energies['U'][0]
    energies_H['p'][0] = energies['p'][0]

energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

# subtract equilibrium hot ion energy for analaytical mappings and full-f
if geometry != 'spline':
    energies['f'] += (control - 1)*equ_PIC.eh_eq(domain.kind_map, domain.params_map)
    
# initial distribution function
fh_loc['eta1_vx'][:, :] = np.histogram2d(particles_loc[0], particles_loc[3], bins=bin_edges['eta1_vx'], weights=particles_loc[6], normed=False)[0]/(Np*dbin['eta1_vx'][0]*dbin['eta1_vx'][1])
mpi_comm.Reduce(fh_loc['eta1_vx'], fh['eta1_vx'], op=MPI.SUM, root=0)

print('initial diagnostics done')
# ===============================================================================


# =============== preconditioners for time integration ==========================
if mpi_rank == 0:
    
    # assemble approximate inverse interpolation/histopolation matrices
    if PRE == 'ILU':
        pro_3d.assemble_approx_inv(tol_inv)
    
    timea = time.time()
    MHD.setPreconditionerA(domain, PRE, drop_tol_A, fill_fac_A)
    timeb = time.time()
    print('Preconditioner for A done!', timeb - timea)
    
    timea = time.time()
    MHD.setPreconditionerS2(domain, PRE, drop_tol_S2, fill_fac_S2)
    timeb = time.time()
    print('Preconditioner for S2 done!', timeb - timea)
    
    if add_step_6 == True:
        timea = time.time()
        MHD.setPreconditionerS6(domain, PRE, drop_tol_S6, fill_fac_S6)
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
        
        
        # <<<<<<<<< set up and solve linear system (only MHD process) <<<<<<<<<<<<<<
        if mpi_rank == 0:
            
            # build global sparse matrix 
            timea = time.time()
            mat   = acc.assemble_step1(Np, b2 + b2_eq)
            timeb = time.time()
            times_elapsed['control_step1'] = timeb - timea
            
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
        up[:], info = spa.linalg.gmres(MHD.S2, RHS, x0=up, tol=tol2, maxiter=maxiter2, M=MHD.S2_PRE, callback=count_iters)
        print('linear solver step 2 : ', info, num_iters)
        
        timeb = time.time()
        times_elapsed['update_step2u'] = timeb - timea
        
        # update magnetic field (strong)
        timea = time.time()
        b2[:] = b2 - dt*tensor_space_FEM.C.dot(MHD.EF((up + up_old)/2))
        timeb = time.time()
        times_elapsed['update_step2b'] = timeb - timea
    
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
            
            # RHS of linear system
            RHS = MHD.A(up) - dt**2/4*mat.dot(up) + dt*vec
            
            # LHS of linear system
            LHS = spa.linalg.LinearOperator(MHD.A.shape, lambda x : MHD.A(x) + dt**2/4*mat.dot(x))
            
            # solve linear system with conjugate gradient method (A + dt**2*mat/4 is a symmetric positive definite matrix) with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess
            timea = time.time()
            
            num_iters = 0
            up[:], info = spa.linalg.cg(LHS, RHS, x0=up, tol=tol3, maxiter=maxiter3, M=MHD.A_PRE, callback=count_iters)
            print('linear solver step 3 : ', info, num_iters)
            
            timeb = time.time()
            times_elapsed['update_step3u'] = timeb - timea
        
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
        
        b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2 + b2_eq)
        
        #pic_pusher.pusher_step5(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
        pic_pusher.pusher_step5_ana(particles_loc, dt, tensor_space_FEM.T[0], tensor_space_FEM.T[1], tensor_space_FEM.T[2], p, Nel, NbaseN, NbaseD, Np_loc, b2_ten_1, b2_ten_2, b2_ten_3, domain.kind_map, domain.params_map, domain.T[0], domain.T[1], domain.T[2], domain.p, domain.Nel, domain.NbaseN, domain.cx, domain.cy, domain.cz)
        
            
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
    if add_step_6 == True and mpi_rank == 0:
        
        # save energies after Hamiltonian steps
        energies_H['U'][0] = 1/2*up.dot(MHD.A(up))
        energies_H['p'][0] = 1/(gamma - 1)*sum(p3.flatten())

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
        energies['p'][0] = 1/(gamma - 1)*sum(p3.flatten())

    energies_loc['f'][0] = particles_loc[6].dot(particles_loc[3]**2 + particles_loc[4]**2 + particles_loc[5]**2)/(2*Np)
    mpi_comm.Reduce(energies_loc['f'], energies['f'], op=MPI.SUM, root=0)

    # subtract equilibrium hot ion energy for analaytical mappings and full-f
    if geometry != 'spline':
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

            file.create_dataset('restart/pressure', sh_3_ini, maxshape=sh_3_max, dtype=float, chunks=True)
            file.create_dataset('restart/density' , sh_3_ini, maxshape=sh_3_max, dtype=float, chunks=True)
            
            if basis_u == 0:
                file.create_dataset('restart/velocity_field/1_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
                file.create_dataset('restart/velocity_field/2_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
                file.create_dataset('restart/velocity_field/3_component', sh_0_ini , maxshape=sh_0_max , dtype=float, chunks=True)
            else:
                file.create_dataset('restart/velocity_field/1_component', sh_21_ini, maxshape=sh_21_max, dtype=float, chunks=True)
                file.create_dataset('restart/velocity_field/2_component', sh_22_ini, maxshape=sh_22_max, dtype=float, chunks=True)
                file.create_dataset('restart/velocity_field/3_component', sh_23_ini, maxshape=sh_23_max, dtype=float, chunks=True)
            
            file.create_dataset('restart/magnetic_field/1_component', sh_21_ini, maxshape=sh_21_max, dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/2_component', sh_22_ini, maxshape=sh_22_max, dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/3_component', sh_23_ini, maxshape=sh_23_max, dtype=float, chunks=True)
            

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
            time_steps_done   = file['restart/time_steps_done'][num_restart]

            p3[:]       = file['restart/pressure'][num_restart].flatten()
            r3[:]       = file['restart/density'][num_restart].flatten()
            
            b2_ten_1    = file['restart/magnetic_field/1_component'][num_restart]
            b2_ten_2    = file['restart/magnetic_field/2_component'][num_restart]
            b2_ten_3    = file['restart/magnetic_field/3_component'][num_restart]
            b2[:]       = np.concatenate((b2_ten_1.flatten(), b2_ten_2.flatten(), b2_ten_3.flatten()))
        
            up_ten_1    = file['restart/velocity_field/1_component'][num_restart]
            up_ten_2    = file['restart/velocity_field/2_component'][num_restart]
            up_ten_3    = file['restart/velocity_field/3_component'][num_restart]
            up[:]       = np.concatenate((up_ten_1.flatten(), up_ten_2.flatten(), up_ten_3.flatten()))
 
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
            if create_restart:
                if mpi_rank == 0:
                    
                    file['restart/time_steps_done'][-1] = time_steps_done
                    
                    file['restart/pressure'][-1] = tensor_space_FEM.extract_3form(p3)
                    file['restart/density'][-1]  = tensor_space_FEM.extract_3form(r3)
                    
                    b2_ten_1, b2_ten_2, b2_ten_3 = tensor_space_FEM.extract_2form(b2)
                    file['restart/magnetic_field/1_component'][-1] = b2_ten_1
                    file['restart/magnetic_field/2_component'][-1] = b2_ten_2
                    file['restart/magnetic_field/3_component'][-1] = b2_ten_3
                    
                    if basis_u == 0:
                        up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_0form_vec(up)
                    else:
                        up_ten_1, up_ten_2, up_ten_3 = tensor_space_FEM.extract_2form(up)
                    
                    file['restart/velocity_field/1_component'][-1] = up_ten_1
                    file['restart/velocity_field/2_component'][-1] = up_ten_2
                    file['restart/velocity_field/3_component'][-1] = up_ten_3
                    
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
