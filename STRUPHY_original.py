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
import hylife.geometry.pull_push_analytical as push
import hylife.geometry.mappings_analytical  as map_ana

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
import input_run.equilibrium_MHD as eq_MHD


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

# boundary conditions
bc_u1          = params['bc_u1']
bc_b1          = params['bc_b1']

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

drop_tol_S6    = params['drop_tol_S6']
fill_fac_S6    = params['fill_fac_S6']

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
Fy = lambda eta1, eta2, eta3 : 1*eta2
Fz = lambda eta1, eta2, eta3 : 1*eta3

pro_F = proj.projectors_local_3d(tensor_space_F, [p_F[0] + 1, p_F[1] + 1, p_F[2] + 1])

cx = pro_F.pi_0(Fx)
cy = pro_F.pi_0(Fy)
cz = pro_F.pi_0(Fz)

del pro_F
# =======================================================================



# ========= for plotting ================================================
import matplotlib.pyplot as plt
    
etaplot = [np.linspace(0., 1., 100), np.linspace(0., 1., 100), np.linspace(0., 1., 20)]
xplot   = [np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float) for i in range(3)]
    
map_ana.kernel_evaluation_tensor(etaplot[0], etaplot[1], etaplot[2], xplot[0], 1, kind_map, params_map)
map_ana.kernel_evaluation_tensor(etaplot[0], etaplot[1], etaplot[2], xplot[1], 2, kind_map, params_map)
map_ana.kernel_evaluation_tensor(etaplot[0], etaplot[1], etaplot[2], xplot[2], 3, kind_map, params_map)
# =======================================================================


"""
B2_2 = np.empty(100, dtype=float)
B2_3 = np.empty(100, dtype=float)
for i in range(100):
    B2_2[i] = eq_MHD.b2_eq_2(etaplot[0][i], 0.5, 0.5, kind_map, params_map)
    B2_3[i] = eq_MHD.b2_eq_3(etaplot[0][i], 0.5, 0.5, kind_map, params_map)
    
plt.plot(etaplot[0], B2_2)
plt.plot(etaplot[0], B2_3)
plt.show()
sys.exit()
"""


# ======= particle accumulator (and delta-f corrections) ================
if add_PIC == True:

    # particle accumulator (all processes)
    acc = pic_accumu.accumulation(tensor_space)
    
    # delta-f corrections (only MHD process)
    if control == True and mpi_rank == 0:
        cont = cv.terms_control_variate(tensor_space, acc, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
# =======================================================================




# ======= reserve memory for FEM cofficients (all MPI processes) ========
pr     = np.empty(Nbase_3form,    dtype=float)     # bulk pressure FEM coefficients

u1     = np.empty(Nbase_0form,    dtype=float)     # bulk velocity FEM coefficients (1 - component)
u2     = np.empty(Nbase_0form,    dtype=float)     # bulk velocity FEM coefficients (2 - component)
u3     = np.empty(Nbase_0form,    dtype=float)     # bulk velocity FEM coefficients (3 - component)

u1_old = np.empty(Nbase_0form,    dtype=float)     # bulk velocity FEM coefficients from previous time step (1 - component)
u2_old = np.empty(Nbase_0form,    dtype=float)     # bulk velocity FEM coefficients from previous time step (2 - component)
u3_old = np.empty(Nbase_0form,    dtype=float)     # bulk velocity FEM coefficients from previous time step (3 - component)

b1     = np.empty(Nbase_2form[0], dtype=float)     # magnetic field FEM coefficients (1 - component)
b2     = np.empty(Nbase_2form[1], dtype=float)     # magnetic field FEM coefficients (2 - component)
b3     = np.empty(Nbase_2form[2], dtype=float)     # magnetic field FEM coefficients (3 - component)

rh     = np.empty(Nbase_3form,    dtype=float)     # bulk mass density FEM coefficients
# =======================================================================


# ==== reserve memory for implicit particle-coupling sub-steps ==========
mat11_loc = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat12_loc = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat13_loc = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat22_loc = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat23_loc = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
mat33_loc = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)

vec1_loc  = np.empty((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)
vec2_loc  = np.empty((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)
vec3_loc  = np.empty((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)

if mpi_rank == 0:
    mat11 = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat12 = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat13 = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat22 = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat23 = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)
    mat33 = np.empty((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), dtype=float)

    vec1  = np.empty((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)
    vec2  = np.empty((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)
    vec3  = np.empty((NbaseN[0], NbaseN[1], NbaseN[2]), dtype=float)

else:
    mat11, mat12, mat13, mat22, mat23, mat33 = None, None, None, None, None, None
    vec1,  vec2,  vec3                       = None, None, None
# =======================================================================



if mpi_rank == 0:
    # ============= projection of initial conditions ==========================
    
    # create object for projecting initial conditions
    pro = proj.projectors_local_3d(tensor_space, nq_pr)
    
    if   mapping == 0:

        pr[:, :, :]                           = pro.pi_3( None,              0,  1,        kind_map, params_map)
        
        u1[:, :, :]                           = pro.pi_0( None,              0,  2,        kind_map, params_map)
        u2[:, :, :]                           = pro.pi_0( None,              0,  3,        kind_map, params_map)
        u3[:, :, :]                           = pro.pi_0( None,              0,  4,        kind_map, params_map)
        
        b1[:, :, :], b2[:, :, :], b3[:, :, :] = pro.pi_2([None, None, None], 0, [5, 6, 7], kind_map, params_map)
        rh[:, :, :]                           = pro.pi_3( None,              0,  8,        kind_map, params_map)

    elif mapping == 1:
        
        pr[:, :, :]                           = pro.pi_3( None,              1,  1,        T_F, p_F, NbaseN_F, [cx, cy, cz])
        
        u1[:, :, :]                           = pro.pi_0( None,              1,  2,        T_F, p_F, NbaseN_F, [cx, cy, cz])
        u2[:, :, :]                           = pro.pi_0( None,              1,  3,        T_F, p_F, NbaseN_F, [cx, cy, cz])
        u3[:, :, :]                           = pro.pi_0( None,              1,  4,        T_F, p_F, NbaseN_F, [cx, cy, cz])
        
        b1[:, :, :], b2[:, :, :], b3[:, :, :] = pro.pi_2([None, None, None], 1, [5, 6, 7], T_F, p_F, NbaseN_F, [cx, cy, cz])
        rh[:, :, :]                           = pro.pi_3( None,              1,  8,        T_F, p_F, NbaseN_F, [cx, cy, cz])
    
    del pro
    
    
    # apply boundary conditions
    if bc[0] == False:
        
        if bc_u1[0] == 'dirichlet':
            u1[0]  = 0.
        
        if bc_u1[1] == 'dirichlet':
            u1[-1] = 0.
            
        if bc_b1[0] == 'dirichlet':
            b1[0]  = 0.
        
        if bc_b1[1] == 'dirichlet':
            b1[-1] = 0.
    
    """
    U2_plot = tensor_space.evaluate_NNN(etaplot[0], etaplot[1], etaplot[2], u2)
    
    plt.contourf(xplot[0][:, :, 0], xplot[1][:, :, 0], U2_plot[:, :, 0], levels=50, cmap='jet')
    plt.colorbar()
    plt.axis('square')
    
    #plt.plot(etaplot[0], U1_plot[:, 25, 0])
    plt.show()
    
    
    sys.exit()
    """
    
    
    """
    B1_plot = tensor_space.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], b1)
    B2_plot = tensor_space.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], b2)
    B3_plot = tensor_space.evaluate_DDN(etaplot[0], etaplot[1], etaplot[2], b3)
    
    B_plot  = [np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float) for i in range(3)]
    
    push.kernel_evaluation_vector(B1_plot, B2_plot, B3_plot, etaplot[0], etaplot[1], etaplot[2], 16, kind_map, params_map, B_plot[0])
    push.kernel_evaluation_vector(B1_plot, B2_plot, B3_plot, etaplot[0], etaplot[1], etaplot[2], 17, kind_map, params_map, B_plot[1])
    push.kernel_evaluation_vector(B1_plot, B2_plot, B3_plot, etaplot[0], etaplot[1], etaplot[2], 18, kind_map, params_map, B_plot[2])
    
    plt.plot(xplot[0][:, 10, 1], B_plot[1][:, 10, 1])
    plt.plot(xplot[0][:, 10, 1], 1e-3*np.sin(xplot[0][:, 10, 1]*0.8), 'k--')
    plt.show()
    sys.exit()
    """
    
    """
    np.random.seed(1234)
    amps = np.random.rand(8, pr.shape[0], pr.shape[1])

    for k in range(pr.shape[2]):
        pr[:, :, k] = amps[0]

        u1[:, :, k] = 2.5e-12*amps[1]
        u2[:, :, k] = 2.5e-12*amps[2]
        u3[:, :, k] = 1e-10*amps[3]

        b1[:, :, :] = 0.
        b2[:, :, :] = 0.
        b3[:, :, k] = 2e-2*amps[6]

        rh[:, :, k] = amps[7]
    """
    
    
    print('projection of initial conditions done!')
    # ==========================================================================


    # ==================== matrices ============================================
    # mass matrices in V2 and V3
    M2 = mass.mass_V2(tensor_space, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
    M3 = mass.mass_V3(tensor_space, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
    Mv = mass.mass_V0_vector(tensor_space, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
    
    print('mass matrices done!')
    
    
    # discrete grad, curl and div matrices
    derivatives = der.discrete_derivatives(tensor_space)

    GRAD = derivatives.grad_3d()
    CURL = derivatives.curl_3d()
    DIV  = derivatives.div_3d()
    
    del derivatives

    print('discrete derivatives done!')
    
    
    # projection matrices
    MHD = mhd.projectors_local_mhd(tensor_space, nq_pr)
    
    Q   = MHD.projection_Q(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_2[rho3_eq * lambda^0]
    W   = MHD.projection_W(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_0[rho0_eq * lambda^0]
    TAU = MHD.projection_T(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_1[b2_eq   x lambda^0]
    S   = MHD.projection_S(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_2[p3_eq   * lambda^0]
    K   = MHD.projection_K(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_3[p0_eq   * lambda^3]  
    N   = MHD.projection_N(mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)     # pi_2[g_sqrt  * lambda^0]
    
    del MHD
    print('projection matrices done!')
    
    """
    test1, test2, test3 = np.split(TAU.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))), [Ntot_1form[0], Ntot_1form[0] + Ntot_1form[1]])
    
    test1 = test1.reshape(Nbase_1form[0])
    test2 = test2.reshape(Nbase_1form[1])
    test3 = test3.reshape(Nbase_1form[2])
    
    #print(np.abs(test1).max())
    #print(np.abs(test2).max())
    #print(np.abs(test3).max())
    print(test2[0], test2[-1])
    print(test3[0], test3[-1])
    
    BxU_plot1 = tensor_space.evaluate_DNN(etaplot[0], etaplot[1], etaplot[2], test1)
    BxU_plot2 = tensor_space.evaluate_NDN(etaplot[0], etaplot[1], etaplot[2], test2)
    BxU_plot3 = tensor_space.evaluate_NND(etaplot[0], etaplot[1], etaplot[2], test3)
    
    plt.contourf(xplot[0][:, :, 15], xplot[1][:, :, 15], BxU_plot3[:, :, 15], levels=50, cmap='jet')
    plt.colorbar()
    plt.axis('square')
    plt.show()
    
    sys.exit()
    """
    
    """
    A1 = np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float)
    A2 = np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float)
    A3 = np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float)
    
    for i in range(100):
        for j in range(100):
            for k in range(20):
                A1[i, j, k] = eq_MHD.curlb_eq_1(etaplot[0][i], etaplot[1][j], etaplot[2][k], kind_map, params_map)
                A2[i, j, k] = eq_MHD.curlb_eq_2(etaplot[0][i], etaplot[1][j], etaplot[2][k], kind_map, params_map)
                A3[i, j, k] = eq_MHD.curlb_eq_3(etaplot[0][i], etaplot[1][j], etaplot[2][k], kind_map, params_map)
    
    
    plt.contourf(xplot[0][:, :, 15], xplot[1][:, :, 15], A3[:, :, 15], levels=50, cmap='jet')
    plt.colorbar()
    plt.axis('square')
    plt.show()
    
    sys.exit()
    """
    
    # curl Beq term
    curl_beq = mhd.term_curl_beq(tensor_space, mapping, kind_map, params_map, tensor_space_F, cx, cy, cz)
    
    """
    rhs = curl_beq.inner_curl_beq(b1, b2, b3)
    test1, test2, test3 = np.split(spa.linalg.spsolve(Mv, rhs), [Ntot_0form, 2*Ntot_0form])
    
    test1 = test1.reshape(Nbase_0form)
    test2 = test2.reshape(Nbase_0form)
    test3 = test3.reshape(Nbase_0form)
    
    curlb1_plot = tensor_space.evaluate_NNN(etaplot[0], etaplot[1], etaplot[2], test1)
    curlb2_plot = tensor_space.evaluate_NNN(etaplot[0], etaplot[1], etaplot[2], test2)
    curlb3_plot = tensor_space.evaluate_NNN(etaplot[0], etaplot[1], etaplot[2], test3)
    
    plt.contourf(xplot[0][:, :, 15], xplot[1][:, :, 15], curlb2_plot[:, :, 15], levels=50, cmap='jet')
    plt.colorbar()
    plt.axis('square')
    plt.show()
    
    sys.exit()
    """
    # ==========================================================================

    
    # ========= compute symmetric matrix A and a ILU preconditioner ============
    A     = 1/2*(W.T.dot(Mv) + Mv.dot(W)).tocsc()
    print('A done')
    
    A_ILU = spa.linalg.spilu(A, drop_tol=drop_tol_A , fill_factor=fill_fac_A)
    print('A_ILU done')
    
    A_PRE = spa.linalg.LinearOperator(A.shape, lambda x : A_ILU.solve(x))
    
    A     = A.tocsr()
    # ==========================================================================


    # ================== matrices and preconditioner for step 2 ================
    S2      = (A + dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tolil()
    
    # apply boundary conditions
    if bc[0] == False:
        
        # eta1 = 0
        if bc_u1[0] == 'dirichlet':
            S2[:Nbase_0form[1]*Nbase_0form[2], :] = 0.
            S2[:, :Nbase_0form[1]*Nbase_0form[2]] = 0.
            S2[:Nbase_0form[1]*Nbase_0form[2], :Nbase_0form[1]*Nbase_0form[2]] = np.identity(Nbase_0form[1]*Nbase_0form[2])
    
        # eta1 = 1
        if bc_u1[1] == 'dirichlet':
            S2[(Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form, :] = 0.
            S2[:, (Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form] = 0.
            S2[(Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form, (Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form] = np.identity(Nbase_0form[1]*Nbase_0form[2])
    
    print('S2 done')
    
    STEP2_1 = (A - dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tocsr()
    print('STEP2_1 done')
    
    STEP2_2 = dt*TAU.T.dot(CURL.T.dot(M2)).tocsr()
    print('STEP2_2 done')

    # incomplete LU decomposition for preconditioning
    S2_ILU  = spa.linalg.spilu(S2.tocsc(), drop_tol=drop_tol_S2, fill_factor=fill_fac_S2)
    print('S2_ILU done')
    
    S2_PRE  = spa.linalg.LinearOperator(S2.shape, lambda x : S2_ILU.solve(x))
    
    S2      = S2.tocsr()
    # ===========================================================================

    
    # ================== matrices and preconditioner for step 6 =================
    if add_pressure == True:
    
        L       = -DIV.dot(S) - (gamma - 1)*K.dot(DIV).dot(N)
        print('L done')

        del S, K
        
        S6      = (A - dt**2/4*N.T.dot(DIV.T).dot(M3).dot(L)).tolil()
        
        # apply boundary conditions
        if bc[0] == False:
            
            # eta1 = 0
            if bc_u1[0] == 'dirichlet':
                S6[:Nbase_0form[1]*Nbase_0form[2], :] = 0.
                S6[:, :Nbase_0form[1]*Nbase_0form[2]] = 0.
                S6[:Nbase_0form[1]*Nbase_0form[2], :Nbase_0form[1]*Nbase_0form[2]] = np.identity(Nbase_0form[1]*Nbase_0form[2])
            
            # eta1 = 1
            f bc_u1[1] == 'dirichlet':
                S6[(Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form, :] = 0.
                S6[:, (Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form] = 0.
                S6[(Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form, (Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form] = np.identity(Nbase_0form[1]*Nbase_0form[2])
        
        print('S6 done')
        
        STEP6_1 = (A + dt**2/4*N.T.dot(DIV.T).dot(M3).dot(L)).tocsr()
        print('STEP6_1 done')
        
        STEP6_2 = dt*N.T.dot(DIV.T).dot(M3).tocsr()
        print('STEP6_2 done')

        # incomplete LU decomposition for preconditioning
        S6_ILU  = spa.linalg.spilu(S6.tocsc(), drop_tol=drop_tol_S6, fill_factor=fill_fac_S6)
        print('S6_ILU done')

        S6_PRE  = spa.linalg.LinearOperator(S6.shape, lambda x : S6_ILU.solve(x))
        
        S6      = S6.tocsr()
    # ==========================================================================
    print('assembly of constant matrices done!')
    

    
if mpi_rank == 0:
    timea = time.time()
    A_PRE(np.random.rand(A.shape[0]))
    timeb = time.time()
    print('Time for evaluation of preconditioner for A : ', timeb - timea)
    
    timea = time.time()
    rhs = STEP2_1.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) + STEP2_2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())))
    
    temp1, temp2, temp3 = np.split(spa.linalg.cg(S2, rhs, x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol2, M=S2_PRE)[0], [Ntot_0form, 2*Ntot_0form])
    timeb = time.time()
    print('Time for solving linear system in step 2 : ', timeb - timea)
    
if mpi_rank == 0 and add_pressure == True:
    timea = time.time()
    rhs = STEP6_1.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) + STEP6_2.dot(pr.flatten()) + dt*curl_beq.inner_curl_beq(b1, b2, b3)
    rhs[(Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form] = 0.
    
    temp1, temp2, temp3 = np.split(spa.linalg.cgs(S6, rhs, x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol6, M=S6_PRE)[0], [Ntot_0form, 2*Ntot_0form])
    timeb = time.time()
    print('Time for solving linear system in step 6 : ', timeb - timea)
    
    
if control == True and mpi_rank == 0:
    timea = time.time()
    mat   = cont.mass_V2_nh_eq(b1, b2, b3)
    timeb = time.time()
    print('Time for control step 1 : ', timeb - timea)

    timea = time.time()
    vec   = cont.inner_prod_V2_jh_eq(b1, b2, b3)
    timeb = time.time()
    print('Time for control step 3 : ', timeb - timea)


    
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


"""
timea = time.time()
pic_accumu_ker.kernel_step3(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, kind_map, params_map, mat11_loc, mat12_loc, mat13_loc, mat22_loc, mat23_loc, mat33_loc, vec1_loc, vec2_loc, vec3_loc)
timeb = time.time()
print(timeb - timea)

vec = np.concatenate((vec1_loc.flatten(), vec2_loc.flatten(), vec3_loc.flatten()))/Np

temp1, temp2, temp3 = np.split(spa.linalg.spsolve(Mv, vec), [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]]) 


Bxj_1 = tensor_space.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], temp1.reshape(Nbase_2form[0]))
Bxj_2 = tensor_space.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], temp2.reshape(Nbase_2form[1]))
Bxj_3 = tensor_space.evaluate_DDN(etaplot[0], etaplot[1], etaplot[2], temp3.reshape(Nbase_2form[2]))

print(np.abs(Bxj_1).max())
print(np.abs(Bxj_2).max())
print(np.abs(Bxj_3).max())
#sys.exit()
    
Bxj_plot  = [np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float) for i in range(3)]

push.kernel_evaluation_vector(Bxj_1, Bxj_2, Bxj_3, etaplot[0], etaplot[1], etaplot[2], 16, kind_map, params_map, Bxj_plot[0])
push.kernel_evaluation_vector(Bxj_1, Bxj_2, Bxj_3, etaplot[0], etaplot[1], etaplot[2], 17, kind_map, params_map, Bxj_plot[1])
push.kernel_evaluation_vector(Bxj_1, Bxj_2, Bxj_3, etaplot[0], etaplot[1], etaplot[2], 18, kind_map, params_map, Bxj_plot[2])

plt.plot(xplot[0][:, 10, 1], Bxj_plot[1][:, 10, 1])
plt.plot(xplot[0][:, 10, 1], 1e-3*np.sin(xplot[0][:, 10, 1]*0.8)*0.05*2.5, 'k--')
plt.show()
"""

"""
#timea = time.time()
#pic_accumu_ker.kernel_step1(particles_loc, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, kind_map, params_map, mat12_loc, mat13_loc, mat23_loc)
#timeb = time.time()
#print(timeb - timea)

#mat = -acc.to_sparse_step1(mat12_loc, mat13_loc, mat23_loc)/Np
mat = -cont.mass_V2_nh_eq(b1, b2, b3)

temp1, temp2, temp3 = np.split(spa.linalg.spsolve(Mv, mat.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())))), [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]]) 

UxB_1 = tensor_space.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], temp1.reshape(Nbase_2form[0]))
UxB_2 = tensor_space.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], temp2.reshape(Nbase_2form[1]))
UxB_3 = tensor_space.evaluate_DDN(etaplot[0], etaplot[1], etaplot[2], temp3.reshape(Nbase_2form[2]))

UxB_plot  = [np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float) for i in range(3)]

push.kernel_evaluation_vector(UxB_1, UxB_2, UxB_3, etaplot[0], etaplot[1], etaplot[2], 16, kind_map, params_map, UxB_plot[0])
push.kernel_evaluation_vector(UxB_1, UxB_2, UxB_3, etaplot[0], etaplot[1], etaplot[2], 17, kind_map, params_map, UxB_plot[1])
push.kernel_evaluation_vector(UxB_1, UxB_2, UxB_3, etaplot[0], etaplot[1], etaplot[2], 18, kind_map, params_map, UxB_plot[2])

plt.plot(xplot[0][:, 10, 1], UxB_plot[0][:, 10, 1])
plt.plot(xplot[0][:, 10, 1], 0.05*1e-3*np.sin(xplot[0][:, 10, 1]*0.8)*1e-3*np.sin(xplot[0][:, 10, 1]*0.8), 'k--')
#plt.plot(xplot[0][:, 10, 1], -0.05*1e-3*np.sin(xplot[0][:, 10, 1]*0.8), 'k--')
plt.show()
"""

"""
vec =  cont.inner_prod_V2_jh_eq(b1, b2, b3) 
mat = -cont.mass_V2_nh_eq(b1, b2, b3)

temp1, temp2, temp3 = np.split(spa.linalg.spsolve(Mv, mat.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())))), [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]]) 

#temp1, temp2, temp3 = np.split(spa.linalg.spsolve(Mv, vec), [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]]) 

UxB_1 = tensor_space.evaluate_NDD(etaplot[0], etaplot[1], etaplot[2], temp1.reshape(Nbase_2form[0]))
UxB_2 = tensor_space.evaluate_DND(etaplot[0], etaplot[1], etaplot[2], temp2.reshape(Nbase_2form[1]))
UxB_3 = tensor_space.evaluate_DDN(etaplot[0], etaplot[1], etaplot[2], temp3.reshape(Nbase_2form[2]))

UxB_plot  = [np.empty((etaplot[0].size, etaplot[1].size, etaplot[2].size), dtype=float) for i in range(3)]

push.kernel_evaluation_vector(UxB_1, UxB_2, UxB_3, etaplot[0], etaplot[1], etaplot[2], 16, kind_map, params_map, UxB_plot[0])
push.kernel_evaluation_vector(UxB_1, UxB_2, UxB_3, etaplot[0], etaplot[1], etaplot[2], 17, kind_map, params_map, UxB_plot[1])
push.kernel_evaluation_vector(UxB_1, UxB_2, UxB_3, etaplot[0], etaplot[1], etaplot[2], 18, kind_map, params_map, UxB_plot[2])

plt.plot(xplot[0][:, 10, 1], UxB_plot[0][:, 10, 1])
#plt.plot(xplot[0][:, 10, 1], 0.05*2.5*1e-3*np.sin(xplot[0][:, 10, 1]*0.8), 'k--')
plt.plot(xplot[0][:, 10, 1], 0.05*1e-3*np.sin(xplot[0][:, 10, 1]*0.8)*1e-3*np.sin(xplot[0][:, 10, 1]*0.8), 'k--')
#plt.plot(xplot[0][:, 10, 1], -0.05*1e-3*np.sin(xplot[0][:, 10, 1]*0.8), 'k--')
plt.show()
sys.exit()
"""




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
    energies['p'][0] = 1/(gamma - 1)*sum(pr.flatten())

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
        
        # <<<<<<<<<<<<<<<<<<<<< charge accumulation <<<<<<<<<<<<<<<<<
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
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        # <<<<<<<<< set up and solve linear system <<<<<<<<<<<<<<<<<<<
        if mpi_rank == 0:
            
            # build global sparse matrix
            mat = -acc.to_sparse_step1(mat12, mat13, mat23)/Np

            # delta-f correction
            if control == True:
                timea = time.time()
                mat  -= cont.mass_V2_nh_eq(b1, b2, b3)
                timeb = time.time()
                times_elapsed['control_step1'] = timeb - timea
        
            # solve linear system with conjugate gradient squared method with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess 
            timea = time.time()
            temp1, temp2, temp3 = np.split(spa.linalg.cgs((A - dt*mat/2).tocsr(), (A + dt*mat/2).dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))), x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol1, M=A_PRE)[0], [Ntot_0form, 2*Ntot_0form])         
            timeb = time.time()
            times_elapsed['update_step1u'] = timeb - timea

            u1[:, :, :] = temp1.reshape(Nbase_0form)
            u2[:, :, :] = temp2.reshape(Nbase_0form)
            u3[:, :, :] = temp3.reshape(Nbase_0form)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
            
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
        
        # compute right-hand side of linear system
        rhs = STEP2_1.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) + STEP2_2.dot(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())))
          
        # apply boundary conditions
        if bc[0] == False:
            if bc_u1[0] == 'dirichlet':
                rhs[:Nbase_0form[1]*Nbase_0form[2]] = 0.
            if bc_u1[1] == 'dirichlet':
                rhs[(Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form] = 0.
        
        
        # solve linear system with conjugate gradient method (S2 is a symmetric positive definite matrix) with an incomplete LU decomposition as preconditioner and values from last time step as initial guess
        timea = time.time()
        temp1, temp2, temp3 = np.split(spa.linalg.cg(S2, rhs, x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol2, M=S2_PRE)[0], [Ntot_0form, 2*Ntot_0form])
        timeb = time.time()
        times_elapsed['update_step2u'] = timeb - timea

        u1[:, :, :] = temp1.reshape(Nbase_0form)
        u2[:, :, :] = temp2.reshape(Nbase_0form)
        u3[:, :, :] = temp3.reshape(Nbase_0form)
        
        
        # apply boundary conditions
        if bc[0] == False:
            if bc_u1[0] == 'dirichlet':
                u1[0]  = 0.
            if bc_u1[1] == 'dirichlet':
                u1[-1] = 0.
        
        
        # update magnetic field
        timea = time.time()
        temp1, temp2, temp3 = np.split(np.concatenate((b1.flatten(), b2.flatten(), b3.flatten())) - dt/2*CURL.dot(TAU.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())) + np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten())))), [Ntot_2form[0], Ntot_2form[0] + Ntot_2form[1]]) 
        timeb = time.time()
        times_elapsed['update_step2b'] = timeb - timea

        b1[:, :, :] = temp1.reshape(Nbase_2form[0])
        b2[:, :, :] = temp2.reshape(Nbase_2form[1])
        b3[:, :, :] = temp3.reshape(Nbase_2form[2])
        
        
        # apply boundary conditions
        if bc[0] == False:
            if bc_b1[0] == 'dirichlet':
                b1[0]  = 0.
            if bc_b1[1] == 'dirichlet':
                b1[-1] = 0.
    
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
        
        # <<<<<<<<<<<<<<<<<< current accumulation <<<<<<<<<<<<<<<<<<<<<<<
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
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        # <<<<<<<<<<<<<< set up and solve linear system <<<<<<<<<<<<<<<<<
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
                vec   += cont.inner_prod_V2_jh_eq(b1, b2, b3)      
                timeb  = time.time()
                times_elapsed['control_step3'] = timeb - timea

            
            # solve linear system with conjugate gradient method (A + dt**2*mat/4 is a symmetric positive definite matrix) with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess
            timea = time.time()
            temp1, temp2, temp3 = np.split(spa.linalg.cg((A + dt**2*mat/4).tocsr(), (A - dt**2*mat/4).dot(np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten()))) + dt*vec, x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol3, M=A_PRE)[0], [Ntot_0form, 2*Ntot_0form])
            timeb = time.time()
            times_elapsed['update_step3u'] = timeb - timea

            u1[:, :, :] = temp1.reshape(Nbase_0form)
            u2[:, :, :] = temp2.reshape(Nbase_0form)
            u3[:, :, :] = temp3.reshape(Nbase_0form)

        
        # broadcast new FEM coefficients
        mpi_comm.Bcast(u1    , root=0)
        mpi_comm.Bcast(u2    , root=0)
        mpi_comm.Bcast(u3    , root=0)
        
        mpi_comm.Bcast(u1_old, root=0)
        mpi_comm.Bcast(u2_old, root=0)
        mpi_comm.Bcast(u3_old, root=0)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        # <<<<<<<<<<<<<<<<<<<< update velocities <<<<<<<<<<<<<<<<<<<<<<<
        timea = time.time()
        
        if mapping == 0:
            pic_pusher.pusher_step3(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, (u1 + u1_old)/2, (u2 + u2_old)/2, (u3 + u3_old)/2, kind_map, params_map)
        elif mapping == 1:
            pic_pusher.pusher_step3(particles_loc, dt, T[0], T[1], T[2], p, Nel, NbaseN, NbaseD, Np_loc, b1, b2, b3, (u1 + u1_old)/2, (u2 + u2_old)/2, (u3 + u3_old)/2, T_F[0], T_F[1], T_F[2], p_F, Nel_F, NbaseN_F, cx, cy, cz)
        
        timeb = time.time()
        times_elapsed['pusher_step3'] = timeb - timea
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
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
        
        # save coefficients from previous time step
        u1_old[:, :, :] = u1[:, :, :]
        u2_old[:, :, :] = u2[:, :, :]
        u3_old[:, :, :] = u3[:, :, :]
        
        # compute right-hand side of linear system
        rhs = STEP6_1.dot(np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))) + STEP6_2.dot(pr.flatten()) + dt*curl_beq.inner_curl_beq(b1, b2, b3)
        
        # apply boundary conditions
        if bc[0] == False:
            if bc_u1[0] == 'dirichlet':
                rhs[:Nbase_0form[1]*Nbase_0form[2]] = 0.
            if bc_u1[1] == 'dirichlet':
                rhs[(Nbase_0form[0] - 1)*Nbase_0form[1]*Nbase_0form[2]:Ntot_0form] = 0.

        # solve linear system of u^(n+1) with conjugate gradient squared method with an incomplete LU decomposition of A as preconditioner and values from last time step as initial guess
        temp1, temp2, temp3 = np.split(spa.linalg.cgs(S6, rhs, x0=np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())), tol=tol6, M=S6_PRE)[0], [Ntot_0form, 2*Ntot_0form])
        
        # update velocity
        u1[:, :, :] = temp1.reshape(Nbase_0form)
        u2[:, :, :] = temp2.reshape(Nbase_0form)
        u3[:, :, :] = temp3.reshape(Nbase_0form)
        
        # apply boundary conditions
        if bc[0] == False:
            if bc_u1[0] == 'dirichlet':
                u1[0]  = 0.
            if bc_u1[1] == 'dirichlet':
                u1[-1] = 0.
        
        # update pressure
        pr[:, :, :] = pr + dt/2*(L.dot(np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten())) + np.concatenate((u1.flatten(), u2.flatten(), u3.flatten())))).reshape(Nbase_3form)
        
        # update density
        rh[:, :, :] = rh - dt/2*(DIV.dot(Q.dot(np.concatenate((u1_old.flatten(), u2_old.flatten(), u3_old.flatten())) + np.concatenate((u1.flatten(), u2.flatten(), u3.flatten()))))).reshape(Nbase_3form)

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
        energies['p'][0] = 1/(gamma - 1)*sum(pr.flatten())

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
            file.create_dataset('time', (1,), maxshape=(None,), dtype=float, chunks=True)
            
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
            file.create_dataset('pressure',                   (1, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]), maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]), dtype=float, chunks=True)
            file.create_dataset('velocity_field/1_component', (1, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]), maxshape=(None, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]), dtype=float, chunks=True)
            file.create_dataset('velocity_field/2_component', (1, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]), maxshape=(None, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]), dtype=float, chunks=True)
            file.create_dataset('velocity_field/3_component', (1, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]), maxshape=(None, Nbase_0form[0],    Nbase_0form[1],    Nbase_0form[2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/1_component', (1, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), maxshape=(None, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/2_component', (1, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), maxshape=(None, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), dtype=float, chunks=True)
            file.create_dataset('magnetic_field/3_component', (1, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), maxshape=(None, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), dtype=float, chunks=True)
            file.create_dataset('density',                    (1, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]), maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    dtype=float, chunks=True)

            # particles
            file.create_dataset('particles', (1, 7, Np), maxshape=(None, 7, Np), dtype=float, chunks=True)
            
            # other diagnostics
            file.create_dataset('bulk_mass', (1,), maxshape=(None,), dtype=float, chunks=True)

            file.create_dataset('magnetic_field/divergence',  (1, Nbase_3form[0], Nbase_3form[1], Nbase_3form[2]), maxshape=(None, Nbase_3form[0], Nbase_3form[1], Nbase_3form[2]), dtype=float, chunks=True)

            file.create_dataset('distribution_function/eta1_vx', (1, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), maxshape=(None, n_bins['eta1_vx'][0], n_bins['eta1_vx'][1]), dtype=float, chunks=True)

            
            # datasets for restart function
            file.create_dataset('restart/time_steps_done', (1,), maxshape=(None,), dtype=int, chunks=True)

            file.create_dataset('restart/pressure',                   (1, Nbase_3form[0],    Nbase_3form[1], Nbase_3form[2]), maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/1_component', (1, Nbase_0form[0],    Nbase_0form[1], Nbase_0form[2]), maxshape=(None, Nbase_0form[0], Nbase_0form[1], Nbase_0form[2]), dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/2_component', (1, Nbase_0form[0],    Nbase_0form[1], Nbase_0form[2]), maxshape=(None, Nbase_0form[0], Nbase_0form[1], Nbase_0form[2]), dtype=float, chunks=True)
            file.create_dataset('restart/velocity_field/3_component', (1, Nbase_0form[0],    Nbase_0form[1], Nbase_0form[2]), maxshape=(None, Nbase_0form[0], Nbase_0form[1], Nbase_0form[2]), dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/1_component', (1, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), maxshape=(None, Nbase_2form[0][0], Nbase_2form[0][1], Nbase_2form[0][2]), dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/2_component', (1, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), maxshape=(None, Nbase_2form[1][0], Nbase_2form[1][1], Nbase_2form[1][2]), dtype=float, chunks=True)
            file.create_dataset('restart/magnetic_field/3_component', (1, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), maxshape=(None, Nbase_2form[2][0], Nbase_2form[2][1], Nbase_2form[2][2]), dtype=float, chunks=True)
            file.create_dataset('restart/density',                    (1, Nbase_3form[0],    Nbase_3form[1], Nbase_3form[2]), maxshape=(None, Nbase_3form[0],    Nbase_3form[1],    Nbase_3form[2]),    dtype=float, chunks=True)

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

            file['velocity_field/1_component'].resize(file['velocity_field/1_component'].shape[0] + 1, axis = 0)
            file['velocity_field/2_component'].resize(file['velocity_field/2_component'].shape[0] + 1, axis = 0)
            file['velocity_field/3_component'].resize(file['velocity_field/3_component'].shape[0] + 1, axis = 0)
            file['velocity_field/1_component'][-1] = u1
            file['velocity_field/2_component'][-1] = u2
            file['velocity_field/3_component'][-1] = u3

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
