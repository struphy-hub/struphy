import time
start_simulation = time.time()

import os

import numpy             as np
import matplotlib.pyplot as plt
import scipy.sparse      as sparse

import utilitis_FEEC.bsplines       as bsp
import utilitis_FEEC.projectors_mhd as mhd
import utilitis_FEEC.projectors     as proj
import utilitis_FEEC.derivatives    as der
import utilitis_FEEC.evaluation     as eva

import utilitis_FEEC.mass_matrices  as mass
import utilitis_FEEC.mappings       as maps

import utilitis_PIC.STRUPHY_fields       as pic_fields
import utilitis_PIC.STRUPHY_pusher       as pic_pusher
import utilitis_PIC.STRUPHY_accumulation as pic_accumu
import utilitis_PIC.STRUPHY_sampling     as pic_sample

import utilitis_PIC.sobol_seq as sobol
import scipy.special as sp



# ========================================== parameters ==============================================================
Nel       = [128, 3, 3]               # mesh generation on logical domain
bc        = [True, True, True]        # boundary conditions (True: periodic, False: else)
p         = [3, 2, 2]                 # spline degrees  
L         = [20., 1., 1.]             # box lengthes of physical domain


el_b      = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                      # element boundaries
delta     = [1/Nel for Nel in Nel]                                             # element sizes
T         = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]  # knot vectors (for N functions)
t         = [T[1:-1] for T in T]                                               # reduced knot vectors (for D function)
Nbase0    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]                 # number of basis functions (N functions)
Nbase_old = [Nel + p for Nel, p, bc in zip(Nel, p, bc)]                        # TODO delete this later
Ntot      =  Nbase0[0]*Nbase0[1]*Nbase0[2]                                     # total number of basis functions


time_int  = True     # do time integration?
dt        = 0.05     # time step
Tend      = 20.      # simulation time
max_time  = 60*60    # maximum runtime of program in minutes


# geometry (slab geometry)
DF        = np.array([[  L[0], 0., 0.], [0.,   L[1], 0.], [0., 0.,   L[2]]])           # Jacobian matrix
DFinv     = np.array([[1/L[0], 0., 0.], [0., 1/L[1], 0.], [0., 0., 1/L[2]]])           # inverse Jacobian matrix

G         = np.array([[  L[0]**2, 0., 0.], [0.,   L[1]**2, 0.], [0., 0.,   L[2]**2]])  # metric tensor
Ginv      = np.array([[1/L[0]**2, 0., 0.], [0., 1/L[1]**2, 0.], [0., 0., 1/L[2]**2]])  # inverse metric tensor

g_sqrt    = L[0]*L[1]*L[2]                                                             # Jacobian determinant

mapping   = maps.mappings(['slab', L[0], L[1], L[2]])                                  # object for mappings in MHD part


# particle parameters
Np        = 100                 # total number of particles
vth       = 1.                  # thermal velocity of particles in all directions

v0x       = 2.5                 # mean velocity of hot ions in x-direction (must be compatible with backgound field)
v0y       = 0.                  # mean velocity of hot ions in y-direction (must be compatible with backgound field)
v0z       = 0.                  # mean velocity of hot ions in z-direction (must be compatible with backgound field)

nuh       = 0.0                 # ratio of hot/bulk equlibrium number densities 

control   = 0                   # control variate? (0: no, 1: yes)

# particle loading
loading   = 'pseudo-random'     # 'pseudo-random': particles[:, :6] = np.random.rand(Np, 6)
                                # 'sobol_standard': particles[:, :6] = sobol.i4_sobol_generate(6, Np, 1000)
                                # 'sobol_antithetic': sobol.i4_sobol_generate(6, int(Np/64), 1000) --> 64 symmetric particles
                                # 'pr_space_uni_velocity': pseudo-random in space, uniform in velocity space
                                # 'external': particles[:, :6] = np.load('name_of_file.npy')
            
        
add_pressure = True             # add pressure terms to simulation?
                           
            
            
# name and directory of output data file
#identifier  = 'STRUPHY_Nel=20_p=3_L=2pidk_dt=0.04_Np=1e5_vth=1.0_v0=2.5_nuh=0.05_k=0.75_amp=1e-4_CV=on_x_sobol_ref'
identifier  = 'STRUPHY_fullMHD_angle=45'
dir_results = 'results/'



# Is this run a restart? If True, locate restart files
restart = False  

name_particles = 'restart_files/' + identifier + '_restart_files/' + identifier + '_restart=particles1.npy'
name_rho_coeff = 'restart_files/' + identifier + '_restart_files/' + identifier + '_restart=rho_coeff1.npy'
name_u_coeff   = 'restart_files/' + identifier + '_restart_files/' + identifier + '_restart=u_coeff1.npy'
name_b_coeff   = 'restart_files/' + identifier + '_restart_files/' + identifier + '_restart=b_coeff1.npy'
name_p_coeff   = 'restart_files/' + identifier + '_restart_files/' + identifier + '_restart=p_coeff1.npy'
name_control   = 'restart_files/' + identifier + '_restart_files/' + identifier + '_restart=CV1.npy'
name_time_step = 'restart_files/' + identifier + '_restart_files/' + identifier + '_restart=time1.npy'


# Create restart files at the end of the simulation? If True, name full directory where to save them
create_restart = False
dir_restart    = '/home/florian/Desktop/PHD/02_Projekte/hylife/restart_files/' + identifier + '_restart_files/'
# =====================================================================================================================



# ===================== coefficients for pp-forms in interval [0, delta] (N and D) ====================================
pp0 = []
pp1 = []

for i in range(3):
    if p[i] == 3:
        pp0.append(np.asfortranarray([[1/6, -1/(2*delta[i]), 1/(2*delta[i]**2), -1/(6*delta[i]**3)], [2/3, 0., -1/delta[i]**2, 1/(2*delta[i]**3)], [1/6, 1/(2*delta[i]), 1/(2*delta[i]**2), -1/(2*delta[i]**3)], [0., 0., 0., 1/(6*delta[i]**3)]]))
        pp1.append(np.asfortranarray([[1/2, -1/delta[i], 1/(2*delta[i]**2)], [1/2, 1/delta[i], -1/delta[i]**2], [0., 0., 1/(2*delta[i]**2)]])/delta[i])
    elif p[i] == 2:
        pp0.append(np.asfortranarray([[1/2, -1/delta[i], 1/(2*delta[i]**2)], [1/2, 1/delta[i], -1/delta[i]**2], [0., 0., 1/(2*delta[i]**2)]]))
        pp1.append(np.asfortranarray([[1., -1/delta[i]], [0., 1/delta[i]]])/delta[i])
    else:
        print('So far only cubic and quadratic splines implemented!')
# =====================================================================================================================




# ====================================== background quantities ========================================================
Ueq_phys   = np.array([0., 0., 0.])     # background bulk flow (vector/1-form on physical domain)
Ueq        = DF.T.dot(Ueq_phys)         # background bulk flow (1-form on logical domain)


Beq_phys   = np.array([1., 1., 0.])     # background magnetic field (vector/2-form on physical domain)
Beq        = g_sqrt*DFinv.dot(Beq_phys) # background magnetic field (2-form on logical domain)

B0_23      = lambda q1, q2, q3 : mapping.g_sqrt(q1, q2, q3) * mapping.DFinv[0][0](q1, q2, q3) * (1.)
B0_31      = lambda q1, q2, q3 : mapping.g_sqrt(q1, q2, q3) * mapping.DFinv[1][1](q1, q2, q3) * (1.)   
B0_12      = lambda q1, q2, q3 : mapping.g_sqrt(q1, q2, q3) * mapping.DFinv[2][2](q1, q2, q3) * (0.)   

B0_hat     = [B0_23, B0_31, B0_12]

rhoeq_phys = 1.                         # background bulk mass density (scalar/3-from on physical domain)
peq_phys   = 1.                         # background bulk pressure (scalar/0-form on physical domain)


rho0_123   = lambda q1, q2, q3 : mapping.g_sqrt(q1, q2, q3) * (rhoeq_phys) # background bulk mass density (3-form on logical domain)
peq        = lambda q1, q2, q3 : np.ones(q1.shape) * (peq_phys)            # background bulk pressure (0-form on logical domain)

gamma      = 5/3                        # adiabatic exponent
# =====================================================================================================================



# ============================================== initial conditions ===================================================
kx     = 0.75  # wavenumber of initial perturbation in x - direction
ky     = 0.    # wavenumber of initial perturbation in y - direction
kz     = 0.    # wavenumber of initial perturbation in z - direction

amp    = 1e-4  # amplitude  of initial perturbation

'''
B1_ini = lambda q1, q2, q3 : mapping.g_sqrt(q1, q2, q3) * mapping.DFinv[0][0](q1, q2, q3) * (0. * q1)
B2_ini = lambda q1, q2, q3 : mapping.g_sqrt(q1, q2, q3) * mapping.DFinv[1][1](q1, q2, q3) * (0. * q1)
B3_ini = lambda q1, q2, q3 : mapping.g_sqrt(q1, q2, q3) * mapping.DFinv[2][2](q1, q2, q3) * (amp * np.sin(kx * q1 * L[0] + ky * q2 * L[1]))

U1_ini = lambda q1, q2, q3 : mapping.DF[0][0](q1, q2, q3) * (0. * q1)  # actually DF.T !!
U2_ini = lambda q1, q2, q3 : mapping.DF[1][1](q1, q2, q3) * (0. * q1)  # actually DF.T !!
U3_ini = lambda q1, q2, q3 : mapping.DF[2][2](q1, q2, q3) * (0. * q1)  # actually DF.T !!
'''


Nmodes = 128
modes  = np.linspace(0, Nmodes, Nmodes + 1) - Nmodes/2
modes  = np.delete(modes, int(Nmodes/2))
amps   = np.random.rand(8, Nmodes)



def U1_ini(q1, q2, q3):
    
    values = np.zeros(q1.shape)
    
    for i in range(Nmodes):
        values += amps[0, i]*np.sin(2*np.pi*modes[i]*q1)
        
    return values

def U2_ini(q1, q2, q3):
    
    values = np.zeros(q2.shape)
    
    for i in range(Nmodes):
        values += amps[1, i]*np.sin(2*np.pi*modes[i]*q1)
        
    return values

def U3_ini(q1, q2, q3):
    
    values = np.zeros(q3.shape)
    
    for i in range(Nmodes):
        values += amps[2, i]*np.sin(2*np.pi*modes[i]*q1)
        
    return values


def B1_ini(q1, q2, q3):
    
    values = np.zeros(q2.shape)
    
    for i in range(Nmodes):
        values += 0*np.sin(2*np.pi*modes[i]*q1)
        
    return values

def B2_ini(q1, q2, q3):
    
    values = np.zeros(q1.shape)
    
    for i in range(Nmodes):
        values += amps[4, i]*np.sin(2*np.pi*modes[i]*q1)
        
    return values

def B3_ini(q1, q2, q3):
    
    values = np.zeros(q1.shape)
    
    for i in range(Nmodes):
        values += amps[5, i]*np.sin(2*np.pi*modes[i]*q1)
        
    return values


def rho_ini(q1, q2, q3):
    
    values = np.zeros(q1.shape)
    
    for i in range(Nmodes):
        values += amps[6, i]*np.sin(2*np.pi*modes[i]*q1)
        
    return values


def p_ini(q1, q2, q3):
    
    values = 0.
    
    for i in range(Nmodes):
        values += amps[7, i]*np.sin(2*np.pi*modes[i]*q1)
        
    return values



nh0_phys = rhoeq_phys*nuh                                    # hot ion number density on physical domain
nh0_123  = nh0_phys*g_sqrt                                   # hot ion number density on logical domain
Eh_eq    = nh0_123/2*(v0x**2 + v0y**2 + v0z**2 + 3*vth**2/2) # hot ion equilibrium energy

# initial hot ion distribution function (3-form on logical domain)
fh0             = lambda q1, q2, q3, vx, vy, vz : nh0_123/((np.pi)**(3/2)*vth**3)*np.exp(-(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2)

# control variate
control_variate = lambda q1, q2, q3, vx, vy, vz : nh0_123/((np.pi)**(3/2)*vth**3)*np.exp(-(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2)

# initial sampling distribution
g_sampling      = lambda q1, q2, q3, vx, vy, vz :       1/((np.pi)**(3/2)*vth**3)*np.exp(-(vx - v0x)**2/vth**2 - (vy - v0y)**2/vth**2 - (vz - v0z)**2/vth**2)
# =====================================================================================================================



# ========================================= reserve memory for unknowns ===============================================
rho   = np.empty(1*Ntot, dtype=float)   # density  FEM coefficients

u     = np.empty(3*Ntot, dtype=float)   # U-field  FEM coefficients
u_old = np.empty(3*Ntot, dtype=float)   # U-field  FEM coefficients from previous time step (needed in integration step 3)

b     = np.empty(3*Ntot, dtype=float)   # B-field  FEM coefficients

pr    = np.empty(1*Ntot, dtype=float)   # pressure FEM coefficients


# matrices and vectors in steps 1 and 3
mat11 = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
mat12 = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
mat13 = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')

mat22 = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
mat23 = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
mat33 = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')

vec1  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
vec2  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
vec3  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')

# particles and knot span indices
particles = np.empty((Np, 7), dtype=float, order='F')
spans0    = np.empty((Np, 3), dtype=int,   order='F')

# fields at particle positions
B_part = np.empty((Np, 3), dtype=float, order='F')
U_part = np.empty((Np, 3), dtype=float, order='F')

# energies (bulk kinetic energy, magnetic energy, bulk internal energy, hot ion kinetic + internal energy (delta f))
energies = np.empty(4, dtype=float)
# =====================================================================================================================


# ======================================== projection of initial conditions ===========================================
# create object for projecting initial conditions
PRO = proj.projectors_3d(p, Nbase_old, T, bc)

# left-hand sides of projectors
PRO.assemble_V0()
PRO.assemble_V1()
PRO.assemble_V2()
PRO.assemble_V3()

# projection of initial conditions
rho[:]                                               = PRO.PI_3(rho_ini)
u[0*Ntot:1*Ntot], u[1*Ntot:2*Ntot], u[2*Ntot:3*Ntot] = PRO.PI_1([U1_ini, U2_ini, U3_ini])
b[0*Ntot:1*Ntot], b[1*Ntot:2*Ntot], b[2*Ntot:3*Ntot] = PRO.PI_2([B1_ini, B2_ini, B3_ini])
pr[:]                                                = PRO.PI_0(p_ini)


print('projection of initial conditions done!')
# =====================================================================================================================


# ============================================ MHD matrices ===========================================================
# create object for projecting MHD matrices
MHD = mhd.projections_mhd(p, Nbase_old, T, bc)

# right-hand side of projection matrices
Q1,   Q2,   Q3   = MHD.projection_Q(rho0_123, mapping.Ginv)
W1,   W2,   W3   = MHD.projection_W(rho0_123, mapping.g_sqrt)
TAU1, TAU2, TAU3 = MHD.projection_T(B0_hat, mapping.Ginv)
S1,   S2,   S3   = MHD.projection_S(peq)
K                = MHD.projection_K(peq)

# mass matrices in V0, V1 and V2
M0 = mass.mass_V0_3d(T, p, bc, mapping.g_sqrt)
M1 = mass.mass_V1_3d(T, p, bc, mapping.Ginv, mapping.g_sqrt)
M2 = mass.mass_V2_3d(T, p, bc, mapping.G, mapping.g_sqrt)

# normalization vector in V0
norm = mass.inner_prod_V0_3d(T, p, bc, mapping.g_sqrt, lambda q1, q2, q3 : np.ones(q1.shape)).flatten()

# discrete grad, curl and div matrices
derivatives = der.discrete_derivatives(T, p, bc)

GRAD = derivatives.GRAD_3d()
CURL = derivatives.CURL_3d()
DIV  = derivatives.DIV_3d()

# Perform projections of Q
Q1   = sparse.linalg.spsolve(PRO.interhistopolation_V2_1, Q1)
Q2   = sparse.linalg.spsolve(PRO.interhistopolation_V2_2, Q2)
Q3   = sparse.linalg.spsolve(PRO.interhistopolation_V2_3, Q3)

Q    = sparse.bmat([[Q1], [Q2], [Q3]], format='csc')

del Q1, Q2, Q3

# perform projections of W1, W2, W3
W1   = sparse.linalg.spsolve(PRO.interhistopolation_V1_1, W1)
W2   = sparse.linalg.spsolve(PRO.interhistopolation_V1_2, W2)
W3   = sparse.linalg.spsolve(PRO.interhistopolation_V1_3, W3)

W    = sparse.bmat([[W1, None, None], [None, W2, None], [None, None, W3]], format='csc')

del W1, W2, W3

# perform projections of TAU1, TAU2, TAU3
TAU1 = sparse.linalg.spsolve(PRO.interhistopolation_V1_1, TAU1)
TAU2 = sparse.linalg.spsolve(PRO.interhistopolation_V1_2, TAU2)
TAU3 = sparse.linalg.spsolve(PRO.interhistopolation_V1_3, TAU3)

TAU  = sparse.bmat([[TAU1], [TAU2], [TAU3]], format='csc')

del TAU1, TAU2, TAU3

# perform projections of S1, S2, S3
S1   = sparse.linalg.spsolve(PRO.interhistopolation_V1_1, S1)
S2   = sparse.linalg.spsolve(PRO.interhistopolation_V1_2, S2)
S3   = sparse.linalg.spsolve(PRO.interhistopolation_V1_3, S3)

S    = sparse.bmat([[S1, None, None], [None, S2, None], [None, None, S3]], format='csc')

del S1, S2, S3

# perform projection of K
K   = sparse.linalg.spsolve(PRO.interhistopolation_V0, K).tocsc()

# compute matrix A
A = 1/2*(M1.dot(W) + W.T.dot(M1))

del W


# LU decompostion of Schur complement in step 2
STEP2_schur_LU = sparse.linalg.splu((A + dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tocsc())

# other matrices needed in step 2
STEP2_1 = (A - dt**2/4*TAU.T.dot(CURL.T.dot(M2.dot(CURL.dot(TAU))))).tocsc()
STEP2_2 = dt*TAU.T.dot(CURL.T.dot(M2)).tocsc()

# matrices for non-Hamiltonian part
MAT = GRAD.T.dot(M1).dot(S) + (gamma - 1)*K.T.dot(GRAD.T).dot(M1)

del S, K

LHS_LU = sparse.linalg.splu((sparse.bmat([[sparse.identity(Ntot),  dt/2*DIV.dot(Q), None], [None, A,  dt/2*M1.dot(GRAD)], [None, -dt/2*MAT, M0]])).tocsc())
RHS    =                     sparse.bmat([[sparse.identity(Ntot), -dt/2*DIV.dot(Q), None], [None, A, -dt/2*M1.dot(GRAD)], [None,  dt/2*MAT, M0]], format='csc')

A = A.toarray()

# delete everything which is not needed to save memory
del PRO, MHD, M0, M1, GRAD, DIV, Q, MAT

print('assembly of constant matrices done!')
# ======================================================================================================================




# ================================================ create particles ====================================================
if   loading == 'pseudo-random':
    # pseudo-random numbers between (0, 1)
    particles[:, :6] = np.random.rand(Np, 6)
    
elif loading == 'sobol_standard':
    # plain sobol numbers between (0, 1) (skip first 1000 numbers)
    particles[:, :6] = sobol.i4_sobol_generate(6, Np, 1000) 
    
elif loading == 'sobol_antithetic':
    # symmetric sobol numbers between (0, 1) (skip first 1000 numbers) in all 6 dimensions
    pic_sample.set_particles_symmetric(sobol.i4_sobol_generate(6, int(Np/64), 1000), particles)  
    
elif loading == 'pr_space_uni_velocity':
    # pseudo-random numbers in space and uniform in velocity space
    particles[:, :3] = np.random.rand(Np, 3)
    
    dv = 1/Np
    particles[:,  3] = np.linspace(dv, 1 - dv, Np)
    particles[:,  4] = np.linspace(dv, 1 - dv, Np)
    particles[:,  5] = np.linspace(dv, 1 - dv, Np)
    
elif loading == 'external':
    # load numbers between (0, 1) from an external file
    particles[:, :6] = np.load('test_particles.npy')
    
else:
    print('particle loading not specified')

# inversion of cumulative distribution function
particles[:, 3]  = sp.erfinv(2*particles[:, 3] - 1)*vth + v0x
particles[:, 4]  = sp.erfinv(2*particles[:, 4] - 1)*vth + v0y
particles[:, 5]  = sp.erfinv(2*particles[:, 5] - 1)*vth + v0z

# compute parameters for control variate and initial weights
g0 = g_sampling(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3], particles[:, 4], particles[:, 5])
w0 = fh0(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3], particles[:, 4], particles[:, 5])/g0

particles[:, 6] = w0 - control*control_variate(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3], particles[:, 4], particles[:, 5])/g0 

# compute initial knot span indices
spans0[:, 0] = np.floor(particles[:, 0]*Nel[0]).astype(int) + p[0]
spans0[:, 1] = np.floor(particles[:, 1]*Nel[1]).astype(int) + p[1]
spans0[:, 2] = np.floor(particles[:, 2]*Nel[2]).astype(int) + p[2]

print('particle initialization done!')
# =====================================================================================================================


# ================ compute initial fields at particle positions and initial energies ==================================
timea = time.time()
pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, np.asfortranarray(b[:Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[Ntot:2*Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[2*Ntot:].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
pic_fields.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, np.asfortranarray(u[:Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(u[Ntot:2*Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(u[2*Ntot:].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), Ueq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part)
timeb = time.time()
print('initial field computation at particles done. Time : ', timeb-timea)


# initial energies
energies[0] = 1/2*u.dot(A.dot(u))
energies[1] = 1/2*b.dot(M2.dot(b))
energies[2] = 1/(gamma - 1)*pr.dot(norm)
energies[3] = 1/2*particles[:, 6].dot(particles[:, 3]**2 + particles[:, 4]**2 + particles[:, 5]**2)/Np + (control - 1)*Eh_eq
# =====================================================================================================================



# =============================================== time integrator =====================================================
def update():
    
    '''
    # step 1 (update u)
    pic_accumu.accumulation_step1(particles, p, spans0, Nbase0, T[0], T[1], T[2], t[0], t[1], t[2], L, B_part, mat12, mat13, mat23)
    
    AJ11A = -np.block([[np.zeros((Ntot, Ntot)), mat12.reshape(Ntot, Ntot), mat13.reshape(Ntot, Ntot)], [-mat12.reshape(Ntot, Ntot).T, np.zeros((Ntot, Ntot)), mat23.reshape(Ntot, Ntot)], [-mat13.reshape(Ntot, Ntot).T, -mat23.reshape(Ntot, Ntot).T, np.zeros((Ntot, Ntot))]])/Np
    
    if control == 1:
        AJ11A -= mass.mass_V1_nh0(T, p, bc, mapping.Ginv, b[0*Ntot:1*Ntot], b[1*Ntot:2*Ntot], b[2*Ntot:3*Ntot], Beq, nh0_123).toarray()
    
    u[:] = np.linalg.solve(A - dt/2*AJ11A, (A + dt/2*AJ11A).dot(u))
    '''
    
    
    # step 2 (update first u, then b and evaluate B-field at particle positions)
    u_old[:] = u
    
    u[:] = STEP2_schur_LU.solve(STEP2_1.dot(u_old) + STEP2_2.dot(b))
    b[:] = b - dt/2*CURL.dot(TAU.dot(u_old + u))
    
    '''
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, np.asfortranarray(b[:Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[Ntot:2*Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[2*Ntot:].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    
    
    # step 3 (update first u, then evaluate U-field at particle positions and then update V)
    pic_accumu.accumulation_step3(particles, p, spans0, Nbase0, T[0], T[1], T[2], t[0], t[1], t[2], L, B_part, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    
    BLOCK = np.block([[mat11.reshape(Ntot, Ntot), mat12.reshape(Ntot, Ntot), mat13.reshape(Ntot, Ntot)], [mat12.reshape(Ntot, Ntot).T, mat22.reshape(Ntot, Ntot), mat23.reshape(Ntot, Ntot)], [mat13.reshape(Ntot, Ntot).T, mat23.reshape(Ntot, Ntot).T, mat33.reshape(Ntot, Ntot)]])/Np
    
    u_old[:] = u
    
    if control == 1:
        CV = mass.inner_prod_V1_jh0(T, p, bc, mapping.Ginv, mapping.DFinv, mapping.g_sqrt, b[0*Ntot:1*Ntot], b[1*Ntot:2*Ntot], b[2*Ntot:3*Ntot], Beq, [nh0_phys*v0x, nh0_phys*v0y, nh0_phys*v0z])
        
        u[:] = np.linalg.solve(A + dt**2/4*BLOCK, (A - dt**2/4*BLOCK).dot(u_old) + dt*np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))/Np + dt*np.concatenate((CV[0].flatten(), CV[1].flatten(), CV[2].flatten())))
    
    
    else:
        u[:] = np.linalg.solve(A + dt**2/4*BLOCK, (A - dt**2/4*BLOCK).dot(u_old) + dt*np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))/Np)
    
    
    pic_fields.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, np.asfortranarray(1/2*(u + u_old)[:Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(1/2*(u + u_old)[Ntot:2*Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(1/2*(u + u_old)[2*Ntot:].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), Ueq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part)
    
    pic_pusher.pusher_step3(particles, L, dt, B_part, U_part)
    
    # step 4 (update Q and spans)
    pic_pusher.pusher_step4(particles, L, dt)
    
    spans0[:, 0] = np.floor(particles[:, 0]*Nel[0]).astype(int) + p[0]
    spans0[:, 1] = np.floor(particles[:, 1]*Nel[1]).astype(int) + p[1]
    spans0[:, 2] = np.floor(particles[:, 2]*Nel[2]).astype(int) + p[2]
    
    # step 5 (update V and weights)
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, np.asfortranarray(b[:Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[Ntot:2*Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[2*Ntot:].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_pusher.pusher_step5(particles, L, dt, B_part)
    
    particles[:, 6] = w0 - control*control_variate(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3], particles[:, 4], particles[:, 5])/g0
    '''
    
    # step 6 (non-Hamiltonian)
    if add_pressure:
        rho[:], u[:], pr[:] = np.split(LHS_LU.solve(RHS.dot(np.concatenate((rho, u, pr)))), [Ntot, 4*Ntot])
    
    # diagnostics (compute energies)
    energies[0] = 1/2*u.dot(A.dot(u))
    energies[1] = 1/2*b.dot(M2.dot(b))
    energies[2] = 1/(gamma - 1)*pr.dot(norm)
    energies[3] = 1/2*particles[:, 6].dot(particles[:, 3]**2 + particles[:, 4]**2 + particles[:, 5]**2)/Np + (control - 1)*Eh_eq
# =====================================================================================================================    





# ========================================== time integration =========================================================
if time_int == True:
    
    if restart == False:
        title = dir_results + identifier + '.txt'
        file  = open(title, 'ab')
        
        
        # == initial data to save ==
        #data  = np.concatenate((energies, np.array([0.])))
        #np.savetxt(file, data.reshape(1, 5), fmt = '%1.16e')
        
        data  = np.concatenate((pr, u[2*Ntot:3*Ntot], energies, np.array([0.])))
        np.savetxt(file, data.reshape(1, len(pr) + len(u[2*Ntot:3*Ntot]) + 5), fmt = '%1.16e')
        # ==========================

        print('initial energies : ', energies)
        
        time_step = 0
        counter   = 0
        
    else:
        title = dir_results + identifier + '.txt'
        file  = open(title, 'ab')
        
        particles[:, :]    = np.load(name_particles)
        rho[:]             = np.load(name_rho_coeff)
        u[:]               = np.load(name_u_coeff)
        b[:]               = np.load(name_b_coeff)
        pr[:]              = np.load(name_p_coeff)
        w0                 = np.load(name_control)[0]
        g0                 = np.load(name_control)[1]
        time_step, counter = np.load(name_time_step)
        
        spans0[:, 0] = np.floor(particles[:, 0]*Nel[0]).astype(int) + p[0]
        spans0[:, 1] = np.floor(particles[:, 1]*Nel[1]).astype(int) + p[1]
        spans0[:, 2] = np.floor(particles[:, 2]*Nel[2]).astype(int) + p[2]
        
        pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, np.asfortranarray(b[:Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[Ntot:2*Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(b[2*Ntot:].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
        pic_fields.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, np.asfortranarray(u[:Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(u[Ntot:2*Ntot].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), np.asfortranarray(u[2*Ntot:].reshape(Nbase0[0], Nbase0[1], Nbase0[2])), Ueq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part)
        
        particles[:, 6] = w0 - control*control_variate(particles[:, 0], particles[:, 1], particles[:, 2], particles[:, 3], particles[:, 4], particles[:, 5])/g0 

        
    print('start time integration! (total number of time steps : ' + str(int(Tend/dt)) + ')')
    
    while True:

        if (time_step*dt >= Tend) or ((time.time() - start_simulation)/60 > max_time):
            
            if create_restart:
                
                if not os.path.exists(dir_restart):
                    os.makedirs(dir_restart)
                
                counter += 1

                np.save(dir_restart + identifier + '_restart=particles' + str(counter), particles)
                np.save(dir_restart + identifier + '_restart=rho_coeff' + str(counter), rho)
                np.save(dir_restart + identifier + '_restart=u_coeff'   + str(counter), u)
                np.save(dir_restart + identifier + '_restart=b_coeff'   + str(counter), b)
                np.save(dir_restart + identifier + '_restart=p_coeff'   + str(counter), pr)
                np.save(dir_restart + identifier + '_restart=CV'        + str(counter), np.vstack((w0, g0)))
                np.save(dir_restart + identifier + '_restart=time'      + str(counter), np.array([time_step, counter]))
            
            break

        if time_step%10 == 0:
            print('time steps finished : ' + str(time_step))
            print('energies : ', energies)

        timea = time.time()
        update()
        timeb = time.time()

        if time_step == 0:
            print('time for one time step : ', timeb-timea)

        # == data to save ==========
        #data  = np.concatenate((energies, np.array([(time_step + 1)*dt])))
        #np.savetxt(file, data.reshape(1, 5), fmt = '%1.16e')
        
        data  = np.concatenate((pr, u[2*Ntot:3*Ntot], energies, np.array([(time_step + 1)*dt])))
        np.savetxt(file, data.reshape(1, len(pr) + len(u[2*Ntot:3*Ntot]) + 5), fmt = '%1.16e')
        # ==========================

        time_step += 1

    file.close()
# =====================================================================================================================