import numpy             as np
import matplotlib.pyplot as plt
import bsplines          as bsp
import Bspline           as bspline

import ECHO_fields
import ECHO_pusher
import ECHO_accumulation

import time

test_case = 'accumulation_step1'


#====================================================================================
#  calling epyccel for particle pusher
#====================================================================================
from pyccel.epyccel import epyccel

if (test_case == 'fields-B') or (test_case == 'fields-U'):

    pic        = epyccel(ECHO_fields, accelerator='openmp')
    
elif (test_case == 'pusher_step3') or (test_case == 'pusher_step4') or (test_case == 'pusher_step5'):
    
    pic_fields = epyccel(ECHO_fields, accelerator='openmp')
    pic_pusher = epyccel(ECHO_pusher, accelerator='openmp')
    
elif (test_case == 'accumulation_step1') or (test_case == 'accumulation_step3'):
    
    pic_fields = epyccel(ECHO_fields      , accelerator='openmp')
    pic_accumu = epyccel(ECHO_accumulation, accelerator='openmp')
    

print('pyccelization done!')
#====================================================================================



Nel = [3, 3, 40]           # mesh generation on logical domain
bc  = [True, True, True]   # boundary conditions
p   = [2, 2, 2]            # splines degrees

L   = [1., 1., 2*np.pi]    # box lengthes of physical domain
Np  = int(5e5)             # number of particles
dt  = 0.05                 # time step

el_b     = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                           # element boundaries
T        = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]       # knot vectors
t        = [T[1:-1] for T in T]                                                    # reduced knot vectors
Nbase0   = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]                      # number of basis functions in V0
delta    = [1/Nel for Nel in Nel]         

particles       = np.empty((Np, 7), dtype=float, order='F')

particles[:, :] = np.random.rand(Np, 7)

spans0       = np.empty((Np, 3), dtype=int, order='F')
spans0[:, 0] = np.floor(particles[:, 0]*Nel[0]).astype(int) + p[0]
spans0[:, 1] = np.floor(particles[:, 1]*Nel[1]).astype(int) + p[1]
spans0[:, 2] = np.floor(particles[:, 2]*Nel[2]).astype(int) + p[2]

b1 = np.zeros((Nel[0], Nel[1], Nel[2]), order='F')
b2 = np.zeros((Nel[0], Nel[1], Nel[2]), order='F')
b3 = np.zeros((Nel[0], Nel[1], Nel[2]), order='F')

u1 = np.zeros((Nel[0], Nel[1], Nel[2]), order='F')
u2 = np.zeros((Nel[0], Nel[1], Nel[2]), order='F')
u3 = np.zeros((Nel[0], Nel[1], Nel[2]), order='F')

b1[:, :, :] = np.random.rand(Nel[0], Nel[1], Nel[2]) - 0.5
b2[:, :, :] = np.random.rand(Nel[0], Nel[1], Nel[2]) - 0.5
b3[:, :, :] = np.random.rand(Nel[0], Nel[1], Nel[2]) - 0.5

u1[:, :, :] = np.random.rand(Nel[0], Nel[1], Nel[2]) - 0.5
u2[:, :, :] = np.random.rand(Nel[0], Nel[1], Nel[2]) - 0.5
u3[:, :, :] = np.random.rand(Nel[0], Nel[1], Nel[2]) - 0.5

DF     = np.array([[  L[0], 0., 0.], [0.,   L[1], 0.], [0., 0.,   L[2]]])
DFinv  = np.array([[1/L[0], 0., 0.], [0., 1/L[1], 0.], [0., 0., 1/L[2]]])

G      = np.array([[  L[0]**2, 0., 0.], [0.,   L[1]**2, 0.], [0., 0.,   L[2]**2]])
Ginv   = np.array([[1/L[0]**2, 0., 0.], [0., 1/L[1]**2, 0.], [0., 0., 1/L[2]**2]])

g_sqrt = L[0]*L[1]*L[2]

Beq    = g_sqrt*DFinv.dot(np.array([0., 0., 1.]))
Ueq    = DF.T.dot(np.array([0.02, 0., 0.]))



# ================ coefficients for pp-forms in interval [0, delta] (N and D) ==================
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
# ==============================================================================================


if test_case == 'fields-B':
    print('-------------fields-B------------------')
    B_part = np.empty((Np, 3), dtype=float, order='F')

    timea = time.time()
    pic.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    timeb = time.time()
    
    print('time : ', timeb - timea)
    
elif test_case == 'fields-U':
    print('-------------fields-U------------------')
    U_part          = np.empty((Np, 3), dtype=float, order='F')

    timea = time.time()
    pic.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, u1, u2, u3, Ueq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part)
    timeb = time.time()
    print('time : ', timeb - timea)

    

elif test_case == 'pusher_step3':
    print('-------------pusher-step3------------------')
    
    B_part          = np.empty((Np, 3), dtype=float, order='F')
    U_part          = np.empty((Np, 3), dtype=float, order='F')
    
    timea = time.time()
    pic_fields.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, u1, u2, u3, Ueq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part)
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_pusher.pusher_step3(particles, L, dt, B_part, U_part)
    timeb = time.time()
    print('time : ', timeb - timea)
    
      
    
elif test_case == 'pusher_step4':
    print('-------------pusher-step4------------------')
    
    timea = time.time()
    pic_pusher.pusher_step4(particles, L, dt)
    timeb = time.time()
    print('time : ', timeb - timea)
    
    
elif test_case == 'pusher_step5':
    print('-------------pusher-step5------------------')
    
    B_part          = np.empty((Np, 3), dtype=float, order='F')
    
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_pusher.pusher_step5(particles, L, dt, B_part)
    timeb = time.time()
    print('time : ', timeb - timea)
    
    
elif test_case == 'accumulation_step1':
    print('-------------accumulation-step1------------------')
    
    B_part         = np.empty((Np, 3), dtype=float, order='F')
    
    mat12          = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat13          = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat23          = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_accumu.accumulation_step1(particles, p, spans0, Nbase0, T[0], T[1], T[2], t[0], t[1], t[2], L, B_part, mat12, mat13, mat23)
    timeb = time.time()
    print('time : ', timeb - timea)
    
    
elif test_case == 'accumulation_step3':
    print('-------------accumulation-step3------------------')
    
    B_part = np.empty((Np, 3), dtype=float, order='F')
    
    mat11  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat12  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat13  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat22  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat23  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat33  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    
    vec1   = np.empty((Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    vec2   = np.empty((Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    vec3   = np.empty((Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    
    
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_accumu.accumulation_step3(particles, p, spans0, Nbase0, T[0], T[1], T[2], t[0], t[1], t[2], L, B_part, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
    timeb = time.time()
    print('time : ', timeb - timea)