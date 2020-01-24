import numpy             as np
import matplotlib.pyplot as plt
import bsplines          as bsp

import ECHO_pusher
import ECHO_fields
import ECHO_accumulation

import time

#====================================================================================
#  calling epyccel for particle pusher
#====================================================================================
from pyccel.epyccel import epyccel

pic_pusher = epyccel(ECHO_pusher, accelerator='openmp')
pic_fields = epyccel(ECHO_fields, accelerator='openmp')
pic_accumu = epyccel(ECHO_accumulation, accelerator='openmp')

print('pyccelization done!')
#====================================================================================


test_case = 'B-field'





Nel = [3, 4, 6]            # mesh generation on logical domain
bc  = [True, True, True]   # boundary conditions
p   = [2, 2, 3]            # splines degrees

L   = [2., 3., 1.]         # box lengthes of physical domain
Np  = int(100)              # number of particles

el_b     = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                           # element boundaries
T        = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]       # knot vectors
t        = [T[1:-1] for T in T]                                                    # reduced knot vectors
Nbase0   = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]                      # number of basis functions in V0

dt       = 0.15


particles = np.empty((Np, 7), dtype=float, order='F')
spans0    = np.empty((Np, 3), dtype=int, order='F')

b1 = np.empty((Nel[0], Nel[1], Nel[2]), order='F')
b2 = np.empty((Nel[0], Nel[1], Nel[2]), order='F')
b3 = np.empty((Nel[0], Nel[1], Nel[2]), order='F')

u1 = np.empty((Nel[0], Nel[1], Nel[2]), order='F')
u2 = np.empty((Nel[0], Nel[1], Nel[2]), order='F')
u3 = np.empty((Nel[0], Nel[1], Nel[2]), order='F')

U_part = np.empty((Np, 3), dtype=float, order='F')
B_part = np.empty((Np, 3), dtype=float, order='F')


particles[:, :] = np.load('test_particles.npy')

b1[:, :, :] = np.load('b1_coeff.npy')
b2[:, :, :] = np.load('b2_coeff.npy')
b3[:, :, :] = np.load('b3_coeff.npy')

u1[:, :, :] = np.load('u1_coeff.npy')
u2[:, :, :] = np.load('u2_coeff.npy')
u3[:, :, :] = np.load('u3_coeff.npy')


spans0[:, 0] = np.floor(particles[:, 0]*Nel[0]).astype(int) + p[0]
spans0[:, 1] = np.floor(particles[:, 1]*Nel[1]).astype(int) + p[1]
spans0[:, 2] = np.floor(particles[:, 2]*Nel[2]).astype(int) + p[2]




DF     = np.array([[  L[0], 0., 0.], [0.,   L[1], 0.], [0., 0.,   L[2]]])
DFinv  = np.array([[1/L[0], 0., 0.], [0., 1/L[1], 0.], [0., 0., 1/L[2]]])

G      = np.array([[  L[0]**2, 0., 0.], [0.,   L[1]**2, 0.], [0., 0.,   L[2]**2]])
Ginv   = np.array([[1/L[0]**2, 0., 0.], [0., 1/L[1]**2, 0.], [0., 0., 1/L[2]**2]])

g_sqrt = L[0]*L[1]*L[2]

Beq    = g_sqrt*DFinv.dot(np.array([0., 0., 1.]))
Ueq    = DF.T.dot(np.array([0.02, 0., 0.]))

#=================== coefficients for pp-forms (1 - component) ======================
if p[0] == 3:
    d1 = 1/Nel[0]
    pp0_1 = np.asfortranarray([[1/6, -1/(2*d1), 1/(2*d1**2), -1/(6*d1**3)], [2/3, 0., -1/d1**2, 1/(2*d1**3)], [1/6, 1/(2*d1), 1/(2*d1**2), -1/(2*d1**3)], [0., 0., 0., 1/(6*d1**3)]])
    pp1_1 = np.asfortranarray([[1/2, -1/d1, 1/(2*d1**2)], [1/2, 1/d1, -1/d1**2], [0., 0., 1/(2*d1**2)]])/d1
elif p[0] == 2:
    d1 = 1/Nel[0]
    pp0_1 = np.asfortranarray([[1/2, -1/d1, 1/(2*d1**2)], [1/2, 1/d1, -1/d1**2], [0., 0., 1/(2*d1**2)]])
    pp1_1 = np.asfortranarray([[1., -1/d1], [0., 1/d1]])/d1
else:
    print('Only cubic and quadratic splines implemented!')
#====================================================================================



#=================== coefficients for pp-forms (2 - component) ======================
if p[1] == 3:
    d2 = 1/Nel[1]
    pp0_2 = np.asfortranarray([[1/6, -1/(2*d2), 1/(2*d2**2), -1/(6*d2**3)], [2/3, 0., -1/d2**2, 1/(2*d2**3)], [1/6, 1/(2*d2), 1/(2*d2**2), -1/(2*d2**3)], [0., 0., 0., 1/(6*d2**3)]])
    pp1_2 = np.asfortranarray([[1/2, -1/d2, 1/(2*d2**2)], [1/2, 1/d2, -1/d2**2], [0., 0., 1/(2*d2**2)]])/d2
elif p[1] == 2:
    d2 = 1/Nel[1]
    pp0_2 = np.asfortranarray([[1/2, -1/d2, 1/(2*d2**2)], [1/2, 1/d2, -1/d2**2], [0., 0., 1/(2*d2**2)]])
    pp1_2 = np.asfortranarray([[1., -1/d2], [0., 1/d2]])/d2
else:
    print('Only cubic and quadratic splines implemented!')
#====================================================================================



#=================== coefficients for pp-forms (3 - component) ======================
if p[2] == 3:
    d3 = 1/Nel[2]
    pp0_3 = np.asfortranarray([[1/6, -1/(2*d3), 1/(2*d3**2), -1/(6*d3**3)], [2/3, 0., -1/d3**2, 1/(2*d3**3)], [1/6, 1/(2*d3), 1/(2*d3**2), -1/(2*d3**3)], [0., 0., 0., 1/(6*d3**3)]])
    pp1_3 = np.asfortranarray([[1/2, -1/d3, 1/(2*d3**2)], [1/2, 1/d3, -1/d3**2], [0., 0., 1/(2*d3**2)]])/d3
elif p[2] == 2:
    d3 = 1/Nel[2]
    pp0_3 = np.asfortranarray([[1/2, -1/d3, 1/(2*d3**2)], [1/2, 1/d3, -1/d3**2], [0., 0., 1/(2*d3**2)]])
    pp1_3 = np.asfortranarray([[1., -1/d3], [0., 1/d3]])/d3
else:
    print('Only cubic and quadratic splines implemented!')
#====================================================================================





# test evaluate_1-form
if test_case == 'U-field':
    timea = time.time()
    pic_fields.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, u1, u2, u3, Ueq, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, U_part)
    np.save('U_part_parallel', U_part)
    timeb = time.time()
    print('evaluate 1-form; time : ', timeb-timea)
    
# test evaluate_2-form
if test_case == 'B-field':
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0_1, pp0_2, pp0_3, pp1_1, pp1_2, pp1_3, B_part)
    np.save('B_part_parallel', B_part)
    timeb = time.time()
    print('evaluate 2-form; time : ', timeb-timea)