import numpy                as np
import matplotlib.pyplot    as plt
import bsplines             as bsp

import STRUPHY_pusher       as pic_pusher
import STRUPHY_fields       as pic_fields
import STRUPHY_accumulation as pic_accumu

import time



test_case = 'accumulation_step3'



Nel = [3, 4, 6]            # mesh generation on logical domain
bc  = [True, True, True]   # boundary conditions
p   = [2, 2, 3]            # splines degrees

L   = [2., 3., 1.]         # box lengthes of physical domain
Np  = int(100)             # number of particles

el_b     = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                           # element boundaries
T        = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]       # knot vectors
t        = [T[1:-1] for T in T]                                                    # reduced knot vectors
Nbase0   = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]                      # number of basis functions in V0
delta    = [1/Nel for Nel in Nel]

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



# test evaluate_1-form
if test_case == 'U-field':
    timea = time.time()
    pic_fields.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, u1, u2, u3, Ueq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part)
    timeb = time.time()
    np.save('U_part_parallel', U_part)
    print('evaluate 1-form; time : ', timeb-timea)
    

# test evaluate_2-form
if test_case == 'B-field':
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    timeb = time.time()
    np.save('B_part_parallel', B_part)
    print('evaluate 2-form; time : ', timeb-timea)
    

# test pusher_step3
if test_case == 'pusher_step3':
    B_part = np.empty((Np, 3), dtype=float, order='F')
    U_part = np.empty((Np, 3), dtype=float, order='F')
    
    timea = time.time()
    pic_fields.evaluate_1form(particles[:, 0:3], p, spans0, Nbase0, Np, u1, u2, u3, Ueq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part)
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_pusher.pusher_step3(particles, L, dt, B_part, U_part)
    timeb = time.time()
    np.save('particles_after_push3', particles)
    print('pusher_step3; time : ', timeb - timea)
    
      
# test pusher_step4    
if test_case == 'pusher_step4':
    timea = time.time()
    pic_pusher.pusher_step4(particles, L, dt)
    timeb = time.time()
    np.save('particles_after_push4', particles)
    print('pusher_step4; time : ', timeb - timea)
    
    
# test pusher_step5   
if test_case == 'pusher_step5':
    B_part = np.empty((Np, 3), dtype=float, order='F')
    
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_pusher.pusher_step5(particles, L, dt, B_part)
    timeb = time.time()
    np.save('particles_after_push5', particles)
    print('pusher_step5; time : ', timeb - timea)


# test accumulation_step1   
if test_case == 'accumulation_step1':
    B_part = np.empty((Np, 3), dtype=float, order='F')
    
    mat12  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat13  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    mat23  = np.empty((Nbase0[0], Nbase0[1], Nbase0[2], Nbase0[0], Nbase0[1], Nbase0[2]), dtype=float, order='F')
    
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, 0:3], p, spans0, Nbase0, Np, b1, b2, b3, Beq, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part)
    pic_accumu.accumulation_step1(particles, p, spans0, Nbase0, T[0], T[1], T[2], t[0], t[1], t[2], L, B_part, mat12, mat13, mat23)
    timeb = time.time()
    np.save('mat12_step1', mat12)
    np.save('mat13_step1', mat13)
    np.save('mat23_step1', mat23)
    print('accumulation_step1; time : ', timeb - timea)
    
    
# test accumulation_step3   
if test_case == 'accumulation_step3':
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
    
    np.save('mat11_step3', mat11)
    np.save('mat12_step3', mat12)
    np.save('mat13_step3', mat13)
    
    np.save('mat22_step3', mat22)
    np.save('mat23_step3', mat23)
    np.save('mat33_step3', mat33)
    
    np.save('vec1_step3', vec1)
    np.save('vec2_step3', vec2)
    np.save('vec3_step3', vec3)
    
    print('accumulation_step3; time : ', timeb - timea)