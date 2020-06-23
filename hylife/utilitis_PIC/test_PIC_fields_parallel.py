import numpy          as np
import bsplines       as bsp
import STRUPHY_fields as pic_fields
import time

parallel  = True


Nel       = [8, 9, 10]                # mesh generation on logical domain
bc        = [False, False, False]     # boundary conditions (True: periodic, False: else)
p         = [3, 2, 3]                 # spline degrees  


el_b      = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                      # element boundaries
delta     = [1/Nel for Nel in Nel]                                             # element sizes
T         = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]  # knot vectors (for N functions)
t         = [T[1:-1] for T in T]                                               # reduced knot vectors (for D function)
NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]                 # number of basis functions (N functions)
NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]              # number of basis functions (D functions)

Nbase     = np.asfortranarray([[NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]]])



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



if parallel == False:
    print('------------ not parallel ------------------')
    Np        = int(1e5)
    particles = np.asfortranarray(np.random.rand(Np, 7))
    
    B_part = np.empty((Np, 3), dtype=float, order='F')
    U_part = np.empty((Np, 3), dtype=float, order='F')
    
    b1 = np.empty((NbaseN[0], NbaseD[1], NbaseD[2]), order='F')
    b2 = np.empty((NbaseD[0], NbaseN[1], NbaseD[2]), order='F')
    b3 = np.empty((NbaseD[0], NbaseD[1], NbaseN[2]), order='F')

    b1[:, :, :] = 0.1*(np.random.rand(NbaseN[0], NbaseD[1], NbaseD[2]) - 0.5)
    b2[:, :, :] = 0.1*(np.random.rand(NbaseD[0], NbaseN[1], NbaseD[2]) - 0.5)
    b3[:, :, :] = 0.1*(np.random.rand(NbaseD[0], NbaseD[1], NbaseN[2]) - 0.5)
    
    u1 = np.empty((NbaseD[0], NbaseN[1], NbaseN[2]), order='F')
    u2 = np.empty((NbaseN[0], NbaseD[1], NbaseN[2]), order='F')
    u3 = np.empty((NbaseN[0], NbaseN[1], NbaseD[2]), order='F')

    u1[:, :, :] = 0.1*(np.random.rand(NbaseD[0], NbaseN[1], NbaseN[2]) - 0.5)
    u2[:, :, :] = 0.1*(np.random.rand(NbaseN[0], NbaseD[1], NbaseN[2]) - 0.5)
    u3[:, :, :] = 0.1*(np.random.rand(NbaseN[0], NbaseN[1], NbaseD[2]) - 0.5)
    
    np.save('test_particles', particles)
    np.save('b1_coeff', b1)
    np.save('b2_coeff', b2)
    np.save('b3_coeff', b3)
    np.save('u1_coeff', u1)
    np.save('u2_coeff', u2)
    np.save('u3_coeff', u3)
    
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, :3], T[0], T[1], T[2], t[0], t[1], t[2], p, Nel, Nbase, Np, b1, b2, b3, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part, 1, [1., 1., 1.])
    timeb = time.time()
    print('time not parallel B: ', timeb - timea)
    
    timea = time.time()
    pic_fields.evaluate_1form(particles[:, :3], T[0], T[1], T[2], t[0], t[1], t[2], p, Nel, Nbase, Np, u1, u2, u3, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part, 1, [1., 1., 1.])
    timeb = time.time()
    print('time not parallel U: ', timeb - timea)
    
    
    np.save('b_part', B_part)
    np.save('u_part', U_part)
    
else:
    print('------------ parallel ------------------')
    particles = np.load('test_particles.npy')
    
    b1 = np.load('b1_coeff.npy')
    b2 = np.load('b2_coeff.npy')
    b3 = np.load('b3_coeff.npy')
    u1 = np.load('u1_coeff.npy')
    u2 = np.load('u2_coeff.npy')
    u3 = np.load('u3_coeff.npy')
    
    Np = len(particles[:, 0])
    
    B_part = np.empty((Np, 3), dtype=float, order='F')
    U_part = np.empty((Np, 3), dtype=float, order='F')
    
    timea = time.time()
    pic_fields.evaluate_2form(particles[:, :3], T[0], T[1], T[2], t[0], t[1], t[2], p, Nel, Nbase, Np, b1, b2, b3, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], B_part, 1, [1., 1., 1.])
    timeb = time.time()
    print('time parallel B: ', timeb - timea)
    print('same result? : ', np.allclose(B_part, np.load('b_part.npy')))
    
    timea = time.time()
    pic_fields.evaluate_1form(particles[:, :3], T[0], T[1], T[2], t[0], t[1], t[2], p, Nel, Nbase, Np, u1, u2, u3, pp0[0], pp0[1], pp0[2], pp1[0], pp1[1], pp1[2], U_part, 1, [1., 1., 1.])
    timeb = time.time()
    print('time parallel U: ', timeb - timea)
    print('same result? : ', np.allclose(U_part, np.load('u_part.npy')))