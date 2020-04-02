import numpy          as np
import bsplines       as bsp
import STRUPHY_accumulation as pic_accumu
import time


test_case = 'step1'
#test_case = 'step3'

parallel  = False


Nel       = [24, 24, 4]                # mesh generation on logical domain
bc        = [True, True, True]     # boundary conditions (True: periodic, False: else)
p         = [3, 3, 2]                 # spline degrees  


el_b      = [np.linspace(0., 1., Nel + 1) for Nel in Nel]                      # element boundaries
delta     = [1/Nel for Nel in Nel]                                             # element sizes
T         = [bsp.make_knots(el_b, p, bc) for el_b, p, bc in zip(el_b, p, bc)]  # knot vectors (for N functions)
t         = [T[1:-1] for T in T]                                               # reduced knot vectors (for D function)
NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]                 # number of basis functions (N functions)
NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]


L      = [3., 1.5, 2.]

DF     = np.array([[  L[0], 0., 0.], [0.,   L[1], 0.], [0., 0.,   L[2]]])
DFinv  = np.array([[1/L[0], 0., 0.], [0., 1/L[1], 0.], [0., 0., 1/L[2]]])

G      = np.array([[  L[0]**2, 0., 0.], [0.,   L[1]**2, 0.], [0., 0.,   L[2]**2]])
Ginv   = np.array([[1/L[0]**2, 0., 0.], [0., 1/L[1]**2, 0.], [0., 0., 1/L[2]**2]])

g_sqrt = L[0]*L[1]*L[2]


Ntot = Nel[0]*Nel[1]*Nel[2]


if test_case == 'step1':
    if parallel == False:
        print('------------ step 1 : not parallel ------------------')
        Np        = int(1e5)

        particles = np.asfortranarray(np.random.rand(Np, 7))

        B_part    = np.asfortranarray(np.random.rand(Np, 3))

        np.save('test_particles', particles)
        np.save('b_part', B_part)
        
        acc = pic_accumu.accumulation(T, p, bc)

        timea = time.time()
        mat = acc.accumulation_step1(particles, B_part, 1, L)
        timeb = time.time()
        print('time not parallel step 1: ', timeb - timea)

        np.save('matrix_step1', mat.toarray())

    else:
        print('------------ step 1 : parallel ------------------')
        particles = np.load('test_particles.npy')

        B_part    = np.load('b_part.npy')

        Np        = len(particles[:, 0])
        
        acc = pic_accumu.accumulation(T, p, bc)

        timea = time.time()
        mat = acc.accumulation_step1(particles, B_part, 1, L)
        timeb = time.time()
        print('time parallel step 1: ', timeb - timea)
        print('same result matrix? : ', np.allclose(mat.toarray(), np.load('matrix_step1.npy')))
        
        
if test_case == 'step3':
    if parallel == False:
        print('------------ step 3 : not parallel ------------------')
        Np        = int(1e5)

        particles = np.asfortranarray(np.random.rand(Np, 7))

        B_part    = np.asfortranarray(np.random.rand(Np, 3))

        np.save('test_particles', particles)
        np.save('b_part', B_part)
        
        acc = pic_accumu.accumulation(T, p, bc)

        timea = time.time()
        mat, vec = acc.accumulation_step3(particles, B_part, 1, L)
        timeb = time.time()
        print('time not parallel step 3: ', timeb - timea)

        np.save('matrix_step3', mat.toarray())
        np.save('vector_step3', vec)

    else:
        print('------------ step 3 : parallel ------------------')
        particles = np.load('test_particles.npy')

        B_part    = np.load('b_part.npy')

        Np        = len(particles[:, 0])
        
        acc = pic_accumu.accumulation(T, p, bc)

        timea = time.time()
        mat, vec = acc.accumulation_step3(particles, B_part, 1, L)
        timeb = time.time()
        print('time parallel step 3: ', timeb - timea)
        print('same result vector? : ', np.allclose(vec, np.load('vector_step3.npy')))
        print('same result matrix? : ', np.allclose(mat.toarray(), np.load('matrix_step3.npy')))