import numpy          as np
import bsplines       as bsp
import STRUPHY_pusher as pic_pusher
import time

#test_case = 'step3'
#test_case = 'step4'
test_case = 'step5'


parallel = True

L =     [2., 1.5, 3.]

DF     = np.array([[  L[0], 0., 0.], [0.,   L[1], 0.], [0., 0.,   L[2]]])
DFinv  = np.array([[1/L[0], 0., 0.], [0., 1/L[1], 0.], [0., 0., 1/L[2]]])

G      = np.array([[  L[0]**2, 0., 0.], [0.,   L[1]**2, 0.], [0., 0.,   L[2]**2]])
Ginv   = np.array([[1/L[0]**2, 0., 0.], [0., 1/L[1]**2, 0.], [0., 0., 1/L[2]**2]])

g_sqrt = L[0]*L[1]*L[2]

Beq    = g_sqrt*DFinv.dot(np.array([1., 0., 0.]))
Ueq    = DF.T.dot(np.array([0., 0., 0.]))


dt = 0.12

if test_case == 'step3':
    if parallel == False:
        print('------------ step 3 : not parallel ------------------')
        Np        = int(1e6)

        particles = np.asfortranarray(np.random.rand(Np, 7))

        B_part    = np.asfortranarray(np.random.rand(Np, 3))
        U_part    = np.asfortranarray(np.random.rand(Np, 3))

        np.save('test_particles_before', particles)
        np.save('b_part', B_part)
        np.save('u_part', U_part)

        timea = time.time()
        pic_pusher.pusher_step3(particles, dt, B_part, U_part, 1, L)
        timeb = time.time()
        print('time not parallel step 3: ', timeb - timea)

        np.save('test_particles_after', particles)

    else:
        print('------------ step 3 : parallel ------------------')
        particles = np.load('test_particles_before.npy')

        B_part    = np.load('b_part.npy')
        U_part    = np.load('u_part.npy')

        Np        = len(particles[:, 0])

        timea = time.time()
        pic_pusher.pusher_step3(particles, dt, B_part, U_part, 1, L)
        timeb = time.time()
        print('time parallel step 3: ', timeb - timea)
        print('same result? : ', np.allclose(particles, np.load('test_particles_after.npy')))
        
        
if test_case == 'step4':
    if parallel == False:
        print('------------ step 4 : not parallel ------------------')
        Np        = int(1e6)

        particles = np.asfortranarray(np.random.rand(Np, 7))

        np.save('test_particles_before', particles)

        timea = time.time()
        pic_pusher.pusher_step4(particles, dt, 1, L)
        timeb = time.time()
        print('time not parallel step 4 ', timeb - timea)

        np.save('test_particles_after', particles)

    else:
        print('------------ step 4 : parallel ------------------')
        particles = np.load('test_particles_before.npy')

        Np        = len(particles[:, 0])

        timea = time.time()
        pic_pusher.pusher_step4(particles, dt, 1, L)
        timeb = time.time()
        print('time parallel step 4: ', timeb - timea)
        print('same result? : ', np.allclose(particles, np.load('test_particles_after.npy')))
        
        
if test_case == 'step5':
    if parallel == False:
        print('------------ step 5 : not parallel ------------------')
        Np        = int(1e6)

        particles = np.asfortranarray(np.random.rand(Np, 7))

        B_part    = np.asfortranarray(np.random.rand(Np, 3))

        np.save('test_particles_before', particles)
        np.save('b_part', B_part)

        timea = time.time()
        pic_pusher.pusher_step5(particles, dt, B_part, 1, L)
        timeb = time.time()
        print('time not parallel step 5: ', timeb - timea)

        np.save('test_particles_after', particles)

    else:
        print('------------ step 5 : parallel ------------------')
        particles = np.load('test_particles_before.npy')

        B_part    = np.load('b_part.npy')

        Np        = len(particles[:, 0])

        timea = time.time()
        pic_pusher.pusher_step5(particles, dt, B_part, 1, L)
        timeb = time.time()
        print('time parallel step 5: ', timeb - timea)
        print('same result? : ', np.allclose(particles, np.load('test_particles_after.npy')))