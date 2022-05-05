# import pyccel decorators
from pyccel.decorators import types

# import background solution
import struphy.kinetic_equil.analytical.background_sol  as bs
import struphy.pic.mat_vec_filler                       as mvf

# ==========================================================================================================
@types(            'double[:,:]','int[:]','double[:]','double[:]','double[:]','int','int[:,:]','int[:,:]','int[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double')
def accum_step_e_w(particles,    p,       t1,         t2,         t3,         np,   indN1,     indN2,     indN3,     mat11,                mat12,                mat13,                mat22,                mat23,                mat33,                vec1,           vec2,           vec3,           v_shift,    v_th,       n0      ):
    """
    Does the Accumulation in the 3rd substep (e_w step, with b=const)

    Parameters :
    ------------
        particles : array
            contains the positions [0:3,], velocities [3:6,], and weights [6,] of the particles

        p : array of integers
            contains the degrees of the B-splines in every direction
        
        t1 : array
            contains the knot sequence in direction 1

        t2 : array
            contains the knot sequence in direction 2

        t3 : array
            contains the knot sequence in direction 3
        
        np : int
            total number of particles
        
        indN1 : array of integers
            contains the global indices of the non-vanishing B-splines in direction 1

        indN2 : array of integers
            contains the global indices of the non-vanishing B-splines in direction 2

        indN3 : array of integers
            contains the global indices of the non-vanishing B-splines in direction 3

        indD1 : array of integers
            contains the global indices of the non-vanishing D-splines in direction 1

        indD2 : array of integers
            contains the global indices of the non-vanishing D-splines in direction 2

        indD3 : array of integers
            contains the global indices of the non-vanishing D-splines in direction 3
        
        mat11 : array
            mu=1,nu=1 element of the accumulation matrix that is to be filled

        mat12 : array
            mu=1,nu=2 element of the accumulation matrix that is to be filled

        mat13 : array
            mu=1,nu=3 element of the accumulation matrix that is to be filled
        
        mat22 : array
            mu=2,nu=2 element of the accumulation matrix that is to be filled

        mat23 : array
            mu=2,nu=3 element of the accumulation matrix that is to be filled

        mat33 : array
            mu=3,nu=3 element of the accumulation matrix that is to be filled
        
        vec1 : array
            mu=1 component of the accumulation vector that is to be filled

        vec2 : array
            mu=2 component of the accumulation vector that is to be filled

        vec3 : array
            mu=3 component of the accumulation vector that is to be filled
        
        v_shift : array
            contains the 3 values of the shift velocity in the background solution (maxwellian) for all 3 directions
        
        v_th : array
            contains the 3 values for the thermal velocity of the background solution (maxwellian) for all 3 directions
        
        n0 : double
            homogeneous density of the background solution (maxwellian)
    """
    from numpy import empty, sqrt

    v = empty( 3, dtype=float )

    #$ omp parallel
    #$ omp do reduction ( + :mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3) private(ip, eta1, eta2, eta3, v, v1, v2, v3, weight, f0, filling_m11, filling_m12, filling_m13, filling_m22, filling_m23, filling_m33, filling_v1, filling_v2, filling_v3)
    for ip in range(np):

        # position
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        # velocity
        v1      = particles[3, ip]
        v2      = particles[4, ip]
        v3      = particles[5, ip]
        v       = [v1, v2, v3]

        weight  = particles[6, ip]

        # background solution
        f0 = bs.maxwellian_point(v, v_shift, v_th, n0)

        # Compute filling for matrix: f_0(x,v) * v^2
        filling_m11 = f0 * v1**2
        filling_m12 = f0 * v1 * v2
        filling_m13 = f0 * v1 * v3
        filling_m22 = f0 * v2**2
        filling_m23 = f0 * v2 * v3
        filling_m33 = f0 * v3**2

        # Compute filling for vector: sqrt(f_0(x,v)) * w_p * v
        filling_v1 = sqrt(f0) * weight * v1
        filling_v2 = sqrt(f0) * weight * v2
        filling_v3 = sqrt(f0) * weight * v3

        # fill matrix and vector
        mvf.m_v_fill_b_v1_symm(p, t1, t2, t3, indN1, indN2, indN3, eta1, eta2, eta3, mat11, mat12, mat13, mat22, mat23, mat33, filling_m11, filling_m12, filling_m13, filling_m22, filling_m23, filling_m33, vec1, vec2, vec3, filling_v1, filling_v2, filling_v3)

    #$ omp end do
    #$ omp end parallel

    ierr = 0