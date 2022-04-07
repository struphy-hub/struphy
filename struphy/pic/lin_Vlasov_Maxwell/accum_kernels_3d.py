# import pyccel decorators
from pyccel.decorators import types

# import background solution
import struphy.kinetic_equil.analytical.background_sol  as bs
import struphy.pic.mat_vec_filler                       as mvf


# ==========================================================================================================
@types(            'double[:,:]','int[:]','double[:]','double[:]','double[:]','int','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','double[:]','double[:]','double')
def accum_step_e_w(particles,    p,       t1,         t2,         t3,         np,   indN1,     indN2,     indN3,     indD1,     indD2,     indD3,     mat11,                mat22,                mat33,                vec1,           vec2,           vec3,           v_shift,    v_th,       n0      ):
    """
    TODO
    """
    from numpy import empty

    v = empty( 3, dtype=float )

    #$ omp parallel
    #$ omp do reduction ( + : mat11, mat22, mat33) private(ip, eta1, eta2, eta3, v1, v2, v3, v_norm2, v_shift, v_th, n0, weight, span, span1, span2, span3, bn1, bn2, bn3, bd1, bd2, bd3, f0, filling, filling_v1, filling_v2, filling_v3)
    # pn1, pn2, pn3, pd1, pd2, pd3, particles don't have to be declared since they are read-only in the code below
    for ip in range(np):

        # only do something if particle is inside the logical domain (s < 1)
        if particles[0, ip] > 1.0 or particles[1, ip] > 1.0 or particles[2, ip] > 1.0:
            continue

        # position
        eta1 = particles[0, ip]
        eta2 = particles[1, ip]
        eta3 = particles[2, ip]

        # velocity
        v1      = particles[3, ip]
        v2      = particles[4, ip]
        v3      = particles[5, ip]
        v      = [v1, v2, v3]
        v_norm2 = v1**2 + v2**2 + v3**2

        weight  = particles[6, ip]

        # background solution
        f0 = bs.maxwellian_point(v, v_shift, v_th, n0)

        # Compute filling for matrix L: f_0(x,v) * v^2
        filling    = f0 * v_norm2

        # Compute filling for vector Y: f_0(x,v) * w_p * v
        filling_v1 = f0 * weight * v1
        filling_v2 = f0 * weight * v2
        filling_v3 = f0 * weight * v3

        # fill matrix and vector
        mvf.m_v_fill_b_v1_diag(p, t1, t2, t3, indN1, indN2, indN3, indD1, indD2, indD3, eta1, eta2, eta3, mat11, mat22, mat33, filling, filling, filling, vec1, vec2, vec3, filling_v1, filling_v2, filling_v3)
