# import modules for matrix-matrix and matrix-vector multiplications
import struphy.kinetic_background.analytical.background_sol  as bs
import struphy.pic.mat_vec_filler                       as mvf
import struphy.geometry.mappings_3d_fast                as mapping_fast

# ==========================================================================================================
def accum_step_e_w(particles: 'float[:,:]', kind_map: 'int', params_map: 'float[:]', p: 'int[:]', t1: 'float[:]', t2: 'float[:]', t3: 'float[:]', np: 'int', indN1: 'int[:,:]', indN2: 'int[:,:]', indN3: 'int[:,:]', mat11: 'float[:,:,:,:,:,:]', mat12: 'float[:,:,:,:,:,:]', mat13: 'float[:,:,:,:,:,:]', mat21: 'float[:,:,:,:,:,:]', mat22: 'float[:,:,:,:,:,:]', mat23: 'float[:,:,:,:,:,:]', mat31: 'float[:,:,:,:,:,:]', mat32: 'float[:,:,:,:,:,:]', mat33: 'float[:,:,:,:,:,:]', vec1: 'float[:,:,:]', vec2: 'float[:,:,:]', vec3: 'float[:,:,:]', v_shift: 'float[:]', v_th: 'float[:]', n0: 'float', cx: 'float[:,:,:]', cy: 'float[:,:,:]', cz: 'float[:,:,:]'):
    """
    Does the Accumulation in the 3rd substep (e_w step, with b=const)

    Parameters :
    ------------
        particles : array
            contains the positions [0:3,], velocities [3:6,], and weights [6,] of the particles

        kind_map : integer
            if kind_map is 0,1,2 then the mapping is given in terms of splines, otherwise an analytical expression is given

        params_map : array
            contains parameters for the analytical mapping

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
        
        cx, cy, cz : array
            contains the spline coefficients for the mapping in each direction
    """
    from numpy import empty, sqrt

    v = empty( 3, dtype=float )

    df      = empty( (3,3), dtype=float )
    df_inv  = empty( (3,3), dtype=float )
    g_inv   = empty( (3,3), dtype=float )
    fx      = empty( 3    , dtype=float )

    # make extra arrays for G_inv.v and Df_inv.v
    Gv = empty ( 3, dtype=float )
    Dv = empty ( 3, dtype=float )

    #$ omp parallel private (ip, eta1, eta2, eta3, df, fx, df_inv, g_inv, Gv, Dv, v, v1, v2, v3, weight, f0, filling_m11, filling_m12, filling_m13, filling_m21, filling_m22, filling_m23, filling_m31, filling_m32, filling_m33, filling_v1, filling_v2, filling_v3)
    #$ omp for reduction ( + :mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33, vec1, vec2, vec3)
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

        mapping_fast.dl_all(kind_map, params_map, t1, t2, t3, p, cx, cy, cz, indN1, indN2, indN3, eta1, eta2, eta3, df, fx, 0)
        
        # evaluate inverse Jacobian matrix
        mapping_fast.df_inv_all(df, df_inv)
        
        # evaluate inverse metric tensor
        mapping_fast.g_inv_all(df_inv, g_inv)

        Gv[0] = g_inv[0,0] * v1 + g_inv[0,1] * v2 + g_inv[0,2] * v3
        Gv[1] = g_inv[1,0] * v1 + g_inv[1,1] * v2 + g_inv[1,2] * v3
        Gv[2] = g_inv[2,0] * v1 + g_inv[2,1] * v2 + g_inv[2,2] * v3

        Dv[0] = df_inv[0,0] * v1 + df_inv[0,1] * v2 + df_inv[0,2] * v3
        Dv[1] = df_inv[1,0] * v1 + df_inv[1,1] * v2 + df_inv[1,2] * v3
        Dv[2] = df_inv[2,0] * v1 + df_inv[2,1] * v2 + df_inv[2,2] * v3

        # Compute filling for matrix: f_0(x,v) * ( G^{-1} v )_mu ( DL^{-1} v )_nu
        filling_m11 = f0 * Gv[0] * Dv[0]
        filling_m12 = f0 * Gv[0] * Dv[1]
        filling_m13 = f0 * Gv[0] * Dv[2]
        filling_m21 = f0 * Gv[1] * Dv[0]
        filling_m22 = f0 * Gv[1] * Dv[1]
        filling_m23 = f0 * Gv[1] * Dv[2]
        filling_m31 = f0 * Gv[2] * Dv[0]
        filling_m32 = f0 * Gv[2] * Dv[1]
        filling_m33 = f0 * Gv[2] * Dv[2]

        # Compute filling for vector: sqrt(f_0(x,v)) * w_p * ( G^{-1} v )
        filling_v1 = sqrt(f0) * weight * Gv[0]
        filling_v2 = sqrt(f0) * weight * Gv[1]
        filling_v3 = sqrt(f0) * weight * Gv[2]

        # fill matrix and vector
        mvf.m_v_fill_b_v1_full(p, t1, t2, t3, indN1, indN2, indN3, eta1, eta2, eta3, mat11, mat12, mat13, mat21, mat22, mat23, mat31, mat32, mat33, filling_m11, filling_m12, filling_m13, filling_m21, filling_m22, filling_m23, filling_m31, filling_m32, filling_m33, vec1, vec2, vec3, filling_v1, filling_v2, filling_v3)

    #$ omp end parallel

    ierr = 0
