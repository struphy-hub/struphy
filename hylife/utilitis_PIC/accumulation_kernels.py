# import pyccel decorators
from pyccel.decorators import types

# import modules for mapping related quantities
import hylife.geometry.mappings_analytical as mapping

# import modules for B-spline evaluation
import hylife.utilitis_FEEC.bsplines_kernels as bsp
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eva

# import module for matrix-matrix and matrix-vector multiplications
import hylife.linear_algebra.core as linalg


# ==============================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:]','int','double[:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def kernel_step1(particles, t1, t2, t3, p, nel, nbase_n, nbase_d, np, b_part, kind_map, params_map, mat12, mat13, mat23):
    
    from numpy import empty, zeros
    
    # polynomial degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # all non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3  = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    # non-vanishing N-splines
    bn1 = empty( pn1 + 1          , dtype=float)
    bn2 = empty( pn2 + 1          , dtype=float)
    bn3 = empty( pn3 + 1          , dtype=float)
    
    # non-vanishing D-splines
    d1  = empty( pn1              , dtype=float)
    d2  = empty( pn2              , dtype=float)
    d3  = empty( pn3              , dtype=float)
    
    bd1 = empty( pd1 + 1          , dtype=float)
    bd2 = empty( pd2 + 1          , dtype=float)
    bd3 = empty( pd3 + 1          , dtype=float)
    
    # magnetic field at particle positions
    b          = empty( 3    , dtype=float)
    b_prod     = zeros((3, 3), dtype=float)
    
    # other needed quantities
    ginv       = empty((3, 3), dtype=float) 
    
    temp_mat1  = empty((3, 3), dtype=float)
    temp_mat2  = empty((3, 3), dtype=float)
    
    components = empty((3, 3), dtype=int)
    
    components[0, 0] = 11
    components[0, 1] = 12
    components[0, 2] = 13
    components[1, 0] = 21
    components[1, 1] = 22
    components[1, 2] = 23
    components[2, 0] = 31
    components[2, 1] = 32
    components[2, 2] = 33
    
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : mat12, mat13, mat23) private (ip, b, pos1, pos2, pos3, span1, span2, span3, ie1, ie2, ie3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, w, i, j, ginv, temp_mat1, temp_mat2, temp12, temp13, temp23, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
    for ip in range(np):

        # magnetic field at particle positions
        b[0]         = b_part[ip, 0]
        b[1]         = b_part[ip, 1]
        b[2]         = b_part[ip, 2]
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
          
        # particle positions, span indices and element indices
        pos1  = particles[ip, 0]
        pos2  = particles[ip, 1]
        pos3  = particles[ip, 2]
        
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        ie1   = span1 - pn1
        ie2   = span2 - pn2
        ie3   = span3 - pn3
        
        # evaluation of basis functions at particle positions
        bsp.basis_funs_all(t1, pn1, pos1, span1, b1, d1)
        bsp.basis_funs_all(t2, pn2, pos2, span2, b2, d2)
        bsp.basis_funs_all(t3, pn3, pos3, span3, b3, d3)
        
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
        
        # particle weight
        w = particles[ip, 6]
        
        # evaluate inverse metric tensor
        for i in range(3):
            for j in range(3):
                ginv[i, j] = mapping.g_inv(pos1, pos2, pos3, kind_map, params_map, components[i, j]) 
        
        
        # perform matrix-matrix multiplications
        linalg.matrix_matrix(ginv, b_prod, temp_mat1)
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        
        temp12 = w * temp_mat2[0, 1]
        temp13 = w * temp_mat2[0, 2]
        temp23 = w * temp_mat2[1, 2]
        
        
        # add contribution to 12 component (DNN NDN) and 13 component (DNN NND)
        for il1 in range(pd1 + 1):
            i1  = (ie1 + il1)%nbase_d[0]
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        
                        for jl2 in range(pd2 + 1):
                            bj2 =  bj1 * bd2[jl2] * temp12
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2] * temp13
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
        
        
        
        
        # add contribution to 23 component (NDN NND)
        for il1 in range(pn1 + 1):
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1] * temp23
            for il2 in range(pd2 + 1):
                i2  = (ie2 + il2)%nbase_d[1]
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                                
    #$ omp end do
    #$ omp end parallel   
    
    ierr = 0
    
    
    
# ==============================================================================
@types('double[:,:]','double[:]','double[:]','double[:]','int[:]','int[:]','int[:]','int[:]','int','double[:,:]','int','double[:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]','double[:,:,:]','double[:,:,:]','double[:,:,:]')
def kernel_step3(particles, t1, t2, t3, p, nel, nbase_n, nbase_d, np, b_part, kind_map, params_map, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3):
    
    from numpy import empty, zeros
    
    # polynomial degrees
    pn1 = p[0]
    pn2 = p[1]
    pn3 = p[2]
    
    pd1 = pn1 - 1
    pd2 = pn2 - 1
    pd3 = pn3 - 1
    
    # all non-vanishing basis functions up tp degree p
    b1  = empty((pn1 + 1, pn1 + 1), dtype=float)
    b2  = empty((pn2 + 1, pn2 + 1), dtype=float)
    b3  = empty((pn3 + 1, pn3 + 1), dtype=float)
    
    # non-vanishing N-splines
    bn1 = empty( pn1 + 1          , dtype=float)
    bn2 = empty( pn2 + 1          , dtype=float)
    bn3 = empty( pn3 + 1          , dtype=float)
    
    # non-vanishing D-splines
    d1  = empty( pn1              , dtype=float)
    d2  = empty( pn2              , dtype=float)
    d3  = empty( pn3              , dtype=float)
    
    bd1 = empty( pd1 + 1          , dtype=float)
    bd2 = empty( pd2 + 1          , dtype=float)
    bd3 = empty( pd3 + 1          , dtype=float)
    
    # magnetic field at particle positions
    b          = empty( 3    , dtype=float)
    b_prod     = zeros((3, 3), dtype=float)
    b_prod_t   = zeros((3, 3), dtype=float)
    
    # other needed quantities
    ginv         = empty((3, 3), dtype=float)
    dfinv        = empty((3, 3), dtype=float)
    
    temp_mat1    = empty((3, 3), dtype=float)
    temp_mat2    = empty((3, 3), dtype=float)
    
    temp_mat_vec = empty((3, 3), dtype=float)
    
    temp_vec     = empty( 3    , dtype=float)
    
    v            = empty( 3    , dtype=float)
    
    components   = empty((3, 3), dtype=int)
    
    components[0, 0] = 11
    components[0, 1] = 12
    components[0, 2] = 13
    components[1, 0] = 21
    components[1, 1] = 22
    components[1, 2] = 23
    components[2, 0] = 31
    components[2, 1] = 32
    components[2, 2] = 33
    
    mat11[:, :, :, :, :, :] = 0.
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat22[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    mat33[:, :, :, :, :, :] = 0.
    
    vec1[:,:,:] = 0.
    vec2[:,:,:] = 0.
    vec3[:,:,:] = 0.
    
    
    #$ omp parallel
    #$ omp do reduction ( + : mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3) private (ip, b, b_prod_t, pos1, pos2, pos3, span1, span2, span3, ie1, ie2, ie3, b1, b2, b3, d1, d2, d3, bn1, bn2, bn3, bd1, bd2, bd3, w, v, i, j, ginv, dfinv, temp_mat1, temp_mat2, temp_mat_vec, temp_vec, temp11, temp12, temp13, temp22, temp23, temp33, temp1, temp2, temp3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
    for ip in range(np):

        # magnetic field at particle positions
        b[0]         = b_part[ip, 0]
        b[1]         = b_part[ip, 1]
        b[2]         = b_part[ip, 2]
        
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
          
        # particle positions, span indices and element indices
        pos1  = particles[ip, 0]
        pos2  = particles[ip, 1]
        pos3  = particles[ip, 2]
        
        span1 = int(pos1*nel[0]) + pn1
        span2 = int(pos2*nel[1]) + pn2
        span3 = int(pos3*nel[2]) + pn3
        
        ie1   = span1 - pn1
        ie2   = span2 - pn2
        ie3   = span3 - pn3
        
        # evaluation of basis functions at particle positions
        bsp.basis_funs_all(t1, pn1, pos1, span1, b1, d1)
        bsp.basis_funs_all(t2, pn2, pos2, span2, b2, d2)
        bsp.basis_funs_all(t3, pn3, pos3, span3, b3, d3)
        
        bn1[:] = b1[pn1, :]
        bn2[:] = b2[pn2, :]
        bn3[:] = b3[pn3, :]
        
        bd1[:] = b1[pd1, :pn1] * d1[:]
        bd2[:] = b2[pd2, :pn2] * d2[:]
        bd3[:] = b3[pd3, :pn3] * d3[:]
    
        # particle weight and velocity
        w    = particles[ip, 6]
        v[:] = particles[ip, 3:6]
        
        
        # evaluate inverse metric tensor
        for i in range(3):
            for j in range(3):
                ginv[i, j] = mapping.g_inv(pos1, pos2, pos3, kind_map, params_map, components[i, j]) 
                
        # evaluate inverse Jacobian matrix
        for i in range(3):
            for j in range(3):
                dfinv[i, j] = mapping.df_inv(pos1, pos2, pos3, kind_map, params_map, components[i, j]) 
        
        
        # perform matrix-matrix multiplications
        linalg.matrix_matrix(ginv, b_prod, temp_mat1)
        linalg.matrix_matrix(temp_mat1, dfinv, temp_mat_vec)
        linalg.matrix_vector(temp_mat_vec, v, temp_vec)
        
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        linalg.transpose(b_prod, b_prod_t)
        linalg.matrix_matrix(temp_mat2, b_prod_t, temp_mat1)
        linalg.matrix_matrix(temp_mat1, ginv, temp_mat2)
        
        temp11 = w * temp_mat2[0, 0]
        temp12 = w * temp_mat2[0, 1]
        temp13 = w * temp_mat2[0, 2]
        temp22 = w * temp_mat2[1, 1]
        temp23 = w * temp_mat2[1, 2]
        temp33 = w * temp_mat2[2, 2]
        
        temp1  = w * temp_vec[0]
        temp2  = w * temp_vec[1]
        temp3  = w * temp_vec[2]
        
        
        # add contribution to 11 component (DNN DNN), 12 component (DNN NDN) and 13 component (DNN NND)
        for il1 in range(pd1 + 1):
            i1  = (ie1 + il1)%nbase_d[0]
            bi1 = bd1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    
                    vec1[i1, i2, i3] += bi3 * temp1 
                    
                    for jl1 in range(pd1 + 1):
                        bj1 = bi3 * bd1[jl1] * temp11
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat11[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1] * temp12
                        for jl2 in range(pd2 + 1):
                            bj2 =  bj1 * bd2[jl2]
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat12[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1] * temp13
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat13[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                                
                                
        # add contribution to 22 component (NDN NDN) and 23 component (NDN NND)
        for il1 in range(pn1 + 1):
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1]
            for il2 in range(pd2 + 1):
                i2  = (ie2 + il2)%nbase_d[1]
                bi2 = bi1 * bd2[il2]
                for il3 in range(pn3 + 1):
                    i3  = (ie3 + il3)%nbase_n[2]
                    bi3 = bi2 * bn3[il3]
                    
                    vec2[i1, i2, i3] += bi3 * temp2 
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1]
                        
                        for jl2 in range(pd2 + 1):
                            bj2 =  bj1 * bd2[jl2] * temp22
                            for jl3 in range(pn3 + 1):
                                bj3 = bj2 * bn3[jl3]
                                
                                mat22[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                   
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2] * temp23
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat23[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                       
        
        # add contribution to 33 component (NND NND)
        for il1 in range(pn1 + 1):
            i1  = (ie1 + il1)%nbase_n[0]
            bi1 = bn1[il1]
            for il2 in range(pn2 + 1):
                i2  = (ie2 + il2)%nbase_n[1]
                bi2 = bi1 * bn2[il2]
                for il3 in range(pd3 + 1):
                    i3  = (ie3 + il3)%nbase_d[2]
                    bi3 = bi2 * bd3[il3]
                    
                    vec3[i1, i2, i3] += bi3 * temp3 
                    
                    for jl1 in range(pn1 + 1):
                        bj1 = bi3 * bn1[jl1] * temp33
                        for jl2 in range(pn2 + 1):
                            bj2 =  bj1 * bn2[jl2]
                            for jl3 in range(pd3 + 1):
                                bj3 = bj2 * bd3[jl3]
                                
                                mat33[i1, i2, i3, pn1 + jl1 - il1, pn2 + jl2 - il2, pn3 + jl3 - il3] += bj3
                                
                                
    #$ omp end do
    #$ omp end parallel   
    
    ierr = 0