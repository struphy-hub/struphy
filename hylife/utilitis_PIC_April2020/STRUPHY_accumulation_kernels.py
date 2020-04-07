from pyccel.decorators import types
#import ..linear_algebra.core as linalg
#import ..geometry.mappings_analytical as mapping

import hylife.linear_algebra.core as linalg
import hylife.geometry.mappings_analytical as mapping


# ==============================================================================
@types('double[:]','int','double','int','double[:]','double[:]','double[:]')
def basis_funs(knots, degree, x, span, left, right, values):
    
    left [:]  = 0.
    right[:]  = 0.

    values[0] = 1.
    
    for j in range(degree):
        left [j] = x - knots[span - j]
        right[j] = knots[span + 1 + j] - x
        saved    = 0.
        for r in range(j + 1):
            temp      = values[r]/(right[r] + left[j - r])
            values[r] = saved + right[r]*temp
            saved     = left[j - r]*temp
        
        values[j + 1] = saved
        
        
        
# ==============================================================================
@types('double[:,:](order=F)','int[:]','int[:]','int[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','int','double[:]','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)')
def kernel_step1(particles, p0, nel, nbase, t0_1, t0_2, t0_3, t1_1, t1_2, t1_3, b_part, kind_map, params_map, mat12, mat13, mat23):
    
    from numpy import empty
    from numpy import zeros
    
    p0_1   = p0[0]
    p0_2   = p0[1]
    p0_3   = p0[2]
    
    p1_1   = p0_1 - 1
    p1_2   = p0_2 - 1
    p1_3   = p0_3 - 1
    
    nl1    = empty(p0_1,     dtype=float)
    nr1    = empty(p0_1,     dtype=float)
    nn1    = zeros(p0_1 + 1, dtype=float)
    
    nl2    = empty(p0_2,     dtype=float)
    nr2    = empty(p0_2,     dtype=float)
    nn2    = zeros(p0_2 + 1, dtype=float)
    
    nl3    = empty(p0_3,     dtype=float)
    nr3    = empty(p0_3,     dtype=float)
    nn3    = zeros(p0_3 + 1, dtype=float)
    
    dl1    = empty(p1_1    , dtype=float)
    dr1    = empty(p1_1    , dtype=float)
    dd1    = zeros(p1_1 + 1, dtype=float)
    
    dl2    = empty(p1_2    , dtype=float)
    dr2    = empty(p1_2    , dtype=float)
    dd2    = zeros(p1_2 + 1, dtype=float)
    
    dl3    = empty(p1_3    , dtype=float)
    dr3    = empty(p1_3    , dtype=float)
    dd3    = zeros(p1_3 + 1, dtype=float)
    
    b          = empty( 3    , dtype=float)
    
    b_prod     = zeros((3, 3), dtype=float, order='F')
    
    ginv       = empty((3, 3), dtype=float, order='F') 
    
    temp_mat1  = empty((3, 3), dtype=float, order='F')
    temp_mat2  = empty((3, 3), dtype=float, order='F')
    
    np         = len(particles[:, 0])
    
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    
    components = empty((3, 3), dtype=int, order='F')
    
    components[0, 0] = 11
    components[0, 1] = 12
    components[0, 2] = 13
    components[1, 0] = 21
    components[1, 1] = 22
    components[1, 2] = 23
    components[2, 0] = 31
    components[2, 1] = 32
    components[2, 2] = 33
    
    #$ omp parallel
    #$ omp do reduction ( + : mat12, mat13, mat23) private (ip, b, pos1, pos2, pos3, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, ie1, ie2, ie3, nl1, nr1, nn1, nl2, nr2, nn2, nl3, nr3, nn3, dl1, dr1, dd1, dl2, dr2, dd2, dl3, dr3, dd3, w, i, j, ginv, temp_mat1, temp_mat2, temp12, temp13, temp23, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, j1, j2, j3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
    for ip in range(np):

        b[0]    = b_part[ip, 0]
        b[1]    = b_part[ip, 1]
        b[2]    = b_part[ip, 2]
        
        pos1    = particles[ip, 0]
        pos2    = particles[ip, 1]
        pos3    = particles[ip, 2]
        
        span0_1 = int(pos1*nel[0]) + p0_1
        span0_2 = int(pos2*nel[1]) + p0_2
        span0_3 = int(pos3*nel[2]) + p0_3
        
        span1_1 = span0_1 - 1
        span1_2 = span0_2 - 1
        span1_3 = span0_3 - 1
        
        ie1 = span0_1 - p0_1
        ie2 = span0_2 - p0_2
        ie3 = span0_3 - p0_3
        
        basis_funs(t0_1, p0_1, pos1, span0_1, nl1, nr1, nn1)
        basis_funs(t0_2, p0_2, pos2, span0_2, nl2, nr2, nn2)
        basis_funs(t0_3, p0_3, pos3, span0_3, nl3, nr3, nn3)

        basis_funs(t1_1, p1_1, pos1, span1_1, dl1, dr1, dd1)
        basis_funs(t1_2, p1_2, pos2, span1_2, dl2, dr2, dd2)
        basis_funs(t1_3, p1_3, pos3, span1_3, dl3, dr3, dd3)
    
                    
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        
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
        for il1 in range(p1_1 + 1):
            i1  = (ie1 + il1)%nbase[3]
            bi1 = dd1[il1]*p0_1/(t1_1[i1 + p0_1] - t1_1[i1])
            for il2 in range(p0_2 + 1):
                i2  = (ie2 + il2)%nbase[1]
                bi2 = bi1 * nn2[il2]
                for il3 in range(p0_3 + 1):
                    i3  = (ie3 + il3)%nbase[2]
                    bi3 = bi2 * nn3[il3]
                    for jl1 in range(p0_1 + 1):
                        j1  = (ie1 + jl1)%nbase[0]
                        bj1 = bi3 * nn1[jl1]
                        
                        for jl2 in range(p1_2 + 1):
                            j2  = (ie2 + jl2)%nbase[4]
                            bj2 =  bj1 * dd2[jl2]*p0_2/(t1_2[j2 + p0_2] - t1_2[j2]) * temp12
                            for jl3 in range(p0_3 + 1):
                                j3  = (ie3 + jl3)%nbase[2]
                                bj3 = bj2 * nn3[jl3]
                                
                                mat12[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
                        for jl2 in range(p0_2 + 1):
                            j2  = (ie2 + jl2)%nbase[1]
                            bj2 =  bj1 * nn2[jl2] * temp13
                            for jl3 in range(p1_3 + 1):
                                j3  = (ie3 + jl3)%nbase[5]
                                bj3 = bj2 * dd3[jl3]*p0_3/(t1_3[j3 + p0_3] - t1_3[j3])
                                
                                mat13[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
        
        
        
        
        # add contribution to 23 component (NDN NND)
        for il1 in range(p0_1 + 1):
            i1  = (ie1 + il1)%nbase[0]
            bi1 = nn1[il1] * temp23
            for il2 in range(p1_2 + 1):
                i2  = (ie2 + il2)%nbase[4]
                bi2 = bi1 * dd2[il2]*p0_2/(t1_2[i2 + p0_2] - t1_2[i2])
                for il3 in range(p0_3 + 1):
                    i3  = (ie3 + il3)%nbase[2]
                    bi3 = bi2 * nn3[il3]
                    for jl1 in range(p0_1 + 1):
                        j1  = (ie1 + jl1)%nbase[0]
                        bj1 = bi3 * nn1[jl1]
                        for jl2 in range(p0_2 + 1):
                            j2  = (ie2 + jl2)%nbase[1]
                            bj2 =  bj1 * nn2[jl2]
                            for jl3 in range(p1_3 + 1):
                                j3  = (ie3 + jl3)%nbase[5]
                                bj3 = bj2 * dd3[jl3]*p0_3/(t1_3[j3 + p0_3] - t1_3[j3])
                                
                                mat23[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
                                
    #$ omp end do
    #$ omp end parallel   
    
    ierr = 0
    
    
    
    
# ==============================================================================
@types('double[:,:](order=F)','int[:]','int[:]','int[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:]','double[:,:](order=F)','int','double[:]','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:,:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)','double[:,:,:](order=F)')
def kernel_step3(particles, p0, nel, nbase, t0_1, t0_2, t0_3, t1_1, t1_2, t1_3, b_part, kind_map, params_map, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3):
    
    from numpy import empty
    from numpy import zeros
    
    p0_1 = p0[0]
    p0_2 = p0[1]
    p0_3 = p0[2]
    
    p1_1 = p0_1 - 1
    p1_2 = p0_2 - 1
    p1_3 = p0_3 - 1
    
    nl1  = empty(p0_1,     dtype=float)
    nr1  = empty(p0_1,     dtype=float)
    nn1  = zeros(p0_1 + 1, dtype=float)
    
    nl2  = empty(p0_2,     dtype=float)
    nr2  = empty(p0_2,     dtype=float)
    nn2  = zeros(p0_2 + 1, dtype=float)
    
    nl3  = empty(p0_3,     dtype=float)
    nr3  = empty(p0_3,     dtype=float)
    nn3  = zeros(p0_3 + 1, dtype=float)
    
    dl1  = empty(p1_1    , dtype=float)
    dr1  = empty(p1_1    , dtype=float)
    dd1  = zeros(p1_1 + 1, dtype=float)
    
    dl2  = empty(p1_2    , dtype=float)
    dr2  = empty(p1_2    , dtype=float)
    dd2  = zeros(p1_2 + 1, dtype=float)
    
    dl3  = empty(p1_3    , dtype=float)
    dr3  = empty(p1_3    , dtype=float)
    dd3  = zeros(p1_3 + 1, dtype=float)
    
    b            = empty( 3    , dtype=float)
    
    b_prod       = zeros((3, 3), dtype=float, order='F')
    b_prod_t     = zeros((3, 3), dtype=float, order='F')
    
    ginv         = empty((3, 3), dtype=float, order='F')
    dfinv        = empty((3, 3), dtype=float, order='F')
    
    temp_mat1    = empty((3, 3), dtype=float, order='F')
    temp_mat2    = empty((3, 3), dtype=float, order='F')
    
    temp_mat_vec = empty((3, 3), dtype=float, order='F')
    
    temp_vec     = empty( 3    , dtype=float)
    
    v            = empty( 3    , dtype=float)
    
    np           = len(particles[:, 0])
    
    mat11[:, :, :, :, :, :] = 0.
    mat12[:, :, :, :, :, :] = 0.
    mat13[:, :, :, :, :, :] = 0.
    mat22[:, :, :, :, :, :] = 0.
    mat23[:, :, :, :, :, :] = 0.
    mat33[:, :, :, :, :, :] = 0.
    
    vec1[:, :, :] = 0.
    vec2[:, :, :] = 0.
    vec3[:, :, :] = 0.
    
    components = empty((3, 3), dtype=int, order='F')
    
    components[0, 0] = 11
    components[0, 1] = 12
    components[0, 2] = 13
    components[1, 0] = 21
    components[1, 1] = 22
    components[1, 2] = 23
    components[2, 0] = 31
    components[2, 1] = 32
    components[2, 2] = 33
    
    #$ omp parallel
    #$ omp do reduction ( + : vec1, mat11, mat12, mat13, vec2, mat22, mat23, vec3, mat33) private (ip, b, pos1, pos2, pos3, span0_1, span0_2, span0_3, span1_1, span1_2, span1_3, ie1, ie2, ie3, nl1, nr1, nn1, nl2, nr2, nn2, nl3, nr3, nn3, dl1, dr1, dd1, dl2, dr2, dd2, dl3, dr3, dd3, v, w, i, j, ginv, dfinv, temp_mat1, temp_mat2, temp_mat_vec, temp_vec, b_prod_t, temp11, temp12, temp13, temp22, temp23, temp33, temp1, temp2, temp3, il1, il2, il3, jl1, jl2, jl3, i1, i2, i3, j1, j2, j3, bi1, bi2, bi3, bj1, bj2, bj3) firstprivate(b_prod)
    for ip in range(np):

        b[0]    = b_part[ip, 0]
        b[1]    = b_part[ip, 1]
        b[2]    = b_part[ip, 2]
        
        pos1    = particles[ip, 0]
        pos2    = particles[ip, 1]
        pos3    = particles[ip, 2]
        
        span0_1 = int(pos1*nel[0]) + p0_1
        span0_2 = int(pos2*nel[1]) + p0_2
        span0_3 = int(pos3*nel[2]) + p0_3
        
        span1_1 = span0_1 - 1
        span1_2 = span0_2 - 1
        span1_3 = span0_3 - 1
        
        ie1 = span0_1 - p0_1
        ie2 = span0_2 - p0_2
        ie3 = span0_3 - p0_3
        
        basis_funs(t0_1, p0_1, pos1, span0_1, nl1, nr1, nn1)
        basis_funs(t0_2, p0_2, pos2, span0_2, nl2, nr2, nn2)
        basis_funs(t0_3, p0_3, pos3, span0_3, nl3, nr3, nn3)

        basis_funs(t1_1, p1_1, pos1, span1_1, dl1, dr1, dd1)
        basis_funs(t1_2, p1_2, pos2, span1_2, dl2, dr2, dd2)
        basis_funs(t1_3, p1_3, pos3, span1_3, dl3, dr3, dd3)
    
                    
        b_prod[0, 1] = -b[2]
        b_prod[0, 2] =  b[1]

        b_prod[1, 0] =  b[2]
        b_prod[1, 2] = -b[0]

        b_prod[2, 0] = -b[1]
        b_prod[2, 1] =  b[0]
        
        v = particles[ip, 3:6]
        w = particles[ip, 6]
        
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
        for il1 in range(p1_1 + 1):
            i1  = (ie1 + il1)%nbase[3]
            bi1 = dd1[il1]*p0_1/(t1_1[i1 + p0_1] - t1_1[i1])
            for il2 in range(p0_2 + 1):
                i2  = (ie2 + il2)%nbase[1]
                bi2 = bi1 * nn2[il2]
                for il3 in range(p0_3 + 1):
                    i3  = (ie3 + il3)%nbase[2]
                    bi3 = bi2 * nn3[il3]
                    
                    vec1[i1, i2, i3] += bi3 * temp1 
                    
                    for jl1 in range(p1_1 + 1):
                        j1  = (ie1 + jl1)%nbase[3]
                        bj1 = bi3 * dd1[jl1]*p0_1/(t1_1[j1 + p0_1] - t1_1[j1]) * temp11
                        for jl2 in range(p0_2 + 1):
                            j2  = (ie2 + jl2)%nbase[1]
                            bj2 =  bj1 * nn2[jl2]
                            for jl3 in range(p0_3 + 1):
                                j3  = (ie3 + jl3)%nbase[2]
                                bj3 = bj2 * nn3[jl3]
                                
                                mat11[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
                    for jl1 in range(p0_1 + 1):
                        j1  = (ie1 + jl1)%nbase[0]
                        bj1 = bi3 * nn1[jl1] * temp12
                        for jl2 in range(p1_2 + 1):
                            j2  = (ie2 + jl2)%nbase[4]
                            bj2 =  bj1 * dd2[jl2]*p0_2/(t1_2[j2 + p0_2] - t1_2[j2])
                            for jl3 in range(p0_3 + 1):
                                j3  = (ie3 + jl3)%nbase[2]
                                bj3 = bj2 * nn3[jl3]
                                
                                mat12[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
                    for jl1 in range(p0_1 + 1):
                        j1  = (ie1 + jl1)%nbase[0]
                        bj1 = bi3 * nn1[jl1] * temp13
                        for jl2 in range(p0_2 + 1):
                            j2  = (ie2 + jl2)%nbase[1]
                            bj2 =  bj1 * nn2[jl2]
                            for jl3 in range(p1_3 + 1):
                                j3  = (ie3 + jl3)%nbase[5]
                                bj3 = bj2 * dd3[jl3]*p0_3/(t1_3[j3 + p0_3] - t1_3[j3])
                                
                                mat13[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
    
        # add contribution to 22 component (NDN NDN) and 23 component (NDN NND)
        for il1 in range(p0_1 + 1):
            i1  = (ie1 + il1)%nbase[0]
            bi1 = nn1[il1]
            for il2 in range(p1_2 + 1):
                i2  = (ie2 + il2)%nbase[4]
                bi2 = bi1 * dd2[il2]*p0_2/(t1_2[i2 + p0_2] - t1_2[i2])
                for il3 in range(p0_3 + 1):
                    i3  = (ie3 + il3)%nbase[2]
                    bi3 = bi2 * nn3[il3]
                    vec2[i1, i2, i3] += bi3 * temp2 
                    for jl1 in range(p0_1 + 1):
                        j1  = (ie1 + jl1)%nbase[0]
                        bj1 = bi3 * nn1[jl1]
                        
                        for jl2 in range(p1_2 + 1):
                            j2  = (ie2 + jl2)%nbase[4]
                            bj2 =  bj1 * dd2[jl2]*p0_2/(t1_2[j2 + p0_2] - t1_2[j2]) * temp22
                            for jl3 in range(p0_3 + 1):
                                j3  = (ie3 + jl3)%nbase[2]
                                bj3 = bj2 * nn3[jl3]
                                
                                mat22[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
                        
                        for jl2 in range(p0_2 + 1):
                            j2  = (ie2 + jl2)%nbase[1]
                            bj2 =  bj1 * nn2[jl2] * temp23
                            for jl3 in range(p1_3 + 1):
                                j3  = (ie3 + jl3)%nbase[5]
                                bj3 = bj2 * dd3[jl3]*p0_3/(t1_3[j3 + p0_3] - t1_3[j3])
                                
                                mat23[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
                                
        # add contribution to 33 component (NND NND)
        for il1 in range(p0_1 + 1):
            i1  = (ie1 + il1)%nbase[0]
            bi1 = nn1[il1]
            for il2 in range(p0_2 + 1):
                i2  = (ie2 + il2)%nbase[1]
                bi2 = bi1 * nn2[il2]
                for il3 in range(p1_3 + 1):
                    i3  = (ie3 + il3)%nbase[5]
                    bi3 = bi2 * dd3[il3]*p0_3/(t1_3[i3 + p0_3] - t1_3[i3])
                    vec3[i1, i2, i3] += bi3 * temp3 
                    for jl1 in range(p0_1 + 1):
                        j1  = (ie1 + jl1)%nbase[0]
                        bj1 = bi3 * nn1[jl1] * temp33
                        for jl2 in range(p0_2 + 1):
                            j2  = (ie2 + jl2)%nbase[1]
                            bj2 =  bj1 * nn2[jl2]
                            for jl3 in range(p1_3 + 1):
                                j3  = (ie3 + jl3)%nbase[5]
                                bj3 = bj2 * dd3[jl3]*p0_3/(t1_3[j3 + p0_3] - t1_3[j3])
                                
                                mat33[i1, i2, i3, p0_1 + jl1 - il1, p0_2 + jl2 - il2, p0_3 + jl3 - il3] += bj3
                                
                                                       
    #$ omp end do
    #$ omp end parallel   
    
    ierr = 0
