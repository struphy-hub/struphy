# ================= 1d =================================
def kernel_1d_mat(el_loc_indices1 : 'int[:]', pi1 : int, pj1 : int,  periodic1 : int, starts1 : int, pads1 : int, nq1 : int, w1 : 'float[:,:]', bi1 : 'float[:,:,:,:]', bj1 : 'float[:,:,:,:]', mat_fun : 'float[:]', data : 'float[:,:]'):
    """
    Performs the integration Lambda_i * mat_fun(eta1) * Lambda_j for the basis functions on the calling process.
    
    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """
    nel = len(el_loc_indices1)

    for iel1 in range(nel):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]

        for il1 in range(pi1 + 1):
            for jl1 in range(pj1 + 1):

                value = 0.

                for q1 in range(nq1):
                    value += w1[iel1, q1] * bi1[iel1, il1, 0, q1] * bj1[iel1, jl1, 0, q1] * mat_fun[iel1*nq1 + q1]

                data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), pads1 + jl1 - il1] += value
                
def kernel_1d_vec(el_loc_indices1 : 'int[:]', pi1 : int,  periodic1 : int, starts1 : int, pads1 : int, nq1 : int, w1 : 'float[:,:]', bi1 : 'float[:,:,:,:]', mat_fun : 'float[:]', data : 'float[:]'):
    """
    Performs the integration Lambda_i * mat_fun(eta1) for the basis functions on the calling process.
    
    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """
    
    nel = len(el_loc_indices1)

    for iel1 in range(nel):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]

        for il1 in range(pi1 + 1):

            value = 0.

            for q1 in range(nq1):
                value += w1[iel1, q1] * bi1[iel1, il1, 0, q1] * mat_fun[iel1*nq1 + q1]

            data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1)] += value
            
def kernel_1d_eval(el_loc_indices1 : 'int[:]', pi1 : int,  periodic1 : int, starts1 : int, pads1 : int, nq1 : int, bi1 : 'float[:,:,:,:]', coeffs_data : 'float[:]', values : 'float[:]'):
    """
    Evaluates sum_i [ coeffs_i * Lambda_i(quad_eta1) ] for all quadrature points on the calling process.
    
    The results are written into values.
    """
    
    values[:] = 0.
    
    nel = len(el_loc_indices1)

    for iel1 in range(nel):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]

        for il1 in range(pi1 + 1):

            value = 0.

            for q1 in range(nq1):
                values[iel1*nq1 + q1] += coeffs_data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1)] * bi1[iel1, il1, 0, q1]
                
# ================= 2d =================================                
def kernel_2d_mat(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', pi1 : int, pi2 : int, pj1 : int, pj2 : int,  periodic1 : int,  periodic2 : int, starts1 : int, starts2 : int, pads1 : int, pads2 : int, nq1 : int, nq2 : int, w1 : 'float[:,:]', w2 : 'float[:,:]', bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bj1 : 'float[:,:,:,:]', bj2 : 'float[:,:,:,:]', mat_fun : 'float[:,:]', data : 'float[:,:,:,:]'):
    """
    Performs the integration Lambda_ij * mat_fun(eta1, eta2) * Lambda_kl for the basis functions on the calling process.
    
    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]

            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):
                    for jl1 in range(pj1 + 1):
                        for jl2 in range(pj2 + 1):

                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):

                                    wvol = w1[iel1, q1] * w2[iel2, q2] * mat_fun[iel1*nq1 + q1, iel2*nq2 + q2]
                                    bi = bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]
                                    bj = bj1[iel1, jl1, 0, q1] * bj2[iel2, jl2, 0, q2]

                                    value += wvol * bi * bj

                            data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2), pads1 + jl1 - il1, pads2 + jl2 - il2] += value

def kernel_2d_vec(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', pi1 : int, pi2 : int,  periodic1 : int,  periodic2 : int, starts1 : int, starts2 : int, pads1 : int, pads2 : int, nq1 : int, nq2 : int, w1 : 'float[:,:]', w2 : 'float[:,:]', bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', mat_fun : 'float[:,:]', data : 'float[:,:]'):
    """
    Performs the integration Lambda_ij * mat_fun(eta1, eta2) for the basis functions on the calling process.
    
    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]

            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):

                    value = 0.

                    for q1 in range(nq1):
                        for q2 in range(nq2):

                            wvol = w1[iel1, q1] * w2[iel2, q2] * mat_fun[iel1*nq1 + q1, iel2*nq2 + q2]
                            bi = bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]

                            value += wvol * bi

                    data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2)] += value

def kernel_2d_eval(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', pi1 : int, pi2 : int,  periodic1 : int,  periodic2 : int, starts1 : int, starts2 : int, pads1 : int, pads2 : int, nq1 : int, nq2 : int, bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', coeffs_data : 'float[:,:]', values : 'float[:,:]'):
    """
    Evaluates sum_ij [ coeffs_ij * Lambda_ij(quad_eta1, quad_eta2) ] for all quadrature points on the calling process.
    
    The results are written into values.
    """
    
    values[:, :] = 0.
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]

            for il1 in range(pi1 + 1):
                for il2 in range(pi2 + 1):

                    for q1 in range(nq1):
                        for q2 in range(nq2):

                            values[iel1*nq1 + q1, iel2*nq2 + q2] += coeffs_data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2)] * bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]
                    
# ================= 3d =================================                
def kernel_3d_mat(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', el_loc_indices3 : 'int[:]', pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int,  periodic1 : int,  periodic2 : int,  periodic3 : int, starts1 : int, starts2 : int, starts3 : int, pads1 : int, pads2 : int, pads3 : int, nq1 : int, nq2 : int, nq3 : int, w1 : 'float[:,:]', w2 : 'float[:,:]', w3 : 'float[:,:]', bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bi3 : 'float[:,:,:,:]', bj1 : 'float[:,:,:,:]', bj2 : 'float[:,:,:,:]', bj3 : 'float[:,:,:,:]', mat_fun : 'float[:,:,:]', data : 'float[:,:,:,:,:,:]'):
    """
    Performs the integration Lambda_ijk * mat_fun(eta1, eta2, eta3) * Lambda_lmn for the basis functions on the calling process.
    
    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)
    nel3 = len(el_loc_indices3)

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]
                
            for iel3 in range(nel3):

                if periodic3 == 1:
                    ie3 = iel3
                else:    
                    ie3 = el_loc_indices3[iel3]

                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):
                            for jl1 in range(pj1 + 1):
                                for jl2 in range(pj2 + 1):
                                    for jl3 in range(pj3 + 1):

                                        value = 0.

                                        for q1 in range(nq1):
                                            for q2 in range(nq2):
                                                for q3 in range(nq3):
                                                    
                                                    wvol = w1[iel1, q1] * w2[iel2, q2] * w3[iel3, q3] * mat_fun[iel1*nq1 + q1, iel2*nq2 + q2, iel3*nq3 + q3]
                                                    bi = bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3]
                                                    bj = bj1[iel1, jl1, 0, q1] * bj2[iel2, jl2, 0, q2] * bj3[iel3, jl3, 0, q3]
                                                    
                                                    value += wvol * bi * bj

                                        data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2), ie3 + il3 - (1 - periodic3)*(starts3 - pads3), pads1 + jl1 - il1, pads2 + jl2 - il2, pads3 + jl3 - il3] += value 

def kernel_3d_vec(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', el_loc_indices3 : 'int[:]', pi1 : int, pi2 : int, pi3 : int,  periodic1 : int,  periodic2 : int,  periodic3 : int, starts1 : int, starts2 : int, starts3 : int, pads1 : int, pads2 : int, pads3 : int, nq1 : int, nq2 : int, nq3 : int, w1 : 'float[:,:]', w2 : 'float[:,:]', w3 : 'float[:,:]', bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bi3 : 'float[:,:,:,:]', mat_fun : 'float[:,:,:]', data : 'float[:,:,:]'):
    """
    Performs the integration Lambda_ijk * mat_fun(eta1, eta2, eta3) for the basis functions on the calling process.
    
    The results are written into data (attention: data is NOT set to zero first, but the results are added to data).
    """
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)
    nel3 = len(el_loc_indices3)

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]
                
            for iel3 in range(nel3):

                if periodic3 == 1:
                    ie3 = iel3
                else:    
                    ie3 = el_loc_indices3[iel3]

                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):

                            value = 0.

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):

                                        wvol = w1[iel1, q1] * w2[iel2, q2] * w3[iel3, q3] * mat_fun[iel1*nq1 + q1, iel2*nq2 + q2, iel3*nq3 + q3]
                                        bi = bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3]

                                        value += wvol * bi

                            data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2), ie3 + il3 - (1 - periodic3)*(starts3 - pads3)] += value                                        
                                        
def kernel_3d_eval(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', el_loc_indices3 : 'int[:]', pi1 : int, pi2 : int, pi3 : int,  periodic1 : int,  periodic2 : int,  periodic3 : int, starts1 : int, starts2 : int, starts3 : int, pads1 : int, pads2 : int, pads3 : int, nq1 : int, nq2 : int, nq3 : int, bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bi3 : 'float[:,:,:,:]', coeffs_data : 'float[:,:,:]', values : 'float[:,:,:]'):
    """
    Evaluates sum_ijk [ coeffs_ijk * Lambda_ijk(quad_eta1, quad_eta2, quad_eta3) ] for all quadrature points on the calling process.
    
    The results are written into values.
    """
    
    values[:, :, :] = 0.
    
    nel1 = len(el_loc_indices1)
    nel2 = len(el_loc_indices2)
    nel3 = len(el_loc_indices3)

    for iel1 in range(nel1):

        if periodic1 == 1:
            ie1 = iel1
        else:    
            ie1 = el_loc_indices1[iel1]
            
        for iel2 in range(nel2):

            if periodic2 == 1:
                ie2 = iel2
            else:    
                ie2 = el_loc_indices2[iel2]
                
            for iel3 in range(nel3):

                if periodic3 == 1:
                    ie3 = iel3
                else:    
                    ie3 = el_loc_indices3[iel3]

                for il1 in range(pi1 + 1):
                    for il2 in range(pi2 + 1):
                        for il3 in range(pi3 + 1):

                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        
                                        values[iel1*nq1 + q1, iel2*nq2 + q2, iel3*nq3 + q3] += coeffs_data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2), ie3 + il3 - (1 - periodic3)*(starts3 - pads3)] * bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3]