# ================= 1d =================================
def kernel_1d(el_loc_indices : 'int[:]', pi : int, pj : int,  periodic : int, starts : int, pads : int, nq : int, w : 'float[:,:]', bi : 'float[:,:,:,:]', bj : 'float[:,:,:,:]', mat_fun : 'float[:,:]', data : 'float[:,:]'):
    
    nel = len(el_loc_indices)

    for iel in range(nel):

        if periodic == 1:
            ie = iel
        else:    
            ie = el_loc_indices[iel]

        for il in range(pi + 1):
            for jl in range(pj + 1):

                value = 0.

                for q in range(nq):
                    value += w[iel, q] * bi[iel, il, 0, q] * bj[iel, jl, 0, q] * mat_fun[iel, q]

                data[ie + il - (1 - periodic)*(starts - pads), pads + jl - il] += value
                
                
                
# ================= 2d =================================                
def kernel_2d(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', pi1 : int, pi2 : int, pj1 : int, pj2 : int,  periodic1 : int,  periodic2 : int, starts1 : int, starts2 : int, pads1 : int, pads2 : int, nq1 : int, nq2 : int, w1 : 'float[:,:]', w2 : 'float[:,:]', bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bj1 : 'float[:,:,:,:]', bj2 : 'float[:,:,:,:]', mat_fun : 'float[:,:,:,:]', data : 'float[:,:,:,:]'):
    
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

                                    wvol = w1[iel1, q1] * w2[iel2, q2] * mat_fun[iel1, q1, iel2, q2]
                                    bi = bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2]
                                    bj = bj1[iel1, jl1, 0, q1] * bj2[iel2, jl2, 0, q2]

                                    value += wvol * bi * bj

                            data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2), pads1 + jl1 - il1, pads2 + jl2 - il2] += value

                

# ================= 3d =================================                
def kernel_3d(el_loc_indices1 : 'int[:]', el_loc_indices2 : 'int[:]', el_loc_indices3 : 'int[:]', pi1 : int, pi2 : int, pi3 : int, pj1 : int, pj2 : int, pj3 : int,  periodic1 : int,  periodic2 : int,  periodic3 : int, starts1 : int, starts2 : int, starts3 : int, pads1 : int, pads2 : int, pads3 : int, nq1 : int, nq2 : int, nq3 : int, w1 : 'float[:,:]', w2 : 'float[:,:]', w3 : 'float[:,:]', bi1 : 'float[:,:,:,:]', bi2 : 'float[:,:,:,:]', bi3 : 'float[:,:,:,:]', bj1 : 'float[:,:,:,:]', bj2 : 'float[:,:,:,:]', bj3 : 'float[:,:,:,:]', mat_fun : 'float[:,:,:,:,:,:]', data : 'float[:,:,:,:,:,:]'):
    
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
                                                    
                                                    wvol = w1[iel1, q1] * w2[iel2, q2] * w3[iel3, q3] * mat_fun[iel1, q1, iel2, q2, iel3, q3]
                                                    bi = bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * bi3[iel3, il3, 0, q3]
                                                    bj = bj1[iel1, jl1, 0, q1] * bj2[iel2, jl2, 0, q2] * bj3[iel3, jl3, 0, q3]
                                                    
                                                    value += wvol * bi * bj

                                        data[ie1 + il1 - (1 - periodic1)*(starts1 - pads1), ie2 + il2 - (1 - periodic2)*(starts2 - pads2), ie3 + il3 - (1 - periodic3)*(starts3 - pads3), pads1 + jl1 - il1, pads2 + jl2 - il2, pads3 + jl3 - il3] += value