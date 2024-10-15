from pyccel.decorators import types

# ========================================================
@types('int[:]','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:,:,:,:,:]','double[:,:,:]')
def kernel_pi0(n, n_int, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_int_ind1, x_int_ind2, x_int_ind3, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, basis)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_int[0]):
                    for j2 in range(n_int[1]):
                        for j3 in range(n_int[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        basis = bs1[x_int_ind1[i1, j1], k1] * bs2[x_int_ind2[i2, j2], k2] * bs3[x_int_ind3[i3, j3], k3]
                                        
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * basis * mat_eq[x_int_ind1[i1, j1], x_int_ind2[i2, j2], x_int_ind3[i3, j3]]
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        
# ========================================================
@types('int[:]','int','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:]','double[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:]')
def kernel_pi1_1(n, n_quad1, n_inthis, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, x_int_ind2, x_int_ind3, wts1, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, f_int, q1)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_inthis[0]):
                    for j2 in range(n_inthis[1]):
                        for j3 in range(n_inthis[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        f_int = 0.
                                        
                                        for q1 in range(n_quad1):
                                            f_int += wts1[x_his_ind1[i1, j1], q1] * bs1[x_his_ind1[i1, j1], q1, k1] * bs2[x_int_ind2[i2, j2], k2] * bs3[x_int_ind3[i3, j3], k3] * mat_eq[x_his_ind1[i1, j1], q1, x_int_ind2[i2, j2], x_int_ind3[i3, j3]]
                                                
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * f_int
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        
                                        
                                        
# ========================================================
@types('int[:]','int','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:,:]','double[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:]')
def kernel_pi1_2(n, n_quad2, n_inthis, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_int_ind1, x_his_ind2, x_int_ind3, wts2, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, f_int, q2)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_inthis[0]):
                    for j2 in range(n_inthis[1]):
                        for j3 in range(n_inthis[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        f_int = 0.
                                        
                                        for q2 in range(n_quad2):
                                            f_int += wts2[x_his_ind2[i2, j2], q2] * bs1[x_int_ind1[i1, j1], k1] * bs2[x_his_ind2[i2, j2], q2, k2] * bs3[x_int_ind3[i3, j3], k3] * mat_eq[x_int_ind1[i1, j1], x_his_ind2[i2, j2], q2, x_int_ind3[i3, j3]]
                                                
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * f_int
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        
# ========================================================
@types('int[:]','int','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:]','double[:,:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:]')
def kernel_pi1_3(n, n_quad3, n_inthis, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_int_ind1, x_int_ind2, x_his_ind3, wts3, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, f_int, q3)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_inthis[0]):
                    for j2 in range(n_inthis[1]):
                        for j3 in range(n_inthis[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        f_int = 0.
                                        
                                        for q3 in range(n_quad3):
                                            f_int += wts3[x_his_ind3[i3, j3], q3] * bs1[x_int_ind1[i1, j1], k1] * bs2[x_int_ind2[i2, j2], k2] * bs3[x_his_ind3[i3, j3], q3, k3] * mat_eq[x_int_ind1[i1, j1], x_int_ind2[i2, j2], x_his_ind3[i3, j3], q3]
                                                
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * f_int
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        
# ========================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:]','double[:,:,:]','double[:,:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:]')
def kernel_pi2_1(n, n_quad, n_inthis, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_int_ind1, x_his_ind2, x_his_ind3, wts2, wts3, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, f_int, q2, q3, wvol)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_inthis[0]):
                    for j2 in range(n_inthis[1]):
                        for j3 in range(n_inthis[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        f_int = 0.
                                        
                                        for q2 in range(n_quad[0]):
                                            for q3 in range(n_quad[1]):
                                                wvol = wts2[x_his_ind2[i2, j2], q2] * wts3[x_his_ind3[i3, j3], q3]
                                                f_int += wvol * bs1[x_int_ind1[i1, j1], k1] * bs2[x_his_ind2[i2, j2], q2, k2] * bs3[x_his_ind3[i3, j3], q3, k3] * mat_eq[x_int_ind1[i1, j1], x_his_ind2[i2, j2], q2, x_his_ind3[i3, j3], q3]
                                                
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * f_int
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        
                                        
# ========================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:]','double[:,:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:]')
def kernel_pi2_2(n, n_quad, n_inthis, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, x_int_ind2, x_his_ind3, wts1, wts3, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, f_int, q1, q3, wvol)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_inthis[0]):
                    for j2 in range(n_inthis[1]):
                        for j3 in range(n_inthis[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        f_int = 0.
                                        
                                        for q1 in range(n_quad[0]):
                                            for q3 in range(n_quad[1]):
                                                wvol = wts1[x_his_ind1[i1, j1], q1] * wts3[x_his_ind3[i3, j3], q3]
                                                f_int += wvol * bs1[x_his_ind1[i1, j1], q1, k1] * bs2[x_int_ind2[i2, j2], k2] * bs3[x_his_ind3[i3, j3], q3, k3] * mat_eq[x_his_ind1[i1, j1], q1, x_int_ind2[i2, j2], x_his_ind3[i3, j3], q3]
                                                
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * f_int
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        
                                        
# ========================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:]')
def kernel_pi2_3(n, n_quad, n_inthis, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, x_his_ind2, x_int_ind3, wts1, wts2, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, f_int, q1, q2, wvol)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_inthis[0]):
                    for j2 in range(n_inthis[1]):
                        for j3 in range(n_inthis[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        f_int = 0.
                                        
                                        for q1 in range(n_quad[0]):
                                            for q2 in range(n_quad[1]):
                                                wvol = wts1[x_his_ind1[i1, j1], q1] * wts2[x_his_ind2[i2, j2], q2]
                                                f_int += wvol * bs1[x_his_ind1[i1, j1], q1, k1] * bs2[x_his_ind2[i2, j2], q2, k2] * bs3[x_int_ind3[i3, j3], k3] * mat_eq[x_his_ind1[i1, j1], q1, x_his_ind2[i2, j2], q2, x_int_ind3[i3, j3]]
                                                
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * f_int
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0
                                        
                                        
# ========================================================
@types('int[:]','int[:]','int[:]','int[:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','int[:]','int[:]','int[:]','double[:,:,:]','double[:,:,:]','double[:,:,:]','int[:,:]','int[:,:]','int[:,:]','double[:,:]','double[:,:]','double[:,:]','double[:,:,:,:,:,:]','double[:,:,:,:,:,:]')
def kernel_pi3(n, n_quad, n_his, n_nvbf, i_glo1, i_glo2, i_glo3, c_loc1, c_loc2, c_loc3, coeff1, coeff2, coeff3, coeff_ind1, coeff_ind2, coeff_ind3, bs1, bs2, bs3, x_his_ind1, x_his_ind2, x_his_ind3, wts1, wts2, wts3, tau, mat_eq):
    
    tau[:, :, :, :, :, :] = 0.
    
    #$ omp parallel
    #$ omp do reduction ( + : tau) private (i1, i2, i3, j1, j2, j3, coeff, kl1, k1, c1, kl2, k2, c2, kl3, k3, c3, f_int, q1, q2, q3, wvol)
    for i1 in range(n[0]):
        for i2 in range(n[1]):
            for i3 in range(n[2]):
                
                for j1 in range(n_his[0]):
                    for j2 in range(n_his[1]):
                        for j3 in range(n_his[2]):
                            
                            coeff = coeff1[coeff_ind1[i1], j1] * coeff2[coeff_ind2[i2], j2] * coeff3[coeff_ind3[i3], j3]
                            
                            for kl1 in range(n_nvbf[0]):
                                
                                k1 = i_glo1[i1, kl1]
                                c1 = c_loc1[i1, kl1]
                                
                                for kl2 in range(n_nvbf[1]):
                                    
                                    k2 = i_glo2[i2, kl2]
                                    c2 = c_loc2[i2, kl2]
                                    
                                    for kl3 in range(n_nvbf[2]):
                                        
                                        k3 = i_glo3[i3, kl3]
                                        c3 = c_loc3[i3, kl3]
                                        
                                        f_int = 0.
                                        
                                        for q1 in range(n_quad[0]):
                                            for q2 in range(n_quad[1]):
                                                for q3 in range(n_quad[2]):
                                                    wvol = wts1[x_his_ind1[i1, j1], q1] * wts2[x_his_ind2[i2, j2], q2] * wts3[x_his_ind3[i3, j3], q3]
                                                    f_int += wvol * bs1[x_his_ind1[i1, j1], q1, k1] * bs2[x_his_ind2[i2, j2], q2, k2] * bs3[x_his_ind3[i3, j3], q3, k3] * mat_eq[x_his_ind1[i1, j1], q1, x_his_ind2[i2, j2], q2, x_his_ind3[i3, j3], q3]
                                                
                                        tau[k1, k2, k3, c1, c2, c3] += coeff * f_int
    #$ omp end do
    #$ omp end parallel
    
    ierr = 0