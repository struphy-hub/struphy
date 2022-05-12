from numpy import shape


# ===============================================================
def g_strong(f0 : 'double[:,:,:]', f1_1 : 'double[:,:,:]', f1_2 : 'double[:,:,:]', f1_3 : 'double[:,:,:]'):
    
    n1, n2, n3 = shape(f0)
    
    for i1 in range(shape(f1_1)[0]):
        for i2 in range(shape(f1_1)[1]):
            for i3 in range(shape(f1_1)[2]):
                f1_1[i1, i2, i3] = f0[(i1 + 1)%n1, i2, i3] - f0[i1, i2, i3]
                
    for i1 in range(shape(f1_2)[0]):
        for i2 in range(shape(f1_2)[1]):
            for i3 in range(shape(f1_2)[2]):
                f1_2[i1, i2, i3] = f0[i1, (i2 + 1)%n2, i3] - f0[i1, i2, i3]
                
    for i1 in range(shape(f1_3)[0]):
        for i2 in range(shape(f1_3)[1]):
            for i3 in range(shape(f1_3)[2]):
                f1_3[i1, i2, i3] = f0[i1, i2, (i3 + 1)%n3] - f0[i1, i2, i3]
                

# ===============================================================
def c_strong(f1_1 : 'double[:,:,:]', f1_2 : 'double[:,:,:]', f1_3 : 'double[:,:,:]', f2_1 : 'double[:,:,:]', f2_2 : 'double[:,:,:]', f2_3 : 'double[:,:,:]'):
    
    n1_1, n1_2, n1_3 = shape(f1_1)
    n2_1, n2_2, n2_3 = shape(f1_2)
    n3_1, n3_2, n3_3 = shape(f1_3)
    
    for i1 in range(shape(f2_1)[0]):
        for i2 in range(shape(f2_1)[1]):
            for i3 in range(shape(f2_1)[2]):
                f2_1[i1, i2, i3] = (f1_3[i1, (i2 + 1)%n3_2, i3] - f1_3[i1, i2, i3]) - (f1_2[i1, i2, (i3 + 1)%n2_3] - f1_2[i1, i2, i3])
                
    for i1 in range(shape(f2_2)[0]):
        for i2 in range(shape(f2_2)[1]):
            for i3 in range(shape(f2_2)[2]):
                f2_2[i1, i2, i3] = (f1_1[i1, i2, (i3 + 1)%n1_3] - f1_1[i1, i2, i3]) - (f1_3[(i1 + 1)%n3_1, i2, i3] - f1_3[i1, i2, i3])
    
    for i1 in range(shape(f2_3)[0]):
        for i2 in range(shape(f2_3)[1]):
            for i3 in range(shape(f2_3)[2]):
                f2_3[i1, i2, i3] = (f1_2[(i1 + 1)%n2_1, i2, i3] - f1_2[i1, i2, i3]) - (f1_1[i1, (i2 + 1)%n1_2, i3] - f1_1[i1, i2, i3])
                
                
# ===============================================================
def d_strong(f2_1 : 'double[:,:,:]', f2_2 : 'double[:,:,:]', f2_3 : 'double[:,:,:]', f3 : 'double[:,:,:]'):
    
    n1_1, n1_2, n1_3 = shape(f2_1)
    n2_1, n2_2, n2_3 = shape(f2_2)
    n3_1, n3_2, n3_3 = shape(f2_3)
    
    for i1 in range(shape(f3)[0]):
        for i2 in range(shape(f3)[1]):
            for i3 in range(shape(f3)[2]):
                f3[i1, i2, i3]  = (f2_1[(i1 + 1)%n1_1, i2, i3] - f2_1[i1, i2, i3]) 
                f3[i1, i2, i3] += (f2_2[i1, (i2 + 1)%n2_2, i3] - f2_2[i1, i2, i3])
                f3[i1, i2, i3] += (f2_3[i1, i2, (i3 + 1)%n3_3] - f2_3[i1, i2, i3])
                
                
# ===============================================================
def g_weak(f1_1 : 'double[:,:,:]', f1_2 : 'double[:,:,:]', f1_3 : 'double[:,:,:]', f0 : 'double[:,:,:]', bc : 'int[:]'):
    
    # contributions from 1st component
    for i1 in range(1 - bc[0], shape(f0)[0] - 1 + bc[0]):
        for i2 in range(shape(f0)[1]):
            for i3 in range(shape(f0)[2]):
                f0[i1, i2, i3]  = (f1_1[(i1 - 1)%shape(f0)[0], i2, i3] - f1_1[i1, i2, i3])
                
    
    # contributions from 2nd component
    for i1 in range(shape(f0)[0]):
        for i2 in range(1 - bc[1], shape(f0)[1] - 1 + bc[1]):
            for i3 in range(shape(f0)[2]):
                f0[i1, i2, i3] += (f1_2[i1, (i2 - 1)%shape(f0)[1], i3] - f1_2[i1, i2, i3])
                
                
    # contributions from 2nd component
    for i1 in range(shape(f0)[0]):
        for i2 in range(shape(f0)[1]):
            for i3 in range(1 - bc[2], shape(f0)[2] - 1 + bc[2]):
                f0[i1, i2, i3] += (f1_3[i1, i2, (i3 - 1)%shape(f0)[2]] - f1_3[i1, i2, i3])
                

# ===============================================================                
def c_weak(f2_1 : 'double[:,:,:]', f2_2 : 'double[:,:,:]', f2_3 : 'double[:,:,:]', f1_1 : 'double[:,:,:]', f1_2 : 'double[:,:,:]', f1_3 : 'double[:,:,:]', bc : 'int[:]'):
    
    # contributions to 1st component from 2nd component
    for i1 in range(shape(f1_1)[0]):
        for i2 in range(shape(f1_1)[1]):
            for i3 in range(1 - bc[2], shape(f1_1)[2] - 1 + bc[2]):
                f1_1[i1, i2, i3] = (f2_2[i1, i2, (i3 - 1)%shape(f1_1)[2]] - f2_2[i1, i2, i3])
                
    # contributions to 1st component from 3rd component
    for i1 in range(shape(f1_1)[0]):
        for i2 in range(1 - bc[1], shape(f1_1)[1] - 1 + bc[1]):
            for i3 in range(shape(f1_1)[2]):
                f1_1[i1, i2, i3] = f1_1[i1, i2, i3] - (f2_3[i1, (i2 - 1)%shape(f1_1)[1], i3] - f2_3[i1, i2, i3])
                
    # contributions to 2nd component from 3rd component
    for i1 in range(1 - bc[0], shape(f1_2)[0] - 1 + bc[0]):
        for i2 in range(shape(f1_2)[1]):
            for i3 in range(shape(f1_2)[2]):
                f1_2[i1, i2, i3] = (f2_3[(i1 - 1)%shape(f1_2)[0], i2, i3] - f2_3[i1, i2, i3])
                
    # contributions to 2nd component from 1st component
    for i1 in range(shape(f1_2)[0]):
        for i2 in range(shape(f1_2)[1]):
            for i3 in range(1 - bc[2], shape(f1_2)[2] - 1 + bc[2]):
                f1_2[i1, i2, i3] = f1_2[i1, i2, i3] - (f2_1[i1, i2, (i3 - 1)%shape(f1_2)[2]] - f2_1[i1, i2, i3])
                
    # contributions to 3rd component from 1st component
    for i1 in range(shape(f1_3)[0]):
        for i2 in range(1 - bc[1], shape(f1_3)[1] - 1 + bc[1]):
            for i3 in range(shape(f1_3)[2]):
                f1_3[i1, i2, i3] = (f2_1[i1, (i2 - 1)%shape(f1_3)[1], i3] - f2_1[i1, i2, i3])
                
    # contributions to 3rd component from 2nd component
    for i1 in range(1 - bc[0], shape(f1_3)[0] - 1 + bc[0]):
        for i2 in range(shape(f1_3)[1]):
            for i3 in range(shape(f1_3)[2]):
                f1_3[i1, i2, i3] = f1_3[i1, i2, i3] - (f2_2[(i1 - 1)%shape(f1_3)[0], i2, i3] - f2_2[i1, i2, i3])
                
                
# ===============================================================                
def d_weak(f3 : 'double[:,:,:]', f2_1 : 'double[:,:,:]', f2_2 : 'double[:,:,:]', f2_3 : 'double[:,:,:]', bc : 'int[:]'):
    
    # contributions to 1st component
    for i1 in range(1 - bc[0], shape(f2_1)[0] - 1 + bc[0]):
        for i2 in range(shape(f2_1)[1]):
            for i3 in range(shape(f2_1)[2]):
                f2_1[i1, i2, i3] = (f3[(i1 - 1)%shape(f2_1)[0], i2, i3] - f3[i1, i2, i3])
                
    # contributions to 2nd component
    for i1 in range(shape(f2_2)[0]):
        for i2 in range(1 - bc[1], shape(f2_2)[1] - 1 + bc[1]):
            for i3 in range(shape(f2_2)[2]):
                f2_2[i1, i2, i3] = (f3[i1, (i2 - 1)%shape(f2_2)[1], i3] - f3[i1, i2, i3])
                
    # contributions to 3rd component
    for i1 in range(shape(f2_3)[0]):
        for i2 in range(shape(f2_3)[1]):
            for i3 in range(1 - bc[2], shape(f2_3)[2] - 1 + bc[2]):
                f2_3[i1, i2, i3] = (f3[i1, i2, (i3 - 1)%shape(f2_3)[2]] - f3[i1, i2, i3])
                
                
# ===============================================================
def g_pol_strong(f0_pol : 'double[:,:]', f0_ten : 'double[:,:,:]', f1_12_pol : 'double[:,:]', f1_1_ten : 'double[:,:,:]', f1_2_ten : 'double[:,:,:]', f1_3_pol : 'double[:,:]', f1_3_ten : 'double[:,:,:]', xi1 : 'double[:,:]'):
    
    # number of radial degrees of freedom (clamped)
    n1 = shape(f0_ten)[0]
    d1 = n1 - 1
    
    # number of poloidal degrees of freedom (periodic)
    n2 = shape(f0_ten)[1]
    d2 = shape(f0_ten)[1]
    
    # number of toroidal degrees of freedom (clamped OR periodic)
    n3 = shape(f0_ten)[2]
    d3 = shape(f1_3_ten)[2]
    
    # 1st/2nd component polar degrees of freedom
    for i3 in range(n3):
        f1_12_pol[0, i3] = f0_pol[1, i3] - f0_pol[0, i3]
        f1_12_pol[1, i3] = f0_pol[2, i3] - f0_pol[0, i3]
        
    # 1st component tensor-product degrees of freedom (i1 = 1, D_{ 1}*N_{i2}*N_{i3})
    for i2 in range(n2):
        for i3 in range(n3):
            f1_1_ten[1, i2, i3] = f0_ten[2, i2, i3]
            
            for s in range(3):
                f1_1_ten[1, i2, i3] -= f0_pol[s, i3]*xi1[s, i2]
                
    # 1st component tensor-product degrees of freedom (i1 > 1, D_{i1}*N_{i2}*N_{i3})        
    for i1 in range(2, d1):
        for i2 in range(n2):
            for i3 in range(n3):
                f1_1_ten[i1, i2, i3] = f0_ten[i1 + 1, i2, i3] - f0_ten[i1, i2, i3]
                
    # 2nd component tensor-product degrees of freedom (i1 > 1, N_{i1}*D_{i2}*N_{i3})        
    for i1 in range(2, n1):
        for i2 in range(d2):
            for i3 in range(n3):
                f1_2_ten[i1, i2, i3] = f0_ten[i1, (i2 + 1)%n2, i3] - f0_ten[i1, i2, i3]
                
    # 3rd component polar degrees of freedom
    for i3 in range(d3):
        for s in range(3):
            f1_3_pol[s, i3] = f0_pol[s, (i3 + 1)%n3] - f0_pol[s, i3]
            
    # 3rd component tensor-product degrees of freedom (i1 > 1, N_{i1}*N_{i2}*D_{i3})
    for i1 in range(2, n1):
        for i2 in range(n2):
            for i3 in range(d3):
                f1_3_ten[i1, i2, i3] = f0_ten[i1, i2, (i3 + 1)%n3] - f0_ten[i1, i2, i3]
                
                
# ===============================================================
def c_pol_strong(f1_12_pol : 'double[:,:]', f1_1_ten : 'double[:,:,:]', f1_2_ten : 'double[:,:,:]', f1_3_pol : 'double[:,:]', f1_3_ten : 'double[:,:,:]', f2_12_pol : 'double[:,:]', f2_1_ten : 'double[:,:,:]', f2_2_ten : 'double[:,:,:]', f2_3_ten : 'double[:,:,:]', xi1 : 'double[:,:]'):
    
    # number of radial degrees of freedom (clamped)
    n1 = shape(f1_2_ten)[0]
    d1 = n1 - 1
    
    # number of poloidal degrees of freedom (periodic)
    n2 = shape(f1_1_ten)[1]
    d2 = shape(f1_1_ten)[1]
    
    # number of toroidal degrees of freedom (clamped OR periodic)
    n3 = shape(f1_1_ten)[2]
    d3 = shape(f1_3_ten)[2]
    
    # 1st/2nd component polar degrees of freedom
    for i3 in range(d3):
        f2_12_pol[0, i3] = (f1_3_pol[1, i3] - f1_3_pol[0, i3]) - (f1_12_pol[0, (i3 + 1)%n3] - f1_12_pol[0, i3])
        f2_12_pol[1, i3] = (f1_3_pol[2, i3] - f1_3_pol[0, i3]) - (f1_12_pol[1, (i3 + 1)%n3] - f1_12_pol[1, i3])
        
    # 1st component tensor-product degrees of freedom (i1 > 1, N_{i1}*D_{i2}*D_{i3})        
    for i1 in range(2, n1):
        for i2 in range(d2):
            for i3 in range(d3):
                f2_1_ten[i1, i2, i3] = (f1_3_ten[i1, (i2 + 1)%n2, i3] - f1_3_ten[i1, i2, i3]) - (f1_2_ten[i1, i2, (i3 + 1)%n3] - f1_2_ten[i1, i2, i3])
                
    # 2nd component tensor-product degrees of freedom (i1 = 1, D_{ 1}*N_{i2}*D_{i3})
    for i2 in range(n2):
        for i3 in range(d3):
            f2_2_ten[1, i2, i3] = (f1_1_ten[1, i2, (i3 + 1)%n3] - f1_1_ten[1, i2, i3]) - f1_3_ten[2, i2, i3]
            
            for s in range(3):
                f2_2_ten[1, i2, i3] += f1_3_pol[s, i3]*xi1[s, i2]
                
    # 2nd component tensor-product degrees of freedom (i1 > 1, D_{i1}*N_{i2}*D_{i3})
    for i1 in range(2, d1):
        for i2 in range(n2):
            for i3 in range(d3):
                f2_2_ten[i1, i2, i3] = (f1_1_ten[i1, i2, (i3 + 1)%n3] - f1_1_ten[i1, i2, i3]) - (f1_3_ten[i1 + 1, i2, i3] - f1_3_ten[i1, i2, i3])
                
    # 3rd component tensor-product degrees of freedom (i1 = 1, D_{ 1}*D_{i2}*N_{i3})
    for i2 in range(d2):
        for i3 in range(n3):
            f2_3_ten[1, i2, i3] = f1_2_ten[2, i2, i3] - (f1_1_ten[1, (i2 + 1)%n2, i3] - f1_1_ten[1, i2, i3])
            
            for s in range(2):
                f2_3_ten[1, i2, i3] = f2_3_ten[1, i2, i3] - f1_12_pol[s, i3]*(xi1[s + 1, (i2 + 1)%n2] - xi1[s + 1, i2])
                
    # 3rd component tensor-product degrees of freedom (i1 > 1, D_{i1}*D_{i2}*N_{i3})
    for i1 in range(2, d1):
        for i2 in range(d2):
            for i3 in range(n3):
                f2_3_ten[i1, i2, i3] = (f1_2_ten[i1 + 1, i2, i3] - f1_2_ten[i1, i2, i3]) - (f1_1_ten[i1, (i2 + 1)%n2, i3] - f1_1_ten[i1, i2, i3])
                
                
# ===============================================================
def d_pol_strong(f2_12_pol : 'double[:,:]', f2_1_ten : 'double[:,:,:]', f2_2_ten : 'double[:,:,:]', f2_3_ten : 'double[:,:,:]', f3_ten : 'double[:,:,:]', xi1 : 'double[:,:]'):
    
    # number of radial degrees of freedom (clamped)
    n1 = shape(f2_1_ten)[0]
    d1 = n1 - 1
    
    # number of poloidal degrees of freedom (periodic)
    n2 = shape(f2_2_ten)[1]
    d2 = shape(f2_2_ten)[1]
    
    # number of toroidal degrees of freedom (clamped OR periodic)
    n3 = shape(f2_3_ten)[2]
    d3 = shape(f2_2_ten)[2]
    
    # tensor-product degrees of freedom (i1 = 1, D_{ 1}*D_{i2}*D_{i3})
    for i2 in range(d2):
        for i3 in range(d3):
            f3_ten[1, i2, i3]  = (f2_2_ten[1, (i2 + 1)%n2, i3] - f2_2_ten[1, i2, i3]) + (f2_3_ten[1, i2, (i3 + 1)%n3] - f2_3_ten[1, i2, i3])
            f3_ten[1, i2, i3] +=  f2_1_ten[2, i2, i3]
            
            for s in range(2):
                f3_ten[1, i2, i3] = f3_ten[1, i2, i3] - f2_12_pol[s, i3]*(xi1[s + 1, (i2 + 1)%n2] - xi1[s + 1, i2])
                
    # tensor-product degrees of freedom (i1 > 1, D_{i1}*D_{i2}*D_{i3})
    for i1 in range(2, d1):
        for i2 in range(d2):
            for i3 in range(d3):
                f3_ten[i1, i2, i3] = (f2_1_ten[i1 + 1, i2, i3] - f2_1_ten[i1, i2, i3]) + (f2_2_ten[i1, (i2 + 1)%n2, i3] - f2_2_ten[i1, i2, i3]) + (f2_3_ten[i1, i2, (i3 + 1)%n3] - f2_3_ten[i1, i2, i3])