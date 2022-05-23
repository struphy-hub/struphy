# ===============================================================
def projector_tensor_strong(pi1 : 'double[:,:,:]', pi2 : 'double[:,:,:]', pi3 : 'double[:,:,:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', a : 'double[:,:,:]', b : 'double[:,:,:]', c : 'double[:,:,:]'):
    
    n1 = len(ind1[0])
    n2 = len(ind2[0])
    n3 = len(ind3[0])
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                c[ind1[0, i1], ind2[0, i2], ind3[0, i3]] += a[ind1[1, i1], ind2[1, i2], ind3[1, i3]] * pi1[ind1[0, i1], ind1[1, i1], ind1[2, i1]] * pi2[ind2[0, i2], ind2[1, i2], ind2[2, i2]] * pi3[ind3[0, i3], ind3[1, i3], ind3[2, i3]] * b[ind1[2, i1], ind2[2, i2], ind3[2, i3]]
                
                
                
# ===============================================================
def projector_tensor_strong_reduced(pi1 : 'double[:,:]', pi2 : 'double[:,:]', pi3 : 'double[:,:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', a : 'double[:,:,:]', c : 'double[:,:,:]'):
    
    n1 = len(ind1[0])
    n2 = len(ind2[0])
    n3 = len(ind3[0])
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                c[ind1[0, i1], ind2[0, i2], ind3[0, i3]] += a[ind1[1, i1], ind2[1, i2], ind3[1, i3]] * pi1[ind1[0, i1], ind1[1, i1]] * pi2[ind2[0, i2], ind2[1, i2]] * pi3[ind3[0, i3], ind3[1, i3]]
                          
                                
                
# ===============================================================                
def projector_tensor_weak(pi1 : 'double[:,:,:]', pi2 : 'double[:,:,:]', pi3 : 'double[:,:,:]', ind1 : 'int[:,:]', ind2 : 'int[:,:]', ind3 : 'int[:,:]', a : 'double[:,:,:]', b : 'double[:,:,:]', c : 'double[:,:,:]'):
    
    n1 = len(ind1[0])
    n2 = len(ind2[0])
    n3 = len(ind3[0])
    
    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                
                c[ind1[2, i1], ind2[2, i2], ind3[2, i3]] += a[ind1[0, i1], ind2[0, i2], ind3[0, i3]] * pi1[ind1[0, i1], ind1[1, i1], ind1[2, i1]] * pi2[ind2[0, i2], ind2[1, i2], ind2[2, i2]] * pi3[ind3[0, i3], ind3[1, i3], ind3[2, i3]] * b[ind1[1, i1], ind2[1, i2], ind3[1, i3]]