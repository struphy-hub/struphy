import numpy as np
from scipy import sparse


def GRAD_1d(Nbase_x, bc):
    
    if bc == False:
        Nbase_x_0 = Nbase_x - 2
        
        M = np.zeros((Nbase_x_0 + 1, Nbase_x_0 + 1))
    
        for i in range(0, Nbase_x_0 + 1):
            M[i, i] = 1.
            if i < Nbase_x_0:
                M[i + 1, i] = -1.

        return sparse.csr_matrix(M[:, 0:-1])
    
    elif bc == True:
        Nbase_x_0 = Nbase_x
        
        M = np.zeros((Nbase_x_0, Nbase_x_0))
        
        for i in range(0, Nbase_x_0):
            M[i, i] = -1.
            if i < Nbase_x_0 - 1:
                M[i, i + 1] = 1.
                
        M[-1, 0] = 1.
        
        return sparse.csr_matrix(M)  
    
    
def GRAD_3d(Nbase_x, Nbase_y, Nbase_z, bc):
    
    if bc == True:
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    grad_x_1d = GRAD_1d(Nbase_x, bc)
    grad_y_1d = GRAD_1d(Nbase_y, bc)
    grad_z_1d = GRAD_1d(Nbase_z, bc)
    
    grad_x = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0)), sparse.identity(Nbase_z_0))
    grad_y = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), grad_y_1d), sparse.identity(Nbase_z_0))
    grad_z = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), sparse.identity(Nbase_y_0)), grad_z_1d)
    
    return sparse.bmat([[grad_x], [grad_y], [grad_z]], format='csr')


def CURL_3d(Nbase_x, Nbase_y, Nbase_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
    
    grad_x_1d = GRAD_1d(Nbase_x, bc)
    grad_y_1d = GRAD_1d(Nbase_y, bc)
    grad_z_1d = GRAD_1d(Nbase_z, bc)

    grad_xy = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), sparse.identity(Nbase_y_0 + bcon)), grad_z_1d)
    grad_xz = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0), grad_y_1d), sparse.identity(Nbase_z_0 + bcon))

    grad_yx = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + bcon), sparse.identity(Nbase_y_0)), grad_z_1d)
    grad_yz = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0)), sparse.identity(Nbase_z_0 + bcon))

    grad_zx = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + bcon), grad_y_1d), sparse.identity(Nbase_z_0))
    grad_zy = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0 + bcon)), sparse.identity(Nbase_z_0))

    ZERO_xx = sparse.csr_matrix((Nbase_x_0*(Nbase_y_0 + bcon)*(Nbase_z_0 + bcon), (Nbase_x_0 + bcon)*Nbase_y_0*Nbase_z_0))
    ZERO_yy = sparse.csr_matrix(((Nbase_x_0 + bcon)*Nbase_y_0*(Nbase_z_0 + bcon), Nbase_x_0*(Nbase_y_0 + bcon)*Nbase_z_0))
    ZERO_zz = sparse.csr_matrix(((Nbase_x_0 + bcon)*(Nbase_y_0 + bcon)*Nbase_z_0, Nbase_x_0*Nbase_y_0*(Nbase_z_0 + bcon)))

    return sparse.bmat([[ZERO_xx, -grad_xy, grad_xz], [grad_yx, ZERO_yy, -grad_yz], [-grad_zx, grad_zy, ZERO_zz]], format='csr')
    
    
    
def DIV_3d(Nbase_x, Nbase_y, Nbase_z, bc):
    
    if bc == True:
        bcon = 0
        
        Nbase_x_0 = Nbase_x
        Nbase_y_0 = Nbase_y
        Nbase_z_0 = Nbase_z
    else:
        bcon = 1
        
        Nbase_x_0 = Nbase_x - 2
        Nbase_y_0 = Nbase_y - 2
        Nbase_z_0 = Nbase_z - 2
    
        
    grad_x_1d = GRAD_1d(Nbase_x, bc)
    grad_y_1d = GRAD_1d(Nbase_y, bc)
    grad_z_1d = GRAD_1d(Nbase_z, bc)

    grad_x = sparse.kron(sparse.kron(grad_x_1d, sparse.identity(Nbase_y_0 + bcon)), sparse.identity(Nbase_z_0 + bcon))
    grad_y = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + bcon), grad_y_1d), sparse.identity(Nbase_z_0 + bcon))
    grad_z = sparse.kron(sparse.kron(sparse.identity(Nbase_x_0 + bcon), sparse.identity(Nbase_y_0 + bcon)), grad_z_1d)

    return sparse.bmat([[grad_x, grad_y, grad_z]], format='csr')
