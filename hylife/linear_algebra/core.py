from pyccel.decorators import types


# =========================================
@types('double[:,:](order=F)','double[:]','double[:]')
def matrix_vector(A, b, c):
    
    c[:] = 0.
    
    for i in range(3):
        for j in range(3):
            c[i] += A[i, j] * b[j]      


# =========================================
@types('double[:,:](order=F)','double[:,:](order=F)','double[:,:](order=F)')
def matrix_matrix(A, B, C):
    
    C[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]      


# =========================================
@types('double[:,:](order=F)','double[:,:](order=F)')
def transpose(A, B):
    
    B[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            B[i, j] = A[j, i]

            
# =========================================
@types('double[:,:](order=F)')
def det(A):
    
    plus  = A[0, 0]*A[1, 1]*A[2, 2] + A[0, 1]*A[1, 2]*A[2, 0] + A[0, 2]*A[1, 0]*A[2, 1]
    minus = A[2, 0]*A[1, 1]*A[0, 2] + A[2, 1]*A[1, 2]*A[0, 0] + A[2, 2]*A[1, 0]*A[0, 1]
    
    return plus - minus