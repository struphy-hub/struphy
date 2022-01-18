from pyccel.decorators import types


# ============================================ 3D ======================================

# =========================================
@types('double[:,:]','double[:]','double[:]')
def matrix_vector(a, b, c):
    
    c[:] = 0.
    
    for i in range(3):
        for j in range(3):
            c[i] += a[i, j] * b[j]      


# =========================================
@types('double[:,:]','double[:,:]','double[:,:]')
def matrix_matrix(a, b, c):
    
    c[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i, j] += a[i, k] * b[k, j]      


# =========================================
@types('double[:,:]','double[:,:]')
def transpose(a, b):
    
    b[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            b[i, j] = a[j, i]

            
# =========================================
@types('double[:,:]')
def det(a):
    
    plus  = a[0, 0]*a[1, 1]*a[2, 2] + a[0, 1]*a[1, 2]*a[2, 0] + a[0, 2]*a[1, 0]*a[2, 1]
    minus = a[2, 0]*a[1, 1]*a[0, 2] + a[2, 1]*a[1, 2]*a[0, 0] + a[2, 2]*a[1, 0]*a[0, 1]
    
    return plus - minus


# =======================================================
@types('double[:]','double[:]','double[:]')
def cross(a, b, c):
    
    c[:] = 0.
    
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    
    
# =========================================
@types('double[:,:]','double[:,:]')
def matrix_inv(a, b):
    
    # inverse determinant
    over_det_a = 1.0 / det(a)

    # inverse matrix
    b[0, 0] = (a[1, 1]*a[2, 2] - a[2, 1]*a[1, 2]) * over_det_a
    b[0, 1] = (a[2, 1]*a[0, 2] - a[0, 1]*a[2, 2]) * over_det_a
    b[0, 2] = (a[0, 1]*a[1, 2] - a[1, 1]*a[0, 2]) * over_det_a

    b[1, 0] = (a[1, 2]*a[2, 0] - a[2, 2]*a[1, 0]) * over_det_a
    b[1, 1] = (a[2, 2]*a[0, 0] - a[0, 2]*a[2, 0]) * over_det_a
    b[1, 2] = (a[0, 2]*a[1, 0] - a[1, 2]*a[0, 0]) * over_det_a

    b[2, 0] = (a[1, 0]*a[2, 1] - a[2, 0]*a[1, 1]) * over_det_a
    b[2, 1] = (a[2, 0]*a[0, 1] - a[0, 0]*a[2, 1]) * over_det_a
    b[2, 2] = (a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1]) * over_det_a