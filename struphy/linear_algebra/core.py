# ============================================ 3D ======================================

# =========================================
def matrix_vector(a : 'double[:,:]', b : 'double[:]', c : 'double[:]'):
    
    c[:] = 0.
    
    for i in range(3):
        for j in range(3):
            c[i] += a[i, j] * b[j]      


# =========================================
def matrix_matrix(a : 'double[:,:]', b : 'double[:,:]', c : 'double[:,:]'):
    
    c[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c[i, j] += a[i, k] * b[k, j]      


# =========================================
def transpose(a : 'double[:,:]', b : 'double[:,:]'):
    
    b[:, :] = 0.
    
    for i in range(3):
        for j in range(3):
            b[i, j] = a[j, i]

            
# =========================================
def det(a : 'double[:,:]') -> 'double':
    
    plus  = a[0, 0]*a[1, 1]*a[2, 2] + a[0, 1]*a[1, 2]*a[2, 0] + a[0, 2]*a[1, 0]*a[2, 1]
    minus = a[2, 0]*a[1, 1]*a[0, 2] + a[2, 1]*a[1, 2]*a[0, 0] + a[2, 2]*a[1, 0]*a[0, 1]
    
    return plus - minus


# =======================================================
def cross(a : 'double[:]', b : 'double[:]', c : 'double[:]'):
    
    c[:] = 0.
    
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    
    
# =========================================
def matrix_inv(a : 'double[:,:]', b : 'double[:,:]'):
    
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