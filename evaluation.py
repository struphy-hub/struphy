import numpy as np
import psydac.core.bsplines as bsp
import scipy.sparse as sparse


def evaluate_field_V0(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, x, bc)[:, bcon:Nbase_x_0 + bcon]) 
    N_y = sparse.csr_matrix(bsp.collocation_matrix(Ty, py, y, bc)[:, bcon:Nbase_y_0 + bcon]) 
    N_z = sparse.csr_matrix(bsp.collocation_matrix(Tz, pz, z, bc)[:, bcon:Nbase_z_0 + bcon]) 
    
    EVAL = sparse.kron(sparse.kron(N_x, N_y), N_z)
    
    return EVAL.dot(vec)


def evaluate_field_V1_x(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    tx = bsp.make_knots(el_b_x, px - 1, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    
    D_x = bsp.collocation_matrix(tx, px - 1, x, bc)
    
    for j in range(Nbase_x_0 + bcon):
        D_x[:, j] = px*D_x[:, j]/(tx[j + px] - tx[j])
        
    D_x = sparse.csr_matrix(D_x)
    
    N_y = sparse.csr_matrix(bsp.collocation_matrix(Ty, py, y, bc)[:, bcon:Nbase_y_0 + bcon])
    N_z = sparse.csr_matrix(bsp.collocation_matrix(Tz, pz, z, bc)[:, bcon:Nbase_z_0 + bcon])
    
    
    EVAL = sparse.kron(sparse.kron(D_x, N_y), N_z)
    
    return EVAL.dot(vec)


def evaluate_field_V1_y(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    ty = bsp.make_knots(el_b_y, py - 1, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    
    D_y = bsp.collocation_matrix(ty, py - 1, y, bc)
    
    for j in range(Nbase_y_0 + bcon):
        D_y[:, j] = py*D_y[:, j]/(ty[j + px] - ty[j])
        
    D_y = sparse.csr_matrix(D_y)
    
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, x, bc)[:, bcon:Nbase_x_0 + bcon])
    N_z = sparse.csr_matrix(bsp.collocation_matrix(Tz, pz, z, bc)[:, bcon:Nbase_z_0 + bcon])
    
    
    EVAL = sparse.kron(sparse.kron(N_x, D_y), N_z)
    
    return EVAL.dot(vec)


def evaluate_field_V1_z(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    tz = bsp.make_knots(el_b_z, pz - 1, bc)
    
    
    D_z = bsp.collocation_matrix(tz, pz - 1, z, bc)
    
    for j in range(Nbase_z_0 + bcon):
        D_z[:, j] = pz*D_z[:, j]/(tz[j + px] - tz[j])
        
    D_z = sparse.csr_matrix(D_z)
    
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, x, bc)[:, bcon:Nbase_x_0 + bcon])
    N_y = sparse.csr_matrix(bsp.collocation_matrix(Ty, py, y, bc)[:, bcon:Nbase_y_0 + bcon])
    
    
    EVAL = sparse.kron(sparse.kron(N_x, N_y), D_z)
    
    return EVAL.dot(vec)


def evaluate_field_V2_x(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    ty = bsp.make_knots(el_b_y, py - 1, bc)
    tz = bsp.make_knots(el_b_z, pz - 1, bc)
    
    
    D_y = bsp.collocation_matrix(ty, py - 1, y, bc)
    D_z = bsp.collocation_matrix(tz, pz - 1, z, bc)
    
    for j in range(Nbase_y_0 + bcon):
        D_y[:, j] = py*D_y[:, j]/(ty[j + py] - ty[j])
        
    for j in range(Nbase_z_0 + bcon):
        D_z[:, j] = pz*D_z[:, j]/(tz[j + pz] - tz[j])
        
    D_y = sparse.csr_matrix(D_y)
    D_z = sparse.csr_matrix(D_z)
    
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, x, bc)[:, bcon:Nbase_x_0 + bcon])
    
    EVAL = sparse.kron(sparse.kron(N_x, D_y), D_z)
    
    return EVAL.dot(vec)


def evaluate_field_V2_y(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    tx = bsp.make_knots(el_b_x, px - 1, bc)
    Ty = bsp.make_knots(el_b_y, py, bc)
    tz = bsp.make_knots(el_b_z, pz - 1, bc)
    
    
    D_x = bsp.collocation_matrix(tx, px - 1, x, bc)
    D_z = bsp.collocation_matrix(tz, pz - 1, z, bc)
    
    for j in range(Nbase_x_0 + bcon):
        D_x[:, j] = px*D_x[:, j]/(tx[j + px] - tx[j])
        
    for j in range(Nbase_z_0 + bcon):
        D_z[:, j] = pz*D_z[:, j]/(tz[j + pz] - tz[j])
        
    D_x = sparse.csr_matrix(D_x)
    D_z = sparse.csr_matrix(D_z)
    
    N_y = sparse.csr_matrix(bsp.collocation_matrix(Ty, py, y, bc)[:, bcon:Nbase_y_0 + bcon])
    
    EVAL = sparse.kron(sparse.kron(D_x, N_y), D_z)
    
    return EVAL.dot(vec)



def evaluate_field_V2_z(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    tx = bsp.make_knots(el_b_x, px - 1, bc)
    ty = bsp.make_knots(el_b_y, py - 1, bc)
    Tz = bsp.make_knots(el_b_z, pz, bc)
    
    
    D_x = bsp.collocation_matrix(tx, px - 1, x, bc)
    D_y = bsp.collocation_matrix(ty, py - 1, y, bc)
    
    for j in range(Nbase_x_0 + bcon):
        D_x[:, j] = px*D_x[:, j]/(tx[j + px] - tx[j])
        
    for j in range(Nbase_y_0 + bcon):
        D_y[:, j] = py*D_y[:, j]/(ty[j + py] - ty[j])
        
    D_x = sparse.csr_matrix(D_x)
    D_y = sparse.csr_matrix(D_y)
    
    N_z = sparse.csr_matrix(bsp.collocation_matrix(Tz, pz, z, bc)[:, bcon:Nbase_z_0 + bcon])
    
    EVAL = sparse.kron(sparse.kron(D_x, D_y), N_z)
    
    return EVAL.dot(vec)



def evaluate_field_V3(vec, x, y, z, Nbase_x, Nbase_y, Nbase_z, px, py, pz, el_b_x, el_b_y, el_b_z, bc):
    
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
    
    tx = bsp.make_knots(el_b_x, px - 1, bc)
    ty = bsp.make_knots(el_b_y, py - 1, bc)
    tz = bsp.make_knots(el_b_z, pz - 1, bc)
    
    
    D_x = bsp.collocation_matrix(tx, px - 1, x, bc)
    D_y = bsp.collocation_matrix(ty, py - 1, y, bc)
    D_z = bsp.collocation_matrix(tz, pz - 1, z, bc)
    
    for j in range(Nbase_x_0 + bcon):
        D_x[:, j] = px*D_x[:, j]/(tx[j + px] - tx[j])
        
    for j in range(Nbase_y_0 + bcon):
        D_y[:, j] = py*D_y[:, j]/(ty[j + py] - ty[j])
        
    for j in range(Nbase_z_0 + bcon):
        D_z[:, j] = pz*D_z[:, j]/(tz[j + pz] - tz[j])
        
    D_x = sparse.csr_matrix(D_x)
    D_y = sparse.csr_matrix(D_y)
    D_z = sparse.csr_matrix(D_z)
    
    EVAL = sparse.kron(sparse.kron(D_x, D_y), D_z)
    
    return EVAL.dot(vec)



def evaluate_field_V0_1d(vec, x, Nbase_x, px, el_b_x, bc):
    
    if bc == True:
        bcon = 0
        Nbase_x_0 = Nbase_x
        
    else:
        bcon = 1 
        Nbase_x_0 = Nbase_x - 2
    
    Tx = bsp.make_knots(el_b_x, px, bc)
    
    N_x = sparse.csr_matrix(bsp.collocation_matrix(Tx, px, x, bc)[:, bcon:Nbase_x_0 + bcon]) 
    
    return N_x.dot(vec)


def evaluate_field_V1_1d(vec, x, Nbase_x, px, el_b_x, bc):
    
    if bc == True:
        bcon = 0
        Nbase_x_0 = Nbase_x
        
    else:
        bcon = 1
        Nbase_x_0 = Nbase_x - 2
    
    tx = bsp.make_knots(el_b_x, px - 1, bc)
    
    D_x = bsp.collocation_matrix(tx, px - 1, x, bc)
    
    for j in range(Nbase_x_0 + bcon):
        D_x[:, j] = px*D_x[:, j]/(tx[j + px] - tx[j])
        
    D_x = sparse.csr_matrix(D_x)
    
    return D_x.dot(vec)