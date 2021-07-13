# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Modules to obtain preconditioners for mass matrices in 3D.
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.spline_space as spl

import hylife.linear_algebra.linalg_kron as linkron



# ================ inverse mass matrix in V0 ===========================
def get_M0_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning
    #spaces_pre[0].set_extraction_operators()
    #spaces_pre[1].set_extraction_operators()
    #spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M0.shape, matvec=lambda x: (linkron.kron_fftsolve_3d(c_pre, x.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2]))).flatten())



# ================ inverse mass matrix in V1 ===========================
def get_M1_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    #spaces_pre[0].set_extraction_operators()
    #spaces_pre[1].set_extraction_operators()
    #spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    spaces_pre[0].assemble_M1(lambda eta : 1/domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta : 1/domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta : 1/domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c11_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    c22_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    c33_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    
    def solve(x):
        
        x1, x2, x3 = np.split(x, 3)
        
        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        
        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()
        
        return np.concatenate((r1, r2, r3))
            
    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M1.shape, matvec=solve)


# ================ inverse mass matrix in V2 ===========================
def get_M2_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    #spaces_pre[0].set_extraction_operators()
    #spaces_pre[1].set_extraction_operators()
    #spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    spaces_pre[0].assemble_M1(lambda eta : 1/domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta : 1/domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta : 1/domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c11_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    c22_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]
    c33_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    def solve(x):
        
        x1, x2, x3 = np.split(x, 3)
        
        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        
        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()
        
        return np.concatenate((r1, r2, r3))
            
    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M2.shape, matvec=solve)


# ================ inverse mass matrix in V3 ===========================
def get_M3_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning
    #spaces_pre[0].set_extraction_operators()
    #spaces_pre[1].set_extraction_operators()
    #spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M1(lambda eta : 1/domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M1(lambda eta : 1/domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M1(lambda eta : 1/domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c_pre = [spaces_pre[0].M1.toarray()[:, 0], spaces_pre[1].M1.toarray()[:, 0], spaces_pre[2].M1.toarray()[:, 0]]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M3.shape, matvec=lambda x: (linkron.kron_fftsolve_3d(c_pre, x.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2]))).flatten())



# ================ inverse mass matrix in V0^3 ===========================
def get_Mv_PRE(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 1d spaces for pre-conditioning with fft:
    Nel_pre      = [tensor_space_FEM.spaces[0].NbaseN, tensor_space_FEM.spaces[1].NbaseN, tensor_space_FEM.spaces[2].NbaseN] 
    spl_kind_pre = [True, True, True]
    spaces_pre   = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel_pre, tensor_space_FEM.p, spl_kind_pre, tensor_space_FEM.n_quad)]
    
    # tensor product mass matrices for pre-conditioning of the three diagonal blocks
    #spaces_pre[0].set_extraction_operators()
    #spaces_pre[1].set_extraction_operators()
    #spaces_pre[2].set_extraction_operators()
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]**3*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c11_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]**3*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]*np.ones(eta.shape, dtype=float))
    
    c22_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    spaces_pre[0].assemble_M0(lambda eta : domain.params_map[0]*np.ones(eta.shape, dtype=float))
    spaces_pre[1].assemble_M0(lambda eta : domain.params_map[1]*np.ones(eta.shape, dtype=float))
    spaces_pre[2].assemble_M0(lambda eta : domain.params_map[2]**3*np.ones(eta.shape, dtype=float))
    
    c33_pre = [spaces_pre[0].M0.toarray()[:, 0], spaces_pre[1].M0.toarray()[:, 0], spaces_pre[2].M0.toarray()[:, 0]]
    
    def solve(x):
        
        x1, x2, x3 = np.split(x, 3)
        
        x1 = x1.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x2 = x2.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        x3 = x3.reshape(Nel_pre[0], Nel_pre[1], Nel_pre[2])
        
        r1 = linkron.kron_fftsolve_3d(c11_pre, x1).flatten()
        r2 = linkron.kron_fftsolve_3d(c22_pre, x2).flatten()
        r3 = linkron.kron_fftsolve_3d(c33_pre, x3).flatten()
        
        return np.concatenate((r1, r2, r3))
            
    return spa.linalg.LinearOperator(shape=tensor_space_FEM.Mv.shape, matvec=solve)



# ========== inverse mass matrix in V0 with decomposition poloidal x toroidal ==============
def get_M0_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    #space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M0_2D(domain)
    space_tor.assemble_M0()
    
    # LU decomposition of poloidal mass matrix
    M0_pol_LU = spa.linalg.splu(space_pol.M0.tocsc())
    
    # vector defining the circulant mass matrix in toroidal direction
    tor_vec = space_tor.M0.toarray()[:, 0]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M0.shape, matvec=lambda x: linkron.kron_fftsolve_2d(M0_pol_LU, tor_vec, x.reshape(tensor_space_FEM.E0_pol.shape[0], tensor_space_FEM.NbaseN[2])).flatten())


# ========== inverse mass matrix in V1 with decomposition poloidal x toroidal ==============
def get_M1_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    #space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M1_2D_blocks(domain)
    space_tor.assemble_M0()
    space_tor.assemble_M1()
    
    # LU decomposition of poloidal mass matrix
    M1_pol_12_LU = spa.linalg.splu(space_pol.M1_12.tocsc())
    M1_pol_33_LU = spa.linalg.splu(space_pol.M1_33.tocsc())
    
    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = space_tor.M0.toarray()[:, 0]
    tor_vec1 = space_tor.M1.toarray()[:, 0]
    
    def solve(x):
        
        x1 = x[:tensor_space_FEM.E1_pol.shape[0]*tensor_space_FEM.NbaseN[2] ].reshape(tensor_space_FEM.E1_pol.shape[0], tensor_space_FEM.NbaseN[2])
        x3 = x[ tensor_space_FEM.E1_pol.shape[0]*tensor_space_FEM.NbaseN[2]:].reshape(tensor_space_FEM.E0_pol.shape[0], tensor_space_FEM.NbaseD[2])
        
        r1 = linkron.kron_fftsolve_2d(M1_pol_12_LU, tor_vec0, x1).flatten()
        r3 = linkron.kron_fftsolve_2d(M1_pol_33_LU, tor_vec1, x3).flatten()
        
        return np.concatenate((r1, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M1.shape, matvec=solve)


# ========== inverse mass matrix in V2 with decomposition poloidal x toroidal ==============
def get_M2_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    #space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M2_2D_blocks(domain)
    space_tor.assemble_M0()
    space_tor.assemble_M1()
    
    # LU decomposition of poloidal mass matrix
    M2_pol_12_LU = spa.linalg.splu(space_pol.M2_12.tocsc())
    M2_pol_33_LU = spa.linalg.splu(space_pol.M2_33.tocsc())
    
    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = space_tor.M0.toarray()[:, 0]
    tor_vec1 = space_tor.M1.toarray()[:, 0]
    
    def solve(x):
        
        x1 = x[:tensor_space_FEM.E2_pol.shape[0]*tensor_space_FEM.NbaseD[2] ].reshape(tensor_space_FEM.E2_pol.shape[0], tensor_space_FEM.NbaseD[2])
        x3 = x[ tensor_space_FEM.E2_pol.shape[0]*tensor_space_FEM.NbaseD[2]:].reshape(tensor_space_FEM.E3_pol.shape[0], tensor_space_FEM.NbaseN[2])
        
        r1 = linkron.kron_fftsolve_2d(M2_pol_12_LU, tor_vec1, x1).flatten()
        r3 = linkron.kron_fftsolve_2d(M2_pol_33_LU, tor_vec0, x3).flatten()
        
        return np.concatenate((r1, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M2.shape, matvec=solve)


# ========== inverse mass matrix in V3 with decomposition poloidal x toroidal ==============
def get_M3_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    #space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_M3_2D(domain)
    space_tor.assemble_M1()
    
    # LU decomposition of poloidal mass matrix
    M3_pol_LU = spa.linalg.splu(space_pol.M3.tocsc())
    
    # vector defining the circulant mass matrix in toroidal direction
    tor_vec = space_tor.M1.toarray()[:, 0]

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.M3.shape, matvec=lambda x: linkron.kron_fftsolve_2d(M3_pol_LU, tor_vec, x.reshape(tensor_space_FEM.E3_pol.shape[0], tensor_space_FEM.NbaseD[2])).flatten())



# ========== inverse mass matrix in V0^3 with decomposition poloidal x toroidal ==============
def get_Mv_PRE_3(tensor_space_FEM, domain):
    """
    TODO
    """
    
    # 2D tensor product space in poloidal plane
    space_pol = spl.tensor_spline_space([tensor_space_FEM.spaces[0], tensor_space_FEM.spaces[1]])
    
    # 1D space in toroidal direction
    space_tor = tensor_space_FEM.spaces[2]
    
    # set extraction operators
    space_pol.set_extraction_operators(tensor_space_FEM.bc, tensor_space_FEM.polar_splines)
    #space_tor.set_extraction_operators()
    
    # mass matrices
    space_pol.assemble_Mv_2D_blocks(domain)
    space_tor.assemble_M0()
    
    # LU decomposition of poloidal mass matrix
    Mv_pol_12_LU = spa.linalg.splu(space_pol.Mv_12.tocsc())
    Mv_pol_33_LU = spa.linalg.splu(space_pol.Mv_33.tocsc())
    
    # vectors defining the circulant mass matrices in toroidal direction
    tor_vec0 = space_tor.M0.toarray()[:, 0]
    
    def solve(x):
        
        x1 = x[:(tensor_space_FEM.E0_pol.shape[0] + tensor_space_FEM.E0_pol_all.shape[0])*tensor_space_FEM.NbaseN[2] ].reshape(tensor_space_FEM.E0_pol.shape[0] + tensor_space_FEM.E0_pol_all.shape[0], tensor_space_FEM.NbaseN[2])
        x3 = x[ (tensor_space_FEM.E0_pol.shape[0] + tensor_space_FEM.E0_pol_all.shape[0])*tensor_space_FEM.NbaseN[2]:].reshape(tensor_space_FEM.E0_pol_all.shape[0], tensor_space_FEM.NbaseN[2])
        
        r1 = linkron.kron_fftsolve_2d(Mv_pol_12_LU, tor_vec0, x1).flatten()
        r3 = linkron.kron_fftsolve_2d(Mv_pol_33_LU, tor_vec0, x3).flatten()
        
        return np.concatenate((r1, r3))

    return spa.linalg.LinearOperator(shape=tensor_space_FEM.Mv.shape, matvec=solve)