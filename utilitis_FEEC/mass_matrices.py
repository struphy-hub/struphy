import numpy                      as np
import scipy.sparse               as spa

import utilitis_FEEC.bsplines     as bsp
import utilitis_FEEC.kernels_mass as ker



#==================================================calling epyccel for acceleration============================================
from pyccel import epyccel
ker = epyccel(ker)
#==============================================================================================================================




#==================================================mass matrix in V0 (3d)======================================================
def mass_V0(T, p, bc, g_sqrt):
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]

    M0        = np.zeros((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), order='F')

    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
                                  
    mat_map   = np.asfortranarray(g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 0, 0, 0, 0, 0, 0, quad[0][1], quad[1][1], quad[2][1], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], M0, mat_map)
                
    indices   = np.indices((NbaseN[0], NbaseN[1], NbaseN[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift     = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]
    
    row       = (NbaseN[1]*NbaseN[2]*indices[0] + NbaseN[2]*indices[1] + indices[2]).flatten()
    
    col1      = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseN[0]
    col2      = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseN[1]
    col3      = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseN[2]

    col       = NbaseN[1]*NbaseN[2]*col1 + NbaseN[2]*col2 + col3
                
    M0        = spa.csr_matrix((M0.flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1]*NbaseN[2], NbaseN[0]*NbaseN[1]*NbaseN[2]))
    M0.eliminate_zeros()
                
    return M0
#==============================================================================================================================





#==================================================mass matrix in V1 (3d)======================================================
def mass_V1(T, p, bc, Ginv, g_sqrt):
    
    t         = [T[1:-1] for T in T]
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbi1      = [NbaseD[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0], NbaseN[0]]
    Nbi2      = [NbaseN[1], NbaseD[1], NbaseD[1], NbaseN[1], NbaseN[1], NbaseN[1]]
    Nbi3      = [NbaseN[2], NbaseN[2], NbaseN[2], NbaseD[2], NbaseD[2], NbaseD[2]]
    
    Nbj1      = [NbaseD[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseN[0]]
    Nbj2      = [NbaseN[1], NbaseN[1], NbaseD[1], NbaseN[1], NbaseD[1], NbaseN[1]]
    Nbj3      = [NbaseN[2], NbaseN[2], NbaseN[2], NbaseN[2], NbaseN[2], NbaseD[2]]
    
    basis     = [[basisD[0], basisN[1], basisN[2]], [basisN[0], basisD[1], basisN[2]], [basisN[0], basisN[1], basisD[2]]]
    ns        = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    M1        = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), order='F') for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
    
    counter   = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            mat_map = np.asfortranarray(Ginv[a][b](quad_mesh[0], quad_mesh[1], quad_mesh[2]) * g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, nj1, nj2, nj3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M1[counter], mat_map)
            
            indices = np.indices((Nbi1[counter], Nbi2[counter], Nbi3[counter], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            shift3  = np.arange(Nbi3[counter]) - p[2]
            
            row     = (Nbi2[counter]*Nbi3[counter]*indices[0] + Nbi3[counter]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift1[:, None, None, None, None, None])%Nbj1[counter]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%Nbj2[counter]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%Nbj3[counter]
            
            col     = Nbj2[counter]*Nbj3[counter]*col1 + Nbj3[counter]*col2 + col3
            
            M1[counter] = spa.csr_matrix((M1[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M1[counter].eliminate_zeros()
            
            counter += 1
    
    M1 = spa.bmat([[M1[0], M1[1].T, M1[3].T], [M1[1], M1[2], M1[4].T], [M1[3], M1[4], M1[5]]], format='csr')
            
    return M1
#==============================================================================================================================







#==================================================mass matrix in V2 (3d)======================================================
def mass_V2(T, p, bc, G, g_sqrt):
    
    t         = [T[1:-1] for T in T]
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbi1      = [NbaseN[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0], NbaseD[0]]
    Nbi2      = [NbaseD[1], NbaseN[1], NbaseN[1], NbaseD[1], NbaseD[1], NbaseD[1]]
    Nbi3      = [NbaseD[2], NbaseD[2], NbaseD[2], NbaseN[2], NbaseN[2], NbaseN[2]]
    
    Nbj1      = [NbaseN[0], NbaseN[0], NbaseD[0], NbaseN[0], NbaseD[0], NbaseD[0]]
    Nbj2      = [NbaseD[1], NbaseD[1], NbaseN[1], NbaseD[1], NbaseN[1], NbaseD[1]]
    Nbj3      = [NbaseD[2], NbaseD[2], NbaseD[2], NbaseD[2], NbaseD[2], NbaseN[2]]
    
    basis     = [[basisN[0], basisD[1], basisD[2]], [basisD[0], basisN[1], basisD[2]], [basisD[0], basisD[1], basisN[2]]]
    ns        = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    
    M2        = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), order='F') for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')

    counter   = 0
    
    for a in range(3):
        for b in range(a + 1):
            
            mat_map = np.asfortranarray(G[a][b](quad_mesh[0], quad_mesh[1], quad_mesh[2]) / g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, nj1, nj2, nj3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M2[counter], mat_map)
            
            indices = np.indices((Nbi1[counter], Nbi2[counter], Nbi3[counter], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            shift3  = np.arange(Nbi3[counter]) - p[2]
            
            row     = (Nbi2[counter]*Nbi3[counter]*indices[0] + Nbi3[counter]*indices[1] + indices[2]).flatten()
            
            col1    = (indices[3] + shift1[:, None, None, None, None, None])%Nbj1[counter]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%Nbj2[counter]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%Nbj3[counter]
            
            col     = Nbj2[counter]*Nbj3[counter]*col1 + Nbj3[counter]*col2 + col3
            
            M2[counter] = spa.csr_matrix((M2[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter]*Nbi3[counter], Nbj1[counter]*Nbj2[counter]*Nbj3[counter]))
            M2[counter].eliminate_zeros()
            
            counter += 1
    
    M2 = spa.bmat([[M2[0], M2[1].T, M2[3].T], [M2[1], M2[2], M2[4].T], [M2[3], M2[4], M2[5]]], format='csr')
            
    return M2
#==============================================================================================================================





#==================================================mass matrix in V3 (3d)======================================================
def mass_V3(T, p, bc, g_sqrt):
    
    t         = [T[1:-1] for T in T]
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisD    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]

    M3        = np.zeros((NbaseD[0], NbaseD[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), order='F')

    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
    
    mat_map   = np.asfortranarray(1/g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    ker.kernel_mass(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 1, 1, 1, 1, 1, 1, quad[0][1], quad[1][1], quad[2][1], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], M3, mat_map)
                
    indices   = np.indices((NbaseD[0], NbaseD[1], NbaseD[2], 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1))
    
    shift     = [np.arange(NbaseD) - p for NbaseD, p in zip(NbaseD, p)]
    
    row       = (NbaseD[1]*NbaseD[2]*indices[0] + NbaseD[2]*indices[1] + indices[2]).flatten()
    
    col1      = (indices[3] + shift[0][:, None, None, None, None, None])%NbaseD[0]
    col2      = (indices[4] + shift[1][None, :, None, None, None, None])%NbaseD[1]
    col3      = (indices[5] + shift[2][None, None, :, None, None, None])%NbaseD[2]

    col       = NbaseD[1]*NbaseD[2]*col1 + NbaseD[2]*col2 + col3
                
    M3 = spa.csr_matrix((M3.flatten(), (row, col.flatten())), shape=(NbaseD[0]*NbaseD[1]*NbaseD[2], NbaseD[0]*NbaseD[1]*NbaseD[2]))
    M3.eliminate_zeros()
                
    return M3
#==============================================================================================================================




#==================================================inner product in V0 (3d)====================================================
def inner_prod_V0(T, p, bc, g_sqrt, fun):
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]

    F0        = np.zeros((NbaseN[0], NbaseN[1], NbaseN[2]), order='F')

    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
                                  
    mat_f     = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    mat_map   = np.asfortranarray(g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 0, 0, 0, quad[0][1], quad[1][1], quad[2][1], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], F0, mat_f, mat_map)
                
    return F0
#==============================================================================================================================
                                  
                                  
                                  

            
#==================================================inner product in V1 (3d)====================================================
def inner_prod_V1(T, p, bc, Ginv, g_sqrt, fun):
    
    t         = [T[1:-1] for T in T]
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbase     = [[NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]]]
    basis     = [[basisD[0], basisN[1], basisN[2]], [basisN[0], basisD[1], basisN[2]], [basisN[0], basisN[1], basisD[2]]]
    ns        = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    F1        = [np.zeros((Nbase[0], Nbase[1], Nbase[2]), order='F') for Nbase in Nbase]
    
    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
    
    for a in range(3):
        
        ni1,    ni2,    ni3    = ns[a]
        bi1,    bi2,    bi3    = basis[a]
        Nbase1, Nbase2, Nbase3 = Nbase[a]
        
        for b in range(3):
            
            mat_map = np.asfortranarray(Ginv[a][b](quad_mesh[0], quad_mesh[1], quad_mesh[2]) * g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            mat_f   = np.asfortranarray(fun[b](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F1[a], mat_f, mat_map)
            
    return F1
#==============================================================================================================================
            
            


#==================================================inner product in V2 (3d)====================================================
def inner_prod_V2(T, p, bc, G, g_sqrt, fun):
    
    t         = [T[1:-1] for T in T]
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbase     = [[NbaseN[0], NbaseD[1], NbaseD[2]], [NbaseD[0], NbaseN[1], NbaseD[2]], [NbaseD[0], NbaseD[1], NbaseN[2]]]
    basis     = [[basisN[0], basisD[1], basisD[2]], [basisD[0], basisN[1], basisD[2]], [basisD[0], basisD[1], basisN[2]]]
    ns        = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    
    F2        = [np.zeros((Nbase[0], Nbase[1], Nbase[2]), order='F') for Nbase in Nbase]
    
    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
    
    for a in range(3):
        
        ni1,    ni2,    ni3    = ns[a]
        bi1,    bi2,    bi3    = basis[a]
        Nbase1, Nbase2, Nbase3 = Nbase[a]
        
        for b in range(3):
            
            mat_map = np.asfortranarray(G[a][b](quad_mesh[0], quad_mesh[1], quad_mesh[2]) / g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            mat_f   = np.asfortranarray(fun[b](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F2[a], mat_f, mat_map)
            
    return F2
#==============================================================================================================================
            
            

        
#==================================================inner product in V3 (3d)====================================================
def inner_prod_V3(T, p, bc, g_sqrt, fun):
    
    t         = [T[1:-1] for T in T]
    
    el_b      = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel       = [len(el_b) - 1 for el_b in el_b]
    NbaseN    = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD    = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc  = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad      = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad      = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisD    = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]

    F3        = np.zeros((NbaseD[0], NbaseD[1], NbaseD[2]), order='F')

    quad_mesh = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
                                  
    mat_f     = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    mat_map   = np.asfortranarray(1/g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
    
    ker.kernel_inner(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 1, 1, 1, quad[0][1], quad[1][1], quad[2][1], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], F3, mat_f, mat_map)
                
    return F3
#==============================================================================================================================        
        
        
        
            

#==================================================L2 error in V0 (3d)=========================================================
def L2_error_V0(coeff, T, p, bc, g_sqrt, fun):
    
    
    el_b               = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel                = [len(el_b) - 1 for el_b in el_b]
    Nbase              = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    
    pts1_loc, wts1_loc = np.polynomial.legendre.leggauss(p[0] + 1)
    pts2_loc, wts2_loc = np.polynomial.legendre.leggauss(p[1] + 1)
    pts3_loc, wts3_loc = np.polynomial.legendre.leggauss(p[2] + 1)
    
    pts1, wts1         = bsp.quadrature_grid(el_b[0], pts1_loc, wts1_loc)
    pts2, wts2         = bsp.quadrature_grid(el_b[1], pts2_loc, wts2_loc)
    pts3, wts3         = bsp.quadrature_grid(el_b[2], pts3_loc, wts3_loc)
    
    pts                = [pts1, pts2, pts3]
    wts                = [wts1, wts2, wts3]
    wts                = [np.asfortranarray(wts) for wts in wts]
    
    basisN             = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, pts, 0)) for T, p, pts in zip(T, p, pts)]

    error              = np.zeros((Nel[0], Nel[1], Nel[2]), order='F')

    quad               = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij')
                                  
    mat_f              = np.asfortranarray(fun(quad[0], quad[1], quad[2]))
    mat_g              = np.asfortranarray(g_sqrt(quad[0], quad[1], quad[2]))
    coeff              = np.asfortranarray(coeff)
    
    ker.kernel_L2error_V0(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], Nbase[0], Nbase[1], Nbase[2], error, mat_f, coeff, mat_g)
                                  
    error = np.sqrt(error.sum())
                
    return error
#==============================================================================================================================