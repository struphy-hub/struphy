import numpy        as np
import scipy.sparse as spa

import utilitis_FEEC.bsplines              as bsp
import utilitis_FEEC.kernels_mass          as ker
import utilitis_FEEC.evaluation            as eva
import utilitis_FEEC.spline_mappings_polar as splmap




# ================================================ mass matrix in V0 (1d) ====================================================
def mass_1d_NN(T, p, bc):
    
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b - 1)
    NbaseN           = Nel + p - bc*p
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisN           = bsp.basis_ders_on_quad_grid(T, p, pts, 0)

    M                = np.zeros((NbaseN, 2*p + 1))

    for ie in range(Nel):

        for il in range(p + 1):
            for jl in range(p + 1):

                value = 0.

                for q in range(p + 1):
                    value += wts[ie, q] * basisN[ie, il, 0, q] * basisN[ie, jl, 0, q]

                M[(ie + il)%NbaseN, p + jl - il] += value
                
    indices = np.indices((NbaseN, 2*p + 1))
    shift   = np.arange(NbaseN) - p
    
    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseN
    
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN, NbaseN))
    M.eliminate_zeros()
                
    return M    
# ============================================================================================================================




# =============================================== mass matrix in V0/V1 (1d DN) (1d) ==========================================
def mass_1d_DN(T, p, bc):
    
    t                = T[1:-1]
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b - 1)
    NbaseN           = Nel + p - bc*p
    NbaseD           = NbaseN - (1 - bc)
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisN           = bsp.basis_ders_on_quad_grid(T, p, pts, 0)
    basisD           = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)
    
    M                = np.zeros((NbaseD, 2*p + 1))

    for ie in range(Nel):

        for il in range(p):
            for jl in range(p + 1):

                value = 0.

                for q in range(p + 1):
                    value += wts[ie, q] * basisD[ie, il, 0, q] * basisN[ie, jl, 0, q]

                M[(ie + il)%NbaseD, p + jl - il] += value
                
    indices = np.indices((NbaseD, 2*p + 1))
    shift   = np.arange(NbaseD) - p

    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseN
    
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD, NbaseN))
    M.eliminate_zeros()
                
    return M
# ============================================================================================================================




# =============================================== mass matrix in V0/V1 (1d ND) (1d) ==========================================
def mass_1d_ND(T, p, bc):
    
    t                = T[1:-1]
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b - 1)
    NbaseN           = Nel + p - bc*p
    NbaseD           = NbaseN - (1 - bc)
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisN           = bsp.basis_ders_on_quad_grid(T, p, pts, 0)
    basisD           = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)
    
    M                = np.zeros((NbaseN, 2*p + 1))

    for ie in range(Nel):

        for il in range(p + 1):
            for jl in range(p):

                value = 0.

                for q in range(p + 1):
                    value += wts[ie, q] * basisN[ie, il, 0, q] * basisD[ie, jl, 0, q]

                M[(ie + il)%NbaseN, p + jl - il] += value
                
    
    indices = np.indices((NbaseN, 2*p + 1))
    shift   = np.arange(NbaseN) - p

    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseD
    
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseN, NbaseD))
    M.eliminate_zeros()
                
    return M
# ============================================================================================================================




# ================================================ mass matrix in V1 (1d) ====================================================
def mass_1d_DD(T, p, bc):
    
    t                = T[1:-1]
    
    el_b             = bsp.breakpoints(T, p)
    Nel              = len(el_b - 1)
    NbaseN           = Nel + p - bc*p
    NbaseD           = NbaseN - (1 - bc)
    
    pts_loc, wts_loc = np.polynomial.legendre.leggauss(p + 1)
    pts,     wts     = bsp.quadrature_grid(el_b, pts_loc, wts_loc)

    basisD           = bsp.basis_ders_on_quad_grid(t, p - 1, pts, 0, normalize=True)
    
    M                = np.zeros((NbaseD, 2*p + 1))

    for ie in range(Nel):

        for il in range(p):
            for jl in range(p):

                value = 0.

                for q in range(p + 1):
                    value += wts[ie, q] * basisD[ie, il, 0, q] * basisD[ie, jl, 0, q]

                M[(ie + il)%NbaseD, p + jl - il] += value
                
    
    indices = np.indices((NbaseD, 2*p + 1))
    shift   = np.arange(NbaseD) - p

    row     = indices[0].flatten()
    col     = (indices[1] + shift[:, None])%NbaseD
    
    M       = spa.csr_matrix((M.flatten(), (row, col.flatten())), shape=(NbaseD, NbaseD))
    M.eliminate_zeros()
                
    return M
# ============================================================================================================================





# ================================================ mass matrix in V0 (2d) ====================================================
def mass_V0_2d(T, p, bc, mapping):
    
    el_b       = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel        = [len(el_b) - 1 for el_b in el_b]
    NbaseN     = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    
    quad_loc   = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad       = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad       = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]

    M0         = np.zeros((NbaseN[0], NbaseN[1], 2*p[0] + 1, 2*p[1] + 1), order='F')
                                  
    spline_map = splmap.discrete_mapping_2d(T, p, bc, mapping)
    mat_map    = np.asfortranarray(spline_map.jacobian_determinant([quad[0][0].flatten(), quad[1][0].flatten()]))
    
    ker.kernel_mass_2d(Nel[0], Nel[1], p[0], p[1], p[0] + 1, p[1] + 1, 0, 0, 0, 0, quad[0][1], quad[1][1], basisN[0], basisN[1], basisN[0], basisN[1], NbaseN[0], NbaseN[1], M0, mat_map)
                
    indices   = np.indices((NbaseN[0], NbaseN[1], 2*p[0] + 1, 2*p[1] + 1))
    
    shift     = [np.arange(NbaseN) - p for NbaseN, p in zip(NbaseN, p)]
    
    row       = (NbaseN[1]*indices[0] + indices[1]).flatten()
    
    col1      = (indices[2] + shift[0][:, None, None, None])%NbaseN[0]
    col2      = (indices[3] + shift[1][None, :, None, None])%NbaseN[1]

    col       = NbaseN[1]*col1 + col2
                
    M0        = spa.csr_matrix((M0.flatten(), (row, col.flatten())), shape=(NbaseN[0]*NbaseN[1], NbaseN[0]*NbaseN[1]))
    M0.eliminate_zeros()
                
    return M0
# ============================================================================================================================



# ================================================ mass matrix in V1 (2d) ====================================================
def mass_V1_2d_curl(T, p, bc, mapping):
    '''
    Corresponds to the sequence grad --> curl
    '''
    
    t          = [T[1:-1] for T in T]
    
    el_b       = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel        = [len(el_b) - 1 for el_b in el_b]
    NbaseN     = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD     = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc   = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad       = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad       = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbi1       = [NbaseD[0], NbaseN[0], NbaseN[0]]
    Nbi2       = [NbaseN[1], NbaseD[1], NbaseD[1]]
    
    Nbj1       = [NbaseD[0], NbaseD[0], NbaseN[0]]
    Nbj2       = [NbaseN[1], NbaseN[1], NbaseD[1]]
    
    basis      = [[basisD[0], basisN[1]], [basisN[0], basisD[1]]]
    ns         = [[1, 0], [0, 1]]
    
    M1         = [np.zeros((Nbi1, Nbi2, 2*p[0] + 1, 2*p[1] + 1), order='F') for Nbi1, Nbi2 in zip(Nbi1, Nbi2)]
    
    spline_map = splmap.discrete_mapping_2d(T, p, bc, mapping)
    components = ['00', '10', '11']
    
    counter    = 0
    
    for a in range(2):
        for b in range(a + 1):
            
            mat_map = np.asfortranarray(spline_map.metric_tensor_inverse([quad[0][0].flatten(), quad[1][0].flatten()], components[counter]) * spline_map.jacobian_determinant([quad[0][0].flatten(), quad[1][0].flatten()]))
            
            ni1, ni2 = ns[a]
            nj1, nj2 = ns[b]
            
            bi1, bi2 = basis[a]
            bj1, bj2 = basis[b]
            
            ker.kernel_mass_2d(Nel[0], Nel[1], p[0], p[1], p[0] + 1, p[1] + 1, ni1, ni2, nj1, nj2, quad[0][1], quad[1][1], bi1, bi2, bj1, bj2, Nbi1[counter], Nbi2[counter], M1[counter], mat_map)
            
            indices = np.indices((Nbi1[counter], Nbi2[counter], 2*p[0] + 1, 2*p[1] + 1))
            
            shift1  = np.arange(Nbi1[counter]) - p[0]
            shift2  = np.arange(Nbi2[counter]) - p[1]
            
            row     = (Nbi2[counter]*indices[0] + indices[1]).flatten()
            
            col1    = (indices[2] + shift1[:, None, None, None])%Nbj1[counter]
            col2    = (indices[3] + shift2[None, :, None, None])%Nbj2[counter]
            
            col     = Nbj2[counter]*col1 + col2
            
            M1[counter] = spa.csr_matrix((M1[counter].flatten(), (row, col.flatten())), shape=(Nbi1[counter]*Nbi2[counter], Nbj1[counter]*Nbj2[counter]))
            M1[counter].eliminate_zeros()
            
            counter += 1
    
    M1 = spa.bmat([[M1[0], M1[1].T], [M1[1], M1[2]]], format='csr')
            
    return M1
# ============================================================================================================================





# ================================================ mass matrix in V0 (3d) ====================================================
def mass_V0_3d(T, p, bc, g_sqrt):
    
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
    
    ker.kernel_mass_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 0, 0, 0, 0, 0, 0, quad[0][1], quad[1][1], quad[2][1], basisN[0], basisN[1], basisN[2], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], M0, mat_map)
                
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
# ============================================================================================================================





# ================================================ mass matrix in V1 (3d) ====================================================
def mass_V1_3d(T, p, bc, Ginv, g_sqrt):
    
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
            
            ker.kernel_mass_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, nj1, nj2, nj3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M1[counter], mat_map)
            
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
# ============================================================================================================================







# ================================================ mass matrix in V2 (3d) ====================================================
def mass_V2_3d(T, p, bc, G, g_sqrt):
    
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
            
            ker.kernel_mass_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, nj1, nj2, nj3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M2[counter], mat_map)
            
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
# ============================================================================================================================





# ================================================ mass matrix in V3 (3d) ====================================================
def mass_V3_3d(T, p, bc, g_sqrt):
    
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
    
    ker.kernel_mass_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 1, 1, 1, 1, 1, 1, quad[0][1], quad[1][1], quad[2][1], basisD[0], basisD[1], basisD[2], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], M3, mat_map)
                
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
# ============================================================================================================================




# ================================================ inner product in V0 (2d) ==================================================
def inner_prod_V0_2d(T, p, bc, mapping, fun):
    
    el_b       = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel        = [len(el_b) - 1 for el_b in el_b]
    NbaseN     = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    
    quad_loc   = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad       = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad       = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]

    F0         = np.zeros((NbaseN[0], NbaseN[1]), order='F')

    spline_map = splmap.discrete_mapping_2d(T, p, bc, mapping)
    mat_map    = np.asfortranarray(spline_map.jacobian_determinant([quad[0][0].flatten(), quad[1][0].flatten()]))
    
    quad_mesh  = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), indexing='ij')
    mat_f      = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1]))
    
    ker.kernel_inner_2d(Nel[0], Nel[1], p[0], p[1], p[0] + 1, p[1] + 1, 0, 0, quad[0][1], quad[1][1], basisN[0], basisN[1], NbaseN[0], NbaseN[1], F0, mat_f, mat_map)
                
    return F0
# ============================================================================================================================



# ================================================ inner product in V1 (2d) ==================================================
def inner_prod_V1_2d_curl(T, p, bc, mapping, fun):
    '''
    Corresponds to the sequence grad --> curl
    '''
    
    t          = [T[1:-1] for T in T]
    
    el_b       = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel        = [len(el_b) - 1 for el_b in el_b]
    NbaseN     = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD     = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc   = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad       = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad       = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbase      = [[NbaseD[0], NbaseN[1]], [NbaseN[0], NbaseD[1]]]
    basis      = [[basisD[0], basisN[1]], [basisN[0], basisD[1]]]
    ns         = [[1, 0], [0, 1]]
    
    F1         = [np.zeros((Nbase[0], Nbase[1]), order='F') for Nbase in Nbase]
    
    quad_mesh  = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), indexing='ij')
    
    spline_map = splmap.discrete_mapping_2d(T, p, bc, mapping)
    components = ['00', '01', '10', '11']
    
    counter    = 0
    
    for a in range(2):
        
        ni1,    ni2    = ns[a]
        bi1,    bi2    = basis[a]
        Nbase1, Nbase2 = Nbase[a]
        
        for b in range(2):
            
            mat_map = np.asfortranarray(spline_map.metric_tensor_inverse([quad[0][0].flatten(), quad[1][0].flatten()], components[counter]) * spline_map.jacobian_determinant([quad[0][0].flatten(), quad[1][0].flatten()]))
            
            mat_f   = np.asfortranarray(fun[b](quad_mesh[0], quad_mesh[1]))
            
            ker.kernel_inner_2d(Nel[0], Nel[1], p[0], p[1], p[0] + 1, p[1] + 1, ni1, ni2, quad[0][1], quad[1][1], bi1, bi2, Nbase1, Nbase2, F1[a], mat_f, mat_map)
            
            counter += 1
            
    return F1
# ============================================================================================================================






# ================================================ inner product in V0 (3d) ==================================================
def inner_prod_V0_3d(T, p, bc, g_sqrt, fun):
    
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
    
    ker.kernel_inner_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 0, 0, 0, quad[0][1], quad[1][1], quad[2][1], basisN[0], basisN[1], basisN[2], NbaseN[0], NbaseN[1], NbaseN[2], F0, mat_f, mat_map)
                
    return F0
# ============================================================================================================================
                                  
                                  
                                  

            
# =============================================== inner product in V1 (3d) ===================================================
def inner_prod_V1_3d(T, p, bc, Ginv, g_sqrt, fun):
    
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
            
            ker.kernel_inner_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F1[a], mat_f, mat_map)
            
    return F1
# ============================================================================================================================
            
            


# =============================================== inner product in V2 (3d) ===================================================
def inner_prod_V2_3d(T, p, bc, G, g_sqrt, fun):
    
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
            
            ker.kernel_inner_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F2[a], mat_f, mat_map)
            
    return F2
# ============================================================================================================================
            
            

        
# =============================================== inner product in V3 (3d) ===================================================
def inner_prod_V3_3d(T, p, bc, g_sqrt, fun):
    
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
    
    ker.kernel_inner_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, 1, 1, 1, quad[0][1], quad[1][1], quad[2][1], basisD[0], basisD[1], basisD[2], NbaseD[0], NbaseD[1], NbaseD[2], F3, mat_f, mat_map)
                
    return F3
# ============================================================================================================================        
        
    
# ================================================= L2 error in V0 (2d) ======================================================
def L2_error_V0_2d(coeff, T, p, bc, mapping, fun):
    
    el_b       = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel        = [len(el_b) - 1 for el_b in el_b]
    NbaseN     = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    
    quad_loc   = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad       = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad       = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]

    error      = np.zeros((Nel[0], Nel[1]), order='F')

    spline_map = splmap.discrete_mapping_2d(T, p, bc, mapping)
    mat_map    = np.asfortranarray(spline_map.jacobian_determinant([quad[0][0].flatten(), quad[1][0].flatten()]))
    
    quad_mesh  = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), indexing='ij')
    mat_f      = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1]))
    coeff      = np.asfortranarray(coeff)
    
    ker.kernel_l2error_v0_2d(Nel[0], Nel[1], p[0], p[1], p[0] + 1, p[1] + 1, quad[0][1], quad[1][1], basisN[0], basisN[1], NbaseN[0], NbaseN[1], error, mat_f, coeff, mat_map)
                                  
    error      = np.sqrt(error.sum())
                
    return error
# ============================================================================================================================
        
            

# =============================================== L2 error in V0 (3d) ========================================================
def L2_error_V0_3d(coeff, T, p, bc, g_sqrt, fun):
    
    
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
    
    ker.kernel_l2error_v0_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, wts[0], wts[1], wts[2], basisN[0], basisN[1], basisN[2], Nbase[0], Nbase[1], Nbase[2], error, mat_f, coeff, mat_g)
                                  
    error = np.sqrt(error.sum())
                
    return error
# ============================================================================================================================



# ==================================== inner product in V1 (3d) of equilibrium Hall term =====================================
def inner_prod_V1_jh0(T, p, bc, Ginv, DFinv, g_sqrt, b1, b2, b3, Beq, jheq):
    '''
    jheq = [jheq_x, jheq_y, jheq_z] on physical domain!!
    b1, b2, b3 must be flattened!
    '''
    
    t          = [T[1:-1] for T in T]
    
    el_b       = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel        = [len(el_b) - 1 for el_b in el_b]
    NbaseN     = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD     = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc   = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad       = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad       = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbase      = [[NbaseD[0], NbaseN[1], NbaseN[2]], [NbaseN[0], NbaseD[1], NbaseN[2]], [NbaseN[0], NbaseN[1], NbaseD[2]]]
    basis      = [[basisD[0], basisN[1], basisN[2]], [basisN[0], basisD[1], basisN[2]], [basisN[0], basisN[1], basisD[2]]]
    ns         = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    F1         = [np.zeros((Nbase[0], Nbase[1], Nbase[2]), order='F') for Nbase in Nbase]
    
    quad_mesh  = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
    
    B1, B2, B3 = eva.FEM_field_V2_3d([b1, b2, b3], [quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten()], T, p, bc)
    
    B1 += Beq[0]
    B2 += Beq[1]
    B3 += Beq[2]
    
    B1 = B1.reshape((p[0] + 1)*Nel[0], (p[1] + 1)*Nel[1], (p[2] + 1)*Nel[2])
    B2 = B2.reshape((p[0] + 1)*Nel[0], (p[1] + 1)*Nel[1], (p[2] + 1)*Nel[2])
    B3 = B3.reshape((p[0] + 1)*Nel[0], (p[1] + 1)*Nel[1], (p[2] + 1)*Nel[2])
    
    for a in range(3):
        
        ni1,    ni2,    ni3    = ns[a]
        bi1,    bi2,    bi3    = basis[a]
        Nbase1, Nbase2, Nbase3 = Nbase[a]
        
        for b in range(3):
            
            mat_map = np.asfortranarray(Ginv[a][b](quad_mesh[0], quad_mesh[1], quad_mesh[2]) * g_sqrt(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            if b == 0:
                mat_f = np.asfortranarray(B2*(DFinv[2][0](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[0] + DFinv[2][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[1] + DFinv[2][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[2]) - B3*(DFinv[1][0](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[0] + DFinv[1][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[1] + DFinv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[2]))
                
            elif b == 1:
                mat_f = np.asfortranarray(B3*(DFinv[0][0](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[0] + DFinv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[1] + DFinv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[2]) - B1*(DFinv[2][0](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[0] + DFinv[2][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[1] + DFinv[2][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[2]))
                
            elif b == 2:
                mat_f = np.asfortranarray(B1*(DFinv[1][0](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[0] + DFinv[1][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[1] + DFinv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[2]) - B2*(DFinv[0][0](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[0] + DFinv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[1] + DFinv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*jheq[2]))
            
            ker.kernel_inner_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, Nbase1, Nbase2, Nbase3, F1[a], mat_f, mat_map)
            
    return F1
# ============================================================================================================================



# ================================ mass matrix in V1 (3d) of equilibrium charge density ======================================
def mass_V1_nh0(T, p, bc, Ginv, b1, b2, b3, Beq, nh0):
    '''
    nh0 on logical domain!!
    b1, b2, b3 must be flattened! 
    '''
    
    t          = [T[1:-1] for T in T]
    
    el_b       = [bsp.breakpoints(T, p) for T, p in zip(T, p)]
    Nel        = [len(el_b) - 1 for el_b in el_b]
    NbaseN     = [Nel + p - bc*p for Nel, p, bc in zip(Nel, p, bc)]
    NbaseD     = [NbaseN - (1 - bc) for NbaseN, bc in zip(NbaseN, bc)]
    
    quad_loc   = [np.polynomial.legendre.leggauss(p + 1) for p in p]
    quad       = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(el_b, quad_loc)]
    
    quad       = [(quad[0], np.asfortranarray(quad[1])) for quad in quad]
    
    basisN     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(T, p, quad)]
    basisD     = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(t, p, quad)]
    
    Nbi1       = [NbaseN[0], NbaseN[0], NbaseN[0]]
    Nbi2       = [NbaseD[1], NbaseN[1], NbaseN[1]]
    Nbi3       = [NbaseN[2], NbaseD[2], NbaseD[2]]
    
    Nbj1       = [NbaseD[0], NbaseD[0], NbaseN[0]]
    Nbj2       = [NbaseN[1], NbaseN[1], NbaseD[1]]
    Nbj3       = [NbaseN[2], NbaseN[2], NbaseN[2]]
    
    basis      = [[basisD[0], basisN[1], basisN[2]], [basisN[0], basisD[1], basisN[2]], [basisN[0], basisN[1], basisD[2]]]
    ns         = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    M1         = [np.zeros((Nbi1, Nbi2, Nbi3, 2*p[0] + 1, 2*p[1] + 1, 2*p[2] + 1), order='F') for Nbi1, Nbi2, Nbi3 in zip(Nbi1, Nbi2, Nbi3)]
    
    quad_mesh  = np.meshgrid(quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten(), indexing='ij')
    
    B1, B2, B3 = eva.FEM_field_V2_3d([b1, b2, b3], [quad[0][0].flatten(), quad[1][0].flatten(), quad[2][0].flatten()], T, p, bc)
    
    B1 += Beq[0]
    B2 += Beq[1]
    B3 += Beq[2]
    
    B1 = B1.reshape((p[0] + 1)*Nel[0], (p[1] + 1)*Nel[1], (p[2] + 1)*Nel[2])
    B2 = B2.reshape((p[0] + 1)*Nel[0], (p[1] + 1)*Nel[1], (p[2] + 1)*Nel[2])
    B3 = B3.reshape((p[0] + 1)*Nel[0], (p[1] + 1)*Nel[1], (p[2] + 1)*Nel[2])
    
    counter    = 0
    
    for a in range(1, 3):
        for b in range(0, a):
            
            if   (a == 1) and (b == 0):
                mat_f = nh0*np.asfortranarray(-Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B3*Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B2*Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[1][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B3*Ginv[0][0](quad_mesh[0], quad_mesh[1], quad_mesh[2]) - Ginv[1][1](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B1*Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2]) - Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B2*Ginv[0][0](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B1*Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
                
            elif (a == 2) and (b == 0):
                mat_f = nh0*np.asfortranarray(-Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B3*Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B2*Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B3*Ginv[0][0](quad_mesh[0], quad_mesh[1], quad_mesh[2]) - Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B1*Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2]) - Ginv[2][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B2*Ginv[0][0](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[2][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B1*Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
                
            elif (a == 2) and (b == 1):
                mat_f = nh0*np.asfortranarray(-Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B3*Ginv[1][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[0][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B2*Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B3*Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]) - Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B1*Ginv[1][2](quad_mesh[0], quad_mesh[1], quad_mesh[2]) - Ginv[2][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B2*Ginv[0][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]) + Ginv[2][2](quad_mesh[0], quad_mesh[1], quad_mesh[2])*B1*Ginv[1][1](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
            
            ni1, ni2, ni3 = ns[a]
            nj1, nj2, nj3 = ns[b]
            
            bi1, bi2, bi3 = basis[a]
            bj1, bj2, bj3 = basis[b]
            
            ker.kernel_mass_3d(Nel[0], Nel[1], Nel[2], p[0], p[1], p[2], p[0] + 1, p[1] + 1, p[2] + 1, ni1, ni2, ni3, nj1, nj2, nj3, quad[0][1], quad[1][1], quad[2][1], bi1, bi2, bi3, bj1, bj2, bj3, Nbi1[counter], Nbi2[counter], Nbi3[counter], M1[counter], mat_f)
            
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
    
    M1 = spa.bmat([[None, -M1[0].T, -M1[1].T], [M1[0], None, -M1[2].T], [M1[1], M1[2], None]], format='csr')
            
    return M1
# ============================================================================================================================