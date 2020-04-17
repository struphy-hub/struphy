import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_FEEC.bsplines     as bsp
import hylife.utilitis_FEEC.kernels_mass as ker



class l2_errors_3d:
    
    def __init__(self, T, p, bc, n_quad, kind_map, params_map):
        
        self.T          = T                    
        self.p          = p
        self.bc         = bc
        self.n_quad     = n_quad
        self.kind_map   = kind_map
        self.params_map = params_map
        
        self.t        = [T[1:-1] for T in self.T]
        self.el_b     = [bsp.breakpoints(T, p) for T, p in zip(self.T, self.p)]
        self.Nel      = [len(el_b) - 1 for el_b in self.el_b]
        self.NbaseN   = [Nel + p - bc*p for Nel, p, bc in zip(self.Nel, self.p, self.bc)]
        self.NbaseD   = [NbaseN - (1 - bc) for NbaseN, bc in zip(self.NbaseN, self.bc)]
        self.quad_loc = [np.polynomial.legendre.leggauss(n_quad) for n_quad in self.n_quad]
        
        self.quad     = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(self.el_b, self.quad_loc)]
        self.quad     = [(quad[0], np.asfortranarray(quad[1])) for quad in self.quad]

        self.basisN   = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(self.T, self.p, self.quad)]
        self.basisD   = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(self.t, self.p, self.quad)]
        
        self.n_pts    = [quad[0].flatten().size for quad in self.quad]
        
        
    def l2_error_V0(self, coeff, fun):
        
        # evaluate function at quadrature points
        quad_mesh = np.meshgrid(self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), indexing='ij')
        mat_f = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        
        # evaluate mapping at quadrature points
        mat_g = np.zeros(self.n_pts, dtype=float, order='F')
        
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=1, kind_map=self.kind_map, params=self.params_map)
        
        # compute error
        error = np.zeros(self.Nel, dtype=float, order='F')
        
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [0, 0, 0], [0, 0, 0], self.basisN[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisN[2], [self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]], [self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]], error, mat_f, mat_f, np.asfortranarray(coeff), np.asfortranarray(coeff), mat_g)
        
        return np.sqrt(error.sum())
    
    
    
    def l2_error_V1(self, coeff, fun):
        
        # evaluate function at quadrature points
        quad_mesh = np.meshgrid(self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), indexing='ij')
        
        mat_f1 = np.asfortranarray(fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        mat_f2 = np.asfortranarray(fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        mat_f3 = np.asfortranarray(fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        
        # global error
        error = np.zeros(self.Nel, dtype=float, order='F')
                                   
        # evaluate mapping
        mat_g = np.zeros(self.n_pts, dtype=float, order='F')
        
        # f1 * G^11 * sqrt(g) * f1
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=11, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [1, 0, 0], [1, 0, 0], self.basisD[0], self.basisN[1], self.basisN[2], self.basisD[0], self.basisN[1], self.basisN[2], [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], error, mat_f1, mat_f1, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[0]), mat_g)
                                   
        # 2 * f1 * G^12 * sqrt(g) * f2
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=12, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [1, 0, 0], [0, 1, 0], self.basisD[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisD[1], self.basisN[2], [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], error, mat_f1, mat_f2, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[1]), 2*mat_g)                        
                                   
        # 2 * f1 * G^13 * sqrt(g) * f3
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=14, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [1, 0, 0], [0, 0, 1], self.basisD[0], self.basisN[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisD[2], [self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]], [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], error, mat_f1, mat_f3, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[2]), 2*mat_g)
                                   
        # f2 * G^22 * sqrt(g) * f2
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=13, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [0, 1, 0], [0, 1, 0], self.basisN[0], self.basisD[1], self.basisN[2], self.basisN[0], self.basisD[1], self.basisN[2], [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], error, mat_f2, mat_f2, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[1]), mat_g)
                                                  
        # 2 * f2 * G^23 * sqrt(g) * f3
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=15, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [0, 1, 0], [0, 0, 1], self.basisN[0], self.basisD[1], self.basisN[2], self.basisN[0], self.basisN[1], self.basisD[2], [self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]], [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], error, mat_f2, mat_f3, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[2]), 2*mat_g)
                                   
        # f3 * G^33 * sqrt(g) * f3
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=16, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [0, 0, 1], [0, 0, 1], self.basisN[0], self.basisN[1], self.basisD[2], self.basisN[0], self.basisN[1], self.basisD[2], [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], [self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]], error, mat_f3, mat_f3, np.asfortranarray(coeff[2]), np.asfortranarray(coeff[2]), mat_g)
                                   
        return np.sqrt(error.sum())
    
    
    def l2_error_V2(self, coeff, fun):
        
        # evaluate function at quadrature points
        quad_mesh = np.meshgrid(self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), indexing='ij')
        
        mat_f1 = np.asfortranarray(fun[0](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        mat_f2 = np.asfortranarray(fun[1](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        mat_f3 = np.asfortranarray(fun[2](quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        
        # global error
        error = np.zeros(self.Nel, dtype=float, order='F')
                                   
        # evaluate mapping
        mat_g = np.zeros(self.n_pts, dtype=float, order='F')
        
        # f1 * G_11 / sqrt(g) * f1
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=21, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [0, 1, 1], [0, 1, 1], self.basisN[0], self.basisD[1], self.basisD[2], self.basisN[0], self.basisD[1], self.basisD[2], [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], error, mat_f1, mat_f1, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[0]), mat_g)
                                   
        # 2 * f1 * G_12 / sqrt(g) * f2
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=22, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [0, 1, 1], [1, 0, 1], self.basisN[0], self.basisD[1], self.basisD[2], self.basisD[0], self.basisN[1], self.basisD[2], [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], error, mat_f1, mat_f2, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[1]), 2*mat_g)                        
                                   
        # 2 * f1 * G_13 / sqrt(g) * f3
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=24, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [0, 1, 1], [1, 1, 0], self.basisN[0], self.basisD[1], self.basisD[2], self.basisD[0], self.basisD[1], self.basisN[2], [self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]], [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], error, mat_f1, mat_f3, np.asfortranarray(coeff[0]), np.asfortranarray(coeff[2]), 2*mat_g)
                                   
        # f2 * G_22 / sqrt(g) * f2
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=23, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [1, 0, 1], [1, 0, 1], self.basisD[0], self.basisN[1], self.basisD[2], self.basisD[0], self.basisN[1], self.basisD[2], [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], error, mat_f2, mat_f2, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[1]), mat_g)
                                                  
        # 2 * f2 * G_23 / sqrt(g) * f3
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=25, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [1, 0, 1], [1, 1, 0], self.basisD[0], self.basisN[1], self.basisD[2], self.basisD[0], self.basisD[1], self.basisN[2], [self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]], [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], error, mat_f2, mat_f3, np.asfortranarray(coeff[1]), np.asfortranarray(coeff[2]), 2*mat_g)
                                   
        # f3 * G_33 / sqrt(g) * f3
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=26, kind_map=self.kind_map, params=self.params_map)
                                   
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [1, 1, 1], [1, 1, 0], self.basisD[0], self.basisD[1], self.basisN[2], self.basisD[0], self.basisD[1], self.basisN[2], [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], [self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]], error, mat_f3, mat_f3, np.asfortranarray(coeff[2]), np.asfortranarray(coeff[2]), mat_g)
                                   
        return np.sqrt(error.sum())
    
    
    def l2_error_V3(self, coeff, fun):
        
        # evaluate function at quadrature points
        quad_mesh = np.meshgrid(self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), indexing='ij')
        mat_f = np.asfortranarray(fun(quad_mesh[0], quad_mesh[1], quad_mesh[2]))
        
        # evaluate mapping at quadrature points
        mat_g = np.zeros(self.n_pts, dtype=float, order='F')
        
        ker.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), mat_g, kind_fun=2, kind_map=self.kind_map, params=self.params_map)
        
        # compute error
        error = np.zeros(self.Nel, dtype=float, order='F')
        
        ker.kernel_l2error_3d(self.Nel, self.p, self.n_quad, self.quad[0][1], self.quad[1][1], self.quad[2][1], [1, 1, 1], [1, 1, 1], self.basisD[0], self.basisD[1], self.basisD[2], self.basisD[0], self.basisD[1], self.basisD[2], [self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]], [self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]], error, mat_f, mat_f, np.asfortranarray(coeff), np.asfortranarray(coeff), mat_g)
        
        return np.sqrt(error.sum())