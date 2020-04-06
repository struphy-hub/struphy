import  numpy                 as np
import  scipy.sparse          as spa

from ..utilitis_FEEC import bsplines       as bsp
from ..utilitis_FEEC import kernels_mass            as ker_mass
from ..utilitis_FEEC import kernels_control_variate as ker_cv
from ..utilitis_FEEC import evaluation              as eva




class terms_control_variate:
    
    def __init__(self, T, p, bc, kind_map, params_map):
        
        self.T        = T
        self.p        = p
        self.bc       = bc
        
        self.t        = [T[1:-1] for T in self.T]
        self.el_b     = [bsp.breakpoints(T, p) for T, p in zip(self.T, self.p)]
        self.Nel      = [len(el_b) - 1 for el_b in self.el_b]
        self.NbaseN   = [Nel + p - bc*p for Nel, p, bc in zip(self.Nel, self.p, self.bc)]
        self.NbaseD   = [NbaseN - (1 - bc) for NbaseN, bc in zip(self.NbaseN, self.bc)]
        self.n_quad   = [p + 1 for p in self.p]
        self.quad_loc = [np.polynomial.legendre.leggauss(n_quad) for n_quad in self.n_quad]
        
        self.quad     = [bsp.quadrature_grid(el_b, quad_loc[0], quad_loc[1]) for el_b, quad_loc in zip(self.el_b, self.quad_loc)]
        self.quad     = [(quad[0], np.asfortranarray(quad[1])) for quad in self.quad]

        self.basisN   = [np.asfortranarray(bsp.basis_ders_on_quad_grid(T, p, quad[0], 0)) for T, p, quad in zip(self.T, self.p, self.quad)]
        self.basisD   = [np.asfortranarray(bsp.basis_ders_on_quad_grid(t, p - 1, quad[0], 0, normalize=True)) for t, p, quad in zip(self.t, self.p, self.quad)]
        
        self.n_pts    = [quad[0].flatten().size for quad in self.quad]
        
        # evaluation of DF^T * jh_eq_phys at quadrature points
        self.mat_jh1  = np.zeros((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        self.mat_jh2  = np.zeros((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        self.mat_jh3  = np.zeros((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_jh1, kind_fun=1, kind_map=kind_map, params_map=params_map)
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_jh2, kind_fun=2, kind_map=kind_map, params_map=params_map)
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_jh3, kind_fun=3, kind_map=kind_map, params_map=params_map)
        
        # evaluation of nh_eq_phys at quadrature points
        self.mat_nh   = np.zeros((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_nh, kind_fun=4, kind_map=kind_map, params_map=params_map)
        
        # evaluation of G / sqrt(g) at quadrature points
        self.mat_g11   = np.empty((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        self.mat_g21   = np.empty((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        self.mat_g22   = np.empty((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        self.mat_g31   = np.empty((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        self.mat_g32   = np.empty((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        self.mat_g33   = np.empty((self.n_pts[0], self.n_pts[1], self.n_pts[2]), dtype=float, order='F')
        
        ker_mass.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_g11, kind_fun=21, kind_map=kind_map, params=params_map)
        ker_mass.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_g21, kind_fun=22, kind_map=kind_map, params=params_map)
        ker_mass.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_g22, kind_fun=23, kind_map=kind_map, params=params_map)
        ker_mass.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_g31, kind_fun=24, kind_map=kind_map, params=params_map)
        ker_mass.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_g32, kind_fun=25, kind_map=kind_map, params=params_map)
        ker_mass.kernel_eva_3d(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), self.mat_g33, kind_fun=26, kind_map=kind_map, params=params_map)
        
        
    # ===== inner product in V1 (3d) of (B x jh_eq) - term =======
    def inner_prod_V1_jh_eq(self, b1, b2, b3, kind_map, params_map):

        F1 = np.zeros((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), order='F')
        F2 = np.zeros((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), order='F')
        F3 = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), order='F')
        
        
        # evaluation of total magnetic field at quadrature points (perturbed + equilibrium)
        B1, B2, B3 = eva.FEM_field_V2_3d([b1, b2, b3], [self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten()], self.T, self.p, self.bc)
        
        B1 = np.asfortranarray(B1)
        B2 = np.asfortranarray(B2)
        B3 = np.asfortranarray(B3)
        
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), B1, kind_fun=11, kind_map=kind_map, params_map=params_map)
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), B2, kind_fun=12, kind_map=kind_map, params_map=params_map)
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), B3, kind_fun=13, kind_map=kind_map, params_map=params_map)
        
        ker_cv.kernel_inner_3d(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 0, self.quad[0][1], self.quad[1][1], self.quad[2][1], self.basisD[0], self.basisN[1], self.basisN[2], self.NbaseD[0], self.NbaseN[1], self.NbaseN[1], F1, self.mat_jh3*(self.mat_g21*B1 + self.mat_g22*B2 + self.mat_g32*B3) - self.mat_jh2*(self.mat_g31*B1 + self.mat_g32*B2 + self.mat_g33*B3))
        
        ker_cv.kernel_inner_3d(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 0, self.quad[0][1], self.quad[1][1], self.quad[2][1], self.basisN[0], self.basisD[1], self.basisN[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[1], F2, self.mat_jh1*(self.mat_g31*B1 + self.mat_g32*B2 + self.mat_g33*B3) - self.mat_jh3*(self.mat_g11*B1 + self.mat_g21*B2 + self.mat_g31*B3))
        
        ker_cv.kernel_inner_3d(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 1, 0, 0, self.quad[0][1], self.quad[1][1], self.quad[2][1], self.basisN[0], self.basisN[1], self.basisD[2], self.NbaseN[0], self.NbaseN[1], self.NbaseD[1], F3, self.mat_jh2*(self.mat_g11*B1 + self.mat_g21*B2 + self.mat_g31*B3) - self.mat_jh1*(self.mat_g21*B1 + self.mat_g22*B2 + self.mat_g32*B3))
        
        return F1, F2, F3
    
    # ===== mass matrix in V1 (3d) of (rhoh_eq * (U x B)) - term =======
    def mass_V1_nh_eq(self, b1, b2, b3, kind_map, params_map):
        
        M21 = np.zeros((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), order='F')
        M31 = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), order='F')
        M32 = np.zeros((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), order='F')
        
        # evaluation of total magnetic field at quadrature points (perturbed + equilibrium)
        B1, B2, B3 = eva.FEM_field_V2_3d([b1, b2, b3], [self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten()], self.T, self.p, self.bc)
        
        B1 = np.asfortranarray(B1)
        B2 = np.asfortranarray(B2)
        B3 = np.asfortranarray(B3)
        
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), B1, kind_fun=11, kind_map=kind_map, params_map=params_map)
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), B2, kind_fun=12, kind_map=kind_map, params_map=params_map)
        ker_cv.kernel_eva(self.n_pts, self.quad[0][0].flatten(), self.quad[1][0].flatten(), self.quad[2][0].flatten(), B3, kind_fun=13, kind_map=kind_map, params_map=params_map)
        
        
        ker_cv.kernel_mass_3d(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 1, 0, 1, 0, 0, self.quad[0][1], self.quad[1][1], self.quad[2][1], self.basisN[0], self.basisD[1], self.basisN[2], self.basisD[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], M21, self.mat_nh*(self.mat_g31*B1 + self.mat_g32*B2 + self.mat_g33*B3))
        
        ker_cv.kernel_mass_3d(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 1, 1, 0, 0, self.quad[0][1], self.quad[1][1], self.quad[2][1], self.basisN[0], self.basisN[1], self.basisD[2], self.basisD[0], self.basisN[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], M31, -self.mat_nh*(self.mat_g21*B1 + self.mat_g22*B2 + self.mat_g32*B3))
        
        ker_cv.kernel_mass_3d(self.Nel[0], self.Nel[1], self.Nel[2], self.p[0], self.p[1], self.p[2], self.n_quad[0], self.n_quad[1], self.n_quad[2], 0, 0, 1, 0, 1, 0, self.quad[0][1], self.quad[1][1], self.quad[2][1], self.basisN[0], self.basisN[1], self.basisD[2], self.basisN[0], self.basisD[1], self.basisN[2], self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], M32, self.mat_nh*(self.mat_g11*B1 + self.mat_g21*B2 + self.mat_g31*B3))
        
        # conversion to sparse matrices
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseD[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        M21     = spa.csr_matrix((M21.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))
        M21.eliminate_zeros()
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseD[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        M31     = spa.csr_matrix((M31.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))
        M31.eliminate_zeros()
        
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseD[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        M32     = spa.csr_matrix((M32.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))
        M32.eliminate_zeros()
        
        M1 = spa.bmat([[None, -M21.T, -M31.T], [M21, None, -M32.T], [M31, M32, None]], format='csr')
        
        return M1