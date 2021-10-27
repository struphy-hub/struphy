# coding: utf-8
#
# Copyright 2020 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for control variates in delta-f method for current coupling scheme.
"""


import numpy        as np
import scipy.sparse as spa


import hylife.utilitis_FEEC.basics.kernels_3d as ker3


class terms_control_variate:
    """
    Contains method for computing the terms (B x jh_eq) and -(rhoh_eq * (B x U)).
    
    Parameters
    ----------
    tensor_space_FEM : tensor_spline_space
        3D tensor product B-spline space
        
    basis_u : int
        representation of MHD bulk velocity
        
    eq_pic : equilibrium_pic
        object for kinetic equilibrium data
    """
    
    def __init__(self, tensor_space_FEM, basis_u, eq_pic):
        
        self.space   = tensor_space_FEM         # 3D B-spline space  
        self.basis_u = basis_u                  # representation of MHD bulk velocity
        
        # physical (x, y, z) components of hot ion current 
        jh_phys = [eq_pic.jh_eq_x, eq_pic.jh_eq_y, eq_pic.jh_eq_z]
        
        if self.space.dim == 2:
            eta3 = np.array([0.])
        else:
            eta3 = self.space.pts[2].flatten()
        
        # evaluation of Jacobian determinant at quadrature points
        det_DF = abs(eq_pic.domain.evaluate(self.space.pts[0].flatten(), self.space.pts[1].flatten(), eta3, 'det_df'))
        
        # evaluation of DF^(-1) * jh_eq_phys (* |det(DF)|) at quadrature points
        self.mat_jh1 = eq_pic.domain.pull(jh_phys, self.space.pts[0].flatten(), self.space.pts[1].flatten(), eta3, 'vector_1')
        self.mat_jh2 = eq_pic.domain.pull(jh_phys, self.space.pts[0].flatten(), self.space.pts[1].flatten(), eta3, 'vector_2')
        self.mat_jh3 = eq_pic.domain.pull(jh_phys, self.space.pts[0].flatten(), self.space.pts[1].flatten(), eta3, 'vector_3')
        
        if self.basis_u == 0:
            self.mat_jh1 = self.mat_jh1*det_DF
            self.mat_jh2 = self.mat_jh2*det_DF
            self.mat_jh3 = self.mat_jh3*det_DF
            
        
        # evaluation of nh_eq_phys */ |det(DF)| at quadrature points
        #self.mat_nh = eq_pic.domain.pull(nh_phys, self.space.pts[0].flatten(), self.space.pts[1].flatten(), eta3, '0_form')
        self.mat_nh = np.outer(eq_pic.nh_eq(self.space.pts[0].flatten()), np.ones(self.space.pts[1].size*eta3.size, dtype=float))
        self.mat_nh = self.mat_nh.reshape(self.space.pts[0].size, self.space.pts[1].size, eta3.size)
        
        if   self.basis_u == 0:
            self.mat_nh = self.mat_nh*det_DF
        elif self.basis_u == 2:
            self.mat_nh = self.mat_nh/det_DF
        
        
        # reshape to format (Nel1, nq1, Nel2, nq2, ...)
        if self.space.dim == 2:
            self.mat_jh1 = self.mat_jh1.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1])
            self.mat_jh2 = self.mat_jh2.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1])
            self.mat_jh3 = self.mat_jh3.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1])

            self.mat_nh = self.mat_nh.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1])
                 
        else:
            self.mat_jh1 = self.mat_jh1.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.Nel[2], self.space.n_quad[2])
            self.mat_jh2 = self.mat_jh2.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.Nel[2], self.space.n_quad[2])
            self.mat_jh3 = self.mat_jh3.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.Nel[2], self.space.n_quad[2])

            self.mat_nh = self.mat_nh.reshape(self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.Nel[2], self.space.n_quad[2])
            
        
        
        # ------------------ correction matrices in step 1 -------------------
        if   self.basis_u == 0:
            
            Ni_1 = self.space.Nbase_0form
            Ni_2 = self.space.Nbase_0form
            Ni_3 = self.space.Nbase_0form
            
        elif self.basis_u == 1:
            
            Ni_1 = self.space.Nbase_1form[0]
            Ni_2 = self.space.Nbase_1form[1]
            Ni_3 = self.space.Nbase_1form[2]
            
        elif self.basis_u == 2:
            
            Ni_1 = self.space.Nbase_2form[0]
            Ni_2 = self.space.Nbase_2form[1]
            Ni_3 = self.space.Nbase_2form[2]
        
        if self.space.dim == 2:
            
            self.M12 = np.zeros((Ni_1[0], Ni_1[1], Ni_1[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2), dtype=float)
            self.M13 = np.zeros((Ni_2[0], Ni_2[1], Ni_2[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2), dtype=float)
            self.M23 = np.zeros((Ni_3[0], Ni_3[1], Ni_3[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2), dtype=float)  
            
        else:
           
            self.M12 = np.zeros((Ni_1[0], Ni_1[1], Ni_1[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)
            self.M13 = np.zeros((Ni_2[0], Ni_2[1], Ni_2[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)
            self.M23 = np.zeros((Ni_3[0], Ni_3[1], Ni_3[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)
        # -------------------------------------------------------------------
            
        
        # --------------- correction vectors in step 3 ----------------------
        self.F = [0, 0, 0]
        
        for a in range(3):
            
            if   self.basis_u == 0:
                Ni = self.space.Nbase_0form
            elif self.basis_u == 1:
                Ni = self.space.Nbase_1form[a]
            elif self.basis_u == 2:
                Ni = self.space.Nbase_2form[a]    
            
            self.F[a] = np.zeros(Ni, dtype=float)
        # -------------------------------------------------------------------
        
        # 2-form magnetic field at quadrature points
        if self.space.dim == 2:
            self.B2_1 = np.empty((self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.NbaseN[2]), dtype=float)
            self.B2_2 = np.empty((self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.NbaseN[2]), dtype=float)
            self.B2_3 = np.empty((self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.NbaseN[2]), dtype=float)
              
        else:
            self.B2_1 = np.empty((self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.Nel[2], self.space.n_quad[2]), dtype=float)
            self.B2_2 = np.empty((self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.Nel[2], self.space.n_quad[2]), dtype=float)
            self.B2_3 = np.empty((self.space.Nel[0], self.space.n_quad[0], self.space.Nel[1], self.space.n_quad[1], self.space.Nel[2], self.space.n_quad[2]), dtype=float)
        
        
        
    
    # ===== inner product in V0^3 resp. V2 of (B x jh_eq) - term ==========
    def correct_step3(self, b1, b2, b3):
        """
        Computes the inner product of the term 
        
        (B x (DF^(-1) * jh_eq_phys) * |det(DF)|) (if MHD bulk velocity is a vector field)
        (B x (DF^(-1) * jh_eq_phys)            ) (if MHD bulk velocity is a 2-form)
        
        with each basis function in V0^3, respectively V2.
        
        Parameters
        ----------
        b1 : array_like
            the B-field FEM coefficients (1-component)
            
        b2 : array_like
            the B-field FEM coefficients (2-component)
            
        b3 : array_like
            the B-field FEM coefficients (3-component)
            
        Returns
        -------
        F : array_like
            inner products with each basis function in V0^3 resp. V2
        """
        
        
        # in 2D: evaluation of perturbed magnetic field at quadrature points
        if self.space.dim == 2:
            print('not yet implemented!')
            
        # in 3D: evaluation of total magnetic field at quadrature points
        else:
            ker3.kernel_evaluate_2form(self.space.Nel, self.space.p, [0, 1, 1], self.space.n_quad, b1, self.space.indN[0], self.space.indD[1], self.space.indD[2], self.space.basisN[0], self.space.basisD[1], self.space.basisD[2], self.B2_1)
            
            ker3.kernel_evaluate_2form(self.space.Nel, self.space.p, [1, 0, 1], self.space.n_quad, b2, self.space.indD[0], self.space.indN[1], self.space.indD[2], self.space.basisD[0], self.space.basisN[1], self.space.basisD[2], self.B2_2)
            
            ker3.kernel_evaluate_2form(self.space.Nel, self.space.p, [1, 1, 0], self.space.n_quad, b3, self.space.indD[0], self.space.indD[1], self.space.indN[2], self.space.basisD[0], self.space.basisD[1], self.space.basisN[2], self.B2_3)
            
        
        # assembly of F components
        if self.space.dim == 2:
            print('not yet implemented!')
            
        else:
        
            if self.basis_u == 0:
                
                ker3.kernel_inner(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 0, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.indN[0], self.space.indN[1], self.space.indN[2], self.F[0], self.B2_2*self.mat_jh3 - self.B2_3*self.mat_jh2)

                
                ker3.kernel_inner(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 0, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.indN[0], self.space.indN[1], self.space.indN[2], self.F[1], self.B2_3*self.mat_jh1 - self.B2_1*self.mat_jh3)

                
                ker3.kernel_inner(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 0, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.indN[0], self.space.indN[1], self.space.indN[2], self.F[2], self.B2_1*self.mat_jh2 - self.B2_2*self.mat_jh1)

            elif self.basis_u == 2:
                
                ker3.kernel_inner(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 1, 1, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisD[1], self.space.basisD[2], self.space.indN[0], self.space.indD[1], self.space.indD[2], self.F[0], self.B2_2*self.mat_jh3 - self.B2_3*self.mat_jh2)

                
                ker3.kernel_inner(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 1, 0, 1, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisD[0], self.space.basisN[1], self.space.basisD[2], self.space.indD[0], self.space.indN[1], self.space.indD[2], self.F[1], self.B2_3*self.mat_jh1 - self.B2_1*self.mat_jh3)

                
                ker3.kernel_inner(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 1, 1, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisD[0], self.space.basisD[1], self.space.basisN[2], self.space.indD[0], self.space.indD[1], self.space.indN[2], self.F[2], self.B2_1*self.mat_jh2 - self.B2_2*self.mat_jh1)
    
    
   
    # ===== mass matrix in V0^3 resp. V2 of -(rhoh_eq * (B x U)) - term =======
    def correct_step1(self, b1, b2, b3):
        """
        Computes the mass matrix in V0^3 respectively V2 weighted with the term 
        
        -(rhoh_eq_phys * |det(DF)| B x) (if MHD bulk velocity is a vector field)
        -(rhoh_eq_phys / |det(DF)| B x) (if MHD bulk velocity is a 2-form)
        
        
        Parameters
        ----------
        b1 : array_like
            the B-field FEM coefficients (1-component)
            
        b2 : array_like
            the B-field FEM coefficients (2-component)
            
        b3 : array_like
            the B-field FEM coefficients (3-component)
            
        Returns
        -------
        M12 : 6D array
            12 block of  weighted mass matrix
            
        M13 : 6D array
            13 block of  weighted mass matrix
            
        M23 : 6D array
            23 block of  weighted mass matrix
        """
        
        
        # in 2D: evaluation of equilibrium magnetic field at quadrature points
        if self.space.dim == 2:
            print('not yet implemented!')
            
        # in 3D: evaluation of total magnetic field at quadrature points
        else:
            ker3.kernel_evaluate_2form(self.space.Nel, self.space.p, [0, 1, 1], self.space.n_quad, b1, self.space.indN[0], self.space.indD[1], self.space.indD[2], self.space.basisN[0], self.space.basisD[1], self.space.basisD[2], self.B2_1)
            
            ker3.kernel_evaluate_2form(self.space.Nel, self.space.p, [1, 0, 1], self.space.n_quad, b2, self.space.indD[0], self.space.indN[1], self.space.indD[2], self.space.basisD[0], self.space.basisN[1], self.space.basisD[2], self.B2_2)
            
            ker3.kernel_evaluate_2form(self.space.Nel, self.space.p, [1, 1, 0], self.space.n_quad, b3, self.space.indD[0], self.space.indD[1], self.space.indN[2], self.space.basisD[0], self.space.basisD[1], self.space.basisN[2], self.B2_3)
        
        
        if self.space.dim == 2:
            print('not yet implemented!')
            
        else:
        
            if self.basis_u == 0:
                
                ker3.kernel_mass(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 0, 0, 0, 0, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.indN[0], self.space.indN[1], self.space.indN[2], self.M12, +self.mat_nh*self.B2_3)

                
                ker3.kernel_mass(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 0, 0, 0, 0, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.indN[0], self.space.indN[1], self.space.indN[2], self.M13, -self.mat_nh*self.B2_2)

                
                ker3.kernel_mass(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 0, 0, 0, 0, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.basisN[0], self.space.basisN[1], self.space.basisN[2], self.space.indN[0], self.space.indN[1], self.space.indN[2], self.M23, +self.mat_nh*self.B2_1)

            elif self.basis_u == 2:
                
                ker3.kernel_mass(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 1, 1, 1, 0, 1, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisD[1], self.space.basisD[2], self.space.basisD[0], self.space.basisN[1], self.space.basisD[2], self.space.indN[0], self.space.indD[1], self.space.indD[2], self.M12, +self.mat_nh*self.B2_3)

                
                ker3.kernel_mass(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 0, 1, 1, 1, 1, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisN[0], self.space.basisD[1], self.space.basisD[2], self.space.basisD[0], self.space.basisD[1], self.space.basisN[2], self.space.indN[0], self.space.indD[1], self.space.indD[2], self.M13, -self.mat_nh*self.B2_2)

                
                ker3.kernel_mass(self.space.Nel[0], self.space.Nel[1], self.space.Nel[2], self.space.p[0], self.space.p[1], self.space.p[2], self.space.n_quad[0], self.space.n_quad[1], self.space.n_quad[2], 1, 0, 1, 1, 1, 0, self.space.wts[0], self.space.wts[1], self.space.wts[2], self.space.basisD[0], self.space.basisN[1], self.space.basisD[2], self.space.basisD[0], self.space.basisD[1], self.space.basisN[2], self.space.indD[0], self.space.indN[1], self.space.indD[2], self.M23, +self.mat_nh*self.B2_1)