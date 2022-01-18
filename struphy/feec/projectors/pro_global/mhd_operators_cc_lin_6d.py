# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

"""
Class for 2D/3D linear MHD projection operators.
"""


import numpy as np
import scipy.sparse as spa

import struphy.feec.projectors.pro_global.mhd_operators_cc_lin_6d_core as mhd

import struphy.feec.basics.mass_matrices_3d_pre as mass_3d_pre


class MHD_operators:
    """
    TODO
    """
    
    def __init__(self, space, equilibrium, domain, basis_u):
        
        # create MHD operators object
        self.MHD = mhd.MHD_operators(space, equilibrium, domain, basis_u)
        
        # get 1D int_N and int_D matrices in third direction
        if space.dim == 2:
            self.ID_tor = spa.identity(space.NbaseN[2], format='csr')
            
            self.int_N3 = spa.identity(space.NbaseN[2], format='csr')
            self.int_D3 = spa.identity(space.NbaseN[2], format='csr')
            
            self.his_N3 = spa.identity(space.NbaseN[2], format='csr')
            self.his_D3 = spa.identity(space.NbaseN[2], format='csr')
        
        else:
            self.int_N3 = space.spaces[2].projectors.I
            self.int_D3 = space.spaces[2].projectors.ID
            
            self.his_N3 = space.spaces[2].projectors.HN
            self.his_D3 = space.spaces[2].projectors.H
            
 
    
    # =================================================================
    def assemble_dofs_EF(self, as_tensor=False, merge_blocks=False):
        """
        TODO
        """
        
        self.EF_as_tensor = as_tensor
        
        EF_12, EF_13, EF_21, EF_23, EF_31, EF_32 = self.MHD.get_blocks_EF(self.EF_as_tensor)
        
        # ------------ full operator : 0-form --------
        if self.MHD.basis_u == 0:
            
            if self.EF_as_tensor:
                
                EF_11 = spa.bmat([[None, EF_12], [EF_21, None]])
                
                EF_12 = spa.bmat([[EF_13], [EF_23]])
                EF_21 = spa.bmat([[EF_31 ,  EF_32]])

                self.dofs_EF_pol_11 = self.MHD.space.projectors.P1_pol_0.dot(EF_11.dot(self.MHD.space.Ev_pol_0.T)).tocsr()
                self.dofs_EF_pol_12 = self.MHD.space.projectors.P1_pol_0.dot(EF_12.dot(self.MHD.space.E0_pol.T  )).tocsr()
                self.dofs_EF_pol_21 = self.MHD.space.projectors.P0_pol_0.dot(EF_21.dot(self.MHD.space.Ev_pol_0.T)).tocsr()
                
                if merge_blocks:
                    
                    self.dofs_EF = spa.bmat([[self.dofs_EF_pol_11, self.dofs_EF_pol_12], [self.dofs_EF_pol_21, None]])
                    self.dofs_EF = spa.kron(self.dofs_EF, self.ID_tor, format='csr')
                
            else:
                
                EF = spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]])

                self.dofs_EF = self.MHD.space.projectors.P1_0.dot(EF.dot(self.MHD.space.Ev_0.T)).tocsr()
        # --------------------------------------------
                
        # ------------ full operator : 1-form --------
        elif self.MHD.basis_u == 1:
            print('1-form MHD is not yet implemented')
        # --------------------------------------------
        
        # ------------ full operator : 2-form --------
        elif self.MHD.basis_u == 2:
            
            if self.EF_as_tensor:

                EF_11 = spa.bmat([[None, EF_12], [EF_21, None]])
                
                EF_12 = spa.bmat([[EF_13], [EF_23]])
                EF_21 = spa.bmat([[EF_31 ,  EF_32]])
                
                self.dofs_EF_pol_11 = self.MHD.space.projectors.P1_pol_0.dot(EF_11.dot(self.MHD.space.E2_pol_0.T)).tocsr()
                self.dofs_EF_pol_12 = self.MHD.space.projectors.P1_pol_0.dot(EF_12.dot(self.MHD.space.E3_pol_0.T)).tocsr()
                self.dofs_EF_pol_21 = self.MHD.space.projectors.P0_pol_0.dot(EF_21.dot(self.MHD.space.E2_pol_0.T)).tocsr()
                
                if merge_blocks:
                    
                    self.dofs_EF = spa.bmat([[self.dofs_EF_pol_11, self.dofs_EF_pol_12], [self.dofs_EF_pol_21, None]])
                    self.dofs_EF = spa.kron(self.dofs_EF, self.ID_tor, format='csr')
                
            else:
                
                EF = spa.bmat([[None, EF_12, EF_13], [EF_21, None, EF_23], [EF_31, EF_32, None]])

                self.dofs_EF = self.MHD.space.projectors.P1_0.dot(EF.dot(self.MHD.space.E2_0.T)).tocsr()
        # --------------------------------------------     
        
        

    # =================================================================
    def assemble_dofs_FL(self, which, as_tensor=False, merge_blocks=False):
        """
        TODO
        """
        
        assert which == 'm' or which == 'p' or which == 'j'
            
        if   which == 'm':
            self.MF_as_tensor = as_tensor
        elif which == 'p':
            self.PF_as_tensor = as_tensor
        elif which == 'j':
            self.JF_as_tensor = as_tensor
            
        
        FL_11, FL_22, FL_33 = self.MHD.get_blocks_FL(which, as_tensor)
        
        # ------------ full operator : 0-form --------
        if self.MHD.basis_u == 0:
            
            if as_tensor:
                
                FL_11 = spa.bmat([[FL_11, None], [None, FL_22]])
                
                dofs_FL_pol_11 = self.MHD.space.projectors.P2_pol_0.dot(FL_11.dot(self.MHD.space.Ev_pol_0.T)).tocsr()
                dofs_FL_pol_22 = self.MHD.space.projectors.P3_pol_0.dot(FL_33.dot(self.MHD.space.E0_pol.T  )).tocsr()
                
                if merge_blocks:
                    dofs_FL = spa.bmat([[dofs_FL_pol_11, None], [None, dofs_FL_pol_22]])
                    dofs_FL = spa.kron(dofs_FL, self.ID_tor, format='csr')
                else:
                    dofs_FL = None
                
                if   which == 'm':
                    self.dofs_MF_pol_11 = dofs_FL_pol_11
                    self.dofs_MF_pol_22 = dofs_FL_pol_22
                    
                    self.dofs_MF = dofs_FL

                elif which == 'p':
                    self.dofs_PF_pol_11 = dofs_FL_pol_11
                    self.dofs_PF_pol_22 = dofs_FL_pol_22
                    
                    self.dofs_PF = dofs_FL

                elif which == 'j':
                    self.dofs_JF_pol_11 = dofs_FL_pol_11
                    self.dofs_JF_pol_22 = dofs_FL_pol_22
                
                    self.dofs_JF = dofs_FL
 
            else:
                
                FL = spa.bmat([[FL_11, None, None], [None, FL_22, None], [None, None, FL_33]])

                if   which == 'm':
                    self.dofs_MF = self.MHD.space.projectors.P2_0.dot(FL.dot(self.MHD.space.Ev_0.T)).tocsr()

                elif which == 'p':
                    self.dofs_PF = self.MHD.space.projectors.P2_0.dot(FL.dot(self.MHD.space.Ev_0.T)).tocsr()

                elif which == 'j':
                    self.dofs_JF = self.MHD.space.projectors.P2_0.dot(FL.dot(self.MHD.space.Ev_0.T)).tocsr()
        # --------------------------------------------
        
        # ------------ full operator : 1-form --------
        elif self.MHD.basis_u == 1:
            print('1-form MHD is not yet implemented')
        # --------------------------------------------
        
        # ------------ full operator : 2-form --------
        elif self.MHD.basis_u == 2:
            
            if as_tensor:
                
                FL_11 = spa.bmat([[FL_11, None], [None, FL_22]])
                
                dofs_FL_pol_11 = self.MHD.space.projectors.P2_pol_0.dot(FL_11.dot(self.MHD.space.E2_pol_0.T)).tocsr()
                dofs_FL_pol_22 = self.MHD.space.projectors.P3_pol_0.dot(FL_33.dot(self.MHD.space.E3_pol_0.T)).tocsr()
                
                if merge_blocks:
                    dofs_FL = spa.bmat([[dofs_FL_pol_11, None], [None, dofs_FL_pol_22]])
                    dofs_FL = spa.kron(dofs_FL, self.ID_tor, format='csr')
                else:
                    dofs_FL = None
                
                if   which == 'm':
                    self.dofs_MF_pol_11 = dofs_FL_pol_11
                    self.dofs_MF_pol_22 = dofs_FL_pol_22
                    
                    self.dofs_MF = dofs_FL

                elif which == 'p':
                    self.dofs_PF_pol_11 = dofs_FL_pol_11
                    self.dofs_PF_pol_22 = dofs_FL_pol_22
                    
                    self.dofs_PF = dofs_FL
                
            else:
                
                FL = spa.bmat([[FL_11, None, None], [None, FL_22, None], [None, None, FL_33]])

                if   which == 'm':
                    self.dofs_MF = self.MHD.space.projectors.P2_0.dot(FL.dot(self.MHD.space.E2_0.T)).tocsr()

                elif which == 'p':
                    self.dofs_PF = self.MHD.space.projectors.P2_0.dot(FL.dot(self.MHD.space.E2_0.T)).tocsr()
        # --------------------------------------------            
                

    
    # =================================================================
    def assemble_dofs_PR(self, as_tensor=False, merge_blocks=False):
        """
        TODO
        """
        
        self.PR_as_tensor = as_tensor
        
        PR = self.MHD.get_blocks_PR(self.PR_as_tensor)
        
        if self.PR_as_tensor:
            self.dofs_PR_pol = self.MHD.space.projectors.P3_pol_0.dot(PR.dot(self.MHD.space.E3_pol_0.T)).tocsr()
            
            if merge_blocks:
                self.dofs_PR = spa.kron(self.dofs_PR_pol, self.ID_tor, format='csr')
            
        else:
            self.dofs_PR = self.MHD.space.projectors.P3_0.dot(PR.dot(self.MHD.space.E3_0.T)).tocsr()
        
    
    # =================================================================
    def assemble_MR(self, as_tensor=True, merge_blocks=False):
        """
        TODO
        """
        
        self.MR_as_tensor = as_tensor
                                                                                                        
        MR = self.MHD.get_blocks_MR(self.MR_as_tensor)
        
        # ----------- 0-form ----------------------
        if self.MHD.basis_u == 0:
            
            if self.MR_as_tensor:
                self.mat_MR_pol_11, self.mat_MR_pol_22 = MR
                
                if merge_blocks:
                    M11 = spa.kron(self.mat_MR_pol_11, self.MHD.space.M0_tor)
                    M22 = spa.kron(self.mat_MR_pol_22, self.MHD.space.M0_tor)
                    
                    self.mat_MR = spa.bmat([[M11, None], [None, M22]], format='csr')
                
            else:
                self.mat_MR = MR
        # -----------------------------------------
        
        # ------------ 1-form ---------------------
        elif self.MHD.basis_u == 1:
            print('1-form MHD is not yet implemented')
        # -----------------------------------------
        
        # ----------- 2-form ----------------------
        elif self.MHD.basis_u == 2:
            
            if self.MR_as_tensor:
                self.mat_MR_pol_11, self.mat_MR_pol_22 = MR
                
                if merge_blocks:
                    M11 = spa.kron(self.mat_MR_pol_11, self.MHD.space.M1_tor)
                    M22 = spa.kron(self.mat_MR_pol_22, self.MHD.space.M0_tor)
                    
                    self.mat_MR = spa.bmat([[M11, None], [None, M22]], format='csr')
                
            else:
                self.mat_MR = MR
        # -----------------------------------------
                
        
    
    
    # =================================================================
    def assemble_MJ(self, as_tensor=False, merge_blocks=False):
        """
        TODO
        """
        
        self.MJ_as_tensor = as_tensor
            
        MJ = self.MHD.get_blocks_MJ(self.MJ_as_tensor)  
        
        # ----------- 0-form ----------------------
        if self.MHD.basis_u == 0:
            
            if self.MJ_as_tensor:
                self.mat_MJ_pol_11, self.mat_MJ_pol_22 = MJ
                
                if merge_blocks:
                    M11 = spa.kron(self.mat_MJ_pol_11, self.MHD.space.M0_tor)
                    M22 = spa.kron(self.mat_MJ_pol_22, self.MHD.space.M0_tor)
                    
                    self.mat_MJ = spa.bmat([[M11, None], [None, M22]], format='csr')
                
            else:
                self.mat_MJ = MJ
        # -----------------------------------------
        
        # ------------ 1-form ---------------------
        elif self.MHD.basis_u == 1:
            print('1-form MHD is not yet implemented')
        # -----------------------------------------
        
        # ----------- 2-form ----------------------
        elif self.MHD.basis_u == 2:
            
            if self.MJ_as_tensor:
                self.mat_MJ_pol_11, self.mat_MJ_pol_22 = MJ
                
                if merge_blocks:
                    M11 = spa.kron(self.mat_MJ_pol_11, self.MHD.space.M1_tor)
                    M22 = spa.kron(self.mat_MJ_pol_22, self.MHD.space.M0_tor)
                    
                    self.mat_MJ = spa.bmat([[M11, None], [None, M22]], format='csr')
                
            else:
                self.mat_MJ = MJ
        # -----------------------------------------
                
        
    # ======================================
    def __EF(self, u):
        """
        TODO
        """
        
        if self.EF_as_tensor:
            
            if self.MHD.basis_u == 0:
                u1, u3 = self.MHD.space.reshape_pol_v(u)
                
                out1 = self.int_N3.dot(self.dofs_EF_pol_11.dot(u1).T).T + self.int_N3.dot(self.dofs_EF_pol_12.dot(u3).T).T
                out3 = self.his_N3.dot(self.dofs_EF_pol_21.dot(u1).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
              
            elif self.MHD.basis_u == 2:
                u1, u3 = self.MHD.space.reshape_pol_2(u)
                
                out1 = self.int_D3.dot(self.dofs_EF_pol_11.dot(u1).T).T + self.int_N3.dot(self.dofs_EF_pol_12.dot(u3).T).T
                out3 = self.his_D3.dot(self.dofs_EF_pol_21.dot(u1).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
 
        else:
            out = self.dofs_EF.dot(u)
        
        return self.MHD.space.projectors.solve_V1(out, False)
    
    # ======================================
    def __EF_transposed(self, e):
        """
        TODO
        """
        
        e = self.MHD.space.projectors.apply_IinvT_V1(e)
        
        if self.EF_as_tensor:
            
            e1, e3 = self.MHD.space.reshape_pol_1(e)
            
            if self.MHD.basis_u == 0:
                out1 = self.int_N3.T.dot(self.dofs_EF_pol_11.T.dot(e1).T).T + self.his_N3.T.dot(self.dofs_EF_pol_21.T.dot(e3).T).T
                out3 = self.int_N3.T.dot(self.dofs_EF_pol_12.T.dot(e1).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
            
            elif self.MHD.basis_u == 2:
                out1 = self.int_D3.T.dot(self.dofs_EF_pol_11.T.dot(e1).T).T + self.his_D3.T.dot(self.dofs_EF_pol_21.T.dot(e3).T).T
                out3 = self.int_N3.T.dot(self.dofs_EF_pol_12.T.dot(e1).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
                           
        else:
            out = self.dofs_EF.T.dot(e)
        
        return out
    
    # ======================================
    def __MF(self, u):
        """
        TODO
        """
        
        if self.MF_as_tensor:
            
            if self.MHD.basis_u == 0:
                u1, u3 = self.MHD.space.reshape_pol_v(u)
                
                out1 = self.his_N3.dot(self.dofs_MF_pol_11.dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_MF_pol_22.dot(u3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
            
            elif self.MHD.basis_u == 2:
                u1, u3 = self.MHD.space.reshape_pol_2(u)
                
                out1 = self.his_D3.dot(self.dofs_MF_pol_11.dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_MF_pol_22.dot(u3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
    
        else:
            out = self.dofs_MF.dot(u)
                
        return self.MHD.space.projectors.solve_V2(out, False)
    
    # ======================================
    def __MF_transposed(self, f):
        """
        TODO
        """
        
        f = self.MHD.space.projectors.apply_IinvT_V2(f)
        
        if self.MF_as_tensor:
            
            f1, f3 = self.MHD.space.reshape_pol_2(f)
            
            if self.MHD.basis_u == 0:
                out1 = self.his_N3.T.dot(self.dofs_MF_pol_11.T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_MF_pol_22.T.dot(f3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
            
            elif self.MHD.basis_u == 2:
                out1 = self.his_D3.T.dot(self.dofs_MF_pol_11.T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_MF_pol_22.T.dot(f3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
                      
        else:
            out = self.dofs_MF.T.dot(f)
        
        return out
    
    # ======================================
    def __PF(self, u):
        """
        TODO
        """
        
        if self.PF_as_tensor:
            
            if self.MHD.basis_u == 0:
                u1, u3 = self.MHD.space.reshape_pol_v(u)
                
                out1 = self.his_N3.dot(self.dofs_PF_pol_11.dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_PF_pol_22.dot(u3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
            
            elif self.MHD.basis_u == 2:
                u1, u3 = self.MHD.space.reshape_pol_2(u)
                
                out1 = self.his_D3.dot(self.dofs_PF_pol_11.dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_PF_pol_22.dot(u3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
                
        else:
            out = self.dofs_PF.dot(u)
                
        return self.MHD.space.projectors.solve_V2(out, False)
    
    # ======================================
    def __PF_transposed(self, f):
        """
        TODO
        """
        
        f = self.MHD.space.projectors.apply_IinvT_V2(f)
        
        if self.PF_as_tensor:
            
            f1, f3 = self.MHD.space.reshape_pol_2(f)
            
            if self.MHD.basis_u == 0:
                out1 = self.his_N3.T.dot(self.dofs_PF_pol_11.T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_PF_pol_22.T.dot(f3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
            
            elif self.MHD.basis_u == 2:
                out1 = self.his_D3.T.dot(self.dofs_PF_pol_11.T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_PF_pol_22.T.dot(f3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
                       
        else:
            out = self.dofs_PF.T.dot(f)
        
        return out
    
    # ======================================
    def __JF(self, u):
        """
        TODO
        """
        
        if self.JF_as_tensor:
            
            if self.MHD.basis_u == 0:
                u1, u3 = self.MHD.space.reshape_pol_v(u)
                
                out1 = self.his_N3.dot(self.dofs_JF_pol_11.dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_JF_pol_22.dot(u3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
            
            elif self.MHD.basis_u == 2:
                u1, u3 = self.MHD.space.reshape_pol_2(u)
                
                out1 = self.his_D3.dot(self.dofs_JF_pol_11.dot(u1).T).T
                out3 = self.int_N3.dot(self.dofs_JF_pol_22.dot(u3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
                    
        else:
            out = self.dofs_JF.dot(u)
                
        return self.MHD.space.projectors.solve_V2(out, False)
    
    # ======================================
    def __JF_transposed(self, f):
        """
        TODO
        """
        
        f = self.MHD.space.projectors.apply_IinvT_V2(f)
        
        if self.JF_as_tensor:
            
            f1, f3 = self.MHD.space.reshape_pol_2(f)
            
            if self.MHD.basis_u == 0:
                out1 = self.his_N3.T.dot(self.dofs_JF_pol_11.T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_JF_pol_22.T.dot(f3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
            
            elif self.MHD.basis_u == 2:
                out1 = self.his_D3.T.dot(self.dofs_JF_pol_11.T.dot(f1).T).T
                out3 = self.int_N3.T.dot(self.dofs_JF_pol_22.T.dot(f3).T).T
                
                out  = np.concatenate((out1.flatten(), out3.flatten()))
                      
        else:
            out = self.dofs_JF.T.dot(f)
        
        return out
    
    # ======================================
    def __PR(self, d):
        """
        TODO
        """
        
        if self.PR_as_tensor:
            d   = self.MHD.space.reshape_pol_3(d)
            out = self.his_D3.dot(self.dofs_PR_pol.dot(d).T).T.flatten()
        else:
            out = self.dofs_PR.dot(d)
        
        return self.MHD.space.projectors.solve_V3(out, False)
    
    # ======================================
    def __PR_transposed(self, d):
        """
        TODO
        """
        
        d = self.MHD.space.projectors.apply_IinvT_V3(d)
        
        if self.PR_as_tensor:
            d   = self.MHD.space.reshape_pol_3(d)
            out = self.his_D3.T.dot(self.dofs_PR_pol.T.dot(d).T).T.flatten()
        else:
            out = self.dofs_PR.T.dot(d)
        
        return out
    
    
    # ======================================
    def __MR(self, u):
        """
        TODO
        """
        
        if self.MR_as_tensor:

            if self.MHD.basis_u == 0:
                out = self.MHD.space.apply_Mv_ten(u, [[self.mat_MR_pol_11, self.MHD.space.M0_tor], [self.mat_MR_pol_22, self.MHD.space.M0_tor]])
            elif self.MHD.basis_u == 2:
                out = self.MHD.space.apply_M2_ten(u, [[self.mat_MR_pol_11, self.MHD.space.M1_tor], [self.mat_MR_pol_22, self.MHD.space.M0_tor]]) 
                     
        else:
            out = self.mat_MR.dot(u)

        return out
    
    
    # ======================================
    def __MJ(self, b):
        """
        TODO
        """
        
        if self.MJ_as_tensor:

            if self.MHD.basis_u == 0:
                out = self.MHD.space.apply_Mv_ten(b, [[self.mat_MJ_pol_11, self.MHD.space.M0_tor], [self.mat_MJ_pol_22, self.MHD.space.M0_tor]]) 
            elif self.MHD.basis_u == 2:
                out = self.MHD.space.apply_M2_ten(b, [[self.mat_MJ_pol_11, self.MHD.space.M1_tor], [self.mat_MJ_pol_22, self.MHD.space.M0_tor]]) 
                   
        else:
            out = self.mat_MJ.dot(b)
        
        return out
    
    # ======================================
    def __MJ_transposed(self, u):
        """
        TODO
        """
        
        if self.MJ_as_tensor:
            
            if self.MHD.basis_u == 0:
                out = self.MHD.space.apply_Mv_ten(u, [[self.mat_MJ_pol_11, self.MHD.space.M0_tor], [self.mat_MJ_pol_22, self.MHD.space.M0_tor]]) 
            elif self.MHD.basis_u == 2:
                out = self.MHD.space.apply_M2_ten(u, [[self.mat_MJ_pol_11, self.MHD.space.M1_tor], [self.mat_MJ_pol_22, self.MHD.space.M0_tor]]) 
                
        else:
            out = self.mat_MJ.dot(u)
        
        return -out
    
    
    # ======================================
    def __A(self, u):
        """
        TODO
        """
        
        return self.__MR(u)
    
    # ======================================
    def __L(self, u):
        """
        TODO
        """
        
        if self.MHD.basis_u == 0:
            out = -self.MHD.space.D0.dot(self.__PF(u)) - (5./3. - 1)*self.__PR(self.MHD.space.D0.dot(self.__JF(u)))
        elif self.MHD.basis_u == 2:
            out = -self.MHD.space.D0.dot(self.__PF(u)) - (5./3. - 1)*self.__PR(self.MHD.space.D0.dot(u))
            
        return out
    
    # ======================================
    def __S2(self, u):
        """
        TODO
        """
        
        bu   = self.MHD.space.C0.dot(self.__EF(u))
        
        out  = self.__A(u)
        out += self.dt_2**2/4*self.__EF_transposed(self.MHD.space.C0.T.dot(self.MHD.space.M2(bu)))
        
        # with additional J_eq x B
        if self.loc_j_eq == 'step_2':
            
            out += self.dt_2**2/4*self.__MJ(bu)

        return out
    
    # ======================================
    def __S6(self, u):
        """
        TODO
        """
        
        out = self.__A(u)
        
        if self.MHD.basis_u == 0:
            out -= self.dt_6**2/4*self.__JF_transposed(self.MHD.space.D0.T.dot(self.MHD.space.M3(self.__L(u))))
        elif self.MHD.basis_u == 2:
            out -= self.dt_6**2/4*self.MHD.space.D0.T.dot(self.MHD.space.M3(self.__L(u)))

        return out
    
    # ======================================
    def setOperators(self, dt_2, dt_6, loc_j_eq):
        """
        TODO
        """
        
        self.dt_2 = dt_2
        self.dt_6 = dt_6
        
        self.loc_j_eq = loc_j_eq
        
        self.MF = spa.linalg.LinearOperator((self.MHD.space.E2_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__MF, rmatvec=self.__MF_transposed)
        
        self.PF = spa.linalg.LinearOperator((self.MHD.space.E2_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__PF, rmatvec=self.__PF_transposed)
        
        if self.MHD.basis_u == 0:
            self.JF = spa.linalg.LinearOperator((self.MHD.space.E2_0.shape[0], self.MHD.space.E0_0.shape[0] + 2*self.MHD.space.E0.shape[0]), matvec=self.__JF, rmatvec=self.__JF_transposed)
        
        self.EF = spa.linalg.LinearOperator((self.MHD.space.E1_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__EF, rmatvec=self.__EF_transposed)
        
        self.PR = spa.linalg.LinearOperator((self.MHD.space.E3_0.shape[0], self.MHD.space.E3_0.shape[0]), matvec=self.__PR, rmatvec=self.__PR_transposed)
        
        self.MJ = spa.linalg.LinearOperator((self.MHD.space.E2_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__MJ, rmatvec=self.__MJ_transposed)
        
        self.A  = spa.linalg.LinearOperator((self.MHD.space.E2_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__A)
        
        self.L  = spa.linalg.LinearOperator((self.MHD.space.E3_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__L)
        
        self.S2 = spa.linalg.LinearOperator((self.MHD.space.E2_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__S2)
        
        self.S6 = spa.linalg.LinearOperator((self.MHD.space.E2_0.shape[0], self.MHD.space.E2_0.shape[0]), matvec=self.__S6)
        
    
    # ======================================
    def RHS2(self, u, b):
        """
        TODO
        """
        
        bu = self.MHD.space.C0.dot(self.EF(u))
        
        out  = self.A(u)
        out -= self.dt_2**2/4*self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(bu)))
        out += self.dt_2*self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b)))
        
        # with additional J_eq x B
        if self.loc_j_eq == 'step_2':
            out -= self.dt_2**2/4*self.MJ(bu) 
            out += self.dt_2*self.MJ(b)
        
        return out
    
    # ======================================
    def RHS6(self, u, p, b):
        """
        TODO
        """
        
        out = self.A(u)
        
        # --- MHD bulk velocity is a 0-form ---
        if self.MHD.basis_u == 0:
            out += self.dt_6**2/4*self.JF.T(self.MHD.space.D0.T.dot(self.MHD.space.M3(self.L(u))))
            out += self.dt_6*self.JF.T(self.MHD.space.D0.T.dot(self.MHD.space.M3(p)))
        # --------------------------------------
        
        # --- MHD bulk velocity is a 2-form ---
        elif self.MHD.basis_u == 2:
            out += self.dt_6**2/4*self.MHD.space.D0.T.dot(self.MHD.space.M3(self.L(u))) 
            out += self.dt_6*self.MHD.space.D0.T.dot(self.MHD.space.M3(p))
        # --------------------------------------
        
        # with additional J_eq x B
        if self.loc_j_eq == 'step_6':
            out += self.dt_6*self.MJ(b)
        
        return out
    
    # ======================================
    def guess_S2(self, u, b, kind):
        """
        TODO
        """
        
        if kind == 'Euler':
            
            k1_u = self.A_inv(self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b))))
            
            u_guess = u + self.dt_2*k1_u
            
        elif kind == 'Heun':
            
            k1_u = self.A_inv(self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b))))
            k1_b = -self.MHD.space.C0.dot(self.EF(u))
            
            k2_u = self.A_inv(self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b + self.dt_2*k1_b))))
            
            u_guess = u + self.dt_2/2*(k1_u + k2_u)
            
        elif kind == 'RK4':
            
            k1_u = self.A_inv(self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b))))
            k1_b = -self.MHD.space.C0.dot(self.EF(u))
            
            k2_u = self.A_inv(self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b + self.dt_2/2*k1_b))))
            k2_b = -self.MHD.space.C0.dot(self.EF(u + self.dt_2/2*k1_u))
            
            k3_u = self.A_inv(self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b + self.dt_2/2*k2_b))))
            k3_b = -self.MHD.space.C0.dot(self.EF(u + self.dt_2/2*k2_u))
            
            k4_u = self.A_inv(self.EF.T(self.MHD.space.C0.T.dot(self.MHD.space.M2(b + self.dt_2*k3_b))))
            k4_b = -self.MHD.space.C0.dot(self.EF(u + self.dt_2*k3_u))
            
            u_guess = u + self.dt_2/6*(k1_u + 2*k2_u + 2*k3_u + k4_u)
            
        else:
            
            u_guess = np.copy(u)
        
        return u_guess
    
    # ======================================
    def guess_S6(self, u, p, b, kind):
        """
        TODO
        """
        
        u_guess = np.copy(u)
        
        return u_guess
    
    # ======================================
    def setInverseA(self):
        """
        TODO
        """
        
        # set fast FFT inverse of A
        # -------- 0-form ---------------
        if self.MHD.basis_u == 0:
            self.A_inv = mass_3d_pre.get_Mv_PRE_3(self.MHD.space, self.MHD.domain, [self.mat_MR_pol_11, self.mat_MR_pol_22])
        # -------------------------------

        # -------- 2-form ---------------
        elif self.MHD.basis_u == 2:
            self.A_inv = mass_3d_pre.get_M2_PRE_3(self.MHD.space, self.MHD.domain, [self.mat_MR_pol_11, self.mat_MR_pol_22])
        # -------------------------------
         
    
    # ======================================
    def setPreconditionerS2(self, which, drop_tol=1e-4, fill_fac=10.):
        """
        TODO
        """
        
        assert which == 'LU' or which == 'ILU' or which == 'FFT'
        
        # LU/ILU preconditioner
        if which == 'ILU' or which == 'LU':
            
            M2 = spa.bmat([[spa.kron(self.MHD.space.M2_pol_11, self.MHD.space.M1_tor), None], [None, spa.kron(self.MHD.space.M2_pol_22, self.MHD.space.M0_tor)]], format='csc')
        
        
            # -------- 0-form ---------------
            if self.MHD.basis_u == 0:
                A = spa.bmat([[spa.kron(self.mat_MR_pol_11, self.MHD.space.M0_tor), None], [None, spa.kron(self.mat_MR_pol_22, self.MHD.space.M0_tor)]], format='csc')
                
                EF_11 = spa.kron(self.dofs_EF_pol_11, self.int_N3)
                EF_12 = spa.kron(self.dofs_EF_pol_12, self.int_N3)
                EF_21 = spa.kron(self.dofs_EF_pol_21, self.his_N3)
                
                # with additional J_eq x B
                if self.loc_j_eq == 'step_2':
                    MJ = spa.bmat([[spa.kron(self.mat_MJ_pol_11, self.MHD.space.M0_tor), None], [None, spa.kron(self.mat_MJ_pol_22, self.MHD.space.M0_tor)]], format='csc')
            # -------------------------------
            
            
            # -------- 2-form ---------------
            elif self.MHD.basis_u == 2:
                A  = spa.bmat([[spa.kron(self.mat_MR_pol_11, self.MHD.space.M1_tor), None], [None, spa.kron(self.mat_MR_pol_22, self.MHD.space.M0_tor)]], format='csc')
                
                EF_11 = spa.kron(self.dofs_EF_pol_11, self.int_D3)
                EF_12 = spa.kron(self.dofs_EF_pol_12, self.int_N3)
                EF_21 = spa.kron(self.dofs_EF_pol_21, self.his_D3)
                
                # with additional J_eq x B
                if self.loc_j_eq == 'step_2':
                    MJ = spa.bmat([[spa.kron(self.mat_MJ_pol_11, self.MHD.space.M1_tor), None], [None, spa.kron(self.mat_MJ_pol_22, self.MHD.space.M0_tor)]], format='csc')
            # -------------------------------
            
            EF_approx = spa.bmat([[EF_11, EF_12], [EF_21, None]], format='csr')
            EF_approx = self.MHD.space.projectors.I1_0_inv_approx.dot(EF_approx)
            
            S2_approx = A + self.dt_2**2/4*EF_approx.T.dot(self.MHD.space.C0.T.dot(M2.dot(self.MHD.space.C0.dot(EF_approx))))
            #self.S2_approx = S2_approx
            
            # with additional J_eq x B
            if self.loc_j_eq == 'step_2':
                S2_approx += self.dt_2**2/4*MJ.dot(self.MHD.space.C0.dot(EF_approx))
                
            del EF_approx
           
            if which == 'LU':
                S2_LU = spa.linalg.splu(S2_approx.tocsc())
            else:
                S2_LU = spa.linalg.spilu(S2_approx.tocsc(), drop_tol=drop_tol , fill_factor=fill_fac)
            
            self.S2_PRE = spa.linalg.LinearOperator(S2_approx.shape, S2_LU.solve)
            
        # FFT preconditioner
        elif which == 'FFT':
            
            def solve_S2(x):

                temp = self.A_inv(x)

                #out = temp - 0*self.dt_2**2/4*self.A_inv(self.__EF_transposed(self.MHD.space.C0.T.dot(self.MHD.space.M2(self.MHD.space.C0.dot(self.__EF(temp))))))

                return temp
                
            self.S2_PRE = spa.linalg.LinearOperator(self.A_inv.shape, solve_S2)
            
            

    # ======================================
    def setPreconditionerS6(self, which, drop_tol=1e-4, fill_fac=10.):
        """
        TODO
        """
        
        assert which == 'LU' or which == 'ILU' or which == 'FFT'
        
        # LU/ILU preconditioner
        if which == 'ILU' or which == 'LU':
            
            M3 = spa.kron(self.MHD.space.M3_pol, self.MHD.space.M1_tor, format='csc')
        
            # -------- 0-form ---------------
            if self.MHD.basis_u == 0:
                A = spa.bmat([[spa.kron(self.mat_MR_pol_11, self.MHD.space.M0_tor), None], [None, spa.kron(self.mat_MR_pol_22, self.MHD.space.M0_tor)]], format='csc')
                
                PF_11 = spa.kron(self.dofs_PF_pol_11, self.his_N3)
                PF_22 = spa.kron(self.dofs_PF_pol_22, self.int_N3)
                
                JF_11 = spa.kron(self.dofs_JF_pol_11, self.his_N3)
                JF_22 = spa.kron(self.dofs_JF_pol_22, self.int_N3)
                
                PR    = spa.kron(self.dofs_PR_pol   , self.his_D3)
                
                PF_approx = spa.bmat([[PF_11, None], [None, PF_22]], format='csr')
                JF_approx = spa.bmat([[JF_11, None], [None, JF_22]], format='csr')
                
                PF_approx = self.MHD.space.projectors.I2_0_inv_approx.dot(PF_approx)
                JF_approx = self.MHD.space.projectors.I2_0_inv_approx.dot(JF_approx)
                PR_approx = self.MHD.space.projectors.I3_0_inv_approx.dot(PR       )

                L_approx  = -self.MHD.space.D0.dot(PF_approx) - (5./3. - 1)*PR_approx.dot(self.MHD.space.D0.dot(JF_approx))

                del PF_approx, PR_approx

                S6_approx = A - self.dt_6**2/4*JF_approx.T.dot(self.MHD.space.D0.T.dot(M3.dot(L_approx)))

                del JF_approx
            # -------------------------------
            
            
            # -------- 2-form ---------------
            elif self.MHD.basis_u == 2:
                A = spa.bmat([[spa.kron(self.mat_MR_pol_11, self.MHD.space.M1_tor), None], [None, spa.kron(self.mat_MR_pol_22, self.MHD.space.M0_tor)]], format='csc')
                
                PF_11 = spa.kron(self.dofs_PF_pol_11, self.his_D3)
                PF_22 = spa.kron(self.dofs_PF_pol_22, self.int_N3)
                
                PR    = spa.kron(self.dofs_PR_pol   , self.his_D3)
                
                PF_approx = spa.bmat([[PF_11, None], [None, PF_22]], format='csr')
                
                PF_approx = self.MHD.space.projectors.I2_0_inv_approx.dot(PF_approx)
                PR_approx = self.MHD.space.projectors.I3_0_inv_approx.dot(PR       )
                
                L_approx  = -self.MHD.space.D0.dot(PF_approx) - (5./3. - 1)*PR_approx.dot(self.MHD.space.D0)

                del PF_approx, PR_approx

                S6_approx = A - self.dt_6**2/4*self.MHD.space.D0.T.dot(M3.dot(L_approx))
            # -------------------------------
            
            #self.S6_approx = S6_approx
            
            if which == 'LU':
                S6_LU = spa.linalg.splu(S6_approx.tocsc())
            else:
                S6_LU = spa.linalg.spilu(S6_approx.tocsc(), drop_tol=drop_tol , fill_factor=fill_fac)
            
            self.S6_PRE = spa.linalg.LinearOperator(S6_approx.shape, S6_LU.solve)
            
        # FFT preconditioner
        elif which == 'FFT':
            
            def solve_S6(x):

                out = self.A_inv(x)

                return out
                
            self.S6_PRE = spa.linalg.LinearOperator(self.A_inv.shape, solve_S6)