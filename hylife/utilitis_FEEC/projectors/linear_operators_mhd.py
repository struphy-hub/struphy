import numpy as np
import scipy.sparse as spa

import source_run.projectors_global as proj_global
import source_run.projectors_local  as proj_local

import hylife.linear_algebra.kernels_tensor_product as ker_la
import hylife.linear_algebra.linalg_kron            as kron_la

"""
Takes and returns flattened coefficient arrays.
"""



class operators_mhd:
    
    def __init__(self, projectors_3d, params_map):
        
        self.projectors_3d  = projectors_3d
        self.kind_projector = projectors_3d.kind          # global or local projector
        self.tensor_space   = projectors_3d.tensor_space  # 3D tensor-product B-splines space
        self.polar          = projectors_3d.polar         # whether polar splines are used in the poloidal plane
        
        
        # 1D projectors for global projector
        if   self.kind_projector == 'global':
            self.projectors_1d = [proj_global.projectors_global_1d(space, n_quad, polar) for space, n_quad, polar in zip(projectors_3d.tensor_space.spaces, projectors_3d.n_quad, [self.polar, self.polar, False])]
        # 1D projectors for local projector
        elif self.kind_projector == 'local':
            self.projectors_1d = [proj_local.projectors_local_1d(space, n_quad) for space, n_quad in zip(self.tensor_space.spaces, self.projectors_3d.n_quad)]
            
        # 1d projection matrices and indices in 1-direction
        self.pi0_x_NN, self.pi0_x_DN, self.pi0_x_ND, self.pi0_x_DD, self.pi1_x_NN, self.pi1_x_DN, self.pi1_x_ND, self.pi1_x_DD, self.pi0_x_NN_i, self.pi0_x_DN_i, self.pi0_x_ND_i, self.pi0_x_DD_i, self.pi1_x_NN_i, self.pi1_x_DN_i, self.pi1_x_ND_i, self.pi1_x_DD_i = self.projectors_1d[0].projection_matrices_1d(params_map)

        # 1d projection matrices and indices in 2-direction
        self.pi0_y_NN, self.pi0_y_DN, self.pi0_y_ND, self.pi0_y_DD, self.pi1_y_NN, self.pi1_y_DN, self.pi1_y_ND, self.pi1_y_DD, self.pi0_y_NN_i, self.pi0_y_DN_i, self.pi0_y_ND_i, self.pi0_y_DD_i, self.pi1_y_NN_i, self.pi1_y_DN_i, self.pi1_y_ND_i, self.pi1_y_DD_i = self.projectors_1d[1].projection_matrices_1d(params_map)

        # 1d projection matrices and indices in 3-direction
        self.pi0_z_NN, self.pi0_z_DN, self.pi0_z_ND, self.pi0_z_DD, self.pi1_z_NN, self.pi1_z_DN, self.pi1_z_ND, self.pi1_z_DD, self.pi0_z_NN_i, self.pi0_z_DN_i, self.pi0_z_ND_i, self.pi0_z_DD_i, self.pi1_z_NN_i, self.pi1_z_DN_i, self.pi1_z_ND_i, self.pi1_z_DD_i = self.projectors_1d[2].projection_matrices_1d(params_map)
            
            
    # =======================================================        
    def T(self, b2_eq, u, basis_u='0_form'):
        
        # equilibrium magnetic field (2-form)
        b2_1_eq, b2_2_eq, b2_3_eq = self.tensor_space.unravel_2form(b2_eq)
        
        # final coefficients of the operation f1 = T(u) (1-form)
        f1_1 = np.zeros(self.tensor_space.Nbase_1form[0], dtype=float)
        f1_2 = np.zeros(self.tensor_space.Nbase_1form[1], dtype=float)
        f1_3 = np.zeros(self.tensor_space.Nbase_1form[2], dtype=float)
        
        # bulk velocity is a 0-form
        if   basis_u == '0_form':
            
            # input bulk velocity coefficients
            u0_1, u0_2, u0_3 = self.tensor_space.unravel_0form(u)
            
            # 1st component
            ker_la.projector_tensor_strong(self.pi1_x_DN, self.pi0_y_NN, self.pi0_z_DN, self.pi1_x_DN_i, self.pi0_y_NN_i, self.pi0_z_DN_i,  b2_2_eq, u0_3, f1_1)
            ker_la.projector_tensor_strong(self.pi1_x_DN, self.pi0_y_DN, self.pi0_z_NN, self.pi1_x_DN_i, self.pi0_y_DN_i, self.pi0_z_NN_i, -b2_3_eq, u0_2, f1_1)

            # 2nd component
            ker_la.projector_tensor_strong(self.pi0_x_DN, self.pi1_y_DN, self.pi0_z_NN, self.pi0_x_DN_i, self.pi1_y_DN_i, self.pi0_z_NN_i,  b2_3_eq, u0_1, f1_2)
            ker_la.projector_tensor_strong(self.pi0_x_NN, self.pi1_y_DN, self.pi0_z_DN, self.pi0_x_NN_i, self.pi1_y_DN_i, self.pi0_z_DN_i, -b2_1_eq, u0_3, f1_2)

            # 3rd component
            ker_la.projector_tensor_strong(self.pi0_x_NN, self.pi0_y_DN, self.pi1_z_DN, self.pi0_x_NN_i, self.pi0_y_DN_i, self.pi1_z_DN_i,  b2_1_eq, u0_2, f1_3)
            ker_la.projector_tensor_strong(self.pi0_x_DN, self.pi0_y_NN, self.pi1_z_DN, self.pi0_x_DN_i, self.pi0_y_NN_i, self.pi1_z_DN_i, -b2_2_eq, u0_1, f1_3)
                   
        # bulk velocity is a 2-form
        elif basis_u == '2_form':
            
            # input bulk velocity coefficients
            if self.polar == False:
                u2_1, u2_2, u2_3 = self.tensor_space.unravel_2form(u)
            else:
                u2_1, u2_2, u2_3 = self.tensor_space.unravel_2form(self.tensor_space.E2.T.dot(u))
            
            # 1st component
            ker_la.projector_tensor_strong(self.pi1_x_DD, self.pi0_y_ND, self.pi0_z_DN, self.pi1_x_DD_i, self.pi0_y_ND_i, self.pi0_z_DN_i,  b2_2_eq, u2_3, f1_1)
            ker_la.projector_tensor_strong(self.pi1_x_DD, self.pi0_y_DN, self.pi0_z_ND, self.pi1_x_DD_i, self.pi0_y_DN_i, self.pi0_z_ND_i, -b2_3_eq, u2_2, f1_1)

            # 2nd component
            ker_la.projector_tensor_strong(self.pi0_x_DN, self.pi1_y_DD, self.pi0_z_ND, self.pi0_x_DN_i, self.pi1_y_DD_i, self.pi0_z_ND_i,  b2_3_eq, u2_1, f1_2)
            ker_la.projector_tensor_strong(self.pi0_x_ND, self.pi1_y_DD, self.pi0_z_DN, self.pi0_x_ND_i, self.pi1_y_DD_i, self.pi0_z_DN_i, -b2_1_eq, u2_3, f1_2)

            # 3rd component
            ker_la.projector_tensor_strong(self.pi0_x_ND, self.pi0_y_DN, self.pi1_z_DD, self.pi0_x_ND_i, self.pi0_y_DN_i, self.pi1_z_DD_i,  b2_1_eq, u2_2, f1_3)
            ker_la.projector_tensor_strong(self.pi0_x_DN, self.pi0_y_ND, self.pi1_z_DD, self.pi0_x_DN_i, self.pi0_y_ND_i, self.pi1_z_DD_i, -b2_2_eq, u2_1, f1_3)
        
        # solve for coefficients with interpolation/histopolation matrix in case of gobal projector
        if   self.kind_projector == 'global':
            return self.projectors_3d.solve_V1(f1_1, f1_2, f1_3)
        else:
            return self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
    
    
    # =======================================================
    def T_transposed(self, b2_eq, f1, basis_u='0_form', bc_u1=['free', 'free']):
        
        # equilibrium magnetic field (2-form)
        b2_1_eq, b2_2_eq, b2_3_eq = self.tensor_space.unravel_2form(b2_eq)
        
        # final coefficients of the operation u = T^T(f1) (either 0-form or 2-form)
        if   basis_u == '0_form':
            u0_1 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            u0_2 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            u0_3 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
        elif basis_u == '2_form':
            u2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
            u2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
            u2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
        
        
        # contract f1 with inverse projection matrices: f1_abc = f1_ijk * I1^(-1)_ia * I2^(-1)_jb * I3^(-1)_kc for global pr.
        if self.kind_projector == 'global':
            
            if self.polar == False:
                
                f1_1, f1_2, f1_3 = self.tensor_space.unravel_1form(f1)
                
                f1_1[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].D_inv.T, self.projectors_1d[1].N_inv.T, self.projectors_1d[2].N_inv.T], f1_1)
                f1_2[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].N_inv.T, self.projectors_1d[1].D_inv.T, self.projectors_1d[2].N_inv.T], f1_2)
                f1_3[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].N_inv.T, self.projectors_1d[1].N_inv.T, self.projectors_1d[2].D_inv.T], f1_3)
                
            else:
                
                n0_pol_2D = self.tensor_space.Nbase0_pol
                n1_pol_2D = self.tensor_space.Nbase1_pol
                
                f1_12 = f1[:n1_pol_2D*self.tensor_space.NbaseN[2] ].reshape(n1_pol_2D, self.tensor_space.NbaseN[2])
                f1_3  = f1[ n1_pol_2D*self.tensor_space.NbaseN[2]:].reshape(n0_pol_2D, self.tensor_space.NbaseD[2])
                
                f1_12 = self.projectors_3d.N_inv[2].T.dot(self.projectors_3d.I1_pol_inv.T.dot(f1_12).T).T
                f1_3  = self.projectors_3d.D_inv[2].T.dot(self.projectors_3d.I0_pol_inv.T.dot(f1_3 ).T).T
                
                # apply transposed projection extraction operator
                f1_1, f1_2, f1_3 = self.tensor_space.unravel_1form(self.projectors_3d.P1.T.dot(np.concatenate((f1_12.flatten(), f1_3.flatten()))))
                         
        # not needed for local projector
        else:
            f1_1, f1_2, f1_3 = self.tensor_space.unravel_1form(f1)
        
        
        # bulk velocity is a 0-form
        if   basis_u == '0_form':
            
            # 1st component
            ker_la.projector_tensor_weak(self.pi0_x_DN, self.pi1_y_DN, self.pi0_z_NN, self.pi0_x_DN_i, self.pi1_y_DN_i, self.pi0_z_NN_i,  f1_2, b2_3_eq, u0_1)
            ker_la.projector_tensor_weak(self.pi0_x_DN, self.pi0_y_NN, self.pi1_z_DN, self.pi0_x_DN_i, self.pi0_y_NN_i, self.pi1_z_DN_i, -f1_3, b2_2_eq, u0_1)

            # 2nd component
            ker_la.projector_tensor_weak(self.pi0_x_NN, self.pi0_y_DN, self.pi1_z_DN, self.pi0_x_NN_i, self.pi0_y_DN_i, self.pi1_z_DN_i,  f1_3, b2_1_eq, u0_2)
            ker_la.projector_tensor_weak(self.pi1_x_DN, self.pi0_y_DN, self.pi0_z_NN, self.pi1_x_DN_i, self.pi0_y_DN_i, self.pi0_z_NN_i, -f1_1, b2_3_eq, u0_2)

            # 3rd component
            ker_la.projector_tensor_weak(self.pi1_x_DN, self.pi0_y_NN, self.pi0_z_DN, self.pi1_x_DN_i, self.pi0_y_NN_i, self.pi0_z_DN_i,  f1_1, b2_2_eq, u0_3)
            ker_la.projector_tensor_weak(self.pi0_x_NN, self.pi1_y_DN, self.pi0_z_DN, self.pi0_x_NN_i, self.pi1_y_DN_i, self.pi0_z_DN_i, -f1_2, b2_1_eq, u0_3)
            
            # apply boundary conditions
            if bc_u1[0] == 'dirichlet':
                u0_1[ 0, :, :] = 0.
            if bc_u1[1] == 'dirichlet':
                u0_1[-1, :, :] = 0.
                
            return self.tensor_space.ravel_pform(u0_1, u0_2, u0_3)
        
        # bulk velocity is a 2-form
        elif basis_u == '2_form':
            
            # 1st component
            ker_la.projector_tensor_weak(self.pi0_x_DN, self.pi1_y_DD, self.pi0_z_ND, self.pi0_x_DN_i, self.pi1_y_DD_i, self.pi0_z_ND_i,  f1_2, b2_3_eq, u2_1)
            ker_la.projector_tensor_weak(self.pi0_x_DN, self.pi0_y_ND, self.pi1_z_DD, self.pi0_x_DN_i, self.pi0_y_ND_i, self.pi1_z_DD_i, -f1_3, b2_2_eq, u2_1)

            # 2nd component
            ker_la.projector_tensor_weak(self.pi0_x_ND, self.pi0_y_DN, self.pi1_z_DD, self.pi0_x_ND_i, self.pi0_y_DN_i, self.pi1_z_DD_i,  f1_3, b2_1_eq, u2_2)
            ker_la.projector_tensor_weak(self.pi1_x_DD, self.pi0_y_DN, self.pi0_z_ND, self.pi1_x_DD_i, self.pi0_y_DN_i, self.pi0_z_ND_i, -f1_1, b2_3_eq, u2_2)

            # 3rd component
            ker_la.projector_tensor_weak(self.pi1_x_DD, self.pi0_y_ND, self.pi0_z_DN, self.pi1_x_DD_i, self.pi0_y_ND_i, self.pi0_z_DN_i,  f1_1, b2_2_eq, u2_3)
            ker_la.projector_tensor_weak(self.pi0_x_ND, self.pi1_y_DD, self.pi0_z_DN, self.pi0_x_ND_i, self.pi1_y_DD_i, self.pi0_z_DN_i, -f1_2, b2_1_eq, u2_3)
            
            # apply boundary conditions
            if bc_u1[0] == 'dirichlet':
                u2_1[ 0, :, :] = 0.
            if bc_u1[1] == 'dirichlet':
                u2_1[-1, :, :] = 0.
            
            # apply polar extraction operator
            if self.polar == True:
                return self.tensor_space.E2.dot(self.tensor_space.ravel_pform(u2_1, u2_2, u2_3))
            else:
                return self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
    
    
    # =======================================================
    def T_as_matrix(self, b2_eq, tol, basis_u='0_form'):
        
        # approximate inverse interpolation-/histopolation matrices in case of global projector
        if self.kind_projector == 'global':
            
            if self.polar == False:
            
                N_inv_reduced = [np.copy(N_inv) for N_inv in self.projectors_3d.N_inv]
                D_inv_reduced = [np.copy(D_inv) for D_inv in self.projectors_3d.D_inv]

                for a in range(3):
                    N_inv_reduced[a][np.abs(N_inv_reduced[a]) < tol] = 0.
                    D_inv_reduced[a][np.abs(D_inv_reduced[a]) < tol] = 0.
                    
            else:
                
                I0_pol_inv_reduced = np.copy(self.projectors_3d.I0_pol_inv)
                I1_pol_inv_reduced = np.copy(self.projectors_3d.I1_pol_inv)
                
                I0_pol_inv_reduced[np.abs(I0_pol_inv_reduced) < tol] = 0.
                I1_pol_inv_reduced[np.abs(I1_pol_inv_reduced) < tol] = 0.
                
                N_inv_z_reduced = np.copy(self.projectors_3d.N_inv[2])
                D_inv_z_reduced = np.copy(self.projectors_3d.D_inv[2])
                
                N_inv_z_reduced[np.abs(N_inv_z_reduced) < tol] = 0.
                D_inv_z_reduced[np.abs(D_inv_z_reduced) < tol] = 0.
                
        
        # equilibrium magnetic field (2-form)
        b2_1_eq, b2_2_eq, b2_3_eq = self.tensor_space.unravel_2form(b2_eq)
            
        # resulting 1-form coefficients in each column
        f1_1 = np.zeros(self.tensor_space.Nbase_1form[0], dtype=float)
        f1_2 = np.zeros(self.tensor_space.Nbase_1form[1], dtype=float)
        f1_3 = np.zeros(self.tensor_space.Nbase_1form[2], dtype=float)
        
        # non-zero values and coefficients in final block matrix
        values = [np.array([], dtype=float) for i in range(3)]
        col    = [np.array([], dtype=int)   for i in range(3)]
        row    = [np.array([], dtype=int)   for i in range(3)]
        
        # bulk velocity is a 0-form
        if basis_u == '0_form':
        
            # =================== 1st column (T_21 & T_31) ================
            for i1 in range(self.tensor_space.Nbase_0form[0]):

                ind21x = np.nonzero(self.pi0_x_DN[:, :, i1])
                ind31x = np.nonzero(self.pi0_x_DN[:, :, i1])
                ind21x = np.vstack((ind21x[0], ind21x[1]))
                ind31x = np.vstack((ind31x[0], ind31x[1]))

                for i2 in range(self.tensor_space.Nbase_0form[1]):

                    ind21y = np.nonzero(self.pi1_y_DN[:, :, i2])
                    ind31y = np.nonzero(self.pi0_y_NN[:, :, i2])
                    ind21y = np.vstack((ind21y[0], ind21y[1]))
                    ind31y = np.vstack((ind31y[0], ind31y[1]))

                    for i3 in range(self.tensor_space.Nbase_0form[2]):

                        ind21z = np.nonzero(self.pi0_z_NN[:, :, i3])
                        ind31z = np.nonzero(self.pi1_z_DN[:, :, i3])
                        ind21z = np.vstack((ind21z[0], ind21z[1]))
                        ind31z = np.vstack((ind31z[0], ind31z[1]))

                        f1_1[:, :, :] = 0.
                        f1_2[:, :, :] = 0.
                        f1_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_DN[:, :, i1], self.pi1_y_DN[:, :, i2], self.pi0_z_NN[:, :, i3], ind21x, ind21y, ind21z,  b2_3_eq, f1_2)
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_DN[:, :, i1], self.pi0_y_NN[:, :, i2], self.pi1_z_DN[:, :, i3], ind31x, ind31y, ind31z, -b2_2_eq, f1_3)   

                        if self.kind_projector == 'global':
                            f1_2[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], D_inv_reduced[1], N_inv_reduced[2]], f1_2)
                            f1_3[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], N_inv_reduced[1], D_inv_reduced[2]], f1_3)

                        column    = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
                        indices1  = np.nonzero(column)[0]
                        values[0] = np.append(values[0], column[indices1])
                        row[0]    = np.append(row[0], indices1)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                        col[0]    = np.append(col[0], col_ind*np.ones(len(indices1), dtype=int))
            # =============================================================             


            # =================== 2nd column (T_12 & T_32) ================
            for i1 in range(self.tensor_space.Nbase_0form[0]):

                ind12x = np.nonzero(self.pi1_x_DN[:, :, i1])
                ind32x = np.nonzero(self.pi0_x_NN[:, :, i1])
                ind12x = np.vstack((ind12x[0], ind12x[1]))
                ind32x = np.vstack((ind32x[0], ind32x[1]))

                for i2 in range(self.tensor_space.Nbase_0form[1]):

                    ind12y = np.nonzero(self.pi0_y_DN[:, :, i2])
                    ind32y = np.nonzero(self.pi0_y_DN[:, :, i2])
                    ind12y = np.vstack((ind12y[0], ind12y[1]))
                    ind32y = np.vstack((ind32y[0], ind32y[1]))

                    for i3 in range(self.tensor_space.Nbase_0form[2]):

                        ind12z = np.nonzero(self.pi0_z_NN[:, :, i3])
                        ind32z = np.nonzero(self.pi1_z_DN[:, :, i3])
                        ind12z = np.vstack((ind12z[0], ind12z[1]))
                        ind32z = np.vstack((ind32z[0], ind32z[1]))

                        f1_1[:, :, :] = 0.
                        f1_2[:, :, :] = 0.
                        f1_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi1_x_DN[:, :, i1], self.pi0_y_DN[:, :, i2], self.pi0_z_NN[:, :, i3], ind12x, ind12y, ind12z, -b2_3_eq, f1_1)
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_NN[:, :, i1], self.pi0_y_DN[:, :, i2], self.pi1_z_DN[:, :, i3], ind32x, ind32y, ind32z,  b2_1_eq, f1_3)

                        if self.kind_projector == 'global':
                            f1_1[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], N_inv_reduced[1], N_inv_reduced[2]], f1_1)
                            f1_3[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], N_inv_reduced[1], D_inv_reduced[2]], f1_3)

                        column    = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
                        indices2  = np.nonzero(column)[0]
                        values[1] = np.append(values[1], column[indices2])
                        row[1]    = np.append(row[1], indices2)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                        col[1]    = np.append(col[1], col_ind*np.ones(len(indices2), dtype=int))
            # =============================================================


            # =================== 3rd column (T_13 & T_23) ================
            for i1 in range(self.tensor_space.Nbase_0form[0]):

                ind13x = np.nonzero(self.pi1_x_DN[:, :, i1])
                ind23x = np.nonzero(self.pi0_x_NN[:, :, i1])
                ind13x = np.vstack((ind13x[0], ind13x[1]))
                ind23x = np.vstack((ind23x[0], ind23x[1]))

                for i2 in range(self.tensor_space.Nbase_0form[1]):

                    ind13y = np.nonzero(self.pi0_y_NN[:, :, i2])
                    ind23y = np.nonzero(self.pi1_y_DN[:, :, i2])
                    ind13y = np.vstack((ind13y[0], ind13y[1]))
                    ind23y = np.vstack((ind23y[0], ind23y[1]))

                    for i3 in range(self.tensor_space.Nbase_0form[2]):

                        ind13z = np.nonzero(self.pi0_z_DN[:, :, i3])
                        ind23z = np.nonzero(self.pi0_z_DN[:, :, i3])
                        ind13z = np.vstack((ind13z[0], ind13z[1]))
                        ind23z = np.vstack((ind23z[0], ind23z[1]))

                        f1_1[:, :, :] = 0.
                        f1_2[:, :, :] = 0.
                        f1_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi1_x_DN[:, :, i1], self.pi0_y_NN[:, :, i2], self.pi0_z_DN[:, :, i3], ind13x, ind13y, ind13z,  b2_2_eq, f1_1)
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_NN[:, :, i1], self.pi1_y_DN[:, :, i2], self.pi0_z_DN[:, :, i3], ind23x, ind23y, ind23z, -b2_1_eq, f1_2)

                        if self.kind_projector == 'global':
                            f1_1[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], N_inv_reduced[1], N_inv_reduced[2]], f1_1)
                            f1_2[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], D_inv_reduced[1], N_inv_reduced[2]], f1_2)

                        column    = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
                        indices3  = np.nonzero(column)[0]
                        values[2] = np.append(values[2], column[indices3])
                        row[2]    = np.append(row[2], indices3)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                        col[2]    = np.append(col[2], col_ind*np.ones(len(indices3), dtype=int))        
            # =============================================================

            T_1 = spa.csr_matrix((values[0], (row[0], col[0])), shape=(sum(self.tensor_space.Ntot_1form), self.tensor_space.Ntot_0form))
            T_2 = spa.csr_matrix((values[1], (row[1], col[1])), shape=(sum(self.tensor_space.Ntot_1form), self.tensor_space.Ntot_0form))
            T_3 = spa.csr_matrix((values[2], (row[2], col[2])), shape=(sum(self.tensor_space.Ntot_1form), self.tensor_space.Ntot_0form))
            
            return spa.bmat([[T_1, T_2, T_3]], format='csr')
            
        # bulk velocity is a 2-form
        elif basis_u == '2_form':
            
            # =================== 1st column (T_21 & T_31) ================
            for i1 in range(self.tensor_space.Nbase_2form[0][0]):

                ind21x = np.nonzero(self.pi0_x_DN[:, :, i1])
                ind31x = np.nonzero(self.pi0_x_DN[:, :, i1])
                ind21x = np.vstack((ind21x[0], ind21x[1]))
                ind31x = np.vstack((ind31x[0], ind31x[1]))

                for i2 in range(self.tensor_space.Nbase_2form[0][1]):

                    ind21y = np.nonzero(self.pi1_y_DD[:, :, i2])
                    ind31y = np.nonzero(self.pi0_y_ND[:, :, i2])
                    ind21y = np.vstack((ind21y[0], ind21y[1]))
                    ind31y = np.vstack((ind31y[0], ind31y[1]))

                    for i3 in range(self.tensor_space.Nbase_2form[0][2]):

                        ind21z = np.nonzero(self.pi0_z_ND[:, :, i3])
                        ind31z = np.nonzero(self.pi1_z_DD[:, :, i3])
                        ind21z = np.vstack((ind21z[0], ind21z[1]))
                        ind31z = np.vstack((ind31z[0], ind31z[1]))

                        f1_1[:, :, :] = 0.
                        f1_2[:, :, :] = 0.
                        f1_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_DN[:, :, i1], self.pi1_y_DD[:, :, i2], self.pi0_z_ND[:, :, i3], ind21x, ind21y, ind21z,  b2_3_eq, f1_2)
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_DN[:, :, i1], self.pi0_y_ND[:, :, i2], self.pi1_z_DD[:, :, i3], ind31x, ind31y, ind31z, -b2_2_eq, f1_3)   

                        if self.kind_projector == 'global':
                            if self.polar == False:
                                f1_2[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], D_inv_reduced[1], N_inv_reduced[2]], f1_2)
                                f1_3[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], N_inv_reduced[1], D_inv_reduced[2]], f1_3)
                                column = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
                            else:
                                
                                rhs_12  = self.projectors_3d.P1_pol.dot(np.block([[f1_1.reshape(f1_1.shape[0]*f1_1.shape[1], f1_1.shape[2])], [f1_2.reshape(f1_2.shape[0]*f1_2.shape[1], f1_2.shape[2])]])) 
                                rhs_3   = self.projectors_3d.P0_pol_2D.dot(f1_3.reshape(f1_3.shape[0]*f1_3.shape[1], f1_3.shape[2]))
                                
                                rhs_12  = N_inv_z_reduced.dot(I1_pol_inv_reduced.dot(rhs_12).T).T
                                rhs_3   = D_inv_z_reduced.dot(I0_pol_inv_reduced.dot(rhs_3).T).T
                                
                                column = np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
                        else:
                            column = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
                            
                        indices1  = np.nonzero(column)[0]
                        values[0] = np.append(values[0], column[indices1])
                        row[0]    = np.append(row[0], indices1)
                        col_ind   = self.tensor_space.NbaseD[1]*self.tensor_space.NbaseD[2]*i1 + self.tensor_space.NbaseD[2]*i2 + i3
                        col[0]    = np.append(col[0], col_ind*np.ones(len(indices1), dtype=int))
            # ============================================================= 
            
            print('Block 1 of TAU_local done!')
            
            # =================== 2nd column (T_12 & T_32) ================
            for i1 in range(self.tensor_space.Nbase_2form[1][0]):

                ind12x = np.nonzero(self.pi1_x_DD[:, :, i1])
                ind32x = np.nonzero(self.pi0_x_ND[:, :, i1])

                ind12x = np.vstack((ind12x[0], ind12x[1]))
                ind32x = np.vstack((ind32x[0], ind32x[1]))

                for i2 in range(self.tensor_space.Nbase_2form[1][1]):

                    ind12y = np.nonzero(self.pi0_y_DN[:, :, i2])
                    ind32y = np.nonzero(self.pi0_y_DN[:, :, i2])

                    ind12y = np.vstack((ind12y[0], ind12y[1]))
                    ind32y = np.vstack((ind32y[0], ind32y[1]))

                    for i3 in range(self.tensor_space.Nbase_2form[1][2]):

                        ind12z = np.nonzero(self.pi0_z_ND[:, :, i3])
                        ind32z = np.nonzero(self.pi1_z_DD[:, :, i3])

                        ind12z = np.vstack((ind12z[0], ind12z[1]))
                        ind32z = np.vstack((ind32z[0], ind32z[1]))

                        f1_1[:, :, :] = 0.
                        f1_2[:, :, :] = 0.
                        f1_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi1_x_DD[:, :, i1], self.pi0_y_DN[:, :, i2], self.pi0_z_ND[:, :, i3], ind12x, ind12y, ind12z, -b2_3_eq, f1_1)
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_ND[:, :, i1], self.pi0_y_DN[:, :, i2], self.pi1_z_DD[:, :, i3], ind32x, ind32y, ind32z,  b2_1_eq, f1_3)

                        if self.kind_projector == 'global':
                            if self.polar == False:
                                f1_1[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], N_inv_reduced[1], N_inv_reduced[2]], f1_1)
                                f1_3[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], N_inv_reduced[1], D_inv_reduced[2]], f1_3)
                                column = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
                            else:
                                
                                rhs_12  = self.projectors_3d.P1_pol.dot(np.block([[f1_1.reshape(f1_1.shape[0]*f1_1.shape[1], f1_1.shape[2])], [f1_2.reshape(f1_2.shape[0]*f1_2.shape[1], f1_2.shape[2])]])) 
                                rhs_3   = self.projectors_3d.P0_pol.dot(f1_3.reshape(f1_3.shape[0]*f1_3.shape[1], f1_3.shape[2]))
                                
                                rhs_12  = N_inv_z_reduced.dot(I1_pol_inv_reduced.dot(rhs_12).T).T
                                rhs_3   = D_inv_z_reduced.dot(I0_pol_inv_reduced.dot(rhs_3).T).T
                                
                                column = np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
                        else:
                            column = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)

                        indices2  = np.nonzero(column)[0]
                        values[1] = np.append(values[1], column[indices2])

                        row[1]    = np.append(row[1], indices2)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseD[2]*i1 + self.tensor_space.NbaseD[2]*i2 + i3

                        col[1]    = np.append(col[1], col_ind*np.ones(len(indices2), dtype=int))
            # =============================================================
            
            print('Block 2 of TAU_local done!')
            
            # =================== 3rd column (T_13 & T_23) ================
            for i1 in range(self.tensor_space.Nbase_2form[2][0]):

                ind13x = np.nonzero(self.pi1_x_DD[:, :, i1])
                ind23x = np.nonzero(self.pi0_x_ND[:, :, i1])

                ind13x = np.vstack((ind13x[0], ind13x[1]))
                ind23x = np.vstack((ind23x[0], ind23x[1]))

                for i2 in range(self.tensor_space.Nbase_2form[2][1]):

                    ind13y = np.nonzero(self.pi0_y_ND[:, :, i2])
                    ind23y = np.nonzero(self.pi1_y_DD[:, :, i2])

                    ind13y = np.vstack((ind13y[0], ind13y[1]))
                    ind23y = np.vstack((ind23y[0], ind23y[1]))

                    for i3 in range(self.tensor_space.Nbase_2form[2][2]):

                        ind13z = np.nonzero(self.pi0_z_DN[:, :, i3])
                        ind23z = np.nonzero(self.pi0_z_DN[:, :, i3])

                        ind13z = np.vstack((ind13z[0], ind13z[1]))
                        ind23z = np.vstack((ind23z[0], ind23z[1]))

                        f1_1[:, :, :] = 0.
                        f1_2[:, :, :] = 0.
                        f1_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi1_x_DD[:, :, i1], self.pi0_y_ND[:, :, i2], self.pi0_z_DN[:, :, i3], ind13x, ind13y, ind13z,  b2_2_eq, f1_1)
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_ND[:, :, i1], self.pi1_y_DD[:, :, i2], self.pi0_z_DN[:, :, i3], ind23x, ind23y, ind23z, -b2_1_eq, f1_2)

                        if self.kind_projector == 'global':
                            if self.polar == False:
                                f1_1[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], N_inv_reduced[1], N_inv_reduced[2]], f1_1)
                                f1_2[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], D_inv_reduced[1], N_inv_reduced[2]], f1_2)
                                column = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)
                            else:
                                
                                rhs_12  = self.projectors_3d.P1_pol.dot(np.block([[f1_1.reshape(f1_1.shape[0]*f1_1.shape[1], f1_1.shape[2])], [f1_2.reshape(f1_2.shape[0]*f1_2.shape[1], f1_2.shape[2])]])) 
                                rhs_3   = self.projectors_3d.P0_pol.dot(f1_3.reshape(f1_3.shape[0]*f1_3.shape[1], f1_3.shape[2]))
                                
                                rhs_12  = N_inv_z_reduced.dot(I1_pol_inv_reduced.dot(rhs_12).T).T
                                rhs_3   = D_inv_z_reduced.dot(I0_pol_inv_reduced.dot(rhs_3).T).T
                                
                                column = np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
                        else:
                            column = self.tensor_space.ravel_pform(f1_1, f1_2, f1_3)

                        indices3  = np.nonzero(column)[0]
                        values[2] = np.append(values[2], column[indices3])

                        row[2]    = np.append(row[2], indices3)
                        col_ind   = self.tensor_space.NbaseD[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3

                        col[2]    = np.append(col[2], col_ind*np.ones(len(indices3), dtype=int))       
            # =============================================================
            
            print('Block 3 of TAU_local done!')
            
            
            
            if self.polar == True:
                
                n0_pol_2D = self.tensor_space.Nbase0_pol
                n1_pol_2D = self.tensor_space.Nbase1_pol
                
                n_tot     = n1_pol_2D*self.tensor_space.NbaseN[2] + n0_pol_2D*self.tensor_space.NbaseD[2]

                T_1 = spa.csr_matrix((values[0], (row[0], col[0])), shape=(n_tot, self.tensor_space.Ntot_2form[0]))
                T_2 = spa.csr_matrix((values[1], (row[1], col[1])), shape=(n_tot, self.tensor_space.Ntot_2form[1]))
                T_3 = spa.csr_matrix((values[2], (row[2], col[2])), shape=(n_tot, self.tensor_space.Ntot_2form[2]))
                
                return spa.bmat([[T_1, T_2, T_3]], format='csr').dot(self.tensor_space.E2.T)
            else:
                
                T_1 = spa.csr_matrix((values[0], (row[0], col[0])), shape=(sum(self.tensor_space.Ntot_1form), self.tensor_space.Ntot_2form[0]))
                T_2 = spa.csr_matrix((values[1], (row[1], col[1])), shape=(sum(self.tensor_space.Ntot_1form), self.tensor_space.Ntot_2form[1]))
                T_3 = spa.csr_matrix((values[2], (row[2], col[2])), shape=(sum(self.tensor_space.Ntot_1form), self.tensor_space.Ntot_2form[2]))
                
                return spa.bmat([[T_1, T_2, T_3]], format='csr')
            
            
    
    # =============================================
    def W(self, rho_eq, u, basis_rho_eq='0_form', basis_u='0_form'):
        
        # bulk velocity is a 0-form
        if   basis_u == '0_form':
            
            # equilibrium mass density (0-form)
            rho0_eq = rho_eq.reshape(self.tensor_space.Nbase_0form)
            
            # input bulk velocity coefficients (0-form)
            u0_1, u0_2, u0_3 = self.tensor_space.unravel_0form(u)

            # final coefficients of operation f0 = W(u0) (0-form)
            f0_1 = np.zeros(self.tensor_space.Nbase_0form, dtype=float)
            f0_2 = np.zeros(self.tensor_space.Nbase_0form, dtype=float)
            f0_3 = np.zeros(self.tensor_space.Nbase_0form, dtype=float)
            
            ker_la.projector_tensor_strong(self.pi0_x_NN, self.pi0_y_NN, self.pi0_z_NN, self.pi0_x_NN_i, self.pi0_y_NN_i, self.pi0_z_NN_i, rho0_eq, u0_1, f0_1)
            ker_la.projector_tensor_strong(self.pi0_x_NN, self.pi0_y_NN, self.pi0_z_NN, self.pi0_x_NN_i, self.pi0_y_NN_i, self.pi0_z_NN_i, rho0_eq, u0_2, f0_2)
            ker_la.projector_tensor_strong(self.pi0_x_NN, self.pi0_y_NN, self.pi0_z_NN, self.pi0_x_NN_i, self.pi0_y_NN_i, self.pi0_z_NN_i, rho0_eq, u0_3, f0_3)
            
            # solve for coefficients with interpolation/histopolation matrix in case of gobal projector
            if self.kind_projector == 'global':
                return np.concatenate((self.projectors_3d.solve_V0(f0_1), self.projectors_3d.solve_V0(f0_2), self.projectors_3d.solve_V0(f0_3)))
            else:
                return self.tensor_space.ravel_pform(f0_1, f0_2, f0_3)
            
        # bulk velocity is a 2-form
        elif basis_u == '2_form':
            
            # equilibrium mass density (3-form)
            rho3_eq = rho_eq.reshape(self.tensor_space.Nbase_3form)
            
            # input bulk velocity coefficients (2-form)
            if self.polar == False:
                u2_1, u2_2, u2_3 = self.tensor_space.unravel_2form(u)
            else:
                u2_1, u2_2, u2_3 = self.tensor_space.unravel_2form(self.tensor_space.E2.T.dot(u))

            # final coefficients of operation f2 = W(u2) (2-form)
            f2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
            f2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
            f2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
            
            ker_la.projector_tensor_strong(self.pi0_x_DN, self.pi1_y_DD, self.pi1_z_DD, self.pi0_x_DN_i, self.pi1_y_DD_i, self.pi1_z_DD_i, rho3_eq, u2_1, f2_1)
            ker_la.projector_tensor_strong(self.pi1_x_DD, self.pi0_y_DN, self.pi1_z_DD, self.pi1_x_DD_i, self.pi0_y_DN_i, self.pi1_z_DD_i, rho3_eq, u2_2, f2_2)
            ker_la.projector_tensor_strong(self.pi1_x_DD, self.pi1_y_DD, self.pi0_z_DN, self.pi1_x_DD_i, self.pi1_y_DD_i, self.pi0_z_DN_i, rho3_eq, u2_3, f2_3)
            
            # solve for coefficients with interpolation/histopolation matrix in case of gobal projector
            if self.kind_projector == 'global':                     
                return self.projectors_3d.solve_V2(f2_1, f2_2, f2_3)
            else:
                return self.tensor_space.ravel_pform(f2_1, f2_2, f2_3)
            
            
    # =============================================
    def W_transposed(self, rho_eq, f, basis_rho_eq='0_form', basis_u='0_form', bc_u1=['free', 'free']):
        
        # final coefficients of the operation u = W^T(f) (either 0-form or 2-form)
        if   basis_u == '0_form':
            u0_1 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            u0_2 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            u0_3 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
        elif basis_u == '2_form':
            u2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
            u2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
            u2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
        
        # contract f with inverse projection matrices: f_abc = f_ijk * I1^(-1)_ia * I2^(-1)_jb * I3^(-1)_kc for global pr.
        if self.kind_projector == 'global':
            
            if self.polar == False:
            
                if   basis_u == '0_form':
                    
                    f0_1, f0_2, f0_3 = self.tensor_space.unravel_0form(f)
                    
                    f0_1[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].N_inv.T, self.projectors_1d[1].N_inv.T, self.projectors_1d[2].N_inv.T], f0_1)
                    f0_2[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].N_inv.T, self.projectors_1d[1].N_inv.T, self.projectors_1d[2].N_inv.T], f0_2)
                    f0_3[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].N_inv.T, self.projectors_1d[1].N_inv.T, self.projectors_1d[2].N_inv.T], f0_3)
                
                elif basis_u == '2_form':
                    
                    f2_1, f2_2, f2_3 = self.tensor_space.unravel_2form(f)
                    
                    f2_1[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].N_inv.T, self.projectors_1d[1].D_inv.T, self.projectors_1d[2].D_inv.T], f2_1)
                    f2_2[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].D_inv.T, self.projectors_1d[1].N_inv.T, self.projectors_1d[2].D_inv.T], f2_2)
                    f2_3[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].D_inv.T, self.projectors_1d[1].D_inv.T, self.projectors_1d[2].N_inv.T], f2_3)
                    
            else:
                
                n2_pol_2D = self.tensor_space.Nbase2_pol
                n3_pol_2D = self.tensor_space.Nbase3_pol
                
                f2_12 = f[:n2_pol_2D*self.tensor_space.NbaseD[2] ].reshape(n2_pol_2D, self.tensor_space.NbaseD[2])
                f2_3  = f[ n2_pol_2D*self.tensor_space.NbaseD[2]:].reshape(n3_pol_2D, self.tensor_space.NbaseN[2])
                
                f2_12 = self.projectors_3d.D_inv[2].T.dot(self.projectors_3d.I2_pol_inv.T.dot(f2_12).T).T
                f2_3  = self.projectors_3d.N_inv[2].T.dot(self.projectors_3d.I3_pol_inv.T.dot(f2_3 ).T).T
                
                # apply transposed projection extraction operator
                f2_1, f2_2, f2_3 = self.tensor_space.unravel_2form(self.projectors_3d.P2.T.dot(np.concatenate((f2_12.flatten(), f2_3.flatten()))))
                
                
        else:
            if   basis_u == '0_form':
                f0_1, f0_2, f0_3 = self.tensor_space.unravel_0form(f)
            elif basis_u == '2_form':
                f2_1, f2_2, f2_3 = self.tensor_space.unravel_2form(f)
              
        if   basis_u == '0_form':
            
            # equilibrium mass density (0-form)
            rho0_eq = rho_eq.reshape(self.tensor_space.Nbase_0form)
            
            ker_la.projector_tensor_weak(self.pi0_x_NN, self.pi0_y_NN, self.pi0_z_NN, self.pi0_x_NN_i, self.pi0_y_NN_i, self.pi0_z_NN_i, f0_1, rho0_eq, u0_1)
            ker_la.projector_tensor_weak(self.pi0_x_NN, self.pi0_y_NN, self.pi0_z_NN, self.pi0_x_NN_i, self.pi0_y_NN_i, self.pi0_z_NN_i, f0_2, rho0_eq, u0_2)
            ker_la.projector_tensor_weak(self.pi0_x_NN, self.pi0_y_NN, self.pi0_z_NN, self.pi0_x_NN_i, self.pi0_y_NN_i, self.pi0_z_NN_i, f0_3, rho0_eq, u0_3)
            
            # apply boundary conditions
            if bc_u1[0] == 'dirichlet':
                u0_1[ 0, :, :] = 0.
            if bc_u1[1] == 'dirichlet':
                u0_1[-1, :, :] = 0.
                
            return self.tensor_space.ravel_pform(u0_1, u0_2, u0_3)
        
        elif basis_u == '2_form':
            
            # equilibrium mass density (3-form)
            rho3_eq = rho_eq.reshape(self.tensor_space.Nbase_3form)
            
            ker_la.projector_tensor_weak(self.pi0_x_DN, self.pi1_y_DD, self.pi1_z_DD, self.pi0_x_DN_i, self.pi1_y_DD_i, self.pi1_z_DD_i, f2_1, rho3_eq, u2_1)
            ker_la.projector_tensor_weak(self.pi1_x_DD, self.pi0_y_DN, self.pi1_z_DD, self.pi1_x_DD_i, self.pi0_y_DN_i, self.pi1_z_DD_i, f2_2, rho3_eq, u2_2)
            ker_la.projector_tensor_weak(self.pi1_x_DD, self.pi1_y_DD, self.pi0_z_DN, self.pi1_x_DD_i, self.pi1_y_DD_i, self.pi0_z_DN_i, f2_3, rho3_eq, u2_3)
            
            # apply boundary conditions
            if bc_u1[0] == 'dirichlet':
                u2_1[ 0, :, :] = 0.
            if bc_u1[1] == 'dirichlet':
                u2_1[-1, :, :] = 0.
        
            # apply polar extraction operator
            if self.polar == True:
                return self.projectors_3d.E2.dot(self.tensor_space.ravel_pform(u2_1, u2_2, u2_3))
            else:
                return self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
    
    
    # =======================================================
    def W_as_matrix(self, rho_eq, tol, basis_rho_eq='0_form', basis_u='0_form'):
        
        # approximate inverse interpolation-/histopolation matrices in case of global projector
        if self.kind_projector == 'global':
            
            if self.polar == False:
                
                N_inv_reduced = [np.copy(N_inv) for N_inv in self.projectors_3d.N_inv]
                D_inv_reduced = [np.copy(D_inv) for D_inv in self.projectors_3d.D_inv]

                for a in range(3):
                    N_inv_reduced[a][np.abs(N_inv_reduced[a]) < tol] = 0.
                    D_inv_reduced[a][np.abs(D_inv_reduced[a]) < tol] = 0.
                    
            else:
                
                I2_pol_inv_reduced = np.copy(self.projectors_3d.I2_pol_inv)
                I3_pol_inv_reduced = np.copy(self.projectors_3d.I3_pol_inv)
                
                I2_pol_inv_reduced[np.abs(I2_2D_pol_reduced) < tol] = 0.
                I3_pol_inv_reduced[np.abs(I3_2D_pol_reduced) < tol] = 0.
                
                N_inv_z_reduced = np.copy(self.projectors_3d.N_inv[2])
                D_inv_z_reduced = np.copy(self.projectors_3d.D_inv[2])
                
                N_inv_z_reduced[np.abs(N_inv_z_reduced) < tol] = 0.
                D_inv_z_reduced[np.abs(D_inv_z_reduced) < tol] = 0.
            
        
        # resulting coefficients in each column
        if   basis_u == '0_form':
            u0_1 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            u0_2 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            u0_3 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
        elif basis_u == '2_form':
            u2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
            u2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
            u2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
        
        # non-zero values and coefficients in final block matrix
        values = [np.array([], dtype=float) for i in range(3)]
        col    = [np.array([], dtype=int)   for i in range(3)]
        row    = [np.array([], dtype=int)   for i in range(3)]
        
        
        # bulk velocity is a 0-form
        if basis_u == '0_form':
            
            rho0_eq = rho_eq.reshape(self.tensor_space.Nbase_0form)
            
            # =================== 1st column (W_11) =======================
            for i1 in range(self.tensor_space.Nbase_0form[0]):

                ind11x = np.nonzero(self.pi0_x_NN[:, :, i1])
                ind11x = np.vstack((ind11x[0], ind11x[1]))

                for i2 in range(self.tensor_space.Nbase_0form[1]):

                    ind11y = np.nonzero(self.pi0_y_NN[:, :, i2])
                    ind11y = np.vstack((ind11y[0], ind11y[1]))

                    for i3 in range(self.tensor_space.Nbase_0form[2]):

                        ind11z = np.nonzero(self.pi0_z_NN[:, :, i3])
                        ind11z = np.vstack((ind11z[0], ind11z[1]))

                        u0_1[:, :, :] = 0.
                        u0_2[:, :, :] = 0.
                        u0_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_NN[:, :, i1], self.pi0_y_NN[:, :, i2], self.pi0_z_NN[:, :, i3], ind11x, ind11y, ind11z, rho0_eq, u0_1)

                        if self.kind_projector == 'global':
                            u0_1[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], N_inv_reduced[1], N_inv_reduced[2]], u0_1)
                        column    = self.tensor_space.ravel_pform(u0_1, u0_2, u0_3)
                        indices1  = np.nonzero(column)[0]
                        values[0] = np.append(values[0], column[indices1])
                        row[0]    = np.append(row[0], indices1)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                        col[0]    = np.append(col[0], col_ind*np.ones(len(indices1), dtype=int))
            # =============================================================             


            # =================== 2nd column (W_22) =======================
            for i1 in range(self.tensor_space.Nbase_0form[0]):

                ind22x = np.nonzero(self.pi0_x_NN[:, :, i1])
                ind22x = np.vstack((ind22x[0], ind22x[1]))

                for i2 in range(self.tensor_space.Nbase_0form[1]):

                    ind22y = np.nonzero(self.pi0_y_NN[:, :, i2])
                    ind22y = np.vstack((ind22y[0], ind22y[1]))

                    for i3 in range(self.tensor_space.Nbase_0form[2]):

                        ind22z = np.nonzero(self.pi0_z_NN[:, :, i3])
                        ind22z = np.vstack((ind22z[0], ind22z[1]))

                        u0_1[:, :, :] = 0.
                        u0_2[:, :, :] = 0.
                        u0_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_NN[:, :, i1], self.pi0_y_NN[:, :, i2], self.pi0_z_NN[:, :, i3], ind22x, ind22y, ind22z, rho0_eq, u0_2)

                        if self.kind_projector == 'global':
                            u0_2[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], N_inv_reduced[1], N_inv_reduced[2]], u0_2)

                        column    = self.tensor_space.ravel_pform(u0_1, u0_2, u0_3)
                        indices2  = np.nonzero(column)[0]
                        values[1] = np.append(values[1], column[indices2])
                        row[1]    = np.append(row[1], indices2)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                        col[1]    = np.append(col[1], col_ind*np.ones(len(indices2), dtype=int))
            # =============================================================           



            # =================== 3rd column (W_33) =======================
            for i1 in range(self.tensor_space.Nbase_0form[0]):

                ind33x = np.nonzero(self.pi0_x_NN[:, :, i1])
                ind33x = np.vstack((ind33x[0], ind33x[1]))

                for i2 in range(self.tensor_space.Nbase_0form[1]):

                    ind33y = np.nonzero(self.pi0_y_NN[:, :, i2])
                    ind33y = np.vstack((ind33y[0], ind33y[1]))

                    for i3 in range(self.tensor_space.Nbase_0form[2]):

                        ind33z = np.nonzero(self.pi0_z_NN[:, :, i3])
                        ind33z = np.vstack((ind33z[0], ind33z[1]))

                        u0_1[:, :, :] = 0.
                        u0_2[:, :, :] = 0.
                        u0_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_NN[:, :, i1], self.pi0_y_NN[:, :, i2], self.pi0_z_NN[:, :, i3], ind33x, ind33y, ind33z, rho0_eq, u0_3)

                        if self.kind_projector == 'global':
                            u0_3[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], N_inv_reduced[1], N_inv_reduced[2]], u0_3)

                        column    = self.tensor_space.ravel_pform(u0_1, u0_2, u0_3)
                        indices3  = np.nonzero(column)[0]
                        values[2] = np.append(values[2], column[indices3])
                        row[2]    = np.append(row[2], indices3)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                        col[2]    = np.append(col[2], col_ind*np.ones(len(indices3), dtype=int))
            # =============================================================           
        
            W_1 = spa.csr_matrix((values[0], (row[0], col[0])), shape=(3*self.tensor_space.Ntot_0form, 3*self.tensor_space.Ntot_0form))
            W_2 = spa.csr_matrix((values[1], (row[1], col[1])), shape=(3*self.tensor_space.Ntot_0form, 3*self.tensor_space.Ntot_0form))
            W_3 = spa.csr_matrix((values[2], (row[2], col[2])), shape=(3*self.tensor_space.Ntot_0form, 3*self.tensor_space.Ntot_0form))

            return spa.bmat([[W_11, W_22, W_33]], format='csr')
        
        # bulk velocity is a 2-form
        elif basis_u == '2_form':
            
            rho3_eq = rho_eq.reshape(self.tensor_space.Nbase_3form)
            
            # =================== 1st block column (W_1) =======================
            for i1 in range(self.tensor_space.Nbase_2form[0][0]):

                ind1x = np.nonzero(self.pi0_x_DN[:, :, i1])
                ind1x = np.vstack((ind1x[0], ind1x[1]))

                for i2 in range(self.tensor_space.Nbase_2form[0][1]):

                    ind1y = np.nonzero(self.pi1_y_DD[:, :, i2])
                    ind1y = np.vstack((ind1y[0], ind1y[1]))

                    for i3 in range(self.tensor_space.Nbase_2form[0][2]):

                        ind1z = np.nonzero(self.pi1_z_DD[:, :, i3])
                        ind1z = np.vstack((ind1z[0], ind1z[1]))

                        u2_1[:, :, :] = 0.
                        u2_2[:, :, :] = 0.
                        u2_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi0_x_DN[:, :, i1], self.pi1_y_DD[:, :, i2], self.pi1_z_DD[:, :, i3], ind1x, ind1y, ind1z, rho3_eq, u2_1)

                        if self.kind_projector == 'global':
                            if self.polar == False:
                                u2_1[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], D_inv_reduced[1], D_inv_reduced[2]], u2_1)
                                column = self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
                            else:
                                rhs_12  = self.projectors_3d.P2_pol.dot(np.block([[u2_1.reshape(u2_1.shape[0]*u2_1.shape[1], u2_1.shape[2])], [u2_2.reshape(u2_2.shape[0]*u2_2.shape[1], u2_2.shape[2])]])) 
                                rhs_3   = self.projectors_3d.P3_pol.dot(u2_3.reshape(u2_3.shape[0]*u2_3.shape[1], u2_3.shape[2]))
                                rhs_12  = D_inv_z_reduced.dot(I2_pol_inv_reduced.dot(rhs_12).T).T
                                rhs_3   = N_inv_z_reduced.dot(I3_pol_inv_reduced.dot(rhs_3).T).T
                                
                                column = np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
                                
                        else:
                            column = self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
                                
                        indices1  = np.nonzero(column)[0]
                        values[0] = np.append(values[0], column[indices1])
                        row[0]    = np.append(row[0], indices1)
                        col_ind   = self.tensor_space.NbaseD[1]*self.tensor_space.NbaseD[2]*i1 + self.tensor_space.NbaseD[2]*i2 + i3
                        col[0]    = np.append(col[0], col_ind*np.ones(len(indices1), dtype=int))
            # =============================================================
            
            print('Block 1 of W_local done!')
            
            # =================== 2nd block column (W_2) =======================
            for i1 in range(self.tensor_space.Nbase_2form[1][0]):

                ind2x = np.nonzero(self.pi1_x_DD[:, :, i1])
                ind2x = np.vstack((ind2x[0], ind2x[1]))

                for i2 in range(self.tensor_space.Nbase_2form[1][1]):

                    ind2y = np.nonzero(self.pi0_y_DN[:, :, i2])
                    ind2y = np.vstack((ind2y[0], ind2y[1]))

                    for i3 in range(self.tensor_space.Nbase_2form[1][2]):

                        ind2z = np.nonzero(self.pi1_z_DD[:, :, i3])
                        ind2z = np.vstack((ind2z[0], ind2z[1]))

                        u2_1[:, :, :] = 0.
                        u2_2[:, :, :] = 0.
                        u2_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi1_x_DD[:, :, i1], self.pi0_y_DN[:, :, i2], self.pi1_z_DD[:, :, i3], ind2x, ind2y, ind2z, rho3_eq, u2_2)

                        if self.kind_projector == 'global':
                            if self.polar == False:
                                u2_2[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], N_inv_reduced[1], D_inv_reduced[2]], u2_2)
                                column = self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
                            else:
                                rhs_12  = self.projectors_3d.P2_pol.dot(np.block([[u2_1.reshape(u2_1.shape[0]*u2_1.shape[1], u2_1.shape[2])], [u2_2.reshape(u2_2.shape[0]*u2_2.shape[1], u2_2.shape[2])]])) 
                                rhs_3   = self.projectors_3d.P3_pol.dot(u2_3.reshape(u2_3.shape[0]*u2_3.shape[1], u2_3.shape[2]))
                                rhs_12  = D_inv_z_reduced.dot(I2_pol_inv_reduced.dot(rhs_12).T).T
                                rhs_3   = N_inv_z_reduced.dot(I3_pol_inv_reduced.dot(rhs_3).T).T
                                
                                column = np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
                                
                        else:
                            column = self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
                                
                        indices2  = np.nonzero(column)[0]
                        values[1] = np.append(values[1], column[indices2])
                        row[1]    = np.append(row[1], indices2)
                        col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseD[2]*i1 + self.tensor_space.NbaseD[2]*i2 + i3
                        col[1]    = np.append(col[1], col_ind*np.ones(len(indices2), dtype=int))
            # =============================================================
            
            print('Block 2 of W_local done!')
            
            # =================== 3rd block column (W_3) =======================
            for i1 in range(self.tensor_space.Nbase_2form[2][0]):

                ind3x = np.nonzero(self.pi1_x_DD[:, :, i1])
                ind3x = np.vstack((ind3x[0], ind3x[1]))

                for i2 in range(self.tensor_space.Nbase_2form[2][1]):

                    ind3y = np.nonzero(self.pi1_y_DD[:, :, i2])
                    ind3y = np.vstack((ind3y[0], ind3y[1]))

                    for i3 in range(self.tensor_space.Nbase_2form[2][2]):

                        ind3z = np.nonzero(self.pi0_z_DN[:, :, i3])
                        ind3z = np.vstack((ind3z[0], ind3z[1]))

                        u2_1[:, :, :] = 0.
                        u2_2[:, :, :] = 0.
                        u2_3[:, :, :] = 0.

                        # compute non-zero coefficients
                        ker_la.projector_tensor_strong_reduced(self.pi1_x_DD[:, :, i1], self.pi1_y_DD[:, :, i2], self.pi0_z_DN[:, :, i3], ind3x, ind3y, ind3z, rho3_eq, u2_3)

                        if self.kind_projector == 'global':
                            if self.polar == False:
                                u2_3[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], D_inv_reduced[1], N_inv_reduced[2]], u2_3)
                                column = self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
                            else:
                                rhs_12  = self.projectors_3d.P2_pol.dot(np.block([[u2_1.reshape(u2_1.shape[0]*u2_1.shape[1], u2_1.shape[2])], [u2_2.reshape(u2_2.shape[0]*u2_2.shape[1], u2_2.shape[2])]])) 
                                rhs_3   = self.projectors_3d.P3_pol.dot(u2_3.reshape(u2_3.shape[0]*u2_3.shape[1], u2_3.shape[2]))
                                rhs_12  = D_inv_z_reduced.dot(I2_pol_inv_reduced.dot(rhs_12).T).T
                                rhs_3   = N_inv_z_reduced.dot(I3_pol_inv_reduced.dot(rhs_3).T).T
                                
                                column = np.concatenate((rhs_12.flatten(), rhs_3.flatten()))
                                
                        else:
                            column = self.tensor_space.ravel_pform(u2_1, u2_2, u2_3)
                                
                        indices3  = np.nonzero(column)[0]
                        values[2] = np.append(values[2], column[indices3])
                        row[2]    = np.append(row[2], indices3)
                        col_ind   = self.tensor_space.NbaseD[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                        col[2]    = np.append(col[2], col_ind*np.ones(len(indices3), dtype=int))
            # =============================================================
            
            print('Block 3 of W_local done!')
            
            n2_pol_2D = self.tensor_space.Nbase2_pol
            n3_pol_2D = self.tensor_space.Nbase3_pol
            n_tot     = n2_pol_2D*self.tensor_space.NbaseD[2] + n3_pol_2D*self.tensor_space.NbaseN[2]
            
            W_1 = spa.csr_matrix((values[0], (row[0], col[0])), shape=(n_tot, self.tensor_space.Ntot_2form[0]))
            W_2 = spa.csr_matrix((values[1], (row[1], col[1])), shape=(n_tot, self.tensor_space.Ntot_2form[1]))
            W_3 = spa.csr_matrix((values[2], (row[2], col[2])), shape=(n_tot, self.tensor_space.Ntot_2form[2]))
            
            if self.polar == True:
                return spa.bmat([[W_1, W_2, W_3]], format='csr').dot(self.tensor_space.E2.T)
            else:
                return spa.bmat([[W_1, W_2, W_3]], format='csr')
    
    
    # =============================================
    def QSN(self, f3, f, basis='0_form', full=True):
        
        # equilibirum density/pressure/Jacobian determinant
        f3 = f3.reshape(self.tensor_space.Nbase_3form)
        
        # final coefficients of operation QSN(f)
        f2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
        f2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
        f2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
        
        if   basis == '0_form':
            f_1, f_2, f_3 = self.tensor_space.unravel_0form(f)
        
            ker_la.projector_tensor_strong(self.pi0_x_DN, self.pi1_y_DN, self.pi1_z_DN, self.pi0_x_DN_i, self.pi1_y_DN_i, self.pi1_z_DN_i, f3, f_1, f2_1)
            ker_la.projector_tensor_strong(self.pi1_x_DN, self.pi0_y_DN, self.pi1_z_DN, self.pi1_x_DN_i, self.pi0_y_DN_i, self.pi1_z_DN_i, f3, f_2, f2_2)
            ker_la.projector_tensor_strong(self.pi1_x_DN, self.pi1_y_DN, self.pi0_z_DN, self.pi1_x_DN_i, self.pi1_y_DN_i, self.pi0_z_DN_i, f3, f_3, f2_3)
            
        elif basis == '2_form':
            f_1, f_2, f_3 = self.tensor_space.unravel_2form(f)
            
        
        # solve for coefficients with interpolation/histopolation matrix in case of gobal projector
        if self.kind_projector == 'global' and full == True:
            f2_1[:, :, :], f2_2[:, :, :], f2_3[:, :, :] = self.I2(f2_1, f2_2, f2_3)
            
        return self.tensor_space.ravel_pform(f2_1, f2_2, f2_3)
    
    
    # =============================================
    def QSN_transposed(self, f3, f2, basis='0_form', full=True):
        
        # equilibirum density/pressure/Jacobian determinant (3-form)
        f3 = f3.reshape(self.tensor_space.Nbase_3form)
        
        # coefficients to contract with
        f2_1, f2_2, f2_3 = self.tensor_space.unravel_2form(f2)
        
        # final coefficients of the operation QSN^T(f2)
        if   basis == '0_form':
            f_1 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            f_2 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
            f_3 = np.zeros(self.tensor_space.Nbase_0form   , dtype=float)
        elif basis == '2_form':
            f_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
            f_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
            f_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
        
        # contract f2 with inverse 1d projection matrices: f2_abc = f2_ijk * I1^(-1)_ia * I2^(-1)_jb * I3^(-1)_kc for global pr.
        if self.kind_projector == 'global' and full == True:
            
            f2_1[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].N_inv.T, self.projectors_1d[1].D_inv.T, self.projectors_1d[2].D_inv.T], f2_1)
            f2_2[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].D_inv.T, self.projectors_1d[1].N_inv.T, self.projectors_1d[2].D_inv.T], f2_2)
            f2_3[:, :, :] = kron_la.kron_matvec_3d([self.projectors_1d[0].D_inv.T, self.projectors_1d[1].D_inv.T, self.projectors_1d[2].N_inv.T], f2_3)
        
        if   basis == '0_form':
            ker_la.projector_tensor_weak(self.pi0_x_DN, self.pi1_y_DN, self.pi1_z_DN, self.pi0_x_DN_i, self.pi1_y_DN_i, self.pi1_z_DN_i, f2_1, f3, f_1)
            ker_la.projector_tensor_weak(self.pi1_x_DN, self.pi0_y_DN, self.pi1_z_DN, self.pi1_x_DN_i, self.pi0_y_DN_i, self.pi1_z_DN_i, f2_2, f3, f_2)
            ker_la.projector_tensor_weak(self.pi1_x_DN, self.pi1_y_DN, self.pi0_z_DN, self.pi1_x_DN_i, self.pi1_y_DN_i, self.pi0_z_DN_i, f2_3, f3, f_3)
            
        elif basis == '2_form':
            print('not yet implemented!')
            
        return self.tensor_space.ravel_pform(f_1, f_2, f_3)
    
    
    
    # =======================================================
    def QSN_as_matrix(self, f3, tol, basis='0_form'):
        
        if self.kind_projector == 'global':
            # approximate inverse interpolation-/histopolation matrices in case of global projector
            N_inv_reduced = [np.copy(projector.N_inv) for projector in self.projectors_1d]
            D_inv_reduced = [np.copy(projector.D_inv) for projector in self.projectors_1d]

            for a in range(3):
                N_inv_reduced[a][np.abs(N_inv_reduced[a]) < tol] = 0.
                D_inv_reduced[a][np.abs(D_inv_reduced[a]) < tol] = 0.
        
        # equilibirum density/pressure/Jacobian determinant
        f3 = f3.reshape(self.tensor_space.Nbase_3form)
        
        # dummy coefficients
        f2_1 = np.zeros(self.tensor_space.Nbase_2form[0], dtype=float)
        f2_2 = np.zeros(self.tensor_space.Nbase_2form[1], dtype=float)
        f2_3 = np.zeros(self.tensor_space.Nbase_2form[2], dtype=float)
        
        # non-zero values and coefficients in final block matrix
        values = [np.array([], dtype=float) for i in range(3)]
        col    = [np.array([], dtype=int)   for i in range(3)]
        row    = [np.array([], dtype=int)   for i in range(3)]
        
        
        # =================== 1st column (Q_11) =======================
        for i1 in range(self.tensor_space.Nbase_0form[0]):
            
            ind11x = np.nonzero(self.pi0_x_DN[:, :, i1])
            ind11x = np.vstack((ind11x[0], ind11x[1]))
            
            for i2 in range(self.tensor_space.Nbase_0form[1]):
                
                ind11y = np.nonzero(self.pi1_y_DN[:, :, i2])
                ind11y = np.vstack((ind11y[0], ind11y[1]))
                
                for i3 in range(self.tensor_space.Nbase_0form[2]):
                    
                    ind11z = np.nonzero(self.pi1_z_DN[:, :, i3])
                    ind11z = np.vstack((ind11z[0], ind11z[1]))
                    
                    f2_1[:, :, :] = 0.
                    
                    # compute non-zero coefficients
                    ker_la.projector_tensor_strong_reduced(self.pi0_x_DN[:, :, i1], self.pi1_y_DN[:, :, i2], self.pi1_z_DN[:, :, i3], ind11x, ind11y, ind11z, f3, f2_1)
                    
                    if self.kind_projector == 'global':
                        f2_1[:, :, :] = kron_la.kron_matvec_3d([N_inv_reduced[0], D_inv_reduced[1], D_inv_reduced[2]], f2_1)
                    
                    indices11 = np.nonzero(f2_1.flatten())[0]
                    values[0] = np.append(values[0], (f2_1.flatten())[indices11])
                    row[0]    = np.append(row[0], indices11)
                    col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                    col[0]    = np.append(col[0], col_ind*np.ones(len(indices11), dtype=int))
        # =============================================================             
                    
        
        
        # =================== 1st column (Q_22) =======================
        for i1 in range(self.tensor_space.Nbase_0form[0]):
            
            ind22x = np.nonzero(self.pi1_x_DN[:, :, i1])
            ind22x = np.vstack((ind22x[0], ind22x[1]))
            
            for i2 in range(self.tensor_space.Nbase_0form[1]):
                
                ind22y = np.nonzero(self.pi0_y_DN[:, :, i2])
                ind22y = np.vstack((ind22y[0], ind22y[1]))
                
                for i3 in range(self.tensor_space.Nbase_0form[2]):
                    
                    ind22z = np.nonzero(self.pi1_z_DN[:, :, i3])
                    ind22z = np.vstack((ind22z[0], ind22z[1]))
                    
                    f2_2[:, :, :] = 0.
                    
                    # compute non-zero coefficients
                    ker_la.projector_tensor_strong_reduced(self.pi1_x_DN[:, :, i1], self.pi0_y_DN[:, :, i2], self.pi1_z_DN[:, :, i3], ind22x, ind22y, ind22z, f3, f2_2)
                    
                    if self.kind_projector == 'global':
                        f2_2[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], N_inv_reduced[1], D_inv_reduced[2]], f2_2)
                    
                    indices22 = np.nonzero(f2_2.flatten())[0]
                    values[1] = np.append(values[1], (f2_2.flatten())[indices22])
                    row[1]    = np.append(row[1], indices22)
                    col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                    col[1]    = np.append(col[1], col_ind*np.ones(len(indices22), dtype=int))
        # =============================================================
        
        
        # =================== 3rd column (Q_33) =======================
        for i1 in range(self.tensor_space.Nbase_0form[0]):
            
            ind33x = np.nonzero(self.pi1_x_DN[:, :, i1])
            ind33x = np.vstack((ind33x[0], ind33x[1]))
            
            for i2 in range(self.tensor_space.Nbase_0form[1]):
                
                ind33y = np.nonzero(self.pi1_y_DN[:, :, i2])
                ind33y = np.vstack((ind33y[0], ind33y[1]))
                
                for i3 in range(self.tensor_space.Nbase_0form[2]):
                    
                    ind33z = np.nonzero(self.pi0_z_DN[:, :, i3])
                    ind33z = np.vstack((ind33z[0], ind33z[1]))
                    
                    f2_3[:, :, :] = 0.
                    
                    # compute non-zero coefficients
                    ker_la.projector_tensor_strong_reduced(self.pi1_x_DN[:, :, i1], self.pi1_y_DN[:, :, i2], self.pi0_z_DN[:, :, i3], ind33x, ind33y, ind33z, f3, f2_3)
                    
                    if self.kind_projector == 'global':
                        f2_3[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], D_inv_reduced[1], N_inv_reduced[2]], f2_3)
                    
                    indices33 = np.nonzero(f2_3.flatten())[0]
                    values[2] = np.append(values[2], (f2_3.flatten())[indices33])
                    row[2]    = np.append(row[2], indices33)
                    col_ind   = self.tensor_space.NbaseN[1]*self.tensor_space.NbaseN[2]*i1 + self.tensor_space.NbaseN[2]*i2 + i3
                    col[2]    = np.append(col[2], col_ind*np.ones(len(indices33), dtype=int))
        # =============================================================           
        
        
        Q_11 = spa.csr_matrix((values[0], (row[0], col[0])), shape=(self.tensor_space.Ntot_2form[0], self.tensor_space.Ntot_0form))
        Q_22 = spa.csr_matrix((values[1], (row[1], col[1])), shape=(self.tensor_space.Ntot_2form[1], self.tensor_space.Ntot_0form))
        Q_33 = spa.csr_matrix((values[2], (row[2], col[2])), shape=(self.tensor_space.Ntot_2form[2], self.tensor_space.Ntot_0form))
            
        return spa.bmat([[Q_11, None, None], [None, Q_22, None], [None, None, Q_33]], format='csr')
    
    
    
    # ===============================================
    def K(self, f0, f3, full=True):
        
        # equilibrium density (0-form)
        f0 = f0.reshape(self.tensor_space.Nbase_0form)
        
        # final coefficients of operation K(f3) (3-form)
        g3 = np.zeros(self.tensor_space.Nbase_3form, dtype=float)
        
        
        f3 = f3.reshape(self.tensor_space.Nbase_3form)
        
        ker_la.projector_tensor_strong(self.pi1_x_ND, self.pi1_y_ND, self.pi1_z_ND, self.pi1_x_ND_i, self.pi1_y_ND_i, self.pi1_z_ND_i, f0, f3, g3)
        
        if self.kind_projector == 'global' and full == True:
            g3[:, :, :] = self.I3(g3)
            
        return g3.flatten()
    
    
    
    # ===============================================
    def K_as_matrix(self, f0, tol):
        
        if self.kind_projector == 'global':
            # approximate inverse interpolation-/histopolation matrices in case of global projector
            N_inv_reduced = [np.copy(projector.N_inv) for projector in self.projectors_1d]
            D_inv_reduced = [np.copy(projector.D_inv) for projector in self.projectors_1d]

            for a in range(3):
                N_inv_reduced[a][np.abs(N_inv_reduced[a]) < tol] = 0.
                D_inv_reduced[a][np.abs(D_inv_reduced[a]) < tol] = 0.
        
        # equilibrium density (0-form)
        f0 = f0.reshape(self.tensor_space.Nbase_0form)
        
        # dummy coefficients
        f3 = np.zeros(self.tensor_space.Nbase_3form, dtype=float)
        
        # non-zero values and coefficients in final block matrix
        values = np.array([], dtype=float)
        col    = np.array([], dtype=int)
        row    = np.array([], dtype=int)
        
        
        for i1 in range(self.tensor_space.Nbase_3form[0]):
            
            indx = np.nonzero(self.pi1_x_ND[:, :, i1])
            indx = np.vstack((indx[0], indx[1]))
            
            for i2 in range(self.tensor_space.Nbase_3form[1]):
                
                indy = np.nonzero(self.pi1_y_ND[:, :, i2])
                indy = np.vstack((indy[0], indy[1]))
                
                for i3 in range(self.tensor_space.Nbase_3form[2]):
                    
                    indz = np.nonzero(self.pi1_z_ND[:, :, i3])
                    indz = np.vstack((indz[0], indz[1]))
                    
                    f3[:, :, :] = 0.
                    
                    # compute non-zero coefficients
                    ker_la.projector_tensor_strong_reduced(self.pi1_x_ND[:, :, i1], self.pi1_y_ND[:, :, i2], self.pi1_z_ND[:, :, i3], indx, indy, indz, f0, f3) 
                    
                    if self.kind_projector == 'global':
                        f3[:, :, :] = kron_la.kron_matvec_3d([D_inv_reduced[0], D_inv_reduced[1], D_inv_reduced[2]], f3)
                    
                    indices = np.nonzero(f3.flatten())[0]
                    values  = np.append(values, (f3.flatten())[indices])
                    row     = np.append(row, indices)
                    col_ind = self.tensor_space.NbaseD[1]*self.tensor_space.NbaseD[2]*i1 + self.tensor_space.NbaseD[2]*i2 + i3
                    col     = np.append(col, col_ind*np.ones(len(indices), dtype=int))
        
            
        return spa.csr_matrix((values, (row, col)), shape=(self.tensor_space.Ntot_3form, self.tensor_space.Ntot_3form))