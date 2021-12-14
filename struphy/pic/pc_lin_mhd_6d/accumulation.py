# coding: utf-8

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

from mpi4py import MPI

import numpy        as np
import scipy.sparse as spa

import struphy.pic.pc_lin_mhd_6d.accumulation_kernels_3d as pic_ker_3d

import time

class Accumulation:
    """
    Class for computing pressure tensor from particles.
    
    Parameters
    ---------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space
        
    domain : domain object
        domain object from struphy.geometry.domain_3d defining the mapping
        
    basis_u : int
        bulk velocity representation (0 : vector-field, 1 : 1-form , 2 : 2-form)
        
    mpi_comm : MPI.COMM_WORLD
        MPI communicator
        
    control : boolean
        whether a full-f (False) of delta-f approach is used
        
    eq_pic : kinetic equilibriumm object
        the equilibrium distribution function (only necessary in case of control = True)
    """
        
    # ===============================================================
    def __init__(self, tensor_space_FEM, domain, basis_u, mpi_comm, control, eq_pic=None):
        
        self.space    = tensor_space_FEM
        self.domain   = domain
        self.basis_u  = basis_u                   
        self.mpi_rank = mpi_comm.Get_rank()
        self.control  = control

        # reserve memory for implicit particle-coupling sub-steps
        self.blocks_loc  = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.vecs_loc    =  [0, 0, 0]
        
        self.blocks      = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.vecs        =  [0, 0, 0]

        self.mat_full = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.vec_full   = [0, 0, 0]
            
        for a in range(3):

            if self.basis_u == 1:
                Ni = self.space.Nbase_1form[a]

            elif self.basis_u == 2:
                Ni = self.space.Nbase_2form[a]
            
            self.vecs_loc[a] = np.empty((Ni[0], Ni[1], Ni[2], 3), dtype=float)
            
            if self.mpi_rank == 0:
                self.vecs[a] = np.empty((Ni[0], Ni[1], Ni[2], 3), dtype=float)
            
            for b in range(a, 3):
                
                if self.space.dim == 2:
                    
                    self.blocks_loc[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, self.space.NbaseN[2], 3, 3), dtype=float)

                    if self.mpi_rank == 0:
                        self.blocks[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, self.space.NbaseN[2], 3, 3), dtype=float)
                    
                else:
                
                    self.blocks_loc[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1, 3, 3), dtype=float)

                    if self.mpi_rank == 0:
                        self.blocks[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1, 3, 3), dtype=float)

    
    # ===============================================================
    def to_sparse_step_ph(self, vp, vq):
        """
        Converts the 6d arrays stored in self.blocks to a sparse block matrix using row-major ordering.
        
        Returns
        -------
        M : sparse matrix in csr-format
            symmetric, sparse block matrix [[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]]
        """
        
        # blocks of global matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        for a in range(3):
            for b in range(a, 3):

                if self.basis_u == 1:
                    Ni = self.space.Nbase_1form[a]
                    Nj = self.space.Nbase_1form[b]
                    
                elif self.basis_u == 2:
                    Ni = self.space.Nbase_2form[a]
                    Nj = self.space.Nbase_2form[b]
                
                indices = np.indices(self.blocks[a][b][:,:,:,:,:,:,vp,vq].shape)
                
                row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
                
                shift   = [np.arange(Ni) - p for Ni, p in zip(Ni[:2], self.space.p[:2])]
            
                if self.space.dim == 2:
                    shift += [np.zeros(self.space.NbaseN[2], dtype=int)]
                else:
                    shift += [np.arange(Ni[2]) - self.space.p[2]]

                col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
                col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
                col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]

                col     = Nj[1]*Nj[2]*col1 + Nj[2]*col2 + col3
                
                M[a][b] = spa.csr_matrix((self.blocks[a][b][:,:,:,:,:,:,vp,vq].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
                M[a][b].eliminate_zeros()
        
        
        # final block matrix
        M = spa.bmat([[M[0][0], M[0][1], M[0][2]], [M[0][1].T, M[1][1], M[1][2]], [M[0][2].T, M[1][2].T, M[2][2]]], format='csr')
        
        # apply extraction operator
        if self.basis_u == 0:
            M = self.space.Ev_0.dot(M.dot(self.space.Ev_0.T)).tocsr()
            
        elif self.basis_u == 1:
            M = self.space.E1_0.dot(M.dot(self.space.E1_0.T)).tocsr()
                
        elif self.basis_u == 2:
            M = self.space.E2_0.dot(M.dot(self.space.E2_0.T)).tocsr()
        
        return M
    

    # ===============================================================
    def accumulate_step_ph_full(self, particles_loc, mpi_comm):
        '''TODO
        '''

        pic_ker_3d.kernel_step_ph_full(particles_loc,  self.basis_u, self.space.T[0], self.space.T[1], self.space.T[2], self.space.p, self.space.Nel, self.space.NbaseN, self.space.NbaseD, particles_loc.shape[1], self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz, self.blocks_loc[0][0], self.blocks_loc[0][1], self.blocks_loc[0][2], self.blocks_loc[1][1], self.blocks_loc[1][2], self.blocks_loc[2][2], self.vecs_loc[0], self.vecs_loc[1], self.vecs_loc[2])
        
        mpi_comm.Reduce(self.blocks_loc[0][0], self.blocks[0][0], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[0][1], self.blocks[0][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[0][2], self.blocks[0][2], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][1], self.blocks[1][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][2], self.blocks[1][2], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[2][2], self.blocks[2][2], op=MPI.SUM, root=0)

        mpi_comm.Reduce(self.vecs_loc[0] , self.vecs[0] , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vecs_loc[1] , self.vecs[1] , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vecs_loc[2] , self.vecs[2] , op=MPI.SUM, root=0)
       

    # ===============================================================
    def assemble_step_ph_full(self, Np, nuh, alpha, charge_over_mass):
        '''TODO
        '''
        
        self.blocks[0][0] /= Np
        self.blocks[0][1] /= Np
        self.blocks[0][2] /= Np
        self.blocks[1][1] /= Np
        self.blocks[1][2] /= Np
        self.blocks[2][2] /= Np
        
        self.vecs[0] /= Np
        self.vecs[1] /= Np
        self.vecs[2] /= Np

        # build global sparse matrix and global vector
        for vp in range(3):
            for vq in range(3):
                self.mat_full[vp][vq] = self.to_sparse_step_ph(vp, vq) * nuh * alpha * charge_over_mass

        for vp in range(3):
            if self.basis_u == 1:
                self.vec_full[vp] = self.space.E1_0.dot(np.concatenate((self.vecs[0][:,:,:,vp].flatten(), self.vecs[1][:,:,:,vp].flatten(), self.vecs[2][:,:,:,vp].flatten()))) * nuh * alpha * charge_over_mass
        
            elif self.basis_u == 2:
                self.vec_full[vp] = self.space.E2_0.dot(np.concatenate((self.vecs[0][:,:,:,vp].flatten(), self.vecs[1][:,:,:,vp].flatten(), self.vecs[2][:,:,:,vp].flatten()))) * nuh * alpha * charge_over_mass


    # ===============================================================
    def assemble_mat_X_step_ph_full(self, MHD, x):
        
        if self.basis_u == 1:
            X_dot = MHD.X1_dot(x)

            RHS_temp_mat1 = self.space.G.T.dot(self.mat_full[0][0].dot(self.space.G.dot(X_dot[0])) + self.mat_full[0][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[0][2].dot(self.space.G.dot(X_dot[2])))
            RHS_temp_mat2 = self.space.G.T.dot(self.mat_full[1][0].dot(self.space.G.dot(X_dot[0])) + self.mat_full[1][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[1][2].dot(self.space.G.dot(X_dot[2])))
            RHS_temp_mat3 = self.space.G.T.dot(self.mat_full[2][0].dot(self.space.G.dot(X_dot[0])) + self.mat_full[2][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[2][2].dot(self.space.G.dot(X_dot[2])))

            RHS = MHD.transpose_X1_dot((RHS_temp_mat1, RHS_temp_mat2, RHS_temp_mat3))

        elif self.basis_u == 2:
            X_dot = MHD.X2_dot(x)

            RHS_temp_mat1 = self.space.G.T.dot(self.mat_full[0][0].dot(self.space.G.dot(X_dot[0])) + self.mat_full[0][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[0][2].dot(self.space.G.dot(X_dot[2])))
            RHS_temp_mat2 = self.space.G.T.dot(self.mat_full[1][0].dot(self.space.G.dot(X_dot[0])) + self.mat_full[1][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[1][2].dot(self.space.G.dot(X_dot[2])))
            RHS_temp_mat3 = self.space.G.T.dot(self.mat_full[2][0].dot(self.space.G.dot(X_dot[0])) + self.mat_full[2][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[2][2].dot(self.space.G.dot(X_dot[2])))

            RHS = MHD.transpose_X2_dot((RHS_temp_mat1, RHS_temp_mat2, RHS_temp_mat3))

        return RHS


    # ===============================================================
    def assemble_vec_X_step_ph_full(self, MHD):

        RHS_temp_vec1 = self.space.G.T.dot(self.vec_full[0])
        RHS_temp_vec2 = self.space.G.T.dot(self.vec_full[1])
        RHS_temp_vec3 = self.space.G.T.dot(self.vec_full[2])

        if self.basis_u == 1:
            RHS = MHD.transpose_X1_dot((RHS_temp_vec1, RHS_temp_vec2, RHS_temp_vec3))

        elif self.basis_u == 2:
            RHS = MHD.transpose_X2_dot((RHS_temp_vec1, RHS_temp_vec2, RHS_temp_vec3))

        return RHS


    # ===============================================================
    def accumulate_step_ph_perp(self, particles_loc, mpi_comm):
        '''TODO
        '''

        pic_ker_3d.kernel_step_ph_perp(particles_loc, self.basis_u, self.space.T[0], self.space.T[1], self.space.T[2], self.space.p, self.space.Nel, self.space.NbaseN, self.space.NbaseD, particles_loc.shape[1], self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz, self.blocks_loc[0][0], self.blocks_loc[0][1], self.blocks_loc[0][2], self.blocks_loc[1][1], self.blocks_loc[1][2], self.blocks_loc[2][2], self.vecs_loc[0], self.vecs_loc[1], self.vecs_loc[2])

        mpi_comm.Reduce(self.blocks_loc[0][0], self.blocks[0][0], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[0][1], self.blocks[0][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[0][2], self.blocks[0][2], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][1], self.blocks[1][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][2], self.blocks[1][2], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[2][2], self.blocks[2][2], op=MPI.SUM, root=0)

        mpi_comm.Reduce(self.vecs_loc[0] , self.vecs[0] , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vecs_loc[1] , self.vecs[1] , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vecs_loc[2] , self.vecs[2] , op=MPI.SUM, root=0)
       

    # ===============================================================
    def assemble_step_ph_perp(self, Np, nuh, alpha, charge_over_mass):
        '''TODO
        '''
        
        self.blocks[0][0] /= Np
        self.blocks[0][1] /= Np
        self.blocks[0][2] /= Np
        self.blocks[1][1] /= Np
        self.blocks[1][2] /= Np
        self.blocks[2][2] /= Np
        
        self.vecs[0] /= Np
        self.vecs[1] /= Np
        self.vecs[2] /= Np

        # build global sparse matrix and global vector
        for vp in [1,2]:
            for vq in [1,2]:
                self.mat_full[vp][vq] = self.to_sparse_step_ph(vp, vq) * nuh * alpha * charge_over_mass

        for vp in [1,2]:
            if self.basis_u == 1:
                self.vec_full[vp] = self.space.E1_0.dot(np.concatenate((self.vecs[0][:,:,:,vp].flatten(), self.vecs[1][:,:,:,vp].flatten(), self.vecs[2][:,:,:,vp].flatten()))) * nuh * alpha * charge_over_mass
        
            elif self.basis_u == 2:
                self.vec_full[vp] = self.space.E2_0.dot(np.concatenate((self.vecs[0][:,:,:,vp].flatten(), self.vecs[1][:,:,:,vp].flatten(), self.vecs[2][:,:,:,vp].flatten()))) * nuh * alpha * charge_over_mass


    # ===============================================================
    def assemble_mat_X_step_ph_perp(self, MHD, x):
        
        if self.basis_u == 1:
            X_dot = MHD.X1_dot(x)

            RHS_temp_mat1 = np.zeros(self.space.Ntot_1form_cum[0])
            RHS_temp_mat2 = self.space.G.T.dot(self.mat_full[1][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[1][2].dot(self.space.G.dot(X_dot[2])))
            RHS_temp_mat3 = self.space.G.T.dot(self.mat_full[2][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[2][2].dot(self.space.G.dot(X_dot[2])))

            RHS = MHD.transpose_X1_dot((RHS_temp_mat1, RHS_temp_mat2, RHS_temp_mat3))

            
            up_ten_1, up_ten_2, up_ten_3 = self.space.extract_1(RHS)
            up_old_ten_1, up_old_ten_2, up_old_ten_3 = self.tensor_space_FEM.extract_1(x)
            RHS[:] = np.concatenate((up_old_ten_1.flatten(), up_ten_2.flatten(), up_ten_3.flatten()))  

        elif self.basis_u == 2:
            X_dot = MHD.X2_dot(x)

            RHS_temp_mat1 = np.zeros(self.space.Ntot_2form_cum[0])
            RHS_temp_mat2 = self.space.G.T.dot(self.mat_full[1][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[1][2].dot(self.space.G.dot(X_dot[2])))
            RHS_temp_mat3 = self.space.G.T.dot(self.mat_full[2][1].dot(self.space.G.dot(X_dot[1])) + self.mat_full[2][2].dot(self.space.G.dot(X_dot[2])))

            RHS = MHD.transpose_X2_dot((RHS_temp_mat1, RHS_temp_mat2, RHS_temp_mat3))

            up_ten_1, up_ten_2, up_ten_3 = self.space.extract_2(RHS)
            up_old_ten_1, up_old_ten_2, up_old_ten_3 = self.space.extract_2(x)
            RHS[:] = np.concatenate((up_old_ten_1.flatten(), up_ten_2.flatten(), up_ten_3.flatten()))  

        return RHS


    # ===============================================================
    def assemble_vec_X_step_ph_perp(self, MHD):

        if self.basis_u == 1:
            
            RHS_temp_vec1 = np.zeros(self.space.Ntot_1form_cum[0]) 
            RHS_temp_vec2 = self.space.G.T.dot(self.vec_full[1])
            RHS_temp_vec3 = self.space.G.T.dot(self.vec_full[2])

            RHS = MHD.transpose_X1_dot((RHS_temp_vec1, RHS_temp_vec2, RHS_temp_vec3))

        elif self.basis_u == 2:
            
            RHS_temp_vec1 = np.zeros(self.space.Ntot_2form_cum[0])
            RHS_temp_vec2 = self.space.G.T.dot(self.vec_full[1])
            RHS_temp_vec3 = self.space.G.T.dot(self.vec_full[2])

            RHS = MHD.transpose_X2_dot((RHS_temp_vec1, RHS_temp_vec2, RHS_temp_vec3))

        return RHS