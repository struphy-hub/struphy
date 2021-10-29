# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

from mpi4py import MPI

import numpy        as np
import scipy.sparse as spa

# TODO: move control variate file to new location
import struphy.pic.control_variate as cv
import struphy.feec.control_variates.control_variate as cv

import time
import struphy.pic.accumulation_kernels as pic_ker_3d



class accumulation:
    """
    Class for computing charge and current densities from particles.
    
    Parameters
    ---------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space
        
    domain : domain object
        domain object from hylife.geometry.domain_3d defining the mapping
        
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
        
        # intialize delta-f correction terms
        if self.control == True and self.mpi_rank == 0:
            self.cont = cv.terms_control_variate(self.space, self.basis_u, eq_pic)
        
        # reserve memory for implicit particle-coupling sub-steps
        self.blocks_loc = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.vecs_loc   =  [0, 0, 0]
        
        self.blocks     = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.vecs       =  [0, 0, 0]
            
        for a in range(3):
            
            if self.basis_u == 0:
                Ni = self.space.Nbase_0form

            elif self.basis_u == 1:
                Ni = self.space.Nbase_1form[a]

            elif self.basis_u == 2:
                Ni = self.space.Nbase_2form[a]
            
            self.vecs_loc[a] = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)
            
            if self.mpi_rank == 0:
                self.vecs[a] = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)
            
            for b in range(a, 3):
                
                if self.space.dim == 2:
                    
                    self.blocks_loc[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2), dtype=float)

                    if self.mpi_rank == 0:
                        self.blocks[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2), dtype=float)
                    
                else:
                
                    self.blocks_loc[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)

                    if self.mpi_rank == 0:
                        self.blocks[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)
        
    
    # ===============================================================
    def to_sparse_step1(self):
        """
        Converts the 6d arrays stored in self.blocks to a sparse block matrix using row-major ordering
        
        Returns
        -------
        M : sparse matrix in csr-format
            anti-symmetric, sparse block matrix [[0, M12, M13], [-M12.T, 0, M23], [-M13.T, -M23.T, 0]]
        """
        
        # blocks of global matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        for a in range(2):
            for b in range(a + 1, 3):
                
                if self.basis_u == 0:
                    Ni = self.space.Nbase_0form
                    Nj = self.space.Nbase_0form
                    
                elif self.basis_u == 1:
                    Ni = self.space.Nbase_1form[a]
                    Nj = self.space.Nbase_1form[b]
                    
                elif self.basis_u == 2:
                    Ni = self.space.Nbase_2form[a]
                    Nj = self.space.Nbase_2form[b]
                
                indices = np.indices(self.blocks[a][b].shape)
                
                row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
            
                shift   = [np.arange(Ni) - p for Ni, p in zip(Ni[:2], self.space.p[:2])]

                if self.space.dim == 2:
                    shift += [np.array([0, 0])]
                else:
                    shift += [np.arange(Ni[2]) - self.space.p[2]]

                col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
                col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
                col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]

                col     = Nj[1]*Nj[2]*col1 + Nj[2]*col2 + col3
                
                M[a][b] = spa.csr_matrix((self.blocks[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
                M[a][b].eliminate_zeros()
        
        
        # final block matrix
        M = spa.bmat([[None, M[0][1], M[0][2]], [-M[0][1].T, None, M[1][2]], [-M[0][2].T, -M[1][2].T, None]], format='csr')
        
        ## apply extraction operator
        #if self.basis_u == 0:
        #    M = self.space.Ev.dot(M.dot(self.space.Ev.T)).tocsr()
        #    
        #elif self.basis_u == 1:
        #    M = self.space.E1.dot(M.dot(self.space.E1.T)).tocsr()
        #        
        #elif self.basis_u == 2:
        #    M = self.space.E2.dot(M.dot(self.space.E2.T)).tocsr()
        
        return M
    
    
    # ===============================================================
    def to_sparse_step3(self):
        """
        Converts the 6d arrays stored in self.blocks to a sparse block matrix using row-major ordering
        
        Returns
        -------
        M : sparse matrix in csr-format
            symmetric, sparse block matrix [[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]]
        """
        
        # blocks of global matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        for a in range(3):
            for b in range(a, 3):
                
                if self.basis_u == 0:
                    Ni = self.space.Nbase_0form
                    Nj = self.space.Nbase_0form
                    
                elif self.basis_u == 1:
                    Ni = self.space.Nbase_1form[a]
                    Nj = self.space.Nbase_1form[b]
                    
                elif self.basis_u == 2:
                    Ni = self.space.Nbase_2form[a]
                    Nj = self.space.Nbase_2form[b]
                
                indices = np.indices(self.blocks[a][b].shape)
                
                row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
                
                shift   = [np.arange(Ni) - p for Ni, p in zip(Ni[:2], self.space.p[:2])]
            
                if self.space.dim == 2:
                    shift += [np.array([0, 0])]
                else:
                    shift += [np.arange(Ni[2]) - self.space.p[2]]

                col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
                col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
                col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]

                col     = Nj[1]*Nj[2]*col1 + Nj[2]*col2 + col3
                
                M[a][b] = spa.csr_matrix((self.blocks[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
                M[a][b].eliminate_zeros()
        
        
        # final block matrix
        M = spa.bmat([[M[0][0], M[0][1], M[0][2]], [M[0][1].T, M[1][1], M[1][2]], [M[0][2].T, M[1][2].T, M[2][2]]], format='csr')
        
        ## apply extraction operator
        #if self.basis_u == 0:
        #    M = self.space.Ev.dot(M.dot(self.space.Ev.T)).tocsr()
        #    
        #elif self.basis_u == 1:
        #    M = self.space.E1.dot(M.dot(self.space.E1.T)).tocsr()
        #        
        #elif self.basis_u == 2:
        #    M = self.space.E2.dot(M.dot(self.space.E2.T)).tocsr()
        
        return M
    
    
    # ===============================================================
    def accumulate_step1(self, particles_loc, b2_eq, b2, mpi_comm):
        
        b2_1, b2_2, b2_3 = self.space.extract_2form(b2)
        
        if self.space.dim == 2:
            
            print('not yet implemented')
            
        else:
            
            pic_ker_3d.kernel_step1(particles_loc, self.space.T[0], self.space.T[1], self.space.T[2], self.space.p, self.space.Nel, self.space.NbaseN, self.space.NbaseD, particles_loc.shape[1], b2_eq[0] + b2_1, b2_eq[1] + b2_2, b2_eq[2] + b2_3, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz, self.blocks_loc[0][1], self.blocks_loc[0][2], self.blocks_loc[1][2], self.basis_u)
        
        mpi_comm.Reduce(self.blocks_loc[0][1], self.blocks[0][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[0][2], self.blocks[0][2], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][2], self.blocks[1][2], op=MPI.SUM, root=0)
        
        
    # ===============================================================
    def accumulate_step3(self, particles_loc, b2_eq, b2, mpi_comm):
        
        b2_1, b2_2, b2_3 = self.space.extract_2form(b2)
        
        if self.space.dim == 2:
            
            print('not yet implemented')
               
        else:
        
            pic_ker_3d.kernel_step3(particles_loc, self.space.T[0], self.space.T[1], self.space.T[2], self.space.p, self.space.Nel, self.space.NbaseN, self.space.NbaseD, particles_loc.shape[1], b2_eq[0] + b2_1, b2_eq[1] + b2_2, b2_eq[2] + b2_3, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz, self.blocks_loc[0][0], self.blocks_loc[0][1], self.blocks_loc[0][2], self.blocks_loc[1][1], self.blocks_loc[1][2], self.blocks_loc[2][2], self.vecs_loc[0], self.vecs_loc[1], self.vecs_loc[2], self.basis_u)
        
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
    def assemble_step1(self, Np, b2_eq, b2):
        
        self.blocks[0][1] /= Np
        self.blocks[0][2] /= Np
        self.blocks[1][2] /= Np
        
        # delta-f correction
        if self.control:
            
            b2_1, b2_2, b2_3 = self.space.extract_2form(b2)
            
            if self.space.dim == 2:
                self.cont.correct_step1(b2_eq[0], b2_eq[1], b2_eq[2])
            else:
                self.cont.correct_step1(b2_eq[0] + b2_1, b2_eq[1] + b2_2, b2_eq[2] + b2_3)
            
            self.blocks[0][1] += self.cont.M12
            self.blocks[0][2] += self.cont.M13
            self.blocks[1][2] += self.cont.M23
            
        # build global sparse matrix
        return self.to_sparse_step1()
        
    
        
    # ===============================================================
    def assemble_step3(self, Np, b2_eq, b2):
        
        self.blocks[0][0] /= Np
        self.blocks[0][1] /= Np
        self.blocks[0][2] /= Np
        self.blocks[1][1] /= Np
        self.blocks[1][2] /= Np
        self.blocks[2][2] /= Np
        
        self.vecs[0] /= Np
        self.vecs[1] /= Np
        self.vecs[2] /= Np
        
        # delta-f correction
        if self.control:
            
            b2_1, b2_2, b2_3 = self.space.extract_2form(b2)
        
            if self.space.dim == 2:
                self.cont.correct_step3(b2_1, b2_2, b2_3)
            else:
                self.cont.correct_step3(b2_eq[0] + b2_1, b2_eq[1] + b2_2, b2_eq[2] + b2_3)
            
            self.vecs[0] += self.cont.F[0]
            self.vecs[1] += self.cont.F[1]
            self.vecs[2] += self.cont.F[2]
            
        # build global sparse matrix and global vector
        if self.basis_u == 0:
            return self.to_sparse_step3(), np.concatenate((self.vecs[0].flatten(), self.vecs[1].flatten(), self.vecs[2].flatten()))
        elif self.basis_u == 1:
            return self.to_sparse_step3(), np.concatenate((self.vecs[0].flatten(), self.vecs[1].flatten(), self.vecs[2].flatten()))
        elif self.basis_u == 2:
            return self.to_sparse_step3(), np.concatenate((self.vecs[0].flatten(), self.vecs[1].flatten(), self.vecs[2].flatten()))