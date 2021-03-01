# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

from mpi4py import MPI

import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_PIC.accumulation_kernels as pic_ker

import hylife.utilitis_FEEC.control_variates.control_variate as cv

class accumulation:
    """
    Class for computing charge and current densities from particles.
    
    Parameters
    ---------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space
        
    basis_u : int
        bulk velocity representation (0 : 0-form, 1 : 1-form , 2 : 2-form)
    """
        
    # ===============================================================
    def __init__(self, tensor_space_FEM, domain, basis_u, mpi_comm, control):
        
        self.domain   = domain
        self.basis_u  = basis_u                   
        self.mpi_rank = mpi_comm.Get_rank()
        self.control  = control
        
        self.tensor_space_FEM = tensor_space_FEM
        
        self.T        = tensor_space_FEM.T        # knot vectors
        self.p        = tensor_space_FEM.p        # spline degrees
        self.bc       = tensor_space_FEM.bc       # boundary conditions (True : periodic, False : clamped)
        
        self.t        = tensor_space_FEM.t        # reduced knot vectors
        self.el_b     = tensor_space_FEM.el_b     # element boundaries
        self.Nel      = tensor_space_FEM.Nel      # number of elements
        self.NbaseN   = tensor_space_FEM.NbaseN   # number of basis functions (N)
        self.NbaseD   = tensor_space_FEM.NbaseD   # number of basis functions (D)
        
        
        # ==== intialize delta-f correction terms ===============================
        if self.control == True and self.mpi_rank == 0:
            self.cont = cv.terms_control_variate(self.tensor_space_FEM, self.domain, self.basis_u)
        

        # ==== reserve memory for implicit particle-coupling sub-steps ==========
        if self.basis_u == 0:
            self.mat11_loc = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat12_loc = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat13_loc = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat22_loc = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat23_loc = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat33_loc = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)

            self.vec1_loc  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.vec2_loc  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.vec3_loc  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)

            if self.mpi_rank == 0:
                self.mat11 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat12 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat13 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat22 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat23 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)

                self.vec1  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
                self.vec2  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
                self.vec3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)

            else:
                self.mat11, self.mat12, self.mat13, self.mat22, self.mat23, self.mat33 = None, None, None, None, None, None
                self.vec1,  self.vec2,  self.vec3 = None, None, None

        elif self.basis_u == 2:
            self.mat11_loc = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat12_loc = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat13_loc = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat22_loc = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat23_loc = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat33_loc = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)

            self.vec1_loc  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]), dtype=float)
            self.vec2_loc  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)
            self.vec3_loc  = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)

            if self.mpi_rank == 0:
                self.mat11 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat12 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat13 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat22 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat23 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat33 = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)

                self.vec1  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]), dtype=float)
                self.vec2  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)
                self.vec3  = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)

            else:
                self.mat11, self.mat12, self.mat13, self.mat22, self.mat23, self.mat33 = None, None, None, None, None, None
                self.vec1,  self.vec2,  self.vec3 = None, None, None
        # =======================================================================
        
        
    
    # ===============================================================
    def to_sparse_step1(self, mat12, mat13, mat23):
        """
        Converts the 6d arrays mat12, mat13, mat23 to a sparse 2d block matrix using row-major ordering
        
        Paramters
        ---------
        mat12 : array_like
            12 - block in final matrix (6d array)
            
        mat13 : array_like
            13 - block in final matrix (6d array)
        
        mat23 : array_like
            23 - block in final matrix (6d array)
        
        Returns
        -------
        M : sparse matrix in csr-format
            2d anti-symmetric, sparse block matrix [[0, M12, mat13], [-M12.T, 0, M23], [-M13.T, -M23.T, 0]]
        """
        
        
        # conversion to sparse matrix if all components of U live in V0
        if self.basis_u == 0:
            indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1 = np.arange(self.NbaseN[0]) - self.p[0]
            shift2 = np.arange(self.NbaseN[1]) - self.p[1]
            shift3 = np.arange(self.NbaseN[2]) - self.p[2]

            row    = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1   = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2   = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3   = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col    = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M12 = spa.csr_matrix((mat12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M12.eliminate_zeros()

            M13 = spa.csr_matrix((mat13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M13.eliminate_zeros()

            M23 = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M23.eliminate_zeros()
        
        # conversion to sparse matrix if U lives in V2
        elif self.basis_u == 2:
            
            # conversion to sparse matrix (12 -block)
            indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

            col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3

            M12   = spa.csr_matrix((mat12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2]))
            M12.eliminate_zeros()


            # conversion to sparse matrix (13 -block)
            indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M13   = spa.csr_matrix((mat13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))
            M13.eliminate_zeros()


            # conversion to sparse matrix (23 -block)
            indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseD[0]) - self.p[0]
            shift2  = np.arange(self.NbaseN[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M23   = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))
            M23.eliminate_zeros()
        
        
        # final block matrix
        M = spa.bmat([[None, M12, M13], [-M12.T, None, M23], [-M13.T, -M23.T, None]], format='csr')
        
        return M
    
    
    # ===============================================================
    def to_sparse_step3(self, mat11, mat12, mat13, mat22, mat23, mat33):
        """
        Converts the 6d arrays mat11, mat12, mat13, mat22, mat23, mat33 to a sparse 2d block matrix using row-major ordering
        
        Paramters
        ---------
        mat11 : array_like
            11 - block in final matrix (6d array)
            
        mat12 : array_like
            12 - block in final matrix (6d array)
        
        mat13 : array_like
            13 - block in final matrix (6d array)
            
        mat22 : array_like
            22 - block in final matrix (6d array)
            
        mat23 : array_like
            23 - block in final matrix (6d array)
        
        mat33 : array_like
            33 - block in final matrix (6d array)
        
        Returns
        -------
        M : sparse matrix in csr-format
            2d symmetric, sparse block matrix [[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]]
        """
        
        # conversion to sparse matrix if all components of U live in V0
        if self.basis_u == 0:
            indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1 = np.arange(self.NbaseN[0]) - self.p[0]
            shift2 = np.arange(self.NbaseN[1]) - self.p[1]
            shift3 = np.arange(self.NbaseN[2]) - self.p[2]

            row    = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1   = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2   = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3   = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col    = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M11 = spa.csr_matrix((mat11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M11.eliminate_zeros()

            M12 = spa.csr_matrix((mat12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M12.eliminate_zeros()

            M13 = spa.csr_matrix((mat13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M13.eliminate_zeros()

            M22 = spa.csr_matrix((mat22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M22.eliminate_zeros()

            M23 = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M23.eliminate_zeros()

            M33 = spa.csr_matrix((mat33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseN[2]))
            M33.eliminate_zeros()
        
        
        # conversion to sparse matrix if U lives in V2
        elif self.basis_u == 2:
            # conversion to sparse matrix (11-block)
            indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

            col     = self.NbaseD[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3

            M11   = spa.csr_matrix((mat11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2]))
            M11.eliminate_zeros()


            # conversion to sparse matrix (12-block)
            indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

            col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3

            M12   = spa.csr_matrix((mat12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2]))
            M12.eliminate_zeros()


            # conversion to sparse matrix (13-block)
            indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M13   = spa.csr_matrix((mat13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))
            M13.eliminate_zeros()


            # conversion to sparse matrix (22-block)
            indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseD[0]) - self.p[0]
            shift2  = np.arange(self.NbaseN[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

            col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3

            M22   = spa.csr_matrix((mat22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2]))
            M22.eliminate_zeros()


            # conversion to sparse matrix (23-block)
            indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseD[0]) - self.p[0]
            shift2  = np.arange(self.NbaseN[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M23   = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))
            M23.eliminate_zeros()


            # conversion to sparse matrix (33-block)
            indices = np.indices((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseD[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseN[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M33   = spa.csr_matrix((mat33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2]))
            M33.eliminate_zeros()
                         
        
        # final block matrix
        M = spa.bmat([[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]], format='csr')
        
        return M
    
    
    # ===============================================================
    def accumulate_step1(self, particles_loc, b2, mpi_comm):
        
        b2_1, b2_2, b2_3 = self.tensor_space_FEM.unravel_2form(self.tensor_space_FEM.E2.T.dot(b2))
        
        pic_ker.kernel_step1(particles_loc, self.T[0], self.T[1], self.T[2], self.p, self.Nel, self.NbaseN, self.NbaseD, particles_loc.shape[1], b2_1, b2_2, b2_3, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz, self.mat12_loc, self.mat13_loc, self.mat23_loc, self.basis_u)
        
        mpi_comm.Reduce(self.mat12_loc, self.mat12, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.mat13_loc, self.mat13, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.mat23_loc, self.mat23, op=MPI.SUM, root=0)
       
    # ===============================================================
    def assemble_step1(self, Np, b2):
        
        b2_1, b2_2, b2_3 = self.tensor_space_FEM.unravel_2form(self.tensor_space_FEM.E2.T.dot(b2))
            
        # delta-f correction
        if self.control == True:
            mat12_df, mat13_df, mat23_df = self.cont.mass_nh_eq(b2_1, b2_2, b2_3)
        else:
            mat12_df = np.zeros(self.mat12.shape, dtype=float)
            mat13_df = np.zeros(self.mat13.shape, dtype=float)
            mat23_df = np.zeros(self.mat23.shape, dtype=float)
            
        # build global sparse matrix
        return self.to_sparse_step1(self.mat12/Np + mat12_df, self.mat13/Np + mat13_df, self.mat23/Np + mat23_df)
        
    # ===============================================================
    def accumulate_step3(self, particles_loc, b2, mpi_comm):
        
        b2_1, b2_2, b2_3 = self.tensor_space_FEM.unravel_2form(self.tensor_space_FEM.E2.T.dot(b2))
        
        pic_ker.kernel_step3(particles_loc, self.T[0], self.T[1], self.T[2], self.p, self.Nel, self.NbaseN, self.NbaseD, particles_loc.shape[1], b2_1, b2_2, b2_3, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz, self.mat11_loc, self.mat12_loc, self.mat13_loc, self.mat22_loc, self.mat23_loc, self.mat33_loc, self.vec1_loc, self.vec2_loc, self.vec3_loc, self.basis_u)
        
        mpi_comm.Reduce(self.mat11_loc, self.mat11, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.mat12_loc, self.mat12, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.mat13_loc, self.mat13, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.mat22_loc, self.mat22, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.mat23_loc, self.mat23, op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.mat33_loc, self.mat33, op=MPI.SUM, root=0)

        mpi_comm.Reduce(self.vec1_loc , self.vec1 , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vec2_loc , self.vec2 , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vec3_loc , self.vec3 , op=MPI.SUM, root=0)
        
    # ===============================================================
    def assemble_step3(self, Np, b2):
        
        b2_1, b2_2, b2_3 = self.tensor_space_FEM.unravel_2form(self.tensor_space_FEM.E2.T.dot(b2))
            
        # delta-f correction
        if self.control == True:
            vec_df = self.cont.inner_prod_jh_eq(b2_1, b2_2, b2_3)     
        else:
            vec_df = np.zeros(self.vec1.size + self.vec2.size, self.vec3.siize, dtype=float)
            
        # build global sparse matrix
        return (self.to_sparse_step3(self.mat11, self.mat12, self.mat13, self.mat22, self.mat23, self.mat23)/Np).tocsr(), np.concatenate((self.vec1.flatten(), self.vec2.flatten(), self.vec3.flatten()))/Np + vec_df