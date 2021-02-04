# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""


import numpy        as np
import scipy.sparse as spa


class accumulation:
    """
    Class for computing charge and current densities from particles.
    
    Parameters
    ---------
    tensor_space : tensor_spline_space
        tensor product B-spline space
    """
        
    # ===============================================================
    def __init__(self, tensor_space_FEM):
         
        self.T      = tensor_space_FEM.T        # knot vectors
        self.p      = tensor_space_FEM.p        # spline degrees
        self.bc     = tensor_space_FEM.bc       # boundary conditions (True : periodic, False : clamped)
        
        self.t      = tensor_space_FEM.t        # reduced knot vectors
        self.el_b   = tensor_space_FEM.el_b     # element boundaries
        self.Nel    = tensor_space_FEM.Nel      # number of elements
        self.NbaseN = tensor_space_FEM.NbaseN   # number of basis functions (N)
        self.NbaseD = tensor_space_FEM.NbaseD   # number of basis functions (D)
        
    
    # ===============================================================
    def to_sparse_step1(self, mat12, mat13, mat23, kind):
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
            
        kind : string
            the basis of U
        
        Returns
        -------
        M : sparse matrix in csr-format
            2d anti-symmetric, sparse block matrix [[0, M12, mat13], [-M12.T, 0, M23], [-M13.T, -M23.T, 0]]
        """
        
        
        # conversion to sparse matrix if all components of U live in V0
        if kind == '0-form':
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
        elif kind == '2-form':
            
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
    def to_sparse_step3(self, mat11, mat12, mat13, mat22, mat23, mat33, kind):
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
            
        kind : string
            the basis of U
        
        Returns
        -------
        M : sparse matrix in csr-format
            2d symmetric, sparse block matrix [[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]]
        """
        
        # conversion to sparse matrix if all components of U live in V0
        if kind == '0-form':
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
        if kind == '2-form':
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