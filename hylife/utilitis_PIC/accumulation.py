# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Basic modules to compute charge and current densities from particles
"""


import numpy        as np
import scipy.sparse as spa

import hylife.utilitis_PIC.accumulation_kernels as ker


class accumulation:
    """
    Class for computing charge and current densities from particles.
    
    Parameters
    ---------
    tensor_space : tensor_spline_space
        tensor product B-spline space
    """
        
    
    def __init__(self, tensor_space):
         
        self.T        = tensor_space.T           # knot vectors
        self.p        = tensor_space.p           # spline degrees
        self.bc       = tensor_space.bc          # boundary conditions (True : periodic, False : clamped)
        
        self.t        = tensor_space.t           # reduced knot vectors
        self.el_b     = tensor_space.el_b        # element boundaries
        self.Nel      = tensor_space.Nel         # number of elements
        self.NbaseN   = tensor_space.NbaseN      # number of basis functions (N)
        self.NbaseD   = tensor_space.NbaseD      # number of basis functions (D)
        
        
    def accumulation_step1(self, particles, b_part, kind_map, params_map):
        """
        Computes the term sum_{ip=1}^Np [lambda^1_i(ip) * g_inv(ip) * B(ip) x g_inv(ip) * lambda^1_j(ip)]
        
        Paramters
        ---------
        particles : array_like
            particles in format (Np, 7) from which the term is computed
            
        b_part : array_like
            magnetic field at particle positions in format (Np, 3)
        
        kind_map : int
            type of mapping
            
        params_map : list of doubles
            paramters for the mapping
        
        Returns
        -------
        mat : sparse matrix in csc-format
            Accumulated term of above expression
        """
        
        Np = particles.shape[0]
        
        # only distinctive non-vanishing blocks
        mat12 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        mat13 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        mat23 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        
        # perform accumulation
        ker.kernel_step1(particles, self.T[0], self.T[1], self.T[2], self.p, self.Nel, self.NbaseN, self.NbaseD, Np, b_part, kind_map, params_map, mat12, mat13, mat23)
        
        # conversion to sparse matrices
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseD[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        mat12   = spa.csr_matrix((mat12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))
        mat12.eliminate_zeros()
        
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseD[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

        col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        mat13   = spa.csr_matrix((mat13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
        mat13.eliminate_zeros()
        
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseD[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

        col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        mat23   = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
        mat23.eliminate_zeros()
        
        mat = spa.bmat([[None, mat12, mat13], [-mat12.T, None, mat23], [-mat13.T, -mat23.T, None]], format='csc')
        
        return mat
    
    
    def accumulation_step3(self, particles, b_part, kind_map, params_map):
        """
        Computes the term sum_{ip=1}^Np [lambda^1_i(ip) * g_inv(ip) * B(ip) x g_inv(ip) * B(ip) x g_inv(ip) * lambda^1_j(ip)].
        Computes the term sum_{ip=1}^Np [lambda^1_i(ip) * g_inv(ip) * DF_inv(ip) * V(ip)].
        
        Paramters
        ---------
        particles : array_like
            particles in format (Np, 7) from which the term is computed
            
        b_part : array_like
            magnetic field at particle positions in format (Np, 3)
        
        kind_map : int
            type of mapping
            
        params_map : list of doubles
            paramters for the mapping
        
        Returns
        -------
        mat : sparse matrix in csc-format
            Accumulated term of first above expression 
            
        vec : array_like
            Accumulated term of second above expression
        """
        
        Np = particles.shape[0]

        mat11 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        mat12 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        mat13 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        mat22 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        mat23 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
        mat33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)

        vec1  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
        vec2  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
        vec3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)
        
        # perform accumulation
        ker.kernel_step3(particles, self.T[0], self.T[1], self.T[2], self.p, self.Nel, self.NbaseN, self.NbaseD, Np, b_part, kind_map, params_map, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)
        
        
        # conversion to sparse matrices
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseD[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        mat11   = spa.csr_matrix((mat11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))
        mat11.eliminate_zeros()
        
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseD[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        mat12   = spa.csr_matrix((mat12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))
        mat12.eliminate_zeros()
        
        indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseD[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

        col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        mat13   = spa.csr_matrix((mat13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
        mat13.eliminate_zeros()
                         
                         
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseD[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

        col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3
        
        mat22   = spa.csr_matrix((mat22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))
        mat22.eliminate_zeros()
                         
        indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseD[1]) - self.p[1]
        shift3  = np.arange(self.NbaseN[2]) - self.p[2]
        
        row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

        col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        mat23   = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
        mat23.eliminate_zeros()
                         
                         
        indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))
        
        shift1  = np.arange(self.NbaseN[0]) - self.p[0]
        shift2  = np.arange(self.NbaseN[1]) - self.p[1]
        shift3  = np.arange(self.NbaseD[2]) - self.p[2]
        
        row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]
        
        col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
        col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
        col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

        col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3
        
        mat33   = spa.csr_matrix((mat33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
        mat33.eliminate_zeros()
                         
        mat = spa.bmat([[mat11, mat12, mat13], [mat12.T, mat22, mat23], [mat13.T, mat23.T, mat33]], format='csc')
        
        return mat, np.concatenate((vec1.flatten(), vec2.flatten(), vec3.flatten()))