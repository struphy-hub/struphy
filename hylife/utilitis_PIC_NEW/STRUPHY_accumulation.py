import numpy as np
import scipy.sparse as spa

#import hylife.utilitis_FEEC.bsplines                as bsp
#import hylife.utilitis_PIC_NEW.STRUPHY_accumulation_kernels as ker

import bsplines as bsp
import STRUPHY_accumulation_kernels as ker



class accumulation:
    
    def __init__(self, T, p, bc):
        
        self.T        = T
        self.p        = p
        self.bc       = bc
        
        self.t        = [T[1:-1] for T in self.T]
        self.el_b     = [bsp.breakpoints(T, p) for T, p in zip(self.T, self.p)]
        self.Nel      = [len(el_b) - 1 for el_b in self.el_b]
        self.NbaseN   = [Nel + p - bc*p for Nel, p, bc in zip(self.Nel, self.p, self.bc)]
        self.NbaseD   = [NbaseN - (1 - bc) for NbaseN, bc in zip(self.NbaseN, self.bc)]
           
    
    
    
    def accumulation_step1(self, particles, b_part, kind_map, params_map):
        
        mat12 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        mat13 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        mat23 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        
        # perform accumulation
        ker.kernel_step1(particles, self.p, self.Nel, [self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]], self.T[0], self.T[1], self.T[2], self.t[0], self.t[1], self.t[2], b_part, kind_map, params_map, mat12, mat13, mat23)
       
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
        
        mat23   = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
        mat23.eliminate_zeros()
        
        mat = spa.bmat([[None, mat12, mat13], [-mat12.T, None, mat23], [-mat13.T, -mat23.T, None]], format='csc')
        
        return mat
        
        
    def accumulation_step3(self, particles, b_part, kind_map, params_map):

        mat11 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        mat12 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        mat13 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        mat22 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        mat23 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')
        mat33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float, order='F')

        vec1  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float, order='F')
        vec2  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float, order='F')
        vec3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float, order='F')

        # perform accumulation
        ker.kernel_step3(particles, self.p, self.Nel, [self.NbaseN[0], self.NbaseN[1], self.NbaseN[2], self.NbaseD[0], self.NbaseD[1], self.NbaseD[2]], self.T[0], self.T[1], self.T[2], self.t[0], self.t[1], self.t[2], b_part, kind_map, params_map, mat11, mat12, mat13, mat22, mat23, mat33, vec1, vec2, vec3)


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