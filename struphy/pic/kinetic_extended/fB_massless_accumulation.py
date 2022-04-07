# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

from mpi4py import MPI

import numpy        as np
import scipy.sparse as spa

import struphy.pic.kinetic_extended.fB_massless_accumulation_kernels as acc_ker

class accumulation:
    """
    Class for computing charge and current densities from particles.
    
    Parameters
    ---------
    tensor_space_FEM : tensor_spline_space
        tensor product B-spline space

    domain : obj
            domain object from geometry/domain_3d.
        
    basis_p : int
        finite element space (0 : 0-form, 1 : 1-form , 2 : 2-form, 3 : 3-form)

    substep : int
        which substep 

    Np_loc  : double particle number in each mpi rank 
    mpi_comm: mpi environment 
    control : bool viriable, delta method is used or not
    p_shape : list of int, B-spline degrees of shape function in three directions
    p_size  : list of double, support of shape function 
    """
        
    # ===============================================================
    def __init__(self, tensor_space_FEM, domain, substep, Np_loc, mpi_comm, control, p_shape, p_size):
        
        self.domain   = domain
        self.substep  = substep
        self.mpi_rank = mpi_comm.Get_rank()
        self.control  = control
        self.Np_loc   = Np_loc
        
        self.tensor_space_FEM = tensor_space_FEM
        
        self.T        = tensor_space_FEM.T        # knot vectors
        self.p        = tensor_space_FEM.p        # spline degrees
        self.bc       = tensor_space_FEM.bc       # boundary conditions (True : periodic, False : clamped)
        
        self.t        = tensor_space_FEM.t        # reduced knot vectors
        self.el_b     = tensor_space_FEM.el_b     # element boundaries
        self.Nel      = tensor_space_FEM.Nel      # number of elements
        self.NbaseN   = tensor_space_FEM.NbaseN   # number of basis functions (N)
        self.NbaseD   = tensor_space_FEM.NbaseD   # number of basis functions (D)
        self.n_quad   = tensor_space_FEM.n_quad
        
        self.p_shape      = p_shape
        self.p_size       = p_size
        self.pts          = tensor_space_FEM.pts
        self.wts          = tensor_space_FEM.wts
        self.index_diffx   = int(np.ceil(0.5*(self.p_shape[0]+1)*self.p_size[0]*self.Nel[0]))
        self.index_diffy   = int(np.ceil(0.5*(self.p_shape[1]+1)*self.p_size[1]*self.Nel[1]))
        self.index_diffz   = int(np.ceil(0.5*(self.p_shape[2]+1)*self.p_size[2]*self.Nel[2]))
        self.index_shapex  = np.empty(self.Nel[0]+2*self.index_diffx, dtype=int)
        self.index_shapey  = np.empty(self.Nel[1]+2*self.index_diffy, dtype=int)
        self.index_shapez  = np.empty(self.Nel[2]+2*self.index_diffz, dtype=int)
        for i in range(self.Nel[0]+2*self.index_diffx):
            self.index_shapex[i] = (i - self.index_diffx)%self.Nel[0]

        for i in range(self.Nel[1]+2*self.index_diffy):
            self.index_shapey[i] = (i - self.index_diffy)%self.Nel[1]

        for i in range(self.Nel[2]+2*self.index_diffz):
            self.index_shapez[i] = (i - self.index_diffz)%self.Nel[2]

        
        

        # ==== reserve memory for implicit particle-coupling sub-steps ==========
        if self.substep == 4:
            # project current into V1 space
            self.mat11_loc = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat12_loc = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat13_loc = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat22_loc = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat23_loc = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
            self.mat33_loc = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)

            self.vec1_loc  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.vec2_loc  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
            self.vec3_loc  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)

            self.oneform_temp1     = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.oneform_temp2     = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
            self.oneform_temp3     = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)

            self.twoform_temp1     = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]), dtype=float)
            self.twoform_temp2     = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)
            self.twoform_temp3     = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)

            self.oneform_temp1_long     = np.empty(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], dtype=float)
            self.oneform_temp2_long     = np.empty(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], dtype=float)
            self.oneform_temp3_long     = np.empty(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], dtype=float)

            self.oneform_temp_long     = np.empty(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]+self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]+self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], dtype=float)

            self.twoform_temp1_long     = np.empty(self.NbaseN[0]*self.NbaseD[1]*self.NbaseD[2], dtype=float)
            self.twoform_temp2_long     = np.empty(self.NbaseD[0]*self.NbaseN[1]*self.NbaseD[2], dtype=float)
            self.twoform_temp3_long     = np.empty(self.NbaseD[0]*self.NbaseD[1]*self.NbaseN[2], dtype=float)

            self.curl_b1     = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseD[2]), dtype=float)
            self.curl_b2     = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)
            self.curl_b3     = np.empty((self.NbaseD[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)

            if self.mpi_rank == 0:
                self.mat11 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat12 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat13 = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat22 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat23 = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)
                self.mat33 = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1), dtype=float)

                self.vec1  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
                self.vec2  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
                self.vec3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)


            else:
                self.mat11, self.mat12, self.mat13, self.mat22, self.mat23, self.mat33 = None, None, None, None, None, None
                self.vec1,  self.vec2,  self.vec3 = None, None, None

        elif substep == 2:
            # vv sbstep 
            # project current into V1 space using L2 projection
            self.vec1_loc  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.vec2_loc  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
            self.vec3_loc  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)

            self.stage1_out_loc = np.empty((3, self.Np_loc), dtype=float)
            self.stage2_out_loc = np.empty((3, self.Np_loc), dtype=float)
            self.stage3_out_loc = np.empty((3, self.Np_loc), dtype=float)
            self.stage4_out_loc = np.empty((3, self.Np_loc), dtype=float)

            self.mid_particles  = np.zeros((3, self.Np_loc), dtype=float)

            if self.control == True:
                self.control = np.empty((3, self.Np_loc), dtype=float)

            self.coe1  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
            self.coe2  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
            self.coe3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)

            if self.mpi_rank == 0:
                self.vec1  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
                self.vec2  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
                self.vec3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)

                self.one_form1  = np.empty((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2]), dtype=float)
                self.one_form2  = np.empty((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2]), dtype=float)
                self.one_form3  = np.empty((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2]), dtype=float)

                self.temp1      = np.empty(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], dtype=float)
                self.temp2      = np.empty(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], dtype=float)
                self.temp3      = np.empty(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], dtype=float)
                    
            else:
                self.vec1,  self.vec2,  self.vec3 = None, None, None
                self.one_form1, self.one_form2, self.one_form3 = None, None, None
                self.temp1, self.temp2, self.temp3 = None, None, None

        # =======================================================================

    # ===============================================================
    def to_sparse_substep4(self, mat11, mat12, mat13, mat22, mat23, mat33):
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
        # conversion to sparse matrix if projection lives in V1
        if self.basis_p == 1:
            # conversion to sparse matrix (11-block)
            indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseD[0]) - self.p[0]
            shift2  = np.arange(self.NbaseN[1]) - self.p[1]
            shift3  = np.arange(self.NbaseN[2]) - self.p[2]

            row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseD[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseN[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M11   = spa.csr_matrix((mat11.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2]))
            M11.eliminate_zeros()


            # conversion to sparse matrix (12-block)
            indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseD[0]) - self.p[0]
            shift2  = np.arange(self.NbaseN[1]) - self.p[1]
            shift3  = np.arange(self.NbaseN[2]) - self.p[2]

            row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M12   = spa.csr_matrix((mat12.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))
            M12.eliminate_zeros()


            # conversion to sparse matrix (13-block)
            indices = np.indices((self.NbaseD[0], self.NbaseN[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseD[0]) - self.p[0]
            shift2  = np.arange(self.NbaseN[1]) - self.p[1]
            shift3  = np.arange(self.NbaseN[2]) - self.p[2]

            row     = self.NbaseN[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

            col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3

            M13   = spa.csr_matrix((mat13.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseD[0]*self.NbaseN[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
            M13.eliminate_zeros()


            # conversion to sparse matrix (22-block)
            indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseN[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseD[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseN[2]

            col     = self.NbaseD[1]*self.NbaseN[2]*col1 + self.NbaseN[2]*col2 + col3

            M22   = spa.csr_matrix((mat22.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2]))
            M22.eliminate_zeros()


            # conversion to sparse matrix (23-block)
            indices = np.indices((self.NbaseN[0], self.NbaseD[1], self.NbaseN[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseD[1]) - self.p[1]
            shift3  = np.arange(self.NbaseN[2]) - self.p[2]

            row     = self.NbaseD[1]*self.NbaseN[2]*indices[0] + self.NbaseN[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

            col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3

            M23   = spa.csr_matrix((mat23.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseD[1]*self.NbaseN[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
            M23.eliminate_zeros()


            # conversion to sparse matrix (33-block)
            indices = np.indices((self.NbaseN[0], self.NbaseN[1], self.NbaseD[2], 2*self.p[0] + 1, 2*self.p[1] + 1, 2*self.p[2] + 1))

            shift1  = np.arange(self.NbaseN[0]) - self.p[0]
            shift2  = np.arange(self.NbaseN[1]) - self.p[1]
            shift3  = np.arange(self.NbaseD[2]) - self.p[2]

            row     = self.NbaseN[1]*self.NbaseD[2]*indices[0] + self.NbaseD[2]*indices[1] + indices[2]

            col1    = (indices[3] + shift1[:, None, None, None, None, None])%self.NbaseN[0]
            col2    = (indices[4] + shift2[None, :, None, None, None, None])%self.NbaseN[1]
            col3    = (indices[5] + shift3[None, None, :, None, None, None])%self.NbaseD[2]

            col     = self.NbaseN[1]*self.NbaseD[2]*col1 + self.NbaseD[2]*col2 + col3

            M33   = spa.csr_matrix((mat33.flatten(), (row.flatten(), col.flatten())), shape=(self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2], self.NbaseN[0]*self.NbaseN[1]*self.NbaseD[2]))
            M33.eliminate_zeros()
                         
        
            # final block matrix
            M = spa.bmat([[M11, M12, M13], [M12.T, M22, M23], [M13.T, M23.T, M33]], format='csr')
        
            return M

    # ===============================================================
    def accumulate_substep4(self, particles_loc, mpi_comm):
        indN = self.tensor_space_FEM.indN
        indD = self.tensor_space_FEM.indD
        acc_ker.kernel_step4(indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], particles_loc, self.T[0], self.T[1], self.T[2], self.p, self.Nel, self.NbaseN, self.NbaseD, particles_loc.shape[1], self.domain.kind_map, self.domain.params_map, self.mat11_loc, self.mat12_loc, self.mat13_loc, self.mat22_loc, self.mat23_loc, self.mat33_loc, self.vec1_loc, self.vec2_loc, self.vec3_loc, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz)
        
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
    def assemble_substep4(self, Np):

        # build global sparse matrix
        return (self.to_sparse_substep4(self.mat11, self.mat12, self.mat13, self.mat22, self.mat23, self.mat33)/Np).tocsr(), np.concatenate((self.vec1.flatten(), self.vec2.flatten(), self.vec3.flatten()))/Np

    # ===============================================================
    def accumulate_substep_vv(self, particles_loc, mpi_comm, index_label, dt):
        indN = self.tensor_space_FEM.indN
        indD = self.tensor_space_FEM.indD
        if index_label == 1:
            acc_ker.accumulate_substepvv(0.0, indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], self.Nel, self.p, self.NbaseN, self.NbaseD, self.T[0], self.T[1], self.T[2], particles_loc.shape[1], self.vec1_loc, self.vec2_loc, self.vec3_loc, particles_loc, self.mid_particles, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz)
        elif index_label == 2:
            acc_ker.accumulate_substepvv(0.5*dt, indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], self.Nel, self.p, self.NbaseN, self.NbaseD, self.T[0], self.T[1], self.T[2], particles_loc.shape[1], self.vec1_loc, self.vec2_loc, self.vec3_loc, particles_loc, self.stage1_out_loc, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz)
        elif index_label == 3:
            acc_ker.accumulate_substepvv(0.5*dt, indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], self.Nel, self.p, self.NbaseN, self.NbaseD, self.T[0], self.T[1], self.T[2], particles_loc.shape[1], self.vec1_loc, self.vec2_loc, self.vec3_loc, particles_loc, self.stage2_out_loc, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz)
        else:
            acc_ker.accumulate_substepvv(dt, indN[0], indN[1], indN[2], indD[0], indD[1], indD[2], self.Nel, self.p, self.NbaseN, self.NbaseD, self.T[0], self.T[1], self.T[2], particles_loc.shape[1], self.vec1_loc, self.vec2_loc, self.vec3_loc, particles_loc, self.stage3_out_loc, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz)

        mpi_comm.Reduce(self.vec1_loc , self.vec1 , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vec2_loc , self.vec2 , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vec3_loc , self.vec3 , op=MPI.SUM, root=0)

