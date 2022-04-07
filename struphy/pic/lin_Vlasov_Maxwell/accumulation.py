# coding: utf-8

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

from mpi4py import MPI

import numpy        as np
import scipy.sparse as spa

from struphy.pic.lin_Vlasov_Maxwell.accum_kernels_3d import accum_step_e_w

class Accumulation:
    """
    Class for computing matrices used in substep 3 of lin_Vlasov_Maxwell (sum over particles)
    
    Parameters
    ---------
    tensor_space_FEM : Tensor_spline_space
        tensor product B-spline space
        
    domain : domain object
        domain object from struphy.geometry.domain_3d defining the mapping
    
    mpi_comm : MPI.COMM_WORLD
        MPI communicator
    
    """
        
    # =======================================================================================================
    def __init__(self, tensor_space_FEM, domain, mpi_comm):
        
        self.space    = tensor_space_FEM
        self.domain   = domain       
        self.mpi_rank = mpi_comm.Get_rank()
        
        # reserve memory for implicit particle-coupling sub-steps
        self.blocks_loc = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.vecs_loc   =  [0, 0, 0]
        
        self.blocks     = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.vecs       =  [0, 0, 0]

        for a in range(3):
            
            Ni = self.space.Nbase_1form[a]

            self.vecs_loc[a] = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)
            
            if self.mpi_rank == 0:
                self.vecs[a] = np.empty((Ni[0], Ni[1], Ni[2]), dtype=float)
            
            for b in range(a, 3):
                
                if self.space.dim == 3:
                
                    self.blocks_loc[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)

                    if self.mpi_rank == 0:
                        self.blocks[a][b] = np.empty((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)
                else:
                    raise ValueError('Only 3D implemented for now !')

        



    # =======================================================================================================
    def accumulate_e_W_step(self, particles_loc, mpi_comm, Np_loc, v_shift, v_th, n0):
        """
        TODO
        """
        
        if self.space.dim == 3:

            accum_step_e_w( particles_loc,
                            self.space.p,
                            self.space.T[0], self.space.T[1], self.space.T[2],
                            Np_loc,
                            self.space.indN[0], self.space.indN[1], self.space.indN[2],
                            self.space.indD[0], self.space.indD[1], self.space.indD[2],
                            self.blocks_loc[0][0], self.blocks_loc[1][1], self.blocks_loc[2][2],
                            self.vecs_loc[0], self.vecs_loc[1], self.vecs_loc[2],
                            v_shift, v_th, n0
                            )
        
        else:
            raise ValueError('Only 3D implemented for now !')
        

        mpi_comm.Reduce(self.blocks_loc[0][0], self.blocks[0][0], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][1], self.blocks[1][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[2][2], self.blocks[2][2], op=MPI.SUM, root=0)

        mpi_comm.Reduce(self.vecs_loc[0] , self.vecs[0] , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vecs_loc[1] , self.vecs[1] , op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.vecs_loc[2] , self.vecs[2] , op=MPI.SUM, root=0)
        

        



    # =======================================================================================================
    def to_sparse_step_e_W(self):
        """
        Converts the 6d arrays stored in self.blocks to a sparse block matrix using row-major ordering.
        
        Returns
        -------
        M : sparse matrix in csr-format
            diagonal, sparse block matrix [[M11, 0, 0], [0, M22, 0], [0, 0, M33]]
        """
        
        # blocks of global matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # entries of global vector
        vec = [0, 0, 0]
        
        for a in range(3):
            
            # reshape matrix and convert to sparse format
            Ni = self.space.Nbase_1form[a]
            
            indices = np.indices(self.blocks[a][a].shape)
            
            row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
        
            shift   = [np.arange(Ni) - p for Ni, p in zip(Ni[:2], self.space.p[:2])]

            if self.space.dim == 3:
                shift += [np.arange(Ni[2]) - self.space.p[2]]
            else:
                raise ValueError('On 3D implemented so far !')

            col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Ni[0]
            col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Ni[1]
            col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Ni[2]

            col     = Ni[1]*Ni[2]*col1 + Ni[2]*col2 + col3
            
            M[a][a] = spa.csr_matrix((self.blocks[a][a].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Ni[0]*Ni[1]*Ni[2]))
            M[a][a].eliminate_zeros()


            # reshape vector
            vec[a] = self.vecs[a].flatten()


        
        # final block matrix
        M = spa.bmat([[M[0][0], None, None], [None, M[1][1], None], [None, None, M[2][2]]], format='csr')
        
        # apply extraction operator
        # M = self.space.E1_0.dot(M.dot(self.space.E1_0.T)).tocsr()
        # M = M.toscr()
                
        return M, np.array(vec)



    def assemble_step_e_W(self, Np):

        # self.blocks[0][0] /= Np
        # self.blocks[1][1] /= Np
        # self.blocks[2][2] /= Np

        # self.vecs[0] /= Np
        # self.vecs[1] /= Np
        # self.vecs[2] /= Np

        # build global sparse matrix
        return self.to_sparse_step_e_W()

        