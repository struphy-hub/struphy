# coding: utf-8

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

from mpi4py import MPI

import numpy        as np
import scipy.sparse as spa

from struphy.pic.lin_Vlasov_Maxwell.accum_kernels_3d   import accum_step_e_w

class Accumulation:
    """
    Class for computing matrix and vector used in substep 3 of lin_Vlasov_Maxwell (sum over particles)
    
    Parameters :
    ------------
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
                self.vecs[a] = np.zeros((Ni[0], Ni[1], Ni[2]), dtype=float)
            
            for b in range(3):
                
                if self.space.dim == 3:
                
                    self.blocks_loc[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)

                    if self.mpi_rank == 0:
                        self.blocks[a][b] = np.zeros((Ni[0], Ni[1], Ni[2], 2*self.space.p[0] + 1, 2*self.space.p[1] + 1, 2*self.space.p[2] + 1), dtype=float)
                else:
                    raise ValueError('Only 3D implemented for now !')


    # =======================================================================================================
    def accumulate_e_W_step(self, particles, mpi_comm, Np, v_shift, v_th, n0):
        """
        calls the accumulation step in accum_kernels_3d

        Parameters : 
        ------------
            particles : array
                contains the positions [0:3,], velocities [3:6,], and weights [6,]
            
            mpi_comm : MPI.COMM_WORLD
                MPI communicator
            
            Np : integer
                number of particles in total
            
            v_shift : array
                contains the values of the shift in velocity for the background solution
            
            v_th : array
                contains the values of the thermal velocity for the background solution

            n0 : double
                value for the homogeneous space-dependent part of the background solution
        """
        from time import time

        if self.space.dim == 3:

            start_time = time()

            accum_step_e_w( particles,
                            self.domain.kind_map, self.domain.params_map,
                            np.array(self.space.p),
                            self.space.T[0], self.space.T[1], self.space.T[2],
                            Np,
                            self.space.indN[0], self.space.indN[1], self.space.indN[2],
                            self.blocks_loc[0][0], self.blocks_loc[0][1], self.blocks_loc[0][2],
                            self.blocks_loc[1][0], self.blocks_loc[1][1], self.blocks_loc[1][2],
                            self.blocks_loc[2][0], self.blocks_loc[2][1], self.blocks_loc[2][2],
                            self.vecs_loc[0], self.vecs_loc[1], self.vecs_loc[2],
                            v_shift, v_th, n0,
                            self.domain.cx, self.domain.cy, self.domain.cz
                            )
            
            end_time = time()
            accum_time = np.round(end_time - start_time, 3)

            print('Accumulation done in '+str(accum_time)+'s.')
            print()
        
        else:
            raise ValueError('Only 3D implemented for now !')
        
        
        for a in range(3):

            assert np.isnan(self.vecs_loc[a]).any() == False, 'NaN found in accumulation vector'

            for b in range(3):
                assert np.isnan(self.blocks_loc[a][b]).any() == False, 'NaN found in accumulation matrix'


        mpi_comm.Reduce(self.blocks_loc[0][0], self.blocks[0][0], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[0][1], self.blocks[0][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[0][2], self.blocks[0][2], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][0], self.blocks[1][0], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][1], self.blocks[1][1], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[1][2], self.blocks[1][2], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[2][0], self.blocks[2][0], op=MPI.SUM, root=0)
        mpi_comm.Reduce(self.blocks_loc[2][1], self.blocks[2][1], op=MPI.SUM, root=0)
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
            symmetric, sparse block matrix [[M11, M12, M13], [0, M22, M23], [0, 0, M33]]
        """
        
        # blocks of global matrix
        M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # entries of global vector
        vec = [0, 0, 0]
        
        for a in range(3):

            # reshape vector
            vec[a] = self.vecs[a].flatten()

            for b in range(3):
            
                # reshape matrix and convert to sparse format
                Ni = self.space.Nbase_1form[a]
                Nj = self.space.Nbase_1form[b]
                
                indices = np.indices(self.blocks[a][b].shape)
                
                row     = (Ni[1]*Ni[2]*indices[0] + Ni[2]*indices[1] + indices[2]).flatten()
            
                shift   = [np.arange(Ni) - p for Ni, p in zip(Ni[:2], self.space.p[:2])]

                if self.space.dim == 3:
                    shift += [np.arange(Ni[2]) - self.space.p[2]]
                else:
                    raise ValueError('On 3D implemented so far !')

                col1    = (indices[3] + shift[0][:, None, None, None, None, None])%Nj[0]
                col2    = (indices[4] + shift[1][None, :, None, None, None, None])%Nj[1]
                col3    = (indices[5] + shift[2][None, None, :, None, None, None])%Nj[2]

                col     = Nj[1]*Nj[2]*col1 + Nj[2]*col2 + col3
                
                M[a][b] = spa.csr_matrix((self.blocks[a][b].flatten(), (row, col.flatten())), shape=(Ni[0]*Ni[1]*Ni[2], Nj[0]*Nj[1]*Nj[2]))
                M[a][b].eliminate_zeros()
        
        # final block matrix
        M = spa.bmat([[M[0][0], M[0][1], M[0][2]], [M[1][0], M[1][1], M[1][2]], [M[2][0], M[2][1], M[2][2]]], format='csr')
        
        return M, np.array(vec)



    def assemble_step_e_W(self, Np):
        """
        assembles the final accumulation matrix and vector needed in e_W step in sparse format

        Parameters : 
        ------------
            Np : integer
                total nuber of particles
        """

        # build global sparse matrix
        return self.to_sparse_step_e_W()

        
