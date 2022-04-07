# coding: utf-8
#
# Copyright 2020 Florian Holderied

"""
Modules to create sparse matrices from 6D sub-matrices in particle accumulation steps
"""

from mpi4py import MPI

import numpy        as np
import scipy.sparse as spa

import struphy.pic.kinetic_extended.fB_gather_kernels as gather_ker

#import struphy.feec.control_variates.kinetic_extended.fB_massless_control_variate as cv

import time

class gather:
    '''
    Class for computing charge and current densities from particles.
    Parameters
    ----------
        tensor_space_FEM : tensor_spline_space, tensor product B-spline space
        
        domain : obj, Domain object from geometry/domain_3d.

        Np_loc : int, Number of all particles.in each rank of mpi

        p_shape : list of int, B-spline degrees of shape function in three directions
        
        p_size  : list of double, support of shape function 
        
        tol : double, threshold related with the denominator

        control: bool variable, delta f is used or not

        MPI_COMM : obj, Environment of MPI
    '''
        
    # ===============================================================
    def __init__(self, tensor_space_FEM, domain, Np_loc, p_shape, p_size, tol, mpi_comm, control):
        
        self.domain   = domain
        self.mpi_rank = mpi_comm.Get_rank()
        self.control  = control
        self.Np_loc   = Np_loc
        self.tol      = tol
        
        self.tensor_space_FEM = tensor_space_FEM
        self.n_quad     = tensor_space_FEM.n_quad
        
        self.T        = tensor_space_FEM.T        # knot vectors
        self.p        = tensor_space_FEM.p        # spline degrees
        self.bc       = tensor_space_FEM.bc       # boundary conditions (True : periodic, False : clamped)
        
        self.t        = tensor_space_FEM.t        # reduced knot vectors
        self.el_b     = tensor_space_FEM.el_b     # element boundaries
        self.Nel      = tensor_space_FEM.Nel      # number of elements
        self.NbaseN   = tensor_space_FEM.NbaseN   # number of basis functions (N)
        self.NbaseD   = tensor_space_FEM.NbaseD   # number of basis functions (D)
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

        self.gather_quadrature_loc = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.gather_grid_loc = np.empty((self.Nel[0], self.Nel[1], self.Nel[2]), dtype=float)

        self.gather_quadrature = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)
        self.gather_grid       = np.empty((self.Nel[0], self.Nel[1], self.Nel[2]), dtype=float)

        self.grid_inverse       = np.empty((self.Nel[0], self.Nel[1], self.Nel[2]), dtype=float)
        self.quadrature_log     = np.empty((self.Nel[0], self.Nel[1], self.Nel[2], self.n_quad[0], self.n_quad[1], self.n_quad[2]), dtype=float)


    def func_gather_quadrature(self, particles_loc, Np_loc, Np, mpi_comm):
        time1 = time.time()
        gather_ker.gather_quadrature(self.index_shapex, self.index_shapey, self.index_shapez, self.index_diffx, self.index_diffy, self.index_diffz, self.p_shape, self.p_size, self.n_quad, self.pts[0], self.pts[1], self.pts[2], self.Nel, particles_loc, Np_loc, Np, self.gather_quadrature_loc, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz)
        mpi_comm.Reduce(self.gather_quadrature_loc, self.gather_quadrature, op=MPI.SUM, root=0)
        time2 = time.time()
        #print('check_gather_inside', time2 - time1)
        #if self.mpi_rank == 0:
        #    if self.control == True:
        #        cv.quadrature_density(self, self.domain)
        mpi_comm.Bcast(self.gather_quadrature,   root=0)




    def func_gather_grid(self, particles_loc, Np_loc, Np, mpi_comm):

        gather_ker.gather_grid(self.index_shapex, self.index_shapey, self.index_shapez, self.index_diffx, self.index_diffy, self.index_diffz, self.p_shape, self.p_size, self.Nel, particles_loc, Np_loc, Np, self.gather_grid_loc, self.domain.kind_map, self.domain.params_map, self.domain.T[0], self.domain.T[1], self.domain.T[2], self.domain.p, self.domain.Nel, self.domain.NbaseN, self.domain.cx, self.domain.cy, self.domain.cz)
        mpi_comm.Reduce(self.gather_grid_loc, self.gather_grid, op=MPI.SUM, root=0)
        #if self.mpi_rank == 0:
        #    if self.control == True:
        #        cv.quadrature_grid(self, self.domain)
        mpi_comm.Bcast(self.gather_grid,   root=0)



    def func_quadrature_inverse(self, LO_inv):
        gather_ker.quadratureinverse(self.gather_quadrature, LO_inv, self.Nel, self.n_quad, self.tol)


    def func_grid_inverse(self):
        gather_ker.gridinverse(self.gather_grid, self.grid_inverse, self.Nel, self.tol)


    def func_quadrature_log(self):
        gather_ker.quadraturelog(self.gather_quadrature, self.quadrature_log, self.Nel, self.n_quad, self.wts[0], self.wts[1], self.wts[2], self.tol)


