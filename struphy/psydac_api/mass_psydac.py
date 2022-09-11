import numpy as np

from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix

from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL

from struphy.psydac_api.mass_kernels_psydac import kernel_1d, kernel_2d, kernel_3d
from struphy.psydac_api.linear_operators import ApplyHomogeneousDirichletToOperator


class WeightedMass:
    """
    Class for assembling weighted mass matrices in 3d.
    
    Parameters
    ----------
        derham : struphy.psydac_api.psydac_derham.Derham
            Discrete de Rham sequence on the logical unit cube.

        domain : struphy.geometry.domains
            All things mapping.
            
        weights : obj
            A general object that provides access to callables that serve as weight functions (e.g. instance of a subclass of struphy.fields_background.mhd_equil.base.EquilibriumMHD).
    """
    
    def __init__(self, derham, domain, **weights):
        
        self._derham = derham
        self._domain = domain
        
        F = domain.F_psy.get_callable_mapping()
        
        # Make sure that mapping matrices correspond to last two indices when evaluating 3d point sets, i.e. (:,:,:,3,3) in order to enable matrix-matrix products with @
        def DF(e1, e2, e3):
            return np.transpose(F.jacobian(e1, e2, e3), axes=(2, 3, 4, 0, 1))

        def DFT(e1, e2, e3):
            return np.transpose(F.jacobian(e1, e2, e3), axes=(2, 3, 4, 1, 0))

        def G(e1, e2, e3):
            return DFT(e1, e2, e3) @ DF(e1, e2, e3) 

        def DFinv(e1, e2, e3):
            return np.transpose(F.jacobian_inv(e1, e2, e3), axes=(2, 3, 4, 0, 1))

        def DFinvT(e1, e2, e3):
            return np.transpose(F.jacobian_inv(e1, e2, e3), axes=(2, 3, 4, 1, 0))

        def Ginv(e1, e2, e3):
            return DFinv(e1, e2, e3) @ DFinvT(e1, e2, e3)
        
        # Cross product matrices and evaluation of cross products
        cross_mask = [[ 1, -1,  1], 
                      [ 1,  1, -1], 
                      [-1,  1,  1]]
        
        def eval_cross(e1, e2, e3, fun_list): 
            
            cross = np.array([[cross_mask[m][n] * fun(e1, e2, e3) for n, fun in enumerate(row)] for m, row in enumerate(fun_list)])
            
            return np.transpose(cross, axes=(2, 3, 4, 0, 1))
        
        
        if 'eq_mhd' in weights:
            j2_cross = [[lambda e1, e2, e3 : 0*e1, weights['eq_mhd'].j2_3, weights['eq_mhd'].j2_2],
                        [weights['eq_mhd'].j2_3, lambda e1, e2, e3 : 0*e2, weights['eq_mhd'].j2_1],
                        [weights['eq_mhd'].j2_2, weights['eq_mhd'].j2_1, lambda e1, e2, e3 : 0*e3]]
       
        # scalar functions
        fun_M0 = [[lambda e1, e2, e3 :   np.sqrt(F.metric_det(e1, e2, e3))]]
        fun_M3 = [[lambda e1, e2, e3 : 1/np.sqrt(F.metric_det(e1, e2, e3))]]
        
        # matrix functions
        fun_M1 = []
        fun_M2 = []
        fun_Mv = []
        
        if 'eq_mhd' in weights:
            fun_M1n = []
            fun_M2n = []
            fun_Mvn = []
            fun_M1J = []
            fun_M2J = []
            fun_MvJ = []
        
        for m in range(3):
            fun_M1 += [[]]
            fun_M2 += [[]]
            fun_Mv += [[]]
            
            if 'eq_mhd' in weights:
                fun_M1n += [[]]
                fun_M2n += [[]]
                fun_Mvn += [[]]
                fun_M1J += [[]]
                fun_M2J += [[]]
                fun_MvJ += [[]]
            
            for n in range(3):
                fun_M1[-1] += [lambda e1, e2, e3, m=m, n=n : Ginv(e1, e2, e3)[:, :, :, m, n]*np.sqrt(F.metric_det(e1, e2, e3))]
                fun_M2[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]/np.sqrt(F.metric_det(e1, e2, e3))]
                fun_Mv[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]*np.sqrt(F.metric_det(e1, e2, e3))]
                
                if 'eq_mhd' in weights:
                    fun_M1n[-1] += [lambda e1, e2, e3, m=m, n=n : Ginv(e1, e2, e3)[:, :, :, m, n]*np.sqrt(F.metric_det(e1, e2, e3))*weights['eq_mhd'].n0(e1, e2, e3, squeeze_output=False)]
                    fun_M2n[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]/np.sqrt(F.metric_det(e1, e2, e3))*weights['eq_mhd'].n0(e1, e2, e3, squeeze_output=False)]
                    fun_Mvn[-1] += [lambda e1, e2, e3, m=m, n=n : G(e1, e2, e3)[:, :, :, m, n]*np.sqrt(F.metric_det(e1, e2, e3))*weights['eq_mhd'].n0(e1, e2, e3, squeeze_output=False)]
                    fun_M1J[-1] += [lambda e1, e2, e3, m=m, n=n : (Ginv(e1, e2, e3) @ eval_cross(e1, e2, e3, j2_cross))[:, :, :, m, n]]
                    fun_M2J[-1] += [lambda e1, e2, e3, m=m, n=n : cross_mask[m][n]*j2_cross[m][n](e1, e2, e3)/np.sqrt(F.metric_det(e1, e2, e3))]
                    fun_MvJ[-1] += [lambda e1, e2, e3, m=m, n=n : cross_mask[m][n]*j2_cross[m][n](e1, e2, e3)]
                    
                
        self._fun_M0 = fun_M0
        self._fun_M3 = fun_M3
        
        self._fun_M1 = fun_M1
        self._fun_M2 = fun_M2
        self._fun_Mv = fun_Mv
        
        if 'eq_mhd' in weights:
            self._fun_M1n = fun_M1n
            self._fun_M2n = fun_M2n
            self._fun_Mvn = fun_Mvn
            self._fun_M1J = fun_M1J
            self._fun_M2J = fun_M2J
            self._fun_MvJ = fun_MvJ
     
        # only for M1 Mac users
        PSYDAC_BACKEND_GPYCCEL['flags'] = '-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none'
    
    
    @property
    def derham(self):
        return self._derham
    
    @property
    def domain(self):
        return self._domain
    
    
    def assemble_M0(self):
        """  Assemble mass matrix for L2-scalar product in V0.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M0 ...')
        self._M0 = ApplyHomogeneousDirichletToOperator('H1', 'H1', self.derham.bc, get_mass(self.derham.V0, self.derham.V0, self._fun_M0))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_M1(self):
        """  Assemble mass matrix for L2-scalar product in V1.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M1 ...')
        self._M1 = ApplyHomogeneousDirichletToOperator('Hcurl', 'Hcurl', self.derham.bc, get_mass(self.derham.V1, self.derham.V1, self._fun_M1))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_M2(self):
        """  Assemble mass matrix for L2-scalar product in V2.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M2 ...')
        self._M2 = ApplyHomogeneousDirichletToOperator('Hdiv', 'Hdiv', self.derham.bc, get_mass(self.derham.V2, self.derham.V2, self._fun_M2))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_M3(self):
        """  Assemble mass matrix for L2-scalar product in V3.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M3 ...')
        self._M3 = ApplyHomogeneousDirichletToOperator('L2', 'L2', self.derham.bc, get_mass(self.derham.V3, self.derham.V3, self._fun_M3))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_Mv(self):
        """  Assemble mass matrix for L2-scalar product in V0vec.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling Mv ...')
        self._Mv = ApplyHomogeneousDirichletToOperator('H1vec', 'H1vec', self.derham.bc, get_mass(self.derham.V0vec, self.derham.V0vec, self._fun_Mv))
        if self.derham.comm.Get_rank() == 0: print('Done.')
              
    def assemble_M1n(self):
        """  Assemble mass matrix for L2-scalar product in V1 weighted with MHD number density.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M1n ...')
        self._M1n = ApplyHomogeneousDirichletToOperator('Hcurl', 'Hcurl', self.derham.bc, get_mass(self.derham.V1, self.derham.V1, self._fun_M1n))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_M2n(self):
        """  Assemble mass matrix for L2-scalar product in V2 weighted with MHD number density.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M2n ...')
        self._M2n = ApplyHomogeneousDirichletToOperator('Hdiv', 'Hdiv', self.derham.bc, get_mass(self.derham.V2, self.derham.V2, self._fun_M2n))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_Mvn(self):
        """  Assemble mass matrix for L2-scalar product in V0vec weighted with MHD number density.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling Mvn ...')
        self._Mvn = ApplyHomogeneousDirichletToOperator('H1vec', 'H1vec', self.derham.bc, get_mass(self.derham.V0vec, self.derham.V0vec, self._fun_Mvn))
        if self.derham.comm.Get_rank() == 0: print('Done.')
            
    def assemble_M1J(self):
        """  Assembles mass matrix for L2-scalar product in V1 weighted with cross product of MHD equilibrium current density.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M1J ...')
        self._M1J = ApplyHomogeneousDirichletToOperator('Hdiv', 'Hcurl', self.derham.bc, get_mass(self.derham.V1, self.derham.V2, self._fun_M1J))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_M2J(self):
        """  Assembles mass matrix for L2-scalar product in V2 weighted with cross product of MHD equilibrium current density.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling M2J ...')
        self._M2J = ApplyHomogeneousDirichletToOperator('Hdiv', 'Hdiv', self.derham.bc, get_mass(self.derham.V2, self.derham.V2, self._fun_M2J))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
    def assemble_MvJ(self):
        """  Assembles mass matrix for L2-scalar product in V0vec weighted with cross product of MHD equilibrium current density.
        """
        if self.derham.comm.Get_rank() == 0: print('Assembling MvJ ...')
        self._MvJ = ApplyHomogeneousDirichletToOperator('Hdiv', 'H1vec', self.derham.bc, get_mass(self.derham.V0vec, self.derham.V2, self._fun_MvJ))
        if self.derham.comm.Get_rank() == 0: print('Done.')
        
        
    @property
    def M0(self):
        """ Mass matrix M0_(ijk lmn) = integral( Lambda^0_(ijk) * Lambda^0_(lmn) * sqrt(g) ). 
        """
        return self._M0
    
    @property
    def M1(self):
        """ Mass matrix M1_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * G_inv_ab * Lambda^1_(b,lmn) * sqrt(g) ). 
        """
        return self._M1
    
    @property
    def M2(self):
        """ Mass matrix M2_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * G_ab * Lambda^2_(b,lmn) / sqrt(g) ). 
        """
        return self._M2
    
    @property
    def M3(self):
        """ Mass matrix M3_(ijk lmn) = integral( Lambda^3_(ijk) * Lambda^3_(lmn) / sqrt(g) ). 
        """
        return self._M3
    
    @property
    def Mv(self):
        """ Mass matrix Mv_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * G_ab * Lambda^v_(b,lmn) * sqrt(g) ). 
        """
        return self._Mv
    
    @property
    def M1n(self):
        """ Mass matrix Mn1_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^1_(b,lmn) * sqrt(g) * n^0_eq * G_inv_ab ).
        """
        return self._M1n
    
    @property
    def M2n(self):
        """ Mass matrix M2n_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * n^0_eq * G_ab ).
        """
        return self._M2n
    
    @property
    def Mvn(self):
        """ Mass matrix Mvn_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^v_(b,lmn) * sqrt(g) * n^0_eq * G_ab ).
        """
        return self._Mvn
    
    @property
    def M1J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c * G_inv_ab ).
        """
        return self._M1J
    
    @property
    def M2J(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * epsilon_(acb) * J^2_eq_c).
        """
        return self._M2J
    
    @property
    def MvJ(self):
        """ Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c ).
        """
        return self._MvJ
    


def get_mass(V, W, weight=None):
    """
    Assembles the weighted mass matrix basis(V) * weight * basis(W). Works in 1d, 2d and 3d.
    
    Parameters
    ----------
        V : TensorFemSpace or ProductFemSpace
            tensor product spline space from psydac.fem.tensor (output space).
            
        W : TensorFemSpace or ProductFemSpace
            tensor product spline space from psydac.fem.tensor (input space).
            
        weight : list[callable], optional
            weight function(s) in a 2d list of shape corresponding to number of components of input/output space.

    Returns
    -------
        M : StencilMatrix of BlockMatrix
            weighted mass matrix.
    """
    
    assert isinstance(V, FemSpace)
    assert isinstance(W, FemSpace)
    
    # Output space: collect tensor fem spaces in a tuple
    if hasattr(V.symbolic_space, 'name'):
        if V.symbolic_space.name in {'H1', 'L2'}:
            Vspaces = (V,)
        else:
            Vspaces = V.spaces
        #print(f'to {V.symbolic_space.name} ...')
    else:
        Vspaces = V.spaces
        #print(f'to H1vec ...')

    # Input space: collect tensor fem spaces in a tuple
    if hasattr(W.symbolic_space, 'name'):
        if W.symbolic_space.name in {'H1', 'L2'}:
            Wspaces = (W,)
        else:
            Wspaces = W.spaces
        #print(f'... from {W.symbolic_space.name}.')
    else:
        Wspaces = W.spaces
        #print(f'... from H1vec.')
    
    blocks = []
    
    for a, vspace in enumerate(Vspaces):
        blocks += [[]]
        
        # periodicity: True (1) or False (0)
        periodic = [int(periodic) for periodic in vspace.periodic]

        # global element indices on process over which integration is performed
        el_loc_indices = [quad_grid.indices for quad_grid in vspace.quad_grids]

        # global start spline index on process
        starts_out = vspace.vector_space.starts

        # pads (ghost regions)
        pads_out = vspace.vector_space.pads

        # global quadrature points and weights in format (local element, local quad_point/weight)
        nq  = [quad_grid.num_quad_pts for quad_grid in vspace.quad_grids]
        pts = [quad_grid.points for quad_grid in vspace.quad_grids]
        wts = [quad_grid.weights for quad_grid in vspace.quad_grids]

        # evaluated basis functions at quadrature points
        basis_o = [quad_grid.basis for quad_grid in vspace.quad_grids]
            
        for b, wspace in enumerate(Wspaces):
                 
            # evaluation of weight function at quadrature points (optional)
            if V.ldim == 1:
                
                if weight is not None:
                    if weight[a][b] is not None:
                        PTS1, = np.meshgrid(pts[0].flatten(), indexing='ij')
                        mat_w = weight[a][b](PTS1).copy()
                        mat_w = mat_w.reshape(pts[0].shape[0], nq[0])
                    else:
                        mat_w = np.ones((pts[0].shape[0], nq[0]), dtype=float)
                else:
                    mat_w = np.ones((pts[0].shape[0], nq[0]), dtype=float)
                
            elif V.ldim == 2:
                
                if weight is not None:
                    if weight[a][b] is not None:
                        PTS1, PTS2 = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij')
                        mat_w = weight[a][b](PTS1, PTS2).copy()
                        mat_w = mat_w.reshape(pts[0].shape[0], nq[0], pts[1].shape[0], nq[1])
                    else:
                        mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1]), dtype=float)
                else:
                    mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1]), dtype=float)
                
            elif V.ldim == 3:
        
                if weight is not None:
                    if weight[a][b] is not None:
                        PTS1, PTS2, PTS3 = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij')
                        mat_w = weight[a][b](PTS1, PTS2, PTS3).copy()
                        mat_w = mat_w.reshape(pts[0].shape[0], nq[0], pts[1].shape[0], nq[1], pts[2].shape[0], nq[2])
                    else:
                        mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1], pts[2].shape[0], nq[2]), dtype=float)
                else:
                    mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1], pts[2].shape[0], nq[2]), dtype=float)

            basis_i = [quad_grid.basis for quad_grid in wspace.quad_grids]

            # assemble matrix if weight is not zero
            if np.any(mat_w):
                M = StencilMatrix(wspace.vector_space, vspace.vector_space, backend=PSYDAC_BACKEND_GPYCCEL)
                
                if V.ldim == 1:
                    
                    kernel_1d(el_loc_indices[0], vspace.degree[0], wspace.degree[0], periodic[0], int(starts_out[0]), pads_out[0], nq[0], wts[0], basis_o[0], basis_i[0], mat_w, M._data)
                    
                elif V.ldim == 2:
                    
                    kernel_2d(el_loc_indices[0], el_loc_indices[1], vspace.degree[0], vspace.degree[1], wspace.degree[0], wspace.degree[1], periodic[0], periodic[1], int(starts_out[0]), int(starts_out[1]), pads_out[0], pads_out[1], nq[0], nq[1], wts[0], wts[1], basis_o[0], basis_o[1], basis_i[0], basis_i[1], mat_w, M._data)
                
                elif V.ldim == 3:

                    kernel_3d(el_loc_indices[0], el_loc_indices[1], el_loc_indices[2], vspace.degree[0], vspace.degree[1], vspace.degree[2], wspace.degree[0], wspace.degree[1], wspace.degree[2], periodic[0], periodic[1], periodic[2], int(starts_out[0]), int(starts_out[1]), int(starts_out[2]), pads_out[0], pads_out[1], pads_out[2], nq[0], nq[1], nq[2], wts[0], wts[1], wts[2], basis_o[0], basis_o[1], basis_o[2], basis_i[0], basis_i[1], basis_i[2], mat_w, M._data)

                
                blocks[-1] += [M]
                
            else:
                blocks[-1] += [None]
                
    if len(blocks) == len(blocks[0]) == 1:
        M = blocks[0][0] 
    else:
        M = BlockMatrix(W.vector_space, V.vector_space, blocks)
        
    #M.update_ghost_regions()
    #M.remove_spurious_entries()
                
    return M