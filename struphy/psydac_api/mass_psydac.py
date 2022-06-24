import numpy as np

from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix

from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from struphy.psydac_api.mass_kernels_psydac import kernel_1d
from struphy.psydac_api.mass_kernels_psydac import kernel_2d
from struphy.psydac_api.mass_kernels_psydac import kernel_3d


class WeightedMass:
    """
    Class for assembling weighted mass matrices.
    
    Parameters
    ----------
        derham : Derham
            Discrete de rham sequence from struphy.psydac_api.psydac_derham.

        domain : Domain
            Mapped domain object from struphy.geometry.domain_3d.
    """
    
    def __init__(self, derham, domain):
        
        self._derham = derham
        self._domain = domain
    
    
    @property
    def derham(self):
        return self._derham
    
    @property
    def domain(self):
        return self._domain
    
    
    def assemble_M0(self):
        """  Assemble mass matrix for L2-scalar product in V0 without psydac's symbolic mapping.
        """

        metric = [[lambda e1, e2, e3 : abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]]
        
        self._M0 = get_mass(self.derham.V0, self.derham.V0, metric)
        
    def assemble_M1(self):
        """  Assemble mass matrix for L2-scalar product in V1 without psydac's symbolic mapping.
        """
        
        keys_metric = [['g_inv_11', 'g_inv_12', 'g_inv_13'],
                       ['g_inv_21', 'g_inv_22', 'g_inv_23'],
                       ['g_inv_31', 'g_inv_32', 'g_inv_33']]
        
        metric = []
        
        metric += [[]]
        
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][0])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][1])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][2])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][0])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][1])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][2])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][0])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][1])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][2])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        self._M1 = get_mass(self.derham.V1, self.derham.V1, metric)
        
    def assemble_M2(self):
        """  Assemble mass matrix for L2-scalar product in V2 without psydac's symbolic mapping.
        """
        
        keys_metric = [['g_11', 'g_12', 'g_13'],
                       ['g_21', 'g_22', 'g_23'],
                       ['g_31', 'g_32', 'g_33']]
        
        metric = []
        
        metric += [[]]
        
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][0])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][1])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][2])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][0])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][1])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][2])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][0])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][1])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][2])/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        self._M2 = get_mass(self.derham.V2, self.derham.V2, metric)
        
    def assemble_M3(self):
        """  Assemble mass matrix for L2-scalar product in V3 without psydac's symbolic mapping.
        """

        metric = [[lambda e1, e2, e3 : 1/abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]]
        
        self._M3 = get_mass(self.derham.V3, self.derham.V3, metric)
        
    def assemble_Mv(self):
        """  Assemble mass matrix for L2-scalar product in V0vec without psydac's symbolic mapping.
        """
        
        keys_metric = [['g_11', 'g_12', 'g_13'],
                       ['g_21', 'g_22', 'g_23'],
                       ['g_31', 'g_32', 'g_33']]
        
        metric = []
        
        metric += [[]]
        
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][0])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][1])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][2])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][0])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][1])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][2])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        metric += [[]]

        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][0])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][1])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        metric[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][2])*abs(self.domain.evaluate(e1, e2, e3, 'det_df'))]
        
        self._Mv = get_mass(self.derham.V0vec, self.derham.V0vec, metric)
              
    def assemble_Mn(self, eq_mhd, basis):
        """  
        Assembles mass matrix for L2-scalar product weighted with an MHD equilibrium number density profile.
        
        Parameters
        ----------
            eq_mhd : EquilibriumMHD
                An MHD equilibrium from struphy.field_equil.mhd_equil.mhd_equils.
                
            basis : string
                The input and output spaces (H1vec, Hcurl or Hdiv).
        """
        
        import operator
        
        if basis == 'Hcurl':
            keys_metric = [['g_inv_11', 'g_inv_12', 'g_inv_13'],
                           ['g_inv_21', 'g_inv_22', 'g_inv_23'],
                           ['g_inv_31', 'g_inv_32', 'g_inv_33']] 
        
        else:
            keys_metric = [['g_11', 'g_12', 'g_13'],
                           ['g_21', 'g_22', 'g_23'],
                           ['g_31', 'g_32', 'g_33']]
            
        if basis == 'Hdiv':
            op = operator.truediv
        else:
            op = operator.mul
        
        weight = []
        
        weight += [[]]
        
        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][0])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][1])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[0][2])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        
        weight += [[]]

        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][0])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][1])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[1][2])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        
        weight += [[]]

        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][0])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][1])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        weight[-1] += [lambda e1, e2, e3 : self.domain.evaluate(e1, e2, e3, keys_metric[2][2])*op(eq_mhd.n0_eq(e1, e2, e3), abs(self.domain.evaluate(e1, e2, e3, 'det_df')))]
        
        if   basis == 'H1vec':
            self._Mn = get_mass(self.derham.V0vec, self.derham.V0vec, weight)
        elif basis == 'Hcurl':
            self._Mn = get_mass(self.derham.V1, self.derham.V1, weight)
        elif basis == 'Hdiv':
            self._Mn = get_mass(self.derham.V2, self.derham.V2, weight)
            
            
    def assemble_MJ(self, eq_mhd, basis):
        """  
        Assembles mass matrix for L2-scalar product weighted with the cross product of an MHD equilibrium current density profile.
        
        Parameters
        ----------
            eq_mhd : EquilibriumMHD
                An MHD equilibrium from struphy.field_equil.mhd_equil.mhd_equils.
                
            basis : string
                The input and output spaces (H1vec, Hcurl or Hdiv).
        """
        
        if   basis == 'Hdiv':
            fun = lambda e1, e2, e3 : abs(self.domain.evaluate(e1, e2, e3, 'det_df'))
        elif basis == 'H1vec':
            fun = lambda e1, e2, e3 : 1 - 0*e1
        
        weight = []
        
        weight += [[]]
        
        weight[-1] += [lambda e1, e2, e3 : 0*eq_mhd.j2_eq_1(e1, e2, e3)/fun(e1, e2, e3)]
        weight[-1] += [lambda e1, e2, e3 :  -eq_mhd.j2_eq_3(e1, e2, e3)/fun(e1, e2, e3)]
        weight[-1] += [lambda e1, e2, e3 :   eq_mhd.j2_eq_2(e1, e2, e3)/fun(e1, e2, e3)]
        
        weight += [[]]

        weight[-1] += [lambda e1, e2, e3 :   eq_mhd.j2_eq_3(e1, e2, e3)/fun(e1, e2, e3)]
        weight[-1] += [lambda e1, e2, e3 : 0*eq_mhd.j2_eq_2(e1, e2, e3)/fun(e1, e2, e3)]
        weight[-1] += [lambda e1, e2, e3 :  -eq_mhd.j2_eq_1(e1, e2, e3)/fun(e1, e2, e3)]
        
        weight += [[]]

        weight[-1] += [lambda e1, e2, e3 :  -eq_mhd.j2_eq_2(e1, e2, e3)/fun(e1, e2, e3)]
        weight[-1] += [lambda e1, e2, e3 :   eq_mhd.j2_eq_1(e1, e2, e3)/fun(e1, e2, e3)]
        weight[-1] += [lambda e1, e2, e3 : 0*eq_mhd.j2_eq_3(e1, e2, e3)/fun(e1, e2, e3)]
        
        if   basis == 'H1vec':
            self._MJ = get_mass(self.derham.V0vec, self.derham.V2, weight)
        elif basis == 'Hdiv':
            self._MJ = get_mass(self.derham.V2, self.derham.V2, weight)
        
        
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
    def Mn(self):
        """ 
        1-form : Mass matrix Mn_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^1_(b,lmn) * sqrt(g) * n^0_eq * G_inv_ab ).
        2-form : Mass matrix Mn_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * n^0_eq * G_ab )
        vector : Mass matrix Mn_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^v_(b,lmn) * sqrt(g) * n^0_eq * G_ab )
        """
        return self._Mn
    
    @property
    def MJ(self):
        """ 
        1-form : Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^1_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c * G_inv_ab ).
        2-form : Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^2_(a,ijk) * Lambda^2_(b,lmn) / sqrt(g) * epsilon_(acb) * J^2_eq_c)
        vector : Mass matrix MJ_(ab, ijk lmn) = integral( Lambda^v_(a,ijk) * Lambda^2_(b,lmn) * epsilon_(acb) * J^2_eq_c )
        """
        return self._MJ
    


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
        print(f'to {V.symbolic_space.name} ...')
    else:
        Vspaces = V.spaces
        print(f'to H1vec ...')

    # Input space: collect tensor fem spaces in a tuple
    if hasattr(W.symbolic_space, 'name'):
        if W.symbolic_space.name in {'H1', 'L2'}:
            Wspaces = (W,)
        else:
            Wspaces = W.spaces
        print(f'... from {W.symbolic_space.name}.')
    else:
        Wspaces = W.spaces
        print(f'... from H1vec.')
    
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
                        PTS1 = np.meshgrid(pts[0].flatten(), indexing='ij')
                        mat_w = weight[a][b](PTS1)
                        mat_w = mat_w.reshape(pts[0].shape[0], nq[0])
                    else:
                        mat_w = np.ones((pts[0].shape[0], nq[0]), dtype=float)
                else:
                    mat_w = np.ones((pts[0].shape[0], nq[0]), dtype=float)
                
            elif V.ldim == 2:
                
                if weight is not None:
                    if weight[a][b] is not None:
                        PTS1, PTS2 = np.meshgrid(pts[0].flatten(), pts[1].flatten(), indexing='ij')
                        mat_w = weight[a][b](PTS1, PTS2)
                        mat_w = mat_w.reshape(pts[0].shape[0], nq[0], pts[1].shape[0], nq[1])
                    else:
                        mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1]), dtype=float)
                else:
                    mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1]), dtype=float)
                
            elif V.ldim == 3:
        
                if weight is not None:
                    if weight[a][b] is not None:
                        PTS1, PTS2, PTS3 = np.meshgrid(pts[0].flatten(), pts[1].flatten(), pts[2].flatten(), indexing='ij')
                        mat_w = weight[a][b](PTS1, PTS2, PTS3)
                        mat_w = mat_w.reshape(pts[0].shape[0], nq[0], pts[1].shape[0], nq[1], pts[2].shape[0], nq[2])
                    else:
                        mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1], pts[2].shape[0], nq[2]), dtype=float)
                else:
                    mat_w = np.ones((pts[0].shape[0], nq[0], pts[1].shape[0], nq[1], pts[2].shape[0], nq[2]), dtype=float)

            basis_i = [quad_grid.basis for quad_grid in wspace.quad_grids]

            # assemble matrix if weight is not zero
            if np.any(mat_w):
                M = StencilMatrix(wspace.vector_space, vspace.vector_space)
                
                if V.ldim == 1:
                    
                    kernel_1d(el_loc_indices[0], vspace.degree[0], wspace.degree[0], periodic[0], int(starts_out[0]), pads_out[0], nq[0], wts[0], basis_o[0], basis_i[0], mat_w, M._data)
                    
                elif V.ldim == 2:
                    
                    kernel_2d(el_loc_indices[0], el_loc_indices[1], vspace.degree[0], vspace.degree[1], wspace.degree[0], wspace.degree[1], periodic[0], periodic[1], int(starts_out[0]), int(starts_out[1]), pads_out[0], pads_out[1], nq[0], nq[1], wts[0], wts[1], basis_o[0], basis_o[1], basis_i[0], basis_i[1], mat_w, M._data)
                
                elif V.ldim == 3:

                    kernel_3d(el_loc_indices[0], el_loc_indices[1], el_loc_indices[2], vspace.degree[0], vspace.degree[1], vspace.degree[2], wspace.degree[0], wspace.degree[1], wspace.degree[2], periodic[0], periodic[1], periodic[2], int(starts_out[0]), int(starts_out[1]), int(starts_out[2]), pads_out[0], pads_out[1], pads_out[2], nq[0], nq[1], nq[2], wts[0], wts[1], wts[2], basis_o[0], basis_o[1], basis_o[2], basis_i[0], basis_i[1], basis_i[2], mat_w, M._data)

                #M.update_ghost_regions()
                
                blocks[-1] += [M]
                
            else:
                blocks[-1] += [None]
                
    if len(blocks) == len(blocks[0]) == 1:
        M = blocks[0][0] 
    else:
        M = BlockMatrix(W.vector_space, V.vector_space, blocks)
                
    return M