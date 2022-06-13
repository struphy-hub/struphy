import numpy as np

from psydac.linalg.stencil import StencilMatrix
from psydac.linalg.block import BlockMatrix

from psydac.fem.basic import FemSpace
from psydac.fem.tensor import TensorFemSpace
from psydac.fem.vector import ProductFemSpace

from struphy.psydac_api.mass_kernels_psydac import kernel_1d
from struphy.psydac_api.mass_kernels_psydac import kernel_2d
from struphy.psydac_api.mass_kernels_psydac import kernel_3d


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