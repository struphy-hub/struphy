# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

import h5py
import numpy as np
from numpy.core.numeric import count_nonzero
import scipy.sparse as spa
from scipy.sparse.linalg import splu

import struphy.geometry.mappings_3d               as mapping
import struphy.geometry.pullback_3d               as pb
import struphy.geometry.pushforward_3d            as pf
import struphy.geometry.transform_3d              as tr
import struphy.geometry.angular_coordinates_torus as angular
import struphy.linear_algebra.linalg_kron as linalg

import struphy.feec.bsplines  as bsp

from sympde.topology import Mapping



# ==================================================
def spline_interpolation_nd(p, grids_1d, values):
    '''
    nd spline interpolation with discrete input (nonuniform).

    The knot vector for the clamped spline interpolant is constructed from grids_1d.

    Parameters
    -----------
        p : list 
            spline degree

        grids_1d : list of np.arrays
            interpolation points

        values: np.array
            function values at interpolation points. values.shape = (grid1.size, ..., gridn.size)

    Returns
    --------
        coeffs : np.array
            spline coefficients as nd array.

        T : list
            Knot vector of spline interpolant
    '''

    # dimension check
    for sh, x_grid in zip(values.shape, grids_1d):
        assert  sh == x_grid.size

    # list of break point arrays
    breaks = []

    for x_grid, p_i in zip(grids_1d, p):

        # dimension of the 1d spline spaces: dim = breaks.size - 1 + p = x_grid.size
        if p_i == 1:
            breaks.append(x_grid)
        elif p_i%2 == 0:
            breaks.append( x_grid[p_i//2 - 1:-p_i//2].copy() )
        else:
            breaks.append( x_grid[(p_i - 1)//2:-(p_i - 1)//2].copy() )

        # cells must be in interval [0, 1] 
        breaks[-1][0]  = 0.
        breaks[-1][-1] = 1.

    # interpolation with clamped splines (periodic=False)
    T     = [bsp.make_knots(breaks_i, p_i, periodic=False) for breaks_i, p_i in zip(breaks, p)]
    I_mat = [bsp.collocation_matrix(T_i, p_i, grids_1d_i, periodic=False) for T_i, p_i, grids_1d_i in zip(T, p, grids_1d)]

    I_LU  = [splu(spa.csc_matrix(I_mat_i)) for I_mat_i in I_mat] 

    # dimension check
    for I, x_grid in zip(I_mat, grids_1d):
        assert I.shape[0] == x_grid.size
        assert I.shape[0] == I.shape[1]

    # solve system
    if len(p) == 1:
        return I_LU[0].solve(values), T
    if len(p) == 2:
        return linalg.kron_lusolve_2d(I_LU, values), T
    elif len(p) == 3:
        return linalg.kron_lusolve_3d(I_LU, values), T
    else:
        raise AssertionError("Only dimensions < 4 are supported.")



# ==================================================
def interp_mapping(Nel, p, spl_kind, X, Y, Z=None):
    '''
    Interpolates the mapping (eta1, eta2, eta3) --> (X, Y, Z) on the given spline space.

    Parameters
    -----------
        Nel, p, spl_kind: array-like
            defining the spline space

        X, Y: callable
            either X(eta1, eta2) in 2D or X(eta1, eta2, eta3) in 3D

        Z: callable Z(eta1, eta2, eta3)

    Returns
    --------
        cx, cy (, cz): np.array
            spline coefficients
    '''
    
    # number of basis functions
    NbaseN = [Nel + p - kind*p for Nel, p, kind in zip(Nel, p, spl_kind)]
    
    # element boundaries
    el_b   = [np.linspace(0., 1., Nel + 1) for Nel in Nel]
    
    # spline knot vectors
    T      = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(el_b, p, spl_kind)]
    
    # greville points
    I_pts  = [bsp.greville(T, p, kind) for T, p, kind in zip(T, p, spl_kind)]
    
    # 1D interpolation matrices
    I_mat  = [spa.csc_matrix(bsp.collocation_matrix(T, p, I_pts, kind)) for T, p, I_pts, kind in zip(T, p, I_pts, spl_kind)]
    
    # 2D interpolation
    if len(Nel) == 2:
        I = spa.kron(I_mat[0], I_mat[1], format='csc')
        
        I_pts = np.meshgrid(I_pts[0], I_pts[1], indexing='ij')
        
        cx = spa.linalg.spsolve(I, X(I_pts[0], I_pts[1]).flatten()).reshape(NbaseN[0], NbaseN[1])
        cy = spa.linalg.spsolve(I, Y(I_pts[0], I_pts[1]).flatten()).reshape(NbaseN[0], NbaseN[1])
        
        return cx, cy
    
    # 3D interpolation
    elif len(Nel) == 3:
        I = spa.kron(I_mat[0], spa.kron(I_mat[1], I_mat[2]), format='csc')
        
        I_pts = np.meshgrid(I_pts[0], I_pts[1], I_pts[2], indexing='ij')
        
        cx = spa.linalg.spsolve(I, X(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(NbaseN[0], NbaseN[1], NbaseN[2])
        cy = spa.linalg.spsolve(I, Y(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(NbaseN[0], NbaseN[1], NbaseN[2])
        cz = spa.linalg.spsolve(I, Z(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(NbaseN[0], NbaseN[1], NbaseN[2])
        
        return cx, cy, cz
    
    else:
        print('wrong number of elements')
        
        return 0.


def prepare_args(x, y, z, flat_eval=False):
    '''Broadcast point sets to correct size for evaluation.
    
    Parameters
    ----------
        x, y, z : float or list or np.array
            Evaluation point sets.
        flat_eval : boolean
            Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)]. 

    Returns
    -------
        E1, E2, E3 : np.arrays
            3d arrays, except for flat_eval=True (1d arrays).

        is_sparse_meshgrid : boolean
            Whether arguments fit sparse_meshgrid shape.
    '''

    # convert float, list type data to numpy array:
    if isinstance(x, float):
        arg_x = np.array([x])
    elif isinstance(x, list):
        arg_x = np.array(x) 
    elif isinstance(x, np.ndarray):
        arg_x = x 
    else:
        print('data type not supported')

    if isinstance(y, float):
        arg_y = np.array([y])
    elif isinstance(y, list):
        arg_y = np.array(y) 
    elif isinstance(y, np.ndarray):
        arg_y = y
    else:
        print('data type not supported')

    if isinstance(z, float):
        arg_z = np.array([z])
    elif isinstance(z, list):
        arg_z = np.array(z) 
    elif isinstance(z, np.ndarray):
        arg_z = z
    else:
        print('data type not supported')

    is_sparse_meshgrid = False

    # flat evaluation
    if flat_eval:
        assert arg_x.ndim == arg_y.ndim == arg_z.ndim == 1
        assert arg_x.size == arg_y.size == arg_z.size

        E1 = arg_x
        E2 = arg_y
        E3 = arg_z

        return E1, E2, E3, is_sparse_meshgrid

    # broadcast to 3d arrays
    else:
        # tensor-product for given three 1D arrays 
        if arg_x.ndim == 1 and arg_y.ndim == 1 and arg_z.ndim == 1:
            #assert arg_x.ndim == arg_y.ndim == arg_z.ndim == 1
            E1, E2, E3 = np.meshgrid(arg_x, arg_y, arg_z, indexing='ij')
        # given xy-plane at point z:
        elif arg_x.ndim == 2 and arg_y.ndim == 2 and arg_z.size == 1:
            E1 = arg_x[:, :, None]
            E2 = arg_y[:, :, None]
            E3 = arg_z*np.ones(E1.shape)
        # given xz-plane at point y:
        elif arg_x.ndim == 2 and arg_y.size == 1 and arg_z.ndim == 2:
            E1 = arg_x[:, None, :]
            E2 = arg_y*np.ones(E1.shape)
            E3 = arg_z[:, None, :]
        # given yz-plane at point x:
        elif arg_x.size == 1 and arg_y.ndim == 2 and arg_z.ndim == 2:
            E2 = arg_y[None, :, :]
            E3 = arg_z[None, :, :]
            E1 = arg_x*np.ones(E2.shape)
        # given three 3D arrays 
        elif arg_x.ndim == 3 and arg_y.ndim == 3 and arg_z.ndim == 3:
            # Distinguish if input coordinates are from sparse or dense meshgrid.
            # Sparse: arg_x.shape = (n1, 1, 1), arg_y.shape = (1, n2, 1), arg_z.shape = (1, 1, n3)
            # Dense : arg_x.shape = (n1, n2, n3), arg_y.shape = (n1, n2, n3) arg_z.shape = (n1, n2, n3)
            E1, E2, E3 = arg_x, arg_y, arg_z

            # `arg_x` `arg_y` `arg_z` are all sparse meshgrids.
            if max(arg_x.shape) == arg_x.size or max(arg_y.shape) == arg_y.size or max(arg_z.shape) == arg_z.size:
                assert max(arg_x.shape) == arg_x.size
                assert max(arg_y.shape) == arg_y.size
                assert max(arg_z.shape) == arg_z.size
                is_sparse_meshgrid = True
            # one of `arg_x` `arg_y` `arg_z` is a dense meshgrid.(i.e., all are dense meshgrid) Process each point as default.

        else:
            raise ValueError('Argument dimensions not supported')

        return E1, E2, E3, is_sparse_meshgrid


# ==================================================
class Domain():
    '''Defines the mapped domain.

    Parameters
    ----------
    kind_map : str
        Type of domain.
    
    params_map: dict
        The parameters needed to define the mappings (see Notes).

    Attributes
    ----------
    kind_map: int
        values <10 indicate a spline mapping

    params_map: array-like
        mapping parameters

    Nel: list
        1d number of elements of discrete spline mapping
    
    p: list
        1d degrees of discrete spline mapping
    
    NbaseN: list
        1d dimensions of discrete spline mapping
    
    T: list
        1d knot vectors of discrete spline mapping

    cx: np.array
        spline coefficients of X(eta1, eta2, eta3)
    
    cy: np.array
        spline coefficients of Y(eta1, eta2, eta3)
    
    cz: np.array
        spline coefficients of Z(eta1, eta2, eta3)

    keys_map: dict
        keys point to values for kind_fun in the 'evaluate' method.

    keys_pull: dict
        keys point to possible values for kind_fun in 'pull' method.

    keys_push: dict
        keys point to possible values for kind_fun in 'push' method.

    Notes
    -----
    Available mappings (choices for "kind_map") are:

        * 'cuboid' :       
            * X = l1 + (r1 - l1)*eta1
            * Y = l2 + (r2 - l2)*eta2
            * Z = l3 + (r3 - l3)*eta3   
        * 'orthogonal' :   
            * X = Lx*(eta1 + alpha*sin(2*pi*eta1))
            * Y = Ly*(eta2 + alpha*sin(2*pi*eta2))
            * Z = Lz*eta3
        * 'colella' :   
            * X = Lx*(eta1 + alpha*sin(2*pi*eta1)*sin(2*pi*eta2))
            * Y = Ly*(eta2 + alpha*sin(2*pi*eta1)*sin(2*pi*eta2))
            * Z = Lz*eta3
        * 'hollow_cyl' :   
            * X = (a1 + (a2 - a1)*eta1)*cos(2*pi*eta2) + R0
            * Y = (a1 + (a2 - a1)*eta1)*sin(2*pi*eta2)
            * Z = 2*pi*R0*eta3
        * 'hollow_torus' : 
            * X = X_hollow_cyl * cos(2*pi*eta3)
            * Y = Y_hollow_cyl
            * Z = X_hollow_cyl * sin(2*pi*eta3)
        * 'ellipse' :
            * X = x0 + (eta1*rx) * cos(2*pi*eta2)
            * Y = y0 + (eta1*ry) * sin(2*pi*eta2)
            * Z = z0 + (eta3*Lz)
        * 'rotated_ellipse' :
            * X = x0 + (eta1*r1) * cos(2*pi*th) * cos(2*pi*eta2) - (eta1*r2) * sin(2*pi*th) * sin(2*pi*eta2)
            * Y = y0 + (eta1*r1) * sin(2*pi*th) * cos(2*pi*eta2) + (eta1*r2) * cos(2*pi*th) * sin(2*pi*eta2)
            * Z = z0 + (eta3*Lz)
        * 'soloviev_approx' :
            * X = x0 + (eta1*rx) * cos(2*pi*eta2) + (1 - eta1**2) * rx * delta
            * Y = y0 + (eta1*ry) * sin(2*pi*eta2)
            * Z = z0 + (eta3*Lz)
        * 'soloviev_sqrt' :
            * Crafted s.t. derivative component 11 does not go to zero at the pole of the map.
            * X = x0 + (eta1*rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * delta
            * Y = y0 + (eta1*ry) * sin(2*pi*eta2)
            * Z = z0 + (eta3*Lz)
        * 'soloviev_cf' :
            * Soloviev equilibrium as described by Cerfon and Freiberg (doi: 10.1063/1.3328818).
            * X = x0 + R0 * [ 1 + (1 - eta1**2) * delta_x + eta1 * epsilon_gs * cos(2*pi*eta2 + arcsin(delta_gs)*eta1*sin(2*pi*eta2)) ]
            * Y = y0 + R0 * [     (1 - eta1**2) * delta_y + eta1 * epsilon_gs * kappa_gs * sin(2*pi*eta2) ]
            * Z = z0 + (eta3*Lz)
        * 'spline': 
            * 3d discrete spline mapping. All information is stored in control points cx, cy, cz.
        * 'spline_cyl': 
            * 2d discrete spline mapping in (eta1, eta2) --> (X, Y) w/ control points cx, cy, cz=None and
            * X = a*eta1*np.cos(2*np.pi*eta2)
            * Y = a*eta1*np.sin(2*np.pi*eta2)
            * Z = Lz*eta3
        * 'spline_torus' : 
            * 2d discrete spline mapping in (eta1, eta2) --> (R, Y) w/ control points cx, cy, cz=None and
            * X = R*cos(2*pi*eta3) = (a*eta1*np.cos(2*np.pi*eta2) + R0)*cos(2*pi*eta3) 
            * Y = Y                =  a*eta1*np.sin(2*np.pi*eta2)
            * Z = R*sin(2*pi*eta3) = (a*eta1*np.cos(2*np.pi*eta2) + R0)*sin(2*pi*eta3) 
    '''

    def __init__(self, kind_map='cuboid', params_map={'l1': 0., 'r1': 1., 'l2': 0., 'r2': 1., 'l3': 0., 'r3': 1.}): 

        if kind_map == 'cuboid':
            self.kind_map = 10
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'l1 + (r1 - l1)*x1',
                                                'y': 'l2 + (r2 - l2)*x2',
                                                'z': 'l3 + (r3 - l3)*x3'}
            # In future versions you only need to specify the Psydac_mapping expressions.

        elif kind_map == 'orthogonal':
            self.kind_map = 13
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1))',
                                                'y': 'Ly*(x2 + alpha*sin(2*pi*x2))',
                                                'z': 'Lz*x3'}

        elif kind_map == 'colella':
            self.kind_map = 12
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                                'y': 'Ly*(x2 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                                'z': 'Lz*x3'}

        elif kind_map == 'hollow_cyl':
            self.kind_map = 11
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': '(a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0',
                                                'y': '(a1 + (a2 - a1)*x1)*sin(2*pi*x2)',
                                                'z': '2*pi*R0*x3'}

        elif kind_map == 'hollow_torus':
            self.kind_map = 14
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * cos(2*pi*x3)',
                                                'y': '(a1 + (a2 - a1)*x1)*sin(2*pi*x2)',
                                                'z': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * sin(2*pi*x3)'}

        elif kind_map == 'ellipse':
            self.kind_map = 15
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2)',
                                                'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}

        elif kind_map == 'rotated_ellipse':
            self.kind_map = 16
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*r1) * cos(2*pi*th) * cos(2*pi*x2) - (x1*r2) * sin(2*pi*th) * sin(2*pi*x2)',
                                                'y': 'y0 + (x1*r1) * sin(2*pi*th) * cos(2*pi*x2) + (x1*r2) * cos(2*pi*th) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}

        elif kind_map == 'soloviev_approx':
            self.kind_map = 17
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2) + (1-x1**2) * rx * delta',
                                                'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}

        elif kind_map == 'soloviev_sqrt':
            self.kind_map = 18
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2) + (1-sqrt(x1)) * rx * delta',
                                                'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}

        elif kind_map == 'soloviev_cf':
            self.kind_map = 19
            self.params_map = list(params_map.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + R0 * ( 1 + (1 - x1**2) * delta_x + x1 * epsilon_gs * cos(2*pi*x2 + asin(delta_gs)*x1*sin(2*pi*x2)) )',
                                                'y': 'y0 + R0 * (     (1 - x1**2) * delta_y + x1 * epsilon_gs * kappa_gs * sin(2*pi*x2) )',
                                                'z': 'z0 + (x3*Lz)'}

        elif kind_map == 'spline':
            # TODO: choose correct params_map
            self.kind_map   = 0
            self.params_map = []

            # print(f'Before popping: list(params_map.values()): {list(params_map.values())}')

            with h5py.File(params_map['file'], 'r') as handle:

                # print(f'Available keys: {tuple(handle.keys())}')
                self.cx = handle['cx'][:]
                self.cy = handle['cy'][:]
                self.cz = handle['cz'][:]

        elif kind_map == 'spline_cyl':
            self.kind_map   = 1
            self.params_map = []

            X = lambda eta1, eta2 : params_map['a']*eta1*np.cos(2*np.pi*eta2) + params_map['R0']
            Y = lambda eta1, eta2 : params_map['a']*eta1*np.sin(2*np.pi*eta2)
        
            self.cx, self.cy = interp_mapping(params_map['Nel'], params_map['p'], params_map['spl_kind'], X, Y)
                
        elif kind_map == 'spline_torus':
            self.kind_map   = 2
            self.params_map = []

            R = lambda eta1, eta2 : params_map['a']*eta1*np.cos(angular.theta(eta1, eta2, params_map['a'], params_map['R0'])) + params_map['R0']
            Y = lambda eta1, eta2 : params_map['a']*eta1*np.sin(angular.theta(eta1, eta2, params_map['a'], params_map['R0']))
        
            self.cx, self.cy = interp_mapping(params_map['Nel'], params_map['p'], params_map['spl_kind'], R, Y)

        else:
            raise ValueError('Specified domain is not implemeted!')

        # other attributes of spline mappings:
        if self.kind_map < 10:

            self.Nel      = params_map['Nel']
            self.p        = params_map['p']
            self.spl_kind = params_map['spl_kind']
            
            self.NbaseN = [Nel + p - kind*p for Nel, p, kind in zip(params_map['Nel'], 
                                                                    params_map['p'], 
                                                                    params_map['spl_kind'])]
            
            el_b        = [np.linspace(0., 1., Nel + 1) for Nel in params_map['Nel']]
            self.T      = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(el_b, 
                                                                                  params_map['p'],
                                                                                  params_map['spl_kind'])]
            
            # adapt 2d spline mappings to 3d
            if self.kind_map != 0:
                
                self.Nel    = self.Nel    + [0]
                self.p      = self.p      + [0]
                self.NbaseN = self.NbaseN + [0]
                
                self.T      = self.T      + [np.zeros((1,), dtype=float)]
                
                # make sure that control points at pole are all the same
                self.cx[0, :] = params_map['R0']
                self.cy[0, :] = 0.
                # self.cx     = self.cx.reshape(self.cx.shape[0], self.cx.shape[1], 1)
                # self.cy     = self.cy.reshape(self.cy.shape[0], self.cy.shape[1], 1)
                self.cx     = self.cx[:, :, None]
                self.cy     = self.cy[:, :, None]
                self.cz = np.zeros((1, 1, 1), dtype=float)
                   
        # create dummy attributes for analytical mappings
        else:
            
            self.Nel    = [0, 0, 0]
            self.p      = [0, 0, 0]
            self.NbaseN = [0, 0, 0]
            
            self.T      = [np.zeros((1,), dtype=float),
                           np.zeros((1,), dtype=float), 
                           np.zeros((1,), dtype=float)]

            self.cx     =  np.zeros((1, 1, 1), dtype=float)
            self.cy     =  np.zeros((1, 1, 1), dtype=float)
            self.cz     =  np.zeros((1, 1, 1), dtype=float)

   
        # keys for evaluating mapping related quantities
        self.keys_map  = {
            'x' : 1, 'y' : 2, 'z' : 3, 'det_df' : 4, 
            'df_11' : 11, 'df_12' : 12, 'df_13' : 13, 
            'df_21' : 14, 'df_22' : 15, 'df_23' : 16, 
            'df_31' : 17, 'df_32' : 18, 'df_33' : 19, 
            'df_inv_11' : 21, 'df_inv_12' : 22, 'df_inv_13' : 23,
            'df_inv_21' : 24, 'df_inv_22' : 25, 'df_inv_23' : 26, 
            'df_inv_31' : 27, 'df_inv_32' : 28, 'df_inv_33' : 29, 
            'g_11' : 31, 'g_12' : 32, 'g_13' : 33, 'g_21' : 34,
            'g_22' : 35, 'g_23' : 36, 'g_31' : 37, 'g_32' : 38, 'g_33' : 39, 
            'g_inv_11' : 41, 'g_inv_12' : 42, 'g_inv_13' : 43, 'g_inv_21' : 44,
            'g_inv_22' : 45, 'g_inv_23' : 46, 'g_inv_31' : 47, 'g_inv_32' : 48,
            'g_inv_33' : 49
            }
        
        # keys for performing pull-backs
        self.keys_pull = {
            '0_form' : 0, '3_form' : 3, '1_form_1' : 11, '1_form_2' : 12, 
            '1_form_3' : 13, '2_form_1' : 21, '2_form_2' : 22, '2_form_3' : 23,
            'vector_1' : 31, 'vector_2' : 32, 'vector_3' : 33
            }
        
        # keys for performing push-forwards
        self.keys_push = {
            '0_form' : 0, '3_form' : 3, '1_form_1' : 11, '1_form_2' : 12, 
            '1_form_3' : 13, '2_form_1' : 21, '2_form_2' : 22, '2_form_3' : 23, 
            'vector_1' : 31, 'vector_2' : 32, 'vector_3' : 33
            }

        # keys for performing transformation
        self.keys_transform = {
            'norm_to_0' : 0, 'norm_to_3' : 3,
            'norm_to_1_1' : 11, 'norm_to_1_2' : 12, 'norm_to_1_3' : 13,
            'norm_to_2_1' : 21, 'norm_to_2_2' : 22, 'norm_to_2_3' : 23,
            'norm_to_vector_1' : 31, 'norm_to_vector_2' : 32, 'norm_to_vector_3' : 33,
            '1_to_1_1' : 41, '1_to_1_1' : 42, '1_to_1_1' : 43,
            '1_to_1_1' : 51, '1_to_1_1' : 52, '1_to_1_1' : 53,
            '0_to_3' : 4, '3_to_0' : 5}
       

    class Psydac_mapping(Mapping):
        '''To create a psydac domain.'''

        _expressions = None
        _ldim        = 3
        _pdim        = 3   


    def evaluate(self, eta1, eta2, eta3, kind_fun, flat_eval=False, squeeze_output=True):
        '''Evaluate mapping/metric coefficients. 

        Depending on the dimension of eta1, eta2, eta3 either point-wise, tensor-product, slice plane or general (see prepare_args).

        Parameters
        -----------
            eta1, eta2, eta3 : point like or array-like or list like 
                logical coordinates at which to evaluate
            kind_fun : string
                what metric coefficient to evaluate, see Notes
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)]. 
            squeeze_output : boolean
                Whether to remove singleton dimensions in output "values". 

        Returns
        --------
            values: np.array
                mapping/metric coefficients evaluated at (eta1, eta2, eta3) 

        Notes
        -----
            Possible choices for kind_fun: 

                * 'x', 'y', 'z': components of F
                * 'det_df': Jacobian determinant
                * 'df_11', 'df_12', 'df_13', 'df_21', 'df_22', 'df_23', 'df_31', 'df_32', 'df_33': Jacobian 
                * 'df_inv_11', 'df_inv_12', 'df_inv_13', 'df_inv_21', 'df_inv_22', 'df_inv_23', 'df_inv_31', 'df_inv_32', 'df_inv_33', Jacobian inverse 
                * 'g_11', 'g_12', 'g_13', 'g_21', 'g_22', 'g_23', 'g_31', 'g_32', 'g_33': metric tensor 
                * 'g_inv_11', 'g_inv_12', 'g_inv_13', 'g_inv_21', 'g_inv_22', 'g_inv_23', 'g_inv_31', 'g_inv_32', 'g_inv_33': inverse metric tensor
        '''

        # # evaluation for point pairs of 1d np.arrays of same length at three directions
        # if flat_eval:
        #     assert eta1.ndim == eta2.ndim == eta3.ndim == 1
        #     assert eta1.size == eta2.size == eta3.size
            
        #     values = np.empty(eta1.size, dtype=float)
            
        #     mapping.kernel_evaluate_flat(eta1, eta2, eta3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
            
        #     return values
        
        # else:

        E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if is_sparse_meshgrid:
            mapping.kernel_evaluate_sparse(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        elif flat_eval:
            mapping.kernel_evaluate_flat(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        else:
            mapping.kernel_evaluate(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)

        if squeeze_output:
            values = values.squeeze()

        return values

       
    # ================================
    def pull(self, a, eta1, eta2, eta3, kind_fun='0-form', flat_eval=False):
        '''Pullback of p-forms. 

        Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see prepare_args).

        Parameters
        ----------
            a:  callable or array-like
                The function a(x, y, z) to be pulled back (as list for components of 1- and 2-forms). 
            eta1, eta2, eta3:   array-like
                Logical coordinates to which to pull back
            kind_fun:   str
                Which p-form pull back to apply, see keys_pull
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)]. 

        Returns
        -------
            values: np.array
                Pullback of p-form (component) evaluated at (eta1, eta2, eta3)

        Notes
        -----
            Possible choices for kind_fun:
                
                * '0_form', '3_form'
                * '1_form_1', '1_form_2', '1_form_3'
                * '2_form_1', '2_form_2', '2_form_3',
                * 'vector_1', 'vector_2', 'vector_3'
        '''

        E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if isinstance(a, list):
            
            if callable(a[0]):
                
                X = self.evaluate(E1, E2, E3, 'x', flat_eval, squeeze_output=False)
                Y = self.evaluate(E1, E2, E3, 'y', flat_eval, squeeze_output=False)
                Z = self.evaluate(E1, E2, E3, 'z', flat_eval, squeeze_output=False)
                
                a_in = np.array([a[0](X, Y, Z), a[1](X, Y, Z), a[2](X, Y, Z)])
            else:
                a_in = np.array(a)
                
        else:
            
            if callable(a):
                
                X = self.evaluate(E1, E2, E3, 'x', flat_eval, squeeze_output=False)
                Y = self.evaluate(E1, E2, E3, 'y', flat_eval, squeeze_output=False)
                Z = self.evaluate(E1, E2, E3, 'z', flat_eval, squeeze_output=False)
                
                a_in = np.array([a(X, Y, Z)])
            else:
                a_in = np.array([a])

        if is_sparse_meshgrid:
            pb.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        elif flat_eval:
            pb.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        else:
            pb.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)

        return values.squeeze()
        
    
    # ================================
    def push(self, a, eta1, eta2, eta3, kind_fun='0-form', flat_eval=False):
        '''Push-forward of p-forms. 

        Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see prepare_args).
        
        Parameters
        -----------
            a:  callable or array-like
                The function a(eta1, eta2, eta3) to be pushed forward (as list for components of 1- and 2-forms).
            eta1, eta2, eta3:   array-like
                Logical coordinates at which to push forward
            kind_fun:   str
                Which p-form push forward to apply, see keys_push
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)]. 

        Returns
        --------
            values: ndarray
                Push forward of p-form (component) evaluated at (eta1, eta2, eta3)

        Notes
        -----
            Possible choices for kind_fun:
                
                * '0_form', '3_form'
                * '1_form_1', '1_form_2', '1_form_3'
                * '2_form_1', '2_form_2', '2_form_3',
                * 'vector_1', 'vector_2', 'vector_3'
        '''
        
        E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if isinstance(a, list):
            if callable(a[0]):
                a_in = np.array([a[0](E1, E2, E3), a[1](E1, E2, E3), a[2](E1, E2, E3)])
            else:
                a_in = np.array(a)
                
        else:
            if callable(a):
                a_in = np.array([a(E1, E2, E3)])
            else:
                a_in = np.array([a])

        if is_sparse_meshgrid:
            pf.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        elif flat_eval:
            pf.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        else:
            pf.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
    
        return values.squeeze()


    # ================================
    def transformation(self, a, eta1, eta2, eta3, kind_fun='norm_to_0', flat_eval=False):
        '''Transformation between different p-forms on logical domain. 
        
        Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see prepare_args).

        Parameters
        ----------
            a:  callable or array-like
                the function a(eta1, eta2, eta3) to be transformed
            eta1, eta2, eta3:   array-like
                logical coordinates to which to transform
            kind_fun:   str
                which transform to apply, see keys_transform
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)]. 

        Returns
        -------
            values: ndarray
                transformed p-form from norm_vector or scalar (component) evaluated at (eta1, eta2, eta3)

        Notes
        -----
            Possible choices for kind_fun:

                * 'norm_to_0', 'norm_to_3',
                * 'norm_to_1_1', 'norm_to_1_2', 'norm_to_1_3',
                * 'norm_to_2_1', 'norm_to_2_2', 'norm_to_2_3',
                * 'norm_to_vector_1', 'norm_to_vector_2', 'norm_to_vector_3',
                * '1_to_1_1', '1_to_1_1', '1_to_1_1',
                * '1_to_1_1', '1_to_1_1', '1_to_1_1',
                * '0_to_3', '3_to_0'
        '''
        
        E1, E2, E3, is_sparse_meshgrid = prepare_args(eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if isinstance(a, list):
            if callable(a[0]):
                a_in = np.array([a[0](E1, E2, E3), a[1](E1, E2, E3), a[2](E1, E2, E3)])
            else:
                a_in = np.array(a)
                
        else:
            if callable(a):
                a_in = np.array([a(E1, E2, E3)])
            else:
                a_in = np.array([a])

        if is_sparse_meshgrid:
            tr.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        elif flat_eval:
            tr.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)
        else:
            tr.kernel_evaluate(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], self.p, self.NbaseN, self.cx, self.cy, self.cz, values)

        return values.squeeze()


