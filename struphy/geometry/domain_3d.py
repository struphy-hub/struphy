# coding: utf-8
#
# Copyright 2021 Florian Holderied (florian.holderied@ipp.mpg.de)

import h5py
import numpy as np

import scipy.sparse as spa
from scipy.sparse.linalg import splu

from struphy.geometry.angular_coordinates_torus import theta

import struphy.geometry.mappings_3d as mapping
import struphy.geometry.pullback_3d as pb
import struphy.geometry.pushforward_3d as pf
import struphy.geometry.transform_3d as tr

import struphy.linear_algebra.linalg_kron as linalg

import struphy.feec.bsplines as bsp

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
        assert sh == x_grid.size

    # list of break point arrays
    breaks = []

    for x_grid, p_i in zip(grids_1d, p):

        # dimension of the 1d spline spaces: dim = breaks.size - 1 + p = x_grid.size
        if p_i == 1:
            breaks.append(x_grid)
        elif p_i % 2 == 0:
            breaks.append(x_grid[p_i//2 - 1:-p_i//2].copy())
        else:
            breaks.append(x_grid[(p_i - 1)//2:-(p_i - 1)//2].copy())

        # cells must be in interval [0, 1]
        breaks[-1][0] = 0.
        breaks[-1][-1] = 1.

    # interpolation with clamped splines (periodic=False)
    T = [bsp.make_knots(breaks_i, p_i, periodic=False)
         for breaks_i, p_i in zip(breaks, p)]
    
    indN = [(np.indices((breaks_i.size - 1, p_i + 1))[1] + np.arange(breaks_i.size - 1)[:, None])%grids_1d_i.size for breaks_i, p_i, grids_1d_i in zip(breaks, p, grids_1d)]

    I_mat = [bsp.collocation_matrix(T_i, p_i, grids_1d_i, periodic=False)
             for T_i, p_i, grids_1d_i in zip(T, p, grids_1d)]

    I_LU = [splu(spa.csc_matrix(I_mat_i)) for I_mat_i in I_mat]

    # dimension check
    for I, x_grid in zip(I_mat, grids_1d):
        assert I.shape[0] == x_grid.size
        assert I.shape[0] == I.shape[1]

    # solve system
    if len(p) == 1:
        return I_LU[0].solve(values), T, indN
    if len(p) == 2:
        return linalg.kron_lusolve_2d(I_LU, values), T, indN
    elif len(p) == 3:
        return linalg.kron_lusolve_3d(I_LU, values), T, indN
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
    el_b = [np.linspace(0., 1., Nel + 1) for Nel in Nel]

    # spline knot vectors
    T = [bsp.make_knots(el_b, p, kind)
         for el_b, p, kind in zip(el_b, p, spl_kind)]

    # greville points
    I_pts = [bsp.greville(T, p, kind) for T, p, kind in zip(T, p, spl_kind)]

    # 1D interpolation matrices
    I_mat = [spa.csc_matrix(bsp.collocation_matrix(T, p, I_pts, kind))
             for T, p, I_pts, kind in zip(T, p, I_pts, spl_kind)]

    # 2D interpolation
    if len(Nel) == 2:
        I = spa.kron(I_mat[0], I_mat[1], format='csc')

        I_pts = np.meshgrid(I_pts[0], I_pts[1], indexing='ij')

        cx = spa.linalg.spsolve(I, X(I_pts[0], I_pts[1]).flatten()).reshape(
            NbaseN[0], NbaseN[1])
        cy = spa.linalg.spsolve(I, Y(I_pts[0], I_pts[1]).flatten()).reshape(
            NbaseN[0], NbaseN[1])

        return cx, cy

    # 3D interpolation
    elif len(Nel) == 3:
        I = spa.kron(I_mat[0], spa.kron(I_mat[1], I_mat[2]), format='csc')

        I_pts = np.meshgrid(I_pts[0], I_pts[1], I_pts[2], indexing='ij')

        cx = spa.linalg.spsolve(I, X(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(
            NbaseN[0], NbaseN[1], NbaseN[2])
        cy = spa.linalg.spsolve(I, Y(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(
            NbaseN[0], NbaseN[1], NbaseN[2])
        cz = spa.linalg.spsolve(I, Z(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(
            NbaseN[0], NbaseN[1], NbaseN[2])

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

    params_map: dict, optional
        The parameters that define the mapping (see Notes). If not given, default values shall be used

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
        * 'shafranov_shift' :
            * X = x0 + (eta1*rx) * cos(2*pi*eta2) + (1 - eta1**2) * rx * delta
            * Y = y0 + (eta1*ry) * sin(2*pi*eta2)
            * Z = z0 + (eta3*Lz)
        * 'shafranov_sqrt' :
            * Crafted s.t. derivative component 11 does not go to zero at the pole of the map.
            * X = x0 + (eta1*rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * delta
            * Y = y0 + (eta1*ry) * sin(2*pi*eta2)
            * Z = z0 + (eta3*Lz)
        * 'shafranov_dshaped' :
            * Soloviev equilibrium as described by Cerfon and Freiberg (doi: 10.1063/1.3328818).
            * X = x0 + R0 * [ 1 + (1 - eta1**2) * delta_x + eta1 * epsilon_gs * cos(2*pi*eta2 + arcsin(delta_gs)*eta1*sin(2*pi*eta2)) ]
            * Y = y0 + R0 * [     (1 - eta1**2) * delta_y + eta1 * epsilon_gs * kappa_gs * sin(2*pi*eta2) ]
            * Z = z0 + (eta3*Lz)
        * 'spline': 
            * 3d IGA spline mapping. All information is stored in control points cx, cy, cz.
        * 'spline_cyl': 
            * 2d IGA spline mapping in (eta1, eta2) --> (X, Y) w/ control points cx, cy, cz=None and
            * X = a*eta1*np.cos(2*np.pi*eta2) + R0
            * Y = a*eta1*np.sin(2*np.pi*eta2)
            * Z = 2*pi*R0*eta3
        * 'spline_torus' : 
            * 2d IGA spline mapping in (eta1, eta2) --> (R, Y) w/ control points cx, cy, cz=None and
            * X = R*cos(2*pi*eta3) = (a*eta1*np.cos(2*np.pi*eta2) + R0)*cos(2*pi*eta3) 
            * Y = Y                =  a*eta1*np.sin(2*np.pi*eta2)
            * Z = R*sin(2*pi*eta3) = (a*eta1*np.cos(2*np.pi*eta2) + R0)*sin(2*pi*eta3) 
    '''

    def __init__(self, kind_map='cuboid', params_map=None):

        # ==============================================================
        #                      analytical mappings
        # ==============================================================

        # Note : in future versions you only need to specify the Psydac_mapping expressions.

        # ================== cuboid ====================
        if kind_map == 'cuboid':
            self._kind_map = 10

            if params_map is None:
                params = {'l1': 0., 'r1': 1., 'l2': 0.,
                          'r2': 1., 'l3': 0., 'r3': 1.}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'l1 + (r1 - l1)*x1',
                                                'y': 'l2 + (r2 - l2)*x2',
                                                'z': 'l3 + (r3 - l3)*x3'}

            self._pole = False

        # ============== hollow cylinder ===============
        elif kind_map == 'hollow_cyl':
            self._kind_map = 11

            if params_map is None:
                params = {'a1': 0.2, 'a2': 1., 'R0': 3.}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': '(a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0',
                                                'y': '(a1 + (a2 - a1)*x1)*sin(2*pi*x2)',
                                                'z': '2*pi*R0*x3'}

            if self.params_map[0] == 0.:
                self._pole = True
            else:
                self._pole = False

        # ================== colella ===================
        elif kind_map == 'colella':
            self._kind_map = 12

            if params_map is None:
                params = {'Lx': 1., 'Ly': 1., 'alpha': 0.1, 'Lz': 1.}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                                'y': 'Ly*(x2 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                                'z': 'Lz*x3'}

            self._pole = False

        # ================= orthogonal =================
        elif kind_map == 'orthogonal':
            self._kind_map = 13

            if params_map is None:
                params = {'Lx': 1., 'Ly': 1., 'alpha': 0.1, 'Lz': 1.}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1))',
                                                'y': 'Ly*(x2 + alpha*sin(2*pi*x2))',
                                                'z': 'Lz*x3'}

            self._pole = False

        # ============== hollow torus ==================
        elif kind_map == 'hollow_torus':
            self._kind_map = 14

            if params_map is None:
                params = {'a1': 0.2, 'a2': 1., 'R0': 3.}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * cos(2*pi*x3)',
                                                'y': '( a1 + (a2 - a1)*x1)*sin(2*pi*x2)',
                                                'z': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * sin(2*pi*x3)'}

            if self.params_map[0] == 0.:
                self._pole = True
            else:
                self._pole = False

        # ============== ellipse =======================
        elif kind_map == 'ellipse':
            self._kind_map = 15

            if params_map is None:
                params = {'x0': 0., 'y0': 0., 'z0': 0.,
                          'rx': 1., 'ry': 2., 'Lz': 1.}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2)',
                                                'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}

            self._pole = True

        # =========== rotated ellipse ==================
        elif kind_map == 'rotated_ellipse':
            self._kind_map = 16

            if params_map is None:
                params = {'x0': 0., 'y0': 0., 'z0': 0.,
                          'r1': 1., 'r2': 2., 'Lz': 1., 'th': 0.2}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*r1) * cos(2*pi*th) * cos(2*pi*x2) - (x1*r2) * sin(2*pi*th) * sin(2*pi*x2)',
                                                'y': 'y0 + (x1*r1) * sin(2*pi*th) * cos(2*pi*x2) + (x1*r2) * cos(2*pi*th) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}
            self._pole = True

        # ============ shafranov shift =================
        elif kind_map == 'shafranov_shift':
            self._kind_map = 17

            if params_map is None:
                params = {'x0': 0., 'y0': 0., 'z0': 0.,
                          'rx': 1., 'ry': 1., 'Lz': 1., 'delta': 0.2}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2) + (1-x1**2) * rx * delta',
                                                'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}

            self._pole = True

        # ============ shafranov sqrt ==================
        elif kind_map == 'shafranov_sqrt':
            self._kind_map = 18

            if params_map is None:
                params = {'x0': 0., 'y0': 0., 'z0': 0.,
                          'rx': 1., 'ry': 1., 'Lz': 1., 'delta': 0.2}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2) + (1-sqrt(x1)) * rx * delta',
                                                'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                                'z': 'z0 + (x3*Lz)'}

            self._pole = True

        # ========= shafranov D-shaped =================
        elif kind_map == 'shafranov_dshaped':
            self._kind_map = 19

            if params_map is None:
                params = {'x0': 0., 'y0': 0., 'z0': 0., 'R0': 3., 'Lz': 1., 'delta_x': 0.1,
                          'delta_y': 0., 'delta_gs': 0.2, 'epsilon_gs': 1/3, 'kappa_gs': 1.5}
            else:
                params = params_map

            self._params_map = list(params.values())
            self.Psydac_mapping._expressions = {'x': 'x0 + R0 * ( 1 + (1 - x1**2) * delta_x + x1 * epsilon_gs * cos(2*pi*x2 + asin(delta_gs)*x1*sin(2*pi*x2)) )',
                                                'y': 'y0 + R0 * (     (1 - x1**2) * delta_y + x1 * epsilon_gs * kappa_gs * sin(2*pi*x2) )',
                                                'z': 'z0 + (x3*Lz)'}

            self._pole = True

        # ==============================================================
        #               IGA mappings (with control points)
        # ==============================================================

        # ================= 3d IGA =====================
        elif kind_map == 'spline':
            # TODO: choose correct params_map
            self._kind_map = 0
            self._params_map = []

            # print(f'Before popping: list(params_map.values()): {list(params_map.values())}')

            with h5py.File(params_map['file'], 'r') as handle:

                # print(f'Available keys: {tuple(handle.keys())}')
                self._cx = handle['cx'][:]
                self._cy = handle['cy'][:]
                self._cz = handle['cz'][:]

            if np.all(self.cx[0, :, 0] == self.cx[0, 0, 0]):
                self._pole = True
            else:
                self._pole = False

        # ============== 2d IGA cylinder ===============
        elif kind_map == 'spline_cyl':
            self._kind_map = 1

            if params_map is None:
                params = {'a': 1., 'R0': 3., 'Nel': [
                    8, 24], 'p': [2, 2], 'spl_kind': [False, True]}
            else:
                params = params_map

            def X(s, chi): return params['a']*s*np.cos(2*np.pi*chi) + params['R0']

            def Y(s, chi): return params['a']*s*np.sin(2*np.pi*chi)

            self._cx, self._cy = interp_mapping(
                params['Nel'], params['p'], params['spl_kind'], X, Y)

            # make sure that control points at pole are all the same
            self._cx[0] = params['R0']
            self._cy[0] = 0.

            self._params_map = [params['a'], params['R0']]

            self._pole = True

            self._cx = self.cx[:, :, None]
            self._cy = self.cy[:, :, None]
            self._cz = np.zeros((1, 1, 1), dtype=float)

        # ============= 2d IGA torus ==================
        elif kind_map == 'spline_torus':
            self._kind_map = 2

            if params_map is None:
                params = {'a': 1., 'R0': 3., 'Nel': [8, 24], 'p': [
                    2, 2], 'spl_kind': [False, True], 'coordinates': 'straight'}
            else:
                params = params_map

            def R(s, chi): return params['a']*s*np.cos(theta(
                s, chi, params['a'], params['R0'], params['coordinates'])) + params['R0']
            def Y(s, chi): return params['a']*s*np.sin(theta(
                s, chi, params['a'], params['R0'], params['coordinates']))

            self._cx, self._cy = interp_mapping(
                params['Nel'], params['p'], params['spl_kind'], R, Y)

            # make sure that control points at pole are all the same
            self._cx[0] = params['R0']
            self._cy[0] = 0.

            self._params_map = [params['a'], params['R0']]

            self._pole = True

            self._cx = self._cx[:, :, None]
            self._cy = self._cy[:, :, None]
            self._cz = np.zeros((1, 1, 1), dtype=float)

        # =========== 2d IGA straight general =========
        elif kind_map == 'spline_straight':
            self._kind_map = 1
            self._params_map = []

            with h5py.File(params_map['file'], 'r') as handle:
                self._cx = handle['cx'][:]
                self._cy = handle['cy'][:]

            assert self._cx.ndim == 2 and self._cy.ndim == 2

            if np.all(self._cx[0, :] == self._cx[0, 0]):
                self._pole = True
            else:
                self._pole = False

            self._cx = self._cx[:, :, None]
            self._cy = self._cy[:, :, None]
            self._cz = np.zeros((1, 1, 1), dtype=float)

        # ========= 2d IGA toroidal general ===========
        elif kind_map == 'spline_toroidal':
            self._kind_map = 2
            self._params_map = []

            with h5py.File(params_map['file'], 'r') as handle:

                self._cx = handle['cx'][:]
                self._cy = handle['cy'][:]

            assert self._cx.ndim == 2 and self._cy.ndim == 2

            if np.all(self._cx[0, :] == self._cx[0, 0]):
                self._pole = True
            else:
                self._pole = False

            self._cx = self._cx[:, :, None]
            self._cy = self._cy[:, :, None]
            self._cz = np.zeros((1, 1, 1), dtype=float)

        else:
            raise ValueError('Specified domain is not implemeted!')

        # create IGA attributes for IGA mappings
        if self._kind_map < 10:

            self._Nel = params['Nel']
            self._p = params['p']
            self._spl_kind = params['spl_kind']

            self._NbaseN = [Nel + p - kind*p for Nel, p,
                            kind in zip(params['Nel'], params['p'], params['spl_kind'])]

            el_b = [np.linspace(0., 1., Nel + 1) for Nel in params['Nel']]
            self._T = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(
                el_b, params['p'], params['spl_kind'])]
            
            self._indN = [(np.indices((Nel, p + 1))[1] + np.arange(Nel)[:, None])%NbaseN for Nel, p, NbaseN in zip(params['Nel'], params['p'], self._NbaseN)] 

            # extend to 3d for 2d IGA mappings
            if self._kind_map != 0:

                self._Nel = self._Nel + [0]
                self._p = self._p + [0]
                self._NbaseN = self._NbaseN + [0]

                self._T = self._T + [np.zeros((1,), dtype=float)]
                
                self._indN = self._indN + [np.zeros((1, 1), dtype=int)]

        # create dummy attributes for analytical mappings
        else:

            self._Nel = [0, 0, 0]
            self._p = [0, 0, 0]
            self._spl_kind = [True, True, True]
            self._NbaseN = [0, 0, 0]

            self._T = [np.zeros((1,), dtype=float),
                       np.zeros((1,), dtype=float),
                       np.zeros((1,), dtype=float)]
            
            self._indN = [np.zeros((1, 1), dtype=int),
                          np.zeros((1, 1), dtype=int),
                          np.zeros((1, 1), dtype=int)]

            self._cx = np.zeros((1, 1, 1), dtype=float)
            self._cy = np.zeros((1, 1, 1), dtype=float)
            self._cz = np.zeros((1, 1, 1), dtype=float)

        # trasform parameter list to numpy array
        self._params_map = np.array(self.params_map)

        # keys for evaluating mapping related quantities
        self._keys_map = {
            'x': 1, 'y': 2, 'z': 3, 'det_df': 4,
            'df_11': 11, 'df_12': 12, 'df_13': 13,
            'df_21': 14, 'df_22': 15, 'df_23': 16,
            'df_31': 17, 'df_32': 18, 'df_33': 19,
            'df_inv_11': 21, 'df_inv_12': 22, 'df_inv_13': 23,
            'df_inv_21': 24, 'df_inv_22': 25, 'df_inv_23': 26,
            'df_inv_31': 27, 'df_inv_32': 28, 'df_inv_33': 29,
            'g_11': 31, 'g_12': 32, 'g_13': 33, 'g_21': 34,
            'g_22': 35, 'g_23': 36, 'g_31': 37, 'g_32': 38, 'g_33': 39,
            'g_inv_11': 41, 'g_inv_12': 42, 'g_inv_13': 43, 'g_inv_21': 44,
            'g_inv_22': 45, 'g_inv_23': 46, 'g_inv_31': 47, 'g_inv_32': 48,
            'g_inv_33': 49
        }

        # keys for performing pull-backs
        self._keys_pull = {
            '0_form': 0, '3_form': 3, '1_form_1': 11, '1_form_2': 12,
            '1_form_3': 13, '2_form_1': 21, '2_form_2': 22, '2_form_3': 23,
            'vector_1': 31, 'vector_2': 32, 'vector_3': 33
        }

        # keys for performing push-forwards
        self._keys_push = {
            '0_form': 0, '3_form': 3, '1_form_1': 11, '1_form_2': 12,
            '1_form_3': 13, '2_form_1': 21, '2_form_2': 22, '2_form_3': 23,
            'vector_1': 31, 'vector_2': 32, 'vector_3': 33
        }

        # keys for performing transformation
        self._keys_transform = {
            'norm_to_0': 0, 'norm_to_3': 3,
            'norm_to_1_1': 11, 'norm_to_1_2': 12, 'norm_to_1_3': 13,
            'norm_to_2_1': 21, 'norm_to_2_2': 22, 'norm_to_2_3': 23,
            'norm_to_vector_1': 31, 'norm_to_vector_2': 32, 'norm_to_vector_3': 33,
            '1_to_1_1': 41, '1_to_1_1': 42, '1_to_1_1': 43,
            '1_to_1_1': 51, '1_to_1_1': 52, '1_to_1_1': 53,
            '0_to_3': 4, '3_to_0': 5}

    class Psydac_mapping(Mapping):
        '''To create a psydac domain.'''

        _expressions = None
        _ldim = 3
        _pdim = 3

    @property
    def kind_map(self):
        '''String that defines the mapping.'''
        return self._kind_map

    @property
    def params_map(self):
        '''List of mapping parameters.'''
        return self._params_map

    @property
    def pole(self):
        '''Bool; True if mapping has one polar point.'''
        return self._pole

    @property
    def cx(self):
        '''3d array of control points for first mapping component Fx.'''
        return self._cx

    @property
    def cy(self):
        '''3d array of control points for second mapping component Fy.'''
        return self._cy

    @property
    def cz(self):
        '''3d array of control points for third mapping component Fz.'''
        return self._cz

    @property
    def Nel(self):
        '''List of number of elements in each direction.'''
        return self._Nel

    @property
    def p(self):
        '''List of spline degrees in each direction.'''
        return self._p

    @property
    def spl_kind(self):
        '''List of spline type (True=periodic, False=clamped) in each direction.'''
        return self._spl_kind

    @property
    def NbaseN(self):
        '''List of number of basis functions for N-splines in each direction.'''
        return self._NbaseN

    @property
    def T(self):
        '''List of knot vectors for N-splines in each direction.'''
        return self._T
    
    @property
    def indN(self):
        '''Global indices of non-vanishing splines in each element. Can be accessed via (element index, local spline index).'''
        return self._indN

    @property
    def keys_map(self):
        '''Dictionary of str->int for kind_map.'''
        return self._keys_map

    @property
    def keys_pull(self):
        '''Dictionary of str->int for pull function.'''
        return self._keys_pull

    @property
    def keys_push(self):
        '''Dictionary of str->int for push function.'''
        return self._keys_push

    @property
    def keys_transform(self):
        '''Dictionary of str->int for transform function.'''
        return self._keys_transform

    def evaluate(self, eta1, eta2, eta3, kind_fun, flat_eval=False, squeeze_output=True):
        '''Evaluate mapping/metric coefficients. 

        Depending on the dimension of eta1, eta2, eta3 either point-wise, tensor-product, slice plane or general (see prepare_args).

        Parameters
        -----------
            eta1, eta2, eta3 : point like or array-like or list like 
                Logical coordinates at which to evaluate.
                
            kind_fun : string
                What metric coefficient to evaluate, see Notes.
                
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)]. 
                
            squeeze_output : boolean
                Whether to remove singleton dimensions in output "values".

        Returns
        --------
            values : np.array
                Mapping/metric coefficients evaluated at (eta1, eta2, eta3). 

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

        E1, E2, E3, is_sparse_meshgrid = prepare_args(
            eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty(
                (E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if is_sparse_meshgrid:
            mapping.kernel_evaluate_sparse(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            mapping.kernel_evaluate_flat(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            mapping.kernel_evaluate(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)

        if squeeze_output:
            values = values.squeeze()

        if values.ndim == 0:
            values = values.item()

        return values

    # ================================

    def pull(self, a, eta1, eta2, eta3, kind_fun='0_form', flat_eval=False, squeeze_output=True):
        '''Pullback of p-forms. 

        Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see prepare_args).

        Parameters
        ----------
            a : callable or array-like
                The function a(x, y, z) to be pulled back (as list for components of 1- and 2-forms). 
                
            eta1, eta2, eta3 : array-like
                Logical coordinates to which to pull back.
                
            kind_fun : str
                Which p-form pull back to apply, see keys_pull.
                
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)].
                
            squeeze_output : boolean
                Whether to remove singleton dimensions in output "values".

        Returns
        -------
            values: np.array
                Pullback of p-form (component) evaluated at (eta1, eta2, eta3).

        Notes
        -----
            Possible choices for kind_fun:

                * '0_form'  , '3_form'
                * '1_form_1', '1_form_2', '1_form_3'
                * '2_form_1', '2_form_2', '2_form_3',
                * 'vector_1', 'vector_2', 'vector_3'
        '''

        E1, E2, E3, is_sparse_meshgrid = prepare_args(
            eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty(
                (E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if isinstance(a, (list, tuple)):

            if callable(a[0]):

                X = self.evaluate(E1, E2, E3, 'x', flat_eval,
                                  squeeze_output=False)
                Y = self.evaluate(E1, E2, E3, 'y', flat_eval,
                                  squeeze_output=False)
                Z = self.evaluate(E1, E2, E3, 'z', flat_eval,
                                  squeeze_output=False)

                a_in = np.array([a[0](X, Y, Z), a[1](X, Y, Z),
                                a[2](X, Y, Z)], dtype=float)
            else:
                a_in = np.array(a)

        else:

            if callable(a):

                X = self.evaluate(E1, E2, E3, 'x', flat_eval,
                                  squeeze_output=False)
                Y = self.evaluate(E1, E2, E3, 'y', flat_eval,
                                  squeeze_output=False)
                Z = self.evaluate(E1, E2, E3, 'z', flat_eval,
                                  squeeze_output=False)

                a_in = np.array([a(X, Y, Z)])
            else:
                a_in = np.array([a])

        if is_sparse_meshgrid:
            pb.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            pb.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            pb.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)

        if squeeze_output:
            values = values.squeeze()

        if values.ndim == 0:
            values = values.item()

        return values

    # ================================

    def push(self, a, eta1, eta2, eta3, kind_fun='0_form', flat_eval=False, squeeze_output=True):
        '''Push-forward of p-forms. 

        Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see prepare_args).

        Parameters
        -----------
            a : callable or array-like
                The function a(eta1, eta2, eta3) to be pushed forward (as list for components of 1- and 2-forms).
                
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to push forward.
                
            kind_fun : str
                Which p-form push forward to apply, see keys_push.
                
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)].
                
            squeeze_output : boolean
                Whether to remove singleton dimensions in output "values".

        Returns
        --------
            values: ndarray
                Push forward of p-form (component) evaluated at (eta1, eta2, eta3).

        Notes
        -----
            Possible choices for kind_fun:

                * '0_form'  , '3_form'
                * '1_form_1', '1_form_2', '1_form_3'
                * '2_form_1', '2_form_2', '2_form_3',
                * 'vector_1', 'vector_2', 'vector_3'
        '''

        E1, E2, E3, is_sparse_meshgrid = prepare_args(
            eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty(
                (E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if isinstance(a, (list, tuple)):
            if callable(a[0]):
                a_in = np.array([a[0](E1, E2, E3), a[1](
                    E1, E2, E3), a[2](E1, E2, E3)], dtype=float)
            else:
                a_in = np.array(a)

        else:
            if callable(a):
                a_in = np.array([a(E1, E2, E3)])
            else:
                a_in = np.array([a])

        if is_sparse_meshgrid:
            pf.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            pf.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            pf.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)

        if squeeze_output:
            values = values.squeeze()

        if values.ndim == 0:
            values = values.item()

        return values

    # ================================

    def transform(self, a, eta1, eta2, eta3, kind_fun='norm_to_0', flat_eval=False, squeeze_output=True):
        '''Transformation between different p-forms on logical domain. 

        Depending on the dimension of eta1 either point-wise, tensor-product, slice plane or general (see prepare_args).

        Parameters
        ----------
            a : callable or array-like
                The function a(eta1, eta2, eta3) to be transformed.
                
            eta1, eta2, eta3 : array-like
                Logical coordinates to which to transform.
                
            kind_fun : str
                Which transform to apply, see keys_transform.
                
            flat_eval : boolean
                Whether to do a flat evaluation, i.e. f([x1, x2], [y1, y2]) = [f(x1, y1) f(x2, y2)].
                
            squeeze_output : boolean
                Whether to remove singleton dimensions in output "values".

        Returns
        -------
            values: ndarray
                Transformed p-form from norm_vector or scalar (component) evaluated at (eta1, eta2, eta3).

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

        E1, E2, E3, is_sparse_meshgrid = prepare_args(
            eta1, eta2, eta3, flat_eval)

        if flat_eval:
            values = np.empty(E1.size, dtype=float)
        else:
            values = np.empty(
                (E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        if isinstance(a, (list, tuple)):
            if callable(a[0]):
                a_in = np.array([a[0](E1, E2, E3), a[1](
                    E1, E2, E3), a[2](E1, E2, E3)], dtype=float)
            else:
                a_in = np.array(a)

        else:
            if callable(a):
                a_in = np.array([a(E1, E2, E3)])
            else:
                a_in = np.array([a])

        if is_sparse_meshgrid:
            tr.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            tr.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            tr.kernel_evaluate(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)

        if squeeze_output:
            values = values.squeeze()

        if values.ndim == 0:
            values = values.item()

        return values
    
    # ================================
    
    def show(self, save_dir=None):
        '''
        Plots isolines (and control point in case on spline mappings) of the 2D domain at eta3 = 0.

        Parameters
        ----------

        save_dir : string (optional)
                If given, the figure is saved according the given directory.
        '''

        import matplotlib.pyplot as plt

        e1 = np.linspace(0., 1., 101)
        e2 = np.linspace(0., 1., 101)

        X = self.evaluate(e1, e2, 0., 'x')
        Y = self.evaluate(e1, e2, 0., 'y')

        # eta1-isolines
        for i in range(e1.size//5 + 1):
            plt.plot(X[i*5, :], Y[i*5, :], 'k')

        # eta2-isolines
        for j in range(e2.size//5 + 1):
            plt.plot(X[:, j*5], Y[:, j*5], 'r')

        if self.kind_map < 10:
            plt.scatter(self.cx[:, :, 0].flatten(),
                        self.cy[:, :, 0].flatten(), s=3, color='b')

        plt.xlabel('x')
        plt.ylabel('y')

        plt.axis('square')

        if save_dir is not None:
            plt.savefig(save_dir)
        else:
            plt.show()
