# coding: utf-8

from abc import ABCMeta, abstractmethod
import h5py
import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import splu

from struphy.geometry import map_eval
from struphy.geometry import pullback
from struphy.geometry import pushforward
from struphy.geometry import transform
from struphy.linear_algebra import linalg_kron 
import struphy.feec.bsplines as bsp

from sympde.topology import Mapping


class Domain(metaclass=ABCMeta):
    '''Base class for mapped domains.'''

    def __init__(self):

        # create IGA attributes for IGA mappings
        if self.kind_map < 10:

            self._Nel = self.params_map['Nel']
            self._p = self.params_map['p']
            self._spl_kind = self.params_map['spl_kind']

            self._NbaseN = [Nel + p - kind*p for Nel, p,
                            kind in zip(self.params_map['Nel'], self.params_map['p'], self.params_map['spl_kind'])]

            el_b = [np.linspace(0., 1., Nel + 1) for Nel in self.params_map['Nel']]
            self._T = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(
                el_b, self.params_map['p'], self.params_map['spl_kind'])]
            
            self._indN = [(np.indices((Nel, p + 1))[1] + np.arange(Nel)[:, None])%NbaseN for Nel, p, NbaseN in zip(self.params_map['Nel'], self.params_map['p'], self._NbaseN)] 

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
        

    class PsydacMapping(Mapping):
        '''To create a psydac domain.'''

        _expressions = None
        _ldim = 3
        _pdim = 3

    @property
    @abstractmethod
    def kind_map(self):
        '''Integer defining the mapping; must be <10 for spline mappings and >=10 otherwise.'''
        pass

    @property
    @abstractmethod
    def params_map(self):
        '''Mapping parameters: as list for analytical mappings, as dict for spline mappings.
        Transformed to numpy array during init.'''
        pass
    
    @property
    @abstractmethod
    def F_psy(self):
        '''Symbolic psydac mapping.'''
        pass

    @property
    @abstractmethod
    def pole(self):
        '''Bool; True if mapping has one polar point.'''
        pass

    @property
    @abstractmethod
    def periodic_eta3(self):
        '''Bool; True if mapping is periodic in eta_3 coordinate.'''
        pass

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
            map_eval.kernel_evaluate_sparse(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            map_eval.kernel_evaluate_flat(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            map_eval.kernel_evaluate(E1, E2, E3, self.keys_map[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
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
            pullback.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            pullback.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            pullback.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
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
            pushforward.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            pushforward.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            pushforward.kernel_evaluate(a_in, E1, E2, E3, self.keys_pull[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
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
            transform.kernel_evaluate_sparse(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        elif flat_eval:
            transform.kernel_evaluate_flat(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
                self.p), self.indN[0], self.indN[1], self.indN[2], self.cx, self.cy, self.cz, values)
        else:
            transform.kernel_evaluate(a_in, E1, E2, E3, self.keys_transform[kind_fun], self.kind_map, self.params_map, self.T[0], self.T[1], self.T[2], np.array(
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
        return linalg_kron.kron_lusolve_2d(I_LU, values), T, indN
    elif len(p) == 3:
        return linalg_kron.kron_lusolve_3d(I_LU, values), T, indN
    else:
        raise AssertionError("Only dimensions < 4 are supported.")