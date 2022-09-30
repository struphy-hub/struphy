# coding: utf-8

from abc import ABCMeta, abstractmethod

from sympde.topology import Mapping

from struphy.geometry import map_eval, pullback, pushforward, transform 
from struphy.linear_algebra import linalg_kron 
import struphy.feec.bsplines as bsp

from scipy.sparse import csc_matrix, kron
from scipy.sparse.linalg import splu, spsolve

import h5py
import numpy as np


class Domain(metaclass=ABCMeta):
    """ Base class for mapped domains.
    """

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
            '0_form': 0, '3_form': 3,
            '1_form_1': 11, '1_form_2': 12, '1_form_3': 13, 
            '2_form_1': 21, '2_form_2': 22, '2_form_3': 23,
            'vector_1': 31, 'vector_2': 32, 'vector_3': 33
        }

        # keys for performing push-forwards
        self._keys_push = {
            '0_form': 0, '3_form': 3,
            '1_form_1': 11, '1_form_2': 12, '1_form_3': 13,
            '2_form_1': 21, '2_form_2': 22, '2_form_3': 23,
            'vector_1': 31, 'vector_2': 32, 'vector_3': 33
        }

        # keys for performing transformation
        self._keys_transform = {
            '0_to_3': 0, '3_to_0': 1,
            'norm_to_v_1': 11, 'norm_to_v_2': 12, 'norm_to_v_3': 13,
            'norm_to_1_1': 21, 'norm_to_1_2': 22, 'norm_to_1_3': 23,
            'norm_to_2_1': 31, 'norm_to_2_2': 32, 'norm_to_2_3': 33,
            '1_to_2_1': 41, '1_to_2_2': 42, '1_to_2_3': 43,
            '2_to_1_1': 51, '2_to_1_2': 52, '2_to_1_3': 53
        }
        

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
        '''Mapping parameters: as numpy array for analytical mappings, as dict for spline mappings.
        Needs to be manually reset to numpy array after __init__ for the latter.'''
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
    def args_map(self):
        '''Tuple of all parameters needed for evaluation of metric coefficients.'''
        
        _args_map = (self._kind_map, self._params_map,
                     np.array(self._p), self._T[0], self._T[1], self._T[2],
                     self._indN[0], self._indN[1], self._indN[2],
                     self._cx, self._cy, self._cz)
        
        return _args_map

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
    
    
    # ========================
    def __call__(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True, change_out_order=False):
        """
        Evaluates the mapping.
        
        Parameters
        ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".
                
            change_out_order : bool
                If True, the axis corresponding to the x, y, z coordinates in the output array is the last one, otherwise the first one.
                
        Returns
        -------
            values : array-like
                The Cartesian coordinates corresponding to the given logical ones.
        """
        
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        values = np.empty((3, 1, E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        map_eval.kernel_evaluate_all(E1, E2, E3, 0, *self.args_map, values, is_sparse_meshgrid)
        
        values = values[:, 0, :, :, :]
            
        if flat_eval:
            values = values[:, :, 0, 0]
            if change_out_order: values = np.transpose(values, axes=(1, 0))
        else:
            if change_out_order: values = np.transpose(values, axes=(1, 2, 3, 0))
            
            if squeeze_output: 
                values = values.squeeze()

            if values.ndim == 0: 
                values = values.item()

        return values
    
    # ========================
    def jacobian(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True, change_out_order=False, transposed=False):
        """
        Evaluates the Jacobian matrix.
        
        Parameters
        ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".
                
            change_out_order : bool
                If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.
                
            transposed : bool
                If True, the transposed Jacobian matrix is evaluated.
                
        Returns
        -------
            values : array-like
                The Jacobian matrix evaluated at the given logical coordinates.
        """
        
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        values = np.empty((3, 3, E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        map_eval.kernel_evaluate_all(E1, E2, E3, 1, *self.args_map, values, is_sparse_meshgrid)
            
        if transposed: values = np.transpose(values, axes=(1, 0, 2, 3, 4))
            
        if flat_eval:
            values = values[:, :, :, 0, 0]
            if change_out_order: values = np.transpose(values, axes=(2, 0, 1))       
        else:
            if change_out_order: values = np.transpose(values, axes=(2, 3, 4, 0, 1))
            
            if squeeze_output: 
                values = values.squeeze()

            if values.ndim == 0: 
                values = values.item()

        return values
    
    # ========================
    def jacobian_det(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True):
        """
        Evaluates the Jacobian determinant.
        
        Parameters
        ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".
                
        Returns
        -------
            values : array-like
                The Jacobian determinant evaluated at the given logical coordinates.
        """
        
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        values = np.empty((1, 1, E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        map_eval.kernel_evaluate_all(E1, E2, E3, 2, *self.args_map, values, is_sparse_meshgrid)
        
        values = values[0, 0, :, :, :]
        
        if flat_eval:
            values = values[:, 0, 0]
        else:
            if squeeze_output: 
                values = values.squeeze()

            if values.ndim == 0: 
                values = values.item()

        return values
    
    # ========================
    def jacobian_inv(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True, change_out_order=False, transposed=False):
        """
        Evaluates the inverse Jacobian matrix.
        
        Parameters
        ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".
                
            change_out_order : bool
                If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.
                
            transposed : bool
                If True, the inverse transposed Jacobian matrix is evaluated.
                
        Returns
        -------
            values : array-like
                The inverse Jacobian matrix evaluated at the given logical coordinates.
        """
        
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        values = np.empty((3, 3, E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        map_eval.kernel_evaluate_all(E1, E2, E3, 3, *self.args_map, values, is_sparse_meshgrid)
            
        if transposed: values = np.transpose(values, axes=(1, 0, 2, 3, 4))
            
        if flat_eval:
            values = values[:, :, :, 0, 0]
            if change_out_order: values = np.transpose(values, axes=(2, 0, 1))       
        else:
            if change_out_order: values = np.transpose(values, axes=(2, 3, 4, 0, 1))
            
            if squeeze_output: 
                values = values.squeeze()

            if values.ndim == 0: 
                values = values.item()

        return values
    
    # ========================
    def metric(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True, change_out_order=False):
        """
        Evaluates the metric tensor.
        
        Parameters
        ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".
                
            change_out_order : bool
                If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.
                
        Returns
        -------
            values : array-like
                The metric tensor evaluated at the given logical coordinates.
        """
        
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        values = np.empty((3, 3, E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        map_eval.kernel_evaluate_all(E1, E2, E3, 4, *self.args_map, values, is_sparse_meshgrid)
            
        if flat_eval:
            values = values[:, :, :, 0, 0]
            if change_out_order: values = np.transpose(values, axes=(2, 0, 1))       
        else:
            if change_out_order: values = np.transpose(values, axes=(2, 3, 4, 0, 1))
            
            if squeeze_output: 
                values = values.squeeze()

            if values.ndim == 0: 
                values = values.item()

        return values
    
    # ========================
    def metric_inv(self, eta1, eta2, eta3, flat_eval=False, squeeze_output=True, change_out_order=False):
        """
        Evaluates the inverse metric tensor.
        
        Parameters
        ----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".
                
            change_out_order : bool
                If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.
                
        Returns
        -------
            values : array-like
                The metric tensor evaluated at the given logical coordinates.
        """
        
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        values = np.empty((3, 3, E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        map_eval.kernel_evaluate_all(E1, E2, E3, 5, *self.args_map, values, is_sparse_meshgrid)
            
        if flat_eval:
            values = values[:, :, :, 0, 0]
            if change_out_order: values = np.transpose(values, axes=(2, 0, 1))       
        else:
            if change_out_order: values = np.transpose(values, axes=(2, 3, 4, 0, 1))
            
            if squeeze_output: 
                values = values.squeeze()

            if values.ndim == 0: 
                values = values.item()

        return values
    
    # ========================
    def evaluate(self, eta1, eta2, eta3, kind_eval, flat_eval=False, squeeze_output=True):
        """
        Evaluate mapping/metric coefficients. 

        Depending on the dimension of eta1, eta2, eta3 either point-wise, tensor-product, slice plane, etc. (see prepare_eval_pts).

        Parameters
        -----------
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to evaluate.
                
            kind_eval : str
                What metric coefficient to evaluate, see Notes.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".

        Returns
        --------
             values : array-like
                Mapping/metric coefficients evaluated at (eta1, eta2, eta3). 

        Notes
        -----
            Possible choices for kind_eval: 

                * 'x', 'y', 'z': components of F
                * 'det_df': Jacobian determinant
                * 'df_11', 'df_12', 'df_13', 'df_21', 'df_22', 'df_23', 'df_31', 'df_32', 'df_33': Jacobian 
                * 'df_inv_11', 'df_inv_12', 'df_inv_13', 'df_inv_21', 'df_inv_22', 'df_inv_23', 'df_inv_31', 'df_inv_32', 'df_inv_33', Jacobian inverse 
                * 'g_11', 'g_12', 'g_13', 'g_21', 'g_22', 'g_23', 'g_31', 'g_32', 'g_33': metric tensor 
                * 'g_inv_11', 'g_inv_12', 'g_inv_13', 'g_inv_21', 'g_inv_22', 'g_inv_23', 'g_inv_31', 'g_inv_32', 'g_inv_33': inverse metric tensor
        """

        # convert evaluation points to 3d array of appropriate shape
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        # call evaluation kernel
        values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)

        map_eval.kernel_evaluate(E1, E2, E3, self.keys_map[kind_eval], *self.args_map, values, is_sparse_meshgrid)

        # convert evaluated values to correct shape
        if flat_eval:
            values = values[:, 0, 0]
        else:
            if squeeze_output:
                values = values.squeeze()

            if values.ndim == 0:
                values = values.item()

        return values

    # ================================
    def pull(self, a, eta1, eta2, eta3, kind_pull, flat_eval=False, squeeze_output=True):
        """
        Pull-back of p-forms. 

        Depending on the dimension of eta1, eta2, eta3 either point-wise, tensor-product, slice plane, etc. (see prepare_eval_pts).

        Parameters
        ----------13:00 -14:30
            a : list of callables or array-like
                The function a(x, y, z) resp. [a_x(x, y, z), a_y(x, y, z), a_z(x, y, z)] to be pulled.
                
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to pull.
                
            kind_pull : str
                Which pull-back to apply, see keys_pull.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12, ...], [e21, e22, ...]) = [f(e11, e21), f(e12, e22), ...].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".

        Returns
        -------
             values : array-like
                Pullback of p-form (component) evaluated at (eta1, eta2, eta3).

        Notes
        -----
            Possible choices for kind_pull:

                * '0_form'  , '3_form'
                * '1_form_1', '1_form_2', '1_form_3'
                * '2_form_1', '2_form_2', '2_form_3',
                * 'vector_1', 'vector_2', 'vector_3'
        """
        
        # convert evaluation points to 3d array of appropriate shape
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)
        
        # convert input to be transformed (a) to 4d array of appropriate shape
        X = self(E1, E2, E3, squeeze_output=False)
        
        A = Domain.prepare_arg(a, X[0], X[1], X[2], flat_eval)

        # call evaluation kernel
        values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        pullback.kernel_evaluate(A, E1, E2, E3, self.keys_pull[kind_pull], *self.args_map, values, is_sparse_meshgrid)
        
        # convert pulled values to correct shape
        if flat_eval:
            values = values[:, 0, 0]
        else:
            if squeeze_output:
                values = values.squeeze()

            if values.ndim == 0:
                values = values.item()

        return values

    # ================================
    def push(self, a, eta1, eta2, eta3, kind_push, flat_eval=False, squeeze_output=True):
        """
        Push-forward of p-forms. 

        Depending on the dimension of eta1, eta2, eta3 either point-wise, tensor-product, slice plane, etc. (see prepare_eval_pts).

        Parameters
        -----------
            a : list of callables or array-like
                The function a(e1, e2, e3) resp. [a_1(e1, e2, e), a_2(e1, e2, e), a_3(e1, e2, e3)] to be pushed.
                
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to push.
                
            kind_push : str
                Which push-forward to apply, see keys_push.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".

        Returns
        --------
             values : array-like
                Push forward of p-form (component) evaluated at (eta1, eta2, eta3).

        Notes
        -----
            Possible choices for kind_push:

                * '0_form'  , '3_form'
                * '1_form_1', '1_form_2', '1_form_3'
                * '2_form_1', '2_form_2', '2_form_3',
                * 'vector_1', 'vector_2', 'vector_3'
        """

        # convert evaluation points to 3d array of appropriate shape
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)
        
        # convert input to be transformed (a) to 4d array of appropriate shape
        A = Domain.prepare_arg(a, E1, E2, E3, flat_eval)

        # call evaluation kernel
        values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        pushforward.kernel_evaluate(A, E1, E2, E3, self.keys_pull[kind_push], *self.args_map, values, is_sparse_meshgrid)

        # convert pushed values to correct shape
        if flat_eval:
            values = values[:, 0, 0]
        else:
            if squeeze_output:
                values = values.squeeze()

            if values.ndim == 0:
                values = values.item()

        return values

    # ================================
    def transform(self, a, eta1, eta2, eta3, kind_trans, flat_eval=False, squeeze_output=True):
        """
        Transformation between different p-forms on logical domain. 

        Depending on the dimension of eta1, eta2, eta3 either point-wise, tensor-product, slice plane, etc. (see prepare_eval_pts).

        Parameters
        ----------
            a : list of callables or array-like
                The function a(e1, e2, e3) resp. [a_1(e1, e2, e), a_2(e1, e2, e), a_3(e1, e2, e3)] to be transformed.
                
            eta1, eta2, eta3 : array-like
                Logical coordinates at which to transform.
                
            kind_trans : str
                Which transformation to apply, see keys_transform.
                
            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].
                
            squeeze_output : bool
                Whether to remove singleton dimensions in output "values".

        Returns
        -------
            values : array-like
                Transformed p-form from norm_vector or scalar (component) evaluated at (eta1, eta2, eta3).

        Notes
        -----
            Possible choices for kind_trans:
            
                * 'norm_to_v_1', 'norm_to_v_2', 'norm_to_v_3',
                * 'norm_to_1_1', 'norm_to_1_2', 'norm_to_1_3',
                * 'norm_to_2_1', 'norm_to_2_2', 'norm_to_2_3',
                * '1_to_2_1', '1_to_2_2', '1_to_2_3',
                * '2_to_1_1', '2_to_1_2', '2_to_1_3',
                * '0_to_3', '3_to_0'
        """

        # convert evaluation points (eta1, eta2, eta3) to 3d array of appropriate shape
        E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
            eta1, eta2, eta3, flat_eval)

        # convert input to be transformed (a) to 4d array of appropriate shape
        A = Domain.prepare_arg(a, E1, E2, E3, flat_eval)

        # call evaluation kernel
        values = np.empty((E1.shape[0], E2.shape[1], E3.shape[2]), dtype=float)
        
        transform.kernel_evaluate(A, E1, E2, E3, self.keys_transform[kind_trans], *self.args_map, values, is_sparse_meshgrid)

        # convert transformed values to correct shape
        if flat_eval:
            values = values[:, 0, 0]
        else:
            if squeeze_output:
                values = values.squeeze()

            if values.ndim == 0:
                values = values.item()

        return values 
    
    # ================================
    @staticmethod
    def prepare_eval_pts(x, y, z, flat_eval=False):
        """
        Broadcasts evaluation point sets to 3d arrays of correct shape.

        Parameters
        ----------
            x, y, z : float | int | list | numpy array 
                Evaluation point sets.

            flat_eval : bool
                Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].

        Returns
        -------
            E1, E2, E3 : numpy array
                3d arrays of correct shape for evaluation.

            is_sparse_meshgrid : bool
                Whether arguments fit sparse_meshgrid shape.
        """

        is_sparse_meshgrid = False

        # flat evaluation (works only if all arguments are 1d numpy arrays/lists of equal length!)
        if flat_eval:

            # convert list type data to numpy array:
            if isinstance(x, list):
                arg_x = np.array(x)
            elif isinstance(x, np.ndarray):
                arg_x = x
            else:
                raise ValueError('Input x must be a 1d list or numpy array')

            if isinstance(y, list):
                arg_y = np.array(y)
            elif isinstance(y, np.ndarray):
                arg_y = y
            else:
                raise ValueError('Input y must be a 1d list or numpy array')

            if isinstance(z, list):
                arg_z = np.array(z)
            elif isinstance(z, np.ndarray):
                arg_z = z
            else:
                raise ValueError('Input z must be a 1d list or numpy array')

            assert arg_x.ndim == arg_y.ndim == arg_z.ndim == 1
            assert arg_x.size == arg_y.size == arg_z.size

            E1 = arg_x[:, None, None]
            E2 = arg_y[:, None, None]
            E3 = arg_z[:, None, None]

            return E1, E2, E3, is_sparse_meshgrid

        # non-flat evaluation (broadcast to 3d arrays)
        else:

            # convert list type data to numpy array:
            if isinstance(x, float):
                arg_x = np.array([x])
            elif isinstance(x, int):
                arg_x = np.array([float(x)])
            elif isinstance(x, list):
                arg_x = np.array(x)
            elif isinstance(x, np.ndarray):
                arg_x = x
            else:
                raise ValueError(f'data type {type(x)} not supported')

            if isinstance(y, float):
                arg_y = np.array([y])
            elif isinstance(y, int):
                arg_y = np.array([float(y)])
            elif isinstance(y, list):
                arg_y = np.array(y)
            elif isinstance(y, np.ndarray):
                arg_y = y
            else:
                raise ValueError(f'data type {type(y)} not supported')

            if isinstance(z, float):
                arg_z = np.array([z])
            elif isinstance(z, int):
                arg_z = np.array([float(z)])
            elif isinstance(z, list):
                arg_z = np.array(z)
            elif isinstance(z, np.ndarray):
                arg_z = z
            else:
                raise ValueError(f'data type {type(z)} not supported')

            # tensor-product for given three 1D arrays
            if arg_x.ndim == 1 and arg_y.ndim == 1 and arg_z.ndim == 1:
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
    
    # ================================
    @staticmethod
    def prepare_arg(a_in, X1, X2, X3, flat_eval=False):
        """
        Broadcasts argument to be pulled, pushed or transformed to 4d array of correct shape.

        Parameters
        ----------
            a_in : list, tuple or array-like
                The argument to be pulled, pushed or transformed. Can be a list/tuple of callables that are first evaluated, OR numpy arrays of appropriate shape.

            X1, X2, X3 : array-like
                The evaluation point sets. Obtained from prepare_eval_pts function (with possible additional __call__ for physical coordinates).

            flat_eval : bool
                    Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].

        Returns
        -------
            a_out : array-like
                The 4d array suitable for evaluation kernels.
        """


        # single callable:
        # scalar function -> must return a 3d array for 3d evaluation points
        # vector-valued function -> must return a 4d array of shape (3,:,:,:)
        if callable(a_in):

            a_out = a_in(X1, X2, X3)

            if   a_out.ndim == 3:
                a_out = a_out[None, :, :, :]
            elif a_out.ndim == 4:
                a_out = a_out[:, :, :, :]
            else:
                raise ValueError('Input callable a_in must return a 3d array (for a scalar function) \
                or 4d array of shape (3,:,:,:) (for a vector-valued function) for 3d evaluation point sets!')

        # list/tuple of length 1 or 3 containing:
        # callable(s) that must return 3d array(s) for 3d evaluation points
        # 1d array(s) (flat_eval=True)
        # 3d array(s) (flat eval=False)
        elif isinstance(a_in, (list, tuple)):
            
            assert len(a_in) == 1 or len(a_in) == 3

            a_out = []
            for component in a_in:

                if callable(component):
                    tmp = component(X1, X2, X3)
                    assert tmp.ndim == 3
                    a_out += [tmp]

                elif isinstance(component, np.ndarray):

                    if flat_eval:
                        assert component.ndim == 1
                        a_out += [component[:, None, None]]
                    else:
                        assert component.ndim == 3
                        a_out += [component[:, :, :]]

            a_out = np.array(a_out, dtype=float)

        # numpy array:
        # 1d array (flat_eval=True and scalar input)
        # 2d array (flat_eval=True and vector-valued input of shape (3,:)) 
        # 3d array (flat_eval=False and scalar input) 
        # 4d array (flat_eval=False and vector-valued input of shape (3,:,:,:)) 
        elif isinstance(a_in, np.ndarray):

            if flat_eval:
                if   a_in.ndim == 1: 
                    a_out = a_in[None, :, None, None]
                elif a_in.ndim == 2:
                    a_out = a_in[:, :, None, None]
                else:
                    raise ValueError('Input array a_in must be either 1d (scalar) or \
                    2d (vector-valued, shape (3,:)) for flat evaluation (flat_eval=True)!')

            else:
                if   a_in.ndim == 3:
                    a_out = a_in[None, :, :, :]
                elif a_in.ndim == 4:
                    a_out = a_in[:, :, :, :]
                else:
                    raise ValueError('Input array a_in must be either 3d (scalar) or \
                    4d (vector-valued, shape (3,:,:,:)) for non-flat evaluation (flat_eval=False)!')

        else:
            raise TypeError('Argument a must be either a list/tuple of 1/3 callable(s)/numpy array(s) \
            OR a single numpy array OR a single callable!')

        # make sure that output array is 4d and of shape (1,:,:,:) or (3,:,:,:)
        assert a_out.ndim == 4
        assert a_out.shape[0] == 1 or a_out.shape[0] == 3

        return a_out

    # ================================
    def show(self, logical=False, grid_info=None, markers=None, marker_coords='logical', save_dir=None):
        """
        Plots isolines (and control point in case on spline mappings) of the 2D physical domain for eta3 = 0.
        Markers can be plotted as well (optional).

        Parameters
        ----------
            logical : bool
                Whether to plot the physical domain (False) or logical domain (True).
        
            grid_info : array-like (optional)
                Information about the grid. If not given, the domain is shown with high resolution. If given, can be either
                    * a list of # of elements [Nel1, Nel2, (Nel3)], OR
                    * a 2d array with information about MPI decomposition.
                    
            markers : array-like (optional)
                Markers to be plotted. Can be of shape (Np, 3) or (:, Np, 3). For the former, all markers are plotted with the same color. For the latter, with different colors (are interpreted as orbits in time).
                
            marker_coords : bool (optional)
                Whether the marker coordinates are logical or physical.

            save_dir : string (optional)
                If given, the figure is saved according the given directory.
        """

        import matplotlib.pyplot as plt
        
        # plot given markers
        if markers is not None:
            
            assert not (logical and marker_coords != 'logical')
            
            # no time series: plot all markers with the same color
            if markers.ndim == 2:
                
                if not logical and marker_coords == 'logical':
                    X = self.evaluate(markers[:, 0].copy(),
                                      markers[:, 1].copy(), 
                                      markers[:, 2].copy()*0,
                                      'x', 'flat')
                    
                    Y = self.evaluate(markers[:, 0].copy(), 
                                      markers[:, 1].copy(), 
                                      markers[:, 2].copy()*0, 
                                      'y', 'flat')   
                else:
                    X = markers[:, 0].copy()
                    Y = markers[:, 1].copy()
                
                plt.scatter(X, Y, s=1, color='b')
                
            # time series: plot markers with different colors
            elif markers.ndim == 3:
                
                for i in range(markers.shape[1]):
                    
                    if not logical and marker_coords == 'logical':
                        X = self.evaluate(markers[:, i, 0].copy(), 
                                          markers[:, i, 1].copy(), 
                                          markers[:, i, 2].copy()*0, 
                                          'x', 'flat')
                        
                        Y = self.evaluate(markers[:, i, 0].copy(), 
                                          markers[:, i, 1].copy(), 
                                          markers[:, i, 2].copy()*0, 
                                          'y', 'flat')
                    else:
                        X = markers[:, i, 0].copy()
                        Y = markers[:, i, 1].copy()
                        
                    plt.scatter(X, Y, s=1)    
            
        # plot domain without MPI decomposition and high resolution
        if grid_info is None:

            e1 = np.linspace(0., 1., 101)
            e2 = np.linspace(0., 1., 101)

            if logical:
                E1, E2 = np.meshgrid(e1, e2, indexing='ij')
                X = np.stack((E1, E2), axis=0)
            else:
                X = self(e1, e2, 0.)

            # eta1-isolines
            for i in range(e1.size//5 + 1):
                plt.plot(X[0, i*5, :], X[1, i*5, :], 'tab:blue')

            # eta2-isolines
            for j in range(e2.size//5 + 1):
                plt.plot(X[0, :, j*5], X[1, :, j*5], 'tab:blue')
                
        # plot domain according to given grid [nel1, nel2, (nel3)]
        elif isinstance(grid_info, list):
            
            assert len(grid_info) > 1
            
            e1 = np.linspace(0., 1., grid_info[0] + 1)
            e2 = np.linspace(0., 1., grid_info[1] + 1)
            
            if logical:
                E1, E2 = np.meshgrid(e1, e2, indexing='ij')
                X = np.stack((E1, E2), axis=0)
            else:
                X = self(e1, e2, 0.)

            # eta1-isolines
            for i in range(e1.size):
                plt.plot(X[0, i, :], X[1, i, :], 'tab:blue')

            # eta2-isolines
            for j in range(e2.size):
                plt.plot(X[0, :, j], X[1, :, j], 'tab:blue')
        
        # plot domain with MPI decomposition
        elif isinstance(grid_info, np.ndarray):
            
            assert grid_info.ndim == 2
            assert grid_info.shape[1] > 5
            
            for i in range(grid_info.shape[0]):
                
                e1 = np.linspace(grid_info[i, 0], grid_info[i, 1], int(
                    grid_info[i, 2]) + 1)
                e2 = np.linspace(grid_info[i, 3], grid_info[i, 4], int(
                    grid_info[i, 5]) + 1)

                if logical:
                    E1, E2 = np.meshgrid(e1, e2, indexing='ij')
                    X = np.stack((E1, E2), axis=0)
                else:
                    X = self(e1, e2, 0.)

                # eta1-isolines
                first_line = plt.plot(X[0, 0, :], X[1, 0, :], label='rank=' + str(i))

                for j in range(e1.size):
                    plt.plot(X[0, j, :], X[1, j, :], color=first_line[0].get_color())

                # eta2-isolines
                for k in range(e2.size):
                    plt.plot(X[0, :, k], X[1, :, k], color=first_line[0].get_color())
                    
        else:
            raise ValueError('given grid_info is not supported!')
            

        # plot control points in case of IGA mappings
        if not logical and self.kind_map < 10:
            plt.scatter(self.cx[:, :, 0].flatten(),
                        self.cy[:, :, 0].flatten(),
                        s=3, color='b')

        plt.axis('square')
        
        if isinstance(grid_info, np.ndarray): plt.legend()
        
        plt.xlabel('x')
        plt.ylabel('y')

        if save_dir is not None:
            plt.savefig(save_dir)
        else:
            plt.show()


            
def interp_mapping(Nel, p, spl_kind, X, Y, Z=None):
    """
    Interpolates the mapping (eta1, eta2, eta3) --> (X, Y, Z) on the given spline space.

    Parameters
    -----------
        Nel, p, spl_kind : array-like
            Defining the spline space.

        X, Y : callable
            Either X(eta1, eta2) in 2D or X(eta1, eta2, eta3) in 3D.

        Z : callable 
            3rd mapping component Z(eta1, eta2, eta3) in 3D.

    Returns
    --------
        cx, cy (, cz) : array-like
            The control points.
    """

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
    I_mat = [csc_matrix(bsp.collocation_matrix(T, p, I_pts, kind))
             for T, p, I_pts, kind in zip(T, p, I_pts, spl_kind)]

    # 2D interpolation
    if len(Nel) == 2:
        I = kron(I_mat[0], I_mat[1], format='csc')

        I_pts = np.meshgrid(I_pts[0], I_pts[1], indexing='ij')

        cx = spsolve(I, X(I_pts[0], I_pts[1]).flatten()).reshape(
            NbaseN[0], NbaseN[1])
        cy = spsolve(I, Y(I_pts[0], I_pts[1]).flatten()).reshape(
            NbaseN[0], NbaseN[1])

        return cx, cy

    # 3D interpolation
    elif len(Nel) == 3:
        I = kron(I_mat[0], kron(I_mat[1], I_mat[2]), format='csc')

        I_pts = np.meshgrid(I_pts[0], I_pts[1], I_pts[2], indexing='ij')

        cx = spsolve(I, X(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(
            NbaseN[0], NbaseN[1], NbaseN[2])
        cy = spsolve(I, Y(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(
            NbaseN[0], NbaseN[1], NbaseN[2])
        cz = spsolve(I, Z(I_pts[0], I_pts[1], I_pts[2]).flatten()).reshape(
            NbaseN[0], NbaseN[1], NbaseN[2])

        return cx, cy, cz

    else:
        print('wrong number of elements')

        return 0.


def spline_interpolation_nd(p, grids_1d, values):
    """
    nd spline interpolation with discrete input (nonuniform).

    The knot vector for the clamped spline interpolant is constructed from grids_1d.

    Parameters
    -----------
        p : list 
            Spline degree.

        grids_1d : list of np.arrays
            Interpolation points.

        values: np.array
            Function values at interpolation points. values.shape = (grid1.size, ..., gridn.size).

    Returns
    --------
        coeffs : np.array
            spline coefficients as nd array.

        T : list
            Knot vector of spline interpolant.
    """

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

    I_LU = [splu(csc_matrix(I_mat_i)) for I_mat_i in I_mat]

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