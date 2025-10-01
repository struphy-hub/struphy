# coding: utf-8
"Base classes for mapped domains (single patch)."

from abc import ABCMeta, abstractmethod

import h5py
import numpy as np
from scipy.sparse import csc_matrix, kron
from scipy.sparse.linalg import splu, spsolve

import struphy.bsplines.bsplines as bsp
from struphy.geometry import evaluation_kernels, transform_kernels
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments
from struphy.linear_algebra import linalg_kron


class Domain(metaclass=ABCMeta):
    r"""Base class for mapped domains (single patch).

    The (physical) domain :math:`\Omega \subset \mathbb R^3` is an open subset of :math:`\mathbb R^3`,
    defined by a diffeomorphism

    .. math::

        F:(0, 1)^3 \to \Omega\,,\qquad \boldsymbol{\eta} \mapsto F(\boldsymbol \eta) = \mathbf x\,,

    mapping points :math:`\boldsymbol{\eta} \in (0, 1)^3 = \hat\Omega` of the (logical)
    unit cube to physical points :math:`\mathbf x \in \Omega`.
    The corresponding Jacobain matrix :math:`DF:\hat\Omega \to \mathbb R^{3\times 3}`,
    its volume element :math:`\sqrt g: \hat\Omega \to \mathbb R`
    and the metric tensor :math:`G:\hat\Omega \to \mathbb R^{3\times 3}` are defined by

    .. math::

        DF_{i,j} = \frac{\partial F_i}{\partial \eta_j}\,,\qquad \sqrt g = |\textnormal{det}(DF)|\,,\qquad G = DF^\top DF\,.

    Only right-handed mappings (:math:`\textnormal{det}(DF) > 0`) are admitted.
    """

    def __init__(
        self,
        Nel: tuple[int] = None,
        p: tuple[int] = None,
        spl_kind: tuple[bool] = None,
    ):
        if Nel is None or p is None or spl_kind is None:
            assert self.kind_map >= 10, "Spline mappings must define Nel, p and spl_kind."
            Nel = (1, 1, 1)
            p = (1, 1, 1)
            spl_kind = (True, True, True)

        # create IGA attributes
        self._Nel = Nel
        self._p = p
        self._spl_kind = spl_kind

        self._NbaseN = [Nel + p - kind * p for Nel, p, kind in zip(Nel, p, spl_kind)]

        el_b = [np.linspace(0.0, 1.0, Nel + 1) for Nel in Nel]

        self._T = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(el_b, p, spl_kind)]

        self._indN = [
            (np.indices((Nel, p + 1))[1] + np.arange(Nel)[:, None]) % NbaseN
            for Nel, p, NbaseN in zip(Nel, p, self._NbaseN)
        ]

        # extend to 3d for 2d IGA mappings
        if 0 < self.kind_map < 10:
            self._Nel = (*self._Nel, 0)
            self._p = (*self._p, 0)
            self._NbaseN = self._NbaseN + [0]

            self._T = self._T + [np.zeros((1,), dtype=float)]

            self._indN = self._indN + [np.zeros((1, 1), dtype=int)]

        # create dummy attributes for analytical mappings
        if self.kind_map >= 10:
            self._cx = np.zeros((1, 1, 1), dtype=float)
            self._cy = np.zeros((1, 1, 1), dtype=float)
            self._cz = np.zeros((1, 1, 1), dtype=float)

        self._transformation_ids = {
            "pull": 0,
            "push": 1,
            "tran": 2,
        }

        # keys for performing pull-backs and push-forwards
        dict_pullpush = {
            "0": 0,
            "3": 1,
            "1": 10,
            "2": 11,
            "v": 12,
        }

        # keys for performing transformation
        dict_tran = {
            "0_to_3": 0,
            "3_to_0": 1,
            "1_to_2": 10,
            "2_to_1": 11,
            "norm_to_v": 12,
            "norm_to_1": 13,
            "norm_to_2": 14,
            "v_to_1": 15,
            "v_to_2": 16,
            "1_to_v": 17,
            "2_to_v": 18,
        }

        self._dict_transformations = {
            "pull": dict_pullpush,
            "push": dict_pullpush,
            "tran": dict_tran,
        }

        self._args_domain = DomainArguments(
            self.kind_map,
            self.params_numpy,
            np.array(self.p),
            self.T[0],
            self.T[1],
            self.T[2],
            self.indN[0],
            self.indN[1],
            self.indN[2],
            self.cx.copy(),  # make sure we don't have stride = 0
            self.cy.copy(),  # make sure we don't have stride = 0
            self.cz.copy(),  # make sure we don't have stride = 0
        )

    @property
    def kind_map(self) -> int:
        """Integer defining the mapping:
        * <=9: spline mappings
        * >=10 and <=19: analytical mappings with cubic domain boundary
        * >=20 and <=29: analytical cylinder and torus mappings
        * >=30 and <=39: Shafranov mappings (cylinder)"""
        if not hasattr(self, "_kind_map"):
            raise AttributeError("Must set 'self.kind_map' for mappings.")
        return self._kind_map

    @kind_map.setter
    def kind_map(self, new):
        assert isinstance(new, int)
        self._kind_map = new

    @property
    def params(self) -> dict:
        """Mapping parameters passed to __init__() of the class in domains.py, as dictionary."""
        if not hasattr(self, "_params"):
            self._params = {}
        return self._params

    @params.setter
    def params(self, new):
        assert isinstance(new, dict)
        if "self" in new:
            new.pop("self")
        if "__class__" in new:
            new.pop("__class__")
        self._params = new

    @property
    def params_numpy(self) -> np.ndarray:
        """Mapping parameters as numpy array (can be empty)."""
        if not hasattr(self, "_params_numpy"):
            self._params_numpy = np.array([0], dtype=float)
        return self._params_numpy

    @params_numpy.setter
    def params_numpy(self, new):
        assert isinstance(new, np.ndarray)
        assert new.ndim == 1
        self._params_numpy = new

    @property
    def pole(self) -> bool:
        """Bool; True if mapping has one polar point."""
        if not hasattr(self, "_pole"):
            self._pole = False
        return self._pole

    @pole.setter
    def pole(self, new):
        assert isinstance(new, bool)
        self._pole = new

    @property
    def periodic_eta3(self) -> bool:
        r"""Bool; True if mapping is periodic in :math:`\eta_3` coordinate."""
        if not hasattr(self, "_periodic_eta3"):
            raise AttributeError("Must specify whether mapping is periodic in eta3.")
        return self._periodic_eta3

    @periodic_eta3.setter
    def periodic_eta3(self, new):
        assert isinstance(new, bool)
        self._periodic_eta3 = new

    @property
    def cx(self):
        """3d array of control points for first mapping component :math:`F_x`."""
        return self._cx

    @property
    def cy(self):
        """3d array of control points for second mapping component :math:`F_y`."""
        return self._cy

    @property
    def cz(self):
        """3d array of control points for third mapping component :math:`F_z`."""
        return self._cz

    @property
    def Nel(self):
        """List of number of elements in each direction."""
        return self._Nel

    @property
    def p(self):
        """List of spline degrees in each direction."""
        return self._p

    @property
    def spl_kind(self):
        """List of spline type (True=periodic, False=clamped) in each direction."""
        return self._spl_kind

    @property
    def NbaseN(self):
        """List of number of basis functions for N-splines in each direction."""
        return self._NbaseN

    @property
    def T(self):
        """List of knot vectors for N-splines in each direction."""
        return self._T

    @property
    def indN(self):
        """Global indices of non-vanishing splines in each element. Can be accessed via (element index, local spline index)."""
        return self._indN

    @property
    def args_domain(self):
        """Object for all parameters needed for evaluation of metric coefficients."""
        return self._args_domain

    @property
    def dict_transformations(self):
        """Dictionary of str->int for pull, push and transformation functions."""
        return self._dict_transformations

    def __call__(
        self,
        *etas,
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
        identity_map=False,
    ):
        r"""
        Evaluates the mapping :math:`F : (0, 1)^3 \to \mathbb R^3,\, \boldsymbol \eta \mapsto \mathbf x`.

        Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1.
        The type of evaluation depends on the shape of the input ``etas``.

        Parameters
        ----------
        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:
                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        change_out_order : bool
            If True, the axis corresponding to x, y, z coordinates in the output array is the last one, otherwise the first one.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        identity_map : bool
            If True, not the mapping F, but the identity map (0, 1)^3 --> (0, 1)^3 is evaluated

        Returns
        -------
        out : ndarray | float
            The Cartesian coordinates corresponding to the given logical ones.
        """

        if identity_map:
            which = -1
        else:
            which = 0

        return self._evaluate_metric_coefficient(
            *etas,
            which=which,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            remove_outside=remove_outside,
        )

    def jacobian(
        self,
        *etas,
        transposed=False,
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
    ):
        r"""
        Evaluates the Jacobian matrix :math:`DF : (0, 1)^3 \to \mathbb R^{3 \times 3}`.
        Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1.

        Parameters
        ----------
        transposed : bool
            If True, the transposed Jacobian matrix is evaluated.

        change_out_order : bool
            If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        Returns
        -------
        out : ndarray | float
            The Jacobian matrix evaluated at given logical coordinates.
        """

        return self._evaluate_metric_coefficient(
            *etas,
            which=1,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            transposed=transposed,
            remove_outside=remove_outside,
        )

    def jacobian_det(
        self,
        *etas,
        squeeze_out=False,
        remove_outside=True,
    ):
        r"""
        Evaluates the Jacobian determinant :math:`\sqrt g : (0, 1)^3 \to \mathbb R^+` (only right-handed mappings allowed).
        Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1.

        Parameters
        ----------
        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:
                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        Returns
        -------
        out : ndarray | float
            The Jacobian determinant evaluated at given logical coordinates.
        """

        return self._evaluate_metric_coefficient(
            *etas,
            which=2,
            squeeze_out=squeeze_out,
            remove_outside=remove_outside,
        )

    def jacobian_inv(
        self,
        *etas,
        transposed=False,
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
        avoid_round_off=True,
    ):
        r"""
        Evaluates the inverse Jacobian matrix :math:`DF^{-1} : (0, 1)^3 \to \mathbb R^{3 \times 3}`.
        Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1.

        Parameters
        ----------
        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:
                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        transposed : bool
            If True, the transposed Jacobian matrix is evaluated.

        change_out_order : bool
            If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        avoid_round_off : bool
            Whether to manually set exact zeros in arrays.

        Returns
        -------
        out : ndarray | float
            The inverse Jacobian matrix evaluated at given logical coordinates.
        """

        return self._evaluate_metric_coefficient(
            *etas,
            which=3,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            transposed=transposed,
            remove_outside=remove_outside,
            avoid_round_off=avoid_round_off,
        )

    def metric(
        self,
        *etas,
        transposed=False,
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
        avoid_round_off=True,
    ):
        r"""
        Evaluates the metric tensor :math:`G: (0, 1)^3 \to \mathbb R^{3\times 3}`.
        Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1.

        Parameters
        ----------
        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:
                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        transposed : bool
            If True, the transposed Jacobian matrix is evaluated.

        change_out_order : bool
            If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        avoid_round_off : bool
            Whether to manually set exact zeros in arrays.

        Returns
        -------
        out : ndarray | float
            The metric tensor evaluated at given logical coordinates.
        """

        return self._evaluate_metric_coefficient(
            *etas,
            which=4,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            transposed=transposed,
            remove_outside=remove_outside,
            avoid_round_off=avoid_round_off,
        )

    def metric_inv(
        self,
        *etas,
        transposed=False,
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
        avoid_round_off=True,
    ):
        r"""
        Evaluates the inverse metric tensor :math:`G^{-1}: (0, 1)^3 \to \mathbb R^{3\times 3}`.
        Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1.

        Parameters
        ----------
        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:
                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        transposed : bool
            If True, the transposed Jacobian matrix is evaluated.

        change_out_order : bool
            If True, the axes corresponding to the 3x3 entries in the output array are the last two, otherwise the first two.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        avoid_round_off : bool
            Whether to manually set exact zeros in arrays.

        Returns
        -------
        out : ndarray | float
            The inverse metric tensor evaluated at given logical coordinates.
        """

        return self._evaluate_metric_coefficient(
            *etas,
            which=5,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            transposed=transposed,
            remove_outside=remove_outside,
            avoid_round_off=avoid_round_off,
        )

    def pull(
        self,
        a,
        *etas,
        flat_eval=False,
        kind="0",
        a_kwargs={},
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
        coordinates="physical",
    ):
        """Pull-back of a Cartesian scalar/vector field to a differential p-form.

        Parameters
        ----------
        a : callable | list | tuple | array-like
            The function a(x, y, z) resp. [a_x(x, y, z), a_y(x, y, z), a_z(x, y, z)] to be pulled.

        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:

            1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
            2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        flat_eval : bool
            Allows to perform flat evaluation when len(etas) == 3 with 1D arrays of same size.

        kind : str
            Which pull-back to apply, '0', '1', '2', '3' or 'v'.

        a_kwargs : dict
            Keyword arguments passed to parameter "a" if "a" is a callable: is called as a(*etas, **a_kwargs).

        change_out_order : bool
            If True, the axes corresponding to the 3 components in the output array are the last two, otherwise the first two.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        coordinates : str
            In which coordinates the input "a" is given (in case of callables). "physical" : a = a(x, y, z).
            "logical"  : a = a(eta1, eta2, eta3).

        Returns
        -------
        out : ndarray | float
            Pullback of Cartesian vector/scalar field to p-form evaluated at given logical coordinates.
        """

        return self._pull_push_transform(
            "pull",
            a,
            kind,
            *etas,
            flat_eval=flat_eval,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            remove_outside=remove_outside,
            coordinates=coordinates,
            a_kwargs=a_kwargs,
        )

    def push(
        self,
        a,
        *etas,
        flat_eval=False,
        kind="0",
        a_kwargs={},
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
    ):
        """Pushforward of a differential p-form to a Cartesian scalar/vector field.

        Parameters
        -----------
        a : callable | list | tuple | array-like
            The function a(e1, e2, e3) resp. [a_1(e1, e2, e3), a_2(e1, e2, e3), a_3(e1, e2, e3)] to be pushed.

        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:

                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        flat_eval : bool
            Allows to perform flat evaluation when len(etas) == 3 with 1D arrays of same size.

        kind : str
            Which pushforward to apply, '0', '1', '2', '3' or 'v'.

        a_kwargs : dict
            Keyword arguments passed to parameter "a" if "a" is a callable: is called as a(*etas, **a_kwargs).

        change_out_order : bool
            If True, the axes corresponding to the 3 components in the output array are the last two, otherwise the first two.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        Returns
        -------
        out : ndarray | float
            Pushforward of p-form to Cartesian vector/scalar field evaluated at given logical coordinates.
        """

        return self._pull_push_transform(
            "push",
            a,
            kind,
            *etas,
            flat_eval=flat_eval,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            remove_outside=remove_outside,
            a_kwargs=a_kwargs,
        )

    def transform(
        self,
        a,
        *etas,
        flat_eval=False,
        kind="0_to_3",
        a_kwargs={},
        change_out_order=False,
        squeeze_out=False,
        remove_outside=True,
    ):
        """Transformation between different differential p-forms and/or vector fields.

        Parameters
        -----------
        a : callable | list | tuple | array-like
            The function a(e1, e2, e3) resp. [a_1(e1, e2, e3), a_2(e1, e2, e3), a_3(e1, e2, e3)] to be transformed.

        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:

                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        flat_eval : bool
            Allows to perform flat evaluation when len(etas) == 3 with 1D arrays of same size.

        kind : str
            Which transformation to apply, such as '0_to_3' for example, see dict_transformations['tran'] for all options.

        a_kwargs : dict
            Keyword arguments passed to parameter "a" if "a" is a callable: is called as a(*etas, **a_kwargs).

        change_out_order : bool
            If True, the axes corresponding to the 3 components in the output array are the last two, otherwise the first two.

        squeeze_out : bool
            Whether to remove singleton dimensions in output array.

        remove_outside : bool
            If True, logical coordinates outside of (0, 1)^3 are NOT evaluated to -1 and are removed in the output array.

        Returns
        -------
        out : ndarray | float
            Transformed p-form evaluated at given logical coordinates.

        Notes
        -----
        Possible choices for kind are '0_to_3', '3_to_0', '1_to_2', '2_to_1', 'norm_to_v', 'norm_to_1', 'norm_to_2', 'v_to_1', 'v_to_2', '1_to_v' and '2_to_v'.
        """

        return self._pull_push_transform(
            "tran",
            a,
            kind,
            *etas,
            flat_eval=flat_eval,
            change_out_order=change_out_order,
            squeeze_out=squeeze_out,
            remove_outside=remove_outside,
            a_kwargs=a_kwargs,
        )

    # ========================
    # private methods :
    # ========================

    # ================================
    def _evaluate_metric_coefficient(self, *etas, which=0, **kwargs):
        """Evaluates metric coefficients. Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1 for markers evaluation.

        Parameters
        ----------
        *etas : array-like | tuple
            Logical coordinates at which to evaluate. Two cases are possible:

                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        which : int
            Which metric coefficients to be evaluated (0 : F, 1 : DF, 2 : det(DF), 3 : DF^(-1), 4 : G, 5 : G^(-1)).

        **kwargs
            Addtional boolean keyword arguments (transposed, change_out_order, squeeze_out, remove_outside, avoid_round_off).

        Returns
        -------
        out : ndarray | float
            The metric coefficient evaluated at the given logical coordinates.
        """

        # set default values
        transposed = kwargs.get("transposed", False)
        change_out_order = kwargs.get("change_out_order", False)
        squeeze_out = kwargs.get("squeeze_out", True)
        remove_outside = kwargs.get("remove_outside", False)
        avoid_round_off = kwargs.get("avoid_round_off", True)

        # markers evaluation
        if len(etas) == 1:
            markers = etas[0]

            # to keep C-ordering the (3, 3)-part is in the last indices
            out = np.empty((markers.shape[0], 3, 3), dtype=float)

            n_inside = evaluation_kernels.kernel_evaluate_pic(
                markers,
                which,
                self.args_domain,
                out,
                remove_outside,
                avoid_round_off,
            )

            # move the (3, 3)-part to front
            out = np.transpose(out, axes=(1, 2, 0))

            # remove holes
            out = out[:, :, :n_inside]

            if transposed:
                out = np.transpose(out, axes=(1, 0, 2))

            # change size of "out" depending on which metric coeff has been evaluated
            if which == 0 or which == -1:
                out = out[:, 0, :]
                if change_out_order:
                    out = np.transpose(out, axes=(1, 0))
            elif which == 2:
                out = out[0, 0, :]
            else:
                if change_out_order:
                    out = np.transpose(out, axes=(2, 0, 1))

        # tensor-product/slice evaluation
        else:
            E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
                etas[0],
                etas[1],
                etas[2],
                flat_eval=False,
            )

            # to keep C-ordering the (3, 3)-part is in the last indices
            out = np.empty(
                (E1.shape[0], E2.shape[1], E3.shape[2], 3, 3),
                dtype=float,
            )
            evaluation_kernels.kernel_evaluate(
                E1,
                E2,
                E3,
                which,
                self.args_domain,
                out,
                is_sparse_meshgrid,
                avoid_round_off,
            )

            # move the (3, 3)-part to front
            out = np.transpose(out, axes=(3, 4, 0, 1, 2))

            if transposed:
                out = np.transpose(out, axes=(1, 0, 2, 3, 4))

            if which == 0:
                out = out[:, 0, :, :, :]
                if change_out_order:
                    out = np.transpose(out, axes=(1, 2, 3, 0))
            elif which == 2:
                out = out[0, 0, :, :, :]
            else:
                if change_out_order:
                    out = np.transpose(out, axes=(2, 3, 4, 0, 1))

            # remove singleton dimensions for slice evaluation
            if squeeze_out:
                out = out.squeeze()

            # remove all "dimensions" for point-wise evaluation
            if out.ndim == 0:
                out = out.item()

        if isinstance(out, float):
            return out
        else:
            return out.copy()

    # ================================
    def _pull_push_transform(self, which, a, kind_fun, *etas, flat_eval=False, **kwargs):
        """Evaluates metric coefficients. Logical coordinates outside of :math:`(0, 1)^3` are evaluated to -1 for markers evaluation.

        Parameters
        ----------
        which : str
            Which transformation to apply (one of "pull", "push" or "tran").

        a : callable | list | tuple | array-like
            The function/values to be transformed.

        kind_fun : str
            The kind of transformation (e.g. "0" or "1" in case of which="pull").

        *etas : array-like| tuple
            Logical coordinates at which to evaluate. Two cases are possible:

                1. 2d numpy array, where coordinates are taken from eta1 = etas[:, 0], eta2 = etas[:, 1], etc. (like markers).
                2. list/tuple (eta1, eta2, ...), where eta1, eta2, ... can be float or array-like of various shapes.

        flat_eval : bool
            Allows to perform flat evaluation when len(etas) == 3 with 1D arrays of same size.

        **kwargs
            Addtional keyword arguments (coordinates, change_out_order, squeeze_out, remove_outside, a_kwargs).

        Returns
        -------
        out : ndarray | float
            4D or 2D (for flat eval) array holding the metric coefficient (first index),
            evaluated at the given logical coordinates (last three indices).
        """

        # set default values
        coordinates = kwargs.get("coordinates", "logical")
        change_out_order = kwargs.get("change_out_order", False)
        squeeze_out = kwargs.get("squeeze_out", True)
        remove_outside = kwargs.get("remove_outside", False)
        a_kwargs = kwargs.get("a_kwargs", {})

        # kind of transformation
        kind_int = self.dict_transformations[which][kind_fun]

        # markers evaluation
        if len(etas) == 1 or flat_eval:
            if flat_eval:
                assert len(etas) == 3
                assert etas[0].shape == etas[1].shape == etas[2].shape
                assert etas[0].ndim == 1
                markers = np.stack(etas, axis=1)
            else:
                markers = etas[0]

            # coordinates (:, 3) and argument evaluation (without holes)
            if callable(a):
                if coordinates == "logical":
                    A = Domain.prepare_arg(
                        a,
                        self(
                            markers,
                            change_out_order=True,
                            remove_outside=remove_outside,
                            identity_map=True,
                        ),
                    )
                else:
                    A = Domain.prepare_arg(
                        a,
                        self(markers, change_out_order=True, remove_outside=remove_outside),
                    )

            elif isinstance(a, (list, tuple)):
                if callable(a[0]):
                    if coordinates == "logical":
                        A = Domain.prepare_arg(
                            a,
                            self(
                                markers,
                                change_out_order=True,
                                remove_outside=remove_outside,
                                identity_map=True,
                            ),
                        )
                    else:
                        A = Domain.prepare_arg(
                            a,
                            self(markers, change_out_order=True, remove_outside=remove_outside),
                        )
                else:
                    A = Domain.prepare_arg(a, markers)

            else:
                A = Domain.prepare_arg(a, markers)

            # check if A includes holes or not
            if A.shape[0] == markers.shape[0]:
                A_has_holes = True
            else:
                A_has_holes = False

            # call evaluation kernel
            out = np.empty((markers.shape[0], 3), dtype=float)

            # make sure we don't have stride = 0
            A = A.copy()

            n_inside = transform_kernels.kernel_pullpush_pic(
                A,
                markers,
                self._transformation_ids[which],
                kind_int,
                self.args_domain,
                out,
                remove_outside,
            )

            # move the (3, 3)-part to front
            out = np.transpose(out, axes=(1, 0))

            # remove holes
            out = out[:, :n_inside]

            # check if A has correct shape
            if not A_has_holes and remove_outside:
                assert A.shape[0] == out.shape[1]

            # change output order
            if kind_int < 10:
                out = out[0, :]
            else:
                if change_out_order:
                    out = np.transpose(out, axes=(1, 0))

        # tensor-product/slice evaluation
        else:
            # convert evaluation points to 3d array of appropriate shape
            E1, E2, E3, is_sparse_meshgrid = Domain.prepare_eval_pts(
                etas[0],
                etas[1],
                etas[2],
                flat_eval=False,
            )

            # convert input to be transformed (a) to 4d array of appropriate shape
            if coordinates == "logical":
                A = Domain.prepare_arg(
                    a,
                    E1,
                    E2,
                    E3,
                    is_sparse_meshgrid=is_sparse_meshgrid,
                    a_kwargs=a_kwargs,
                )
            else:
                X = self(E1, E2, E3)
                A = Domain.prepare_arg(a, X[0], X[1], X[2], a_kwargs=a_kwargs)

            # call evaluation kernel
            out = np.empty(
                (E1.shape[0], E2.shape[1], E3.shape[2], 3),
                dtype=float,
            )
            transform_kernels.kernel_pullpush(
                A,
                E1,
                E2,
                E3,
                self._transformation_ids[which],
                kind_int,
                self.args_domain,
                is_sparse_meshgrid,
                out,
            )

            # move the (3, 3)-part to front
            out = np.transpose(out, axes=(3, 0, 1, 2))

            # change output order
            if kind_int < 10:
                out = out[0, :, :, :]
            else:
                if change_out_order:
                    out = np.transpose(out, axes=(1, 2, 3, 0))

            # remove singleton dimensions for slice evaluation
            if squeeze_out:
                out = out.squeeze()

            # remove all "dimensions" for point-wise evaluation
            if out.ndim == 0:
                out = out.item()

        if isinstance(out, float):
            return out
        else:
            return out.copy()

    # ========================
    # static methods :
    # ========================

    # ================================
    @staticmethod
    def prepare_eval_pts(x, y, z, flat_eval=False):
        """Broadcasts evaluation point sets to 3d arrays of correct shape.

        Parameters
        ----------
        x, y, z : float | int | list | array-like
            Evaluation point sets.

        flat_eval : bool
            Whether to do a flat evaluation, i.e. f([e11, e12], [e21, e22]) = [f(e11, e21), f(e12, e22)].

        Returns
        -------
        E1, E2, E3 : array-like
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
                raise ValueError("Input x must be a 1d list or numpy array")

            if isinstance(y, list):
                arg_y = np.array(y)
            elif isinstance(y, np.ndarray):
                arg_y = y
            else:
                raise ValueError("Input y must be a 1d list or numpy array")

            if isinstance(z, list):
                arg_z = np.array(z)
            elif isinstance(z, np.ndarray):
                arg_z = z
            else:
                raise ValueError("Input z must be a 1d list or numpy array")

            assert arg_x.ndim == arg_y.ndim == arg_z.ndim == 1
            assert arg_x.size == arg_y.size == arg_z.size

            E1 = arg_x[:, None, None]
            E2 = arg_y[:, None, None]
            E3 = arg_z[:, None, None]

            # Make sure we don't have stride 0
            return E1.copy(), E2.copy(), E3.copy(), is_sparse_meshgrid

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
                arg_x = x.copy()
            else:
                raise ValueError(f"data type {type(x)} not supported")

            if isinstance(y, float):
                arg_y = np.array([y])
            elif isinstance(y, int):
                arg_y = np.array([float(y)])
            elif isinstance(y, list):
                arg_y = np.array(y)
            elif isinstance(y, np.ndarray):
                arg_y = y.copy()
            else:
                raise ValueError(f"data type {type(y)} not supported")

            if isinstance(z, float):
                arg_z = np.array([z])
            elif isinstance(z, int):
                arg_z = np.array([float(z)])
            elif isinstance(z, list):
                arg_z = np.array(z)
            elif isinstance(z, np.ndarray):
                arg_z = z.copy()
            else:
                raise ValueError(f"data type {type(z)} not supported")

            # tensor-product for given three 1D arrays
            if arg_x.ndim == 1 and arg_y.ndim == 1 and arg_z.ndim == 1:
                E1, E2, E3 = np.meshgrid(arg_x, arg_y, arg_z, indexing="ij")
            # given xy-plane at point z:
            elif arg_x.ndim == 2 and arg_y.ndim == 2 and arg_z.size == 1:
                E1 = arg_x[:, :, None]
                E2 = arg_y[:, :, None]
                E3 = arg_z * np.ones(E1.shape)
            # given xz-plane at point y:
            elif arg_x.ndim == 2 and arg_y.size == 1 and arg_z.ndim == 2:
                E1 = arg_x[:, None, :]
                E2 = arg_y * np.ones(E1.shape)
                E3 = arg_z[:, None, :]
            # given yz-plane at point x:
            elif arg_x.size == 1 and arg_y.ndim == 2 and arg_z.ndim == 2:
                E2 = arg_y[None, :, :]
                E3 = arg_z[None, :, :]
                E1 = arg_x * np.ones(E2.shape)
            # given three 3D arrays
            elif arg_x.ndim == 3 and arg_y.ndim == 3 and arg_z.ndim == 3:
                # Distinguish if input coordinates are from sparse or dense meshgrid.
                # Sparse: arg_x.shape = (n1, 1, 1), arg_y.shape = (1, n2, 1), arg_z.shape = (1, 1, n3)
                # Dense : arg_x.shape = (n1, n2, n3), arg_y.shape = (n1, n2, n3) arg_z.shape = (n1, n2, n3)
                E1, E2, E3 = arg_x, arg_y, arg_z

                # `arg_x` `arg_y` `arg_z` are all sparse meshgrids.
                if (
                    arg_x.shape[1] == 1
                    and arg_x.shape[2] == 1
                    and arg_y.shape[0] == 1
                    and arg_y.shape[2] == 1
                    and arg_z.shape[0] == 1
                    and arg_z.shape[1] == 1
                ):
                    is_sparse_meshgrid = True
                # one of `arg_x` `arg_y` `arg_z` is a dense meshgrid.(i.e., all are dense meshgrid) Process each point as default.

            else:
                raise ValueError("Argument dimensions not supported")

            # Make sure we don't have stride 0
            return E1.copy(), E2.copy(), E3.copy(), is_sparse_meshgrid

    # ================================
    @staticmethod
    def prepare_arg(a_in, *Xs, is_sparse_meshgrid=False, a_kwargs={}):
        """Broadcasts argument to be pulled, pushed or transformed to array of correct shape (2d for markers, 4d else).

        Parameters
        ----------
        a_in : callable | list | tuple | array-like
            The argument to be pulled, pushed or transformed.

        *Xs : array-like | tuple
            The evaluation point sets. Obtained from prepare_eval_pts function.

        is_sparse_meshgrid : bool
            Whether arguments fit sparse_meshgrid shape. Obtained from prepare_eval_pts function.

        a_kwargs : dict
            Keyword arguments passed to parameter "a_in" if "a_in" is a callable: is called as a_in(*etas, **a_kwargs).

        Returns
        -------
        a_out : array-like
            The 2d/4d array suitable for evaluation kernels.
        """

        if len(Xs) == 1:
            flat_eval = True
        else:
            flat_eval = False

        # float (point-wise, scalar function)
        if isinstance(a_in, float):
            a_out = np.array([[[[a_in]]]])

        # single callable:
        # scalar function -> must return a 3d array for 3d evaluation points
        # vector-valued function -> must return a 4d array of shape (3,:,:,:)
        elif callable(a_in):
            if flat_eval:
                a_out = a_in(Xs[0][:, 0], Xs[0][:, 1], Xs[0][:, 2], **a_kwargs)
                if a_out.ndim == 1:
                    a_out = a_out[None, :]
            else:
                if is_sparse_meshgrid:
                    a_out = a_in(
                        *np.meshgrid(Xs[0][:, 0, 0], Xs[1][0, :, 0], Xs[2][0, 0, :], indexing="ij"),
                        **a_kwargs,
                    )
                else:
                    a_out = a_in(*Xs, **a_kwargs)

                # case of Field.__call__
                if isinstance(a_out, list):
                    a_out = np.array(a_out)

                if a_out.ndim == 3:
                    a_out = a_out[None, :, :, :]

        # list/tuple of length 1 or 3 containing:
        # callable(s) that must return 3d array(s) for 3d evaluation points
        # 1d array(s) (flat_eval=True)
        # 3d array(s) (flat eval=False)
        elif isinstance(a_in, (list, tuple)):
            assert len(a_in) == 1 or len(a_in) == 3

            a_out = []
            for component in a_in:
                if callable(component):
                    if flat_eval:
                        a_out += [
                            component(
                                Xs[0][:, 0],
                                Xs[0][:, 1],
                                Xs[0][:, 2],
                                **a_kwargs,
                            ),
                        ]
                    else:
                        if is_sparse_meshgrid:
                            a_out += [
                                component(
                                    *np.meshgrid(
                                        Xs[0][:, 0, 0],
                                        Xs[1][0, :, 0],
                                        Xs[2][0, 0, :],
                                        indexing="ij",
                                    ),
                                    **a_kwargs,
                                ),
                            ]
                        else:
                            a_out += [component(*Xs, **a_kwargs)]

                elif isinstance(component, np.ndarray):
                    if flat_eval:
                        assert component.ndim == 1, print(f"{component.ndim = }")
                    else:
                        assert component.ndim == 3, print(f"{component.ndim = }")

                    a_out += [component]

                elif isinstance(component, float):
                    a_out += [np.array([component])[:, None, None]]

            a_out = np.array(a_out, dtype=float)

        # numpy array:
        # 1d array (flat_eval=True and scalar input or flat_eval=False and length 1 (scalar) or length 3 (vector))
        # 2d array (flat_eval=True and vector-valued input of shape (3,:))
        # 3d array (flat_eval=False and scalar input)
        # 4d array (flat_eval=False and vector-valued input of shape (3,:,:,:))
        elif isinstance(a_in, np.ndarray):
            if flat_eval:
                if a_in.ndim == 1:
                    a_out = a_in[None, :]
                elif a_in.ndim == 2:
                    a_out = a_in[:, :]
                else:
                    raise ValueError(
                        "Input array a_in must be either 1d (scalar) or \
                    2d (vector-valued, shape (3,:)) for flat evaluation!"
                    )

            else:
                # point-wise evaluation for scalar (len=1) or vector (len=3) input
                if a_in.ndim == 1:
                    assert a_in.size == 1 or a_in.size == 3
                    a_out = a_in[:, None, None, None]

                # tensor-product evaluation (scalar)
                elif a_in.ndim == 3:
                    a_out = a_in[None, :, :, :]

                # tensor-product evaluation (vector)
                elif a_in.ndim == 4:
                    a_out = a_in[:, :, :, :]
                else:
                    raise ValueError(
                        "Input array a_in must be either 3d (scalar) or \
                    4d (vector-valued, shape (3,:,:,:)) for non-flat evaluation!"
                    )

        else:
            raise TypeError(
                "Argument a must be either a float OR a list/tuple of 1 or 3 callable(s)/numpy array(s)/float(s) \
            OR a single numpy array OR a single callable!"
            )

        # make sure that output array is 2d and of shape (:, 1) or (:, 3) for flat evaluation
        if flat_eval:
            assert a_out.ndim == 2
            assert a_out.shape[0] == 1 or a_out.shape[0] == 3
            a_out = np.ascontiguousarray(np.transpose(a_out, axes=(1, 0))).copy()  # Make sure we don't have stride 0

        # make sure that output array is 4d and of shape (:,:,:, 1) or (:,:,:, 3) for tensor-product/slice evaluation
        else:
            assert a_out.ndim == 4
            assert a_out.shape[0] == 1 or a_out.shape[0] == 3
            a_out = np.ascontiguousarray(
                np.transpose(a_out, axes=(1, 2, 3, 0)),
            ).copy()  # Make sure we don't have stride 0

        return a_out

    # ================================

    def get_params_numpy(self) -> np.ndarray:
        """Convert parameter dict into numpy array."""
        params_numpy = []
        for k, v in self.params.items():
            params_numpy.append(v)
        return np.array(params_numpy)

    def show(
        self,
        logical=False,
        grid_info=None,
        markers=None,
        marker_coords="logical",
        show_control_pts=False,
        figsize=(12, 5),
        save_dir=None,
    ):
        """Plots isolines (and control point in case on spline mappings) of the 2D physical domain for eta3 = 0.
        Markers can be plotted as well (optional).

        Parameters
        ----------
        logical : bool
            Whether to plot the physical domain (False) or logical domain (True).

        plane : str
            Which physical coordinates to plot (xy, xz or yz) in case of logical=False.

        grid_info : array-like
            Information about the grid. If not given, the domain is shown with high resolution. If given, can be either
                * a list of # of elements [Nel1, Nel2, (Nel3)], OR
                * a 2d array with information about MPI decomposition.

        markers : array-like
            Markers to be plotted. Can be of shape (Np, 3) or (:, Np, 3). For the former, all markers are plotted with the same color. For the latter, with different colors (are interpreted as orbits in time).

        marker_coords : bool
            Whether the marker coordinates are logical or physical.

        save_dir : str
            If given, the figure is saved according the given directory.
        """

        import matplotlib.pyplot as plt

        is_not_cube = self.kind_map < 10 or self.kind_map > 19
        torus_mappings = (
            "Tokamak",
            "GVECunit",
            "DESCunit",
            "IGAPolarTorus",
            "HollowTorus",
        )

        # plot domain without MPI decomposition and high resolution
        if grid_info is None:
            e1 = np.linspace(0.0, 1.0, 16)
            e2 = np.linspace(0.0, 1.0, 65)

            if logical:
                E1, E2 = np.meshgrid(e1, e2, indexing="ij")
                X = np.stack((E1, E2), axis=0)
            else:
                XYZ = self(e1, e2, 0.0, squeeze_out=True)

            X = XYZ[0]
            if self.__class__.__name__ in torus_mappings:
                Y = XYZ[2]
            else:
                Y = XYZ[1]

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 2, 1)

            # eta1-isolines
            for i in range(e1.size):
                ax.plot(X[i, :], Y[i, :], "tab:blue", alpha=0.5, zorder=0)

            # eta2-isolines
            for j in range(e2.size - int(is_not_cube)):
                ax.plot(X[:, j], Y[:, j], "tab:blue", alpha=0.5, zorder=0)

            ax.scatter(X[0, 0], Y[0, 0], 20, "red", zorder=10)
            if is_not_cube:
                ax.scatter(X[0, 32], Y[0, 32], 20, "red", zorder=10)

            tstr = ""
            for key, val in self.params.items():
                if key not in {"cx", "cy", "cz"}:
                    tstr += key + ": " + str(val) + "\n"
            ax.set_title(self.__class__.__name__ + " at $\\eta_3=0$")
            ax.text(
                0.01,
                0.99,
                tstr,
                ha="left",
                va="top",
                transform=ax.transAxes,
            )

            # top view
            e3 = np.linspace(0.0, 1.0, 65)

            if logical:
                E1, E2 = np.meshgrid(e1, e2, indexing="ij")
                X = np.stack((E1, E2), axis=0)
            else:
                theta_0 = self(e1, 0.0, e3, squeeze_out=True)
                theta_pi = self(e1, 0.5, e3, squeeze_out=True)

            X_0 = theta_0[0]
            X_pi = theta_pi[0]
            if self.__class__.__name__ in torus_mappings:
                Z_0 = theta_0[1]
                Z_pi = theta_pi[1]
            else:
                Z_0 = theta_0[2]
                Z_pi = theta_pi[2]

            ax2 = fig.add_subplot(1, 2, 2)

            # eta1-isolines
            for i in range(e1.size):
                ax2.plot(X_0[i, :], Z_0[i, :], "tab:blue", alpha=0.5, zorder=0)

            # eta3-isolines
            for j in range(e2.size):
                ax2.plot(X_0[:, j], Z_0[:, j], "tab:blue", alpha=0.5, zorder=0)

            if is_not_cube:
                # eta1-isolines
                for i in range(e1.size):
                    ax2.plot(
                        X_pi[i, :],
                        Z_pi[i, :],
                        "tab:blue",
                        alpha=0.5,
                        zorder=0,
                    )

                # eta3-isolines
                for j in range(e2.size):
                    ax2.plot(
                        X_pi[:, j],
                        Z_pi[:, j],
                        "tab:blue",
                        alpha=0.5,
                        zorder=0,
                    )

            # magnetic axis
            ax2.plot(X_0[0, :], Z_0[0, :], "tab:red", alpha=1.0, zorder=10)
            ax2.plot(X_pi[0, :], Z_pi[0, :], "tab:red", alpha=1.0, zorder=10)

            if self.__class__.__name__ in torus_mappings:
                ylab = "y"
            else:
                ylab = "z"
            ax2.set_xlabel("x")
            ax2.set_ylabel(ylab)
            ax2.set_title("top view")
            ax2.axis("equal")

            # coordinates
            # e3 = [0., .25, .5, .75]
            # x, y, z = self(e1, e2, e3)
            # R = np.sqrt(x**2 + y**2)

            # fig = plt.figure(figsize=(13, 13))
            # for n in range(4):
            #     plt.subplot(2, 2, n + 1)
            #     plt.contourf(R[:, :, n], z[:, :, n], x[:, :, n])
            #     plt.title(f'x at {e3[n] = }')
            #     plt.colorbar()

            # fig = plt.figure(figsize=(13, 13))
            # for n in range(4):
            #     plt.subplot(2, 2, n + 1)
            #     plt.contourf(R[:, :, n], z[:, :, n], y[:, :, n])
            #     plt.title(f'y at {e3[n] = }')
            #     plt.colorbar()

            # fig = plt.figure(figsize=(13, 13))
            # for n in range(4):
            #     plt.subplot(2, 2, n + 1)
            #     plt.contourf(R[:, :, n], z[:, :, n], z[:, :, n])
            #     plt.title(f'z at {e3[n] = }')
            #     plt.colorbar()

        # plot domain according to given grid [nel1, nel2, (nel3)]
        elif isinstance(grid_info, list):
            assert len(grid_info) > 1

            e1 = np.linspace(0.0, 1.0, grid_info[0] + 1)
            e2 = np.linspace(0.0, 1.0, grid_info[1] + 1)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

            if logical:
                E1, E2 = np.meshgrid(e1, e2, indexing="ij")

                # eta1-isolines
                for i in range(e1.size):
                    ax.plot(E1[i, :], E2[i, :], "tab:blue", alpha=0.5)

                # eta2-isolines
                for j in range(e2.size):
                    ax.plot(E1[:, j], E2[:, j], "tab:blue", alpha=0.5)

            else:
                X = self(e1, e2, 0.0, squeeze_out=True)

                # plot xz-plane for torus mappings, xy-plane else
                if self.__class__.__name__ in torus_mappings:
                    co1, co2 = 0, 2
                else:
                    co1, co2 = 0, 1

                # eta1-isolines
                for i in range(e1.size):
                    ax.plot(X[co1, i, :], X[co2, i, :], "tab:blue", alpha=0.5)

                # eta2-isolines
                for j in range(e2.size):
                    ax.plot(X[co1, :, j], X[co2, :, j], "tab:blue", alpha=0.5)

        # plot domain with MPI decomposition
        elif isinstance(grid_info, np.ndarray):
            assert grid_info.ndim == 2
            assert grid_info.shape[1] > 5

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)

            for i in range(grid_info.shape[0]):
                e1 = np.linspace(
                    grid_info[i, 0],
                    grid_info[i, 1],
                    int(
                        grid_info[i, 2],
                    )
                    + 1,
                )
                e2 = np.linspace(
                    grid_info[i, 3],
                    grid_info[i, 4],
                    int(
                        grid_info[i, 5],
                    )
                    + 1,
                )

                if logical:
                    E1, E2 = np.meshgrid(e1, e2, indexing="ij")

                    # eta1-isolines
                    first_line = ax.plot(
                        E1[0, :],
                        E2[0, :],
                        label="rank=" + str(i),
                        alpha=0.25,
                    )

                    for j in range(e1.size):
                        ax.plot(
                            E1[j, :],
                            E2[j, :],
                            color=first_line[0].get_color(),
                            alpha=0.25,
                        )

                    # eta2-isolines
                    for k in range(e2.size):
                        ax.plot(
                            E1[:, k],
                            E2[:, k],
                            color=first_line[0].get_color(),
                            alpha=0.25,
                        )

                else:
                    X = self(e1, e2, 0.0, squeeze_out=True)

                    # plot xz-plane for torus mappings, xy-plane else
                    if self.__class__.__name__ in torus_mappings:
                        co1, co2 = 0, 2
                    else:
                        co1, co2 = 0, 1

                    # eta1-isolines
                    first_line = ax.plot(
                        X[co1, 0, :],
                        X[co2, 0, :],
                        label="rank=" + str(i),
                        alpha=0.25,
                    )

                    for j in range(e1.size):
                        ax.plot(
                            X[co1, j, :],
                            X[co2, j, :],
                            color=first_line[0].get_color(),
                            alpha=0.25,
                        )

                    # eta2-isolines
                    for k in range(e2.size):
                        ax.plot(
                            X[co1, :, k],
                            X[co2, :, k],
                            color=first_line[0].get_color(),
                            alpha=0.25,
                        )

        else:
            raise ValueError("given grid_info is not supported!")

        # plot control points in case of IGA mappings
        if not logical and self.kind_map < 10 and show_control_pts:
            if self.__class__.__name__ == "GVECunit" or self.__class__.__name__ == "DESCunit":
                Yc = self.cz[:, :, 0].flatten()
            else:
                Yc = self.cy[:, :, 0].flatten()
            ax.scatter(self.cx[:, :, 0].flatten(), Yc, s=1, color="b")

        # plot given markers
        if markers is not None:
            assert not (logical and marker_coords != "logical")

            if self.__class__.__name__ in torus_mappings:
                co1, co2 = 0, 2
            else:
                co1, co2 = 0, 1

            # no time series: plot all markers with the same color
            if markers.ndim == 2:
                if not logical and marker_coords == "logical":
                    tmp = markers.copy()  # TODO: needed for eta3=0
                    tmp[:, 2] = 0.0  # TODO: needed for eta3=0
                    X = self(tmp, remove_outside=True)
                else:
                    X = (
                        markers[:, 0].copy(),
                        markers[
                            :,
                            1,
                        ].copy(),
                        markers[:, 2].copy(),
                    )

                ax.scatter(X[co1], X[co2], s=1, color="b")

            # time series: plot markers with different colors
            elif markers.ndim == 3:
                colors = ["k", "m", "b", "g", "r", "c", "y"]

                for i in range(markers.shape[1]):
                    if not logical and marker_coords == "logical":
                        # TODO: needed for eta3=0
                        tmp = markers[:, i, :].copy()
                        tmp[:, 2] = 0.0  # TODO: needed for eta3 = 0
                        X = self(tmp, remove_outside=True, squeeze_out=True)
                    else:
                        X = (
                            markers[:, i, 0].copy(),
                            markers[
                                :,
                                i,
                                1,
                            ].copy(),
                            markers[:, i, 2].copy(),
                        )

                    # ax.scatter(X[co1], X[co2], s=2, color=colors[i%len(colors)])
                    ax.scatter(X[co1], X[co2], s=2)

        ax.axis("equal")

        if isinstance(grid_info, np.ndarray):
            plt.legend()

        if self.__class__.__name__ in torus_mappings:
            ylab = "z"
        else:
            ylab = "y"
        ax.set_xlabel("x")
        ax.set_ylabel(ylab)

        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches="tight")
        else:
            plt.show()


class Spline(Domain):
    r"""3D IGA spline mapping.

    .. math::

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ijk} c^x_{ijk} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,,

        y &= \sum_{ijk} c^y_{ijk} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,,

        z &= \sum_{ijk} c^z_{ijk} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,.
        \end{aligned}\right.
    """

    def __init__(
        self,
        Nel: tuple[int] = (8, 24, 6),
        p: tuple[int] = (2, 3, 1),
        spl_kind: tuple[bool] = (False, True, True),
        cx: np.ndarray = None,
        cy: np.ndarray = None,
        cz: np.ndarray = None,
    ):
        self.kind_map = 0

        # get default control points from default GVEC equilibrium
        if cx is None or cy is None or cz is None:
            from struphy.fields_background.equils import GVECequilibrium

            mhd_equil = GVECequilibrium()
            cx = mhd_equil.domain.cx
            cx = mhd_equil.domain.cy
            cx = mhd_equil.domain.cz

        # assign control points
        self._cx = cx
        self._cy = cy
        self._cz = cz

        # check dimensions
        assert self.cx.ndim == 3
        assert self.cy.ndim == 3
        assert self.cz.ndim == 3

        # make sure that control points are compatible with given spline data
        expected_shape = tuple([Nel[n] + (not spl_kind[n]) * p[n] for n in range(3)])

        assert self.cx.shape == expected_shape
        assert self.cy.shape == expected_shape
        assert self.cz.shape == expected_shape

        # identify polar singularity at eta1=0
        if np.all(self.cx[0, :, 0] == self.cx[0, 0, 0]):
            self.pole = True
        else:
            self.pole = False

        self.periodic_eta3 = spl_kind[-1]

        # base class
        super().__init__(Nel=Nel, p=p, spl_kind=spl_kind)


class PoloidalSpline(Domain):
    r"""Base class for all mappings that use a 2D spline representation
    :math:`S:(\eta_1, \eta_2) \to (R, Z) \in \mathbb R^2` in the poloidal plane:

    .. math::

        S: (\eta_1, \eta_2) \mapsto (R, Z) \textnormal{ as } \left\{\begin{aligned}
        R &= \sum_{ij} c^R_{ij} N_i(\eta_1) N_j(\eta_2) \,,

        Z &= \sum_{ij} c^Z_{ij} N_i(\eta_1) N_j(\eta_2) \,.
        \end{aligned}\right.

    The full map :math:`F: [0, 1]^3 \to \Omega` is obtained by defining :math:`(R, Z, \eta_3) \mapsto (x, y, z)` in the child class.
    """

    def __init__(
        self,
        Nel: tuple[int] = (8, 24),
        p: tuple[int] = (2, 3),
        spl_kind: tuple[bool] = (False, True),
        cx: np.ndarray = None,
        cy: np.ndarray = None,
    ):
        # get default control points
        if cx is None or cy is None:

            def X(eta1, eta2):
                return eta1 * np.cos(2 * np.pi * eta2) + 3.0

            def Y(eta1, eta2):
                return eta1 * np.sin(2 * np.pi * eta2)

            cx, cy = interp_mapping(Nel, p, spl_kind, X, Y)

            # make sure that control points at pole are all the same (eta1=0 there)
            cx[0] = 3.0
            cy[0] = 0.0

        # set control point properties
        self._cx = cx
        self._cy = cy

        # make sure that control points are 2D
        assert self.cx.ndim == 2
        assert self.cy.ndim == 2

        # make sure that control points are compatible with given spline data
        expected_shape = tuple([Nel[n] + (not spl_kind[n]) * p[n] for n in range(2)])

        assert self.cx.shape == expected_shape
        assert self.cy.shape == expected_shape

        # identify polar singularity at eta1=0
        if np.all(self.cx[0, :] == self.cx[0, 0]):
            self.pole = True
        else:
            self.pole = False

        # reshape control points to 3D
        self._cx = self.cx[:, :, None]
        self._cy = self.cy[:, :, None]
        self._cz = np.zeros((1, 1, 1), dtype=float)

        # init base class
        super().__init__(Nel=Nel, p=p, spl_kind=spl_kind)


class PoloidalSplineStraight(PoloidalSpline):
    r"""Cylinder where the poloidal planes are described by a 2D IGA-spline mapping.

    .. math::

        F: (R, Z, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= R \,,

        y &= Z \,,

        z &= L_z\eta_3\,.
        \end{aligned}\right.
    """

    def __init__(
        self,
        Nel: tuple[int] = (8, 24),
        p: tuple[int] = (2, 3),
        spl_kind: tuple[bool] = (False, True),
        cx: np.ndarray = None,
        cy: np.ndarray = None,
        Lz: float = 4.0,
    ):
        self.kind_map = 1

        # get default control points
        if cx is None or cy is None:

            def X(eta1, eta2):
                return eta1 * np.cos(2 * np.pi * eta2)

            def Y(eta1, eta2):
                return eta1 * np.sin(2 * np.pi * eta2)

            cx, cy = interp_mapping(Nel, p, spl_kind, X, Y)

            # make sure that control points at pole are all 0 (eta1=0 there)
            cx[0] = 0.0
            cy[0] = 0.0

        self.params_numpy = np.array([Lz])
        self.periodic_eta3 = False

        # init base class
        super().__init__(Nel=Nel, p=p, spl_kind=spl_kind, cx=cx, cy=cy)


class PoloidalSplineTorus(PoloidalSpline):
    r"""Torus where the poloidal planes are described by a 2D IGA-spline mapping.

    .. math::

        F: (R, Z, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= R \cos(2\pi\eta_3)  \,,

        y &= R \sin(- 2\pi\eta_3) \,,

        z &= Z \,.
        \end{aligned}\right.

    Parameters
    ----------
    Nel : tuple[int]
        Number of elements in each poloidal direction.

    p : tuple[int]
        Spline degree in each poloidal direction.

    spl_kind : tuple[bool]
        Kind of spline in each poloidal direction (True=periodic, False=clamped).

    cx, cy : np.ndarray
        Control points (spline coefficients) of the poloidal mapping.
        If None, a default square-to-disc mapping of radius 1 centered around (x, y) = (3, 0) is interpolated.

    tor_period : int
        The toroidal angle is between [0, 2*pi/tor_period).
    """

    def __init__(
        self,
        Nel: tuple[int] = (8, 24),
        p: tuple[int] = (2, 3),
        spl_kind: tuple[bool] = (False, True),
        cx: np.ndarray = None,
        cy: np.ndarray = None,
        tor_period: int = 3,
    ):
        # use setters for mapping attributes
        self.kind_map = 2
        self.params_numpy = np.array([float(tor_period)])
        self.periodic_eta3 = True

        # get default control points
        if cx is None or cy is None:

            def X(eta1, eta2):
                return eta1 * np.cos(2 * np.pi * eta2) + 3.0

            def Y(eta1, eta2):
                return eta1 * np.sin(2 * np.pi * eta2)

            cx, cy = interp_mapping(Nel, p, spl_kind, X, Y)

            # make sure that control points at pole are all 0 (eta1=0 there)
            cx[0] = 3.0
            cy[0] = 0.0

        # init base class
        super().__init__(
            Nel=Nel,
            p=p,
            spl_kind=spl_kind,
            cx=cx,
            cy=cy,
        )


def interp_mapping(Nel, p, spl_kind, X, Y, Z=None):
    r"""Interpolates the mapping :math:`F: (0, 1)^3 \to \mathbb R^3` on the given spline space.

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
    NbaseN = [Nel + p - kind * p for Nel, p, kind in zip(Nel, p, spl_kind)]

    # element boundaries
    el_b = [np.linspace(0.0, 1.0, Nel + 1) for Nel in Nel]

    # spline knot vectors
    T = [bsp.make_knots(el_b, p, kind) for el_b, p, kind in zip(el_b, p, spl_kind)]

    # greville points
    I_pts = [bsp.greville(T, p, kind) for T, p, kind in zip(T, p, spl_kind)]

    # 1D interpolation matrices
    I_mat = [csc_matrix(bsp.collocation_matrix(T, p, I_pts, kind)) for T, p, I_pts, kind in zip(T, p, I_pts, spl_kind)]

    # 2D interpolation
    if len(Nel) == 2:
        I = kron(I_mat[0], I_mat[1], format="csc")

        I_pts = np.meshgrid(I_pts[0], I_pts[1], indexing="ij")

        cx = spsolve(I, X(I_pts[0], I_pts[1]).flatten()).reshape(
            NbaseN[0],
            NbaseN[1],
        )
        cy = spsolve(I, Y(I_pts[0], I_pts[1]).flatten()).reshape(
            NbaseN[0],
            NbaseN[1],
        )

        return cx, cy

    # 3D interpolation
    elif len(Nel) == 3:
        I_LU = [splu(mat) for mat in I_mat]

        x_size = X(I_pts[0], I_pts[1], I_pts[2])
        y_size = Y(I_pts[0], I_pts[1], I_pts[2])
        z_size = Z(I_pts[0], I_pts[1], I_pts[2])

        cx = linalg_kron.kron_lusolve_3d(I_LU, x_size)
        cy = linalg_kron.kron_lusolve_3d(I_LU, y_size)
        cz = linalg_kron.kron_lusolve_3d(I_LU, z_size)

        return cx, cy, cz

    else:
        print("wrong number of elements")

        return 0.0


def spline_interpolation_nd(p: list, spl_kind: list, grids_1d: list, values: np.ndarray):
    """n-dimensional tensor-product spline interpolation with discrete input.

    The interpolation points are passed as a list of 1d arrays, each array with increasing entries g[0]=0 < g[1] < ...
    The last element must be g[-1] = 1 for clamped interpolation and g[-1] < 1 for periodic interpolation.

    Parameters
    -----------
    p : list[int]
        Spline degree.

    grids_1d : list[array]
        Interpolation points in [0, 1].

    spl_kind : list[bool]
        True: periodic splines, False: clamped splines.

    values: array
        Function values at interpolation points. values.shape = (grid1.size, ..., gridn.size).

    Returns
    --------
    coeffs : np.array
        spline coefficients as nd array.

    T : list[array]
        Knot vector of spline interpolant.

    indN : list[array]
        Global indices of non-vanishing splines in each element. Can be accessed via (element, local index).
    """

    T = []
    indN = []
    I_mat = []
    I_LU = []
    for sh, x_grid, p_i, kind_i in zip(values.shape, grids_1d, p, spl_kind):
        assert isinstance(x_grid, np.ndarray)
        assert sh == x_grid.size
        assert (
            np.all(
                np.roll(x_grid, 1)[1:] < x_grid[1:],
            )
            and x_grid[-1] > x_grid[-2]
        )
        assert x_grid[0] == 0.0

        if kind_i:
            assert x_grid[-1] < 1.0, "Interpolation points must be <1 for periodic interpolation."
            breaks = np.ones(x_grid.size + 1)

            if p_i % 2 == 0:
                breaks[1:-1] = (x_grid[1:] + np.roll(x_grid, 1)[1:]) / 2.0
                breaks[0] = 0.0
            else:
                breaks[:-1] = x_grid

        else:
            assert (
                np.abs(
                    x_grid[-1] - 1.0,
                )
                < 1e-14
            ), "Interpolation points must include x=1 for clamped interpolation."
            # dimension of the 1d spline spaces: dim = breaks.size - 1 + p = x_grid.size
            if p_i == 1:
                breaks = x_grid
            elif p_i % 2 == 0:
                breaks = x_grid[p_i // 2 - 1 : -p_i // 2].copy()
            else:
                breaks = x_grid[(p_i - 1) // 2 : -(p_i - 1) // 2].copy()

            # cells must be in interval [0, 1]
            breaks[0] = 0.0
            breaks[-1] = 1.0

        # breaks = np.linspace(0., 1., x_grid.size - (not kind_i)*p_i + 1)

        T += [bsp.make_knots(breaks, p_i, periodic=kind_i)]

        indN += [
            (np.indices((breaks.size - 1, p_i + 1))[1] + np.arange(breaks.size - 1)[:, None]) % x_grid.size,
        ]

        I_mat += [bsp.collocation_matrix(T[-1], p_i, x_grid, periodic=kind_i)]

        I_LU += [splu(csc_matrix(I_mat[-1]))]

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
