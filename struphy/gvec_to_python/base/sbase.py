# import logging
from gvec_to_python.util.logger import logger
# logger = logging.getLogger(__name__)

import warnings

import numpy as np

from gvec_to_python.hylife.utilities_FEEC import bsplines as bsp
from gvec_to_python.hylife.utilities_FEEC import spline_space as spl



class sBase:

    Nel  = None # Number of elements.
    el_b = None # Element boundaries.
    p    = None # Spline degree.
    bc   = None # Periodic boundary conditions (use 'False' if clamped).
    # coef = None # Spline coefficients, a.k.a. control points.

    knots_N    = None # Knot sequence for N-splines. Called "T" in STRUPHY.
    greville_N = None # Greville points.
    knots_D    = None # Knot sequence for D-splines.

    def __init__(self, Nel, el_b, p, bc=False, use_hylife=True, use_cache=False):

        logger.debug('sBase.__init__()')

        assert Nel + 1 == len(el_b), 'Number of element boundaries should be equal to number of elements + 1.'

        # Input quantities.
        self.Nel  = Nel   # Number of elements.
        self.el_b = el_b  # Element boundaries.
        self.p    = p     # Spline degree.
        self.bc   = bc    # Periodic boundary conditions (use 'False' if clamped).
        # self.coef = coef  # Spline coefficients, a.k.a. control points.

        # Derived quantities.
        self.knots_N    = bsp.make_knots(el_b, p, bc)        # Knot sequence for N-splines. Called "T" in STRUPHY.
        self.greville_N = bsp.greville(self.knots_N, p, bc)  # Greville points.
        self.knots_D    = self.knots_N[1:-1]                 # Knot sequence for D-splines.

        if bc: # Periodic
            assert self.Nel == len(self.greville_N), 'Number of elements should be equal to number of Greville points.' # Also equal to # basis functions.
        else: # Clamped
            assert self.Nel + 1 + 2 * p == len(self.knots_N), 'Number of knots should be equal to number of element boundaries + 2 * spline degrees.'
            assert self.Nel + p == len(self.greville_N), 'Number of elements + spline degrees should be equal to number of Greville points.'

        # logger.debug('{:d} knots           for N-splines:\n{}\n'.format(len(self.knots_N),       self.knots_N))
        # logger.debug('{:d} greville points for N-splines:\n{}\n'.format(len(self.greville_N), self.greville_N))
        # logger.debug('{:d} knots           for D-splines:\n{}\n'.format(len(self.knots_D),       self.knots_D))

        self.struphy.sbase = spl.Spline_space_1d(self.knots_N, p, bc) # Reverted.
        # Class constructor signature CHANGED in struphy.s `devel_standard` branch!
        # The first parameter is now `Nel`, instead of the knot vector!
        # Now reverted.
        # self.struphy.sbase = Spline_space_1d(Nel, p, bc)

        # Cache evaluations in memory.
        self.use_cache = use_cache
        self.cache = {
            'eval_s_struphy': {},
            'eval_s_1st_der_struphy': {},
        }
        if self.use_cache:
            warnings.warn("PerformanceWarning: Hashing `coef` will take far longer than the actual evaluation! Do not enable cache in this class!")



    def eval_s(self, s, coef):
        """Evaluate B-spline at point `s` given control points `coef`.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.
            Point of evaluation.

        coef : array_like
            B-spline control points (coefficients). Please be np.array.

        Returns
        -------
        value : float or meshgrid numpy.ndarray
            Finite element B-spline evaluated at the point `s`.

        Notes
        -----
        Not implemented. Passthrough directly to `eval_s_struphy`.
        """

        # logger.debug('sBase.eval_s()')

        return self.eval_s_struphy(s, coef)

    def eval_s_struphy(self, s, coef):
        """Evaluate B-spline at point `s` given control points `coef` using `struphy. class `Spline_space_1d`.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.
            Point of evaluation.

        coef : array_like
            B-spline control points (coefficients). Please be np.array.

        Returns
        -------
        value : float or meshgrid numpy.ndarray
            Finite element B-spline evaluated at point `s`.
        """

        # logger.debug('sBase.eval_s_struphy.)')

        if self.use_cache:
            param_hash = hash(str(s) + str(coef))
            if param_hash in self.cache['eval_s_struphy']:
                return self.cache['eval_s_struphy'][param_hash]

        # After `struphy. update, if `s` is a numpy array, `spline_space` will only process 1D numpy array!
        # Therefore it cannot handle meshgrids!
        # If we have a sparse meshgrid, we can flatten `s`.
        # If we have a dense meshgrid, then we'll have to treat each element as a scalar, and loop many times!

        # `s` is a numpy array.
        if isinstance(s, np.ndarray):

            # `s` is a simple 1D array.
            if s.ndim == 1:

                # Old implementation:
                # N_eval_at_s = np.zeros_like(s)
                # for i, si in enumerate(s):
                #     N_eval_at_s[i] = self.struphy.sbase.evaluate_N(si, coef)

                # New implementation:
                # Transform (s, theta, zeta) into Cartesian (x, y, z) coordinates.
                N_eval_at_s = self.struphy.sbase.evaluate_N(s, coef)

            # `s` is a mesh grid. 
            else:

                # Distinguish if it is sparse or dense.
                # Sparse: s.shape = (ns,  1,  1)
                # Dense : s.shape = (ns, nu, nv)
                # print('sBase: s (eta1) has dim {} > 1'.format(s.ndim))
                # print('sBase: s.shape: {}'.format(s.shape))
                # print('sBase: s.size: {}'.format(s.size))

                # `s` is a sparse meshgrid. Just flatten it.
                if max(s.shape) == s.size:

                    # print('sBase: `s` is a sparse meshgrid.')

                    N_eval_at_s = self.struphy.sbase.evaluate_N(s.flatten(), coef).reshape(s.shape)

                # `s` is a dense meshgrid. Process each point as a scalar.
                else:

                    # print('sBase: `s` is a dense meshgrid.')

                    N_eval_at_s = np.empty(s.shape, dtype=float)

                    if s.ndim == 2:
                        for i in range(s.shape[0]):
                            for j in range(s.shape[1]):
                                N_eval_at_s[i, j] = self.struphy.sbase.evaluate_N(s[i, j], coef)

                    elif s.ndim == 3:
                        for i in range(s.shape[0]):
                            for j in range(s.shape[1]):
                                for k in range(s.shape[2]):
                                    N_eval_at_s[i, j, k] = self.struphy.sbase.evaluate_N(s[i, j, k], coef)

                    else:

                        raise ValueError('`s` has dimension {}, which sBase cannot handle.'.format(s.ndim))

        # `s` is a scalar.
        else:

            # Transform (s, theta, zeta) into Cartesian (x, y, z) coordinates.
            N_eval_at_s = self.struphy.sbase.evaluate_N(s, coef)

        # logger.debug('s-coordinate is: {}'.format(N_eval_at_s))

        if self.use_cache:
            self.cache['eval_s_struphy'][param_hash] = N_eval_at_s
        return N_eval_at_s



    def eval_s_1st_der(self, s, coef):
        """Evaluate the first derivative of B-spline at point `s` given control points `coef` using `struphy. class `Spline_space_1d`.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.
            Point of evaluation.

        coef : array_like
            B-spline control points (coefficients). Please be np.array.

        Returns
        -------
        value : float or meshgrid numpy.ndarray
            First derivative of B-spline evaluated at point `s`.

        Notes
        -----
        Not implemented. Passthrough directly to `eval_s_1st_der_struphy`.
        """

        # logger.debug('sBase.eval_s_1st_der()')

        return self.eval_s_1st_der_struphy(s, coef)

    def eval_s_1st_der_struphy(self, s, coef):
        """Evaluate the first derivative of B-spline at point `s` given control points `coef` using `struphy. class `Spline_space_1d`.

        Parameters
        ----------
        s : float or meshgrid numpy.ndarray
            Logical coordinate in radial direction.
            Point of evaluation.

        coef : array_like
            B-spline control points (coefficients). Please be np.array.

        Returns
        -------
        value : float or meshgrid numpy.ndarray
            First derivative of B-spline evaluated at point `s`.
        """

        # logger.debug('sBase.eval_s_1st_der_struphy.)')

        if self.use_cache:
            param_hash = hash(str(s) + str(coef))
            if param_hash in self.cache['eval_s_1st_der_struphy']:
                return self.cache['eval_s_1st_der_struphy'][param_hash]

        # After `struphy. update, if `s` is a numpy array, `spline_space` will only process 1D numpy array!
        # Therefore it cannot handle meshgrids!
        # If we have a sparse meshgrid, we can flatten `s`.
        # If we have a dense meshgrid, then we'll have to treat each element as a scalar, and loop many times!

        # `s` is a numpy array.
        if isinstance(s, np.ndarray):

            # `s` is a simple 1D array.
            if s.ndim == 1:

                # Old implementation:
                # dN_eval_at_s = np.zeros_like(s)
                # for i, si in enumerate(s):
                #     dN_eval_at_s[i] = self.struphy.sbase.evaluate_dN(si, coef)

                # New implementation:
                # Transform (s, theta, zeta) into Cartesian (x, y, z) coordinates?
                dN_eval_at_s = self.struphy.sbase.evaluate_dN(s, coef)

            # `s` is a mesh grid. 
            else:

                # Distinguish if it is sparse or dense.
                # Sparse: s.shape = (ns,  1,  1)
                # Dense : s.shape = (ns, nu, nv)
                # print('sBase: s (eta1) has dim {} > 1'.format(s.ndim))
                # print('sBase: s.shape: {}'.format(s.shape))
                # print('sBase: s.size: {}'.format(s.size))

                # `s` is a sparse meshgrid. Just flatten it.
                if max(s.shape) == s.size:

                    # print('sBase: `s` is a sparse meshgrid.')

                    dN_eval_at_s = self.struphy.sbase.evaluate_dN(s.flatten(), coef).reshape(s.shape)

                # `s` is a dense meshgrid. Process each point as a scalar.
                else:

                    # print('sBase: `s` is a dense meshgrid.')

                    dN_eval_at_s = np.empty(s.shape, dtype=float)

                    if s.ndim == 2:
                        for i in range(s.shape[0]):
                            for j in range(s.shape[1]):
                                dN_eval_at_s[i, j] = self.struphy.sbase.evaluate_dN(s[i, j], coef)

                    elif s.ndim == 3:
                        for i in range(s.shape[0]):
                            for j in range(s.shape[1]):
                                for k in range(s.shape[2]):
                                    dN_eval_at_s[i, j, k] = self.struphy.sbase.evaluate_dN(s[i, j, k], coef)

                    else:

                        raise ValueError('`s` has dimension {}, which sBase cannot handle.'.format(s.ndim))

        # `s` is a scalar.
        else:

            dN_eval_at_s = self.struphy.sbase.evaluate_dN(s, coef)

        # logger.debug('Derivative is: {}'.format(dN_eval_at_s))

        if self.use_cache:
            self.cache['eval_s_1st_der_struphy'][param_hash] = dN_eval_at_s
        return dN_eval_at_s
