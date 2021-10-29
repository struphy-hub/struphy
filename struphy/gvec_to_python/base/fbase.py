# import logging
from gvec_to_python.util.logger import logger
# logger = logging.getLogger(__name__)

import numpy as np
from numba import jit, njit



class fBase:

    NFP          = None # Number of field periods (symmetry).
    modes        = None # Number of all m-n mode combinations.
    sin_cos      = None # Whether the data has only sine, only cosine, or both sine and cosine basis.
    excl_mn_zero = None # 
    mn           = None # mn-mode number, with NFP premultiplied into the n-modes.
    mn_max       = None # Maximum m-mode and n-mode numbers, without NFP.
    modes_sin    = None # Number of sine modes.
    modes_cos    = None # Number of cosine modes.
    range_sin    = None # Index range of sine modes in `mn` list.
    range_cos    = None # Index range of cosine modes in `mn` list.

    # Most parameters are unused!
    def __init__(self, NFP, modes, sin_cos, excl_mn_zero, mn, mn_max, modes_sin, modes_cos, range_sin, range_cos):

        logger.debug('fBase.__init__()')

        self.NFP          = NFP          # Number of field periods (symmetry).
        self.modes        = modes        # Number of all m-n mode combinations.
        self.sin_cos      = sin_cos      # Whether the data has only sine, only cosine, or both sine and cosine basis. 1=sin, 2=cos, 3=sin&cos.
        self.excl_mn_zero = excl_mn_zero # 
        self.mn           = mn           # mn-mode number, with NFP premultiplied into the n-modes.
        self.mn_max       = mn_max       # Maximum m-mode and n-mode numbers, without NFP.
        self.modes_sin    = modes_sin    # Number of sine modes.
        self.modes_cos    = modes_cos    # Number of cosine modes.
        self.range_sin    = range_sin    # Index range of sine modes in `mn` list.
        self.range_cos    = range_cos    # Index range of cosine modes in `mn` list.
        if self.range_sin is not None:
            self.range_sin    = np.array(self.range_sin)
        if self.range_cos is not None:
            self.range_cos    = np.array(self.range_cos)

        # To get rid of if-else statements inside eval() functions, which will be evaluated many times in a for-loop.
        # 1=sin, 2=cos, 3=sin&cos.
        if   sin_cos == 1: # Only sine.
            self.evaluator        = fBase.eval_sin
            self.evaluator_dtheta = fBase.eval_sin_dtheta
            self.evaluator_dzeta  = fBase.eval_sin_dzeta
        elif sin_cos == 2: # Only cosine.
            self.evaluator        = fBase.eval_cos
            self.evaluator_dtheta = fBase.eval_cos_dtheta
            self.evaluator_dzeta  = fBase.eval_cos_dzeta
        elif sin_cos == 3: # Both sine and cosine terms.
            self.evaluator        = fBase.eval_sin_cos
            self.evaluator_dtheta = fBase.eval_sin_cos_dtheta
            self.evaluator_dzeta  = fBase.eval_sin_cos_dzeta
        else:
            # Throws error.
            pass



    @staticmethod
    @njit
    def eval_sin(idx, m, theta, n, zeta, range_sin):

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        return np.sin(m * theta - n * zeta)

    @staticmethod
    @njit
    def eval_cos(idx, m, theta, n, zeta, range_sin):

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        return np.cos(m * theta - n * zeta)

    @staticmethod
    @njit
    def eval_sin_cos(idx, m, theta, n, zeta, range_sin):

        # If it is guaranteed to evaluate in sequential order, one may be able to neglect this if-else check.
        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        if idx <= range_sin[1] and idx >= range_sin[0]:
            return np.sin(m * theta - n * zeta)
        else:
            return np.cos(m * theta - n * zeta)



    @staticmethod
    @njit
    def eval_sin_dtheta(idx, m, theta, n, zeta, range_sin):

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        return   np.cos(m * theta - n * zeta) * m

    @staticmethod
    @njit
    def eval_cos_dtheta(idx, m, theta, n, zeta, range_sin):

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        return - np.sin(m * theta - n * zeta) * m

    @staticmethod
    @njit
    def eval_sin_cos_dtheta(idx, m, theta, n, zeta, range_sin):

        # If it is guaranteed to evaluate in sequential order, one may be able to neglect this if-else check.
        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        if idx <= range_sin[1] and idx >= range_sin[0]:
            return   np.cos(m * theta - n * zeta) * m
        else:
            return - np.sin(m * theta - n * zeta) * m



    @staticmethod
    @njit
    def eval_sin_dzeta(idx, m, theta, n, zeta, range_sin):

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        return   np.cos(m * theta - n * zeta) * (- n) # * NFP

    @staticmethod
    @njit
    def eval_cos_dzeta(idx, m, theta, n, zeta, range_sin):

        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        return - np.sin(m * theta - n * zeta) * (- n) # * NFP

    @staticmethod
    @njit
    def eval_sin_cos_dzeta(idx, m, theta, n, zeta, range_sin):

        # If it is guaranteed to evaluate in sequential order, one may be able to neglect this if-else check.
        # Nfp is already premultiplied into n-modes. Otherwise n * NFP * zeta.
        if idx <= range_sin[1] and idx >= range_sin[0]:
            return   np.cos(m * theta - n * zeta) * (- n) # * NFP
        else:
            return - np.sin(m * theta - n * zeta) * (- n) # * NFP



    # A better function name may be 'eval_base'.
    def eval_f(self, idx, m, theta, n, zeta):
        """Evaluate a Fourier term given mn-mode numbers and angles `theta` and `zeta`.

        Parameters
        ----------
        idx : int
            Index of the given mode number pair (m, n) in `self.mn`.

        m : int
            Mode number m along poloidal direction.

        theta : float or meshgrid numpy.ndarray
            Angle theta in Tokamak coordinate along poloidal direction.

        n : int
            Mode number n along toroidal direction.

        zeta : float or meshgrid numpy.ndarray
            Angle zeta in Tokamak coordinate along toroidal direction.

        Returns
        -------
        value : float or meshgrid numpy.ndarray
            Fourier contribution from mode (m, n) evaluated at angles `theta` and `zeta`.
        """

        # logger.debug('fBase.eval_f()')

        return self.evaluator(idx, m, theta, n, zeta, self.range_sin)



    def eval_f_dtheta(self, idx, m, theta, n, zeta):
        """Evaluate partial derivative of a Fourier term w.r.t. `theta` given mn-mode numbers and angles `theta` and `zeta`.

        Parameters
        ----------
        idx : int
            Index of the given mode number pair (m, n) in `self.mn`.

        m : int
            Mode number m along poloidal direction.

        theta : float or meshgrid numpy.ndarray
            Angle theta in Tokamak coordinate along poloidal direction.

        n : int
            Mode number n along toroidal direction.

        zeta : float or meshgrid numpy.ndarray
            Angle zeta in Tokamak coordinate along toroidal direction.

        Returns
        -------
        value : float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `theta` of the Fourier contribution from mode (m, n) evaluated at angles `theta` and `zeta`.
        """

        # logger.debug('fBase.eval_f_dtheta()')

        return self.evaluator_dtheta(idx, m, theta, n, zeta, self.range_sin)



    def eval_f_dzeta(self, idx, m, theta, n, zeta):
        """Evaluate partial derivative of a Fourier term w.r.t. `zeta` given mn-mode numbers and angles `theta` and `zeta`.

        Parameters
        ----------
        idx : int
            Index of the given mode number pair (m, n) in `self.mn`.

        m : int
            Mode number m along poloidal direction.

        theta : float or meshgrid numpy.ndarray
            Angle theta in Tokamak coordinate along poloidal direction.

        n : int
            Mode number n along toroidal direction.

        zeta : float or meshgrid numpy.ndarray
            Angle zeta in Tokamak coordinate along toroidal direction.

        Returns
        -------
        value : float or meshgrid numpy.ndarray
            Partial derivative w.r.t. `zeta` of the Fourier contribution from mode (m, n) evaluated at angles `theta` and `zeta`.
        """

        # logger.debug('fBase.eval_f_dzeta()')

        return self.evaluator_dzeta(idx, m, theta, n, zeta, self.range_sin)
