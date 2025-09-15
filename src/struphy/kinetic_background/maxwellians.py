"Maxwellian (Gaussian) distributions in velocity space."

from typing import Callable

import numpy as np

from struphy.fields_background.base import FluidEquilibriumWithB
from struphy.fields_background.equils import set_defaults
from struphy.initial.base import Perturbation
from struphy.kinetic_background.base import Maxwellian


class Maxwellian3D(Maxwellian):
    r"""A :class:`~struphy.kinetic_background.base.Maxwellian` depending on three (:math:`n=3`) Cartesian velocities.

    Parameters
    ----------
    n, ui, vthi : tuple
        Moments of the Maxwellian as tuples. The first entry defines the background
        (float for constant background or callable), the second entry defines a Perturbation (can be None).
    """

    def __init__(
        self,
        n: tuple[float | Callable, Perturbation] = (1.0, None),
        u1: tuple[float | Callable, Perturbation] = (0.0, None),
        u2: tuple[float | Callable, Perturbation] = (0.0, None),
        u3: tuple[float | Callable, Perturbation] = (0.0, None),
        vth1: tuple[float | Callable, Perturbation] = (1.0, None),
        vth2: tuple[float | Callable, Perturbation] = (1.0, None),
        vth3: tuple[float | Callable, Perturbation] = (1.0, None),
    ):
        self._maxw_params = {}
        self._maxw_params["n"] = n
        self._maxw_params["u1"] = u1
        self._maxw_params["u2"] = u2
        self._maxw_params["u3"] = u3
        self._maxw_params["vth1"] = vth1
        self._maxw_params["vth2"] = vth2
        self._maxw_params["vth3"] = vth3

        self.check_maxw_params()

        # factors multiplied onto the defined moments n, u and vth (can be set via setter)
        self._moment_factors = {
            "n": 1.0,
            "u": [1.0, 1.0, 1.0],
            "vth": [1.0, 1.0, 1.0],
        }

    @property
    def maxw_params(self):
        return self._maxw_params

    @property
    def coords(self):
        """Coordinates of the Maxwellian6D, :math:`(v_1, v_2, v_3)`."""
        return "cartesian"

    @property
    def vdim(self):
        """Dimension of the velocity space."""
        return 3

    @property
    def is_polar(self):
        """List of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp)."""
        return [False, False, False]

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """Jacobian determinant of the velocity coordinate transformation from Maxwellian6D('cartesian') to Particles6D('cartesian').

        Input parameters should be slice of 2d numpy marker array. (i.e. *self.phasespace_coords.T)

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The Jacobian determinant evaluated at given logical coordinates.
        -------
        """

        assert eta1.ndim == 1
        assert eta2.ndim == 1
        assert eta3.ndim == 1
        assert len(v) == 3

        return 1.0 + 0 * eta1

    @property
    def volume_form(self):
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian)."""
        return False

    @property
    def moment_factors(self):
        """Collection of factors multiplied onto the defined moments n, u, and vth."""
        return self._moment_factors

    @moment_factors.setter
    def moment_factors(self, **kwargs):
        for kw, arg in kwargs:
            if kw in {"u", "vth"}:
                assert len(arg) == 3
            self._moment_factors[kw] = arg

    def n(self, eta1, eta2, eta3):
        """Zero-th moment (density)."""
        out = self._evaluate_moment(eta1, eta2, eta3, name="n")
        return out * self.moment_factors["n"]

    def u(self, eta1, eta2, eta3):
        """Mean velocities."""
        out = []
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u1")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u2")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u3")]
        return [ou * mom_fac for ou, mom_fac in zip(out, self.moment_factors["u"])]

    def vth(self, eta1, eta2, eta3):
        """Thermal velocities."""
        out = []
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth1")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth2")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth3")]
        return [ou * mom_fac for ou, mom_fac in zip(out, self.moment_factors["vth"])]


class GyroMaxwellian2D(Maxwellian):
    r"""A gyrotropic :class:`~struphy.kinetic_background.base.Maxwellian` depending on
    two velocities :math:`(v_\parallel, v_\perp)`, :math:`n=2`,
    where :math:`v_\parallel = \matbf v \cdot \mathbf b_0` and :math:`v_\perp`
    is the radial component of a polar coordinate system perpendicular
    to the magentic direction :math:`\mathbf b_0`.

    Parameters
    ----------
    n, u_para, u_perp, vth_para, vth_perp : tuple
        Moments of the Maxwellian as tuples. The first entry defines the background
        (float for constant background or callable), the second entry defines a Perturbation (can be None).

    maxw_params : dict
        Parameters for the kinetic background.

    pert_params : dict
        Parameters for the kinetic perturbation added to the background.

    equil : FluidEquilibriumWithB
        Fluid background.

    volume_form : bool
        Whether to represent the Maxwellian as a volume form;
        if True it is multiplied by the Jacobian determinant |v_perp|
        of the polar coordinate transofrmation (default = False).
    """

    def __init__(
        self,
        n: tuple[float | Callable, Perturbation] = (1.0, None),
        u_para: tuple[float | Callable, Perturbation] = (0.0, None),
        u_perp: tuple[float | Callable, Perturbation] = (0.0, None),
        vth_para: tuple[float | Callable, Perturbation] = (1.0, None),
        vth_perp: tuple[float | Callable, Perturbation] = (1.0, None),
        equil: FluidEquilibriumWithB = None,
        volume_form: bool = True,
    ):
        self._maxw_params = {}
        self._maxw_params["n"] = n
        self._maxw_params["u_para"] = u_para
        self._maxw_params["u_perp"] = u_perp
        self._maxw_params["vth_para"] = vth_para
        self._maxw_params["vth_perp"] = vth_perp

        self.check_maxw_params()

        # volume form represenation
        self._volume_form = volume_form
        self._equil = equil

        # factors multiplied onto the defined moments n, u and vth (can be set via setter)
        self._moment_factors = {
            "n": 1.0,
            "u": [1.0, 1.0],
            "vth": [1.0, 1.0],
        }

    @property
    def maxw_params(self):
        return self._maxw_params

    @property
    def coords(self):
        r"""Coordinates of the Maxwellian5D, :math:`(v_\parallel, v_\perp)`."""
        return "vpara_vperp"

    @property
    def vdim(self):
        """Dimension of the velocity space."""
        return 2

    @property
    def is_polar(self):
        """List of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp)."""
        return [False, True]

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        r"""Jacobian determinant of the velocity coordinate transformation from Maxwellian5D('vpara_vperp') to Particles5D('vpara_mu').

        .. math::

            \begin{aligned}
            F &: (v_\parallel, v_\perp) \to (v_\parallel, \mu) \,,
            \\[3mm]
            DF &= \begin{bmatrix} \frac{\partial v_\parallel}{\partial v_\parallel} & \frac{\partial v_\parallel}{\partial v_\perp} \\
                 \frac{\partial \mu}{\partial v_\parallel} & \frac{\partial \mu}{\partial v_\perp}  \end{bmatrix} =
                 \begin{bmatrix} 1 & 0 \\
                 0 & \frac{v_\perp}{B}  \end{bmatrix} \,,
            \\[3mm]
            J_F &= \frac{v_\perp}{B} \,,
            \end{aligned}

        where :math:`\mu = \frac{v_\perp^2}{2B}`.

        Input parameters should be slice of 2d numpy marker array. (i.e. *self.phasespace_coords.T)

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        *v : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The Jacobian determinant evaluated at given logical coordinates.
        -------
        """

        assert eta1.ndim == 1
        assert eta2.ndim == 1
        assert eta3.ndim == 1
        assert len(v) == 2

        # call equilibrium
        etas = (np.vstack((eta1, eta2, eta3)).T).copy()
        absB0 = self.equil.absB0(etas)

        # J = v_perp/B
        jacobian_det = v[1] / absB0

        return jacobian_det

    @property
    def volume_form(self) -> bool:
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian |v_perp|)."""
        return self._volume_form

    @property
    def equil(self) -> FluidEquilibriumWithB:
        """Fluid background with B-field."""
        return self._equil

    @property
    def moment_factors(self):
        """Collection of factors multiplied onto the defined moments n, u, and vth."""
        return self._moment_factors

    @moment_factors.setter
    def moment_factors(self, **kwargs):
        for kw, arg in kwargs:
            if kw in {"u", "vth"}:
                assert len(arg) == 2
            self._moment_factors[kw] = arg

    def n(self, eta1, eta2, eta3):
        """Zero-th moment (density)."""
        out = self._evaluate_moment(eta1, eta2, eta3, name="n")
        return out * self.moment_factors["n"]

    def u(self, eta1, eta2, eta3):
        """Mean velocities."""
        out = []
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u_para")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u_perp")]
        return [ou * mom_fac for ou, mom_fac in zip(out, self.moment_factors["u"])]

    def vth(self, eta1, eta2, eta3):
        """Thermal velocities."""
        out = []
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth_para")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth_perp")]
        return [ou * mom_fac for ou, mom_fac in zip(out, self.moment_factors["vth"])]


class CanonicalMaxwellian:
    r"""canonical Maxwellian distribution function.
    It is defined by three constants of motion in the axissymmetric toroidal system:

    - Shifted canonical toroidal momentum

    .. math::

        \psi_c = \psi + \frac{m_s F}{q_s B}v_\parallel - \text{sign}(v_\parallel)\sqrt{2(\epsilon - \mu B)}\frac{m_sF}{q_sB} \mathcal{H}(\epsilon - \mu B),

    - Energy

    .. math::

        \epsilon = \frac{1}{2}m_sv_\parallel² + \mu B,

    - Magnetic moment

    .. math::

        \mu = \frac{m_s v_\perp²}{2B},

    where :math:`\psi` is the poloidal magnetic flux function, :math:`F=F(\psi)` is the poloidal current function and :math:`\mathcal{H}` is the Heaviside function.

    With the three constants of motion, a canonical Maxwellian distribution function is defined as

    .. math::

        F(\psi_c, \epsilon, \mu) = \frac{n(\psi_c)}{(2\pi)^{3/2}v_\text{th}³(\psi_c)} \text{exp}\left[ - \frac{\epsilon}{v_\text{th}²(\psi_c)}\right].

    Parameters
    ----------
    maxw_params : dict
        Parameters for the kinetic background.

    pert_params : dict
        Parameters for the kinetic perturbation added to the background.

    equil : FluidEquilibriumWithB
        Fluid background.

    volume_form : bool
        Whether to represent the Maxwellian as a volume form;
        if True it is multiplied by the Jacobian determinant |v_perp|
        of the polar coordinate transofrmation (default = False).
    """

    def __init__(
        self,
        n: tuple[float | Callable, Perturbation] = (1.0, None),
        vth: tuple[float | Callable, Perturbation] = (1.0, None),
        equil: FluidEquilibriumWithB = None,
        volume_form: bool = True,
    ):
        self._maxw_params = {}
        self._maxw_params["n"] = n
        self._maxw_params["vth"] = vth

        self.check_maxw_params()

        # volume form represenation
        self._volume_form = volume_form
        self._equil = equil

        # factors multiplied onto the defined moments n and vth (can be set via setter)
        self._moment_factors = {
            "n": 1.0,
            "vth": 1.0,
        }

    @property
    def coords(self):
        r"""Coordinates of the CanonicalMaxwellian, :math:`(\epsilon, \mu, \psi_c)`."""
        return "constants_of_motion"

    @property
    def maxw_params(self):
        """Parameters dictionary defining constant moments of the Maxwellian."""
        return self._maxw_params

    @property
    def equil(self) -> FluidEquilibriumWithB:
        """One of :mod:`~struphy.fields_background.equils`
        in case that moments are to be set in that way, None otherwise.
        """
        return self._equil

    def check_maxw_params(self):
        for k, v in self.maxw_params.items():
            assert isinstance(k, str)
            assert isinstance(v, tuple), f"Maxwallian parameter {k} must be tuple, but is {v}"
            assert len(v) == 2

            assert isinstance(v[0], (float, int, Callable))
            assert isinstance(v[1], Perturbation) or v[1] is None

    def velocity_jacobian_det(self, eta1, eta2, eta3, energy):
        r"""TODO"""

        assert eta1.ndim == 1
        assert eta2.ndim == 1
        assert eta3.ndim == 1

        if self.maxw_params["type"] == "Particles6D":
            return np.sqrt(2.0 * energy) * 4.0 * np.pi

        else:
            # call equilibrium
            etas = (np.vstack((eta1, eta2, eta3)).T).copy()
            absB0 = self.equil.absB0(etas)

            return np.sqrt(energy) * 2.0 * np.sqrt(2.0) / absB0

    def gaussian(self, e, vth=1.0):
        """3-dim. normal distribution, to which array-valued thermal velocities can be passed.

        Parameters
        ----------
        e : float | array-like
            Energy.

        vth : float | array-like
            Thermal velocity evaluated at psic.

        Returns
        -------
        An array of size(e).
        """

        if isinstance(vth, np.ndarray):
            assert e.shape == vth.shape, f"{e.shape = } but {vth.shape = }"

        return 2.0 * np.sqrt(e / np.pi) / vth**3 * np.exp(-e / vth**2)

    def __call__(self, *args):
        """Evaluates the canonical Maxwellian distribution function.

        There are two use-cases for this function in the code:

        1. Evaluating for particles ("flat evaluation", inputs are all 1D of length N_p)
        2. Evaluating the function on a meshgrid (in phase space).

        Hence all arguments must always have

        1. the same shape
        2. either ndim = 1 or ndim = 3.

        Parameters
        ----------
        *args : array_like
            Position-velocity arguments in the order energy, magnetic moment, canonical toroidal momentum.

        Returns
        -------
        f : np.ndarray
            The evaluated Maxwellian.
        """

        # Check that all args have the same shape
        shape0 = np.shape(args[0])
        for i, arg in enumerate(args):
            assert np.shape(arg) == shape0, f"Argument {i} has {np.shape(arg) = }, but must be {shape0 = }."
            assert np.ndim(arg) == 1 or np.ndim(arg) == 3, (
                f"{np.ndim(arg) = } not allowed for canonical Maxwellian evaluation."
            )  # flat or meshgrid evaluation

        # Get result evaluated with each particles' psic
        res = self.n(args[2])
        vths = self.vth(args[2])

        # take care of correct broadcasting, assuming args come from phase space meshgrid
        if np.ndim(args[0]) == 3:
            # move eta axes to the back
            arg_t = np.moveaxis(args[0], 0, -1)
            arg_t = np.moveaxis(arg_t, 0, -1)
            arg_t = np.moveaxis(arg_t, 0, -1)

            # broadcast
            res_broad = res + 0.0 * arg_t

            # move eta axes to the front
            res = np.moveaxis(res_broad, -1, 0)
            res = np.moveaxis(res, -1, 0)
            res = np.moveaxis(res, -1, 0)

        # Multiply result with gaussian in energy
        if np.ndim(args[0]) == 3:
            vth_broad = vths + 0.0 * arg_t
            vth = np.moveaxis(vth_broad, -1, 0)
            vth = np.moveaxis(vth, -1, 0)
            vth = np.moveaxis(vth, -1, 0)
        else:
            vth = vths

        res *= self.gaussian(args[0], vth=vth)

        return res

    @property
    def volume_form(self):
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian |v_perp|)."""
        return self._volume_form

    @property
    def moment_factors(self):
        """Collection of factors multiplied onto the defined moments n, u, and vth."""
        return self._moment_factors

    @moment_factors.setter
    def moment_factors(self, **kwargs):
        for kw, arg in kwargs:
            self._moment_factors[kw] = arg

    def rc(self, psic):
        r""" Square root of radially normalized canonical toroidal momentum.

        .. math::
            \begin{aligned}
            r_c^2 &= \frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}} \,,
            \\[3mm]
            r_c &= \begin{cases}
            \sqrt{\frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}}} & \text{if} \quad \frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}} \geq 0 \,, \\
            -\sqrt{\frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}}} & \text{if} \quad \frac{\psi_c - \psi_\text{axis}}{\psi_\text{edge} - \psi_\text{axis}} < 0 \,,
            \end{cases}
            \end{aligned}

        where :math:`\psi_\text{axis}` and :math:`\psi_\text{edge}` are poloidal magnetic flux function at the center and edge of poloidal plane respectively.

        Parameters
        ----------
        psic : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array of the evaluated :math:`r_c`.

        """

        # calculate rc²
        rc_squared = (psic - self.equil.psi_range[0]) / (self.equil.psi_range[1] - self.equil.psi_range[0])

        # sorting out indices of negative rc²
        neg_index = np.logical_not(rc_squared >= 0)

        # make them positive
        rc_squared[neg_index] *= -1

        # calculate rc
        rc = np.sqrt(rc_squared)
        rc[neg_index] *= -1

        return rc

    def n(self, psic, add_perturbation: bool = None):
        """Density as background + perturbation.

        Parameters
        ----------
        psic : numpy.array
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A float (background value) or a numpy.array of the evaluated density.
        """
        # collect arguments
        assert isinstance(psic, np.ndarray)

        # assuming that input comes from meshgrid.
        if psic.ndim == 3:
            psic = psic[0, 0, :]

        # set background density
        if isinstance(self.maxw_params["n"][0], (float, int)):
            res = self.maxw_params["n"][0] + 0.0 * psic
        else:
            nfun = self.maxw_params["n"][1]
            # for typ, params in mom_funcs.items():
            #     nfun = getattr(moment_functions, typ)(**params)
            res = nfun(eta1=self.rc(psic))

        # add perturbation
        if add_perturbation is None:
            add_perturbation = self.add_perturbation

        perturbation = self.maxw_params["n"][1]
        if perturbation is not None and add_perturbation:
            assert isinstance(perturbation, Perturbation)
            res = perturbation(eta1=self.rc(psic))
            # if eta1.ndim == 1:
            #     out += perturbation(eta1, eta2, eta3)
            # else:
            #     out += perturbation(*etas)

        return res * self.moment_factors["n"]

    def vth(self, psic):
        """Thermal velocities as background + perturbation.

        Parameters
        ----------
        psic : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background value) or a list[numpy.array] of the evaluated thermal velocities.
        """

        # collect arguments
        assert isinstance(psic, np.ndarray)

        # assuming that input comes from meshgrid.
        if psic.ndim == 3:
            psic = psic[0, 0, :]

        res = self.maxw_params["vth"][0] + 0.0 * psic

        # TODO: add perturbation

        return res * self.moment_factors["vth"]

    @property
    def add_perturbation(self) -> bool:
        if not hasattr(self, "_add_perturbation"):
            self._add_perturbation = True
        return self._add_perturbation

    @add_perturbation.setter
    def add_perturbation(self, new):
        assert isinstance(new, bool)
        self._add_perturbation = new


class ColdPlasma(Maxwellian):
    r"""Base class for a distribution as a Dirac-delta in velocity (vth = 0).
    The __call__ method returns the density evaluation."""

    @classmethod
    def default_maxw_params(cls):
        """Default parameters dictionary defining the constant value of the constant background."""
        return {
            "n": 5.0,
            "u1": 0.0,
            "u2": 0.0,
            "u3": 0.0,
            "vth1": 0.0,
            "vth2": 0.0,
            "vth3": 0.0,
        }

    def __init__(
        self,
        maxw_params: dict = None,
        pert_params: dict = None,
        equil: FluidEquilibriumWithB = None,
    ):
        super().__init__(
            maxw_params=maxw_params,
            pert_params=pert_params,
            equil=equil,
        )

        # make sure temperatures are zero
        self._maxw_params["vth1"] = 0.0
        self._maxw_params["vth2"] = 0.0
        self._maxw_params["vth3"] = 0.0

    @property
    def coords(self):
        """Coordinates of the constant background."""
        return None

    @property
    def vdim(self):
        """Dimension of the velocity space (vdim = 0)."""
        return 0

    @property
    def is_polar(self):
        """List of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp)."""
        return []

    @property
    def volume_form(self):
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian)."""
        return False

    @property
    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """Jacobian determinant of the velocity coordinate transformation."""
        return 1.0

    def n(self, eta1, eta2, eta3):
        """Zero-th moment (density)."""
        out = self._evaluate_moment(eta1, eta2, eta3, name="n")
        return out

    def u(self, eta1, eta2, eta3):
        """Mean velocities."""
        out = []
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u1")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u2")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="u3")]
        return out

    def vth(self, eta1, eta2, eta3):
        """Thermal velocities (are zero here, see __init__)."""
        out = []
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth1")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth2")]
        out += [self._evaluate_moment(eta1, eta2, eta3, name="vth3")]
        return out

    def __call__(self, eta1, eta2, eta3):
        return self.n(eta1, eta2, eta3)

    @property
    def add_perturbation(self) -> bool:
        if not hasattr(self, "_add_perturbation"):
            self._add_perturbation = True
        return self._add_perturbation

    @add_perturbation.setter
    def add_perturbation(self, new):
        assert isinstance(new, bool)
        self._add_perturbation = new
