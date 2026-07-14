"Maxwellian (Gaussian) distributions in velocity space."

import copy
from typing import Callable

import cunumpy as xp

from struphy.fields_background.base import FluidEquilibriumWithB
from struphy.geometry.base import Domain
from struphy.initial.base import Perturbation
from struphy.io.options import LiteralOptions
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
        # use setter to store input parameters
        self.params = copy.deepcopy(locals())

        self.check_maxw_params()

        # factors multiplied onto the defined moments n, u and vth (can be set via setter)
        self._moment_factors = {
            "n": 1.0,
            "u": [1.0, 1.0, 1.0],
            "vth": [1.0, 1.0, 1.0],
        }

    @property
    def vdim(self):
        """Dimension of the velocity space."""
        return 3

    @property
    def is_polar(self):
        """Tuple of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp)."""
        return (False, False, False)

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
    where :math:`v_\parallel = \mathbf v \cdot \mathbf b_0` and :math:`v_\perp`
    is the radial component of a polar coordinate system perpendicular
    to the magentic direction :math:`\mathbf b_0`.

    Parameters
    ----------
    n, u_para, u_perp, vth_para, vth_perp : tuple
        Moments of the Maxwellian as tuples. The first entry defines the background
        (float for constant background or callable), the second entry defines a Perturbation (can be None).

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
        # use setter to store input parameters
        self.params = copy.deepcopy(locals())

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
    def vdim(self):
        """Dimension of the velocity space."""
        return 2

    @property
    def is_polar(self):
        """Tuple of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp)."""
        return (False, True)

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

        # collect arguments
        assert isinstance(eta1, xp.ndarray)
        assert isinstance(eta2, xp.ndarray)
        assert isinstance(eta3, xp.ndarray)
        assert isinstance(v[0], xp.ndarray)
        assert isinstance(v[1], xp.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape == v[0].shape == v[1].shape
        assert eta1.ndim == 1, "Input arguments must be a marker array."

        etas = [
            xp.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]),
                axis=1,
            ),
        ]

        absB0 = self.equil.absB0(*etas)

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

    def plot_density_profile(
        self,
        dim_1: LiteralOptions.KineticDimensionsToPlot = "e1",
        dim_2: LiteralOptions.KineticDimensionsToPlot | None = None,
        v_lim: float = 5.0,
        resol: int = 100,
        integrate_resol: int = 10,
        logical_coord: tuple[float] = (0.5, 0.5, 0.5),
        in_physical: bool = False,
        domain: Domain | None = None,
        proj_axis: tuple[float,] = (0, 1),
        plot_3D: bool = False,
        title: str | None = None,
        use_mu: bool = False,
        equil: FluidEquilibriumWithB | None = None,
    ):
        if equil is None:
            equil = self.equil
        super().plot_density_profile(
            dim_1,
            dim_2,
            v_lim,
            resol,
            integrate_resol,
            logical_coord,
            in_physical,
            domain,
            proj_axis,
            plot_3D,
            title,
            use_mu=use_mu,
            equil=equil,
        )


class CanonicalMaxwellian(Maxwellian):
    r"""Canonical Maxwellian distribution function in constants-of-motion coordinates.

    The distribution is parameterized by the density and thermal speed as functions of the
    canonical toroidal momentum :math:`\psi_c`:

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
    """

    @classmethod
    def gaussian(self, e, vth=1.0):
        """3-dim. normal distribution, to which array-valued thermal velocities can be passed.

        Parameters
        ----------
        v : float | array-like
            Velocity coordinate(s).

        vth : float | array-like
            Thermal velocity evaluated at position array.

        Returns
        -------
        An array of size(e).
        """

        if isinstance(vth, xp.ndarray):
            assert e.shape == vth.shape, f"{e.shape = } but {vth.shape = }"

        out = 2.0 * xp.sqrt(e / xp.pi) / vth**3 * xp.exp(-e / vth**2)

        return out

    def __call__(self, *args):
        """Evaluates the canonical Maxwellian distribution function.

        There are two use-cases for this function in the code:

        1. Evaluating for particles ("flat evaluation", inputs are all 1D of length N_p)
        2. Evaluating the function on a meshgrid (constants of motion).

        Hence all arguments must always have

        1. the same shape
        2. either ndim = 1 or ndim = 3 (energy, mu, canonical toroidal momentum).

        Parameters
        ----------
        *args : array_like
            Constants of motion arguments in the order eta1, eta2, eta3, v1, ..., vn.

        Returns
        -------
        f : xp.ndarray
            The evaluated Maxwellian.
        """

        # Check that all args have the same shape
        shape0 = xp.shape(args[0])
        for i, arg in enumerate(args):
            assert xp.shape(arg) == shape0, f"Argument {i} has {xp.shape(arg) = }, but must be {shape0 = }."
            assert xp.ndim(arg) == 1 or xp.ndim(arg) == 3, (
                f"{xp.ndim(arg) = } not allowed for canonical Maxwellian evaluation."
            )  # flat or meshgrid evaluation

        # Get result evaluated at eta's
        res = self.n(*args)
        vths = self.vth(*args)

        # take care of correct broadcasting, assuming args come from constants of motion meshgrid
        if xp.ndim(args[0]) == 3:
            # move eta axes to the back
            arg_t = xp.moveaxis(args[0], 0, -1)
            arg_t = xp.moveaxis(arg_t, 0, -1)
            arg_t = xp.moveaxis(arg_t, 0, -1)

            # broadcast
            res_broad = res + 0.0 * arg_t

            # move eta axes to the front
            res = xp.moveaxis(res_broad, -1, 0)
            res = xp.moveaxis(res, -1, 0)
            res = xp.moveaxis(res, -1, 0)

        # Multiply result with gaussian in energy
        # correct broadcasting
        if xp.ndim(args[0]) == 3:
            vth_broad = vths + 0.0 * arg_t
            vth = xp.moveaxis(vth_broad, -1, 0)
            vth = xp.moveaxis(vth, -1, 0)
            vth = xp.moveaxis(vth, -1, 0)
        else:
            vth = vths

        e = self.eval_energy(*args)
        res *= self.gaussian(e, vth=vth)

        return res

    def _evaluate_moment(self, eta1, eta2, eta3, vparallel, mu, *, name: str = "n", add_perturbation: bool = None):
        """Scalar moment evaluation as background + perturbation.

        Parameters
        ----------
        eta1, eta2, eta3 : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        vparallel : numpy.array
            Parallel velocity.

        mu : numpy.array
            Magnetic moment.

        name : str
            Which moment to evaluate (see varaible "dct" below).

        add_perturbation : bool | None
            Whether to add the perturbation defined in maxw_params. If None, is taken from self.add_perturbation.

        Returns
        -------
        A float (background value) or a numpy.array of the evaluated scalar moment.
        """

        # collect arguments
        assert isinstance(eta1, xp.ndarray)
        assert isinstance(eta2, xp.ndarray)
        assert isinstance(eta3, xp.ndarray)
        assert isinstance(vparallel, xp.ndarray)
        assert isinstance(mu, xp.ndarray)

        params = self.maxw_params[name]
        assert isinstance(params, tuple)
        assert len(params) == 2

        # flat evaluation for markers
        if eta1.ndim == 1:
            etas = [
                xp.concatenate(
                    (eta1[:, None], eta2[:, None], eta3[:, None]),
                    axis=1,
                ),
            ]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 4:
            etas = (
                eta1[:, :, :, 0],
                eta2[:, :, :, 0],
                eta3[:, :, :, 0],
            )
        elif eta1.ndim == 5:
            etas = (
                eta1[:, :, :, 0, 0],
                eta2[:, :, :, 0, 0],
                eta3[:, :, :, 0, 0],
            )
        elif eta1.ndim == 6:
            etas = (
                eta1[:, :, :, 0, 0, 0],
                eta2[:, :, :, 0, 0, 0],
                eta3[:, :, :, 0, 0, 0],
            )
        else:
            etas = (eta1, eta2, eta3)

        # initialize output
        if eta1.ndim == 1:
            out = 0.0 * eta1
        else:
            out = 0.0 * etas[0]

        # evaluate background
        background = params[0]
        if isinstance(background, (float, int)):
            out += background
        else:
            assert callable(background)
            # if eta1.ndim == 1:
            #     out += background(eta1, eta2, eta3)
            # else:
            out += background(self.eval_rc(eta1, eta2, eta3, vparallel, mu))

        # add perturbation
        if add_perturbation is None:
            add_perturbation = self.add_perturbation

        perturbation = params[1]
        if perturbation is not None and add_perturbation:
            assert isinstance(perturbation, Perturbation)
            out += perturbation(self.eval_rc(eta1, eta2, eta3, vparallel, mu))

        return out

    @property
    def add_perturbation(self) -> bool:
        if not hasattr(self, "_add_perturbation"):
            self._add_perturbation = True
        return self._add_perturbation

    @add_perturbation.setter
    def add_perturbation(self, new):
        assert isinstance(new, bool)
        self._add_perturbation = new


class CanonicalMaxwellian2D(GyroMaxwellian2D):
    r"""Canonical Maxwellian distribution function in constants-of-motion coordinates.
    Standard evaluation methods in :math:`(v_\parallel, v_\perp)` coordinates are available through :class:`~struphy.kinetic_background.maxwellians.GyroMaxwellian2D`.

    The distribution is parameterized by the density and thermal speed as functions of the
    canonical toroidal momentum :math:`\psi_c`:

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
    n, vth : tuple
        Moments of the canonical Maxwellian as tuples. The first entry defines the background
        (float for constant background or callable), the second entry defines a Perturbation (can be None).

    maxw_params : dict
        Parameters for the kinetic background.

    vth : tuple[float | Callable, Perturbation]
        Thermal-speed background and optional perturbation.

    equil : FluidEquilibriumWithB, optional
        Fluid equilibrium used to evaluate background profiles in the magnetic geometry.

    volume_form : bool, default=True
        If ``True``, represent the distribution as a volume form and include the appropriate
        velocity-space Jacobian when evaluating it.
    """

    def __init__(
        self,
        n: tuple[float | Callable, Perturbation] = (1.0, None),
        vth: tuple[float | Callable, Perturbation] = (1.0, None),
        equil: FluidEquilibriumWithB = None,
        volume_form: bool = True,
        epsilon: float = 1.0,
    ):
        # use setter to store input parameters
        self.params = copy.deepcopy(locals())

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
    def vdim(self):
        """Dimension of the velocity space."""
        return 2

    @property
    def is_polar(self):
        """Tuple of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp)."""
        return (False, True)

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
        for k, v in self.params.items():
            assert isinstance(k, str)
            if isinstance(v, tuple):
                assert len(v) == 2
                assert isinstance(v[0], (float, int, Callable))
                assert isinstance(v[1], Perturbation) or v[1] is None

    def velocity_jacobian_det(self, eta1, eta2, eta3, vparallel, mu):
        r"""TODO"""
        # collect arguments
        assert isinstance(eta1, xp.ndarray)
        assert isinstance(eta2, xp.ndarray)
        assert isinstance(eta3, xp.ndarray)
        assert isinstance(vparallel, xp.ndarray)
        assert isinstance(mu, xp.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape == vparallel.shape == mu.shape
        assert eta1.ndim == 1, "Input arguments must be a marker array."

        etas = [
            xp.concatenate(
                (eta1[:, None], eta2[:, None], eta3[:, None]),
                axis=1,
            ),
        ]

        absB0 = self.equil.absB0(*etas)

        energy = self.eval_energy(eta1, eta2, eta3, vparallel, mu)

        return xp.sqrt(energy) * 2.0 * xp.sqrt(2.0) / absB0

    @property
    def volume_form(self) -> bool:
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian |v_perp|)."""
        return self._volume_form

    @property
    def moment_factors(self):
        """Collection of factors multiplied onto the defined moments n and vth."""
        return self._moment_factors

    @moment_factors.setter
    def moment_factors(self, **kwargs):
        for kw, arg in kwargs.items():
            self._moment_factors[kw] = arg

    def eval_energy(self, eta1, eta2, eta3, vparallel, mu):
        r"""Energy evaluated at given particle positions and velocities."""
        # call domain and equilibrium information
        if eta1.ndim == 1:
            etas = [
                xp.concatenate(
                    (eta1[:, None], eta2[:, None], eta3[:, None]),
                    axis=1,
                ),
            ]
            absB0 = self.equil.absB0(*etas)
        else:
            absB0 = self.equil.absB0(eta1, eta2, eta3)

        # calculate energy
        energy = 1 / 2 * vparallel**2 + mu * absB0

        return energy

    def eval_psic(self, eta1, eta2, eta3, vparallel, mu):
        r"""Shifted canonical toroidal momentum evaluated at given particle positions and velocities."""
        # call domain and equilibrium information
        a1 = self.equil.domain.params["a1"]
        B0 = self.equil.params["B0"]
        R0 = self.equil.params["R0"]
        if eta1.ndim == 1:
            etas = [
                xp.concatenate(
                    (eta1[:, None], eta2[:, None], eta3[:, None]),
                    axis=1,
                ),
            ]
            absB0 = self.equil.absB0(*etas)
        else:
            absB0 = self.equil.absB0(eta1, eta2, eta3)
        psi = self.equil.psi_r(eta1 * (1 - a1) + a1)

        # calculate energy
        energy = self.eval_energy(eta1, eta2, eta3, vparallel, mu)

        # calculate psic
        psic = psi - self._epsilon * B0 * R0 / absB0 * vparallel

        positive_mask = (energy - mu * B0) > 0
        correction = xp.zeros_like(psic)
        correction[positive_mask] = (
            self._epsilon
            * xp.sign(vparallel[positive_mask])
            * xp.sqrt(2 * (energy[positive_mask] - mu[positive_mask] * B0))
            * R0
        )
        psic += correction

        return psic

    def eval_rc(self, eta1, eta2, eta3, vparallel, mu):
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
        """
        # calculate psic
        psic = self.eval_psic(eta1, eta2, eta3, vparallel, mu)

        # calculate rc²
        rc_squared = (psic - self.equil.psi_range[0]) / (self.equil.psi_range[1] - self.equil.psi_range[0])

        # sorting out indices of negative rc²
        neg_index = xp.logical_not(rc_squared >= 0)

        # make them positive
        rc_squared[neg_index] *= -1

        # calculate rc
        rc = xp.sqrt(rc_squared)
        rc[neg_index] *= -1

        return rc

    def n(self, eta1, eta2, eta3, vparallel, mu):
        """Zero-th moment (density)."""
        out = self._evaluate_moment(eta1, eta2, eta3, vparallel, mu, name="n")
        return out * self.moment_factors["n"]

    def u(self, eta1, eta2, eta3, vparallel, mu):
        """Mean velocities."""
        pass

    def vth(self, eta1, eta2, eta3, vparallel, mu):
        """Thermal velocities."""
        out = self._evaluate_moment(eta1, eta2, eta3, vparallel, mu, name="vth")
        return out * self.moment_factors["vth"]

    @property
    def add_perturbation(self) -> bool:
        if not hasattr(self, "_add_perturbation"):
            self._add_perturbation = True
        return self._add_perturbation

    @add_perturbation.setter
    def add_perturbation(self, new):
        assert isinstance(new, bool)
        self._add_perturbation = new

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
        n: tuple[float | Callable, Perturbation] = (1.0, None),
        u1: tuple[float | Callable, Perturbation] = (0.0, None),
        u2: tuple[float | Callable, Perturbation] = (0.0, None),
        u3: tuple[float | Callable, Perturbation] = (0.0, None),
        equil: FluidEquilibriumWithB = None,
    ):
        # use setter to store input parameters
        self.params = copy.deepcopy(locals())
        self._params["vth1"] = (0.0, None)
        self._params["vth2"] = (0.0, None)
        self._params["vth3"] = (0.0, None)

        self.check_maxw_params()

        self._equil = equil

    @property
    def vdim(self):
        """Dimension of the velocity space (vdim = 0)."""
        return 0

    @property
    def is_polar(self):
        """Tuple of booleans of length vdim. True for a velocity coordinate that is a radial polar coordinate (v_perp)."""
        return ()

    @property
    def volume_form(self):
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian)."""
        return False

    @property
    def equil(self) -> FluidEquilibriumWithB:
        """Fluid background with B-field."""
        return self._equil

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
