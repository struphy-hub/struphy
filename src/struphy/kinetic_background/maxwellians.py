"Maxwellian (Gaussian) distributions in velocity space."

import copy
import logging
from inspect import signature
from typing import Callable

import cunumpy as xp

from struphy.fields_background.base import AxisymmMHDequilibrium, FluidEquilibriumWithB
from struphy.geometry.base import Domain
from struphy.initial.base import Perturbation
from struphy.io.options import LiteralOptions
from struphy.kinetic_background.base import Maxwellian

logger = logging.getLogger("struphy")


class Maxwellian3D(Maxwellian):
    r"""A :class:`~struphy.kinetic_background.base.Maxwellian` depending :math:`(\eta_1, \eta_2, \eta_3)`
    and on three (:math:`n=3`) Cartesian velocities.

    Parameters
    ----------
    n, ui, vthi : tuple
        Moments of the Maxwellian as tuples. The first entry defines the background
        (float for constant background or callable), the second entry defines a Perturbation (can be None).

    uniform_on_disc : bool
        Whether the density n is uniform on the disc.
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
        uniform_on_disc: bool = False,
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
    def velocity_coords(self) -> LiteralOptions.VelocityCoordinates:
        """Velocity coordinates of the background."""
        return "cartesian"

    def velocity_jacobian_det(self, eta1, eta2, eta3, vx, vy, vz):
        """Jacobian determinant is 1 (Cartesian velocity coordinates).

        Input parameters should be slice of 2d numpy marker array. (i.e. *self.phasespace_coords.T)

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        vx, vy, vz : array_like
            Velocity evaluation points.

        Returns
        -------
        out : array-like
            The Jacobian determinant evaluated at given logical coordinates.
        -------
        """

        assert eta1.ndim == eta2.ndim == eta3.ndim == 1
        assert vx.ndim == vy.ndim == vz.ndim == 1

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
        for kw, arg in kwargs.items():
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
    two velocities :math:`(v_\parallel, \mu)`, :math:`n=2`,
    where :math:`v_\parallel = \mathbf v \cdot \mathbf b_0` and
    :math:`\mu = v_\perp^2/(2B_0)` is the magnetic moment, with :math:`B_0` the background magnetic field strength.

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

    B0: float | Callable
        Constant or callable background magnetic field strength (default = 2.0).

    uniform_on_disc : bool
        Whether the density n is uniform on the disc.
    """

    def __init__(
        self,
        n: tuple[float | Callable, Perturbation] = (1.0, None),
        u_para: tuple[float | Callable, Perturbation] = (0.0, None),
        u_perp: tuple[float | Callable, Perturbation] = (0.0, None),
        vth_para: tuple[float | Callable, Perturbation] = (1.0, None),
        vth_perp: tuple[float | Callable, Perturbation] = (1.0, None),
        volume_form: bool = True,
        B0: float | Callable = 2.0,
        uniform_on_disc: bool = False,
    ):
        # use setter to store input parameters
        self.params = copy.deepcopy(locals())

        self.check_maxw_params()

        # volume form represenation
        self._volume_form = volume_form

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
    def velocity_coords(self) -> LiteralOptions.VelocityCoordinates:
        """Velocity coordinates of the background."""
        return "vpara_mu"

    def velocity_jacobian_det(self, eta1, eta2, eta3, v_para, mu):
        r"""Jacobian determinant of the velocity coordinate transformation to :math:`(v_\parallel, mu)`, is :math:`B_0`.

        Input parameters should be slice of 2d numpy marker array. (i.e. *self.phasespace_coords.T)

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, mu : array_like
            Parallel velocity and magnetic moment evaluation points.

        Returns
        -------
        out : array-like
            The Jacobian determinant evaluated at given logical coordinates.
        -------
        """
        assert eta1.ndim == eta2.ndim == eta3.ndim == 1
        assert v_para.ndim == mu.ndim == 1

        B0 = self.params["B0"]

        return B0(eta1, eta2, eta3) if callable(B0) else B0 + 0 * eta1

    @property
    def volume_form(self) -> bool:
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian |v_perp|)."""
        return self._volume_form

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


class GyroMaxwellian2Dvperp(Maxwellian):
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

    uniform_on_disc : bool
        Whether the density n is uniform on the disc (default = False).
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
        uniform_on_disc: bool = False,
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
    def velocity_coords(self) -> LiteralOptions.VelocityCoordinates:
        """Velocity coordinates of the background."""
        return "vpara_vperp"

    def velocity_jacobian_det(self, eta1, eta2, eta3, v_para, v_perp):
        r"""Jacobian determinant of the velocity coordinate transformation to :math:`(v_\parallel, v_\perp)`, is :math:`v_\perp`.

        Input parameters should be slice of 2d numpy marker array. (i.e. *self.phasespace_coords.T)

        Parameters
        ----------
        eta1, eta2, eta3 : array_like
            Logical evaluation points.

        v_para, v_perp : array_like
            Parallel and perpendicular velocity evaluation points.

        Returns
        -------
        out : array-like
            The Jacobian determinant evaluated at given logical coordinates.
        -------
        """
        assert eta1.ndim == eta2.ndim == eta3.ndim == 1
        assert v_para.ndim == v_perp.ndim == 1

        return v_perp

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


class CanonicalMaxwellian2D(GyroMaxwellian2D):
    r"""Canonical Maxwellian distribution function in
    :math:`(\eta_1, \eta_2, \eta_3, v_\parallel, \mu)` coordinates.
    Uses caching for evaluation of the canonical toroidal momentum in these coordinates.

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

    equil : AxisymmMHDequilibrium, optional
        Fluid equilibrium used to evaluate background profiles in the magnetic geometry.

    volume_form : bool, default=True
        If ``True``, represent the distribution as a volume form and include the appropriate
        velocity-space Jacobian when evaluating it.

    cache_size : int, optional
        Number of rows in the cache buffer for :math:`\psi_c` evaluation. If ``None``, no caching is used.
        Must be able to accomodate all markers on the current process.
    """

    def __init__(
        self,
        n: tuple[float | Callable, Perturbation] = (1.0, None),
        vth: tuple[float | Callable, Perturbation] = (1.0, None),
        volume_form: bool = True,
        uniform_on_disc: bool = False,
        equil: AxisymmMHDequilibrium = None,
        epsilon: float = 1.0,
        cache_size: int | None = None,
    ):
        assert isinstance(equil, AxisymmMHDequilibrium)

        super().__init__(
            n=n,
            u_para=(0.0, None),
            u_perp=(0.0, None),
            vth_para=vth,
            vth_perp=vth,
            volume_form=volume_form,
            B0=equil.absB0,
            uniform_on_disc=uniform_on_disc,
        )

        # store additional parameters
        self._equil = equil
        self._epsilon = epsilon
        self.params["vth"] = vth
        self.params["equil"] = equil
        self.params["epsilon"] = epsilon
        self.params["cache_size"] = cache_size

        # factors multiplied onto the defined moments n and vth (can be set via setter)
        self._moment_factors = {
            "n": 1.0,
            "vth": 1.0,
        }

        # create cache for psi_c evaluation
        if cache_size is not None:
            self.cbufs = {}
            self.cbufs["absB0"] = xp.empty(cache_size, dtype=float)
            self.cbufs["x"] = xp.empty(cache_size, dtype=float)
            self.cbufs["y"] = xp.empty(cache_size, dtype=float)
            self.cbufs["z"] = xp.empty(cache_size, dtype=float)
            self.cbufs["R"] = xp.empty(cache_size, dtype=float)
            self.cbufs["P"] = xp.empty(cache_size, dtype=float)
            self.cbufs["Z"] = xp.empty(cache_size, dtype=float)
            self.cbufs["psi"] = xp.empty(cache_size, dtype=float)
            self.cbufs["energy"] = xp.empty(cache_size, dtype=float)
            self.cbufs["psic"] = xp.empty(cache_size, dtype=float)
            self.cbufs["positive_mask"] = xp.empty(cache_size, dtype=bool)
            self.cbufs["correction"] = xp.empty(cache_size, dtype=float)
            logger.debug(f"Created {len(self.cbufs)} cache buffers for psi_c evaluation, each with size {cache_size}.")
        else:
            self.cbufs = None

    @property
    def equil(self) -> AxisymmMHDequilibrium:
        """One of :mod:`~struphy.fields_background.equils`
        in case that moments are to be set in that way, None otherwise.
        """
        return self._equil

    @property
    def epsilon(self) -> float:
        """Epsilon parameter in the canonical toroidal momentum."""
        return self._epsilon

    def _evaluate_moment(
        self,
        eta1,
        eta2,
        eta3,
        v_parallel,
        mu,
        *,
        name: str = "n",
        add_perturbation: bool = None,
    ):
        """Scalar moment evaluation as background + perturbation.
        Incontrast to standard Maxwellians, here the moments are evaluated
        at the phase space coordinates.

        Parameters
        ----------
        eta1, eta2, eta3, v_parallel, mu : numpy.arrays
            Phase space evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        name : str
            Which moment to evaluate (see varaible "dct" below).

        add_perturbation : bool | None
            Whether to add the perturbation defined in params. If None, is taken from self.add_perturbation.

        Returns
        -------
        A float (background value) or a numpy.array of the evaluated scalar moment.
        """

        # collect arguments
        assert isinstance(eta1, xp.ndarray)
        assert isinstance(eta2, xp.ndarray)
        assert isinstance(eta3, xp.ndarray)
        assert isinstance(v_parallel, xp.ndarray)
        assert isinstance(mu, xp.ndarray)
        assert eta1.shape == eta2.shape == eta3.shape == v_parallel.shape == mu.shape

        params = self.params[name]
        assert isinstance(params, tuple)
        assert len(params) == 2

        # flat evaluation for markers
        if eta1.ndim == 1:
            coords = [
                xp.concatenate(
                    (eta1[:, None], eta2[:, None], eta3[:, None], v_parallel[:, None], mu[:, None]),
                    axis=1,
                ),
            ]
        # assuming that input comes from meshgrid.
        elif eta1.ndim == 5:
            coords = (eta1, eta2, eta3, v_parallel, mu)
        else:
            raise ValueError(f"Input arrays must be 1d or 5d (from meshgrid), but got {eta1.ndim}d.")

        # initialize output
        if eta1.ndim == 1:
            out = 0.0 * eta1
        else:
            out = 0.0 * coords[0]

        # evaluate background
        background = params[0]
        if isinstance(background, (float, int)):
            out += background
        else:
            assert callable(background)
            sig = signature(background)
            assert len(sig.parameters) == 1, (
                f"Background function {background} must take one argument (psi_c), but takes {len(sig.parameters)}."
            )

            cached = self._check_psi_c_cached(*coords)
            logger.debug(f"{'Using cached psi_c' if cached else 'Evaluating psi_c'} for background evaluation.")

            if not cached:
                self.psi_c = self.eval_psic(*coords)
            out += background(self.psi_c)

        # add perturbation
        if add_perturbation is None:
            add_perturbation = self.add_perturbation

        perturbation = params[1]
        if perturbation is not None and add_perturbation:
            assert isinstance(perturbation, Perturbation)
            if eta1.ndim == 1:
                out += perturbation(eta1, eta2, eta3)
            else:
                raise NotImplementedError("Perturbation evaluation for meshgrid input not implemented yet.")

        # uniform density on disc (n=2 eta_1)
        if name == "n" and self.params.get("uniform_on_disc", False):
            if eta1.ndim == 1:
                out *= 2.0 * eta1
            else:
                out *= 2.0 * coords[0]

        return out

    def _check_psi_c_cached(self, *coords):
        """Check if psi_c has been cached for the given coordinates."""
        cached = False
        if hasattr(self, "psi_c"):
            if self.psi_c is not None:
                if self.psi_c.shape == coords[0].shape:
                    if len(coords) == 1:
                        test_coords = coords[0][self.test_mask]
                        cached_psi_c = self.eval_psic(test_coords)
                        cached = xp.allclose(self.psi_c[self.test_mask], cached_psi_c)
                else:
                    if len(coords) == 1:
                        self.test_mask = xp.zeros_like(coords[0], dtype=bool)
                        n_markers = coords[0].shape[0]
                        self.test_mask[0] = True
                        self.test_mask[-1] = True
                        self.test_mask[1 * n_markers // 3] = True
                        self.test_mask[2 * n_markers // 3] = True
        return cached

    def eval_psic(self, *coords):
        r"""Shifted canonical toroidal momentum evaluated at given particle positions and velocities."""

        a1 = self.equil.domain.params["a1"]
        B0 = self.equil.params["B0"]
        R0 = self.equil.params["R0"]

        if len(coords) == 1:
            if self.cbufs is None:
                logger.warning(
                    f"Initialize {self.__class__.__name__} with `cache_size` for faster psi_c evaluation for markers!"
                )
            etas = coords[0][:, :3]  # these are views (no mem allocation)
            vparallel = coords[0][:, 3]
            mu = coords[0][:, 4]
            n_markers = etas.shape[0]
            if self.cbufs is None:
                absB0 = self.equil.absB0(etas)
                x, y, z = self.equil.domain(etas)
            else:
                absB0 = self.cbufs["absB0"][:n_markers]
                x = self.cbufs["x"][:n_markers]
                y = self.cbufs["y"][:n_markers]
                z = self.cbufs["z"][:n_markers]
                absB0[:] = self.equil.absB0(etas)
                x[:], y[:], z[:] = self.equil.domain(etas)
        else:
            assert len(coords) == 5
            assert coords[0].ndim == coords[1].ndim == coords[2].ndim == coords[3].ndim == coords[4].ndim == 5
            eta1 = coords[0][:, :, :, 0, 0]
            eta2 = coords[1][:, :, :, 0, 0]
            eta3 = coords[2][:, :, :, 0, 0]
            etas = (eta1, eta2, eta3)
            absB0 = self.equil.absB0(*etas)[:, :, :, None, None]
            x, y, z = self.equil.domain(*etas)
            vparallel = coords[3]
            mu = coords[4]

        if self.cbufs is None or len(coords) != 1:
            R, P, Z = self.equil.inverse_map(x, y, z)
            psi = self.equil.psi(R, Z)
            if len(coords) != 1:
                psi = psi[:, :, :, None, None]

            energy = 1 / 2 * vparallel**2 + mu * absB0

            psi_c = psi - self._epsilon * B0 * R0 / absB0 * vparallel

            positive_mask = (energy - mu * B0) > 0
            correction = xp.zeros_like(psi_c)
            correction[positive_mask] = (
                self._epsilon
                * xp.sign(vparallel[positive_mask])
                * xp.sqrt(2 * (energy[positive_mask] - mu[positive_mask] * B0))
                * R0
            )
            psi_c += correction
        else:
            R = self.cbufs["R"][:n_markers]
            P = self.cbufs["P"][:n_markers]
            Z = self.cbufs["Z"][:n_markers]
            psi = self.cbufs["psi"][:n_markers]
            energy = self.cbufs["energy"][:n_markers]
            psi_c = self.cbufs["psic"][:n_markers]
            positive_mask = self.cbufs["positive_mask"][:n_markers]
            correction = self.cbufs["correction"][:n_markers]

            R[:], P[:], Z[:] = self.equil.inverse_map(x, y, z)
            psi[:] = self.equil.psi(R, Z)

            energy[:] = 1 / 2 * vparallel**2 + mu * absB0

            psi_c[:] = psi - self._epsilon * B0 * R0 / absB0 * vparallel

            positive_mask[:] = (energy - mu * B0) > 0
            correction[:] = 0.0
            correction[positive_mask] = (
                self._epsilon
                * xp.sign(vparallel[positive_mask])
                * xp.sqrt(2 * (energy[positive_mask] - mu[positive_mask] * B0))
                * R0
            )
            psi_c[:] += correction
        return psi_c

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
        """Mean velocities (zero for the canonical Maxwellian)."""
        return [0.0 * eta1, 0.0 * eta1]

    def vth(self, eta1, eta2, eta3, vparallel, mu):
        """Thermal velocities."""
        out = self._evaluate_moment(eta1, eta2, eta3, vparallel, mu, name="vth")
        out = out * self.moment_factors["vth"]
        return [out, out]


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
        uniform_on_disc: bool = False,
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
    def velocity_coords(self) -> LiteralOptions.VelocityCoordinates:
        """Velocity coordinates of the background."""
        return None

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
        return 1.0 + 0.0 * eta1

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
