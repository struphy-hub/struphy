"Base classes for kinetic backgrounds."

import copy
from abc import ABCMeta, abstractmethod
from typing import Callable

import cunumpy as xp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from struphy.fields_background.base import FluidEquilibriumWithB
from struphy.geometry.base import Domain
from struphy.initial.base import Perturbation
from struphy.io.options import LiteralOptions
from struphy.utils.utils import __class_with_params_repr_no_defaults__


class KineticBackground(metaclass=ABCMeta):
    r"""Base class for kinetic background distributions.

    Kinetic backgrounds are mainly used for particle weight computation:

    * they appear as initial conditions in the numerator of particle weights
    * they are evaluated at particle coordinates in the control-variate method for noise reduction.

    Kinetic backgrounds can be defined in arbitrary phase space coordinates.
    A determinant of the velocity Jacobian must be provided in the subclasses.
    """

    @property
    @abstractmethod
    def vdim(self):
        """Dimension of the velocity space (vdim = n)."""
        pass

    @property
    @abstractmethod
    def velocity_coords(self) -> LiteralOptions.VelocityCoordinates:
        """Velocity coordinates of the background."""
        pass

    @abstractmethod
    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """Jacobian determinant of the velocity coordinate transformation (starting from Cartesian velocity coordinates)."""
        pass

    @property
    @abstractmethod
    def volume_form(self) -> bool:
        """True if the background is represented as a volume form (thus including the velocity Jacobian)."""
        pass

    @abstractmethod
    def n(self, *coords):
        """Number density (0-form).

        Parameters
        ----------
        coords : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A numpy.array with the density evaluated at evaluation points (same shape as coords).
        """
        pass

    @abstractmethod
    def u(self, *coords):
        """Mean velocities (Cartesian components).

        Parameters
        ----------
        coords : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background values) or a list[numpy.array] of the evaluated velocities.
        """
        pass

    @abstractmethod
    def __call__(self, *phase_space_coords):
        """Evaluates the background distribution function f0 at the given phase space coordinates.

        There are two use-cases for this function in the code:

        1. Evaluating for particles ("flat evaluation", inputs are all 1D of length N_p)
        2. Evaluating the function on a meshgrid (in phase space).

        Hence all arguments must always have

        1. the same shape
        2. either ndim = 1 or ndim = 3 + vdim.

        Parameters
        ----------
        *phase_space_coords : array_like
            Position-velocity arguments.

        Returns
        -------
        f0 : xp.ndarray
            The evaluated background.
        """
        pass

    @property
    def params(self) -> dict:
        """Parameters passed to __init__(), as dictionary."""
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

    def __repr__(self):
        out = f"{self.__class__.__name__}(\n"
        for k, v in self.params.items():
            out += " " * 4
            out += f"{k}={v},\n"
        out += ")"
        return out

    def __repr_no_defaults__(self):
        return __class_with_params_repr_no_defaults__(self)

    def plot_density_profile(
        self,
        dim_1: LiteralOptions.KineticDimensionsToPlot = "e1",
        dim_2: LiteralOptions.KineticDimensionsToPlot | None = None,
        v_lim: float = 5.0,
        resol: int = 100,
        integrate_resol: int = 50,
        logical_coord: tuple[float] = (0.5, 0.5, 0.5),
        in_physical: bool = False,
        domain: Domain | None = None,
        proj_axis: tuple[float,] = (0, 1),
        plot_3D: bool = False,
        title: str | None = None,
        use_mu=False,
        equil=None,
    ):
        """
        Plots the density profile (slices) of the phase space background distribution. The slice can either be 1D or 2D, in logical or in Cartesian coordinates.

        Parameters
        ----------
        dim_1, dim_2 : LiteralOptions.KineticDimensionsToPlot = ["e1","e2","e3","v1","v2","v3"]
            The axes used in the projection, they refere to logical space axes. If dim_2 is not defined the projection is 1D, it is 2D if dim_2 is attributed.

        v_lim : float = 5.0
            Limit value of the velocity axes.

        resol : int = 100
            Resolution along each axes

        integrate_resol : int = 10
            Resolution along not used velocity axes. The density is reduced (with a maximum function) over these axes before being plotted. High values (>50) may require much memory.

        logical_coord : tuple[float] = (0.5, 0.5, 0.5)
            Refere to the default coordinate (in logical space) attributed to each axe which is not used in the projection.

        in_physical : bool = False
            Specify if the result is plotted in logical coordinates or in Cartesian coordinates, has a effect in 2D plotting. If True, you must specify a domain.

        domain : Domain | None = None
            Domain used to plot the density if in_physical=True.

        proj_axis : tuple[float] = (0,1)
            Axes of the Cartesian coordinates used to plot the density: 0->x, 1->y, 2->z.
            I you do not see the density profile in 2D, you may change these axes.

        plot_3D : bool = False
            Plots a surface in a 3D environment. Only for physical projection.
        """
        integrate_resol = min(integrate_resol, 100)  # to avoid memory issues
        if in_physical:
            if not (dim_1 in ["e1", "e2", "e3"] and dim_2 in ["e1", "e2", "e3"]):
                AssertionError(
                    'To perform a plot in physical space you must use two space axes (dim_1, dim_2 in ["e1","e2","e3"]).'
                )
        if plot_3D and not in_physical:
            AssertionError("To perform a 3D plot you must plot in physical space (activate in_physical).")
        assert 0 <= proj_axis[0] < proj_axis[1] < 3
        if dim_2 is None:
            if dim_1 == "e1":
                axe_to_plot = 0
            elif dim_1 == "e2":
                axe_to_plot = 1
            elif dim_1 == "e3":
                axe_to_plot = 2
            elif dim_1 == "v1":
                axe_to_plot = 3
            elif dim_1 == "v2":
                axe_to_plot = 4
            elif dim_1 == "v3":
                axe_to_plot = 5
            else:
                AssertionError("dim_1argument must match an exiting dimension")
            if axe_to_plot - 3 > self.vdim:
                AssertionError("Coordinate " + dim_1 + " does not exist with this background")
            linspace_space = xp.array([0.0])
            integrate_linspace_vel = xp.linspace(0.0, v_lim, integrate_resol)
            if axe_to_plot < 3:
                plot_linspace = xp.linspace(0.0, 1.0, resol)
            else:
                plot_linspace = xp.linspace(-v_lim, v_lim, resol)
            tabs = 3 * [linspace_space] + self.vdim * [integrate_linspace_vel]
            for i in range(3):
                tabs[i][0] = logical_coord[i]
            tabs[axe_to_plot] = plot_linspace
            etas = xp.abs(xp.meshgrid(*tabs, indexing="ij"))
            total_density = self(*etas)
            if use_mu and axe_to_plot == 4:
                B_norm_tab = equil.absB0(
                    etas[0][tuple([slice(None), slice(None), slice(None)] + self.vdim * [0])],
                    etas[1][tuple([slice(None), slice(None), slice(None)] + self.vdim * [0])],
                    etas[2][tuple([slice(None), slice(None), slice(None)] + self.vdim * [0])],
                )
                B_norm_tab = xp.broadcast_to(
                    B_norm_tab[
                        tuple(
                            [
                                ...,
                            ]
                            + self.vdim * [None]
                        )
                    ],
                    etas[0].shape,
                )
                total_density *= B_norm_tab / etas[4]
                plot_linspace = plot_linspace**2
            axes_to_integrate = [i for i in range(3 + self.vdim)]
            axes_to_integrate.remove(axe_to_plot)
            total_density = xp.mean(total_density, tuple(axes_to_integrate))
            fig, ax = plt.subplots(1, 1)
            ax.plot(plot_linspace, total_density)
            ax.set_xlabel(dim_1)
            ax.set_ylabel("density")
            ax.set_title("background profile")
            plt.show(block=True)
        else:
            if dim_1 == "e1":
                axe_to_plot1 = 0
            elif dim_1 == "e2":
                axe_to_plot1 = 1
            elif dim_1 == "e3":
                axe_to_plot1 = 2
            elif dim_1 == "v1":
                axe_to_plot1 = 3
            elif dim_1 == "v2":
                axe_to_plot1 = 4
            elif dim_1 == "v3":
                axe_to_plot1 = 5
            else:
                AssertionError("dim_1 argument must match an existing dimension")
            if dim_2 == "e1":
                axe_to_plot2 = 0
            elif dim_2 == "e2":
                axe_to_plot2 = 1
            elif dim_2 == "e3":
                axe_to_plot2 = 2
            elif dim_2 == "v1":
                axe_to_plot2 = 3
            elif dim_2 == "v2":
                axe_to_plot2 = 4
            elif dim_2 == "v3":
                axe_to_plot2 = 5
            else:
                AssertionError("dim_2 argument must match an existing dimension")
            if axe_to_plot2 == axe_to_plot1:
                AssertionError("You must specify different dimensions for dim_1 and dim_2")
            integrate_linspace_vel = xp.linspace(0.0, v_lim, integrate_resol)
            tabs = [xp.array([logical_coord[i]]) for i in range(3)] + self.vdim * [integrate_linspace_vel]
            if axe_to_plot1 < 3:
                plot_linspace1 = xp.linspace(0.0, 1.0, resol)
            elif axe_to_plot1 != 4 or (not use_mu):
                plot_linspace1 = xp.linspace(-v_lim, v_lim, resol)
            else:
                plot_linspace1 = xp.linspace(0.0, xp.sqrt(v_lim), resol)
            if axe_to_plot2 < 3:
                plot_linspace2 = xp.linspace(0.0, 1.0, resol)
            elif axe_to_plot2 != 4 or (not use_mu):
                plot_linspace2 = xp.linspace(-v_lim, v_lim, resol)
            else:
                plot_linspace2 = xp.linspace(0.0, xp.sqrt(v_lim), resol)
            tabs[axe_to_plot1] = plot_linspace1
            tabs[axe_to_plot2] = plot_linspace2
            etas = xp.meshgrid(*tabs, indexing="ij")
            if in_physical:
                physical_coords = domain(*tabs[:3])
                physical_coords = list(physical_coords)
                for i in range(3):
                    physical_coords[i] = physical_coords[i][tuple([...] + self.vdim * [None])]
                    physical_coords[i] = xp.broadcast_to(physical_coords[i], etas[0].shape)
                for i in range(self.vdim):
                    physical_coords.append(etas[i + 3])
                total_density = self(*xp.abs(etas))
            else:
                physical_coords = list(etas)
                total_density = self(*xp.abs(etas))
            if use_mu:
                if axe_to_plot1 == 4 or axe_to_plot2 == 4:
                    B_norm_tab = equil.absB0(
                        etas[0][tuple([slice(None), slice(None), slice(None)] + self.vdim * [0])],
                        etas[1][tuple([slice(None), slice(None), slice(None)] + self.vdim * [0])],
                        etas[2][tuple([slice(None), slice(None), slice(None)] + self.vdim * [0])],
                    )
                    B_norm_tab = xp.broadcast_to(
                        B_norm_tab[
                            tuple(
                                [
                                    ...,
                                ]
                                + self.vdim * [None]
                            )
                        ],
                        etas[0].shape,
                    )
                    total_density *= B_norm_tab / etas[4]
                    physical_coords[4] = physical_coords[4] ** 2
            axes_to_integrate = [i for i in range(3 + self.vdim)]
            axes_to_integrate.remove(axe_to_plot1)
            axes_to_integrate.remove(axe_to_plot2)
            total_density = xp.mean(total_density, tuple(axes_to_integrate))
            id_dim = [0] * len(etas)
            id_dim[axe_to_plot1] = slice(None)
            id_dim[axe_to_plot2] = slice(None)
            if not plot_3D:
                if in_physical:
                    X = physical_coords[proj_axis[0]][tuple(id_dim)]
                    Y = physical_coords[proj_axis[1]][tuple(id_dim)]
                else:
                    X = physical_coords[axe_to_plot1][tuple(id_dim)]
                    Y = physical_coords[axe_to_plot2][tuple(id_dim)]
                fig, ax = plt.subplots()
                for_color = ax.pcolor(X, Y, total_density)
                if in_physical:
                    ax.set_xlabel(["x", "y", "z"][proj_axis[0]])
                    ax.set_ylabel(["x", "y", "z"][proj_axis[1]])
                else:
                    if use_mu:
                        if dim_1 == "v2":
                            dim_1 = "mu"
                        if dim_2 == "v2":
                            dim_2 = "mu"
                    ax.set_xlabel(dim_1)
                    ax.set_ylabel(dim_2)
            else:
                norm = Normalize(total_density.min(), total_density.max() + 0.01)
                colors = cm.viridis(norm(total_density))
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                for_color = ax.plot_surface(
                    X=physical_coords[0][tuple(id_dim)],
                    Y=physical_coords[1][tuple(id_dim)],
                    Z=physical_coords[2][tuple(id_dim)],
                    facecolors=colors,
                )
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            if title is None:
                ax.set_title(f"Density in ({dim_1}, {dim_2}) space")
            else:
                ax.set_title(title)
            fig.colorbar(for_color)
            plt.show(block=True)

    def __add__(self, other_f0):
        return SumKineticBackground(self, other_f0)

    def __mul__(self, a):
        return ScalarMultiplyKineticBackground(self, a)

    def __rmul__(self, a):
        return ScalarMultiplyKineticBackground(self, a)

    def __div__(self, a):
        assert isinstance(a, float) or isinstance(a, int) or isinstance(a, xp.int64)
        assert a != 0, "Cannot divide by zero!"
        return ScalarMultiplyKineticBackground(self, 1 / a)

    def __rdiv__(self, a):
        assert isinstance(a, float) or isinstance(a, int) or isinstance(a, xp.int64)
        assert a != 0, "Cannot divide by zero!"
        return ScalarMultiplyKineticBackground(self, 1 / a)

    def __sub__(self, other_f0):
        return SumKineticBackground(self, ScalarMultiplyKineticBackground(other_f0, -1.0))


class SumKineticBackground(KineticBackground):
    def __init__(self, f1, f2):
        # use setter to store input parameters
        self.params = copy.deepcopy(locals())

        assert isinstance(f1, KineticBackground)
        assert isinstance(f2, KineticBackground)
        assert f1.vdim == f2.vdim
        assert f1.velocity_coords == f2.velocity_coords
        assert f1.volume_form == f2.volume_form

        self._f1 = f1
        self._f2 = f2

        if hasattr(f1, "_equil"):
            assert f1.equil is f2.equil
            self._equil = f1.equil

    @property
    def vdim(self):
        """Dimension of the velocity space (vdim = n)."""
        return self._f1.vdim

    @property
    def velocity_coords(self):
        """Velocity coordinates of the background."""
        return self._f1.velocity_coords

    @property
    def volume_form(self):
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian)."""
        return self._f1.volume_form

    @property
    def equil(self) -> FluidEquilibriumWithB:
        """Fluid background with B-field."""
        if not hasattr(self, "_equil"):
            self._equil = None
        return self._equil

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """Jacobian determinant of the velocity coordinate transformation."""
        return self._f1.velocity_jacobian_det(eta1, eta2, eta3, *v)

    def n(self, *coords):
        return self._f1.n(*coords) + self._f2.n(*coords)

    def u(self, *coords):
        n1 = self._f1.n(*coords)
        n2 = self._f2.n(*coords)
        u1s = self._f1.u(*coords)
        u2s = self._f2.u(*coords)

        return [(n1 * u1 + n2 * u2) / (n1 + n2) for u1, u2 in zip(u1s, u2s)]

    def __call__(self, *phase_space_coords):
        return self._f1(*phase_space_coords) + self._f2(*phase_space_coords)


class ScalarMultiplyKineticBackground(KineticBackground):
    def __init__(self, f0, a):
        # use setter to store input parameters
        self.params = copy.deepcopy(locals())

        assert isinstance(f0, KineticBackground)
        assert isinstance(a, float) or isinstance(a, int) or isinstance(a, xp.int64)

        self._f = f0
        self._a = a

    @property
    def vdim(self):
        """Dimension of the velocity space (vdim = n)."""
        return self._f.vdim

    @property
    def velocity_coords(self):
        """Velocity coordinates of the background."""
        return self._f.velocity_coords

    @property
    def volume_form(self):
        """Boolean. True if the background is represented as a volume form (thus including the velocity Jacobian)."""
        return self._f.volume_form

    def velocity_jacobian_det(self, eta1, eta2, eta3, *v):
        """Jacobian determinant of the velocity coordinate transformation."""
        return self._f.velocity_jacobian_det(eta1, eta2, eta3, *v)

    def n(self, *coords):
        return self._a * self._f.n(*coords)

    def u(self, *coords):
        return self._f.u(*coords)

    def __call__(self, *phase_space_coords):
        return self._a * self._f(*phase_space_coords)


class Maxwellian(KineticBackground):
    r"""Base class for a Maxwellian distribution function.
    It is defined on :math:`[0, 1]^3 \times \mathbb R^n, n \geq 1,`
    with logical position coordinates :math:`\boldsymbol{\eta} \in [0, 1]^3`:

    .. math::

        f(\boldsymbol{\eta}, v_1,\ldots,v_n) = n(\boldsymbol{\eta}) \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\,v_{\mathrm{th},i}(\boldsymbol{\eta})}
        \exp\left[-\frac{(v_i-u_i(\boldsymbol{\eta}))^2}{2\,v_{\mathrm{th},i}(\boldsymbol{\eta})^2}\right],

    defined by its velocity moments: the density :math:`n(\boldsymbol{\eta})`,
    the mean-velocities :math:`u_i(\boldsymbol{\eta})`,
    and the thermal velocities :math:`v_{\mathrm{th},i}(\boldsymbol{\eta})`.
    """

    @abstractmethod
    def vth(self, *coords):
        """Thermal velocities (0-forms).

        Parameters
        ----------
        coords : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        Returns
        -------
        A list[float] (background values) or a list[numpy.array] of the evaluated thermal velocities.
        """
        pass

    def check_maxw_params(self):
        for k, v in self.params.items():
            assert isinstance(k, str)
            if isinstance(v, tuple):
                assert len(v) == 2
                assert isinstance(v[0], (float, int, Callable))
                assert isinstance(v[1], Perturbation) or v[1] is None

    # def __repr__(self):
    #     out = f"    {self.__class__.__name__}:"
    #     out += "\n        maxw_params: (background, perturbation)"
    #     for k, v in self.maxw_params.items():
    #         out += f"\n            {k}: {v}"
    #     return out
    
    @property
    def gauss_types(self) -> tuple[LiteralOptions.OptsGaussianCoordinate]:
        """Velocity coordinate types of the Maxwellian (one per velocity dimension)."""
        if self.velocity_coords == "cartesian":
            self._gauss_types = ("cartesian",) * self.vdim
        elif self.velocity_coords == "vpara_vperp":
            self._gauss_types = ("cartesian", "polar")
        elif self.velocity_coords == "vpara_mu":
            self._gauss_types = ("cartesian", "mu")
        elif self.velocity_coords == "vpara_energy":
            self._gauss_types = ("cartesian", "energy")
        else:
            raise ValueError(
                f"Unknown velocity coordinates {self.velocity_coords}, must be one of ['cartesian', 'vpara_vperp', 'vpara_mu', 'vpara_energy']"
            )
        return self._gauss_types

    @classmethod
    def gaussian(self, v, u=0.0, vth=1.0, B0=2.0, type: LiteralOptions.OptsGaussianCoordinate = "cartesian", volume_form=False):
        r"""1-dim. normal distribution, to which array-valued mean- and thermal velocities can be passed.

        The ``type`` selects the velocity coordinate of the Maxwellian:

        - ``"cartesian"``: standard Gaussian,

          .. math::
              G(v) = \frac{1}{\sqrt{2\pi}\,v_{\mathrm{th}}}\exp\left[-\frac{(v-u)^2}{2\,v_{\mathrm{th}}^2}\right]\,.

        - ``"polar"``: :math:`v \geq 0` is the radial coordinate of a polar representation
          :math:`(v, \theta)` of a 2d isotropic Gaussian velocity space (e.g. :math:`v_\perp`
          in gyro-/drift-kinetic Maxwellians, gyro-angle already integrated out), requires :math:`u=0`,

          .. math::
              G_{\mathrm{polar}}(v) = \frac{1}{v_{\mathrm{th}}^2}\exp\left[-\frac{v^2}{2\,v_{\mathrm{th}}^2}\right]\,.

        - ``"energy"``: :math:`v \geq 0` is an energy-like coordinate such as :math:`\mu|\mathbf B| = m v_\perp^2/2`,
        requires :math:`u=0`, ``volume_form`` must be ``False`` (its Jacobian depends on :math:`B^*`),

          .. math::
              G_{\mathrm{energy}}(v) = \frac{1}{v_{\mathrm{th}}^2}\exp\left[-\frac{v}{v_{\mathrm{th}}^2}\right]\,.

        For ``"polar"``, ``volume_form=True`` multiplies by the polar velocity Jacobian
        :math:`|v|` (needed to integrate to 1 over :math:`v \in [0,\infty)`; for :math:`u=0`
        this reduces to the Rayleigh distribution); ``volume_form=False`` leaves the Jacobian
        out, corresponding to the 0-form (density) representation used elsewhere in the
        discretization.

        Parameters
        ----------
        v : float | array-like
            Velocity coordinate; must be non-negative if ``type`` is ``"polar"`` or ``"energy"``.

        u : float | array-like
            Mean velocity evaluated at position array, same shape as v.
            Must be 0 unless ``type == "cartesian"``.

        vth : float | array-like
            Thermal velocity evaluated at position array, same shape as v.
            
        B0: float | array-like
            Background magnetic field evaluated at position array, same shape as v. 
            Only used for ``type == "mu"``.

        type : str
            Velocity coordinate type, one of ``"cartesian"``, ``"polar"``, ``"mu"``, ``"energy"``.

        volume_form : bool
            If True, multiply by the polar velocity Jacobian |v|. Only valid for ``type == "polar"``.

        Returns
        -------
        An array of size(v).
        """

        if isinstance(v, xp.ndarray):
            if isinstance(u, xp.ndarray):
                assert v.shape == u.shape, f"{v.shape =} but {u.shape =}"
            if isinstance(vth, xp.ndarray):
                assert v.shape == vth.shape, f"{v.shape =} but {vth.shape =}"
            if isinstance(B0, xp.ndarray):
                assert v.shape == B0.shape, f"{v.shape =} but {B0.shape =}"

        if type == "cartesian":
            out = 1.0 / vth * 1.0 / xp.sqrt(2.0 * xp.pi) * xp.exp(-((v - u) ** 2) / (2.0 * vth**2))
        elif type == "polar":
            assert xp.all(v >= 0.0)
            assert xp.all(u == 0.0)
            out = 1.0 / vth**2 * xp.exp(-(v**2) / (2.0 * vth**2))
            if volume_form:
                out *= v
        elif type == "mu":
            assert xp.all(v >= 0.0)
            assert xp.all(u == 0.0)
            out = 1.0 / vth**2 * xp.exp(-v*B0 / vth**2)
            if volume_form:
                out *= B0
        elif type == "energy":
            assert xp.all(v >= 0.0)
            assert xp.all(u == 0.0)
            out = 1.0 / vth**2 * xp.exp(-v / vth**2)
        else:
            raise ValueError(
                f"Unknown Gaussian coordinate type {type}. Must be one of ['cartesian', 'polar', 'mu', 'energy']."
            )

        return out

    def __call__(self, *phase_space_coords):
        """Evaluates the Maxwellian distribution function.

        There are two use-cases for this function in the code:

        1. Evaluating for particles ("flat evaluation", inputs are all 1D of length N_p)
        2. Evaluating the function on a meshgrid (in phase space).

        Hence all arguments must always have

        1. the same shape
        2. either ndim = 1 or ndim = 3 + vdim.

        Parameters
        ----------
        *phase_space_coords : array_like
            Phase space coordinates (position and velocity).

        Returns
        -------
        f : xp.ndarray
            The evaluated Maxwellian.
        """
        args = phase_space_coords

        # Check that all args have the same shape
        shape0 = xp.shape(args[0])
        for i, arg in enumerate(args):
            assert xp.shape(arg) == shape0, f"Argument {i} has {xp.shape(arg) =}, but must be {shape0 =}."
            assert xp.ndim(arg) == 1 or xp.ndim(arg) == 3 + self.vdim, (
                f"{xp.ndim(arg) =} not allowed for Maxwellian evaluation."
            )  # flat or meshgrid evaluation

        # Get result evaluated at eta's
        res = self.n(*args[: -self.vdim])
        us = self.u(*args[: -self.vdim])
        vths = self.vth(*args[: -self.vdim])

        # take care of correct broadcasting, assuming args come from phase space meshgrid
        if xp.ndim(args[0]) > 3:
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

        # Multiply result with gaussian in v's
        for i, v in enumerate(args[-self.vdim :]):
            # correct broadcasting
            if xp.ndim(args[0]) > 3:
                u_broad = us[i] + 0.0 * arg_t
                u = xp.moveaxis(u_broad, -1, 0)
                u = xp.moveaxis(u, -1, 0)
                u = xp.moveaxis(u, -1, 0)

                vth_broad = vths[i] + 0.0 * arg_t
                vth = xp.moveaxis(vth_broad, -1, 0)
                vth = xp.moveaxis(vth, -1, 0)
                vth = xp.moveaxis(vth, -1, 0)
            else:
                u = us[i]
                vth = vths[i]

            res *= self.gaussian(v, u=u, vth=vth, type=self.gauss_types[i], volume_form=self.volume_form)

        return res

    def _evaluate_moment(self, *coords, name: str = "n", add_perturbation: bool = None):
        """Scalar moment evaluation as background + perturbation.

        Parameters
        ----------
        coords : numpy.arrays
            Evaluation points. All arrays must be of same shape (can be 1d for flat evaluation).

        name : str
            Which moment to evaluate (see varaible "dct" below).

        add_perturbation : bool | None
            Whether to add the perturbation defined in params. If None, is taken from self.add_perturbation.

        Returns
        -------
        A float (background value) or a numpy.array of the evaluated scalar moment.
        """

        # collect arguments
        for n, coord in enumerate(coords):
            assert isinstance(coord, xp.ndarray)
            if n == 0:
                shp = coord.shape
            else:
                assert coord.shape == shp, f"Argument {n} has shape {coord.shape}, but must match {shp}."

        params = self.params[name]
        assert isinstance(params, tuple)
        assert len(params) == 2

        # flat evaluation for markers
        if coords[0].ndim == 1:
            etas = [
                xp.concatenate(
                    [coord[:, None] for coord in coords],
                    axis=1,
                ),
            ]
        # assuming that input comes from meshgrid.
        elif coords[0].ndim == 4:
            etas = tuple(coord[:, :, :, 0] for coord in coords)
        elif coords[0].ndim == 5:
            etas = tuple(coord[:, :, :, 0, 0] for coord in coords)
        elif coords[0].ndim == 6:
            etas = tuple(coord[:, :, :, 0, 0, 0] for coord in coords)
        else:
            etas = coords

        # initialize output
        if coords[0].ndim == 1:
            out = 0.0 * coords[0]
        else:
            out = 0.0 * etas[0]

        # evaluate background
        background = params[0]
        if isinstance(background, (float, int)):
            out += background
        else:
            assert callable(background)
            out += background(*etas)

        # add perturbation
        if add_perturbation is None:
            add_perturbation = self.add_perturbation

        perturbation = params[1]
        if perturbation is not None and add_perturbation:
            assert isinstance(perturbation, Perturbation)
            if coords[0].ndim == 1:
                out += perturbation(*coords)
            else:
                out += perturbation(*etas)

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
