"Base classes for kinetic backgrounds."

import copy
from abc import ABCMeta, abstractmethod
from typing import Callable

import cunumpy as xp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import logging

from struphy.fields_background.base import FluidEquilibriumWithB
from struphy.geometry.base import Domain
from struphy.initial.base import Perturbation
from struphy.io.options import LiteralOptions
from struphy.utils.utils import __class_with_params_repr_no_defaults__

logger = logging.getLogger("struphy")

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

    @property
    def axes_transform(self) -> dict:
        """Mapping from logical dimension key to its LaTeX-formatted axis label, used for plot labeling.

        Keys are "e1", "e2", "e3" (always :math:`\\eta_1, \\eta_2, \\eta_3`) and "v1", "v2", "v3", whose labels
        depend on :attr:`velocity_coords`: "cartesian" (:math:`v_x, v_y, v_z`), "vpara_mu"
        (:math:`v_\\parallel, \\mu`), "vpara_vperp" (:math:`v_\\parallel, v_\\perp`), or "vpara_energy"
        (:math:`v_\\parallel, E`). The "v3" entry is None unless velocity_coords is "cartesian".
        """
        dct = {}
        dct["e1"] = "$\\eta_1$"
        dct["e2"] = "$\\eta_2$"
        dct["e3"] = "$\\eta_3$"
        if self.velocity_coords == "cartesian":
            dct["v1"] = "$v_x$"
            dct["v2"] = "$v_y$"
            dct["v3"] = "$v_z$"
        elif self.velocity_coords == "vpara_mu":
            dct["v1"] = "$v_\\parallel$"
            dct["v2"] = "$\\mu$"
            dct["v3"] = None
        elif self.velocity_coords == "vpara_vperp":
            dct["v1"] = "$v_\\parallel$"
            dct["v2"] = "$v_\\perp$"
            dct["v3"] = None
        elif self.velocity_coords == "vpara_energy":
            dct["v1"] = "$v_\\parallel$"
            dct["v2"] = "$E$"
            dct["v3"] = None
        return dct

    def __repr__(self):
        out = f"{self.__class__.__name__}(\n"
        for k, v in self.params.items():
            out += " " * 4
            out += f"{k}={v},\n"
        out += ")"
        return out

    def __repr_no_defaults__(self):
        return __class_with_params_repr_no_defaults__(self)

    def reduced_eval(
        self,
        dim_1: LiteralOptions.KineticDimensionsToPlot = "e1",
        dim_2: LiteralOptions.KineticDimensionsToPlot | None = None,
        v_lim: float | tuple[float] = 5.0,
        resol: int | tuple[int] = 100,
        integrate_resol: tuple[int | float] | None = None,
        max_points: int = 1e8,
        domain: Domain | None = None,
    ):
        """Evaluate a "reduced" version of the background, where all but 1 or 2 dimensions have been integrated out.
        See :ref:`binning`.

        Integration is performed via a simple midpoint rule, with the number of integration points specified by ``integrate_resol``.
        One can set a maximum for the total evaluation points in order to avoid memory issues (default is 1e8, corresponding to ~1 GB for double precision).

        Parameters
        ----------
        dim_1, dim_2 : LiteralOptions.KineticDimensionsToPlot = ["e1","e2","e3","v1","v2","v3"]
            The axis (or axes) along which the reduced distribution is evaluated (i.e. the axes that are not integrated out).
            They refere to logical phase space axes.
            If dim_2 is not defined the reduced distribution is 1D, otherwise it is 2D.

        v_lim : float | tuple[float]
            Limit values for the velocity axes (default: 5.0), given as ``(v_lim_1, v_lim_2)`` for ``dim_1`` and
            ``dim_2`` respectively (a single float is broadcast to both entries). ``v_lim[0]`` also sets the
            integration bounds of any velocity axis that is integrated out (i.e. neither ``dim_1`` nor ``dim_2``);
            ``v_lim[1]`` is only used when ``dim_2`` is itself a velocity axis.
            For a Cartesian velocity coordinate (and v_parallel), the limits are [-v_lim, v_lim]. For a positive
            velocity coordinate (such as mu or v_perp), the limits are [0, v_lim].

        resol : int | tuple[int]
            Resolution of the evaluation grid along the plotted axis (axes). If a single integer is provided,
            it is used for both dim_1 and dim_2.

        integrate_resol : tuple[int | float] | None
            Number of quadrature points for integration along each phase space axis.
            If None, is determined as :math:`max\\_points^{1/N}` where :math:`N` is the number of axes to integrate out.
            If tuple, length must be the dimension of the phase space (3 + vdim), where the plotted axes (dim_1, dim_2)
            must hold the value None. A float value means evaluation at that point rather than intgration.
            Example: integrate_resol=(None, 0.5, 20, None, 15, 15) for a 3D integral in (eta_3, vy, vz),
            evaluated at eta_2=0.5, and dim_1="e1", dim_2="v1" for 2D evaluation.
            High number of quadrature points can lead to memory issues.

        max_points : int = 1e8
            Maximum number of points to evaluate the background on (default is 1e8, corresponding to ~1 GB for double precision).

        domain : Domain | None = None
            Mapping to physical space. If given, dim_1 and dim_2 must both be space axes (["e1","e2","e3"]) and the
            returned ``physical_coords`` holds the corresponding "x", "y", "z" arrays; otherwise ``physical_coords``
            is None.

        Returns
        -------
        reduced_density : xp.ndarray
            The background integrated over all axes except dim_1 (and dim_2, if given); 1D if dim_2 is None, else 2D.

        plot_pts1, plot_pts2 : xp.ndarray | None
            Evaluation points along dim_1 and dim_2. ``plot_pts2`` is None if dim_2 is not given.

        physical_coords : dict | None
            Dictionary with keys "x", "y", "z" holding the domain-mapped position arrays (broadcast to the shape
            of ``reduced_density``), or None if no domain was given.
        """

        if domain is not None:
            assert dim_1 in ["e1", "e2", "e3"] and dim_2 in ["e1", "e2", "e3"], (
                'To perform a plot in physical space you must use two space axes (dim_1, dim_2 in ["e1","e2","e3"]).'
            )

        n_axes_plot = 1 + (dim_2 is not None)
        n_v_to_plot = ("v" in dim_1) + ("v" in dim_2 if dim_2 is not None else 0)

        if isinstance(v_lim, float):
            v_lim = (v_lim, v_lim)
        else:
            assert isinstance(v_lim, tuple)
            if len(v_lim) == 1:
                v_lim = (v_lim[0], v_lim[0])

        if isinstance(resol, int):
            resol = (resol,) * n_axes_plot

        n_axes_integration = 3 + self.vdim - n_axes_plot
        max_quad_points = max_points
        for r in resol:
            max_quad_points //= r

        # phase space grid, first add plotting points for the axes that are plotted
        tabs = [None] * (3 + self.vdim)
        axe_to_plot1, plot_pts1 = self._get_plot_pts(dim_1, v_lim[0], resol[0])
        tabs[axe_to_plot1] = plot_pts1

        if dim_2 is not None:
            axe_to_plot2, plot_pts2 = self._get_plot_pts(dim_2, v_lim[1], resol[1])
            assert axe_to_plot2 != axe_to_plot1, "You must specify different dimensions for dim_1 and dim_2"
            tabs[axe_to_plot2] = plot_pts2
        else:
            axe_to_plot2 = None
            plot_pts2 = None

        # add integration points for the axes that are not plotted
        if integrate_resol is None:
            n_int = int(max_quad_points ** (1 / n_axes_integration))
            integrate_resol = (n_int,) * (3 + self.vdim)
        else:
            assert len(integrate_resol) == 3 + self.vdim, (
                f"integrate_resol must have length {3 + self.vdim} for this background"
            )
            assert integrate_resol[axe_to_plot1] is None, (
                f"integrate_resol must be None for the axis {dim_1} that is plotted"
            )
            if axe_to_plot2 is not None:
                assert integrate_resol[axe_to_plot2] is None, (
                    f"integrate_resol must be None for the axis {dim_2} that is plotted"
                )
            cp = 1
            for r in integrate_resol:
                if r is not None:
                    cp *= r
            assert cp <= max_quad_points, (
                f"Too many quadrature points for integration, reduce integrate_resol or increase max_points (current: {cp} > {max_quad_points})"
            )

        logger.info(f"Reduced evaluation with {integrate_resol = }")
        velocity_space_volume = 1.0
        for i, tab in enumerate(tabs):
            if tab is None:
                if i < 3:
                    if isinstance(integrate_resol[i], float):
                        tabs[i] = xp.array([integrate_resol[i]])
                    else:
                        raise NotImplementedError("Integration over spatial axes is not implemented yet.")
                        tabs[i] = xp.linspace(0.0, 1.0, integrate_resol[i])
                else:
                    if i == 3:  # Cartesian and v_parallel
                        tabs[i] = xp.linspace(-v_lim[0], v_lim[0], integrate_resol[i])
                        velocity_space_volume *= 2 * v_lim[0]
                    else:
                        if self.velocity_coords == "cartesian":  # Cartesian
                            tabs[i] = xp.linspace(-v_lim[0], v_lim[0], integrate_resol[i])
                            velocity_space_volume *= 2 * v_lim[0]
                        else:  # v_perp, mu and energy
                            assert i == 4
                            tabs[i] = xp.linspace(0.0, v_lim[0], integrate_resol[i])
                            velocity_space_volume *= v_lim[0]

        # push to physical position space if needed
        if domain is not None:
            tmp = domain(*tabs[:3])
            physical_coords = {}

            # keep the plotted spatial axes (e1, e2, e3) as full slices, fix the rest at index 0
            idx = [0, 0, 0]
            for axe_to_plot in (axe_to_plot1, axe_to_plot2):
                if axe_to_plot is not None and axe_to_plot < 3:
                    idx[axe_to_plot] = slice(None)
            idx = tuple(idx)

            physical_coords["x"] = tmp[(0,) + idx]
            physical_coords["y"] = tmp[(1,) + idx]
            physical_coords["z"] = tmp[(2,) + idx]
        else:
            physical_coords = None

        # memory intensive evaluation of the background on the phase space grid
        phase_space_mesh = xp.meshgrid(*tabs, indexing="ij")
        total_density = self(*phase_space_mesh)

        axes_to_integrate = [i for i in range(3 + self.vdim)]
        axes_to_integrate.remove(axe_to_plot1)
        if dim_2 is not None:
            axes_to_integrate.remove(axe_to_plot2)

        reduced_density = xp.mean(total_density, tuple(axes_to_integrate))
        reduced_density *= velocity_space_volume

        return reduced_density, plot_pts1, plot_pts2, physical_coords

    def _get_plot_pts(
        self,
        dim: LiteralOptions.KineticDimensionsToPlot,
        v_lim: float,
        resol: int,
    ):
        """Resolve a single dimension key to its phase-space axis index and its array of evaluation points.

        Parameters
        ----------
        dim : LiteralOptions.KineticDimensionsToPlot
            The axis to resolve, one of "e1", "e2", "e3", "v1", "v2", "v3".

        v_lim : float
            Limit value for the axis if it is a velocity axis (ignored for logical space axes, which always
            span [0, 1]). For a Cartesian velocity coordinate (and v_parallel) the range is [-v_lim, v_lim];
            for a positive velocity coordinate (such as mu or v_perp) the range is [0, v_lim].

        resol : int
            Number of evaluation points along the axis.

        Returns
        -------
        axe_to_plot : int
            Index of ``dim`` in the phase space (0, 1, 2 for e1, e2, e3 and 3, 4, 5 for v1, v2, v3).

        plot_pts : xp.ndarray
            Array of ``resol`` evaluation points spanning the axis range.
        """
        if dim == "e1":
            axe_to_plot = 0
        elif dim == "e2":
            axe_to_plot = 1
        elif dim == "e3":
            axe_to_plot = 2
        elif dim == "v1":
            axe_to_plot = 3
        elif dim == "v2":
            axe_to_plot = 4
        elif dim == "v3":
            axe_to_plot = 5
        else:
            AssertionError("dim argument must match an exiting dimension")

        if axe_to_plot - 3 > self.vdim:
            AssertionError("Coordinate " + dim + " does not exist with this background")

        if axe_to_plot == 3:  # Cartesian and v_parallel
            v_left = -v_lim
            v_right = v_lim
        else:
            if self.velocity_coords == "cartesian":  # Cartesian
                v_left = -v_lim
                v_right = v_lim
            else:  # v_perp, mu and energy
                v_left = 0.0
                v_right = v_lim

        if axe_to_plot < 3:
            plot_pts = xp.linspace(0.0, 1.0, resol)
        else:
            plot_pts = xp.linspace(v_left, v_right, resol)

        return axe_to_plot, plot_pts

    def plot(
        self,
        dim_1: LiteralOptions.KineticDimensionsToPlot = "e1",
        dim_2: LiteralOptions.KineticDimensionsToPlot | None = None,
        v_lim: float = 5.0,
        resol: int | tuple[int] = 100,
        integrate_resol: tuple[int | float] | None = None,
        max_points: int = 1e7,
        domain: Domain | None = None,
        proj_axis: tuple[str,] = ("x", "y"),
        plot_3D: bool = False,
    ):
        """
        Plots the density profile (slice) of the phase space background distribution.
        The slice can be 1D or 2D, in logical coordinates. If a domain is given, the 2D slice is also plotted
        in physical (Cartesian) coordinates, and optionally as a 3D surface.

        See :meth:`reduced_eval` for how the profile is computed.

        Parameters
        ----------
        dim_1, dim_2 : LiteralOptions.KineticDimensionsToPlot = ["e1","e2","e3","v1","v2","v3"]
            The axes used in the projection, they refere to logical phase space axes.
            If dim_2 is not defined the projection is 1D, it is 2D if dim_2 is attributed.

        v_lim : float = 5.0
            Limit value of the velocity axes (broadcast to dim_1 and dim_2, see :meth:`reduced_eval`).

        resol : int | tuple[int] = 100
            Resolution of the plot along each plotted axis. If a single integer is provided, it is used
            for both dim_1 and dim_2.

        integrate_resol : tuple[int | float] | None = None
            Number of quadrature points for integration along each phase space axis that is not plotted.
            See :meth:`reduced_eval` for details; if None it is chosen automatically from max_points.
            A float value means evaluation at that point rather than intgration.

        max_points : int = 1e7
            Maximum number of points to evaluate the background on, to avoid memory issues.

        domain : Domain | None = None
            Domain used to map the plot to physical (Cartesian) space, producing an additional 2D plot (and,
            if plot_3D=True, a 3D surface plot) alongside the logical-space plot. If given, dim_1 and dim_2
            must both be space axes (["e1", "e2", "e3"]).

        proj_axis : tuple[str] = ("x", "y")
            The two Cartesian axes ("x", "y", "z") used for the 2D physical-space plot (only used if domain
            is given). If you do not see the density profile in 2D, you may change these axes.

        plot_3D : bool = False
            Also plot the density as a colored surface in 3D physical space. Requires domain to be given.
        """

        if plot_3D:
            assert domain is not None, "To perform a 3D plot you must provide a domain."

        assert all([pa in ["x", "y", "z"] for pa in proj_axis]), (
            f"proj_axis must be a tuple of two axes among ['x', 'y', 'z'], but is {proj_axis}"
        )

        reduced_density, plot_pts1, plot_pts2, physical_coords = self.reduced_eval(
            dim_1=dim_1,
            dim_2=dim_2,
            v_lim=v_lim,
            resol=resol,
            integrate_resol=integrate_resol,
            max_points=max_points,
            domain=domain,
        )

        fig, ax = plt.subplots(1, 1)
        if dim_2 is None:
            ax.plot(plot_pts1, reduced_density)
            ax.set_xlabel(self.axes_transform[dim_1])
            ax.set_ylabel("density")
            ax.set_title("Kinetic background profile")
        else:
            X, Y = xp.meshgrid(plot_pts1, plot_pts2, indexing="ij")
            for_color = ax.pcolor(X, Y, reduced_density)
            ax.set_xlabel(self.axes_transform[dim_1])
            ax.set_ylabel(self.axes_transform[dim_2])
            fig.colorbar(for_color)
            ax.set_title("Kinetic background profile (logical space)")

            if physical_coords is not None:
                fig, ax = plt.subplots(1, 1)
                for_color_phys = ax.pcolor(
                    physical_coords[proj_axis[0]], physical_coords[proj_axis[1]], reduced_density
                )
                ax.set_xlabel(proj_axis[0])
                ax.set_ylabel(proj_axis[1])
                fig.colorbar(for_color_phys)
                ax.set_title("Kinetic background profile (physical space)")

            if plot_3D:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                norm = Normalize(reduced_density.min(), reduced_density.max() + 0.01)
                colors = cm.viridis(norm(reduced_density))
                ax.plot_surface(
                    X=physical_coords["x"],
                    Y=physical_coords["y"],
                    Z=physical_coords["z"],
                    facecolors=colors,
                )
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.set_title("Kinetic background profile (physical space)")

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

        # check for uniform drawing on disc
        if self.params.get("uniform_on_disc", False):
            assert self.params.get("n") == (1.0, None), "Uniform drawing on disc requires n=1.0 without perturbation."

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
    def gaussian(
        self, v, u=0.0, vth=1.0, B0=2.0, type: LiteralOptions.OptsGaussianCoordinate = "cartesian", volume_form=False
    ):
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
            out = 1.0 / vth**2 * xp.exp(-v * B0 / vth**2)
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

        # uniform density on disc (n=2 eta_1)
        if name == "n" and self.params.get("uniform_on_disc", False):
            if coords[0].ndim == 1:
                out *= 2.0 * coords[0]
            else:
                out *= 2.0 * etas[0]

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
