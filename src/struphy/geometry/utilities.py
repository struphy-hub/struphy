# from __future__ import annotations
"Domain-related utility functions."

import numpy as np
# from typing import TYPE_CHECKING
from scipy.optimize import newton, root, root_scalar
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from typing import Callable

from struphy.bsplines import bsplines as bsp
from struphy.geometry.base import PoloidalSplineTorus
from struphy.geometry.utilities_kernels import weighted_arc_lengths_flux_surface
from struphy.linear_algebra.linalg_kron import kron_lusolve_2d

# if TYPE_CHECKING:
from struphy.geometry.base import Domain
from struphy.io.options import GivenInBasis


def field_line_tracing(
    psi,
    psi_axis_R,
    psi_axis_Z,
    psi0,
    psi1,
    Nel,
    p,
    psi_power=1,
    xi_param="equal_angle",
    Nel_pre=(64, 256),
    p_pre=(3, 3),
    r0=0.3,
):
    r"""
    Given a poloidal flux function :math:`\psi(R, Z)`, constructs a flux-aligned spline mapping :math:`(R, Z) = F(s(\psi), \xi)`.

    The radial coordinate :math:`s \in [0, 1]` is parametrized in terms of powers of :math:`\psi`:

    .. math::

        s(\psi) = [ (\psi - \psi_0)/(\psi_1 - \psi_0) ]^p,

    where :math:`\psi_0` is the value of the innermost flux surface of the mapping,
    :math:`\psi_1` the value of the outermost flux surface, and :math:`p \in \mathbb Q` is some power.

    The angular coordinate :math:`\xi \in (0, 2\pi)` can be parametrized in five different ways:

        1. ``equal_angle``
        2. ``equal_arc_length``
        3. ``sfl`` (straight field line angle)
        4. ``equal_area``
        5. ``equal_volume``

    All :math:`\xi`-parametrizations other than ``equal_angle`` involve a two step procedure:

        1. First, a flux-aligned mapping with parameters ``Nel_pre``, ``p_pre`` is constructed with ``xi=equal angle``.
        2. Second, a mapping with lower resolution is constructed with the desired :math:`\xi`-parametrization.

    The field-line tracing algorithm for the ``equal_angle``-parametrization is as follows:
    given a callable mapping :math:`s(R, Z) = [ (\psi(R, Z) - \psi_0)/(\psi_1 - \psi_0) ]^p \in [0, 1]`,
    we want to find the spline mapping

    .. math::

        R(s, \xi) &= \sum_{i=1}^{N_1}\sum_{j=1}^{N_2}c^R_{ij}N_1(s)N_2(\xi)\,,

        Z(s, \xi) &= \sum_{i=1}^{N_1}\sum_{j=1}^{N_2}c^Z_{ij}N_1(s)N_2(\xi).

    This will be achieved by interpolation, which means we need a set of function values :math:`(R_{ij}, Z_{ij})`
    at the interpolation point sets :math:`(s_i, \xi_j)_{i=1,j=1}^{N_1, N_2}`. At first, we need to obtain these function values.
    For this we draw lines for :math:`\xi_j=j\Delta\xi` for :math:`j\in\{0,1,\cdots,N_2-1\}`:

    .. math::

        R_j(r) &= R_\text{axis} + r\cos(\xi_j)\,,

        Z_j(r) &= Z_\text{axis} + r\sin(\xi_j)\,.

    For each :math:`j`, we calculate the intersections with certain :math:`s_i`-values by computing the root of the function

    .. math::

        f(r) = s(R_j(r),Z_j(r)) - s_i\,.

    This yields a :math:`r_{i}` which leads to :math:`R_{ij}=R_j(r_i)` and :math:`Z_{ij}=Z_j(r_i)`.
    Finally, after having found the :math:`(R_{ij}, Z_{ij})`,
    we solve a spline interpolation problem to find the :math:`(c^R_{ij}, c^Z_{ij})`.

    Parameters
    ----------
    psi : callable
        The poloidal flux function psi(R, Z).

    psi_axis_R : float
        R coordinate of the minimum of psi.

    psi_axis_Z : float
        Z coordinate of the minimum of psi.

    psi0 : float
        Value of the innermost flux surface of the mapping.

    psi1 : float
        Value of the outermost flux surface of the mapping.

    Nel : list[int]
        Number of elements to be used for spline inerpolation.

    p : list[int]
        Spline degrees for spline interpolation.

    psi_power : int, optional
        Power of normalized poloidal flux used in s parametrization.

    xi_param : str
        Which angular (xi) parametrization.

    Nel_pre : tuple | list, optional
        Number of elements to be used for the pre-mapping.

    p_pre : tuple | list, optional
        Spline degrees to be used for the pre-mapping.

    r0 : float, optional
        Initial guess for radial distance from (psi_axis_R, psi_axis_Z) used in Newton root-finding method for flux surfaces.

    Returns
    -------
    cR : np.ndarray
        Control points (2d) of flux aligned spline mapping (R-component).

    cZ : np.ndarray
        Control points (2d) of flux aligned spline mapping (Z-component).
    """

    # for equal_angle one mapping is enough
    if xi_param == "equal_angle":
        ns, nx = Nel
        ps, px = p
    else:
        ns, nx = Nel_pre
        ps, px = p_pre

    # spline knots
    Ts = bsp.make_knots(np.linspace(0.0, 1.0, ns + 1), ps, False)
    Tx = bsp.make_knots(np.linspace(0.0, 1.0, nx + 1), px, True)

    # interpolation (Greville) points
    s_gr = bsp.greville(Ts, ps, False)
    x_gr = bsp.greville(Tx, px, True)

    if p[1] % 2 == 1:
        assert x_gr[0] == 0.0

    # collocation matrices
    Is = bsp.collocation_matrix(Ts, ps, s_gr, False)
    Ix = bsp.collocation_matrix(Tx, px, x_gr, True)

    ILUs = [
        splu(csc_matrix(Is)),
        splu(csc_matrix(Ix)),
    ]

    # check if pole is included
    if np.abs(psi(psi_axis_R, psi_axis_Z) - psi0) < 1e-14:
        pole = True
    else:
        pole = False

    R = np.zeros((s_gr.size, x_gr.size), dtype=float)
    Z = np.zeros((s_gr.size, x_gr.size), dtype=float)

    # function whose root must be found
    for j, x in enumerate(x_gr):
        for i, s in enumerate(s_gr):
            if pole and i == 0:
                R[i, j] = psi_axis_R
                Z[i, j] = psi_axis_Z
                continue

            if i < s_gr.size // 2:
                r_guess = 1 * r0
            else:
                r_guess = 1 * r_flux_surface

            # function whose root must be found
            def f(r):
                _R = psi_axis_R + r * np.cos(2 * np.pi * x)
                _Z = psi_axis_Z + r * np.sin(2 * np.pi * x)

                psi_norm = (psi(_R, _Z) - psi0) / (psi1 - psi0)

                if psi_norm < 0.0:
                    return -((-psi_norm) ** psi_power) - s
                else:
                    return psi_norm**psi_power - s

            r_flux_surface = newton(f, x0=r_guess)

            R[i, j] = psi_axis_R + r_flux_surface * np.cos(2 * np.pi * x)
            Z[i, j] = psi_axis_Z + r_flux_surface * np.sin(2 * np.pi * x)

    # get control points
    cR_equal_angle = kron_lusolve_2d(ILUs, R)
    cZ_equal_angle = kron_lusolve_2d(ILUs, Z)

    if pole:
        cR_equal_angle[0, :] = psi_axis_R
        cZ_equal_angle[0, :] = psi_axis_Z

    # for equal angle parametrization stop here and return the control points
    if xi_param == "equal_angle":
        return cR_equal_angle, cZ_equal_angle

    # for all other parametrizations continue
    else:
        print("Calculation of pre-mapping successful! Start angle parametrization " + xi_param + ".")

        # create temporary domain
        domain_eq_angle = PoloidalSplineTorus(Nel=Nel_pre, p=p_pre, cx=cR_equal_angle, cy=cZ_equal_angle)

        # create new interpolation data
        ns, nx = Nel
        ps, px = p

        # spline knots
        Ts = bsp.make_knots(np.linspace(0.0, 1.0, ns + 1), ps, False)
        Tx = bsp.make_knots(np.linspace(0.0, 1.0, nx + 1), px, True)

        # interpolation (Greville) points
        s_gr = bsp.greville(Ts, ps, False)
        x_gr = bsp.greville(Tx, px, True)

        if p[1] % 2 == 1:
            assert x_gr[0] == 0.0

        # collocation matrices
        Is = bsp.collocation_matrix(Ts, ps, s_gr, False)
        Ix = bsp.collocation_matrix(Tx, px, x_gr, True)

        ILUs = [
            splu(csc_matrix(Is)),
            splu(csc_matrix(Ix)),
        ]

        xi_param_dict = {
            "equal_arc_length": 1,
            "sfl": 2,
            "equal_area": 3,
            "equal_volume": 4,
        }

        # target function for xi parametrization
        def f_angles(xis, s_val):
            assert np.all(np.logical_and(xis > 0.0, xis < 1.0))

            # add 0 and 1 to angles array
            xis_extended = np.array([0.0] + list(xis) + [1.0])

            # compute (R, Z) coordinates for given xis on fixed flux surface corresponding to s_val
            _RZ = domain_eq_angle(s_val, xis_extended, 0.0)

            _R = _RZ[0]
            _Z = _RZ[2]

            # |grad(psi)| at xis
            gp = np.sqrt(psi(_R, _Z, dR=1) ** 2 + psi(_R, _Z, dZ=1) ** 2)

            # compute weighted arc_lengths between two successive points in xis_extended array
            dl = np.zeros(xis_extended.size - 1, dtype=float)
            weighted_arc_lengths_flux_surface(_R, _Z, gp, dl, xi_param_dict[xi_param])

            # total length of the flux surface
            l = np.sum(dl)

            # cumulative sum of arc lengths, start with 0!
            l_cum = np.cumsum(dl)

            # odd spline degree
            if px % 2 == 1:
                xi_diff = l_cum[:-1] / l - x_gr[1:]
            # even spline degree
            else:
                xi_diff = l_cum[:-1] / l - x_gr

            return xi_diff

        # loop over flux surfaces and find xi parametrization
        R = np.zeros((s_gr.size, x_gr.size), dtype=float)
        Z = np.zeros((s_gr.size, x_gr.size), dtype=float)

        if px % 2 == 1:
            xis0 = x_gr[1:].copy()
        else:
            xis0 = x_gr.copy()

        # loop over flux surfaces and finds roots of F_single
        for i in range(s_gr.size):
            s_flux = s_gr[i]

            if i == 0 and pole:
                R[i, :] = psi_axis_R
                Z[i, :] = psi_axis_Z
                continue

            # find root of target function and check for convergence
            tracing = root(f_angles, x0=xis0, args=(s_flux,), method="hybr")
            assert tracing["success"]

            # set new initial guess
            xis0 = tracing["x"]

            # add zero angle for odd degree
            if px % 2 == 1:
                R[i, 1:] = domain_eq_angle(s_flux, tracing["x"], 0.0)[0]
                Z[i, 1:] = domain_eq_angle(s_flux, tracing["x"], 0.0)[2]

                R[i, 0] = domain_eq_angle(s_flux, 0.0, 0.0)[0]
                Z[i, 0] = domain_eq_angle(s_flux, 0.0, 0.0)[2]

            else:
                R[i, :] = domain_eq_angle(s_flux, tracing["x"], 0.0)[0]
                Z[i, :] = domain_eq_angle(s_flux, tracing["x"], 0.0)[2]

        # get control points
        cR = kron_lusolve_2d(ILUs, R)
        cZ = kron_lusolve_2d(ILUs, Z)

        if pole:
            cR[0, :] = psi_axis_R
            cZ[0, :] = psi_axis_Z

        return cR, cZ


class TransformedPformComponent:
    r"""
    Construct callable component of p-form on logical domain (unit cube).

    Parameters
    ----------
    fun : Callable | list
        Callable function (components). Has to be length three for vector-valued funnctions,.

    given_in_basis : GivenInBasis
        In which basis fun is represented: either a p-form,
        then '0' or '3' for scalar
        and 'v', '1' or '2' for vector-valued,
        'physical' when defined on the physical (mapped) domain,
        'physical_at_eta' when given the Cartesian components defined on the logical domain,
        and 'norm' when given in the normalized contra-variant basis (:math:`\delta_i / |\delta_i|`).

    out_form : str
        The p-form representation of the output: '0', '1', '2' '3' or 'v'.
        
    comp : int
        Which component of the vector-valued function to return (=0 for scalars).

    domain: struphy.geometry.domains
        All things mapping. If None, the input fun is just evaluated and not transformed at __call__.
    """

    def __init__(self, 
                 fun: Callable | list, 
                 given_in_basis: GivenInBasis, 
                 out_form: str,
                 comp: int = 0,
                 domain: Domain=None,
                 ):
        
        if isinstance(fun, list):
            assert len(fun) == 1 or len(fun) == 3
        else:
            fun = [fun]

        self._fun = []
        for f in fun:
            if f is None:
                def f_zero(x, y, z):
                    return 0 * x
                self._fun += [f_zero]
            else:
                assert callable(f)
                self._fun += [f]

        self._given_in_basis = given_in_basis
        self._out_form = out_form
        self._comp = comp
        self._domain = domain

        self._is_scalar = len(fun) == 1

        # define which component of the field is evaluated (=0 for scalar fields)
        if self._is_scalar:
            self._fun = self._fun[0]
            assert callable(self._fun)
        else:
            assert len(self._fun) == 3
            assert all([callable(f) for f in self._fun])

    def __call__(self, eta1, eta2, eta3):
        """
        Evaluate the component of the transformed p-form specified 'comp'.

        Depending on the dimension of eta1 either point-wise, tensor-product,
        slice plane or general (see :ref:`struphy.geometry.base.prepare_arg`).
        """

        if self._given_in_basis == self._out_form or self._domain is None:
            if self._is_scalar:
                out = self._fun(eta1, eta2, eta3)
            else:
                out = self._fun[self._comp](eta1, eta2, eta3)

        elif self._given_in_basis == "physical":
            if self._is_scalar:
                out = self._domain.pull(
                    self._fun,
                    eta1,
                    eta2,
                    eta3,
                    kind=self._out_form,
                )
            else:
                out = self._domain.pull(
                    self._fun,
                    eta1,
                    eta2,
                    eta3,
                    kind=self._out_form,
                )[self._comp]

        elif self._given_in_basis == "physical_at_eta":
            if self._is_scalar:
                out = self._domain.pull(
                    self._fun,
                    eta1,
                    eta2,
                    eta3,
                    kind=self._out_form,
                    coordinates="logical",
                )
            else:
                out = self._domain.pull(
                    self._fun,
                    eta1,
                    eta2,
                    eta3,
                    kind=self._out_form,
                    coordinates="logical",
                )[self._comp]

        else:
            dict_tran = self._given_in_basis + "_to_" + self._out_form

            if self._is_scalar:
                out = self._domain.transform(
                    self._fun,
                    eta1,
                    eta2,
                    eta3,
                    kind=dict_tran,
                )
            else:
                out = self._domain.transform(
                    self._fun,
                    eta1,
                    eta2,
                    eta3,
                    kind=dict_tran,
                )[self._comp]

        return out
