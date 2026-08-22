"""Mapped domains (single patch).

Module providing mapping classes for single-patch geometries used by Struphy.

This module includes classes such as `Tokamak`, `GVECunit`, `DESCunit`,
`IGAPolarCylinder`, `IGAPolarTorus`, and `Cuboid`. Mappings transform
reference coordinates to Cartesian coordinates and integrate with spline-based
grid constructions and field-line tracing.
"""

import copy

import cunumpy as xp

from struphy.fields_background.base import AxisymmMHDequilibrium
from struphy.fields_background.equils import EQDSKequilibrium
from struphy.geometry.base import (
    Domain,
    PoloidalSplineStraight,
    PoloidalSplineTorus,
    Spline,
    interp_mapping,
)
from struphy.geometry.utilities import field_line_tracing


class Tokamak(PoloidalSplineTorus):
    r"""Mappings for Tokamak MHD equilibria constructed via :ref:`field-line tracing <field_tracing>` of a poloidal flux function :math:`\psi`.

    .. image:: ../../pics/mappings/tokamak.png

    Parameters
    ----------
    equilibrium : struphy.fields_background.base.AxisymmMHDequilibrium
        The axisymmetric MHD equilibrium for which a flux-aligned grid shall be constructed (default: AdhocTorus).
    num_elements : tuple[int]
        Number of cells in (radial, angular) direction to be used in spline mapping (default: [8, 32]).
    degree : tuple[int]
        Spline degrees in (radial, angular) direction to be used in spline mapping (default: [2, 3]).
    psi_power : float
        Parametrization of radial flux coordinate :math:`\eta_1=\psi_{\mathrm{norm}}^p`, where :math:`\psi_{\mathrm{norm}}` is the normalized poloidal flux (default: 0.75).
    psi_shifts : tuple[float]
        Start and end shifts of polidal flux in % --> cuts away regions at the axis and edge (default: [2., 2.])
    r_min : float
        Inner radius of poloidal section (optional, default: 0.0). If >0.0, then r_0 = r_min.
    xi_param : str
        Parametrization of angular coordinate ("equal_angle", "equal_arc_length" or "sfl" (straight field line), default: "equal_angle").
    r0 : float
        Initial guess for radial distance from axis used in Newton root-finding method (default: 0.3).
    num_elements_pre : tuple[int]
        Number of cells in (radial, angular) direction of pre-mapping needed for equal_arc_length and sfl parametrizations (default: [64, 256]).
    p_pre : tuple[int]
        Spline degrees in (radial, angular) direction of pre-mapping needed for equal_arc_length and sfl parametrizations (default: [3, 3]).
    tor_period : int
        Toroidal periodicity built into the mapping: :math:`\phi=2\pi\,\eta_3/\mathrm{torperiod}` (default: 1 --> full torus).

    Note
    ----
    Regarding r_min and psi_shifts:
        If r_min is left at 0.0, psi_shifts defines both the inner and outer boundaries of the computational
        domain in terms of the normalized flux coordinate \psi.
        When r_min > 0.0, however, psi_shifts[0] is no longer used. Instead, the code computes the flux value
        corresponding to the physical radius r_min (measured from the magnetic axis),
        which then defines the inner boundary of the domain.
        This allows the user to specify the inner boundary using a more intuitive physical radius rather
        than a flux coordinate. The outer boundary is still controlled by psi_shifts[1].

    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : Tokamak
            Tokamak :
                num_elements        : [8, 32]     # number of poloidal grid cells for spline mapping, >degree
                degree          : [3, 3]      # poloidal spline degrees for spline mapping, >1
                psi_power  : 0.7         # parametrization of radial flux coordinate eta1=psi_norm^psi_power, where psi_norm is normalized flux
                psi_shifts : [2., 2.]    # start and end shifts of polidal flux in % --> cuts away regions at the axis and edge
                r_min : 0.0              # Inner radius of poloidal section. If >0.0, then r_0 = r_min.
                xi_param   : equal_angle # parametrization of angular coordinate (equal_angle, equal_arc_length or sfl (straight field line))
                r0         : 0.3         # initial guess for radial distance from axis used in Newton root-finding method for flux surfaces
                num_elements_pre    : [64, 256]   # number of poloidal grid cells of pre-mapping needed for equal_arc_length and sfl
                p_pre      : [3, 3]      # poloidal spline degrees of pre-mapping needed for equal_arc_length and sfl
                tor_period : 1           # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
    """

    def __init__(
        self,
        equilibrium: AxisymmMHDequilibrium = None,
        num_elements: tuple = (8, 32),
        degree: tuple = (2, 3),
        psi_power: float = 0.75,
        psi_shifts: tuple = (0.01, 2.0),
        r_min: float = 0.0,
        xi_param: str = "equal_angle",
        r0: float = 0.3,
        num_elements_pre: tuple = (64, 256),
        p_pre: tuple = (3, 3),
        tor_period: int = 1,
    ):
        if r_min != 0.0:
            r0 = r_min
        if equilibrium is None:
            equilibrium = EQDSKequilibrium()
        else:
            assert isinstance(equilibrium, AxisymmMHDequilibrium)

        # use the params setter
        self.params = copy.deepcopy(locals())

        # get control points via field tracing between fluxes [psi_s, psi_e]
        psi0, psi1 = equilibrium.psi_range[0], equilibrium.psi_range[1]

        assert r_min >= 0.0, f"Inner radius must be non-negative, got {r_min = }."

        if r_min == 0.0:
            # Default behaviour: keep exactly the historical psi_shifts logic.
            psi_s = psi0 + psi_shifts[0] * 0.01 * (psi1 - psi0)
        else:
            # Annular domain: eta1=0 is the flux surface crossing the outboard
            # midplane at distance r_min from the magnetic axis.
            psi_s = equilibrium.psi(
                equilibrium.psi_axis_RZ[0] + r_min,
                equilibrium.psi_axis_RZ[1],
            )

        psi_e = psi1 - psi_shifts[1] * 0.01 * (psi1 - psi0)

        assert (psi_s - psi0) * (psi_s - psi1) <= 0.0, (
            f"Inner radius gives a flux outside equilibrium.psi_range: "
            f"{r_min = }, {psi_s = }, {equilibrium.psi_range = }."
        )

        assert (psi_e - psi_s) * (psi1 - psi0) > 0.0, (
            f"Invalid radial interval: {psi_s = }, {psi_e = }, {equilibrium.psi_range = }."
        )

        cx, cy = field_line_tracing(
            equilibrium.psi,
            equilibrium.psi_axis_RZ[0],
            equilibrium.psi_axis_RZ[1],
            psi_s,
            psi_e,
            num_elements,
            degree,
            psi_power=psi_power,
            xi_param=xi_param,
            num_elements_pre=num_elements_pre,
            p_pre=p_pre,
            r0=r0,
        )

        # init base class
        super().__init__(
            num_elements=num_elements,
            degree=degree,
            spl_kind=(False, True),
            cx=cx,
            cy=cy,
            tor_period=tor_period,
        )


class GVECunit(Spline):
    """The mapping from `pygvec <https://gvec.readthedocs.io/latest/index.html>`_, computed by the GVEC MHD equilibrium code.

    .. image:: ../../pics/mappings/gvec.png

    Parameters
    ----------
    gvec_equil : struphy.fields_background.equils.GVECequilibrium
        GVEC MHD equilibrium object.

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : GVECunit
    """

    def __init__(self, gvec_equil=None):
        import gvec

        from struphy.fields_background.equils import GVECequilibrium

        if gvec_equil is None:
            gvec_equil = GVECequilibrium()
        else:
            assert isinstance(gvec_equil, GVECequilibrium)

        # do not set params here because of a pickling error

        num_elements = gvec_equil.params["num_elements"]
        degree = gvec_equil.params["degree"]
        if gvec_equil.params["use_nfp"]:
            spl_kind = (False, True, False)
        else:
            spl_kind = (False, True, True)

        # project mapping to splines
        _rmin = gvec_equil.params["rmin"]

        def XYZ(e1, e2, e3):
            rho = _rmin + e1 * (1.0 - _rmin)
            theta = 2 * xp.pi * e2
            zeta = 2 * xp.pi * e3 / gvec_equil._nfp
            if gvec_equil.params["use_boozer"]:
                ev = gvec.EvaluationsBoozer(rho=rho, theta_B=theta, zeta_B=zeta, state=gvec_equil.state)
            else:
                ev = gvec.Evaluations(rho=rho, theta=theta, zeta=zeta, state=gvec_equil.state)
            gvec_equil.state.compute(ev, "pos")
            x = ev.pos.data[0]
            y = ev.pos.data[1]
            z = ev.pos.data[2]
            return x, y, z

        def X(e1, e2, e3):
            return XYZ(e1, e2, e3)[0]

        def Y(e1, e2, e3):
            return XYZ(e1, e2, e3)[1]

        def Z(e1, e2, e3):
            return XYZ(e1, e2, e3)[2]

        cx, cy, cz = interp_mapping(num_elements, degree, spl_kind, X, Y, Z)

        super().__init__(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)


class DESCunit(Spline):
    r"""The mapping :math:`(\rho, \theta,\zeta) \mapsto (X, Y, Z)` to Cartesian coordinates computed by the `DESC MHD equilibrium code
    <https://desc-docs.readthedocs.io/en/latest/theory_general.html#flux-coordinates>`_.

    .. image:: ../../pics/mappings/desc.png

    Parameters
    ----------
    desc_equil : struphy.fields_background.equils.DESCequilibrium
        DESC MHD equilibrium object.

    Note
    ----
    In the parameter .yml file, use the following::

        geometry :
            type : DESCunit
    """

    def __init__(self, desc_equil=None):
        from struphy.fields_background.equils import DESCequilibrium

        if desc_equil is None:
            desc_equil = DESCequilibrium()
        else:
            assert isinstance(desc_equil, DESCequilibrium)

        num_elements = desc_equil.params["num_elements"]
        degree = desc_equil.params["degree"]

        if desc_equil.eq.NFP > 1 and desc_equil.use_nfp:
            spl_kind = (False, True, False)
        else:
            spl_kind = (False, True, True)

        _rmin = desc_equil.params["rmin"]

        nfp = desc_equil.eq.NFP
        if not desc_equil.use_nfp:
            nfp = 1

        # project mapping to splines
        def X(e1, e2, e3):
            return desc_equil.desc_eval("X", e1, e2, e3, nfp=nfp)

        def Y(e1, e2, e3):
            return desc_equil.desc_eval("Y", e1, e2, e3, nfp=nfp)

        def Z(e1, e2, e3):
            return desc_equil.desc_eval("Z", e1, e2, e3, nfp=nfp)

        cx, cy, cz = interp_mapping(num_elements, degree, spl_kind, X, Y, Z)

        super().__init__(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)


class IGAPolarCylinder(PoloidalSplineStraight):
    r"""A cylinder with the cross section approximated by a spline mapping.

    .. math:: 

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &\sum_{ij} c^x_{ij} N_i(\eta_1) N_j(\eta_2)\approx a\,\eta_1\cos(2\pi\eta_2)\,\,\\
        \,\,y= &\sum_{ij} c^y_{ij} N_i(\eta_1) N_j(\eta_2)\approx a\,\eta_1\sin(2\pi\eta_2)\,\,\\
        \,\,z= &L_z\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/iga_cylinder.png

    Parameters
    ----------
    num_elements : list[int]
        Number of cells in (radial, angular) direction used for spline mapping (default: [8, 24]).
    degree : list[int]
        Splines degrees in (radial, angular) direction used for spline mapping (default: [2, 3]).   
    a : float
        Radius of cylinder (default: 1.).
    Lz : float
        Length of cylinder (default: 4.).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : IGAPolarCylinder
            IGAPolarCylinder :
                num_elements : [8, 24] # number of poloidal grid cells, >degree
                degree   : [3, 3] # poloidal spline degree, >1
                Lz  : 6. # Length in third direction
                a   : 1. # minor radius
    """

    def __init__(
        self,
        num_elements: tuple[int] = (8, 24),
        degree: tuple[int] = (2, 3),
        a: float = 1.0,
        Lz: float = 4.0,
    ):
        # use params setter
        self.params = copy.deepcopy(locals())

        # get control points
        def X(eta1, eta2):
            return a * eta1 * xp.cos(2 * xp.pi * eta2)

        def Y(eta1, eta2):
            return a * eta1 * xp.sin(2 * xp.pi * eta2)

        spl_kind = (False, True)

        cx, cy = interp_mapping(num_elements, degree, spl_kind, X, Y)

        # make sure that control points at pole are all the same (eta1=0 there)
        cx[0] = 0.0
        cy[0] = 0.0

        # init base class
        super().__init__(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, Lz=Lz)


class IGAPolarTorus(PoloidalSplineTorus):
    r""" A torus with the poloidal cross-section approximated by a spline mapping.

    .. math::

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &\sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \cos(\phantom{-}2\pi\eta_3) \approx \left[a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0\right]\cos(\phantom{-}2\pi\eta_3)\,\,\\
        \,\,y= &\sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \sin(-2\pi\eta_3)\approx \left[a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0\right]\sin(-2\pi\eta_3)\,\,\\
        \,\,z= &\sum_{ij} c^{Z}_{ij} N_i(\eta_1) N_j(\eta_2)\approx a\,\eta_1\sin(2\pi\theta(\eta_1, \eta_2))\,\,\end{bmatrix}

    The angular parametrization :math:`\theta(\eta_1, \eta_2)` can either be equal angle or straight field line (see parameters below).

    .. image:: ../../pics/mappings/iga_torus.png

    Parameters
    ----------
    num_elements : tuple[int]
        Number of cells in (radial, angular) direction used for spline mapping (default: [8, 24]).
    degree : tuple[int]
        Splines degrees in (radial, angular) direction used for spline mapping (default: [2, 3]).   
    a : float
        Minor radius of torus (default: 1.).
    R0 : float
        Major radius of torus (default: 3.).
    tor_period : int
        Toroidal periodicity built into the mapping: :math:`\phi=2\pi\,\eta_3/\mathrm{torperiod}` (default: 3 --> one third of a torus).
    sfl : bool
        Whether to use straight field line coordinates (default: False).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : IGAPolarTorus
            IGAPolarTorus :
                num_elements        : [8, 24] # number of poloidal grid cells, >degree
                degree          : [3, 3] # poloidal spline degree, >1
                a          : 1. # minor radius
                R0         : 3. # major radius
                tor_period : 2 # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
                sfl        : False # whether to use straight field line coordinates (particular theta parametrization) 
    """

    def __init__(
        self,
        num_elements: tuple[int] = (8, 24),
        degree: tuple[int] = (2, 3),
        a: float = 1.0,
        R0: float = 3.0,
        sfl: bool = False,
        tor_period: int = 3,
    ):
        # use params setter
        self.params = copy.deepcopy(locals())

        # get control points
        if sfl:

            def theta(eta1, eta2):
                return 2 * xp.arctan(xp.sqrt((1 + a * eta1 / R0) / (1 - a * eta1 / R0)) * xp.tan(xp.pi * eta2))
        else:

            def theta(eta1, eta2):
                return 2 * xp.pi * eta2

        def R(eta1, eta2):
            return a * eta1 * xp.cos(theta(eta1, eta2)) + R0

        def Z(eta1, eta2):
            return a * eta1 * xp.sin(theta(eta1, eta2))

        spl_kind = (False, True)

        cx, cy = interp_mapping(num_elements, degree, spl_kind, R, Z)

        # make sure that control points at pole are all the same (eta1=0 there)
        cx[0] = R0
        cy[0] = 0.0

        # init base class
        super().__init__(
            num_elements=num_elements,
            degree=degree,
            spl_kind=spl_kind,
            cx=cx,
            cy=cy,
            tor_period=tor_period,
        )


class Cuboid(Domain):
    r""" Slab geometry (Cartesian coordinates).

    .. math::

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &l_1 + (r_1 - l_1)\,\eta_1\,\,\\
        \,\,y= &l_2 + (r_2 - l_2)\,\eta_2\,\,\\
        \,\,z= &l_3 + (r_3 - l_3)\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/cuboid.png

    Parameters
    ----------
    l1 : float
        Start of x-interval (default: 0.).
    r1 : float
        End of x-interval, r1>l1 (default: 1.).
    l2 : float
        Start of y-interval (default: 0.).
    r2 : float
        End of y-interval, r2>l2 (default: 1.).
    l3 : float
        Start of z-interval (default: 0.).
    r3 : float
        End of z-interval, r3>l3 (default: 1.).
    """

    def __init__(
        self,
        l1: float = 0.0,
        r1: float = 1.0,
        l2: float = 0.0,
        r2: float = 1.0,
        l3: float = 0.0,
        r3: float = 1.0,
    ):
        self.kind_map = 10

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = False

        super().__init__()


class Orthogonal(Domain):
    r""" Slab geometry with orthogonal mesh distortion.

    .. math:: 

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &L_x\,\left[\,\eta_1 + \alpha\sin(2\pi\,\eta_1)\right]\,\,\\
        \,\,y= &L_y\,\left[\,\eta_2 + \alpha\sin(2\pi\,\eta_2)\right]\,\,\\
        \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/orthogonal.png

    Parameters
    ----------
    Lx : float
        Length of x-interval (default: 2.).
    Ly : float
        Length of y-interval (default: 3.).
    alpha: float
        Distortion factor (default: 0.1).
    Lz : float
        Length of z-interval (default: 6.).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : Orthogonal
            Orthogonal :
                Lx    : 2. # length in x-direction
                Ly    : 2. # length in y-direction
                alpha : .1 # x-distortion and y-distortion
                Lz    : 1. # length in z-direction
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 3.0,
        alpha: float = 0.1,
        Lz: float = 6.0,
    ):
        self.kind_map = 11

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = False

        super().__init__()


class Colella(Domain):
    r""" Slab geometry with Colella mesh distortion.

    .. math::

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &L_x\,\left[\,\eta_1 + \alpha\sin(2\pi\,\eta_1)\sin(2\pi\,\eta_2)\,\right]\,\,\\
        \,\,y= &L_y\,\left[\,\eta_2 + \alpha\sin(2\pi\,\eta_2)\sin(2\pi\,\eta_1)\,\right]\,\,\\
        \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/colella.png

    Parameters
    ----------
    Lx : float
        Length of x-interval (default: 2.).
    Ly : float
        Length of y-interval (default: 3.).
    alpha: float
        Distortion factor (default: 0.1).
    Lz : float
        Length of z-interval (default: 6.).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : Colella
            Colella :
                Lx    : 2. # length in x-direction
                Ly    : 2. # length in y-direction
                alpha : .1 # distortion factor
                Lz    : 1. # length in third direction
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 3.0,
        alpha: float = 0.1,
        Lz: float = 6.0,
    ):
        self.kind_map = 12

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = False

        super().__init__()


class MagnetotailSlab(Domain):
    r"""Elongated slab with a pinched center.

    This mapping is intended as a simple geometry for studying reconnection in
    the Earth's night-side magnetotail. The domain is a long slab in the
    tail-aligned direction with a smooth reduction of the cross-tail width near
    the center plane.

    Parameters
    ----------
    Lx : float
        Length in the tail-aligned direction (default: 12.0).
    Ly : float
        Full width of the slab in the cross-tail direction away from the pinch (default: 4.0).
    Lz : float
        Thickness in the normal direction (default: 2.0).
    pinch : float
        Relative pinch strength in [0, 1). Larger values give a narrower center (default: 0.6).
    pinch_width : float
        Characteristic width of the pinched region in physical x-units (default: 2.0).
    x_center : float
        Center position of the pinch in physical x-coordinate (default: 0.0).
    """

    def __init__(
        self,
        Lx: float = 12.0,
        Ly: float = 4.0,
        Lz: float = 2.0,
        pinch: float = 0.6,
        pinch_width: float = 2.0,
        x_center: float = 0.0,
    ):
        self.kind_map = 13

        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        assert Lx > 0.0, f"Need positive slab length, got {Lx =}"
        assert Ly > 0.0, f"Need positive slab width, got {Ly =}"
        assert Lz > 0.0, f"Need positive slab thickness, got {Lz =}"
        assert 0.0 <= pinch < 1.0, f"Pinch strength must satisfy 0 <= pinch < 1, got {pinch =}"
        assert pinch_width > 0.0, f"Need positive pinch width, got {pinch_width =}"

        self.periodic_eta3 = False
        self.pole = False

        super().__init__()


class HollowCylinder(Domain):
    r""" Cylinder with possible hole around the axis.

    .. math::

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2 / poc)\,\,\\
        \,\,y= &\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin(2\pi\,\eta_2 / poc)\,\,\\
        \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/hollow_cylinder.png

    Parameters
    ----------
    a1 : float
        Inner radius of cylinder (default: 0.2).
    a2 : float
        Outer radius of cylinder (default: 1.0).
    Lz: float
        Length of cylinder (default: 4.)
    poc: int
        Which periodicity used in the mapping, i.e. :math: `\theta = 2*\pi*\eta_2 / \mathrm{poc}` (piece of cake) (default: 1).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : HollowCylinder
            HollowCylinder :
                a1 : .2 # inner radius
                a2 : 1. # outer radius
                Lz : 4. # length of cylinder
                poc: 2. # periodicity of theta used in the mapping
    """

    def __init__(
        self,
        a1: float = 0.2,
        a2: float = 1.0,
        Lz: float = 4.0,
        poc: int = 1,
    ):
        self.kind_map = 20

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False

        if a1 == 0.0:
            self.pole = True
        else:
            self.pole = False

        super().__init__()


class PoweredEllipticCylinder(Domain):
    r""" Cylinder with elliptic cross section and radial power law.

    .. math::

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &r_x\,\eta_1^s\cos(2\pi\,\eta_2)\,\,\\
        \,\,y= &r_y\,\eta_1^s\sin(2\pi\,\eta_2)\,\,\\
        \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/pow_elliptic_cyl.png

    Parameters
    ----------
    rx : float
        Radius in x-direction (default: 1.0).
    ry : float
        Radius in y-direction (default: 2.0).
    Lz: float
        Length in z-direction (default: 6.0).
    s : float
        Power of radial coordinate (default: 0.5).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : PoweredEllipticCylinder
            PoweredEllipticCylinder :
                rx : 1. # axis length in x-direction
                ry : 2. # axis length in y-direction
                Lz : 4. # length in z-direction
                s  : .5 # power of radial coordinate
    """

    def __init__(
        self,
        rx: float = 1.0,
        ry: float = 2.0,
        Lz: float = 6.0,
        s: float = 0.5,
    ):
        self.kind_map = 21

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = True

        super().__init__()


class HollowTorus(Domain):
    r""" Torus with possible hole around the magnetic axis (center of the smaller circle).

    .. math::

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &\lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos\left[\theta(\eta_1,\eta_2)\right]+R_0\rbrace\cos(\phantom{-}2\pi\,\eta_3 / n)\,\,\\
        \,\,y= &\lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos\left[\theta(\eta_1,\eta_2)\right]+R_0\rbrace\sin(-2\pi\,\eta_3 / n)\,\,\\
        \,\,z= &\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin\left[\theta(\eta_1,\eta_2)\right]\,\,\end{bmatrix}

    with the following possible poloidal angle parametrizations:

    .. math::

        &\theta(\eta_1,\eta_2) = \left\{\begin{aligned} 

        & 2\pi\,\eta_2\,, \quad &&\textnormal{if}\quad \textnormal{sfl}=\textnormal{False}\,,

        &2\arctan\left[\sqrt{\frac{1 + \epsilon(\eta_1)}{1 - \epsilon(\eta_1)}}\,\tan\left(\pi\,\eta_2\right)\right]\quad &&\textnormal{if}\quad \textnormal{sfl}=\textnormal{True}\,,

        &\qquad \textrm {with}\qquad \epsilon(\eta_1) = \frac{a_1 + (a_2-a_1)\,\eta_1}{R_0}\,.
        \end{aligned}\right.

    .. image:: ../../pics/mappings/hollow_torus.png

    Parameters
    ----------
    a1 : float
        Inner minor radius of hollow torus (default: 0.2).
    a2 : float
        Outer minor radius of hollow torus (default: 1.0).
    R0 : float
        Major radius of torus (default: 3.0).
    sfl : bool
        Whether to use straight field line coordinates (True) or not (False) (default: False).
    pol_period: int
        Which periodicity used in the mapping, i.e. :math: `\theta = 2*\pi*\eta_2 / \mathrm{pol_period}` (piece of cake) (default: 1, only for sfl=False).
    tor_period : int
        Toroidal periodicity built into the mapping: :math:`\phi=2\pi\,\eta_3/\mathrm{torperiod}` (default: 3 --> one third of a torus).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : HollowTorus
            HollowTorus :
                a1  : 0.2   # inner radius
                a2  : 1.0   # minor radius
                R0  : 3.0   # major radius
                sfl : False # straight field line coordinates?
                pol_period: 2. # periodicity of theta used in the mapping: theta = 2*pi * eta2 / pol_period (if not sfl)
                tor_period : 2 # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
    """

    def __init__(
        self,
        a1: float = 0.1,
        a2: float = 1.0,
        R0: float = 3.0,
        sfl: bool = False,
        pol_period: int = 1,
        tor_period: int = 3,
    ):
        self.kind_map = 22

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        assert a2 <= R0, f"The minor radius must be smaller or equal than the major radius! {a2 =}, {R0 =}"

        if sfl:
            assert pol_period == 1, (
                "Piece-of-cake is only implemented for torus coordinates, not for straight field line coordinates!"
            )

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = True

        if a1 == 0.0:
            self.pole = True
        else:
            self.pole = False

        super().__init__()

    def inverse_map(self, x, y, z, bounded=True, change_out_order=False):
        """Analytical inverse map of HollowTorus"""

        mr = xp.sqrt(x**2 + y**2) - self.params["R0"]

        eta3 = xp.arctan2(-y, x) % (2 * xp.pi / self.params["tor_period"]) / (2 * xp.pi) * self.params["tor_period"]
        eta2 = xp.arctan2(z, mr) % (2 * xp.pi / self.params["pol_period"]) / (2 * xp.pi / self.params["pol_period"])
        eta1 = (z / xp.sin(2 * xp.pi * eta2 / self.params["pol_period"]) - self.params["a1"]) / (
            self.params["a2"] - self.params["a1"]
        )

        if bounded:
            eta1[eta1 > 1] = 1.0
            eta1[eta1 < 0] = 0.0
            assert xp.all(xp.logical_and(eta1 >= 0, eta1 <= 1))

        assert xp.all(xp.logical_and(eta2 >= 0, eta2 <= 1))
        assert xp.all(xp.logical_and(eta3 >= 0, eta3 <= 1))

        if change_out_order:
            return xp.transpose((eta1, eta2, eta3))

        else:
            return eta1, eta2, eta3


class GeospaceFluxDomain(Domain):
    r"""Global magnetospheric flux-surface inspired mapping.

    This domain maps the logical unit cube onto an Earth-centered, solar-wind
    distorted geometry with compressed dayside and elongated nightside tail.
    It provides a single smooth coordinate mapping useful for idealized studies
    of bow-shock/magnetosheath/magnetotail coupling and tail reconnection setup.

    Coordinates
    -----------
    * :math:`\eta_1`: radial/normal-like coordinate from ionosphere to bow shock
    * :math:`\eta_2`: dayside-to-nightside poloidal angle coordinate
    * :math:`\eta_3`: clock angle around Earth-Sun axis

    Parameters
    ----------
    r_iono : float
        Inner reference radius (ionosphere-like boundary).
    r_mp_dayside : float
        Magnetopause radius on dayside.
    r_mp_tail : float
        Magnetopause radius in the nightside tail.
    r_bs_dayside : float
        Bow-shock radius on dayside.
    r_bs_tail : float
        Bow-shock radius in the nightside tail.
    mp_eta1 : float
        Logical location of magnetopause in :math:`\eta_1` (0 < mp_eta1 < 1).
    sheet_flatten : float
        Tail current-sheet flattening strength in [0, 1).
    """

    def __init__(
        self,
        r_iono: float = 1.0,
        r_mp_dayside: float = 8.0,
        r_mp_tail: float = 30.0,
        r_bs_dayside: float = 12.0,
        r_bs_tail: float = 45.0,
        mp_eta1: float = 0.72,
        sheet_flatten: float = 0.45,
    ):
        self.kind_map = 24

        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        assert r_iono > 0.0, f"Need positive ionosphere radius, got {r_iono =}"
        assert r_mp_dayside > r_iono, f"Need r_mp_dayside > r_iono, got {r_mp_dayside =}, {r_iono =}"
        assert r_mp_tail > r_iono, f"Need r_mp_tail > r_iono, got {r_mp_tail =}, {r_iono =}"
        assert r_bs_dayside > r_mp_dayside, f"Need r_bs_dayside > r_mp_dayside, got {r_bs_dayside =}, {r_mp_dayside =}"
        assert r_bs_tail > r_mp_tail, f"Need r_bs_tail > r_mp_tail, got {r_bs_tail =}, {r_mp_tail =}"
        assert 0.0 < mp_eta1 < 1.0, f"Need 0 < mp_eta1 < 1, got {mp_eta1 =}"
        assert 0.0 <= sheet_flatten < 1.0, f"Need 0 <= sheet_flatten < 1, got {sheet_flatten =}"

        self.periodic_eta3 = True
        self.pole = False

        super().__init__()


class WarpedAccretionDisk(Domain):
    r"""Warped accretion disk mapping.

    This mapping describes a cylindrical disk with finite thickness and a smooth,
    radius-dependent vertical warp of the disk midplane.

    Coordinates
    -----------
    * :math:`\eta_1`: radial coordinate from inner to outer disk radius
    * :math:`\eta_2`: azimuthal coordinate around the central object
    * :math:`\eta_3`: vertical coordinate across disk thickness

    Parameters
    ----------
    r_in : float
        Inner disk radius.
    r_out : float
        Outer disk radius.
    thickness : float
        Half-thickness scaling in the vertical direction.
    warp_amp : float
        Warp amplitude factor.
    warp_power : float
        Power-law exponent for radial growth of the warp.
    node_angle : float
        Azimuthal node angle of the warp in radians.
    tor_period : int
        Azimuthal periodicity: :math:`\phi = 2\pi\eta_2/\mathrm{tor\_period}`.
    """

    def __init__(
        self,
        r_in: float = 2.0,
        r_out: float = 12.0,
        thickness: float = 0.3,
        warp_amp: float = 0.15,
        warp_power: float = 1.5,
        node_angle: float = 0.0,
        tor_period: int = 1,
    ):
        self.kind_map = 25

        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        assert r_in > 0.0, f"Need positive inner radius, got {r_in =}"
        assert r_out > r_in, f"Need r_out > r_in, got {r_out =}, {r_in =}"
        assert thickness > 0.0, f"Need positive thickness, got {thickness =}"
        assert warp_amp >= 0.0, f"Need non-negative warp amplitude, got {warp_amp =}"
        assert warp_power >= 0.0, f"Need non-negative warp power, got {warp_power =}"
        assert tor_period > 0, f"Need positive toroidal periodicity, got {tor_period =}"

        self.periodic_eta3 = True
        self.pole = False

        super().__init__()


class Spheromak(Domain):
    r"""Compact toroidal plasma proxy in a spherical-like shell.

    This mapping provides a simple geometry for spheromak studies using nested
    closed flux-surface-like shells with optional vertical elongation.

    Coordinates
    -----------
    * :math:`\eta_1`: radial shell coordinate
    * :math:`\eta_2`: poloidal angle coordinate
    * :math:`\eta_3`: toroidal/azimuthal angle coordinate

    Parameters
    ----------
    r0 : float
        Inner radius.
    a : float
        Plasma minor-size scale (outer radius is r0 + a).
    kappa : float
        Vertical elongation factor.
    tor_period : int
        Azimuthal periodicity: :math:`\phi=2\pi\eta_3/\mathrm{tor\_period}`.
    """

    def __init__(
        self,
        r0: float = 0.0,
        a: float = 1.0,
        kappa: float = 1.0,
        tor_period: int = 1,
    ):
        self.kind_map = 26

        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        assert r0 >= 0.0, f"Need non-negative inner radius, got {r0 =}"
        assert a > 0.0, f"Need positive minor size a, got {a =}"
        assert kappa > 0.0, f"Need positive elongation kappa, got {kappa =}"
        assert tor_period > 0, f"Need positive toroidal periodicity, got {tor_period =}"

        self.periodic_eta3 = True
        self.pole = r0 == 0.0

        super().__init__()


class HallEffectThrusterChannel(Domain):
    r"""Coaxial annular Hall thruster channel with axial end packing.

    The mapping models an annular channel where :math:`\eta_3` is the axial
    coordinate (anode to exit), with increased point packing near both ends
    controlled by a smooth high-frequency modulation.

    Parameters
    ----------
    r_in : float
        Inner channel radius.
    r_out : float
        Outer channel radius.
    length : float
        Axial channel length.
    pack_strength : float
        End-packing strength in [0, 1). Higher values cluster more points at
        the anode and exit.
    tor_period : int
        Azimuthal periodicity: :math:`\phi=2\pi\eta_2/\mathrm{tor\_period}`.
    """

    def __init__(
        self,
        r_in: float = 1.0,
        r_out: float = 2.0,
        length: float = 5.0,
        pack_strength: float = 0.7,
        tor_period: int = 1,
    ):
        self.kind_map = 27

        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        assert r_in > 0.0, f"Need positive inner radius, got {r_in =}"
        assert r_out > r_in, f"Need r_out > r_in, got {r_out =}, {r_in =}"
        assert length > 0.0, f"Need positive channel length, got {length =}"
        assert 0.0 <= pack_strength < 1.0, f"Need 0 <= pack_strength < 1, got {pack_strength =}"
        assert tor_period > 0, f"Need positive toroidal periodicity, got {tor_period =}"

        self.periodic_eta3 = False
        self.pole = False

        super().__init__()


class ShafranovShiftCylinder(Domain):
    r""" Cylinder with quadratic Shafranov shift.

    .. math:: 

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\eta_1^2)\,r_x\Delta\,\,\\
        \,\,y= &r_y\,\eta_1\sin(2\pi\,\eta_2)\,\,\\
        \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/shafranov_shift.png

    Parameters
    ----------
    rx : float
        Radius in x-direction (default: 1.0).
    ry : float
        Radius in y-direction (default: 1.0).
    Lz: float
        Length in z-direction (default: 4.0).
    delta : float
        Shift factor, should be in [0, 0.1] (default: 0.2).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : ShafranovShiftCylinder
            ShafranovShiftCylinder :
                rx    : 1. # axis length
                ry    : 1. # axis length
                Lz    : 4. # length in z-direction
                delta : .2 # shift factor, should be in [0, 0.1]
    """

    def __init__(
        self,
        rx: float = 1.0,
        ry: float = 1.0,
        Lz: float = 4.0,
        delta: float = 0.2,
    ):
        self.kind_map = 30

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = True

        super().__init__()


class ShafranovSqrtCylinder(Domain):
    r""" Cylinder with square-root Shafranov shift.

    .. math:: 

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\sqrt \eta_1)r_x\Delta\,\,\\
        \,\,y= &r_y\,\eta_1\sin(2\pi\,\eta_2)\,\,\\
        \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/shafranov_sqrt.png

    Parameters
    ----------
    rx : float
        Radius in x-direction (default: 1.0).
    ry : float
        Radius in y-direction (default: 1.0).
    Lz: float
        Length in z-direction (default: 4.0).
    delta : float
        Shift factor, should be in [0, 0.1] (default: 0.2).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : ShafranovSqrtCylinder
            ShafranovSqrtCylinder :
                rx    : 1. # axis length
                ry    : 1. # axis length
                Lz    : 4. # length in third direction
                delta : .2 # shift factor, should be in [0, 0.1]
    """

    def __init__(
        self,
        rx: float = 1.0,
        ry: float = 1.0,
        Lz: float = 4.0,
        delta: float = 0.2,
    ):
        self.kind_map = 31

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = True

        super().__init__()


class ShafranovDshapedCylinder(Domain):
    r""" Cylinder with D-shaped cross section and quadratic Shafranov shift.

    .. math:: 

        F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
        \,\,x= &R_0\left[1 + (1 - \eta_1^2)\Delta_x + \eta_1\epsilon\cos(2\pi\,\eta_2 + \arcsin(\delta)\eta_1\sin(2\pi\,\eta_2)) \right]\,\,\\
        \,\,y= &R_0\left[    (1 - \eta_1^2)\Delta_y + \eta_1\epsilon\kappa\sin(2\pi\,\eta_2)\right]\,\,\\
        \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}

    .. image:: ../../pics/mappings/shafranov_dshaped.png

    Parameters
    ----------
    R0 : float 
        Base radius (default: 2.).
    Lz : float 
        Length in z-direction (default: 4.).
    delta_x : float 
        Shafranov shift in x-direction (default: 0.05).
    delta_y : float 
        Shafranov shift in y-direction (default: 0.025).
    delta_gs : float 
        Delta = sin(alpha): triangularity, shift of high point  (default: 0.05).
    epsilon_gs : float
        Epsilon: inverse aspect ratio a/r0 (default: 0.5).
    kappa_gs : float 
        Kappa: ellipticity (elongation) (default: 2.).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : ShafranovDshapedCylinder
            ShafranovDshapedCylinder :
                R0         : 2. # base radius
                Lz         : 4. # length in third direction
                delta_x    : .05 # Shafranov shift in x-direction
                delta_y    : .025 # Shafranov shift in y-direction
                delta_gs   : .05 # delta = sin(alpha): triangularity, shift of high point
                epsilon_gs : .5 # epsilon: inverse aspect ratio a/r0
                kappa_gs   : 2. # Kappa: ellipticity (elongation)
    """

    def __init__(
        self,
        R0: float = 2.0,
        Lz: float = 3.0,
        delta_x: float = 0.1,
        delta_y: float = 0.0,
        delta_gs: float = 0.33,
        epsilon_gs: float = 0.32,
        kappa_gs: float = 1.7,
    ):
        self.kind_map = 32

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = True

        super().__init__()

class MoebiusStrip(Domain):
    r"""Thickened Moebius strip domain.
    
    A toroidal ribbon that performs a 180-degree twist over one revolution.
    This serves as a test for non-standard periodicity and metric tensor shifts.

    Parameters
    ----------
    R : float
        Major radius of the loop (default: 3.0).
    width : float
        Width of the ribbon (default: 1.0).
    thickness : float
        Thickness of the ribbon (default: 0.1).
    """

    def __init__(
        self,
        R: float = 3.0,
        width: float = 1.0,
        thickness: float = 0.1,
    ):
        self.kind_map = 50  # Unique ID for the new domain

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # Periodicity logic: 
        # Technically, eta3 is periodic, but the boundary at eta3=1 
        # maps to a flipped version of eta3=0.
        self.periodic_eta3 = True
        self.pole = False

        super().__init__()

    def map(self, eta1, eta2, eta3):
        """Analytical mapping for MoebiusStrip"""
        
        # Center the coordinates for width and thickness
        w_hat = self.params['width'] * (eta1 - 0.5)
        t_hat = self.params['thickness'] * (eta2 - 0.5)
        
        # Half-twist angle
        alpha = xp.pi * eta3
        # Standard toroidal angle
        phi = 2 * xp.pi * eta3

        # Effective radius in the XY plane
        r_eff = self.params['R'] + w_hat * xp.cos(alpha) - t_hat * xp.sin(alpha)

        x = r_eff * xp.cos(phi)
        y = r_eff * xp.sin(phi)
        z = w_hat * xp.sin(alpha) + t_hat * xp.cos(alpha)

        return x, y, z


class DiagnosticPortHoleTorus(Domain):
    r"""Torus with a localized diagnostic port deformation.

    This mapping models a toroidal vessel with a smooth, localized outward bulge
    attached to the torus tube. The bulge approximates the effect of a small
    radial/poloidal cylindrical diagnostic port while remaining a single smooth
    patch, making it suitable for studying perturbations near reactor openings.

    Parameters
    ----------
    a1 : float
        Inner minor radius of the torus tube (default: 0.1).
    a2 : float
        Outer minor radius of the torus tube (default: 1.0).
    R0 : float
        Major radius of the torus (default: 3.0).
    tor_period : int
        Toroidal periodicity built into the mapping: :math:`\phi=2\pi\,\eta_3/\mathrm{torperiod}` (default: 1).
    port_depth : float
        Maximum outward radial deformation of the localized port (default: 0.25).
    port_eta2_center : float
        Center of the port in logical poloidal coordinate :math:`\eta_2` (default: 0.0).
    port_eta3_center : float
        Center of the port in logical toroidal coordinate :math:`\eta_3` (default: 0.0).
    port_eta2_width : float
        Width parameter of the port in logical poloidal coordinate (default: 0.08).
    port_eta3_width : float
        Width parameter of the port in logical toroidal coordinate (default: 0.08).
    """

    def __init__(
        self,
        a1: float = 0.1,
        a2: float = 1.0,
        R0: float = 3.0,
        tor_period: int = 1,
        port_depth: float = 0.25,
        port_eta2_center: float = 0.0,
        port_eta3_center: float = 0.0,
        port_eta2_width: float = 0.08,
        port_eta3_width: float = 0.08,
    ):
        self.kind_map = 23

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        assert a2 <= R0, f"The minor radius must be smaller or equal than the major radius! {a2 =}, {R0 =}"
        assert a2 + port_depth <= R0, (
            f"The localized port deformation must keep the torus non-self-intersecting! {a2 =}, {port_depth =}, {R0 =}"
        )
        assert port_eta2_width > 0.0, f"Need positive port width in eta2, got {port_eta2_width =}"
        assert port_eta3_width > 0.0, f"Need positive port width in eta3, got {port_eta3_width =}"

        self.periodic_eta3 = True
        self.pole = a1 == 0.0

        super().__init__()