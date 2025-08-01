"Mapped domains (single patch)."

import copy

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
from struphy.utils.arrays import xp as np


class Tokamak(PoloidalSplineTorus):
    r"""Mappings for Tokamak MHD equilibria constructed via :ref:`field-line tracing <field_tracing>` of a poloidal flux function :math:`\psi`.

    .. image:: ../../pics/mappings/tokamak.png

    Parameters
    ----------
    equilibrium : struphy.fields_background.base.AxisymmMHDequilibrium
        The axisymmetric MHD equilibrium for which a flux-aligned grid shall be constructed (default: AdhocTorus).
    Nel : tuple[int]
        Number of cells in (radial, angular) direction to be used in spline mapping (default: [8, 32]).
    p : tuple[int]
        Spline degrees in (radial, angular) direction to be used in spline mapping (default: [2, 3]).
    psi_power : float
        Parametrization of radial flux coordinate :math:`\eta_1=\psi_{\mathrm{norm}}^p`, where :math:`\psi_{\mathrm{norm}}` is the normalized poloidal flux (default: 0.75).
    psi_shifts : tuple[float]
        Start and end shifts of polidal flux in % --> cuts away regions at the axis and edge (default: [2., 2.])
    xi_param : str
        Parametrization of angular coordinate ("equal_angle", "equal_arc_length" or "sfl" (straight field line), default: "equal_angle").
    r0 : float
        Initial guess for radial distance from axis used in Newton root-finding method (default: 0.3).
    Nel_pre : tuple[int]
        Number of cells in (radial, angular) direction of pre-mapping needed for equal_arc_length and sfl parametrizations (default: [64, 256]).
    p : tuple[int]
        Spline degrees in (radial, angular) direction of pre-mapping needed for equal_arc_length and sfl parametrizations (default: [3, 3]).
    tor_period : int
        Toroidal periodicity built into the mapping: :math:`\phi=2\pi\,\eta_3/\mathrm{torperiod}` (default: 1 --> full torus).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : Tokamak
            Tokamak :
                Nel        : [8, 32]     # number of poloidal grid cells for spline mapping, >p
                p          : [3, 3]      # poloidal spline degrees for spline mapping, >1
                psi_power  : 0.7         # parametrization of radial flux coordinate eta1=psi_norm^psi_power, where psi_norm is normalized flux
                psi_shifts : [2., 2.]    # start and end shifts of polidal flux in % --> cuts away regions at the axis and edge
                xi_param   : equal_angle # parametrization of angular coordinate (equal_angle, equal_arc_length or sfl (straight field line))
                r0         : 0.3         # initial guess for radial distance from axis used in Newton root-finding method for flux surfaces
                Nel_pre    : [64, 256]   # number of poloidal grid cells of pre-mapping needed for equal_arc_length and sfl
                p_pre      : [3, 3]      # poloidal spline degrees of pre-mapping needed for equal_arc_length and sfl
                tor_period : 1           # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
    """

    def __init__(
        self,
        equilibrium: AxisymmMHDequilibrium = None,
        Nel: tuple = (8, 32),
        p: tuple = (2, 3),
        psi_power: float = 0.75,
        psi_shifts: tuple = (0.01, 2.0),
        xi_param: str = "equal_angle",
        r0: float = 0.3,
        Nel_pre: tuple = (64, 256),
        p_pre: tuple = (3, 3),
        tor_period: int = 1,
    ):
        if equilibrium is None:
            equilibrium = EQDSKequilibrium()
        else:
            assert isinstance(equilibrium, AxisymmMHDequilibrium)

        # use the params setter
        self.params = copy.deepcopy(locals())

        # get control points via field tracing between fluxes [psi_s, psi_e]
        psi0, psi1 = equilibrium.psi_range[0], equilibrium.psi_range[1]

        psi_s = psi0 + psi_shifts[0] * 0.01 * (psi1 - psi0)
        psi_e = psi1 - psi_shifts[1] * 0.01 * (psi1 - psi0)

        cx, cy = field_line_tracing(
            equilibrium.psi,
            equilibrium.psi_axis_RZ[0],
            equilibrium.psi_axis_RZ[1],
            psi_s,
            psi_e,
            Nel,
            p,
            psi_power=psi_power,
            xi_param=xi_param,
            Nel_pre=Nel_pre,
            p_pre=p_pre,
            r0=r0,
        )

        # init base class
        super().__init__(
            Nel=Nel,
            p=p,
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

        Nel = gvec_equil.params["Nel"]
        p = gvec_equil.params["p"]
        if gvec_equil.params["use_nfp"]:
            spl_kind = (False, True, False)
        else:
            spl_kind = (False, True, True)

        # project mapping to splines
        _rmin = gvec_equil.params["rmin"]

        def XYZ(e1, e2, e3):
            rho = _rmin + e1 * (1.0 - _rmin)
            theta = 2 * np.pi * e2
            zeta = 2 * np.pi * e3 / gvec_equil._nfp
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

        cx, cy, cz = interp_mapping(Nel, p, spl_kind, X, Y, Z)

        super().__init__(Nel=Nel, p=p, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)


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

        Nel = desc_equil.params["Nel"]
        p = desc_equil.params["p"]

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

        cx, cy, cz = interp_mapping(Nel, p, spl_kind, X, Y, Z)

        super().__init__(Nel=Nel, p=p, spl_kind=spl_kind, cx=cx, cy=cy, cz=cz)


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
    Nel : list[int]
        Number of cells in (radial, angular) direction used for spline mapping (default: [8, 24]).
    p : list[int]
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
                Nel : [8, 24] # number of poloidal grid cells, >p
                p   : [3, 3] # poloidal spline degree, >1
                Lz  : 6. # Length in third direction
                a   : 1. # minor radius
    """

    def __init__(
        self,
        Nel: tuple[int] = (8, 24),
        p: tuple[int] = (2, 3),
        a: float = 1.0,
        Lz: float = 4.0,
    ):
        # use params setter
        self.params = copy.deepcopy(locals())

        # get control points
        def X(eta1, eta2):
            return a * eta1 * np.cos(2 * np.pi * eta2)

        def Y(eta1, eta2):
            return a * eta1 * np.sin(2 * np.pi * eta2)

        spl_kind = (False, True)

        cx, cy = interp_mapping(Nel, p, spl_kind, X, Y)

        # make sure that control points at pole are all the same (eta1=0 there)
        cx[0] = 0.0
        cy[0] = 0.0

        # init base class
        super().__init__(Nel=Nel, p=p, spl_kind=spl_kind, cx=cx, cy=cy, Lz=Lz)


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
    Nel : tuple[int]
        Number of cells in (radial, angular) direction used for spline mapping (default: [8, 24]).
    p : tuple[int]
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
                Nel        : [8, 24] # number of poloidal grid cells, >p
                p          : [3, 3] # poloidal spline degree, >1
                a          : 1. # minor radius
                R0         : 3. # major radius
                tor_period : 2 # toroidal periodicity built into the mapping: phi = 2*pi * eta3 / tor_period
                sfl        : False # whether to use straight field line coordinates (particular theta parametrization) 
    """

    def __init__(
        self,
        Nel: tuple[int] = (8, 24),
        p: tuple[int] = (2, 3),
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
                return 2 * np.arctan(np.sqrt((1 + a * eta1 / R0) / (1 - a * eta1 / R0)) * np.tan(np.pi * eta2))
        else:

            def theta(eta1, eta2):
                return 2 * np.pi * eta2

        def R(eta1, eta2):
            return a * eta1 * np.cos(theta(eta1, eta2)) + R0

        def Z(eta1, eta2):
            return a * eta1 * np.sin(theta(eta1, eta2))

        spl_kind = (False, True)

        cx, cy = interp_mapping(Nel, p, spl_kind, R, Z)

        # make sure that control points at pole are all the same (eta1=0 there)
        cx[0] = R0
        cy[0] = 0.0

        # init base class
        super().__init__(
            Nel=Nel,
            p=p,
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
        End of x-interval, r1>l1 (default: 2.).
    l2 : float
        Start of y-interval (default: 0.).
    r2 : float
        End of y-interval, r2>l2 (default: 3.).
    l3 : float
        Start of z-interval (default: 0.).
    r3 : float
        End of z-interval, r3>l3 (default: 6.).

    Note
    ----
    In the parameter .yml, use the following in the section `geometry`::

        geometry :
            type : Cuboid
            Cuboid :
                l1 : 0. # start of x-interval
                r1 : 2. # end of x-interval, r1>l1
                l2 : 0. # start of y-interval
                r2 : 2. # end of y-interval, r2>l2
                l3 : 0. # start of z-interval
                r3 : 1. # end of z-interval, r3>l3
    """

    def __init__(self, l1: float = 0.0,
                       r1: float = 2.0,
                       l2: float = 0.0,
                       r2: float = 3.0,
                       l3: float = 0.0,
                       r3: float = 6.0,):
        
        self._kind_map = 10

        self._params_map, self._params_numpy = Domain.prepare_params_map_new(
            l1=l1,
            r1=r1,
            l2=l2,
            r2=r2,
            l3=l3,
            r3=r3,
        )

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

        assert a2 <= R0, f"The minor radius must be smaller or equal than the major radius! {a2 = }, {R0 = }"

        if sfl:
            assert pol_period == 1, (
                f"Piece-of-cake is only implemented for torus coordinates, not for straight field line coordinates!"
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

        mr = np.sqrt(x**2 + y**2) - self.params["R0"]

        eta3 = np.arctan2(-y, x) % (2 * np.pi / self.params["tor_period"]) / (2 * np.pi) * self.params["tor_period"]
        eta2 = np.arctan2(z, mr) % (2 * np.pi / self.params["pol_period"]) / (2 * np.pi / self.params["pol_period"])
        eta1 = (z / np.sin(2 * np.pi * eta2 / self.params["pol_period"]) - self.params["a1"]) / (
            self.params["a2"] - self.params["a1"]
        )

        if bounded:
            eta1[eta1 > 1] = 1.0
            eta1[eta1 < 0] = 0.0
            assert np.all(np.logical_and(eta1 >= 0, eta1 <= 1))

        assert np.all(np.logical_and(eta2 >= 0, eta2 <= 1))
        assert np.all(np.logical_and(eta3 >= 0, eta3 <= 1))

        if change_out_order:
            return np.transpose((eta1, eta2, eta3))

        else:
            return eta1, eta2, eta3


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
