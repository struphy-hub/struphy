import copy

import cunumpy as xp

from struphy.fields_background.base import AxisymmMHDequilibrium
from struphy.fields_background.equils import EQDSKequilibrium
from struphy.geometry.base import PoloidalSplineTorus
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
    """

    @classmethod
    def doc_mapping(cls):
        r"""Regarding r_min and psi_shifts:
        If r_min is left at 0.0, psi_shifts defines both the inner and outer boundaries of the computational
        domain in terms of the normalized flux coordinate \psi.
        When r_min > 0.0, however, psi_shifts[0] is no longer used. Instead, the code computes the flux value
        corresponding to the physical radius r_min (measured from the magnetic axis),
        which then defines the inner boundary of the domain.
        This allows the user to specify the inner boundary using a more intuitive physical radius rather
        than a flux coordinate. The outer boundary is still controlled by psi_shifts[1].
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
        # The field-line tracing evaluates psi on the active backend (batched Newton over all rays per flux surface);
        # its small spline interpolation runs on the host, and the control points are copied to the active backend once.
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

        cx, cy = xp.to_cunumpy(cx), xp.to_cunumpy(cy)

        # init base class
        super().__init__(
            num_elements=num_elements,
            degree=degree,
            spl_kind=(False, True),
            cx=cx,
            cy=cy,
            tor_period=tor_period,
        )
