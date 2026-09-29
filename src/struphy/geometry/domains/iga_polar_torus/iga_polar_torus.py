import copy

import cunumpy as xp

from struphy.geometry.base import PoloidalSplineTorus, interp_mapping


class IGAPolarTorus(PoloidalSplineTorus):
    r"""A torus with the poloidal cross-section approximated by a spline mapping.

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
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &\sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \cos(\phantom{-}2\pi\eta_3) \approx \left[a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0\right]\cos(\phantom{-}2\pi\eta_3)\,\,\\
            \,\,y= &\sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \sin(-2\pi\eta_3)\approx \left[a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0\right]\sin(-2\pi\eta_3)\,\,\\
            \,\,z= &\sum_{ij} c^{Z}_{ij} N_i(\eta_1) N_j(\eta_2)\approx a\,\eta_1\sin(2\pi\theta(\eta_1, \eta_2))\,\,\end{bmatrix}

        The angular parametrization :math:`\theta(\eta_1, \eta_2)` can either be equal angle or straight field line (see parameters below).
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
