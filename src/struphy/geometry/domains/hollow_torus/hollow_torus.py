import copy

import cunumpy as xp

from struphy.geometry.base import Domain


class HollowTorus(Domain):
    r"""Torus with possible hole around the magnetic axis (center of the smaller circle).

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
    """

    @classmethod
    def doc_mapping(cls):
        r"""
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
        r = xp.sqrt(mr**2 + z**2)

        eta3 = xp.arctan2(-y, x) % (2 * xp.pi / self.params["tor_period"]) / (2 * xp.pi) * self.params["tor_period"]
        eta1 = (r - self.params["a1"]) / (self.params["a2"] - self.params["a1"])

        if self.params["sfl"]:
            # invert theta = 2*arctan(sqrt((1 + eps)/(1 - eps)) * tan(pi*eta2)) with eps = r/R0
            theta = xp.arctan2(z, mr) % (2 * xp.pi)
            g = xp.sqrt((1 + r / self.params["R0"]) / (1 - r / self.params["R0"]))
            eta2 = xp.arctan2(xp.sin(theta / 2), g * xp.cos(theta / 2)) / xp.pi % 1.0
        else:
            eta2 = xp.arctan2(z, mr) % (2 * xp.pi / self.params["pol_period"]) / (2 * xp.pi / self.params["pol_period"])

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
