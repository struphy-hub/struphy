import copy

from struphy.geometry.base import Domain


class Cuboid(Domain):
    r"""Slab geometry (Cartesian coordinates).

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

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &l_1 + (r_1 - l_1)\,\eta_1\,\,\\
            \,\,y= &l_2 + (r_2 - l_2)\,\eta_2\,\,\\
            \,\,z= &l_3 + (r_3 - l_3)\,\eta_3\,\,\end{bmatrix}
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
