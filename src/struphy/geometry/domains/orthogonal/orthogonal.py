import copy

from struphy.geometry.base import Domain


class Orthogonal(Domain):
    r"""Slab geometry with orthogonal mesh distortion.

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
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &L_x\,\left[\,\eta_1 + \alpha\sin(2\pi\,\eta_1)\right]\,\,\\
            \,\,y= &L_y\,\left[\,\eta_2 + \alpha\sin(2\pi\,\eta_2)\right]\,\,\\
            \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}
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
