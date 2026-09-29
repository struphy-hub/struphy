import copy

from struphy.geometry.base import Domain


class PoweredEllipticCylinder(Domain):
    r"""Cylinder with elliptic cross section and radial power law.

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
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &r_x\,\eta_1^s\cos(2\pi\,\eta_2)\,\,\\
            \,\,y= &r_y\,\eta_1^s\sin(2\pi\,\eta_2)\,\,\\
            \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}
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
