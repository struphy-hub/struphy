import copy

from struphy.geometry.base import Domain


class ShafranovShiftCylinder(Domain):
    r"""Cylinder with quadratic Shafranov shift.

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
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\eta_1^2)\,r_x\Delta\,\,\\
            \,\,y= &r_y\,\eta_1\sin(2\pi\,\eta_2)\,\,\\
            \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}
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
