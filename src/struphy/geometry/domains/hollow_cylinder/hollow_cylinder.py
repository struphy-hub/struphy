import copy

from struphy.geometry.base import Domain


class HollowCylinder(Domain):
    r"""Cylinder with possible hole around the axis.

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
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2 / poc)\,\,\\
            \,\,y= &\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin(2\pi\,\eta_2 / poc)\,\,\\
            \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}
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
