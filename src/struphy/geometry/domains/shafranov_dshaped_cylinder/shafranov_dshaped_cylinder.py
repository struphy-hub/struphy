import copy

from struphy.geometry.base import Domain


class ShafranovDshapedCylinder(Domain):
    r"""Cylinder with D-shaped cross section and quadratic Shafranov shift.

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
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &R_0\left[1 + (1 - \eta_1^2)\Delta_x + \eta_1\epsilon\cos(2\pi\,\eta_2 + \arcsin(\delta)\eta_1\sin(2\pi\,\eta_2)) \right]\,\,\\
            \,\,y= &R_0\left[    (1 - \eta_1^2)\Delta_y + \eta_1\epsilon\kappa\sin(2\pi\,\eta_2)\right]\,\,\\
            \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}
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
