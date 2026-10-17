import copy

from struphy.geometry.base import Domain


class RoundedSquareCylinder(Domain):
    r"""Cylinder whose cross-section is a rounded square, the zero set of
    :math:`g = 1 - x^2 - y^2 - \varepsilon (x^4 - 6x^2y^2 + y^4)` (for :math:`a = 1`).

    This is the domain of the Poisson and curl-curl examples of PolarFEEC.jl (``examples/05_polar_pde.jl``).

    Parameters
    ----------
    a : float
        Scale of the cross-section (default: 1.0).
    eps : float
        Squareness of the cross-section (default: 0.15).
    Lz : float
        Length of the cylinder (default: 1.0).
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &a\,\eta_1\,\rho(\eta_2)\cos(2\pi\,\eta_2)\,\,\\
            \,\,y= &a\,\eta_1\,\rho(\eta_2)\sin(2\pi\,\eta_2)\,\,\\
            \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}\,, \qquad
            \rho(\eta_2) = \sqrt{\frac{2}{1 + \sqrt{1 + 4\varepsilon\cos(8\pi\,\eta_2)}}}
        """

    def __init__(
        self,
        a: float = 1.0,
        eps: float = 0.15,
        Lz: float = 1.0,
    ):
        self.kind_map = 24

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = True

        super().__init__()
