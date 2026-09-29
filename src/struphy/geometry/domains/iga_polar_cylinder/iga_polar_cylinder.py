import copy

import cunumpy as xp

from struphy.geometry.base import PoloidalSplineStraight, interp_mapping


class IGAPolarCylinder(PoloidalSplineStraight):
    r"""A cylinder with the cross section approximated by a spline mapping.

    .. image:: ../../pics/mappings/iga_cylinder.png

    Parameters
    ----------
    num_elements : list[int]
        Number of cells in (radial, angular) direction used for spline mapping (default: [8, 24]).
    degree : list[int]
        Splines degrees in (radial, angular) direction used for spline mapping (default: [2, 3]).
    a : float
        Radius of cylinder (default: 1.).
    Lz : float
        Length of cylinder (default: 4.).
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &\sum_{ij} c^x_{ij} N_i(\eta_1) N_j(\eta_2)\approx a\,\eta_1\cos(2\pi\eta_2)\,\,\\
            \,\,y= &\sum_{ij} c^y_{ij} N_i(\eta_1) N_j(\eta_2)\approx a\,\eta_1\sin(2\pi\eta_2)\,\,\\
            \,\,z= &L_z\eta_3\,\,\end{bmatrix}
        """

    def __init__(
        self,
        num_elements: tuple[int] = (8, 24),
        degree: tuple[int] = (2, 3),
        a: float = 1.0,
        Lz: float = 4.0,
    ):
        # use params setter
        self.params = copy.deepcopy(locals())

        # get control points
        def X(eta1, eta2):
            return a * eta1 * xp.cos(2 * xp.pi * eta2)

        def Y(eta1, eta2):
            return a * eta1 * xp.sin(2 * xp.pi * eta2)

        spl_kind = (False, True)

        cx, cy = interp_mapping(num_elements, degree, spl_kind, X, Y)

        # make sure that control points at pole are all the same (eta1=0 there)
        cx[0] = 0.0
        cy[0] = 0.0

        # init base class
        super().__init__(num_elements=num_elements, degree=degree, spl_kind=spl_kind, cx=cx, cy=cy, Lz=Lz)
