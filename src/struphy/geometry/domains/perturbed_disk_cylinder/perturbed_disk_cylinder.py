import copy

from struphy.geometry.base import Domain


class PerturbedDiskCylinder(Domain):
    r"""Cylinder whose cross-section is the perturbed (``k``-petal) disk of the polar FEEC paper.

    The cross-section is the disk-like map :math:`R(\eta_1, \eta_2) = R_1 \eta_1 (1 + \varepsilon \eta_1^q \cos(2\pi k
    \eta_2))` (``q = 1``: the perturbed disk, ``q = 0``: a star-shaped disk) of R. Remmerswaal, *High-order FEEC discretization on 2D and 3D domains with a disk-like coordinate
    singularity*, which is exactly representable by the polar splines (``r``-compatible) only with trigonometric
    functions in :math:`\eta_2`. With :math:`\varepsilon = 0` it is the circular cylinder of radius :math:`R_1`.

    Parameters
    ----------
    R1 : float
        Radius of the unperturbed disk (default: 1.0).
    eps : float
        Amplitude of the perturbation (default: 0.2).
    k : int
        Number of petals (default: 3).
    Lz : float
        Length of the cylinder (default: 1.0).
    power : float
        Power ``q`` of ``eta1`` in the perturbation (default: 1.0; 0.0 gives the star-shaped disk
        ``R = R_1 eta1 (1 + eps cos(2 pi k eta2))`` of the regularity study of the talk).
    """

    @classmethod
    def doc_mapping(cls):
        r"""
        .. math::

            F: \begin{bmatrix}\eta_1\\ \eta_2\\ \eta_3\end{bmatrix}\mapsto \begin{bmatrix}
            \,\,x= &R_1\,\eta_1\,\left(1 + \varepsilon\,\eta_1^q\cos(2\pi k\,\eta_2)\right)\cos(2\pi\,\eta_2)\,\,\\
            \,\,y= &R_1\,\eta_1\,\left(1 + \varepsilon\,\eta_1^q\cos(2\pi k\,\eta_2)\right)\sin(2\pi\,\eta_2)\,\,\\
            \,\,z= &L_z\,\eta_3\,\,\end{bmatrix}
        """

    def __init__(
        self,
        R1: float = 1.0,
        eps: float = 0.2,
        k: int = 3,
        Lz: float = 1.0,
        power: float = 1.0,
    ):
        self.kind_map = 23

        # use params setter
        self.params = copy.deepcopy(locals())
        self.params_numpy = self.get_params_numpy()

        # periodicity in eta3-direction and pole at eta1=0
        self.periodic_eta3 = False
        self.pole = True

        super().__init__()
