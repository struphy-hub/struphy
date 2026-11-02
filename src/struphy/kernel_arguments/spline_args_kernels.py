# NOTE: This file must use ONLY numpy for pyccel compilation compatibility.
# Backend conversion (NumPy/CuPy) happens at the Python wrapper level.
import numpy as np


class SplineArguments:
    """Holds the arguments pertaining to one component of a :class:`~struphy.feec.psydac_derham.SplineFunction`
    passed to the spline evaluation kernels (``bsplines/kernels/eval_spline_mpi_*``).

    Paramaters
    ----------
    kind : array[int]
        Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

    pn : array[int]
        Spline degrees of V0 in each direction.

    tn1, tn2, tn3 : array[float]
        Knot vectors of V0 in each direction.

    starts : array[int]
        Start indices of the splines of this component on the current process.

    The scratch arrays ``bn1``, ..., ``bd3`` hold the B- and D-spline values at the current evaluation point.
    """

    def __init__(
        self,
        kind: "int[:]",
        pn: "int[:]",
        tn1: "float[:]",
        tn2: "float[:]",
        tn3: "float[:]",
        starts: "int[:]",
    ):
        self.kind = kind
        self.pn = pn
        self.tn1 = tn1
        self.tn2 = tn2
        self.tn3 = tn3
        self.starts = starts

        self.bn1 = np.empty(int(pn[0] + 1), dtype=float)
        self.bn2 = np.empty(int(pn[1] + 1), dtype=float)
        self.bn3 = np.empty(int(pn[2] + 1), dtype=float)
        self.bd1 = np.empty(int(pn[0]), dtype=float)
        self.bd2 = np.empty(int(pn[1]), dtype=float)
        self.bd3 = np.empty(int(pn[2]), dtype=float)
