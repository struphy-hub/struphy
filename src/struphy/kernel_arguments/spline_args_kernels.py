# NOTE: This file must use ONLY numpy for pyccel compilation compatibility.
# Backend conversion (NumPy/CuPy) happens at the Python wrapper level.


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
