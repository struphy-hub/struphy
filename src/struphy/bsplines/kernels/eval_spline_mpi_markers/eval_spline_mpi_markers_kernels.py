"""Flat (marker) evaluation of a tensor-product spline, distributed."""

from numpy import shape

from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi


def eval_spline_mpi_markers(
    markers: "float[:,:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
    values: "float[:]",
):
    """
    Flat (marker) evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        markers : array[float]
            Marker coordinates in format [Np, 3]; markers not on local process domain must be flagged as -1.
            Spline values are obtained as S_p = S(*markers[p, :]).

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return 1D array for spline values S_p = S(*markers[p, :]).
    """

    Np = shape(markers)[0]

    for ip in range(Np):
        if markers[ip, 0] == -1:
            continue  # point not in process domain

        values[ip] = eval_spline_mpi(
            markers[ip, 0],
            markers[ip, 1],
            markers[ip, 2],
            _data,
            kind,
            pn,
            tn1,
            tn2,
            tn3,
            starts,
        )
