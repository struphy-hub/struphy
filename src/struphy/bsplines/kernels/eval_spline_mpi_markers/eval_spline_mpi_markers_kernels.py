"""Flat (marker) evaluation of a tensor-product spline, distributed."""

from numpy import shape

import struphy.kernel_arguments.spline_args_kernels as spline_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi
from struphy.kernel_arguments.spline_args_kernels import SplineArguments


def eval_spline_mpi_markers(
    markers: "float[:,:]",
    _data: "float[:,:,:]",
    args_spline: "SplineArguments",
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

        args_spline : SplineArguments
            Kind of 1d basis, spline degrees and knot vectors of V0, and start indices of the splines on the
            current process.

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
            args_spline.kind,
            args_spline.pn,
            args_spline.tn1,
            args_spline.tn2,
            args_spline.tn3,
            args_spline.starts,
        )
