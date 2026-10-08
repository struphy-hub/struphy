"""3d array evaluation of a tensor-product spline, distributed."""

from numpy import shape

import struphy.kernel_arguments.spline_args_kernels as spline_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi
from struphy.kernel_arguments.spline_args_kernels import SplineArguments


def eval_spline_mpi_matrix(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    _data: "float[:,:,:]",
    args_spline: "SplineArguments",
    values: "float[:,:,:]",
):
    """
    3d array evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]
            Evaluation points as 3d arrays; points not on local process domain must be flagged as -1.
            Spline values are obtained as S_ijk = S(eta1[i,j,k], eta2[i,j,k], eta3[i,j,k]).

        _data : array[float]
            The spline coefficients c_ijk.

        args_spline : SplineArguments
            Kind of 1d basis, spline degrees and knot vectors of V0, and start indices of the splines on the
            current process.

        values : array[float]
            Return array for spline values S_ijk = S(eta1[i,j,k], eta2[i,j,k], eta3[i,j,k]).
    """

    shp = shape(eta1)

    for i in range(shp[0]):
        for j in range(shp[1]):
            for k in range(shp[2]):
                if eta1[i, j, k] == -1.0:
                    continue  # point not in process domain
                if eta2[i, j, k] == -1.0:
                    continue  # point not in process domain
                if eta3[i, j, k] == -1.0:
                    continue  # point not in process domain

                values[i, j, k] = eval_spline_mpi(
                    eta1[i, j, k],
                    eta2[i, j, k],
                    eta3[i, j, k],
                    _data,
                    args_spline.kind,
                    args_spline.pn,
                    args_spline.tn1,
                    args_spline.tn2,
                    args_spline.tn3,
                    args_spline.starts,
                )
