"""Sparse meshgrid evaluation of a tensor-product spline, distributed."""

import struphy.kernel_arguments.spline_args_kernels as spline_args_kernels  # do not remove; needed to identify dependencies
from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi
from struphy.kernel_arguments.spline_args_kernels import SplineArguments


def eval_spline_mpi_sparse_meshgrid(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    _data: "float[:,:,:]",
    args_spline: "SplineArguments",
    values: "float[:,:,:]",
):
    """
    Sparse meshgrid evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]
            Evaluation points as 3d arrays obtained from sparse meshgrid; points not on local process domain must be flagged as -1.
            Spline values are obtained as S_ijk = S(eta1[i,0,0], eta2[0,j,0], eta3[0,0,k]).

        _data : array[float]
            The spline coefficients c_ijk.

        args_spline : SplineArguments
            Kind of 1d basis, spline degrees and knot vectors of V0, and start indices of the splines on the
            current process.

        values : array[float]
            Return array for spline values S_ijk = S(eta1[i,0,0], eta2[0,j,0], eta3[0,0,k]).
    """

    from numpy import size

    n1 = size(eta1)
    n2 = size(eta2)
    n3 = size(eta3)

    for i in range(n1):
        if eta1[i, 0, 0] == -1.0:
            continue  # point not in process domain
        for j in range(n2):
            if eta2[0, j, 0] == -1.0:
                continue  # point not in process domain
            for k in range(n3):
                if eta3[0, 0, k] == -1.0:
                    continue  # point not in process domain

                values[i, j, k] = eval_spline_mpi(
                    eta1[i, 0, 0],
                    eta2[0, j, 0],
                    eta3[0, 0, k],
                    _data,
                    args_spline.kind,
                    args_spline.pn,
                    args_spline.tn1,
                    args_spline.tn2,
                    args_spline.tn3,
                    args_spline.starts,
                )
