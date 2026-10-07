"""3d array evaluation of a tensor-product spline, distributed."""

from numpy import shape

from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi


def eval_spline_mpi_matrix(
    eta1: "float[:,:,:]",
    eta2: "float[:,:,:]",
    eta3: "float[:,:,:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts: "int[:]",
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

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        tn1, tn2, tn3 : array[float]
            Knot vectors of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

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
                    kind,
                    pn,
                    tn1,
                    tn2,
                    tn3,
                    starts,
                )
