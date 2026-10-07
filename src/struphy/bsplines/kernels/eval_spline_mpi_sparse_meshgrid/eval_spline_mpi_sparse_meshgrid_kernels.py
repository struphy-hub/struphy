"""Sparse meshgrid evaluation of a tensor-product spline, distributed."""

from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi


def eval_spline_mpi_sparse_meshgrid(
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
    Sparse meshgrid evaluation of a tensor-product spline, distributed.

    Parameters
    ----------
        eta1, eta2, eta3 : array[float]
            Evaluation points as 3d arrays obtained from sparse meshgrid; points not on local process domain must be flagged as -1.
            Spline values are obtained as S_ijk = S(eta1[i,0,0], eta2[0,j,0], eta3[0,0,k]).

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
                    kind,
                    pn,
                    tn1,
                    tn2,
                    tn3,
                    starts,
                )
