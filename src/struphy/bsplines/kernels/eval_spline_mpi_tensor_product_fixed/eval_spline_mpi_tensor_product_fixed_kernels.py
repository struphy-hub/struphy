"""Tensor-product evaluation of a distributed spline on a fixed grid (spans and basis values pre-evaluated)."""

from numpy import empty
from pyccel.decorators import stack_array

from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi_kernel


@stack_array("b1", "b2", "b3")
def eval_spline_mpi_tensor_product_fixed(
    span1s: "int[:]",
    span2s: "int[:]",
    span3s: "int[:]",
    b1s: "float[:,:]",
    b2s: "float[:,:]",
    b3s: "float[:,:]",
    _data: "float[:,:,:]",
    kind: "int[:]",
    pn: "int[:]",
    starts: "int[:]",
    values: "float[:,:,:]",
):
    """
    Tensor-product evaluation of a tensor-product spline, distributed,
    and optimized for a fixed grid (spans and spline values have been pre-evaluated).

    Parameters
    ----------
        span1s, span2s, span3s : array[int]
            Knot span indices.

        b1s, b2s, b3s : array[float]
            Values of p+1 non-zero basis functions at evaluation points.

        _data : array[float]
            The spline coefficients c_ijk.

        kind : array[int]
            Kind of 1d basis in each direction: 0 = N-spline, 1 = D-spline.

        pn : array[int]
            Spline degrees of V0 in each direction.

        starts : array[float]
            Start indices of splines on current process.

        values : array[float]
            Return array for spline values S_ijk where ijk are the flattened indices corresponding to spans.
    """

    # allocate spline values
    b1 = empty(pn[0] - kind[0] + 1, dtype=float)
    b2 = empty(pn[1] - kind[1] + 1, dtype=float)
    b3 = empty(pn[2] - kind[2] + 1, dtype=float)

    ni = span1s.size
    nj = span2s.size
    nk = span3s.size

    for i in range(ni):
        span1 = span1s[i]
        b1[:] = b1s[i, :]
        for j in range(nj):
            span2 = span2s[j]
            b2[:] = b2s[j, :]
            for k in range(nk):
                span3 = span3s[k]
                b3[:] = b3s[k, :]

                values[i, j, k] = eval_spline_mpi_kernel(
                    pn[0] - kind[0],
                    pn[1] - kind[1],
                    pn[2] - kind[2],
                    b1,
                    b2,
                    b3,
                    span1,
                    span2,
                    span3,
                    _data,
                    starts,
                )
