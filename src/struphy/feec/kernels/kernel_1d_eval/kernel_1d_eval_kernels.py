"""Kernel kernel_1d_eval: integral kernel for mass matrices and L2-projections."""

from numpy import shape


def kernel_1d_eval(
    spans1: "int[:]",
    pi1: int,
    starts1: int,
    pads1: int,
    bi1: "float[:,:,:,:]",
    coeffs_data: "float[:]",
    values: "float[:]",
):
    """
    Evaluates sum_i1 [ coeffs_i1 * Lambda_i1(quad_eta1) ] for all quadrature points on the calling process.

    The results are written into values.

    Parameters
    ----------
    spans1 : array[int]
        Array of span indices; the span is the index of the last non-vanishing spline on each grid element
        (cell). The length of the returned array is the number of elements (cells).
    pi1 : int
        Degree of the basis functions.
    starts1 : int
        Starting index on the current rank.
    pads1 : int
        Padding (=spline degree) for ghost regions in coeffs_data.
    bi1 : "float[:,:,:,:]"
        Values of basis functions. The indexing is [global element, local basis function, derivative, quadrature point].
    coeffs_data : "float[:]"
        _data array of StencilVector holding the spline coefficients of the function to be evaluated.
    values : "float[:]"
        Output array (flattened over elements and quadrature points) holding the evaluated function values;
        it is set to zero at the start of the kernel, i.e. it is overwritten, not added to.
    """

    values[:] = 0.0

    ne1 = spans1.size

    nq1 = shape(bi1)[3]

    for iel1 in range(ne1):
        for il1 in range(pi1 + 1):
            # global spline indices
            i_global1 = spans1[iel1] - pi1 + il1

            # local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
            i_local1 = i_global1 - starts1

            for q1 in range(nq1):
                values[iel1 * nq1 + q1] += coeffs_data[pads1 + i_local1] * bi1[iel1, il1, 0, q1]
