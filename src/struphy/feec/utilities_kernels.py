from numpy import empty
from pyccel.decorators import pure, stack_array

import struphy.bsplines.bsplines_kernels as bsplines_kernels
import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
import struphy.pic.accumulation.filler_kernels as filler_kernels
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, DomainArguments


@pure
@stack_array("bn1", "bn2", "bn3")
def l2_projection_V0(
    pn: "int[:]",
    tn1: "float[:]",
    tn2: "float[:]",
    tn3: "float[:]",
    starts0: "int[:]",
    vec: "float[:,:,:]",
    quad_locs_1: "float[:]",
    quad_locs_2: "float[:]",
    quad_locs_3: "float[:]",
    scaled_wts_1: "float[:]",
    scaled_wts_2: "float[:]",
    scaled_wts_3: "float[:]",
    fun_vals: "float[:,:,:]",
):
    """Kernel for integration of a function times basis functions in V0.

    Do integration using quadrature points and weights from Gauss-Legendre quadrature.

    Parameters
    ----------
    pn : array[int]
        Spline degrees in each direction.

    tn1, tn2, tn3 : array[float]
        Spline knot vectors in each direction.

    starts0 : array[int]
        Start indices of the current process in space V0.

    vec : array[float]
        Vector in which the basis functions times the function values is to be written.

    quad_locs_1, _2, _3 : array[float]
        quadrature points in each direction.

    scaled_wts_1, _2, _3 : array[float]
        Quadrature weights scaled by the cell size in each direction.

    fun_vals : array[float]
        Function values, each axis is a direction in space.
    """

    nr_points_1 = len(quad_locs_1)
    nr_points_2 = len(quad_locs_2)
    nr_points_3 = len(quad_locs_3)

    # degrees of the basis functions : B-splines (pn) and D-splines (pd)
    pn1 = pn[0]
    pn2 = pn[1]
    pn3 = pn[2]

    # non-vanishing B-splines at particle position
    bn1 = empty(pn[0] + 1, dtype=float)
    bn2 = empty(pn[1] + 1, dtype=float)
    bn3 = empty(pn[2] + 1, dtype=float)

    for i in range(nr_points_1):
        eta1 = quad_locs_1[i]
        span1 = bsplines_kernels.find_span(tn1, pn1, eta1)
        bsplines_kernels.b_splines_slim(tn1, pn1, eta1, span1, bn1)

        for j in range(nr_points_2):
            eta2 = quad_locs_2[j]
            span2 = bsplines_kernels.find_span(tn2, pn2, eta2)
            bsplines_kernels.b_splines_slim(tn2, pn2, eta2, span2, bn2)

            for k in range(nr_points_3):
                eta3 = quad_locs_3[k]
                span3 = bsplines_kernels.find_span(tn3, pn3, eta3)
                bsplines_kernels.b_splines_slim(tn3, pn3, eta3, span3, bn3)

                fill = fun_vals[i, j, k] * scaled_wts_1[i] * scaled_wts_2[j] * scaled_wts_3[k]

                filler_kernels.fill_vec(pn1, pn2, pn3, bn1, bn2, bn3, span1, span2, span3, starts0, vec, fill)


# # ================= 3d =================================
# def hybrid_curlA(starts1: 'float[:,:]', starts2: 'float[:,:]', starts3: 'float[:,:]',
#                  spans1: 'int[:]', spans2: 'int[:]', spans3: 'int[:]',
#                  pi1: int, pi2: int, pi3: int,
#                  nq1: int, nq2: int, nq3: int,
#                  bi1: 'float[:,:,:,:]', bi2: 'float[:,:,:,:]', bi3: 'float[:,:,:,:]',
#                  data: 'float[:,:,:]', coeffs: 'float[:,:,:]'):

#     nel1 = spans1.size
#     nel2 = spans2.size
#     nel3 = spans3.size

#     # -- removed omp: #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, value, il1, il2, il3, i1, i2, i3)
#     for iel1 in range(nel1):
#         for iel2 in range(nel2):
#             for iel3 in range(nel3):

#                 for q1 in range(nq1):
#                     for q2 in range(nq2):
#                         for q3 in range(nq3):

#                             value = 0.0
#                             for il1 in range(pi1 + 1):
#                                 i1 = spans1[iel1] - pi1 + il1 - starts1
#                                 for il2 in range(pi2 + 1):
#                                     i2 = spans2[iel2] - pi2 + il2 - starts2
#                                     for il3 in range(pi3 + 1):
#                                         i3 = spans3[iel3] - pi3 + il3 - starts3
#                                         value += bi1[iel1, il1, 0, q1] * bi2[iel2, il2, 0, q2] * \
#                                             bi3[iel3, il3, 0, q3] * \
#                                             coeffs[i1, i2, i3]

#                             data[iel1*nq1+q1, iel2*nq2+q2, iel3*nq3+q3] = value
#     # -- removed omp: #$ omp end parallel


# ================= 3d =================================
def hybrid_weight(
    pads1: int,
    pads2: int,
    pads3: int,
    pts1: "float[:,:]",
    pts2: "float[:,:]",
    pts3: "float[:,:]",
    spans1: "int[:]",
    spans2: "int[:]",
    spans3: "int[:]",
    nq1: int,
    nq2: int,
    nq3: int,
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    data1: "float[:,:,:]",
    data2: "float[:,:,:]",
    data3: "float[:,:,:]",
    n_data: "float[:,:,:,:,:,:]",
    args_domain: "DomainArguments",
):
    nel1 = spans1.size
    nel2 = spans2.size
    nel3 = spans3.size

    df_out = empty((3, 3), dtype=float)
    G = empty((3, 3), dtype=float)
    value_new = empty(3, dtype=float)

    # -- removed omp: #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, value1, value2, value3, eta1, eta2, eta3, df_out, G, overn, value_new)
    for iel1 in range(nel1):
        for iel2 in range(nel2):
            for iel3 in range(nel3):
                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            value1 = data1[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]
                            value2 = data2[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]
                            value3 = data3[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3]

                            eta1 = pts1[iel1, q1]
                            eta2 = pts2[iel2, q2]
                            eta3 = pts3[iel3, q3]

                            evaluation_kernels.df(eta1, eta2, eta3, args_domain, df_out)
                            # sqrtg = evaluation_kernels.det_df(eta1, eta2, eta3, kind_map, params_map, t1, t2, t3, p_map, ind1, ind2, ind3, cx, cy, cz)

                            G[0, 0] = (
                                df_out[0, 0] * df_out[0, 0] + df_out[1, 0] * df_out[1, 0] + df_out[2, 0] * df_out[2, 0]
                            )
                            G[0, 1] = (
                                df_out[0, 0] * df_out[0, 1] + df_out[1, 0] * df_out[1, 1] + df_out[2, 0] * df_out[2, 1]
                            )
                            G[0, 2] = (
                                df_out[0, 0] * df_out[0, 2] + df_out[1, 0] * df_out[1, 2] + df_out[2, 2] * df_out[2, 2]
                            )

                            G[1, 1] = (
                                df_out[0, 1] * df_out[0, 1] + df_out[1, 1] * df_out[1, 1] + df_out[2, 1] * df_out[2, 1]
                            )
                            G[1, 2] = (
                                df_out[0, 1] * df_out[0, 2] + df_out[1, 1] * df_out[1, 2] + df_out[2, 1] * df_out[2, 2]
                            )

                            G[2, 2] = (
                                df_out[0, 2] * df_out[0, 2] + df_out[1, 2] * df_out[1, 2] + df_out[2, 2] * df_out[2, 2]
                            )

                            G[1, 0] = G[0, 1]
                            G[2, 0] = G[0, 2]
                            G[2, 1] = G[1, 2]

                            if n_data[pads1 + iel1, pads2 + iel2, pads3 + iel3, q1, q2, q3] < 0.001:
                                overn = 0.0
                            else:
                                overn = 1.0 / n_data[pads1 + iel1, pads2 + iel2, pads3 + iel3, q1, q2, q3]

                            value_new[0] = (G[0, 0] * value1 + G[0, 1] * value2 + G[0, 2] * value3) * overn
                            value_new[1] = (G[1, 0] * value1 + G[1, 1] * value2 + G[1, 2] * value3) * overn
                            value_new[2] = (G[2, 0] * value1 + G[2, 1] * value2 + G[2, 2] * value3) * overn

                            data1[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3] = value_new[0]
                            data2[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3] = value_new[1]
                            data3[iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3] = value_new[2]
    # -- removed omp: #$ omp end parallel
