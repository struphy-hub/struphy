"""Kernel hybrid_weight."""

from numpy import empty

import struphy.geometry.evaluation_kernels as evaluation_kernels
import struphy.kernel_arguments.pusher_args_kernels as pusher_args_kernels  # do not remove; needed to identify dependencies
from struphy.kernel_arguments.pusher_args_kernels import DomainArguments


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

    # fmt: off
    #$ omp parallel private (iel1, iel2, iel3, q1, q2, q3, value1, value2, value3, eta1, eta2, eta3, df_out, G, overn, value_new)
    # fmt: on
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
                                df_out[0, 0] * df_out[0, 2] + df_out[1, 0] * df_out[1, 2] + df_out[2, 0] * df_out[2, 2]
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
