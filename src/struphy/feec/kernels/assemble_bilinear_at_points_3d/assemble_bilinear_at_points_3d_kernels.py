"""Kernel assemble_bilinear_at_points_3d: assembly of a weighted bilinear form of two spline bases at a tensor-product
point set (e.g. the quadrature points of a commuting projector)."""


def assemble_bilinear_at_points_3d(
    mat: "float[:,:,:,:,:,:]",
    starts_out: "int[:]",
    ends_out: "int[:]",
    pads_out: "int[:]",
    pads_in: "int[:]",
    npts_out: "int[:]",
    npts_in: "int[:]",
    periods: "int[:]",
    span_out1: "int[:]",
    span_out2: "int[:]",
    span_out3: "int[:]",
    basis_out1: "float[:,:]",
    basis_out2: "float[:,:]",
    basis_out3: "float[:,:]",
    span_in1: "int[:]",
    span_in2: "int[:]",
    span_in3: "int[:]",
    basis_in1: "float[:,:]",
    basis_in2: "float[:,:]",
    basis_in3: "float[:,:]",
    weight: "float[:,:,:]",
):
    """Adds the matrix

    A_(ijk,mno) = sum_q weight_q Lambda^out_ijk(x_q) Lambda^in_mno(x_q)

    to the _data attribute of a StencilMatrix (rows: basis of the output space, columns: basis of the input space),
    for the points x_q = (x1_q1, x2_q2, x3_q3). Only the rows of the calling process are written (the points must
    contain all points in the support of its rows; the others are skipped), so no exchange of assembly data is
    needed.

    Parameters
    ----------
        mat : 6d float array
            _data attribute of the StencilMatrix.

        starts_out, ends_out : 1d int arrays
            Starting and ending indices of the output space (codomain) on the process.

        pads_out, pads_in : 1d int arrays
            Paddings of the rows (output space) and of the columns (input space) of the StencilMatrix.

        npts_out, npts_in : 1d int arrays
            Number of basis functions of the output and of the input space in each direction.

        periods : 1d int array
            1 for a periodic direction, else 0.

        span_out1, span_out2, span_out3 : 1d int arrays
            Knot span indices of the output basis at the points of each direction.

        basis_out1, basis_out2, basis_out3 : 2d float arrays
            Values of the p + 1 non-zero output basis functions at the points, format (point, basis function).

        span_in1, span_in2, span_in3, basis_in1, basis_in2, basis_in3 :
            The same for the input basis.

        weight : 3d float array
            The weight at the points, format (point 1, point 2, point 3).
    """

    po1 = basis_out1.shape[1] - 1
    po2 = basis_out2.shape[1] - 1
    po3 = basis_out3.shape[1] - 1
    pi1 = basis_in1.shape[1] - 1
    pi2 = basis_in2.shape[1] - 1
    pi3 = basis_in3.shape[1] - 1

    for q1 in range(weight.shape[0]):
        for q2 in range(weight.shape[1]):
            for q3 in range(weight.shape[2]):
                w = weight[q1, q2, q3]
                if w == 0.0:
                    continue

                for a1 in range(po1 + 1):
                    # global row index
                    i1 = span_out1[q1] - po1 + a1
                    # local row index, periodic representative in the rows of the process (else skipped)
                    i1 = i1 - starts_out[0]
                    if periods[0] == 1:
                        i1 = i1 % npts_out[0]
                    if i1 < 0 or i1 > ends_out[0] - starts_out[0]:
                        continue
                    v1 = w * basis_out1[q1, a1]

                    for a2 in range(po2 + 1):
                        i2 = span_out2[q2] - po2 + a2
                        # local row index, periodic representative in the rows of the process (else skipped)
                        i2 = i2 - starts_out[1]
                        if periods[1] == 1:
                            i2 = i2 % npts_out[1]
                        if i2 < 0 or i2 > ends_out[1] - starts_out[1]:
                            continue
                        v2 = v1 * basis_out2[q2, a2]

                        for a3 in range(po3 + 1):
                            i3 = span_out3[q3] - po3 + a3
                            # local row index, periodic representative in the rows of the process (else skipped)
                            i3 = i3 - starts_out[2]
                            if periods[2] == 1:
                                i3 = i3 % npts_out[2]
                            if i3 < 0 or i3 > ends_out[2] - starts_out[2]:
                                continue
                            v3 = v2 * basis_out3[q3, a3]

                            for b1 in range(pi1 + 1):
                                # offset of the column, periodic representative next to the row
                                k1 = span_in1[q1] - pi1 + b1 - (i1 + starts_out[0])
                                if periods[0] == 1:
                                    if k1 > npts_in[0] // 2:
                                        k1 = k1 - npts_in[0]
                                    elif -k1 > npts_in[0] // 2:
                                        k1 = k1 + npts_in[0]
                                u1 = v3 * basis_in1[q1, b1]

                                for b2 in range(pi2 + 1):
                                    k2 = span_in2[q2] - pi2 + b2 - (i2 + starts_out[1])
                                    if periods[1] == 1:
                                        if k2 > npts_in[1] // 2:
                                            k2 = k2 - npts_in[1]
                                        elif -k2 > npts_in[1] // 2:
                                            k2 = k2 + npts_in[1]
                                    u2 = u1 * basis_in2[q2, b2]

                                    for b3 in range(pi3 + 1):
                                        k3 = span_in3[q3] - pi3 + b3 - (i3 + starts_out[2])
                                        if periods[2] == 1:
                                            if k3 > npts_in[2] // 2:
                                                k3 = k3 - npts_in[2]
                                            elif -k3 > npts_in[2] // 2:
                                                k3 = k3 + npts_in[2]

                                        mat[
                                            pads_out[0] + i1,
                                            pads_out[1] + i2,
                                            pads_out[2] + i3,
                                            pads_in[0] + k1,
                                            pads_in[1] + k2,
                                            pads_in[2] + k3,
                                        ] += u2 * basis_in3[q3, b3]
