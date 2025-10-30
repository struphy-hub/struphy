from numpy import shape


# ========= kernel for integration in 1d ==================
def kernel_int_1d(nq1: "int", w1: "float[:]", mat_f: "float[:]") -> "float":
    f_loc = 0.0

    for q1 in range(nq1):
        f_loc += w1[q1] * mat_f[q1]

    return f_loc


# ========= kernel for integration in 2d ==================
def kernel_int_2d(nq1: "int", nq2: "int", w1: "float[:]", w2: "float[:]", mat_f: "float[:,:]") -> "float":
    f_loc = 0.0

    for q1 in range(nq1):
        for q2 in range(nq2):
            f_loc += w1[q1] * w2[q2] * mat_f[q1, q2]

    return f_loc


# ========= kernel for integration in 3d ==================
def kernel_int_3d(
    nq1: "int",
    nq2: "int",
    nq3: "int",
    w1: "float[:]",
    w2: "float[:]",
    w3: "float[:]",
    mat_f: "float[:,:,:]",
) -> "float":
    f_loc = 0.0

    for q1 in range(nq1):
        for q2 in range(nq2):
            for q3 in range(nq3):
                f_loc += w1[q1] * w2[q2] * w3[q3] * mat_f[q1, q2, q3]

    return f_loc


# ===========================================================================================================
#                                                   2d
# ===========================================================================================================

# ===========================================================================================================
#                                               line integrals
# ===========================================================================================================


# ========= kernel for integration along eta1 direction, reducing to a 2d array  ============================
def kernel_int_2d_eta1(
    subs1: "int[:]",
    subs_cum1: "int[:]",
    w1: "float[:,:]",
    mat_f: "float[:,:,:]",
    f_int: "float[:,:]",
):
    n1, n2 = shape(f_int)

    nq1 = shape(w1)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            value = 0.0

            for j1 in range(subs1[i1]):
                for q1 in range(nq1):
                    value += w1[i1 + j1 + subs_cum1[i1], q1] * mat_f[i1 + j1 + subs_cum1[i1], q1, i2]

            f_int[i1, i2] = value


# ========= kernel for integration along eta2 direction, reducing to a 2d array  ============================
def kernel_int_2d_eta2(
    subs2: "int[:]",
    subs_cum2: "int[:]",
    w2: "float[:,:]",
    mat_f: "float[:,:,:]",
    f_int: "float[:,:]",
):
    n1, n2 = shape(f_int)

    nq2 = shape(w2)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            value = 0.0

            for j2 in range(subs2[i2]):
                for q2 in range(nq2):
                    value += w2[i2 + j2 + subs_cum2[i2], q2] * mat_f[i1, i2 + j2 + subs_cum2[i2], q2]

            f_int[i1, i2] = value


# ========= kernel for integration along eta1 direction, reducing to a 2d array  ============================
def kernel_int_2d_eta1_old(w1: "float[:,:]", mat_f: "float[:,:,:]", f_int: "float[:,:]"):
    ne1, nq1, n2 = shape(mat_f)

    for ie1 in range(ne1):
        for i2 in range(n2):
            f_int[ie1, i2] = 0.0

            for q1 in range(nq1):
                f_int[ie1, i2] += w1[ie1, q1] * mat_f[ie1, q1, i2]


# ========= kernel for integration along eta2 direction, reducing to a 2d array  ============================
def kernel_int_2d_eta2_old(w2: "float[:,:]", mat_f: "float[:,:,:]", f_int: "float[:,:]"):
    n1, ne2, nq2 = shape(mat_f)

    for i1 in range(n1):
        for ie2 in range(ne2):
            f_int[i1, ie2] = 0.0

            for q2 in range(nq2):
                f_int[i1, ie2] += w2[ie2, q2] * mat_f[i1, ie2, q2]


# ===========================================================================================================
#                                            surface integrals
# ===========================================================================================================


# ========= kernel for integration in eta1-eta2 plane, reducing to a 2d array  ==============================
def kernel_int_2d_eta1_eta2(
    subs1: "int[:]",
    subs2: "int[:]",
    subs_cum1: "int[:]",
    subs_cum2: "int[:]",
    w1: "float[:,:]",
    w2: "float[:,:]",
    mat_f: "float[:,:,:,:]",
    f_int: "float[:,:]",
):
    n1, n2 = shape(f_int)

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            value = 0.0

            for j1 in range(subs1[i1]):
                for j2 in range(subs2[i2]):
                    for q1 in range(nq1):
                        for q2 in range(nq2):
                            wvol = w1[i1 + j1 + subs_cum1[i1], q1] * w2[i2 + j2 + subs_cum2[i2], q2]

                            value += wvol * mat_f[i1 + j1 + subs_cum1[i1], q1, i2 + j2 + subs_cum2[i2], q2]

            f_int[i1, i2] = value


# ========= kernel for integration in eta1-eta2 plane, reducing to a 2d array  =======================
def kernel_int_2d_eta1_eta2_old(w1: "float[:,:]", w2: "float[:,:]", mat_f: "float[:,:,:,:]", f_int: "float[:,:]"):
    ne1, nq1, ne2, nq2 = shape(mat_f)

    for ie1 in range(ne1):
        for ie2 in range(ne2):
            f_int[ie1, ie2] = 0.0

            for q1 in range(nq1):
                for q2 in range(nq2):
                    f_int[ie1, ie2] += w1[ie1, q1] * w2[ie2, q2] * mat_f[ie1, q1, ie2, q2]


# ===========================================================================================================
#                                                   3d
# ===========================================================================================================

# ===========================================================================================================
#                                               line integrals
# ===========================================================================================================


# ========= kernel for integration along eta1 direction, reducing to a 3d array  ============================
def kernel_int_3d_eta1(
    subs1: "int[:]",
    subs_cum1: "int[:]",
    w1: "float[:,:]",
    mat_f: "float[:,:,:,:]",
    f_int: "float[:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq1 = shape(w1)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0

                for j1 in range(subs1[i1]):
                    for q1 in range(nq1):
                        value += w1[i1 + j1 + subs_cum1[i1], q1] * mat_f[i1 + j1 + subs_cum1[i1], q1, i2, i3]

                f_int[i1, i2, i3] = value


def kernel_int_3d_eta1_transpose(
    subs1: "int[:]",
    subs_cum1: "int[:]",
    w1: "float[:,:]",
    f_int: "float[:,:,:]",
    mat_f: "float[:,:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq1 = shape(w1)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                for j1 in range(subs1[i1]):
                    for q1 in range(nq1):
                        mat_f[i1 + j1 + subs_cum1[i1], q1, i2, i3] = w1[i1 + j1 + subs_cum1[i1], q1] * f_int[i1, i2, i3]


# ============================================================================================================


# ========= kernel for integration along eta2 direction, reducing to a 3d array  ============================
def kernel_int_3d_eta2(
    subs2: "int[:]",
    subs_cum2: "int[:]",
    w2: "float[:,:]",
    mat_f: "float[:,:,:,:]",
    f_int: "float[:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq2 = shape(w2)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0

                for j2 in range(subs2[i2]):
                    for q2 in range(nq2):
                        value += w2[i2 + j2 + subs_cum2[i2], q2] * mat_f[i1, i2 + j2 + subs_cum2[i2], q2, i3]

                f_int[i1, i2, i3] = value


def kernel_int_3d_eta2_transpose(
    subs2: "int[:]",
    subs_cum2: "int[:]",
    w2: "float[:,:]",
    f_int: "float[:,:,:]",
    mat_f: "float[:,:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq2 = shape(w2)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                for j2 in range(subs2[i2]):
                    for q2 in range(nq2):
                        mat_f[i1, i2 + j2 + subs_cum2[i2], q2, i3] = w2[i2 + j2 + subs_cum2[i2], q2] * f_int[i1, i2, i3]


# ============================================================================================================


# ========= kernel for integration along eta3 direction, reducing to a 3d array  ============================
def kernel_int_3d_eta3(
    subs3: "int[:]",
    subs_cum3: "int[:]",
    w3: "float[:,:]",
    mat_f: "float[:,:,:,:]",
    f_int: "float[:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0

                for j3 in range(subs3[i3]):
                    for q3 in range(nq3):
                        value += w3[i3 + j3 + subs_cum3[i3], q3] * mat_f[i1, i2, i3 + j3 + subs_cum3[i3], q3]

                f_int[i1, i2, i3] = value


def kernel_int_3d_eta3_transpose(
    subs3: "int[:]",
    subs_cum3: "int[:]",
    w3: "float[:,:]",
    f_int: "float[:,:,:]",
    mat_f: "float[:,:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                for j3 in range(subs3[i3]):
                    for q3 in range(nq3):
                        mat_f[i1, i2, i3 + j3 + subs_cum3[i3], q3] = w3[i3 + j3 + subs_cum3[i3], q3] * f_int[i1, i2, i3]


# ============================================================================================================


# ========= kernel for integration along eta1 direction, reducing to a 3d array  ============================
def kernel_int_3d_eta1_old(w1: "float[:,:]", mat_f: "float[:,:,:,:]", f_int: "float[:,:,:]"):
    ne1, nq1, n2, n3 = shape(mat_f)

    for ie1 in range(ne1):
        for i2 in range(n2):
            for i3 in range(n3):
                f_int[ie1, i2, i3] = 0.0

                for q1 in range(nq1):
                    f_int[ie1, i2, i3] += w1[ie1, q1] * mat_f[ie1, q1, i2, i3]


# ========= kernel for integration along eta2 direction, reducing to a 3d array  ============================
def kernel_int_3d_eta2_old(w2: "float[:,:]", mat_f: "float[:,:,:,:]", f_int: "float[:,:,:]"):
    n1, ne2, nq2, n3 = shape(mat_f)

    for i1 in range(n1):
        for ie2 in range(ne2):
            for i3 in range(n3):
                f_int[i1, ie2, i3] = 0.0

                for q2 in range(nq2):
                    f_int[i1, ie2, i3] += w2[ie2, q2] * mat_f[i1, ie2, q2, i3]


# ========= kernel for integration along eta3 direction, reducing to a 3d array  ============================
def kernel_int_3d_eta3_old(w3: "float[:,:]", mat_f: "float[:,:,:,:]", f_int: "float[:,:,:]"):
    n1, n2, ne3, nq3 = shape(mat_f)

    for i1 in range(n1):
        for i2 in range(n2):
            for ie3 in range(ne3):
                f_int[i1, i2, ie3] = 0.0

                for q3 in range(nq3):
                    f_int[i1, i2, ie3] += w3[ie3, q3] * mat_f[i1, i2, ie3, q3]


# ===========================================================================================================
#                                            surface integrals
# ===========================================================================================================


# ========= kernel for integration in eta2-eta3 plane, reducing to a 3d array  ==============================
def kernel_int_3d_eta2_eta3(
    subs2: "int[:]",
    subs3: "int[:]",
    subs_cum2: "int[:]",
    subs_cum3: "int[:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    mat_f: "float[:,:,:,:,:]",
    f_int: "float[:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0

                for j2 in range(subs2[i2]):
                    for j3 in range(subs3[i3]):
                        for q2 in range(nq2):
                            for q3 in range(nq3):
                                wvol = w2[i2 + j2 + subs_cum2[i2], q2] * w3[i3 + j3 + subs_cum3[i3], q3]

                                value += wvol * mat_f[i1, i2 + j2 + subs_cum2[i2], q2, i3 + j3 + subs_cum3[i3], q3]

                f_int[i1, i2, i3] = value


def kernel_int_3d_eta2_eta3_transpose(
    subs2: "int[:]",
    subs3: "int[:]",
    subs_cum2: "int[:]",
    subs_cum3: "int[:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    f_int: "float[:,:,:]",
    mat_f: "float[:,:,:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                for j2 in range(subs2[i2]):
                    for j3 in range(subs3[i3]):
                        for q2 in range(nq2):
                            for q3 in range(nq3):
                                wvol = w2[i2 + j2 + subs_cum2[i2], q2] * w3[i3 + j3 + subs_cum3[i3], q3]

                                mat_f[i1, i2 + j2 + subs_cum2[i2], q2, i3 + j3 + subs_cum3[i3], q3] = (
                                    wvol * f_int[i1, i2, i3]
                                )


# ========= kernel for integration in eta1-eta3 plane, reducing to a 3d array  ==============================
def kernel_int_3d_eta1_eta3(
    subs1: "int[:]",
    subs3: "int[:]",
    subs_cum1: "int[:]",
    subs_cum3: "int[:]",
    w1: "float[:,:]",
    w3: "float[:,:]",
    mat_f: "float[:,:,:,:,:]",
    f_int: "float[:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq1 = shape(w1)[1]
    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0

                for j1 in range(subs1[i1]):
                    for j3 in range(subs3[i3]):
                        for q1 in range(nq1):
                            for q3 in range(nq3):
                                wvol = w1[i1 + j1 + subs_cum1[i1], q1] * w3[i3 + j3 + subs_cum3[i3], q3]

                                value += wvol * mat_f[i1 + j1 + subs_cum1[i1], q1, i2, i3 + j3 + subs_cum3[i3], q3]

                f_int[i1, i2, i3] = value


def kernel_int_3d_eta1_eta3_transpose(
    subs1: "int[:]",
    subs3: "int[:]",
    subs_cum1: "int[:]",
    subs_cum3: "int[:]",
    w1: "float[:,:]",
    w3: "float[:,:]",
    f_int: "float[:,:,:]",
    mat_f: "float[:,:,:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq1 = shape(w1)[1]
    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                for j1 in range(subs1[i1]):
                    for j3 in range(subs3[i3]):
                        for q1 in range(nq1):
                            for q3 in range(nq3):
                                wvol = w1[i1 + j1 + subs_cum1[i1], q1] * w3[i3 + j3 + subs_cum3[i3], q3]

                                mat_f[i1 + j1 + subs_cum1[i1], q1, i2, i3 + j3 + subs_cum3[i3], q3] = (
                                    wvol * f_int[i1, i2, i3]
                                )


# ========= kernel for integration in eta1-eta2 plane, reducing to a 3d array  ==============================
def kernel_int_3d_eta1_eta2(
    subs1: "int[:]",
    subs2: "int[:]",
    subs_cum1: "int[:]",
    subs_cum2: "int[:]",
    w1: "float[:,:]",
    w2: "float[:,:]",
    mat_f: "float[:,:,:,:,:]",
    f_int: "float[:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0

                for j1 in range(subs1[i1]):
                    for j2 in range(subs2[i2]):
                        for q1 in range(nq1):
                            for q2 in range(nq2):
                                wvol = w1[i1 + j1 + subs_cum1[i1], q1] * w2[i2 + j2 + subs_cum2[i2], q2]

                                value += wvol * mat_f[i1 + j1 + subs_cum1[i1], q1, i2 + j2 + subs_cum2[i2], q2, i3]

                f_int[i1, i2, i3] = value


def kernel_int_3d_eta1_eta2_transpose(
    subs1: "int[:]",
    subs2: "int[:]",
    subs_cum1: "int[:]",
    subs_cum2: "int[:]",
    w1: "float[:,:]",
    w2: "float[:,:]",
    f_int: "float[:,:,:]",
    mat_f: "float[:,:,:,:,:]",
):
    n1, n2, n3 = shape(f_int)

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                for j1 in range(subs1[i1]):
                    for j2 in range(subs2[i2]):
                        for q1 in range(nq1):
                            for q2 in range(nq2):
                                wvol = w1[i1 + j1 + subs_cum1[i1], q1] * w2[i2 + j2 + subs_cum2[i2], q2]

                                mat_f[i1 + j1 + subs_cum1[i1], q1, i2 + j2 + subs_cum2[i2], q2, i3] = (
                                    wvol * f_int[i1, i2, i3]
                                )


# ========= kernel for integration in eta2-eta3 plane, reducing to a 3d array  ==============================
def kernel_int_3d_eta2_eta3_old(w2: "float[:,:]", w3: "float[:,:]", mat_f: "float[:,:,:,:,:]", f_int: "float[:,:,:]"):
    n1, ne2, nq2, ne3, nq3 = shape(mat_f)

    for i1 in range(n1):
        for ie2 in range(ne2):
            for ie3 in range(ne3):
                f_int[i1, ie2, ie3] = 0.0

                for q2 in range(nq2):
                    for q3 in range(nq3):
                        f_int[i1, ie2, ie3] += w2[ie2, q2] * w3[ie3, q3] * mat_f[i1, ie2, q2, ie3, q3]


# ========= kernel for integration eta1-eta3 plane, reducing to a 3d array  ================================
def kernel_int_3d_eta1_eta3_old(w1: "float[:,:]", w3: "float[:,:]", mat_f: "float[:,:,:,:,:]", f_int: "float[:,:,:]"):
    ne1, nq1, n2, ne3, nq3 = shape(mat_f)

    for ie1 in range(ne1):
        for i2 in range(n2):
            for ie3 in range(ne3):
                f_int[ie1, i2, ie3] = 0.0

                for q1 in range(nq1):
                    for q3 in range(nq3):
                        f_int[ie1, i2, ie3] += w1[ie1, q1] * w3[ie3, q3] * mat_f[ie1, q1, i2, ie3, q3]


# ========= kernel for integration in eta1-eta2 plane, reducing to a 3d array  ============================
def kernel_int_3d_eta1_eta2_old(w1: "float[:,:]", w2: "float[:,:]", mat_f: "float[:,:,:,:,:]", f_int: "float[:,:,:]"):
    ne1, nq1, ne2, nq2, n3 = shape(mat_f)

    for ie1 in range(ne1):
        for ie2 in range(ne2):
            for i3 in range(n3):
                f_int[ie1, ie2, i3] = 0.0

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        f_int[ie1, ie2, i3] += w1[ie1, q1] * w2[ie2, q2] * mat_f[ie1, q1, ie2, q2, i3]


# ===========================================================================================================
#                                            cell integrals
# ===========================================================================================================


# ========= kernel for integration in eta1-eta2-eta3 cell, reducing to a 3d array  ==============================
def kernel_int_3d_eta1_eta2_eta3(
    subs1: "int[:]",
    subs2: "int[:]",
    subs3: "int[:]",
    subs_cum1: "int[:]",
    subs_cum2: "int[:]",
    subs_cum3: "int[:]",
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    mat_f: "float[:,:,:,:,:,:]",
    f_int: "float[:,:,:]",
):
    n1 = len(subs1)
    n2 = len(subs2)
    n3 = len(subs3)

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                value = 0.0

                for j1 in range(subs1[i1]):
                    for j2 in range(subs2[i2]):
                        for j3 in range(subs3[i3]):
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        wvol = (
                                            w1[i1 + j1 + subs_cum1[i1], q1]
                                            * w2[i2 + j2 + subs_cum2[i2], q2]
                                            * w3[i3 + j3 + subs_cum3[i3], q3]
                                        )

                                        value += (
                                            wvol
                                            * mat_f[
                                                i1 + j1 + subs_cum1[i1],
                                                q1,
                                                i2 + j2 + subs_cum2[i2],
                                                q2,
                                                i3 + j3 + subs_cum3[i3],
                                                q3,
                                            ]
                                        )

                f_int[i1, i2, i3] = value


def kernel_int_3d_eta1_eta2_eta3_transpose(
    subs1: "int[:]",
    subs2: "int[:]",
    subs3: "int[:]",
    subs_cum1: "int[:]",
    subs_cum2: "int[:]",
    subs_cum3: "int[:]",
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    f_int: "float[:,:,:]",
    mat_f: "float[:,:,:,:,:,:]",
):
    n1 = len(subs1)
    n2 = len(subs2)
    n3 = len(subs3)

    nq1 = shape(w1)[1]
    nq2 = shape(w2)[1]
    nq3 = shape(w3)[1]

    for i1 in range(n1):
        for i2 in range(n2):
            for i3 in range(n3):
                for j1 in range(subs1[i1]):
                    for j2 in range(subs2[i2]):
                        for j3 in range(subs3[i3]):
                            for q1 in range(nq1):
                                for q2 in range(nq2):
                                    for q3 in range(nq3):
                                        wvol = (
                                            w1[i1 + j1 + subs_cum1[i1], q1]
                                            * w2[i2 + j2 + subs_cum2[i2], q2]
                                            * w3[i3 + j3 + subs_cum3[i3], q3]
                                        )

                                        mat_f[
                                            i1 + j1 + subs_cum1[i1],
                                            q1,
                                            i2 + j2 + subs_cum2[i2],
                                            q2,
                                            i3 + j3 + subs_cum3[i3],
                                            q3,
                                        ] = wvol * f_int[i1, i2, i3]


# ========= kernel for integration in eta1-eta2-eta3 cell, reducing to a 3d array  =======================
def kernel_int_3d_eta1_eta2_eta3_old(
    w1: "float[:,:]",
    w2: "float[:,:]",
    w3: "float[:,:]",
    mat_f: "float[:,:,:,:,:,:]",
    f_int: "float[:,:,:]",
):
    ne1, nq1, ne2, nq2, ne3, nq3 = shape(mat_f)

    for ie1 in range(ne1):
        for ie2 in range(ne2):
            for ie3 in range(ne3):
                f_int[ie1, ie2, ie3] = 0.0

                for q1 in range(nq1):
                    for q2 in range(nq2):
                        for q3 in range(nq3):
                            f_int[ie1, ie2, ie3] += (
                                w1[ie1, q1] * w2[ie2, q2] * w3[ie3, q3] * mat_f[ie1, q1, ie2, q2, ie3, q3]
                            )
