// Port of filler_kernels.fill_mat_vec_pressure_full: like fill_mat_vec_dev
// but scatters into 6 matrix blocks (scaled by vx*vx, vx*vy, vx*vz, vy*vy,
// vy*vz, vz*vz) and 3 vector blocks (scaled by vx, vy, vz) in one pass.
__device__ void fill_mat_vec_pressure_full_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_13, double* mat_22, double* mat_23, double* mat_33,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double* vec_1, double* vec_2, double* vec_3, int vn2, int vn3,
    double filling_vec,
    double vx, double vy, double vz)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];
                size_t vidx = (size_t)i1*vn2*vn3 + (size_t)i2*vn3 + i3;
                double bv = b3 * filling_vec;
                atomicAdd(&vec_1[vidx], bv * vx);
                atomicAdd(&vec_2[vidx], bv * vy);
                atomicAdd(&vec_3[vidx], bv * vz);

                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_13[idx], b6 * vx * vz);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                            atomicAdd(&mat_23[idx], b6 * vy * vz);
                            atomicAdd(&mat_33[idx], b6 * vz * vz);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat_pressure_full: same as above minus the
// vector part (off-diagonal spatial blocks have no associated vector).
__device__ void fill_mat_pressure_full_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_13, double* mat_22, double* mat_23, double* mat_33,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double vx, double vy, double vz)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];
                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_13[idx], b6 * vx * vz);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                            atomicAdd(&mat_23[idx], b6 * vy * vz);
                            atomicAdd(&mat_33[idx], b6 * vz * vz);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat_vec_pressure: the "perp" (xy-plane only)
// variant -- 3 matrix blocks (vx*vx, vx*vy, vy*vy) and 2 vector blocks
// (vx, vy).
__device__ void fill_mat_vec_pressure_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_22,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double* vec_1, double* vec_2, int vn2, int vn3,
    double filling_vec,
    double vx, double vy)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];
                size_t vidx = (size_t)i1*vn2*vn3 + (size_t)i2*vn3 + i3;
                double bv = b3 * filling_vec;
                atomicAdd(&vec_1[vidx], bv * vx);
                atomicAdd(&vec_2[vidx], bv * vy);

                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                        }
                    }
                }
            }
        }
    }
}

// Port of filler_kernels.fill_mat_pressure: "perp" matrix-only variant.
__device__ void fill_mat_pressure_dev(
    int pi1, int pi2, int pi3, int pj1, int pj2, int pj3,
    const double* bi1, const double* bi2, const double* bi3,
    const double* bj1, const double* bj2, const double* bj3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    int pad0, int pad1, int pad2,
    double* mat_11, double* mat_12, double* mat_22,
    int d2, int d3, int d4, int d5, int d6,
    double filling_mat,
    double vx, double vy)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1];
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];
                for (int jl1 = 0; jl1 <= pj1; jl1++) {
                    int j1 = pad0 + jl1 - il1;
                    double b4 = b3 * bj1[jl1] * filling_mat;
                    for (int jl2 = 0; jl2 <= pj2; jl2++) {
                        int j2 = pad1 + jl2 - il2;
                        double b5 = b4 * bj2[jl2];
                        for (int jl3 = 0; jl3 <= pj3; jl3++) {
                            int j3 = pad2 + jl3 - il3;
                            double b6 = b5 * bj3[jl3];
                            size_t idx = (((((size_t)i1*d2+i2)*d3+i3)*d4+j1)*d5+j2)*d6+j3;
                            atomicAdd(&mat_11[idx], b6 * vx * vx);
                            atomicAdd(&mat_12[idx], b6 * vx * vy);
                            atomicAdd(&mat_22[idx], b6 * vy * vy);
                        }
                    }
                }
            }
        }
    }
}

