__device__ void fill_vec_dev(
    int pi1, int pi2, int pi3,
    const double* bi1, const double* bi2, const double* bi3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    double* vec, int vn2, int vn3,
    double filling)
{
    for (int il1 = 0; il1 <= pi1; il1++) {
        int i1 = span1 + il1 - start0;
        double b1 = bi1[il1] * filling;
        for (int il2 = 0; il2 <= pi2; il2++) {
            int i2 = span2 + il2 - start1;
            double b2 = b1 * bi2[il2];
            for (int il3 = 0; il3 <= pi3; il3++) {
                int i3 = span3 + il3 - start2;
                double b3 = b2 * bi3[il3];
                atomicAdd(&vec[(size_t)i1*vn2*vn3 + (size_t)i2*vn3 + i3], b3);
            }
        }
    }
}

