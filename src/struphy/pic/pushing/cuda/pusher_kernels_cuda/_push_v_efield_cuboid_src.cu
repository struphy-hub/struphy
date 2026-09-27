#define MAXP 8

__device__ int find_span_dev(const double* t, int p, int len_t, double eta)
{
    int low = p;
    int high = len_t - 1 - p;

    if (eta <= t[low]) return low;
    if (eta >= t[high]) return high - 1;

    int span = (low + high) / 2;
    while (eta < t[span] || eta >= t[span + 1]) {
        if (eta < t[span]) high = span;
        else low = span;
        span = (low + high) / 2;
    }
    return span;
}

// Combined N-spline (bn, p+1 values) and D-spline (bd, p values) evaluation,
// matching struphy.bsplines.bsplines_kernels.b_d_splines_slim exactly.
__device__ void b_d_splines_dev(const double* t, int p, double eta, int span, double* bn, double* bd)
{
    double left[MAXP];
    double right[MAXP];
    int pd = p - 1;

    for (int i = 0; i <= p; i++) bn[i] = 0.0;
    for (int i = 0; i < p; i++) bd[i] = 0.0;
    bn[0] = 1.0;

    for (int j = 0; j < p; j++) {
        left[j] = eta - t[span - j];
        right[j] = t[span + 1 + j] - eta;
        double saved = 0.0;

        if (j == p - 1) {
            for (int il = 0; il <= pd; il++) {
                bd[pd - il] = (double)p / (t[span - il + p] - t[span - il]) * bn[pd - il];
            }
        }

        for (int r = 0; r <= j; r++) {
            double temp = bn[r] / (right[r] + left[j - r]);
            bn[r] = saved + right[r] * temp;
            saved = left[j - r] * temp;
        }
        bn[j + 1] = saved;
    }
}

extern "C" __global__
void push_v_with_efield_cuboid(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1,
    const int p2,
    const int p3,
    const double* tn1,
    const int len_tn1,
    const double* tn2,
    const int len_tn2,
    const double* tn3,
    const int len_tn3,
    const int start0,
    const int start1,
    const int start2,
    const double* e1_1,
    const int n2x1,
    const int n3x1,
    const double* e1_2,
    const int n2x2,
    const int n3x2,
    const double* e1_3,
    const int n2x3,
    const int n3x3,
    const double sx,
    const double sy,
    const double sz,
    const double dt_const)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;

    // skip holes and ghost/boundary particles, matching Particles.valid_mks
    if (row[0] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double eta1 = row[0];
    const double eta2 = row[1];
    const double eta3 = row[2];

    double bn1[MAXP + 1], bd1[MAXP];
    double bn2[MAXP + 1], bd2[MAXP];
    double bn3[MAXP + 1], bd3[MAXP];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    // e_form[0]: D-spline in direction 1, N-splines in directions 2, 3
    double e_form0 = 0.0;
    for (int il1 = 0; il1 < p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form0 += e1_1[(size_t)i1 * n2x1 * n3x1 + (size_t)i2 * n3x1 + i3] * bd1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }

    // e_form[1]: N-spline in direction 1, D-spline in direction 2, N-spline in direction 3
    double e_form1 = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 < p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form1 += e1_2[(size_t)i1 * n2x2 * n3x2 + (size_t)i2 * n3x2 + i3] * bn1[il1] * bd2[il2] * bn3[il3];
            }
        }
    }

    // e_form[2]: N-splines in directions 1, 2, D-spline in direction 3
    double e_form2 = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 < p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form2 += e1_3[(size_t)i1 * n2x3 * n3x3 + (size_t)i2 * n3x3 + i3] * bn1[il1] * bn2[il2] * bd3[il3];
            }
        }
    }

    // Cartesian E-field is DF^-T @ e_form; for Cuboid, DF is diag(sx^-1, sy^-1, sz^-1)
    // so DF^-T is diag(sx, sy, sz) -- same convention as push_eta_stage_cuboid's scale.
    row[3] += dt_const * sx * e_form0;
    row[4] += dt_const * sy * e_form1;
    row[5] += dt_const * sz * e_form2;
}

