#define MAXP 8

__device__ void matrix_inv_dev(const double* a, double* b)
{
    double det_a = a[0]*(a[4]*a[8] - a[5]*a[7])
                 - a[1]*(a[3]*a[8] - a[5]*a[6])
                 + a[2]*(a[3]*a[7] - a[4]*a[6]);

    b[0] = (a[4]*a[8] - a[7]*a[5]) / det_a;
    b[1] = (a[7]*a[2] - a[1]*a[8]) / det_a;
    b[2] = (a[1]*a[5] - a[4]*a[2]) / det_a;
    b[3] = (a[5]*a[6] - a[8]*a[3]) / det_a;
    b[4] = (a[8]*a[0] - a[2]*a[6]) / det_a;
    b[5] = (a[2]*a[3] - a[5]*a[0]) / det_a;
    b[6] = (a[3]*a[7] - a[6]*a[4]) / det_a;
    b[7] = (a[6]*a[1] - a[0]*a[7]) / det_a;
    b[8] = (a[0]*a[4] - a[3]*a[1]) / det_a;
}

// c = a^T @ b  (used for both DF^-1 @ v and DF^-T @ e_form: pass dfinv or
// its transpose accordingly -- here we need dfinv @ v (not transposed) for
// push_eta_stage, and dfinvT @ e_form for push_v_with_efield, so both a
// plain and a transposed matvec are provided).
__device__ void matvec_dev(const double* a, const double* v, double* out)
{
    out[0] = a[0]*v[0] + a[1]*v[1] + a[2]*v[2];
    out[1] = a[3]*v[0] + a[4]*v[1] + a[5]*v[2];
    out[2] = a[6]*v[0] + a[7]*v[1] + a[8]*v[2];
}

// c = a @ b, 3x3 row-major matrices.
__device__ void matmat_dev(const double* a, const double* b, double* c)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            c[3*i+j] = a[3*i+0]*b[0*3+j] + a[3*i+1]*b[1*3+j] + a[3*i+2]*b[2*3+j];
        }
    }
}

__device__ void matvecT_dev(const double* a, const double* v, double* out)
{
    out[0] = a[0]*v[0] + a[3]*v[1] + a[6]*v[2];
    out[1] = a[1]*v[0] + a[4]*v[1] + a[7]*v[2];
    out[2] = a[2]*v[0] + a[5]*v[1] + a[8]*v[2];
}

// df_out is row-major 3x3 (df_out[3*i+j] = dF_i/deta_j), matching
// struphy.geometry.mappings_kernels.cuboid_df / colella_df exactly.
__device__ void cuboid_df_dev(const double* params, double* df_out)
{
    // params = (l1, r1, l2, r2, l3, r3)
    for (int k = 0; k < 9; k++) df_out[k] = 0.0;
    df_out[0] = params[1] - params[0];
    df_out[4] = params[3] - params[2];
    df_out[8] = params[5] - params[4];
}

__device__ void colella_df_dev(double eta1, double eta2, const double* params, double* df_out)
{
    // params = (Lx, Ly, alpha, Lz)
    const double lx = params[0], ly = params[1], alpha = params[2], lz = params[3];
    const double twopi = 6.283185307179586;
    const double s1 = sin(twopi * eta1), c1 = cos(twopi * eta1);
    const double s2 = sin(twopi * eta2), c2 = cos(twopi * eta2);

    df_out[0] = lx * (1.0 + alpha * c1 * s2 * twopi);
    df_out[1] = lx * alpha * s1 * c2 * twopi;
    df_out[2] = 0.0;
    df_out[3] = ly * alpha * c1 * s2 * twopi;
    df_out[4] = ly * (1.0 + alpha * s1 * c2 * twopi);
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}

__device__ void orthogonal_df_dev(double eta1, double eta2, const double* params, double* df_out)
{
    // params = (Lx, Ly, alpha, Lz)
    const double lx = params[0], ly = params[1], alpha = params[2], lz = params[3];
    const double twopi = 6.283185307179586;

    for (int k = 0; k < 9; k++) df_out[k] = 0.0;
    df_out[0] = lx * (1.0 + alpha * cos(twopi * eta1) * twopi);
    df_out[4] = ly * (1.0 + alpha * cos(twopi * eta2) * twopi);
    df_out[8] = lz;
}

__device__ void hollow_cyl_df_dev(double eta1, double eta2, const double* params, double* df_out)
{
    // params = (a1, a2, Lz, poc); faithful port of
    // struphy.geometry.mappings_kernels.hollow_cyl_df, including its
    // existing df_out[0,0]/df_out[1,0] not dividing eta2's argument by poc
    // (unlike f_out and every other entry here) -- not "fixed" here, since
    // this is a port, not a bugfix.
    const double a1 = params[0], a2 = params[1], lz = params[2], poc = params[3];
    const double twopi = 6.283185307179586;
    const double da = a2 - a1;
    const double r = a1 + eta1 * da;

    df_out[0] = da * cos(twopi * eta2);
    df_out[1] = -twopi / poc * r * sin(twopi * eta2 / poc);
    df_out[2] = 0.0;
    df_out[3] = da * sin(twopi * eta2);
    df_out[4] = twopi / poc * r * cos(twopi * eta2 / poc);
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}

__device__ void powered_ellipse_df_dev(double eta1, double eta2, const double* params, double* df_out)
{
    // params = (rx, ry, Lz, s)
    const double rx = params[0], ry = params[1], lz = params[2], s = params[3];
    const double twopi = 6.283185307179586;
    const double c2 = cos(twopi * eta2), s2 = sin(twopi * eta2);
    const double e_sm1 = pow(eta1, s - 1.0);
    const double e_s = pow(eta1, s);

    df_out[0] = e_sm1 * rx * c2;
    df_out[1] = -twopi * e_s * rx * s2;
    df_out[2] = 0.0;
    df_out[3] = e_sm1 * ry * s2;
    df_out[4] = twopi * e_s * ry * c2;
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}

__device__ void hollow_torus_df_dev(double eta1, double eta2, double eta3, const double* params, double* df_out)
{
    // params = (a1, a2, R0, sfl, pol_period, tor_period)
    const double a1 = params[0], a2 = params[1], r0 = params[2];
    const double sfl = params[3], pol_period = params[4], tor_period = params[5];
    const double pi = 3.14159265358979323846;
    const double twopi = 6.283185307179586;
    const double da = a2 - a1;

    if (sfl == 1.0) {
        const double r = a1 + da * eta1;
        const double eps = r / r0;
        const double eps_p = da / r0;
        const double tpe = tan(pi * eta2);
        const double cpe = cos(pi * eta2);
        const double tpe_p = pi / (cpe * cpe);
        const double g = sqrt((1.0 + eps) / (1.0 - eps));
        const double g_p = 1.0 / (2.0 * g) * (eps_p * (1.0 - eps) + (1.0 + eps) * eps_p) / ((1.0 - eps) * (1.0 - eps));
        const double theta = 2.0 * atan(g * tpe);
        const double denom = 1.0 + (g * tpe) * (g * tpe);
        const double dtheta_deta1 = 2.0 / denom * g_p * tpe;
        const double dtheta_deta2 = 2.0 / denom * g * tpe_p;
        const double ct = cos(theta), st = sin(theta);
        const double cf = cos(twopi * eta3 / tor_period), sf = sin(twopi * eta3 / tor_period);

        df_out[0] = (da * ct - r * st * dtheta_deta1) * cf;
        df_out[1] = -r * st * dtheta_deta2 * cf;
        df_out[2] = -twopi / tor_period * (r * ct + r0) * sf;

        df_out[3] = (da * ct - r * st * dtheta_deta1) * (-1.0) * sf;
        df_out[4] = -r * st * dtheta_deta2 * (-1.0) * sf;
        df_out[5] = twopi / tor_period * (r * ct + r0) * (-1.0) * cf;

        df_out[6] = da * st + r * ct * dtheta_deta1;
        df_out[7] = r * ct * dtheta_deta2;
        df_out[8] = 0.0;
    } else {
        const double r = a1 + eta1 * da;
        const double cp = cos(twopi * eta2 / pol_period), sp = sin(twopi * eta2 / pol_period);
        const double cf = cos(twopi * eta3 / tor_period), sf = sin(twopi * eta3 / tor_period);

        df_out[0] = da * cp * cf;
        df_out[1] = -twopi / pol_period * r * sp * cf;
        df_out[2] = -twopi / tor_period * (r * cp + r0) * sf;

        df_out[3] = da * cp * (-1.0) * sf;
        df_out[4] = -twopi / pol_period * r * sp * (-1.0) * sf;
        df_out[5] = (r * cp + r0) * (-1.0) * cf * twopi / tor_period;

        df_out[6] = da * sp;
        df_out[7] = r * cp * twopi / pol_period;
        df_out[8] = 0.0;
    }
}

__device__ void shafranov_shift_df_dev(double eta1, double eta2, const double* params, double* df_out)
{
    // params = (rx, ry, Lz, delta)
    const double rx = params[0], ry = params[1], lz = params[2], de = params[3];
    const double twopi = 6.283185307179586;
    const double c2 = cos(twopi * eta2), s2 = sin(twopi * eta2);

    df_out[0] = rx * c2 - 2.0 * eta1 * rx * de;
    df_out[1] = -twopi * (eta1 * rx) * s2;
    df_out[2] = 0.0;
    df_out[3] = ry * s2;
    df_out[4] = twopi * (eta1 * ry) * c2;
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}

__device__ void shafranov_sqrt_df_dev(double eta1, double eta2, const double* params, double* df_out)
{
    // params = (rx, ry, Lz, delta)
    const double rx = params[0], ry = params[1], lz = params[2], de = params[3];
    const double twopi = 6.283185307179586;
    const double c2 = cos(twopi * eta2), s2 = sin(twopi * eta2);

    df_out[0] = rx * c2 - 0.5 / sqrt(eta1) * rx * de;
    df_out[1] = -twopi * (eta1 * rx) * s2;
    df_out[2] = 0.0;
    df_out[3] = ry * s2;
    df_out[4] = twopi * (eta1 * ry) * c2;
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}

__device__ void shafranov_dshaped_df_dev(double eta1, double eta2, const double* params, double* df_out)
{
    // params = (R0, Lz, delta_x, delta_y, delta_gs, epsilon_gs, kappa_gs)
    const double r0 = params[0], lz = params[1], dx = params[2], dy = params[3];
    const double dg = params[4], eg = params[5], kg = params[6];
    const double pi = 3.14159265358979323846;
    const double twopi = 6.283185307179586;
    const double asin_dg = asin(dg);
    const double s2 = sin(twopi * eta2), c2 = cos(twopi * eta2);
    const double phase = eta1 * s2 * asin_dg + twopi * eta2;

    df_out[0] = r0 * (
        -2.0 * dx * eta1
        - eg * eta1 * s2 * asin_dg * sin(phase)
        + eg * cos(phase)
    );
    df_out[1] = -r0 * eg * eta1 * (twopi * eta1 * c2 * asin_dg + twopi) * sin(phase);
    df_out[2] = 0.0;
    df_out[3] = r0 * (-2.0 * dy * eta1 + eg * kg * s2);
    df_out[4] = twopi * r0 * eg * eta1 * kg * c2;
    df_out[5] = 0.0;
    df_out[6] = 0.0;
    df_out[7] = 0.0;
    df_out[8] = lz;
}

// Returns 1 if kind_map is supported and df_out was filled, 0 otherwise.
__device__ int df_dispatch_dev(int kind_map, double eta1, double eta2, double eta3,
                                const double* params, double* df_out)
{
    if (kind_map == 10) { cuboid_df_dev(params, df_out); return 1; }
    if (kind_map == 11) { orthogonal_df_dev(eta1, eta2, params, df_out); return 1; }
    if (kind_map == 12) { colella_df_dev(eta1, eta2, params, df_out); return 1; }
    if (kind_map == 20) { hollow_cyl_df_dev(eta1, eta2, params, df_out); return 1; }
    if (kind_map == 21) { powered_ellipse_df_dev(eta1, eta2, params, df_out); return 1; }
    if (kind_map == 22) { hollow_torus_df_dev(eta1, eta2, eta3, params, df_out); return 1; }
    if (kind_map == 30) { shafranov_shift_df_dev(eta1, eta2, params, df_out); return 1; }
    if (kind_map == 31) { shafranov_sqrt_df_dev(eta1, eta2, params, df_out); return 1; }
    if (kind_map == 32) { shafranov_dshaped_df_dev(eta1, eta2, params, df_out); return 1; }
    return 0;
}

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

// Same as pusher_kernels_cuda.py's push_v_with_efield_cuboid's b_d_splines_dev,
// duplicated here because each cp.RawKernel source string is compiled
// independently (no cross-source linking).
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

__device__ double det3_dev(const double* a)
{
    return a[0]*(a[4]*a[8] - a[5]*a[7])
         - a[1]*(a[3]*a[8] - a[5]*a[6])
         + a[2]*(a[3]*a[7] - a[4]*a[6]);
}

__device__ void cross_dev(const double* a, const double* b, double* out)
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ double dot3_dev(const double* a, const double* b)
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Single-point evaluation of a Derham 2-form spline, matching
// struphy.bsplines.evaluation_kernels_3d.eval_2form_spline_mpi (N-D-D /
// D-N-D / D-D-N tensor-product sums, the dual basis combination of the
// 1-form evaluation in push_v_with_efield_general above).
__device__ void eval_2form_dev(
    int p1, int p2, int p3,
    const double* bn1, const double* bd1,
    const double* bn2, const double* bd2,
    const double* bn3, const double* bd3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    const double* c1, int n2x1, int n3x1,
    const double* c2, int n2x2, int n3x2,
    const double* c3, int n2x3, int n3x3,
    double* out)
{
    out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;

    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 < p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 < p3; il3++) {
                int i3 = span3 + il3 - start2;
                out[0] += c1[(size_t)i1 * n2x1 * n3x1 + (size_t)i2 * n3x1 + i3] * bn1[il1] * bd2[il2] * bd3[il3];
            }
        }
    }
    for (int il1 = 0; il1 < p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 < p3; il3++) {
                int i3 = span3 + il3 - start2;
                out[1] += c2[(size_t)i1 * n2x2 * n3x2 + (size_t)i2 * n3x2 + i3] * bd1[il1] * bn2[il2] * bd3[il3];
            }
        }
    }
    for (int il1 = 0; il1 < p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 < p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                out[2] += c3[(size_t)i1 * n2x3 * n3x3 + (size_t)i2 * n3x3 + i3] * bd1[il1] * bd2[il2] * bn3[il3];
            }
        }
    }
}

extern "C" __global__
void push_eta_stage_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int first_free_idx,
    const int kind_map,
    const double* params,
    const double dt_a,
    const double dt_b,
    const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0 || row[n_cols - 1] == -2.0) return;

    double dfm[9], dfinv[9], v[3], k[3];
    v[0] = row[3]; v[1] = row[4]; v[2] = row[5];

    df_dispatch_dev(kind_map, row[0], row[1], row[2], params, dfm);
    matrix_inv_dev(dfm, dfinv);
    matvec_dev(dfinv, v, k);

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
}

extern "C" __global__
void push_v_with_efield_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* e1_1, const int n2x1, const int n3x1,
    const double* e1_2, const int n2x2, const int n3x2,
    const double* e1_3, const int n2x3, const int n3x3,
    const int kind_map,
    const double* params,
    const double dt_const)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];

    double bn1[MAXP + 1], bd1[MAXP];
    double bn2[MAXP + 1], bd2[MAXP];
    double bn3[MAXP + 1], bd3[MAXP];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double e_form[3] = {0.0, 0.0, 0.0};
    for (int il1 = 0; il1 < p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form[0] += e1_1[(size_t)i1 * n2x1 * n3x1 + (size_t)i2 * n3x1 + i3] * bd1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 < p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form[1] += e1_2[(size_t)i1 * n2x2 * n3x2 + (size_t)i2 * n3x2 + i3] * bn1[il1] * bd2[il2] * bn3[il3];
            }
        }
    }
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 < p3; il3++) {
                int i3 = span3 + il3 - start2;
                e_form[2] += e1_3[(size_t)i1 * n2x3 * n3x3 + (size_t)i2 * n3x3 + i3] * bn1[il1] * bn2[il2] * bd3[il3];
            }
        }
    }

    double dfm[9], dfinv[9], dfinvT_e[3];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    matrix_inv_dev(dfm, dfinv);
    matvecT_dev(dfinv, e_form, dfinvT_e);

    row[3] += dt_const * dfinvT_e[0];
    row[4] += dt_const * dfinvT_e[1];
    row[5] += dt_const * dfinvT_e[2];
}

// Shared setup for push_vxb_analytic_general / push_vxb_implicit_general:
// evaluate DF(eta), its determinant, and the Cartesian B-field at the
// marker's position. Returns 0 (and leaves b_cart untouched) if the marker
// is a hole/ghost, matching both CPU kernels' skip check.
__device__ int eval_b_cart_dev(
    const double* row, const int n_cols, const int first_init_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int n2x1, const int n3x1,
    const double* b2_2, const int n2x2, const int n3x2,
    const double* b2_3, const int n2x3, const int n3x3,
    const int kind_map, const double* params,
    double* b_cart)
{
    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];

    double bn1[MAXP + 1], bd1[MAXP];
    double bn2[MAXP + 1], bd2[MAXP];
    double bn3[MAXP + 1], bd3[MAXP];

    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);

    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b_form[3];
    eval_2form_dev(
        p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3,
        span1, span2, span3, start0, start1, start2,
        b2_1, n2x1, n3x1, b2_2, n2x2, n3x2, b2_3, n2x3, n3x3,
        b_form
    );

    double dfm[9];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    const double det_df = det3_dev(dfm);

    matvec_dev(dfm, b_form, b_cart);
    b_cart[0] /= det_df;
    b_cart[1] /= det_df;
    b_cart[2] /= det_df;
    return 1;
}

extern "C" __global__
void push_vxb_analytic_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int n2x1, const int n3x1,
    const double* b2_2, const int n2x2, const int n3x2,
    const double* b2_3, const int n2x3, const int n3x3,
    const int kind_map,
    const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[first_init_idx] == -1.0 || row[n_cols - 1] == -2.0) return;

    double b_cart[3];
    eval_b_cart_dev(
        row, n_cols, first_init_idx, p1, p2, p3,
        tn1, len_tn1, tn2, len_tn2, tn3, len_tn3,
        start0, start1, start2,
        b2_1, n2x1, n3x1, b2_2, n2x2, n3x2, b2_3, n2x3, n3x3,
        kind_map, params, b_cart
    );

    const double b_abs = sqrt(b_cart[0]*b_cart[0] + b_cart[1]*b_cart[1] + b_cart[2]*b_cart[2]);
    if (b_abs == 0.0) return;

    double b_norm[3] = {b_cart[0]/b_abs, b_cart[1]/b_abs, b_cart[2]/b_abs};
    double v[3] = {row[3], row[4], row[5]};

    const double vpar = dot3_dev(v, b_norm);

    double vxb_norm[3], vperp[3], b_normxvperp[3];
    cross_dev(v, b_norm, vxb_norm);
    cross_dev(b_norm, vxb_norm, vperp);
    cross_dev(b_norm, vperp, b_normxvperp);

    const double cbt = cos(b_abs * dt), sbt = sin(b_abs * dt);
    row[3] = vpar * b_norm[0] + cbt * vperp[0] - sbt * b_normxvperp[0];
    row[4] = vpar * b_norm[1] + cbt * vperp[1] - sbt * b_normxvperp[1];
    row[5] = vpar * b_norm[2] + cbt * vperp[2] - sbt * b_normxvperp[2];
}

extern "C" __global__
void push_vxb_implicit_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int n2x1, const int n3x1,
    const double* b2_2, const int n2x2, const int n3x2,
    const double* b2_3, const int n2x3, const int n3x3,
    const int kind_map,
    const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    // NOTE: the CPU push_vxb_implicit only checks the hole flag, not the
    // ghost flag (unlike push_vxb_analytic) -- faithfully reproduced here.
    if (row[first_init_idx] == -1.0) return;

    double b_cart[3];
    eval_b_cart_dev(
        row, n_cols, first_init_idx, p1, p2, p3,
        tn1, len_tn1, tn2, len_tn2, tn3, len_tn3,
        start0, start1, start2,
        b2_1, n2x1, n3x1, b2_2, n2x2, n3x2, b2_3, n2x3, n3x3,
        kind_map, params, b_cart
    );

    // b_prod = [[0, bz, -by], [-bz, 0, bx], [by, -bx, 0]] (row-major), such
    // that b_prod @ v == v x b_cart (matches the CPU kernel's b_prod, which
    // solves v x B via a matrix product rather than a cross product).
    double b_prod[9] = {
        0.0,        b_cart[2], -b_cart[1],
        -b_cart[2], 0.0,        b_cart[0],
        b_cart[1], -b_cart[0],  0.0,
    };

    double rhs[9], lhs[9];
    for (int k = 0; k < 9; k++) {
        const double id = (k == 0 || k == 4 || k == 8) ? 1.0 : 0.0;
        rhs[k] = id + 0.5 * dt * b_prod[k];
        lhs[k] = id - 0.5 * dt * b_prod[k];
    }

    double lhs_inv[9];
    matrix_inv_dev(lhs, lhs_inv);

    double v[3] = {row[3], row[4], row[5]};
    double vec[3], res[3];
    matvec_dev(rhs, v, vec);
    matvec_dev(lhs_inv, vec, res);

    row[3] = res[0];
    row[4] = res[1];
    row[5] = res[2];
}

// Single-point evaluation of a Derham 1-form spline (D-N-N / N-D-N / N-N-D),
// matching struphy.bsplines.evaluation_kernels_3d.eval_1form_spline_mpi.
// A standalone copy of the same math already inlined in
// push_v_with_efield_general above -- kept separate (not factored out and
// reused there) to avoid touching that already-validated kernel.
__device__ void eval_1form_dev(
    int p1, int p2, int p3,
    const double* bn1, const double* bd1,
    const double* bn2, const double* bd2,
    const double* bn3, const double* bd3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    const double* c1, int n2x1, int n3x1,
    const double* c2, int n2x2, int n3x2,
    const double* c3, int n2x3, int n3x3,
    double* out)
{
    out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;

    for (int il1 = 0; il1 < p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                out[0] += c1[(size_t)i1 * n2x1 * n3x1 + (size_t)i2 * n3x1 + i3] * bd1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 < p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                out[1] += c2[(size_t)i1 * n2x2 * n3x2 + (size_t)i2 * n3x2 + i3] * bn1[il1] * bd2[il2] * bn3[il3];
            }
        }
    }
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 < p3; il3++) {
                int i3 = span3 + il3 - start2;
                out[2] += c3[(size_t)i1 * n2x3 * n3x3 + (size_t)i2 * n3x3 + i3] * bn1[il1] * bn2[il2] * bd3[il3];
            }
        }
    }
}

// Single-point evaluation of a vector-field spline (H^1)^3 (N-N-N for every
// component), matching
// struphy.bsplines.evaluation_kernels_3d.eval_vectorfield_spline_mpi.
__device__ void eval_vectorfield_dev(
    int p1, int p2, int p3,
    const double* bn1, const double* bn2, const double* bn3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    const double* c1, int n2x1, int n3x1,
    const double* c2, int n2x2, int n3x2,
    const double* c3, int n2x3, int n3x3,
    double* out)
{
    out[0] = 0.0; out[1] = 0.0; out[2] = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                double b123 = bn1[il1] * bn2[il2] * bn3[il3];
                out[0] += c1[(size_t)i1 * n2x1 * n3x1 + (size_t)i2 * n3x1 + i3] * b123;
                out[1] += c2[(size_t)i1 * n2x2 * n3x2 + (size_t)i2 * n3x2 + i3] * b123;
                out[2] += c3[(size_t)i1 * n2x3 * n3x3 + (size_t)i2 * n3x3 + i3] * b123;
            }
        }
    }
}

// Shared setup for push_bxu_{Hdiv,Hcurl,H1vec}_general: evaluate DF(eta),
// its determinant, and the Cartesian B-field (always a 2-form) at the
// marker's position. Also computes and caches the local N-/D-spline basis
// values and span indices, reused by the caller for its own U-field
// evaluation (which differs per FEEC space).
__device__ void eval_b_cart_and_basis_dev(
    double eta1, double eta2, double eta3,
    int p1, int p2, int p3,
    const double* tn1, int len_tn1,
    const double* tn2, int len_tn2,
    const double* tn3, int len_tn3,
    int start0, int start1, int start2,
    const double* b2_1, int n2x1, int n3x1,
    const double* b2_2, int n2x2, int n3x2,
    const double* b2_3, int n2x3, int n3x3,
    int kind_map, const double* params,
    double* bn1, double* bd1, double* bn2, double* bd2, double* bn3, double* bd3,
    int* span1_out, int* span2_out, int* span3_out,
    double* dfm, double* b_cart)
{
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    *span1_out = span1; *span2_out = span2; *span3_out = span3;

    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double b_form[3];
    eval_2form_dev(
        p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3,
        span1, span2, span3, start0, start1, start2,
        b2_1, n2x1, n3x1, b2_2, n2x2, n3x2, b2_3, n2x3, n3x3,
        b_form
    );

    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    const double det_df = det3_dev(dfm);
    matvec_dev(dfm, b_form, b_cart);
    b_cart[0] /= det_df;
    b_cart[1] /= det_df;
    b_cart[2] /= det_df;
}

extern "C" __global__
void push_bxu_Hdiv_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int n2x1, const int n3x1,
    const double* b2_2, const int n2x2, const int n3x2,
    const double* b2_3, const int n2x3, const int n3x3,
    const double* u2_1, const int m2x1, const int m3x1,
    const double* u2_2, const int m2x2, const int m3x2,
    const double* u2_3, const int m2x3, const int m3x3,
    const int kind_map,
    const double* params,
    const double boundary_cut,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;
    if (row[0] < boundary_cut || row[0] > 1.0 - boundary_cut) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    int span1, span2, span3;
    double dfm[9], b_cart[3];
    eval_b_cart_and_basis_dev(
        eta1, eta2, eta3, p1, p2, p3, tn1, len_tn1, tn2, len_tn2, tn3, len_tn3,
        start0, start1, start2, b2_1, n2x1, n3x1, b2_2, n2x2, n3x2, b2_3, n2x3, n3x3,
        kind_map, params, bn1, bd1, bn2, bd2, bn3, bd3, &span1, &span2, &span3, dfm, b_cart
    );

    double u_form[3];
    eval_2form_dev(
        p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3,
        span1, span2, span3, start0, start1, start2,
        u2_1, m2x1, m3x1, u2_2, m2x2, m3x2, u2_3, m2x3, m3x3,
        u_form
    );
    const double det_df = det3_dev(dfm);
    double u_cart[3];
    matvec_dev(dfm, u_form, u_cart);
    u_cart[0] /= det_df; u_cart[1] /= det_df; u_cart[2] /= det_df;

    double e_cart[3];
    cross_dev(b_cart, u_cart, e_cart);
    row[3] += dt * e_cart[0];
    row[4] += dt * e_cart[1];
    row[5] += dt * e_cart[2];
}

extern "C" __global__
void push_bxu_Hcurl_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int n2x1, const int n3x1,
    const double* b2_2, const int n2x2, const int n3x2,
    const double* b2_3, const int n2x3, const int n3x3,
    const double* u1_1, const int m2x1, const int m3x1,
    const double* u1_2, const int m2x2, const int m3x2,
    const double* u1_3, const int m2x3, const int m3x3,
    const int kind_map,
    const double* params,
    const double boundary_cut,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;
    if (row[0] < boundary_cut || row[0] > 1.0 - boundary_cut) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    int span1, span2, span3;
    double dfm[9], b_cart[3];
    eval_b_cart_and_basis_dev(
        eta1, eta2, eta3, p1, p2, p3, tn1, len_tn1, tn2, len_tn2, tn3, len_tn3,
        start0, start1, start2, b2_1, n2x1, n3x1, b2_2, n2x2, n3x2, b2_3, n2x3, n3x3,
        kind_map, params, bn1, bd1, bn2, bd2, bn3, bd3, &span1, &span2, &span3, dfm, b_cart
    );

    double u_form[3];
    eval_1form_dev(
        p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3,
        span1, span2, span3, start0, start1, start2,
        u1_1, m2x1, m3x1, u1_2, m2x2, m3x2, u1_3, m2x3, m3x3,
        u_form
    );
    double dfinv[9], dfinvT[9], u_cart[3];
    matrix_inv_dev(dfm, dfinv);
    matvecT_dev(dfinv, u_form, u_cart);

    double e_cart[3];
    cross_dev(b_cart, u_cart, e_cart);
    row[3] += dt * e_cart[0];
    row[4] += dt * e_cart[1];
    row[5] += dt * e_cart[2];
}

extern "C" __global__
void push_bxu_H1vec_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* b2_1, const int n2x1, const int n3x1,
    const double* b2_2, const int n2x2, const int n3x2,
    const double* b2_3, const int n2x3, const int n3x3,
    const double* uv_1, const int m2x1, const int m3x1,
    const double* uv_2, const int m2x2, const int m3x2,
    const double* uv_3, const int m2x3, const int m3x3,
    const int kind_map,
    const double* params,
    const double boundary_cut,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;
    if (row[0] < boundary_cut || row[0] > 1.0 - boundary_cut) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    int span1, span2, span3;
    double dfm[9], b_cart[3];
    eval_b_cart_and_basis_dev(
        eta1, eta2, eta3, p1, p2, p3, tn1, len_tn1, tn2, len_tn2, tn3, len_tn3,
        start0, start1, start2, b2_1, n2x1, n3x1, b2_2, n2x2, n3x2, b2_3, n2x3, n3x3,
        kind_map, params, bn1, bd1, bn2, bd2, bn3, bd3, &span1, &span2, &span3, dfm, b_cart
    );

    double u_form[3];
    eval_vectorfield_dev(
        p1, p2, p3, bn1, bn2, bn3,
        span1, span2, span3, start0, start1, start2,
        uv_1, m2x1, m3x1, uv_2, m2x2, m3x2, uv_3, m2x3, m3x3,
        u_form
    );
    double u_cart[3];
    matvec_dev(dfm, u_form, u_cart);

    double e_cart[3];
    cross_dev(b_cart, u_cart, e_cart);
    row[3] += dt * e_cart[0];
    row[4] += dt * e_cart[1];
    row[5] += dt * e_cart[2];
}

// Shared setup for push_pc_GXu{_full,}_general: DF(eta)/dfinv/dfinvT plus
// span/basis values, reused by the caller to evaluate the 3 (or 2) rows of
// the GXu matrix via eval_1form_dev.
__device__ void eval_dfinvt_and_basis_dev(
    double eta1, double eta2, double eta3,
    int p1, int p2, int p3,
    const double* tn1, int len_tn1,
    const double* tn2, int len_tn2,
    const double* tn3, int len_tn3,
    int kind_map, const double* params,
    double* bn1, double* bd1, double* bn2, double* bd2, double* bn3, double* bd3,
    int* span1_out, int* span2_out, int* span3_out,
    double* dfinvt)
{
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    *span1_out = span1; *span2_out = span2; *span3_out = span3;

    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9], dfinv[9];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    matrix_inv_dev(dfm, dfinv);
    // dfinvt = dfinv^T, stored explicitly (row-major) since the caller needs
    // it as a plain matrix for matvec_dev, not just for a single matvecT_dev
    // application.
    dfinvt[0] = dfinv[0]; dfinvt[1] = dfinv[3]; dfinvt[2] = dfinv[6];
    dfinvt[3] = dfinv[1]; dfinvt[4] = dfinv[4]; dfinvt[5] = dfinv[7];
    dfinvt[6] = dfinv[2]; dfinvt[7] = dfinv[5]; dfinvt[8] = dfinv[8];
}

extern "C" __global__
void push_pc_GXu_full_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* g11, const double* g12, const double* g13,
    const double* g21, const double* g22, const double* g23,
    const double* g31, const double* g32, const double* g33,
    const int n2xc1, const int n3xc1,
    const int n2xc2, const int n3xc2,
    const int n2xc3, const int n3xc3,
    const int kind_map,
    const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    int span1, span2, span3;
    double dfinvt[9];
    eval_dfinvt_and_basis_dev(
        eta1, eta2, eta3, p1, p2, p3, tn1, len_tn1, tn2, len_tn2, tn3, len_tn3,
        kind_map, params, bn1, bd1, bn2, bd2, bn3, bd3, &span1, &span2, &span3, dfinvt
    );

    // components 1/2/3 of a 1-form generally have different shapes
    // (D-N-N / N-D-N / N-N-D), but the shape only depends on the component
    // index, not on which "row" of GXu is being evaluated -- so the same
    // (n2xc1,n3xc1)/(n2xc2,n3xc2)/(n2xc3,n3xc3) apply to all three rows.
    double gxu_row0[3], gxu_row1[3], gxu_row2[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   g11, n2xc1, n3xc1, g12, n2xc2, n3xc2, g13, n2xc3, n3xc3, gxu_row0);
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   g21, n2xc1, n3xc1, g22, n2xc2, n3xc2, g23, n2xc3, n3xc3, gxu_row1);
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   g31, n2xc1, n3xc1, g32, n2xc2, n3xc2, g33, n2xc3, n3xc3, gxu_row2);

    // GXu[i][j] = gxu_row_i[j]; e[j] = sum_i GXu[i][j] * v[i]
    double v[3] = {row[3], row[4], row[5]};
    double e[3];
    e[0] = gxu_row0[0]*v[0] + gxu_row1[0]*v[1] + gxu_row2[0]*v[2];
    e[1] = gxu_row0[1]*v[0] + gxu_row1[1]*v[1] + gxu_row2[1]*v[2];
    e[2] = gxu_row0[2]*v[0] + gxu_row1[2]*v[1] + gxu_row2[2]*v[2];

    double e_cart[3];
    matvec_dev(dfinvt, e, e_cart);

    row[3] -= dt * e_cart[0] / 2.0;
    row[4] -= dt * e_cart[1] / 2.0;
    row[5] -= dt * e_cart[2] / 2.0;
}

extern "C" __global__
void push_pc_GXu_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* g11, const double* g12, const double* g13,
    const double* g21, const double* g22, const double* g23,
    const int n2xc1, const int n3xc1,
    const int n2xc2, const int n3xc2,
    const int n2xc3, const int n3xc3,
    const int kind_map,
    const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    int span1, span2, span3;
    double dfinvt[9];
    eval_dfinvt_and_basis_dev(
        eta1, eta2, eta3, p1, p2, p3, tn1, len_tn1, tn2, len_tn2, tn3, len_tn3,
        kind_map, params, bn1, bd1, bn2, bd2, bn3, bd3, &span1, &span2, &span3, dfinvt
    );

    double gxu_row0[3], gxu_row1[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   g11, n2xc1, n3xc1, g12, n2xc2, n3xc2, g13, n2xc3, n3xc3, gxu_row0);
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   g21, n2xc1, n3xc1, g22, n2xc2, n3xc2, g23, n2xc3, n3xc3, gxu_row1);

    double v[3] = {row[3], row[4], row[5]};
    double e[3];
    e[0] = gxu_row0[0]*v[0] + gxu_row1[0]*v[1];
    e[1] = gxu_row0[1]*v[0] + gxu_row1[1]*v[1];
    e[2] = gxu_row0[2]*v[0] + gxu_row1[2]*v[1];

    double e_cart[3];
    matvec_dev(dfinvt, e, e_cart);

    row[3] -= dt * e_cart[0] / 2.0;
    row[4] -= dt * e_cart[1] / 2.0;
    row[5] -= dt * e_cart[2] / 2.0;
}

extern "C" __global__
void push_pc_eta_stage_Hcurl_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int first_free_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* u_1, const int n2x1, const int n3x1,
    const double* u_2, const int n2x2, const int n3x2,
    const double* u_3, const int n2x3, const int n3x3,
    const int use_perp_model,
    const int kind_map,
    const double* params,
    const double dt_a,
    const double dt_b,
    const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};

    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9], dfinv[9], dfinvt[9], ginv[9];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    matrix_inv_dev(dfm, dfinv);
    dfinvt[0]=dfinv[0]; dfinvt[1]=dfinv[3]; dfinvt[2]=dfinv[6];
    dfinvt[3]=dfinv[1]; dfinvt[4]=dfinv[4]; dfinvt[5]=dfinv[7];
    dfinvt[6]=dfinv[2]; dfinvt[7]=dfinv[5]; dfinvt[8]=dfinv[8];
    matmat_dev(dfinv, dfinvt, ginv);

    double k_v[3];
    matvec_dev(dfinv, v, k_v);

    double u[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   u_1, n2x1, n3x1, u_2, n2x2, n3x2, u_3, n2x3, n3x3, u);
    if (use_perp_model) u[2] = 0.0;

    double k_u[3];
    matvec_dev(ginv, u, k_u);

    double k[3] = {k_v[0]+k_u[0], k_v[1]+k_u[1], k_v[2]+k_u[2]};

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
}

extern "C" __global__
void push_pc_eta_stage_Hdiv_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int first_free_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* u_1, const int n2x1, const int n3x1,
    const double* u_2, const int n2x2, const int n3x2,
    const double* u_3, const int n2x3, const int n3x3,
    const int use_perp_model,
    const int kind_map,
    const double* params,
    const double dt_a,
    const double dt_b,
    const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};

    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9], dfinv[9];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    const double det_df = det3_dev(dfm);
    matrix_inv_dev(dfm, dfinv);

    double k_v[3];
    matvec_dev(dfinv, v, k_v);

    double u[3];
    eval_2form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   u_1, n2x1, n3x1, u_2, n2x2, n3x2, u_3, n2x3, n3x3, u);
    if (use_perp_model) u[2] = 0.0;

    double k_u[3] = {u[0]/det_df, u[1]/det_df, u[2]/det_df};
    double k[3] = {k_v[0]+k_u[0], k_v[1]+k_u[1], k_v[2]+k_u[2]};

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
}

extern "C" __global__
void push_pc_eta_stage_H1vec_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int first_free_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* u_1, const int n2x1, const int n3x1,
    const double* u_2, const int n2x2, const int n3x2,
    const double* u_3, const int n2x3, const int n3x3,
    const int use_perp_model,
    const int kind_map,
    const double* params,
    const double dt_a,
    const double dt_b,
    const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};

    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9], dfinv[9];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    matrix_inv_dev(dfm, dfinv);

    double k_v[3];
    matvec_dev(dfinv, v, k_v);

    double u[3];
    eval_vectorfield_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3, start0, start1, start2,
                          u_1, n2x1, n3x1, u_2, n2x2, n3x2, u_3, n2x3, n3x3, u);
    if (use_perp_model) u[2] = 0.0;

    double k[3] = {k_v[0]+u[0], k_v[1]+u[1], k_v[2]+u[2]};

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
}

extern "C" __global__
void push_weights_with_efield_lin_va_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* e1_1, const int n2x1, const int n3x1,
    const double* e1_2, const int n2x2, const int n3x2,
    const double* e1_3, const int n2x3, const int n3x3,
    const double* f0_values,
    const double kappa,
    const double vth,
    const int kind_map,
    const double* params,
    const double dt)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0 || row[n_cols - 1] == -2.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};

    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    double dfm[9], dfinv[9];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    matrix_inv_dev(dfm, dfinv);

    double dfinv_v[3];
    matvec_dev(dfinv, v, dfinv_v);

    double e_vec[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                   e1_1, n2x1, n3x1, e1_2, n2x2, n3x2, e1_3, n2x3, n3x3, e_vec);

    const double update = (dfinv_v[0]*e_vec[0] + dfinv_v[1]*e_vec[1] + dfinv_v[2]*e_vec[2])
                         * f0_values[ip] * kappa * dt / (2.0 * row[7] * vth * vth);
    row[6] += update;
}

// Single-point evaluation of a Derham 0-form spline (N-N-N), matching
// struphy.bsplines.evaluation_kernels_3d.eval_0form_spline_mpi.
__device__ double eval_0form_dev(
    int p1, int p2, int p3,
    const double* bn1, const double* bn2, const double* bn3,
    int span1, int span2, int span3,
    int start0, int start1, int start2,
    const double* c, int n2x, int n3x)
{
    double out = 0.0;
    for (int il1 = 0; il1 <= p1; il1++) {
        int i1 = span1 + il1 - start0;
        for (int il2 = 0; il2 <= p2; il2++) {
            int i2 = span2 + il2 - start1;
            for (int il3 = 0; il3 <= p3; il3++) {
                int i3 = span3 + il3 - start2;
                out += c[(size_t)i1 * n2x * n3x + (size_t)i2 * n3x + i3] * bn1[il1] * bn2[il2] * bn3[il3];
            }
        }
    }
    return out;
}

extern "C" __global__
void push_deterministic_diffusion_stage_general(
    double* markers,
    const int n_cols,
    const int n_markers,
    const int first_init_idx,
    const int first_free_idx,
    const int p1, const int p2, const int p3,
    const double* tn1, const int len_tn1,
    const double* tn2, const int len_tn2,
    const double* tn3, const int len_tn3,
    const int start0, const int start1, const int start2,
    const double* pi_u, const int n2xu, const int n3xu,
    const double* pi_grad_u1, const int n2x1, const int n3x1,
    const double* pi_grad_u2, const int n2x2, const int n3x2,
    const double* pi_grad_u3, const int n2x3, const int n3x3,
    const double diffusion_coeff,
    const int kind_map,
    const double* params,
    const double dt_a,
    const double dt_b,
    const double last)
{
    int ip = blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= n_markers) return;

    double* row = markers + (size_t)ip * n_cols;
    if (row[0] == -1.0) return;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];

    double bn1[MAXP + 1], bd1[MAXP], bn2[MAXP + 1], bd2[MAXP], bn3[MAXP + 1], bd3[MAXP];
    const int span1 = find_span_dev(tn1, p1, len_tn1, eta1);
    const int span2 = find_span_dev(tn2, p2, len_tn2, eta2);
    const int span3 = find_span_dev(tn3, p3, len_tn3, eta3);
    b_d_splines_dev(tn1, p1, eta1, span1, bn1, bd1);
    b_d_splines_dev(tn2, p2, eta2, span2, bn2, bd2);
    b_d_splines_dev(tn3, p3, eta3, span3, bn3, bd3);

    const double pi_u_value = eval_0form_dev(p1, p2, p3, bn1, bn2, bn3, span1, span2, span3,
                                              start0, start1, start2, pi_u, n2xu, n3xu);

    double pi_du_value[3];
    eval_1form_dev(p1, p2, p3, bn1, bd1, bn2, bd2, bn3, bd3, span1, span2, span3, start0, start1, start2,
                    pi_grad_u1, n2x1, n3x1, pi_grad_u2, n2x2, n3x2, pi_grad_u3, n2x3, n3x3, pi_du_value);

    // ginv = G^-1 = DF^-1 @ DF^-T, matching struphy.geometry.evaluation_kernels.g_inv
    // (computed there as (DF^T @ DF)^-1 instead -- same result, different
    // intermediate path, reusing the dfinv this file already needs elsewhere).
    double dfm[9], dfinv[9], dfinvt[9], ginv[9];
    df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm);
    matrix_inv_dev(dfm, dfinv);
    dfinvt[0]=dfinv[0]; dfinvt[1]=dfinv[3]; dfinvt[2]=dfinv[6];
    dfinvt[3]=dfinv[1]; dfinvt[4]=dfinv[4]; dfinvt[5]=dfinv[7];
    dfinvt[6]=dfinv[2]; dfinvt[7]=dfinv[5]; dfinvt[8]=dfinv[8];
    matmat_dev(dfinv, dfinvt, ginv);

    double tmp[3] = {
        -diffusion_coeff * pi_du_value[0] / pi_u_value,
        -diffusion_coeff * pi_du_value[1] / pi_u_value,
        -diffusion_coeff * pi_du_value[2] / pi_u_value,
    };
    double k[3];
    matvec_dev(ginv, tmp, k);

    row[first_free_idx + 0] += dt_b * k[0];
    row[first_free_idx + 1] += dt_b * k[1];
    row[first_free_idx + 2] += dt_b * k[2];

    row[0] = row[first_init_idx + 0] + dt_a * k[0] + last * row[first_free_idx + 0];
    row[1] = row[first_init_idx + 1] + dt_a * k[1] + last * row[first_free_idx + 1];
    row[2] = row[first_init_idx + 2] + dt_a * k[2] + last * row[first_free_idx + 2];
}

