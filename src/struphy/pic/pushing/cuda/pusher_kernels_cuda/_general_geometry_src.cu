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

// c = a^T @ v (used for DF^-T @ e_form in push_v_with_efield_general below).
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
