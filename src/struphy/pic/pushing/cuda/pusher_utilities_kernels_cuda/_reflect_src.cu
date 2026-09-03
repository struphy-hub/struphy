extern "C" __global__
void reflect_cuda(
    double* markers, const int n_cols,
    const long long* outside_inds, const int n_outside,
    const int axis,
    const int kind_map, const double* params)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_outside) return;

    const long long ip = outside_inds[i];
    double* row = markers + (size_t)ip * n_cols;

    const double eta1 = row[0], eta2 = row[1], eta3 = row[2];
    double v[3] = {row[3], row[4], row[5]};

    double dfm[9];
    if (!df_dispatch_dev(kind_map, eta1, eta2, eta3, params, dfm)) return;

    double dfinv[9], v_logical[3];
    matrix_inv_dev(dfm, dfinv);

    // pull back of the velocity
    matvec_dev(dfinv, v, v_logical);

    // reverse the velocity component along `axis`
    v_logical[axis] *= -1.0;

    // push forward of the velocity
    matvec_dev(dfm, v_logical, v);

    row[3] = v[0];
    row[4] = v[1];
    row[5] = v[2];
}

