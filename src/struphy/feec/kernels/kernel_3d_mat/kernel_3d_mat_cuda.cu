// CUDA version of kernel_3d_mat (kernel_3d_mat_kernels.py), with the same arguments in the same order.
#include "cunumpy/array_view.cuh"
#include "cunumpy/atomic.cuh"
#include "cunumpy/index.cuh"

/**
 * Integral of Lambda_i * mat_fun * Lambda_j over the local elements, added to the stencil matrix data, as in
 * kernel_3d_mat_kernels.kernel_3d_mat.
 *
 * One thread per (element, codomain basis function (il1, il2, il3), domain basis function (jl1, jl2, jl3)):
 * n_threads = ne1 * ne2 * ne3 * (pi1 + 1) * (pi2 + 1) * (pi3 + 1) * (pj1 + 1) * (pj2 + 1) * (pj3 + 1) (see
 * __init__.py), the element index varying slowest and jl3 fastest. Each thread computes the quadrature sum of its
 * pair exactly as the pyccel kernel (same factors, same order) and adds it to its entry of `data` with an atomic
 * addition: neighbouring elements contribute to the same entries, so only the summation order over elements
 * differs from pyccel (round-off).
 *
 * @param spans1, spans2, spans3 Knot span index of each local element, per direction.
 * @param pi1, pi2, pi3 Degrees of the codomain basis functions.
 * @param pj1, pj2, pj3 Degrees of the domain basis functions.
 * @param starts1, starts2, starts3 Global start index of the rows on this process.
 * @param pads1, pads2, pads3 Paddings (ghost regions) of `data`.
 * @param w1, w2, w3 Quadrature weights [element, quadrature point].
 * @param bi1, bi2, bi3 Codomain basis functions [element, local basis function, derivative, quadrature point].
 * @param bj1, bj2, bj3 Domain basis functions, same layout.
 * @param mat_fun Weight function at the quadrature points [flattened point 1, flattened point 2, flattened point 3].
 * @param data StencilMatrix data (rows with ghost regions..., diagonals...); the results are added (not set).
 */
extern "C" __global__ void kernel_3d_mat(Array1D<long long> spans1, Array1D<long long> spans2,
                                         Array1D<long long> spans3, int pi1, int pi2, int pi3, int pj1, int pj2,
                                         int pj3, int starts1, int starts2, int starts3, int pads1, int pads2,
                                         int pads3, Array2D<double> w1, Array2D<double> w2, Array2D<double> w3,
                                         Array4D<double> bi1, Array4D<double> bi2, Array4D<double> bi3,
                                         Array4D<double> bj1, Array4D<double> bj2, Array4D<double> bj3,
                                         Array3D<double> mat_fun, Array6D<double> data)
{
    const long long ne1 = spans1.size();
    const long long ne2 = spans2.size();
    const long long ne3 = spans3.size();

    const long long ni1 = pi1 + 1, ni2 = pi2 + 1, ni3 = pi3 + 1;
    const long long nj1 = pj1 + 1, nj2 = pj2 + 1, nj3 = pj3 + 1;
    const long long n_pairs = ni1 * ni2 * ni3 * nj1 * nj2 * nj3;

    CUNUMPY_THREAD_1D(thread, ne1 * ne2 * ne3 * n_pairs);

    // CUDA-only: split the thread index into the pyccel loop variables (iel1, iel2, iel3, il1, il2, il3, jl1, jl2,
    // jl3), jl3 fastest
    long long rest = thread;
    const long long jl3 = rest % nj3; rest /= nj3;
    const long long jl2 = rest % nj2; rest /= nj2;
    const long long jl1 = rest % nj1; rest /= nj1;
    const long long il3 = rest % ni3; rest /= ni3;
    const long long il2 = rest % ni2; rest /= ni2;
    const long long il1 = rest % ni1; rest /= ni1;
    const long long iel3 = rest % ne3; rest /= ne3;
    const long long iel2 = rest % ne2; rest /= ne2;
    const long long iel1 = rest;

    const long long nq1 = w1.shape[1];
    const long long nq2 = w2.shape[1];
    const long long nq3 = w3.shape[1];

    double value = 0.0;

    for (long long q1 = 0; q1 < nq1; ++q1) {
        for (long long q2 = 0; q2 < nq2; ++q2) {
            for (long long q3 = 0; q3 < nq3; ++q3) {
                const double wvol = w1(iel1, q1) * w2(iel2, q2) * w3(iel3, q3) *
                                    mat_fun(iel1 * nq1 + q1, iel2 * nq2 + q2, iel3 * nq3 + q3);

                const double bi = bi1(iel1, il1, 0, q1) * bi2(iel2, il2, 0, q2) * bi3(iel3, il3, 0, q3);
                const double bj = bj1(iel1, jl1, 0, q1) * bj2(iel2, jl2, 0, q2) * bj3(iel3, jl3, 0, q3);

                value += wvol * bi * bj;
            }
        }
    }

    // global spline indices
    const long long i_global1 = spans1(iel1) - pi1 + il1;
    const long long i_global2 = spans2(iel2) - pi2 + il2;
    const long long i_global3 = spans3(iel3) - pi3 + il3;

    // local spline indices (- starts --> can be negative, will therefore be written to ghost regions)
    const long long i_local1 = i_global1 - starts1;
    const long long i_local2 = i_global2 - starts2;
    const long long i_local3 = i_global3 - starts3;

    cunumpy_atomic_add(&data(pads1 + i_local1, pads2 + i_local2, pads3 + i_local3, pads1 + jl1 - il1,
                             pads2 + jl2 - il2, pads3 + jl3 - il3),
                       value);
}
