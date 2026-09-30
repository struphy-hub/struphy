"""CUDA counterpart of :mod:`struphy.pic.pushing.demo_kernels`.

Run ``python -m struphy.pic.pushing.demo_cuda`` to push markers with both backends and compare.
"""

import cunumpy
import numpy as np
from cunumpy import PyccelKernel

from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments
from struphy.pic.pushing import demo_kernels
from struphy.utils.cuda_arguments import CudaDomainArguments, CudaMarkerArguments
from struphy.utils.kernel_backends import CudaKernel, Kernel, is_cuda_backend

# Arguments: (dt, stage, CudaMarkerArguments, CudaDomainArguments), see struphy.utils.cuda_arguments.
PUSH_ETA_LINEAR_SRC = r"""
extern "C" __global__
void push_eta_linear(
    double dt, int stage,
    double* markers, bool* valid_mks, int n_markers, int n_cols,
    int Np, int vdim, int weight_idx, int first_diagnostics_idx, int first_init_idx,
    int first_shift_idx, int residual_idx, int first_free_idx, int mu_idx, long long* bc_type,
    int kind_map, double* params, long long* degree,
    double* t1, double* t2, double* t3,
    long long* ind1, long long* ind2, long long* ind3,
    double* cx, double* cy, double* cz)
{
    int ip = blockDim.x * blockIdx.x + threadIdx.x;

    // only do something if particle is valid (i.e. not a hole or ghost)
    if (ip >= n_markers || !valid_mks[ip]) return;

    double* mk = markers + (long long)ip * n_cols;
    mk[0] += dt * mk[3];
    mk[1] += dt * mk[4];
    mk[2] += dt * mk[5];
}
"""

push_eta_linear = Kernel(
    pyccel_kernel=PyccelKernel(demo_kernels.push_eta_linear),
    cuda_kernel=CudaKernel(PUSH_ETA_LINEAR_SRC, "push_eta_linear"),
)


def make_demo_arguments(n_markers: int, seed: int = 0):
    """Random markers and a Cuboid domain, as kernel arguments for the active cunumpy backend.

    The arrays are created on the active backend (on the device for CuPy); the arguments reference them without copies.
    """
    rng = np.random.default_rng(seed)
    markers = cunumpy.asarray(rng.random((n_markers, 25)))
    valid_mks = cunumpy.asarray(rng.random(n_markers) > 0.1)
    bc_type = cunumpy.zeros(3, dtype=int)

    domain = Cuboid()
    if not is_cuda_backend():
        return MarkerArguments(markers, valid_mks, n_markers, 3, 6, 7, 8, 14, 17, 18, 4, bc_type), domain.args_domain

    args_markers = CudaMarkerArguments(markers, valid_mks, n_markers, 3, 6, 7, 8, 14, 17, 18, 4, bc_type)
    args_domain = CudaDomainArguments(
        domain.kind_map,
        domain.params_numpy,
        cunumpy.asarray(domain.degree),
        *domain.T,
        *domain.indN,
        domain.cx,
        domain.cy,
        domain.cz,
    )
    return args_markers, args_domain


def main(n_markers: int = 1_000_000, n_steps: int = 100, dt: float = 1e-3):
    results = {}
    for backend in ("numpy", "cupy"):
        with cunumpy.use_backend(backend):
            args_markers, args_domain = make_demo_arguments(n_markers)
            for _ in range(n_steps):
                push_eta_linear(dt, 0, args_markers, args_domain)
            results[backend] = cunumpy.to_numpy(args_markers.markers)

    print(f"max |pyccel - cuda| = {np.max(np.abs(results['numpy'] - results['cupy'])):.2e}")


if __name__ == "__main__":
    main()
