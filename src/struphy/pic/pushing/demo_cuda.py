"""CUDA counterpart of :mod:`struphy.pic.pushing.demo_kernels` and the corresponding
:class:`~struphy.utils.kernel_backends.Kernel` registered in the kernel catalog.

Run as a script to push markers with both backends and compare::

    python -m struphy.pic.pushing.demo_cuda --n-markers 1000000 --n-steps 100
"""

import argparse
import time

import cunumpy
import numpy as np
from cunumpy import PyccelKernel

from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments
from struphy.pic.pushing import demo_kernels
from struphy.utils.cuda_arguments import CudaMarkerArguments
from struphy.utils.kernel_backends import CudaKernel, Kernel, catalog, is_cuda_backend

# Argument order = (dt, stage, CudaMarkerArguments.values, CudaDomainArguments.values),
# see struphy.utils.cuda_arguments.
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

push_eta_linear = catalog.register(
    Kernel(
        pyccel_kernel=PyccelKernel(demo_kernels.push_eta_linear),
        cuda_kernel=CudaKernel(PUSH_ETA_LINEAR_SRC, "push_eta_linear"),
    ),
)


def make_demo_arguments(n_markers: int, seed: int = 0):
    """Random markers (positions, velocities, some holes) and a Cuboid domain on the active cunumpy backend.

    The marker arrays are created on the active backend (on the device for CuPy), as ``Particles`` does;
    the kernel arguments reference them without copies.

    Returns
    -------
    args_markers : MarkerArguments | CudaMarkerArguments
        Marker arguments for the kernel of the active backend.

    args_domain : DomainArguments | CudaDomainArguments
        Domain arguments for the kernel of the active backend.
    """
    # same random numbers on both backends, such that results can be compared
    rng = np.random.default_rng(seed)
    markers = cunumpy.asarray(rng.random((n_markers, 25)))
    valid_mks = cunumpy.asarray(rng.random(n_markers) > 0.1)
    bc_type = cunumpy.zeros(3, dtype=int)

    domain = Cuboid()
    if is_cuda_backend():
        args_markers = CudaMarkerArguments(markers, valid_mks, n_markers, 3, 6, 7, 8, 14, 17, 18, 4, bc_type)
        args_domain = domain.cuda_args_domain
    else:
        args_markers = MarkerArguments(markers, valid_mks, n_markers, 3, 6, 7, 8, 14, 17, 18, 4, bc_type)
        args_domain = domain.args_domain
    return args_markers, args_domain


def run_push_eta_linear(args_markers, args_domain, dt: float, n_steps: int):
    """Push markers ``n_steps`` times with :data:`push_eta_linear` on the active cunumpy backend.

    Inside the time loop only the kernel is called; no arrays are converted or copied.

    Returns
    -------
    markers : numpy.ndarray
        Markers after the last step (copied to the host once, at the end).

    time_per_step : float
        Wall-clock time per step in seconds.
    """
    # warm-up (compiles the CUDA kernel on first call)
    push_eta_linear(0.0, 0, args_markers, args_domain)
    cunumpy.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        push_eta_linear(dt, 0, args_markers, args_domain)
    cunumpy.synchronize()
    time_per_step = (time.perf_counter() - t0) / n_steps

    return cunumpy.to_numpy(args_markers.markers), time_per_step


def main():
    parser = argparse.ArgumentParser(
        description="Push markers with the pyccel and the CUDA version of push_eta_linear."
    )
    parser.add_argument("--n-markers", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1e-3)
    args = parser.parse_args()

    backends = ["numpy"] + (["cupy"] if cunumpy.cupy_available() else [])
    results = {}
    for backend in backends:
        with cunumpy.use_backend(backend):
            args_markers, args_domain = make_demo_arguments(args.n_markers)
            results[backend] = run_push_eta_linear(args_markers, args_domain, args.dt, args.n_steps)
        print(
            f"{backend:>5}: {results[backend][1] * 1e3:8.3f} ms/step ({args.n_markers} markers, {args.n_steps} steps)"
        )

    if "cupy" in results:
        max_diff = np.max(np.abs(results["numpy"][0] - results["cupy"][0]))
        print(f"max |pyccel - cuda| = {max_diff:.2e}, speed-up = {results['numpy'][1] / results['cupy'][1]:.1f}x")
    else:
        print("CuPy/GPU not available, only the pyccel kernel was run.")


if __name__ == "__main__":
    main()
