"""CUDA implementations of matrix-free FEEC mass-operator kernels.

These kernels are used only when ``ARRAY_BACKEND=cupy``.  The corresponding
Pyccel kernels accept CuPy arrays, but execute their nested loops on the host;
the routines below keep both the quadrature data and coefficient vectors on
the device.
"""
from struphy.cuda import CudaKernel, launch_1d, load_cuda_source

_H1VEC_DIVERGENCE_SRC = load_cuda_source(__file__, "mass_kernels_cuda/_h1vec_divergence_src.cu")

_divergence_eval_kernel = CudaKernel(_H1VEC_DIVERGENCE_SRC, "h1vec_divergence_eval_cuda")
_divergence_transpose_kernel = CudaKernel(_H1VEC_DIVERGENCE_SRC, "h1vec_divergence_transpose_cuda")

_MASS_ASSEMBLY_SRC = load_cuda_source(__file__, "mass_kernels_cuda/_mass_assembly_src.cu")

_mass_assembly_kernel = CudaKernel(_MASS_ASSEMBLY_SRC, "mass_3d_assemble_cuda")

_WEAK_DIV_ASSEMBLY_SRC = load_cuda_source(__file__, "mass_kernels_cuda/_weak_div_assembly_src.cu")

_weak_div_assembly_kernel = CudaKernel(_WEAK_DIV_ASSEMBLY_SRC, "weak_div_assemble_cuda")

_H1VEC_DIVDIV_ASSEMBLY_SRC = load_cuda_source(__file__, "mass_kernels_cuda/_h1vec_divdiv_assembly_src.cu")

_h1vec_divdiv_assembly_kernel = CudaKernel(_H1VEC_DIVDIV_ASSEMBLY_SRC, "h1vec_divdiv_assemble_cuda")


def _kernel_args(spans, degree, starts, pads, bases, dlogj, component):
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    bases = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases)
    dlogj = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in dlogj)
    return (
        *spans,
        np.int32(spans[0].size),
        np.int32(spans[1].size),
        np.int32(spans[2].size),
        *(np.int32(x) for x in degree),
        *(np.int32(x) for x in starts),
        *(np.int32(x) for x in pads),
        *bases,
        np.int32(bases[0].shape[2]),
        np.int32(bases[1].shape[2]),
        np.int32(bases[2].shape[2]),
        np.int32(bases[0].shape[3]),
        np.int32(bases[1].shape[3]),
        np.int32(bases[2].shape[3]),
        *dlogj,
        np.int32(component),
    )


def h1vec_divergence_eval_gpu(spans, degree, starts, pads, bases, dlogj, component, coeffs, values):
    """Add one H1-vector component's divergence to device ``values``."""
    import numpy as np

    args = _kernel_args(spans, degree, starts, pads, bases, dlogj, component)
    nvalues = values.size
    launch_1d(
        _divergence_eval_kernel,
        nvalues,
        (*args, coeffs, np.int32(coeffs.shape[1]), np.int32(coeffs.shape[2]), values),
    )


def h1vec_divergence_transpose_gpu(spans, degree, starts, pads, bases, dlogj, component, values, coeffs):
    """Accumulate the transpose of one H1-vector divergence component."""
    import numpy as np

    args = _kernel_args(spans, degree, starts, pads, bases, dlogj, component)
    nvalues = values.size
    launch_1d(
        _divergence_transpose_kernel,
        nvalues,
        (*args, values, np.int32(coeffs.shape[1]), np.int32(coeffs.shape[2]), coeffs),
    )


def mass_3d_assemble_gpu(spans, degree_i, degree_j, starts, pads, weights, bases_i, bases_j, mat_fun, data):
    """Assemble a 3D weighted mass matrix directly into device stencil data."""
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    weights = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in weights)
    bases_i = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_i)
    bases_j = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_j)
    mat_fun = cp.ascontiguousarray(mat_fun)
    total = int(
        np.prod([x.size for x in spans]) * np.prod([x + 1 for x in degree_i]) * np.prod([x + 1 for x in degree_j])
    )
    launch_1d(
        _mass_assembly_kernel,
        total,
        (
            *spans,
            *(np.int32(x.size) for x in spans),
            *(np.int32(x) for x in degree_i),
            *(np.int32(x) for x in degree_j),
            *(np.int32(x) for x in starts),
            *(np.int32(x) for x in pads),
            *weights,
            *(np.int32(x.shape[1]) for x in weights),
            *bases_i,
            *bases_j,
            *(np.int32(x.shape[2]) for x in bases_i),
            *(np.int32(x.shape[2]) for x in bases_j),
            mat_fun,
            data,
            *(np.int32(x) for x in data.shape[1:]),
        ),
    )


def weak_divergence_assemble_gpu(
    spans,
    degree_i,
    degree_j,
    starts,
    pads,
    weights,
    bases_i,
    bases_j,
    mat_fun,
    dlogj,
    component,
    data,
):
    """Assemble one L2-by-H1 weak-divergence block on the GPU."""
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    weights = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in weights)
    bases_i = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_i)
    bases_j = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases_j)
    dlogj = tuple(cp.ascontiguousarray(x) for x in dlogj)
    mat_fun = cp.ascontiguousarray(mat_fun)
    total = int(
        np.prod([x.size for x in spans]) * np.prod([p + 1 for p in degree_i]) * np.prod([p + 1 for p in degree_j])
    )
    launch_1d(
        _weak_div_assembly_kernel,
        total,
        (
            *spans,
            *(np.int32(x.size) for x in spans),
            *(np.int32(x) for x in degree_i),
            *(np.int32(x) for x in degree_j),
            *(np.int32(x) for x in starts),
            *(np.int32(x) for x in pads),
            *weights,
            *(np.int32(x.shape[1]) for x in weights),
            *bases_i,
            *bases_j,
            *(np.int32(x.shape[2]) for x in bases_i),
            *(np.int32(x.shape[2]) for x in bases_j),
            mat_fun,
            *dlogj,
            np.int32(component),
            data,
            *(np.int32(x) for x in data.shape[1:]),
        ),
    )


def h1vec_divdiv_assemble_gpu(spans, degree, starts, pads, bases, weighted_rho, component_test, component_trial, data):
    """Assemble one H1-vector div-div block on the GPU.

    This mirrors the existing Pyccel kernel exactly, including its current
    affine-mapping formulation where the log-Jacobian terms vanish.
    """
    import cupy as cp
    import numpy as np

    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    bases = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases)
    weighted_rho = cp.ascontiguousarray(weighted_rho)
    nloc = int(np.prod([x + 1 for x in degree]))
    total = int(np.prod([x.size for x in spans]) * nloc * nloc)
    launch_1d(
        _h1vec_divdiv_assembly_kernel,
        total,
        (
            *spans,
            *(np.int32(x.size) for x in spans),
            *(np.int32(x) for x in degree),
            *(np.int32(x) for x in starts),
            *(np.int32(x) for x in pads),
            *bases,
            *(np.int32(x.shape[2]) for x in bases),
            *(np.int32(x.shape[3]) for x in bases),
            weighted_rho,
            np.int32(component_test),
            np.int32(component_trial),
            data,
            *(np.int32(x) for x in data.shape[1:]),
        ),
    )
