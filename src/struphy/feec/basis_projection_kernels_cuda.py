"""CUDA kernels for dynamic weighted basis-projection matrices."""

from struphy.cuda import load_cuda_source

_ASSEMBLE_SRC = load_cuda_source(__file__, "basis_projection_kernels_cuda/_assemble_src.cu")

_kernel = None


def assemble_dofs_for_weighted_basisfuns_3d_gpu(
    mat,
    starts_in,
    ends_in,
    pads_in,
    starts_out,
    ends_out,
    pads_out,
    fun,
    weights,
    spans,
    bases,
    subs,
    dims_in,
    dims_out,
    degrees_out,
):
    import cupy as cp
    import numpy as np

    global _kernel
    if _kernel is None:
        _kernel = cp.RawKernel(_ASSEMBLE_SRC, "assemble_weighted_basis_3d_cuda")
    spans = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.int64)) for x in spans)
    weights = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in weights)
    bases = tuple(cp.ascontiguousarray(cp.asarray(x, dtype=cp.float64)) for x in bases)
    rows = tuple(cp.arange(len(x), dtype=cp.int64) - cp.cumsum(cp.asarray(x, dtype=cp.int64)) for x in subs)
    fun = cp.ascontiguousarray(fun)
    mat.fill(0.0)
    ni = tuple(x.shape[0] for x in spans)
    nq = tuple(x.shape[1] for x in spans)
    degree = tuple(x.shape[2] - 1 for x in bases)
    total = int(np.prod(ni) * np.prod(nq) * np.prod([p + 1 for p in degree]))
    threads = 256
    _kernel(
        ((total + threads - 1) // threads,),
        (threads,),
        (
            *rows,
            *spans,
            *weights,
            *bases,
            fun,
            *(np.int32(x) for x in (*ni, *nq, *degree)),
            *(np.int32(x) for x in starts_out),
            *(np.int32(x) for x in pads_in),
            *(np.int32(x) for x in pads_out),
            *(np.int32(x) for x in dims_in),
            *(np.int32(x) for x in dims_out),
            *(np.int32(x) for x in degrees_out),
            mat,
            *(np.int32(x) for x in mat.shape[1:]),
        ),
    )
