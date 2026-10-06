"""Run struphy's CUDA kernels on the CPU, for parity tests without a GPU.

``cunumpy.kernel_testing.emulate_cuda_kernel`` compiles a CUDA kernel as C++ and calls it once per thread,
but it does not accept struct parameters, and every struphy kernel takes ``MarkerArgs``, ``DomainArgs``
or ``DerhamArgs``. :func:`emulate_struct_kernel` therefore emulates a generated wrapper kernel that takes
the fields of each struct as separate parameters, rebuilds the structs and calls the real kernel.
"""

import numpy as np
from cunumpy.cuda import CudaKernel
from cunumpy.kernel_testing import emulate_cuda_kernel

from struphy.utils.cuda_arguments import CUDA_OPTIONS

# the CUDA mapping dispatch traps on unsupported mappings with inline PTX, which a CPU compiler rejects
EMULATION_OPTIONS = ("-Dasm(x)=__trap()",)


def _declaration(param, name):
    if param.view_ndim is not None:
        return f"{param.ctype} {name}"
    return f"{param.ctype}{'*' if param.pointer else ''} {name}"


def _wrapper(kernel: CudaKernel):
    """Source and name of a kernel that takes every struct field as a parameter and calls `kernel`."""
    params, builds, call = [], [], []
    for p in kernel.signature:
        if p.struct is None:
            params.append(_declaration(p, p.name))
            call.append(p.name)
            continue
        fields = [f"{p.name}__{f.name}" for f in p.struct.fields]
        params += [_declaration(f, name) for f, name in zip(p.struct.fields, fields)]
        builds.append(f"    {p.struct.name} {p.name}_struct{{{', '.join(fields)}}};")
        call.append(f"{p.name}_struct")
    name = f"emulated_{kernel.name}"
    source = (
        kernel.source
        + f'\nextern "C" __global__ void {name}({", ".join(params)}) {{\n'
        + "\n".join(builds)
        + f"\n    {kernel.name}({', '.join(call)});\n}}\n"
    )
    return source, name


def _struct_field(value, name):
    """Field `name` of a struct, read from the pyccel argument object `value`.

    The CUDA structs have the attributes of their pyccel classes plus the knot lengths ``nt1``, ``nt2``,
    ``nt3`` of ``DerhamArgs``, which pyccel gets from ``len(tn1)``.
    """
    if name in ("nt1", "nt2", "nt3"):
        return len(getattr(value, "tn" + name[-1]))
    return getattr(value, name)


def emulate_struct_kernel(kernel: CudaKernel, *args, n_threads=None):
    """Emulate `kernel` with host arguments; struct arguments are the pyccel argument objects (NumPy arrays).

    Arrays (also those inside the argument objects) are updated in place, as by a launch. Without `n_threads`,
    one thread per row of the first array is launched, as in a real launch.
    """
    flat = []
    for p, value in zip(kernel.signature, args):
        if p.struct is None:
            flat.append(value)
        else:
            flat += [_struct_field(value, f.name) for f in p.struct.fields]
    if n_threads is None:
        n_threads = next(value.shape[0] for value in flat if isinstance(value, np.ndarray))
    grid, block = kernel.launch_shape(n_threads)
    source, name = _wrapper(kernel)
    wrapper = CudaKernel(source, name, source_dir=kernel.source_dir, **CUDA_OPTIONS)
    emulate_cuda_kernel(wrapper, *flat, grid=grid, block=block, options=EMULATION_OPTIONS)
