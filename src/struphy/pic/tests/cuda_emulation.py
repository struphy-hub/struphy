"""Run struphy's CUDA kernels on the CPU, for parity tests without a GPU.

``cunumpy.kernel_testing.emulate_cuda_kernel`` compiles a CUDA kernel as C++ and calls it once per thread,
but it does not accept struct parameters, and every struphy kernel takes ``MarkerArgs``, ``DomainArgs``
or ``DerhamArgs``. :func:`emulate_struct_kernel` therefore emulates a generated wrapper kernel that takes
the fields of each struct as separate parameters, rebuilds the structs and calls the real kernel.
"""

import contextlib

import numpy as np
from cunumpy.kernel_testing import emulate_cuda_kernel, fake_cupy_active
from cunumpy.kernels import CudaKernel

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


def emulate_struct_kernel(kernel: CudaKernel, *args, n_threads=None, grid=None, block=None):
    """Emulate `kernel` with host arguments; struct arguments are the pyccel argument objects (NumPy arrays).

    Arrays (also those inside the argument objects) are updated in place, as by a launch. Without `n_threads` or
    `grid`, one thread per row of the first array is launched, as in a real launch; `grid` and `block` give the
    launch shape explicitly.
    """
    flat = []
    for p, value in zip(kernel.signature, args):
        if p.struct is None:
            flat.append(value)
        else:
            flat += [_struct_field(value, f.name) for f in p.struct.fields]
    if grid is None:
        if n_threads is None:
            n_threads = next(value.shape[0] for value in flat if isinstance(value, np.ndarray))
        grid, block = kernel.launch_shape(n_threads, block=block)
    source, name = _wrapper(kernel)
    wrapper = CudaKernel(source, name, source_dir=kernel.source_dir, **CUDA_OPTIONS)
    emulate_cuda_kernel(wrapper, *flat, grid=grid, block=block, options=EMULATION_OPTIONS)


def _host(value):
    """The host buffer behind a fake CuPy array (the same memory, no copy); other values unchanged."""
    if isinstance(value, np.ndarray) or not hasattr(value, "__cuda_array_interface__"):
        return value
    return object.__getattribute__(value, "_a")


class _HostFields:
    """A CUDA argument object (``Cuda*Arguments``) whose array attributes are their fake CuPy host buffers."""

    def __init__(self, args):
        self._args = args

    def __getattr__(self, name):
        return _host(getattr(self._args, name))


@contextlib.contextmanager
def emulated_launches():
    """On cunumpy's fake CuPy, run every ``CudaKernel`` launch by CPU emulation, in place on the fake device arrays.

    The fake CuPy (``CUNUMPY_FAKE_CUPY=1``) keeps device arrays in host memory and rejects host/device mixing, but
    cannot launch kernels. Inside this context a launch emulates the kernel (struct arguments included, through
    :func:`emulate_struct_kernel`) on the host buffers of the fake arrays, so code that launches CUDA kernels (struphy's
    and feectools') runs end to end on the CuPy backend without a GPU. Each launch compiles the kernel, so this is slow.
    """
    if not fake_cupy_active():
        raise RuntimeError("emulated_launches() needs cunumpy's fake CuPy (CUNUMPY_FAKE_CUPY=1)")
    original = CudaKernel.__call__

    def launch(self, *args, n_threads=None, grid=None, block=None, shared_mem=0, stream=None):
        grid, block = self.launch_shape(n_threads, grid=grid, block=block, args=args)
        if any(p.struct is not None for p in self.signature):
            host_args = [_host(a) if p.struct is None else _HostFields(a) for p, a in zip(self.signature, args)]
            emulate_struct_kernel(self, *host_args, grid=grid, block=block)
        else:
            host_args = [_host(a) for a in args]
            emulate_cuda_kernel(
                self, *host_args, grid=grid, block=block, shared_mem=shared_mem, options=EMULATION_OPTIONS
            )

    CudaKernel.__call__ = launch
    try:
        yield
    finally:
        CudaKernel.__call__ = original
