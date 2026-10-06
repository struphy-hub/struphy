# CUDA strategy for Struphy kernels

Plan for running Struphy's compute kernels on NVIDIA GPUs, next to the existing pyccel (CPU) kernels.
The work is split into small PRs that can be reviewed and merged one at a time. Nothing here has to be done in one go.

Progress is tracked in struphy-hub/struphy#650.

## PR checklist

- [x] **PR 1: Proof of concept** (#643)
  `Kernel` (pyccel/CUDA pair), `CudaKernel`, `CudaMarkerArguments`/`CudaDomainArguments`, and one test file with a demo kernel pair run on both backends. Also adds this document and the `gpu` optional dependency.
- [x] **PR 2: CUDA source files** (#644) — `CudaKernel` loads CUDA source from a `<name>_cuda.cu` file next to the pyccel file; `.cu`/`.cuh` files are shipped as package data.
- [x] **PR 3: Kernel catalog** (#645) — kernels are defined once in the `__init__.py` of the folder that contains them; a missing CUDA kernel raises an error on the GPU backend.
- [x] **PR 4: `Pusher` accepts `Kernel`** (#646) — the kernel for the active backend is chosen once, when the pusher is created; a plain `PyccelKernel` is wrapped, so the propagators do not change (no behaviour change on CPU).
- [x] **PR 5: `Domain` on the GPU** (#649) — `Domain.args_domain` is selected at construction; CUDA arguments reference device arrays, and deepcopy/unpickling rebuilds them from the copied or restored arrays.
- [x] **PR 6: `Particles` on the GPU** (#651) — `Particles` can be created on the CuPy backend, and `Particles.args_markers` is selected at construction.
- [x] **PR 7: `Derham` on the GPU** (#652) — `Derham` can be created on the CuPy backend, and `Derham.args_derham` is selected at construction.
- [x] **PR 8: Shared CUDA header for the argument classes** (#653, open) — one struct per argument class in `kernel_arguments/pusher_args.cuh`, passed by value, instead of long flat kernel signatures.
- [x] **PR 9: One folder per kernel, starting with `pic/pushing`** (#663, open) — pure refactor; the propagators take their pusher and evaluation kernels from `struphy.pic.pushing.kernels.catalog`.
- [ ] **PR 10: Device versions of helper kernels** — B-spline evaluation, mapping evaluation (per domain), small linear algebra, boundary conditions, as `__device__` functions in `.cuh` headers, each tested against its pyccel version. Starts with an H100 run of the PR 8 struct tests.
- [ ] **PR 11: First real CUDA kernel** — `push_eta_stage` with a pyccel/CUDA parity test and an end-to-end run on the GPU without host/device transfers in the time loop.
- [ ] **PR 12+: Port kernels one by one**, in the order they are needed by the models we want on the GPU (see [Porting order](#porting-order)). Accumulation needs its own design step first.
- [ ] **PR 14: Accumulation catalog and first Vlasov–Ampère kernels** — `pic/accumulation` is split into one folder per kernel with its own `catalog`; `Accumulator`/`AccumulatorVector` take catalog kernels and launch the CUDA version with one thread per marker row. CUDA versions of `push_v_with_efield`, `push_weights_with_efield_lin_va` and `charge_density_0form` (atomic fp64 adds). All CUDA kernels are checked against pyccel without a GPU by CPU emulation. `vlasov_maxwell` and `linear_vlasov_ampere` wait for 6D array views in cunumpy.
- [ ] **PR 15: Use cunumpy as intended** — every kernel folder declares its kernel in its `__init__.py` (`Kernel.from_folder`), and the code imports it: `from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage` instead of `catalog["push_eta_stage"]`; the module-level catalogs are gone. `Pusher`, `KernelSetup` and the accumulators call the `Kernel` itself (no `get_kernel()`, no `n_threads` branches: cunumpy infers one thread per marker from `args_markers`). Parity test arguments live next to each kernel in `<name>_test_args.py`, driven by `cunumpy.kernel_testing.parity_cases`/`check_parity`. Built against cunumpy `devel`; once the usage here is settled, cunumpy 0.5.0 is released on PyPI and struphy pins `cunumpy>=0.5.0, <0.6`.
- [ ] **PR 16: One CPU and one GPU argument class per argument type** — every pyccel argument class (`MarkerArguments`, `DerhamArguments`, `DomainArguments`, `LocalProjectorsArguments`) has a CUDA counterpart of the same name with a `Cuda` prefix in `kernel_arguments/*_cuda.py`, with the same constructor and attributes. Each owner creates one of the two in `__init__`, depending on the backend; `PyccelStructArguments`, `__host_args__()` and host copies are no longer used. Every kernel that takes argument objects is called through `Kernel(PyccelKernel(...))`; without a CUDA version it raises on CuPy. The remaining entry kernels called from Python (spline, geometry and SPH evaluation, marker diagnostics, `reflect`, local projectors) move into one folder per kernel, like the pushers. Geometry evaluations have no CUDA version yet, so CuPy particle runs fail at setup until they are ported (see [PR 16 implementation notes](#pr-16-implementation-notes)).
- [ ] **PR 17: struphy on the feectools CUDA stack** — the `feectools` submodule points at the top of the feectools CUDA stack (`cuda-4-device-kernels`, [feectools#88](https://github.com/struphy-hub/feectools/pull/88)) instead of `devel-tiny`, so the struphy CUDA PRs run against feectools with device stencil operations, MPI with device buffers and one GPU per rank. Moves along with the stack; before the struphy stack is merged into `devel`, the feectools stack is merged into `devel-tiny` and the submodule points there again (see [feectools](#feectools)).
- [ ] **PR 18: Geometry evaluation on the GPU for all analytic mappings** — CUDA versions of the four geometry entry kernels (`kernel_evaluate_pic`, `kernel_evaluate`, `kernel_pullpush_pic`, `kernel_pullpush` in `geometry/kernels/`), built on device versions of the whole metric chain (`f`, `df`, `det_df`, `df_inv`, `g`, `g_inv`, `select_metric_coeff`, `pull`/`push`/`tran`) for every analytic mapping (`kind_map` 10–12, 20–22, 30–32). Restores CuPy particle runs (weight initialization evaluates `jacobian_det`), and removes the Cuboid-only checks in `Pusher`, the accumulators and `reflect`. Parity arguments cover every analytic mapping (see [PR 18](#pr-18-geometry-evaluation-for-all-analytic-mappings)).
- [ ] **PR 19: Spline mappings on the GPU** — `kind_map` 0–2 (`IGAPolarCylinder`, `IGAPolarTorus`, `Tokamak`, GVEC, DESC): `DomainArgs` gets array views for `ind1..3` and `cx/cy/cz` (shape needed on the device), spline-mapped `Domain`s can be created on the CuPy backend, and `spline_3d`, `spline_2d_straight`, `spline_2d_torus` get device versions (see [PR 19](#pr-19-spline-mappings)).
- [ ] **Next (order to be confirmed)**: 6D array views in cunumpy and the blocked matrix accumulations `vlasov_maxwell`, `linear_vlasov_ampere` (steps 1–2 of the [porting order](#porting-order)); one complete model (`VlasovAmpereOneSpecies`) on the GPU end to end, including when to compile the kernels, which needs the [feectools stack](#feectools) merged and released first; a decision on hand-written CUDA vs. code generation before the guiding-center kernels (step 3).
- [ ] **feectools**: the FEEC side (stencil vectors and matrices, MPI exchange, GPU binding) in feectools, stacked PRs [#85](https://github.com/struphy-hub/feectools/pull/85) (merged into `cuda-development`), [#86](https://github.com/struphy-hub/feectools/pull/86), [#87](https://github.com/struphy-hub/feectools/pull/87), [#88](https://github.com/struphy-hub/feectools/pull/88), integrated by [#90](https://github.com/struphy-hub/feectools/pull/90); needed before the end-to-end model run, not for PR 18/19 (see [feectools](#feectools)).
- [x] **PR 13: Move the kernel infrastructure to cunumpy** — `Kernel`, `KernelCatalog`, `CudaKernel` and `Argument` are replaced by their cunumpy counterparts; each owner has a single `args_*` object on both backends, and `pusher_args.cuh` is generated (see [Moving to cunumpy](#moving-to-cunumpy-pr-13)). Kernels and device helpers only change their includes.
- [ ] **CI**: a GPU runner that runs the CUDA tests (can happen any time; until then the GPU tests are run by hand on an H100 before each PR that touches CUDA code is merged).

Unrelated bugs found along the way go into their own PRs, not into these ones.

## Goal

A developer who adds a new kernel (e.g. for a new model) should only have to:

1. write the pyccel kernel `<name>_kernels.py` in a new folder `<package>/kernels/<name>/`, next to an `__init__.py` that declares `<name> = Kernel.from_folder(__name__, structs=CUDA_STRUCTS)`, and
2. optionally write `<name>_cuda.cu` and `<name>_test_args.py` **in the same folder**.

Everything else (loading, dispatch, argument passing, tests for agreement between the two versions) is done by the infrastructure.
CUDA kernels can be added one by one. If the code runs on the GPU and needs a kernel that has no CUDA version yet, it raises a clear error instead of silently falling back to the CPU.

## Principles

- **1:1 correspondence.** Each CUDA kernel has the same name and the same arguments (in the same order) as its pyccel kernel. The CUDA kernel takes the CUDA versions of the argument classes, plus the number of threads (`n_threads`).
- **The backend decides.** The cunumpy backend (`CUNUMPY_BACKEND=cupy` or `cunumpy.set_backend("cupy")`, queried with `cunumpy.get_backend()`) selects the CUDA kernels; with NumPy the pyccel kernels run as today.
- **No conversions at call time.** When a kernel is called, its arguments are already in the right format. There are no host/device copies per kernel call.
- **Data already lives on the GPU.** On the CuPy backend, `xp` is `cupy`, so markers, spline coefficients etc. are CuPy arrays from the start. The CUDA argument objects only collect *references* to these arrays and raise if they get host arrays.
- **No silent CPU fallback on the GPU.** A kernel without a CUDA version raises an error on the GPU backend. Falling back would mean copying data to the host and back at every call.
- **Kernels see only the C structs.** CUDA kernels and `__device__` helpers take `MarkerArgs`, `DomainArgs` and `DerhamArgs` from `pusher_args.cuh` and nothing else of the Python side. The Python classes that fill the structs can change (as they did when they moved to cunumpy in PR 13) without touching a single kernel.
- **Own infrastructure first, cunumpy later.** The kernel infrastructure (`Kernel`, `KernelCatalog`, `CudaKernel`, `Argument`) stayed in struphy while it was being designed (PRs 1–12) and moved to [cunumpy](https://github.com/struphy-hub/cunumpy), which we maintain and release on PyPI ourselves, once it was stable (PR 13, see [Moving to cunumpy](#moving-to-cunumpy-pr-13)).
- **One folder per kernel.** Every kernel that Python calls (an *entry kernel*) lives in its own folder `<package>/kernels/<name>/`: `<name>_kernels.py` (pyccel), `__init__.py` (declares the `Kernel`), and once ported `<name>_cuda.cu` and `<name>_test_args.py`. Code imports it (`from struphy.geometry.kernels.kernel_evaluate import kernel_evaluate`) and calls it; there is no `Kernel(PyccelKernel(...))` at call sites. **Any PR that adds a kernel, or ports one that still lives in a shared module, puts it into its own folder first** (moved with its imports, as in PR 16) and adds its package to `test_cuda_parity.PACKAGES`. Helpers called only from other kernels (`@pure` functions, `__device__` functions) stay in shared modules and headers, e.g. `geometry/evaluation_kernels.py` and `geometry/evaluation_kernels.cuh`; per-mapping device helpers go next to their domain (`geometry/domains/<name>/<name>_cuda.cuh`, like `cuboid_cuda.cuh`).
- **Small steps.** Every PR keeps the CPU code path working and tested.

## Current state (after PR 16)

| File | Content |
|---|---|
| `cunumpy.kernels`, `cunumpy.cuda` | `Kernel`, `KernelCatalog`, `PyccelKernel`, `CudaStructArguments` and `CudaKernel`; Struphy uses cunumpy for dispatch and struct packing |
| `src/struphy/kernel_arguments/` | the argument classes in pairs: the pyccel classes in `pusher_args_kernels.py` / `local_projectors_args_kernels.py` (NumPy backend) and their CUDA versions `Cuda<Name>` in `pusher_args_cuda.py` / `local_projectors_args_cuda.py` (CuPy backend; subclasses of `CudaStructArguments` whose `fields` define the C struct) |
| `src/struphy/utils/cuda_arguments.py` | `CUDA_STRUCTS`, `CUDA_OPTIONS`, `write_pusher_header()` and `write_local_projectors_header()` |
| `src/struphy/kernel_arguments/pusher_args.cuh`, `local_projectors_args.cuh` | the C structs `MarkerArgs`, `DerhamArgs`, `DomainArgs` and `LocalProjectorsArgs`, generated from the CUDA classes, using cunumpy array views |
| `src/struphy/geometry/base.py`, `src/struphy/pic/base.py`, `src/struphy/feec/psydac_derham.py` | `Domain.args_domain`, `Particles.args_markers` and `Derham.args_derham` are the pyccel class on NumPy and the CUDA class on CuPy, chosen once at construction; every kernel call goes through a `Kernel` |
| `src/struphy/*/kernels/` (`pic/pushing`, `pic/accumulation`, `pic/diagnostics`, `pic/sph`, `bsplines`, `geometry`, `feec`, `feec/local_projectors`) | every kernel called from Python with argument objects, one folder each: 44 pusher/evaluation (incl. `reflect`), 16 accumulation, 10 marker diagnostics, 4 SPH evaluation, 3 spline evaluation, 4 geometry, 1 FEEC utility and 8 local projector kernels. Each folder's `__init__.py` declares its `Kernel`; the code imports it (`from struphy.geometry.kernels.kernel_evaluate import kernel_evaluate`). Eleven have CUDA versions and a `<name>_test_args.py` |
| `src/struphy/pic/tests/test_cuda_parity.py`, `test_cuda_emulation.py` | parity of every CUDA kernel with pyccel from the `<name>_test_args.py` modules: on a GPU (`check_parity`), and without one by CPU emulation |
| `src/struphy/pic/tests/test_kernel_backends.py` | the generated-header tests, the check that each CUDA class mirrors its pyccel class, signature check of all kernels, owner class selection, and (on a GPU) struct layout and copy/pickle checks |

Things we learned so far:

- The pyccel-compiled argument classes (`MarkerArguments`, `DomainArguments`, `DerhamArguments`) hold references to their owner's arrays, but only accept **NumPy** arrays, and a pyccel class cannot inherit from a Python base class. Hence the CUDA counterparts are separate Python classes that hold the same arrays.
- `cupy.RawKernel` accepts only device arrays (host arrays raise) and does **not** check the kernel signature: a wrongly typed scalar or a shifted argument gives a wrong value **without an error**. This is why the argument classes are passed as structs (PR 8). Scalars passed directly to a kernel (`dt`, `stage`) are still unchecked; see [Open questions](#open-questions).
- Flattening the argument classes at each call costs well under 1 µs, compared to about 70 µs for launching the kernel. Since PR 8 each class is one struct, packed once when the argument object is created.
- `struphy compile` compiles every `.py` file whose name contains `kernels`. Non-pyccel modules must not contain `kernels` in their name; `.cu` files are ignored by it.
- Kernel names can have at most 48 characters: with the Fortran backend, pyccel names the wrapper module `bind_c_<name>_kernels`, and Fortran names have at most 63 characters (PR 9).
- Running the particle tests without an MPI launcher uses feectools' `MockComm`, whose collectives do nothing. Code that gathers into a buffer must fill its own entry first (fixed in PR 6).
- On an H100, the demo kernel pushes 10⁶ markers in about 0.13 ms per step (PR 1, flat arguments). Passing the structs by value (PR 8) has **not** run on a GPU yet; this is the first thing PR 10 does.

## Target layout

Each kernel gets its own folder with the pyccel and the CUDA version side by side, and the folder's `__init__.py` declares the kernel (since PR 15; before, the parent `__init__.py` held a catalog):

```
src/struphy/pic/pushing/kernels/
├── __init__.py                          # documentation only
├── push_eta_stage/
│   ├── __init__.py                      # push_eta_stage = Kernel.from_folder(__name__, structs=CUDA_STRUCTS)
│   ├── push_eta_stage_kernels.py        # pyccel (compiled by `struphy compile`)
│   ├── push_eta_stage_cuda.cu           # CUDA (compiled at runtime by CuPy/NVRTC)
│   └── push_eta_stage_test_args.py      # make_args(backend, seed) and CASES for the parity tests
├── push_bxu_Hdiv/
│   ├── __init__.py
│   └── push_bxu_Hdiv_kernels.py         # no CUDA version yet -> error on the GPU
└── ...
```

Conventions:

- The folder name, the pyccel function name, the CUDA `extern "C" __global__` function name and the name of the declared `Kernel` are all the same.
- The pyccel file is `<name>_kernels.py` (so `struphy compile` picks it up); the CUDA file is `<name>_cuda.cu`.
- Shared CUDA code (device helper functions, argument structs) lives in `.cuh` headers next to the pyccel module it mirrors, e.g. `bsplines/bsplines_kernels.cuh` for `bsplines/bsplines_kernels.py`. A `.cu` file includes them relative to the source root, e.g. `#include "struphy/kernel_arguments/pusher_args.cuh"`.
- Every function in a `.cu` or `.cuh` file has a documentation comment (`/** ... */`, the C++ equivalent of a docstring) immediately above its definition. Identify the pyccel counterpart and explain the operation, parameters, outputs or return value, array shapes/layouts, and any CUDA-specific limits or storage requirements. Document CUDA-only helpers by identifying the pyccel calculation they extract.
- CUDA parameter, local variable and scratch-field names exactly match the corresponding pyccel names, including spelling and capitalization (e.g. `eta1`, `args_markers`, `args_domain`, `args_derham`, `df_out`, `dfm`, `dfinv`, `v_logical`, `tn`, `pn`, `pd`, `il`, `det_a`, `span1`, `bn1`, `bd1`). Do not abbreviate them to `x`, `m`, `d`, `a`, or `out` when the pyccel counterpart uses a different name. Additional CUDA-only variables or arguments, such as pointer lengths, thread indices or per-thread scratch storage, must have descriptive names and documented purposes.
- Kernels read and write markers through the `Array2D<double>` view: `args.markers(ip, j)`. The view carries the existing device pointer, shape and element strides; no array data is copied. The column count is `args.markers.shape[1]`.

The geometry domains already follow a similar layout (`geometry/domains/cuboid/cuboid_kernels.py`), which can be extended with `cuboid_cuda.cuh` for the device version of the mapping.

Required in every CUDA porting branch (PR 10, PR 11 and PR 12+), before review:

- [ ] Document every added or modified `.cu`/`.cuh` function following the conventions above.
- [ ] Compare parameter, local variable and scratch-field names against the pyccel source; match every corresponding name and document CUDA-only additions.
- [ ] Update callers when helper signatures or scratch fields change, and run the affected pyccel/CUDA parity tests with fresh header compilation.

Usage at a call site (e.g. in a propagator):

```python
from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage  # a Kernel: pyccel or CUDA by backend
```

## Details per PR

### PR 2: CUDA source files (complete)

- `CudaKernel.from_file(path)` reads `<name>_cuda.cu`; the kernel name is taken from the file name. The kernel is compiled lazily on first call. CuPy caches compiled kernels on disk (`~/.cupy/kernel_cache`), so the compile cost is paid once per machine. Note: CuPy keys the cache on the source text and options only, so a changed `.cuh` header does not trigger a recompile by itself; delete `~/.cupy/kernel_cache` after editing a header (or add an include hash to the options, which cunumpy does).
- `"**/*.cu"` and `"**/*.cuh"` are package data in `pyproject.toml`.
- Shared headers are found through the NVRTC include path (`-I<folder containing the struphy package>`, added in PR 8).

### PR 3: Kernel catalog (complete)

- `KernelCatalog.from_package(package)` scans the subfolders of a package. For every `<name>/<name>_kernels.py` it creates a `Kernel` with `PyccelKernel(<name>_kernels.<name>)` and, if `<name>/<name>_cuda.cu` exists, a `CudaKernel`. Developers add files, not registration code.
- `Kernel` without a CUDA version: on the CuPy backend, `get_kernel()` raises `NotImplementedError: No CUDA version of kernel 'push_vxb_analytic' (expected .../push_vxb_analytic_cuda.cu)`.
- The error comes as early as possible: propagators/pushers call `get_kernel()` when they are set up, not at the first time step.
- `catalog.missing_cuda` lists the kernels without a CUDA version.

### PR 4: `Pusher` accepts `Kernel` (complete)

- `Pusher` takes a `Kernel` or a `PyccelKernel` (wrapped into a `Kernel` without CUDA version) and calls `get_kernel()` once in its constructor.
- The propagators did not change; they switched to catalog lookups in PR 9.

### PR 5–7: Owners build their CUDA arguments (complete)

- `Particles`, `Domain` and `Derham` own the arrays, so they build the CUDA argument objects from their own `xp` arrays. `args_markers`, `args_domain` and `args_derham` are the pyccel bundle on NumPy and the CUDA bundle on CuPy, selected once at construction; each owner keeps a private pyccel bundle (`_pyccel_args_*`) for existing direct pyccel calls.
- The CUDA argument objects hold references. If an owner reallocates an array, it must rebuild its arguments at the same place, exactly like for the pyccel arguments.
- `Domain` (PR 5): arrays that already have the dtype and layout the kernels expect (`float64`/`int64`, C-contiguous) are referenced, not copied; otherwise one device copy is made when the arguments are built (e.g. `degree`, a tuple). Arguments are recreated after deepcopy or unpickling. Spline mappings (e.g. `IGAPolarCylinder`) cannot be created on CuPy yet (`interp_mapping` passes CuPy arrays to `scipy.sparse`), so CUDA `args_domain` exists only for analytic mappings for now.
- `Particles` (PR 6): particle arrays, validity masks and boundary-condition codes are allocated through `xp`. Scalar MPI gathers use small NumPy buffers (the own entry filled first, see `MockComm` above) and copy the result back to the active backend.
- `Derham` (PR 7): depends on feectools running on the CuPy backend (struphy-hub/feectools#85, merged). Data that describes the spline spaces (knots, quadrature and projection grids, `spline_types_pyccel`) is host data on every backend; only coefficients and stencil matrices live on the device. `domain_array`, `index_array(_N/_D)` and `neighbours` are gathered with NumPy MPI buffers and converted with `xp.asarray`. `args_derham` is built from the host knots, degrees and starts on NumPy, and from one device copy of them on CuPy. The pyccel scratch arrays (`bn1`, ..., `bd3`) are not part of the CUDA arguments; they become per-thread local arrays in CUDA (PR 10). Not supported on CuPy yet: local projectors (`NotImplementedError` at construction), polar splines (need a spline mapping), and field evaluation (`SplineFunction.__call__` still calls pyccel kernels with device coefficients).

### PR 8: Argument structs in a shared header (complete, open)

Before, every CUDA kernel repeated the full flat signature (26 parameters for markers and domain alone). CuPy does not check it, so adding a field to `MarkerArguments` would have shifted all following arguments of all CUDA kernels **without an error**.

- `kernel_arguments/pusher_args.cuh` defines `struct MarkerArgs`, `DerhamArgs` and `DomainArgs`. A kernel takes them by value, e.g. `void push_eta_linear(double dt, int stage, MarkerArgs args_markers, DomainArgs args_domain)`, and reads `args_markers.markers`, `args_markers.n_markers`, ... The member names are the attribute names of the pyccel classes; the marker view carries its shape and strides, so a separate `n_cols` member is unnecessary.
- Each `Argument` subclass lists its members as `fields = (("double*", "markers"), ...)`, from which `struct_dtype()` builds a NumPy structured dtype with `align=True` (C alignment and padding). Pointer members are `uint64` holding `array.data.ptr`. The struct (a `numpy.void`, which CuPy passes by value) is packed once, in the constructor, so a kernel call does no extra work.
- The struct holds device addresses: it is repacked when an argument object is deepcopied or unpickled (`__getstate__`/`__setstate__`), and owners that reallocate an array must rebuild their argument object, as before.
- Scalar members are range-checked when packing: a `float` for an `int` member raises `TypeError`, a value that does not fit raises `OverflowError`.
- Tests: the header is parsed and compared with `fields` (names, C types, order) without a GPU; on a GPU, an NVRTC-compiled kernel reports `sizeof` and the member offsets, which are compared with the dtype. On the host, `offsetof`/`sizeof` from a C++ compiler agree with the dtypes (x86-64/arm64 lay these structs out like CUDA).
- Why structs and not flat parameters: see the discussion in #653. The struct is what lets a `__device__` helper take one `const DomainArgs&` instead of 12+ forwarded parameters (PR 10), and it mirrors how the pyccel helpers take `args_domain`. What is given up: CuPy's own rejection of host arrays at launch (`_cupy_array()` does it at construction instead) and compatibility with `cupyx.jit`-style kernels (which could take the flat tuple from `fields` if ever needed).
- `_cupy_array()` checks that the array is on the current device (neither CuPy's `RawKernel` nor the struct checks it; it matters with one GPU per MPI rank).

### PR 9: One folder per kernel (complete, open)

- The 43 kernels of `pic/pushing` (`pusher_kernels.py`: 16, `pusher_kernels_gc.py`: 15, `pusher_kernels_sph.py`: 3, `eval_kernels_gc.py`: 5, `eval_kernels_sph.py`: 4) are now in `pic/pushing/kernels/<name>/<name>_kernels.py`; the five old modules are removed. The function bodies are unchanged (checked by comparing the ASTs with the old modules).
- `pusher_utilities_kernels.py` stays a shared module: it holds helpers called by the kernels (boundary conditions), not kernels. It gets its device version in PR 10.
- Each new module imports only what its kernel uses, plus `pusher_args_kernels` as a module (`# do not remove; needed to identify dependencies`): `dependencies.py` only sees imported modules.
- Call sites use the catalog: `catalog["push_eta_stage"]` is passed to `Pusher`, and `KernelSetup` also accepts a `Kernel`, resolved with `get_kernel()` when the setup is created. `CurrentCoupling5DGradB` calls its pusher kernels itself and resolves them once. `Particles` runs the SPH evaluation kernels on its host bundle with `catalog[...].pyccel_kernel`.
- No re-export modules for the old import paths (they would have to contain `kernels` in their name). Code outside struphy that imports e.g. `struphy.pic.pushing.pusher_kernels` has to switch to the catalog.
- `push_gc_bxEstar_discrete_gradient_1st_order_newton` was renamed to `push_gc_bxEstar_dg_1st_order_newton` (Fortran name length); `test_pushing_catalog` checks the layout and the name lengths.
- Import time: loading the catalog (43 compiled modules) takes about 75 ms, once per process.
- `pic/accumulation` (`accum_kernels.py`: 8, `accum_kernels_gc.py`: 8) is split the same way when the first accumulation kernel is ported (PR 12+); the remaining kernel modules as needed.

### PR 10: Device helper functions

The pusher kernels call helpers from other pyccel modules: B-spline evaluation (`bsplines_kernels`, `evaluation_kernels_3d`), mapping evaluation (`geometry/evaluation_kernels`, one module per domain), small linear algebra (`linalg_kernels`), boundary conditions (`pusher_utilities_kernels`). Each needs a `__device__` version in a `.cuh` header before the kernels using it can be ported.

- First: run the PR 8 GPU tests (`test_cuda_struct_layout`, `test_cuda_struct_scalars_are_checked`, the demo kernels) on an H100. Everything below assumes structs arrive by value correctly.
- Port only what `push_eta_stage` needs: `get_spans` and the basis functions for the spline degrees in use, the analytic mapping evaluation (`kind_map` switch, starting with `Cuboid`), the 3×3 linear algebra, and the kinetic boundary conditions. Everything else comes with the kernel that needs it.
- Device helpers take the argument structs by reference (`const DomainArgs&`), exactly like the pyccel helpers take `args_domain`, and read markers through `args.markers(ip, j)`.
- Scratch arrays that the pyccel classes allocate once (e.g. `DerhamArguments.bn1`, ..., `bd3`) become per-thread local arrays in CUDA with a fixed maximum spline degree (compile-time constant, checked against `pn` when the arguments are built) or a template parameter.
- Tests: each device helper is called from a small elementwise test kernel (one thread per evaluation point, inputs and outputs as arrays) and compared with its pyccel version from Python. One generic test-kernel template in `test_kernel_backends.py` (or a new `test_device_helpers.py`) serves all scalar-returning helpers. Tests are skipped without a GPU.
- Headers are included through the source root (`#include "struphy/bsplines/bsplines_kernels.cuh"`). Remember the CuPy cache caveat from PR 2 when editing headers.

### PR 11: First real kernel: `push_eta_stage`

- Write `push_eta_stage_cuda.cu`, using the device helpers from PR 10; one thread per marker, `if (ip >= args_markers.n_markers) return;`.
- Parity: same markers on both backends, results agree to round-off (`rtol ~ 1e-13`). The test is written once against the catalog: it parametrizes over every kernel that has a CUDA version (`[n for n in catalog.names if n not in catalog.missing_cuda]`), builds the arguments with a per-kernel factory, runs the pyccel and the CUDA kernel, and compares every array argument. Later kernels then only add their factory.
- `Pusher` on the CuPy backend passes `n_threads=particles.markers.shape[0]` and device arrays in `args_kernel` (the Butcher tableau arrays `a_stage`, `b`, `c` are tiny NumPy arrays today; `PushEta` makes device copies once at setup).
- End-to-end: run `PushEta` (a propagator that only needs this kernel) with `CUNUMPY_BACKEND=cupy` on 1 and 2 ranks, and check that no host/device transfers happen inside the time loop: `nsys profile` (look for `cudaMemcpy` between the kernel launches) or monkeypatch `cupy.ndarray.get`/`cupy.asarray` to count calls during the loop.

### PR 12+: Port kernels and particle boundary handling

For each kernel: add `<name>_cuda.cu`; the parity test from PR 11 picks it up; `catalog.missing_cuda` shrinks (a `struphy compile --status` line "CUDA kernels: 3 of 43" would be a nice addition).

- Port the remaining particle boundary handling, including the `reflect` helper called from `Particles.apply_kinetic_bc`, so the whole push stays on the device. The CUDA path must use `Particles.args_markers` and `Domain.args_domain`; remove the temporary direct-pyccel use of `Domain._pyccel_args_domain` from this path once reflection runs in a CUDA kernel.
- Accumulation (before the first accumulation kernel): many markers write to the same grid cells. Options are atomics (`atomicAdd` on `double`, native since sm_60) and sort-then-reduce. Start with atomics, which match the pyccel structure one-to-one; measure before optimizing. The accumulated arrays are owned by feectools stencil vectors, which are host arrays on the NumPy backend and device arrays on CuPy (feectools#85), so the kernel writes into `vec._data` directly on both. Split `pic/accumulation` into one folder per kernel at the same time. Accumulation kernels over grid points need their own launch sizes; `CudaKernel` gets a `grid`/`block` override then.
- Field evaluation on the device (`SplineFunction.__call__` and friends) is needed by diagnostics and by some propagators; it reuses the B-spline device helpers from PR 10 with 3D launch shapes.

### PR 18: Geometry evaluation for all analytic mappings

- Device helpers, ported from pyccel one to one: per mapping `<name>_cuda.cuh` next to `<name>_kernels.py` in `geometry/domains/<name>/` (`f` and `df` of Cuboid, Orthogonal, Colella, HollowCylinder, PoweredEllipticCylinder, HollowTorus, ShafranovShift/Sqrt/DshapedCylinder), and in `geometry/evaluation_kernels.cuh` the `kind_map` switch for `f`/`df` plus `det_df`, `df_inv`, `g`, `g_inv`, `select_metric_coeff`. `pull`, `push`, `tran` go into `geometry/transform_kernels.cuh`. Spline mappings (`kind_map` 0–2) trap on the device until PR 19.
- Entry kernels: `<name>_cuda.cu` and `<name>_test_args.py` in the four `geometry/kernels/<name>/` folders (one thread per marker, or per grid point with `N_THREADS`). The parity arguments loop over every analytic mapping, so a transposed `DF` (invisible with Cuboid's diagonal Jacobian) is caught, also by the CPU emulation.
- Device helper tests for the metric chain, against the pyccel helpers, as in PR 10.
- With `df`/`df_inv` defined for every analytic mapping, the Cuboid-only checks in `Pusher`, `_accumulation_kernel` and `Particles` (reflection) are replaced by a check for spline mappings.
- The GPU tests are not run on the H100 in this PR; CPU emulation and the CPU regression tests are the gate.

### PR 19: Spline mappings

- `DomainArgs`: `ind1..3` become `Array2D<long long>` and `cx/cy/cz` `Array3D<double>` (the device needs their shapes); `CudaDomainArguments.fields` and the generated `pusher_args.cuh` change, the pyccel `DomainArguments` does not.
- Spline-mapped domains (`IGAPolarCylinder`, `IGAPolarTorus`, `Tokamak`, GVEC, DESC) can be created on the CuPy backend (the PR 5 leftover: polar splines and feectools arrays).
- Device versions of `spline_3d(_df)`, `spline_2d_straight(_df)`, `spline_2d_torus(_df)` in `geometry/spline_mappings_kernels.cuh`, using the B-spline helpers of PR 10; the parity arguments of the geometry kernels add the spline mappings.

## Moving to cunumpy (PR 13)

cunumpy 0.5.0 contains equivalents of everything struphy built in PRs 1–8, with more checks. The move was deliberately postponed until a few kernels ran on the GPU: keeping the small struphy classes while the design settled made them easier to understand and change, and the kernels never see the Python side, so nothing written in PRs 10–12 had to change beyond includes and argument names. The table uses the names of the 0.4.0 plan; in 0.5.0 the classes live in the submodules `cunumpy.kernels`, `cunumpy.cuda`, `cunumpy.kernel_testing` and `cunumpy.profiling`.

| struphy (PRs 1–12) | cunumpy |
|---|---|
| `utils.kernel_backends.Kernel`, `KernelCatalog`, `CudaKernel` | `xp.Kernel`, `xp.KernelCatalog.from_package` (same folder convention, source root on the include path by default), `xp.CudaKernel` (parses the `__global__` signature once and checks every call: argument count, array dtypes and contiguity, scalars cast and range-checked; 1D–3D launches via `n_threads` or `grid`; header-aware compile cache; `compile()` at setup; debug mode with `-lineinfo`, bounds checks and synchronized launches) |
| `utils.cuda_arguments.Argument` + `_pack()` | `xp.CudaStructArguments` (`struct_name`, `fields`, `pack()`, repack on copy/unpickle, `cls.struct.declaration`/`to_header()`), `xp.CudaStruct.from_signature` to derive a struct from pyccel-style annotations |
| the two bundles per owner (`args_*` and `_pyccel_args_*`) | `xp.KernelArguments`: one object with `__host_args__()` (the pyccel instance) and `__cuda_args__()` (the packed struct), resolved by `Kernel` and `PyccelKernel`, so call sites never branch on the backend |
| Local `Array2D<double>` marker view | `Array2D<double>` view (pointer, shape, strides) from `cunumpy/array_view.cuh`; kernels write `markers(ip, j)` |
| the "reference or copy once" rule, written by hand | `xp.as_device_array(value, dtype, ndim)` |
| the parity test of PR 11 | `cunumpy.testing.assert_kernels_agree`, `KernelCatalog.parity_cases()`, `requires_cupy`, `backend` fixture |
| the device-helper test kernels of PR 10 | `cunumpy.testing.device_function_kernel(header, prototype)` |
| transfer check of PR 11 | `xp.count_transfers()`, `xp.assert_no_transfers()` |
| atomics, thread-index macros | `cunumpy/atomic.cuh`, `cunumpy/index.cuh` |
| one GPU per MPI rank | `xp.bind_local_device()`, `xp.require_cuda_aware_mpi()`, `xp.synchronize_for_mpi()` |

Steps (status in PR 13):

1. Depend on cunumpy 0.5.0 from PyPI in `pyproject.toml`. **Done.**
2. Delete `utils/kernel_backends.py`; the catalog `__init__.py` files, `Pusher` and `KernelSetup` use the cunumpy classes. **Done.**
3. The three argument classes subclass `PyccelStructArguments` and provide `__host_args__()`, becoming the single `args_*` object of their owner on both backends; the `_pyccel_args_*` bundles and the backend branches that select between them are gone. The pyccel class stays a plain compiled class built from the same arrays (it cannot inherit from anything). **Done; replaced in PR 16** by one pyccel and one CUDA class per argument type.
4. `pusher_args.cuh` is generated from the Python `fields` (`write_pusher_header()`); `test_generated_header` asserts the committed header equals the generated one. **Done.** Markers use `Array2D<double>` and spline coefficients use `Array3D<double>` from `cunumpy/array_view.cuh`; cunumpy packs both views and replaces struphy's own `array_view.cuh`.
5. Replace the hand-written parity, device-helper and transfer tests by the cunumpy helpers (`assert_kernels_agree`, `parity_cases()`, `device_function_kernel`, `assert_no_transfers`, `requires_cupy`). **Done.**

What this buys, beyond less code in struphy: scalar checks against the kernel signature, launch shapes for grid kernels, the include-hash compile cache, debug mode, and one argument object per owner instead of two.

## feectools

Struphy's FEEC data (Derham spaces, stencil vectors and matrices, the MPI ghost-region exchange) lives in
[feectools](https://github.com/struphy-hub/feectools). Its CUDA support is a stack of PRs on branches
`cuda-<n>-<topic>`, each targeting `devel-tiny` and reviewed commit by commit; `cuda-development` collects them:

| feectools PR | What | Struphy needs it for |
|---|---|---|
| [#85](https://github.com/struphy-hub/feectools/pull/85) `cuda-1-xp-arrays` (merged into `cuda-development`) | feectools runs on the CuPy backend: stencil data are `xp` arrays | `Derham` on CuPy (PR 7) and every struphy GPU test that builds a `Derham` |
| [#86](https://github.com/struphy-hub/feectools/pull/86) `cuda-2-mpi-sync` | MPI with device buffers (CUDA-aware MPI, `synchronize_for_mpi` before every MPI call) | runs on more than one rank on CuPy: ghost-region exchange, reductions |
| [#87](https://github.com/struphy-hub/feectools/pull/87) `cuda-3-device-binding` | one GPU per MPI rank (`bind_local_device` before MPI starts) | multi-GPU runs |
| [#88](https://github.com/struphy-hub/feectools/pull/88) `cuda-4-device-kernels` | stencil `dot`, `transpose`, `inner`, `axpy` on the device (matvec 0.5 ms instead of 165 ms on an H100) | field solves in the time loop without host copies (e.g. Poisson and Ampère in Vlasov–Ampère) |
| [#90](https://github.com/struphy-hub/feectools/pull/90) `cuda-development` | integrates the stack into `devel-tiny` | the feectools release struphy pins |

**When.** Struphy follows the top of the stack through the `feectools` submodule since PR 17, so every later struphy
CUDA PR (and the first end-to-end model run) is tested against it. PR 18 and PR 19 do not depend on its changes:
they port struphy kernels, run on one rank, and are gated by the CPU emulation. Before the struphy CUDA stack is
merged into `devel`, the feectools stack is merged into `devel-tiny` (bottom up), released, and the submodule and
the `pyproject.toml` pin (today `feectools>=0.3.0, <=0.3.0`) point at that release again; the CI check
`pr-feectools-submodule` (submodule = latest `devel-tiny`) fails on the CUDA PRs until then, by design.

**Before merging the stack:** it imports `CudaKernel`, `CudaKernelVariants`, `bind_local_device` and
`synchronize_for_mpi` from the top level of cunumpy and requires `cunumpy>=0.3.0`. In cunumpy 0.5 these names are
deprecated (removed in 0.6): import them from `cunumpy.cuda` and `cunumpy.mpi`, and require `cunumpy>=0.5.0, <0.6`
like struphy.

New feectools work for the GPU follows the same pattern: a `cuda-<n>-<topic>` PR, linked in the table above and
from the struphy tracking issue ([#650](https://github.com/struphy-hub/struphy/issues/650)).

## Porting order

Kernels are ported in the order that completes one model after the other on the GPU. Each step
lists only the kernels it adds; P = pusher, A = accumulation, E = marker evaluation. Status:
done (✓), in this PR (PR 14), blocked (⏸).

**Step 0 – Vlasov** (PRs 11–12, ✓): `push_eta_stage`, `push_vxb_analytic`, `push_vxb_implicit`, `reflect`.

**Geometry** (PR 18, PR 19): `kernel_evaluate_pic`, `kernel_evaluate`, `kernel_pullpush_pic`, `kernel_pullpush` with the metric chain, first for all analytic mappings, then for spline mappings. Needed by every particle run on the GPU (weight initialization) and by the diagnostics; it also lifts the Cuboid-only restriction of all CUDA kernels.

**Step 1 – Vlasov–Ampère, Vlasov–Maxwell, ColdPlasmaVlasov**

| # | Kernel | Type | Used by | Status |
|---|---|---|---|---|
| 1 | `push_v_with_efield` | P | `VlasovAmpereCoupling`, `PushVinForceField` | PR 14 |
| 2 | `charge_density_0form` | A (vector) | initial Poisson solve of these models | PR 14 |
| 3 | `vlasov_maxwell` | A (matrix + vector) | `VlasovAmpereCoupling` | ⏸ 6D views |

**Step 2 – LinearVlasovAmpère/Maxwell (δf)**

| # | Kernel | Type | Used by | Status |
|---|---|---|---|---|
| 4 | `push_weights_with_efield_lin_va` | P | `EfieldWeightsCoupling` | PR 14 |
| 5 | `linear_vlasov_ampere` | A (matrix + vector) | `EfieldWeightsCoupling` | ⏸ 6D views |

**Step 3 – Guiding center: ToyDrift, then DriftKineticElectrostaticAdiabatic.** The default
algorithm is `discrete_gradient_1st_order`; its chain comes first.

| # | Kernel | Type |
|---|---|---|
| 6 | `driftkinetic_hamiltonian` | E |
| 7 | `bstar_parallel_3form` | E |
| 8 | `unit_b_1form` | E |
| 9 | `push_gc_bxEstar_discrete_gradient_1st_order` | P |
| 10 | `gc_density_0form` | A (vector) — ToyDrift complete |
| 11 | `bstar_2form` | E |
| 12 | `push_gc_Bstar_discrete_gradient_1st_order` | P — DKEA complete |
| 13 | `grad_driftkinetic_hamiltonian` | E (Newton variants) |
| 14–15 | `push_gc_bxEstar_dg_1st_order_newton`, `push_gc_Bstar_discrete_gradient_1st_order_newton` | P |
| 16–17 | `push_gc_bxEstar_discrete_gradient_2nd_order`, `push_gc_Bstar_discrete_gradient_2nd_order` | P |
| 18–19 | `push_gc_bxEstar_explicit_multistage`, `push_gc_Bstar_explicit_multistage` | P |

These are the largest kernels; evaluate code generation (see [Open questions](#open-questions)) before porting 13–19 by hand.

**Step 4 – Hybrid MHD–kinetic 6D (current and pressure coupling)**

| # | Kernel | Used by |
|---|---|---|
| 20 | `cc_lin_mhd_6d_1` (A) | `CurrentCoupling6DDensity` |
| 21–23 | `push_bxu_Hcurl`, `push_bxu_Hdiv`, `push_bxu_H1vec` (P) | `CurrentCoupling6DCurrent` |
| 24 | `cc_lin_mhd_6d_2` (A) | `CurrentCoupling6DCurrent` |
| 25–27 | `push_pc_eta_stage_Hcurl`, `push_pc_eta_stage_Hdiv`, `push_pc_eta_stage_H1vec` (P) | `PushEtaPC` |
| 28–29 | `push_pc_GXu` (P), `pc_lin_mhd_6d` (A) | `PressureCoupling6D` |
| 30–31 | `push_pc_GXu_full` (P), `pc_lin_mhd_6d_full` (A) | `PressureCoupling6D` (full) |

The three spaces (H1vec/Hcurl/Hdiv) differ only in the basis; port one, then the other two.

**Step 5 – Hybrid 5D**

| # | Kernel |
|---|---|
| 32 | `gc_mag_density_0form` (A) |
| 33 | `cc_lin_mhd_5d_D` (A) |
| 34–36 | `cc_lin_mhd_5d_curlb` (A), `push_gc_cc_J1_Hdiv`, `push_gc_cc_J1_H1vec` (P) |
| 37–39 | `cc_lin_mhd_5d_gradB` (A), `push_gc_cc_J2_stage_Hdiv`, `push_gc_cc_J2_stage_H1vec` (P) |
| 40–43 | `cc_lin_mhd_5d_gradB_dg_init`, `cc_lin_mhd_5d_gradB_dg` (A), `push_gc_cc_J2_dg_init_Hdiv`, `push_gc_cc_J2_dg_Hdiv` (P) |
| 44 | `cc_lin_mhd_5d_M` (A) |

**Step 6 – Diffusion and SPH (last)**: `push_random_diffusion_stage` (needs device RNG),
`push_deterministic_diffusion_stage`; then, after box sorting runs on the device,
`sph_pressure_coeffs`, `sph_mean_velocity_coeffs`, `sph_viscosity_tensor`, `sph_isotherm_kappa`,
`push_v_sph_pressure`, `push_v_sph_pressure_ideal_gas`, `push_v_viscosity`, `div_u_weak_1form`.

Infrastructure that gates the steps, independent of the kernels: mappings other than Cuboid (every
CUDA kernel rejects other mappings at setup until PR 18 (analytic) and PR 19 (spline)), multi-rank marker sorting without host round trips,
and array views with more than 4 dimensions in cunumpy (all matrix accumulations write 6D stencil
matrix data).

## Testing

- Every kernel with a CUDA version has a parity test (pyccel vs. CUDA, same inputs). Since PR 15 its arguments live in `<name>_test_args.py` next to the kernel (`make_args(backend, seed)`, `CASES`, tolerances), used by the GPU test (`check_parity`) and by the CPU emulation test.
- Each device helper is tested against its pyccel version through a small test kernel (PR 10).
- CUDA tests are skipped when no GPU is available (`cunumpy.kernel_testing.requires_cupy`), so the normal CI keeps working. Until a GPU runner exists, the GPU tests are run by hand on an H100 before a PR that touches CUDA code is merged, and the PR description says so.
- Regression tests on the CPU (`pic/tests/test_pushers.py`, `pic/tests/test_kernel_setup.py`, the model tests, ...) must pass in every PR. Particle tests need an MPI launcher (`mpirun -n 1 pytest ...`).

## Open questions

- **Scalar types.** Members of the argument structs are checked when they are packed (PR 8). Scalars passed directly to a kernel (`dt`, `stage`) are not checked against the kernel signature: a Python `float` for an `int` parameter arrives wrong without an error. Either `CudaKernel` parses the `extern "C"` signature once and casts/checks (what cunumpy's `CudaKernel` does), or we accept it until the move to cunumpy. Until then: kernels take `double dt, int stage` in exactly this order, and the pyccel signature is the reference.
- **Kernel launch configuration.** One thread per marker with a fixed block size for now. Kernels over grid points (accumulation, FEEC) need their own launch sizes; `CudaKernel` needs a `grid`/`block` override (PR 12+), or cunumpy's.
- **Accumulation strategy.** Atomics vs. sort-then-reduce; see PR 12+. Decided by measurement on the first accumulation kernel.
- **Marker layout.** The markers array is row-major (`n_markers × n_cols`). With one thread per marker, the memory accesses are strided. This is fine for now (each thread reads a few neighbouring columns), but a column-major or struct-of-arrays layout may be faster later. This would affect the CPU code too, so it is out of scope here. The array view represents strides explicitly on the CUDA side.
- **MPI + GPUs.** One GPU per MPI rank (`xp.bind_local_device()` before `MPI_Init`, with feectools#86/#87), and GPU-aware MPI for the marker exchange, so markers do not go through the host. `xp.mpi_is_cuda_aware()` detects it; the exchange in `Particles.mpi_sort_markers` has to be checked for host staging buffers.
- **Single-source alternatives.** Hand-written CUDA stays the default. Generating whole kernels from the Python source (`cupyx.jit`, numba-cuda, or a pyccel CUDA backend) is worth a look before the guiding-center kernels (the largest ones) are ported. Those tools take flat arguments, which `fields` also provides.
- **Spline mappings and polar splines on the GPU** (PR 5/7 leftovers): needed once a model with an IGA mapping is run on the GPU.

## PR 10 implementation notes

Device headers now provide Cuboid Jacobians, matrix inversion/vector products,
spline span and N/D basis evaluation (degrees 1–8), and per-marker boundary
conditions. `DerhamArgs` carries knot lengths because raw device pointers have
no shape. Argument construction checks spline degrees and the current device.
GPU parity tests cover non-block-aligned batches, domain endpoints, and mixed
boundaries. H100 struct and helper validation is still required; the development
workspace is macOS and has no CUDA device.

## PR 11 implementation notes

`push_eta_stage` now has a CUDA implementation for Cuboid. Both signatures take
an explicit final `n_stages` argument because a raw CUDA tableau pointer has no
length. PushEta prepares tableau arrays once; Pusher passes the current marker
count at launch and rejects unsupported mappings during setup. Empty launches
are no-ops. Catalog parity tests require a factory for every ported kernel and
cover holes, boundary particles, Euler/RK4 and periodic/reflect/remove boundaries.
The single-rank pusher test guards host array conversions after warmup.
H100 execution and the two-rank no-transfer requirement remain unverified;
existing MPI sorting still synchronizes dynamic counts with the host.

## PR 12 implementation notes

Scoped to the Vlasov model: both magnetic rotation algorithms now have CUDA
versions, using tensor-product N/D spline evaluation and strided coefficient
views. Reflection and spline evaluation are called like every other kernel: the
owning class builds a `Kernel(PyccelKernel(...), CudaKernel.from_file(...))` pair in
its `__init__` and calls it with the same arguments on both backends.
`Particles` builds the `reflect` pair when a direction has reflecting boundaries
and calls it as `reflect(markers, args_domain, outside_inds, axis)` (issue #675);
CUDA receives marker and integer-index array views, so it needs neither a marker
argument bundle nor a separate index count. The Cuboid-only check of the CUDA
version is done once, when the particles are created.
`SplineFunction` builds the `eval_spline_mpi_markers`, `eval_spline_mpi_matrix` and
`eval_spline_mpi_sparse_meshgrid` pairs and, once per component, the arguments
`(kind, pn, tn1, tn2, tn3, starts)` as arrays of the active backend, which both
versions take unchanged; spline degrees outside 1–8 are rejected there on CuPy.
CUDA array views carry coordinate, coefficient and output shapes/strides;
sparse grids are evaluated directly without broadcasting or flattening coordinates.
The `_evaluate_cuda` helper and its separate CUDA argument convention are removed
(issue #674).
Accumulation is deliberately left for the next Vlasov–Ampere/Maxwell step.
GPU tests are provided but not run here, at the maintainer's request.

## PR 13 implementation notes

Struphy's dispatch code (`utils/kernel_backends.py`, `kernel_arguments/array_view.cuh`) is
deleted; `cunumpy.kernels` and `cunumpy.cuda` replace it. Every `CudaKernel` is built with
`CUDA_OPTIONS`, so it knows the three structs and has the struphy source root on its include
path. The argument classes check dtype, contiguity and device of each owner array without
copying; `CudaDomainArguments` sets `host_copies = True` because geometry evaluations on the
CuPy backend still call pyccel with host copies. pyccel getters return a new NumPy view of the
same buffer on every access, so tests compare data pointers, not object identity.
`from cunumpy import PyccelKernel` is deprecated in cunumpy 0.5 (removed in 0.6); all imports use
`cunumpy.kernels`. GPU tests are provided but not run here (no CUDA device on the development
machine).

## PR 14 implementation notes

`accum_kernels.py` and `accum_kernels_gc.py` are split into `pic/accumulation/kernels/<name>/`
(16 kernels) with a catalog, like `pic/pushing` in PR 9; models and propagators take
`accum_catalog["<name>"]`. `Accumulator` and `AccumulatorVector` select the kernel once, as
`Pusher` does, and launch CUDA kernels with one thread per marker row (the pyccel loop runs over
all rows and skips holes). The CUDA fillers (`filler_kernels.cuh`) add with `cunumpy_atomic_add`;
the summation order differs from the serial loop, so GPU parity uses `rtol=1e-12`.
`push_v_with_efield`'s last parameter is renamed from `const` (a C++ keyword) to `const_factor`;
callers pass it positionally.

Without a GPU, `test_cuda_emulation.py` compiles each CUDA kernel as C++ (cunumpy's
`emulate_cuda_kernel`) and compares it with pyccel. cunumpy's emulator does not take struct
parameters, so `pic/tests/cuda_emulation.py` emulates a generated wrapper that takes the struct
fields one by one and rebuilds the structs. Emulation runs threads serially, so it does not test
races; with Cuboid's diagonal Jacobian it cannot catch a transposed `DF`.

Blocked: `vlasov_maxwell` and `linear_vlasov_ampere` accumulate into 6D stencil matrix data, and
cunumpy's views stop at `Array4D`. Adding `Array5D`/`Array6D` in cunumpy unblocks every matrix
accumulation. GPU tests are provided but not run here.

## PR 15 implementation notes

The kernels are plain imports now. Each of the 59 kernel folders declares its kernel in its
`__init__.py` with `Kernel.from_folder(__name__, structs=CUDA_STRUCTS)`; importing one kernel no longer
imports all 43 compiled modules of a package. `kernels/__init__.py` is documentation only.

A `Kernel` calls the version of the active backend itself, so `Pusher`, `KernelSetup`, the
accumulators, `SplineFunction` (spline evaluation) and `Particles` (reflection) keep the `Kernel` and call it. `prepare_kernel()` (in `utils/cuda_arguments.py`) checks it
at setup instead of `get_kernel()`: on CuPy a kernel without CUDA version raises there, and the CUDA kernel
is compiled. Launch sizes are not passed any more where cunumpy infers them: the first array of a pusher
or accumulation call is `args_markers.markers`, so one thread per marker row. `KernelSetup` passes
`n_threads=args_markers.n_markers` because its first array is `alpha`; `reflect` and the spline evaluation
pass theirs explicitly.

Parity tests follow cunumpy's convention: `<name>_test_args.py` in the kernel folder defines
`make_args(backend, seed)`; struphy adds `CASES` (the test cases, selected by `seed`) and the tolerances.
`test_cuda_parity.py` runs `check_parity` for every case of every kernel with a CUDA version, and
`test_cuda_emulation.py` runs the same cases through the CPU emulation. Tests that go through all kernels
build a `KernelCatalog.from_package` of each package; `test_folders_declare_their_kernels` checks that
the folders declare the same kernels.

Built and tested against cunumpy `devel` (`d2e9235`).

## PR 16 implementation notes

The argument classes come in pairs, one for each backend:

| CPU (pyccel) | GPU (CUDA struct) |
| --- | --- |
| `pusher_args_kernels.MarkerArguments` | `pusher_args_cuda.CudaMarkerArguments` (`MarkerArgs`) |
| `pusher_args_kernels.DerhamArguments` | `pusher_args_cuda.CudaDerhamArguments` (`DerhamArgs`) |
| `pusher_args_kernels.DomainArguments` | `pusher_args_cuda.CudaDomainArguments` (`DomainArgs`) |
| `local_projectors_args_kernels.LocalProjectorsArguments` | `local_projectors_args_cuda.CudaLocalProjectorsArguments` (`LocalProjectorsArgs`) |

The two classes of a pair take the same constructor arguments and have the same attributes; the CUDA struct
adds only derived members that CUDA pointers cannot carry (`n_markers` is also a pyccel attribute; `nt1`,
`nt2`, `nt3` are the knot lengths) and leaves out the pyccel scratch arrays (`bn1`, ..., `bd3`), which are
per-thread arrays in the kernels. `test_argument_classes_correspond` checks this for every pair. The CUDA
classes subclass cunumpy's `CudaStructArguments`, accept only CuPy arrays and pack their struct in the
constructor; the pyccel classes are unchanged.

`Particles`, `Derham`, `Domain` and the local projectors create one of the two classes, depending on the
backend, so on NumPy `args_markers` is a `MarkerArguments` and pyccel kernels receive it as it is. There
are no host copies and no `__host_args__()` any more. Every kernel that takes argument objects is called
through `Kernel(PyccelKernel(...))`, built at the call site where the owner is pickled (`Kernel` objects
cannot be pickled) and kept by `Pusher`, `KernelSetup`, the accumulators, reflection and the spline
evaluation; a kernel without a CUDA version raises `NotImplementedError` when it is called on CuPy.
`prepare_kernel()` (PR 15) is removed: kernels are no longer checked or compiled at setup, CUDA kernels
compile on their first call. When to compile them (e.g. once at simulation start) is decided later.

Consequence: geometry evaluations (`Domain.__call__`, `jacobian_det`, `pull`/`push`, ...) have no CUDA
version yet and raise on CuPy. `Particles` evaluates `jacobian_det` when it initializes weights, so
**particle runs on the CuPy backend fail at setup until the geometry evaluation kernels are ported**. Next
step: CUDA versions of `kernel_evaluate_pic`, `kernel_evaluate` and the pull/push kernels for `Cuboid`.

`CudaLocalProjectorsArguments` is not used by any CUDA kernel yet; local projectors are rejected on the
CuPy backend by `Derham`.

### Kernel folders

Every kernel that Python code calls with argument objects now lives in its own folder, like the pusher and
accumulation kernels since PR 9/14: `<package>/kernels/<name>/` with `<name>_kernels.py` (pyccel),
`<name>_cuda.cu` (if ported), `<name>_test_args.py` (if ported) and an `__init__.py` that declares
`<name> = Kernel.from_folder(__name__, structs=CUDA_STRUCTS)`. The code imports the kernel and calls it, with
no `Kernel(PyccelKernel(...))` at call sites. New packages: `pic/diagnostics/kernels` (marker energies,
moments, guiding-center coordinates, from `pic/utilities_kernels.py`), `pic/sph/kernels` (from
`pic/sph_eval_kernels.py`), `bsplines/kernels` (`eval_spline_mpi_markers/_matrix/_sparse_meshgrid`, from
`bsplines/evaluation_kernels_3d.py`), `geometry/kernels` (`kernel_evaluate(_pic)`, `kernel_pullpush(_pic)`),
`feec/kernels` (`hybrid_weight`) and `feec/local_projectors/kernels`; `reflect` joins `pic/pushing/kernels`.
Only entry kernels moved; the `@pure` helpers they call stay in the shared modules and are imported from
there.

The CUDA spline evaluation is split accordingly: the shared device function `eval_spline_mpi` is in
`bsplines/evaluation_kernels_3d.cuh`, and each folder has its `__global__` kernel. All packages are in
`test_cuda_parity.PACKAGES`, so their signatures are checked, and every CUDA kernel has a parity test and a
CPU emulation test; `N_THREADS` in a `<name>_test_args.py` sets the launch size where it is not one thread
per row of the first array (spline evaluation on grids, `reflect`).
