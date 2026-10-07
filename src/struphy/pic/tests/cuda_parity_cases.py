"""Inputs for the pyccel/CUDA parity tests: for every kernel with a CUDA version, the cases it is checked on.

Test-only: ``test_cuda_parity.py`` (on a GPU) and ``test_cuda_emulation.py`` (without one) run each case on both
kernel versions and compare every array. ``build(case)`` returns the kernel's positional arguments on the active
backend; a kernel that is ported to CUDA gets an entry here.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np

from struphy.geometry.base import inside_logical_cube
from struphy.geometry.domains import Cuboid
from struphy.ode.utils import ButcherTableau
from struphy.pic.tests.kernel_test_args import (
    BOUNDARY_CONDITIONS,
    N_ANALYTIC_DOMAINS,
    N_GEOMETRY_DOMAINS,
    N_MARKERS,
    analytic_domains,
    butcher_arguments,
    derham_arguments,
    evaluation_grid,
    geometry_domain,
    logical_markers,
    marker_arguments,
    spline_coefficients,
    spline_evaluation_arguments,
    v1_symm_accumulation_data,
)


@dataclass(frozen=True)
class ParityCases:
    """The cases of one kernel, how to build its arguments, and how to compare."""

    cases: Sequence[Any]
    build: Callable[[Any], tuple]
    """The kernel's positional arguments for one case, on the active backend."""
    rtol: float = 1e-12
    atol: float = 0.0
    n_threads: Callable[[tuple], int] | None = None
    """Launch size, if it is not one thread per row of the first array."""


# ---------------------------------------------------------------- pushers


def push_eta_stage_args(case):
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage

    bc, method, stage = case
    args = (*marker_arguments(bc), *butcher_arguments(method))
    # reach this stage independently on each backend
    for previous in range(stage):
        push_eta_stage(0.2, previous, *args)
    return (0.2, stage, *args)


def push_weights_with_efield_lin_va_args(bc):
    f0_values = xp.asarray(np.random.default_rng(13).random(N_MARKERS))
    return (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients(), f0_values, 1.3, 0.8)


def reflect_args(axis):
    """Reverse one logical velocity component of 10 markers in a Cuboid."""
    markers = np.random.default_rng(32).random((140, 8))
    outside_inds = np.arange(139, 129, -1, dtype=np.int64)
    return (xp.asarray(markers), Cuboid(r1=2.0, r2=3.0, r3=4.0).args_domain, xp.asarray(outside_inds), axis)


# ---------------------------------------------------------------- accumulation


def charge_density_0form_args(bc):
    args_markers, args_domain = marker_arguments(bc)
    return (args_markers, derham_arguments(), args_domain, xp.zeros((18, 20, 16)))


def linear_vlasov_ampere_args(domain):
    """129 markers (a hole and a boundary particle) in domain `domain` of geometry_domain(); zeroed V1 data."""
    args_markers, _ = marker_arguments((0, 0, 0))
    f0_values = xp.asarray(np.random.default_rng(14).random(N_MARKERS))
    return (
        args_markers,
        derham_arguments(),
        geometry_domain(domain).args_domain,
        *v1_symm_accumulation_data(),
        f0_values,
    )


# ---------------------------------------------------------------- spline evaluation


def eval_spline_mpi_markers_args(kind):
    """Spline values at 129 markers, one of them flagged."""
    markers = np.random.default_rng(5).random((129, 3))
    markers[0, 0] = -1.0  # not on the process domain: skipped
    return (xp.asarray(markers), *spline_evaluation_arguments(kind), xp.zeros(129))


def eval_spline_mpi_grid_args(sparse):
    """Spline values on a 7 x 5 x 4 grid (full or sparse meshgrid), one point flagged."""
    return lambda kind: (*evaluation_grid(sparse=sparse), *spline_evaluation_arguments(kind), xp.zeros((7, 5, 4)))


# ---------------------------------------------------------------- geometry

# metric coefficients at 129 markers: every mapping for F, det(DF), DF^(-1) and G^(-1); the identity, DF and G for
# a diagonal, two non-diagonal analytic and three spline mappings; remove_outside and avoid_round_off alternate.
# (index into geometry_domain(), kind_coeff, remove_outside, avoid_round_off)
KERNEL_EVALUATE_PIC_CASES = tuple(
    (domain, kind_coeff, (domain + n) % 2 == 0, n % 2 == 1)
    for domain in range(N_GEOMETRY_DOMAINS)
    for n, kind_coeff in enumerate((0, 2, 3, 5))
) + tuple((domain, kind_coeff, kind_coeff == 1, True) for domain in (1, 2, 6, 10, 11, 12) for kind_coeff in (-1, 1, 4))


def kernel_evaluate_pic_args(case):
    domain, kind_coeff, remove_outside, avoid_round_off = case
    markers = logical_markers()
    mat_f = np.random.default_rng(6).random((markers.shape[0], 3, 3))
    return (
        xp.asarray(markers),
        kind_coeff,
        geometry_domain(domain).args_domain,
        xp.asarray(mat_f),
        remove_outside,
        avoid_round_off,
    )


# metric coefficients on a 7 x 5 x 4 grid: every mapping on a full and a sparse meshgrid, kind_coeff -1 to 5 and
# avoid_round_off varying. (index into geometry_domain(), kind_coeff, is_sparse_meshgrid, avoid_round_off)
KIND_COEFFS = (-1, 0, 1, 2, 3, 4, 5)
KERNEL_EVALUATE_CASES = tuple(
    (domain, KIND_COEFFS[(2 * domain + sparse) % len(KIND_COEFFS)], bool(sparse), (domain + sparse) % 2 == 0)
    for domain in range(N_GEOMETRY_DOMAINS)
    for sparse in (0, 1)
)


def kernel_evaluate_args(case):
    domain, kind_coeff, is_sparse_meshgrid, avoid_round_off = case
    eta1, eta2, eta3 = evaluation_grid(sparse=is_sparse_meshgrid)
    mat_f = np.random.default_rng(8).random((7, 5, 4, 3, 3))
    return (
        eta1,
        eta2,
        eta3,
        kind_coeff,
        geometry_domain(domain).args_domain,
        xp.asarray(mat_f),
        is_sparse_meshgrid,
        avoid_round_off,
    )


# every kind_fun of pull (0, 1, 10, 11, 12), push (the same) and tran (0, 1, 10-21) once, cycling through the
# analytic mappings; each spline mapping gets three further transforms
PULLPUSH_KINDS = (0, 1, 10, 11, 12)
TRAN_KINDS = (0, 1, *range(10, 22))
TRANSFORMS = (
    tuple((0, k) for k in PULLPUSH_KINDS) + tuple((1, k) for k in PULLPUSH_KINDS) + tuple((2, k) for k in TRAN_KINDS)
)
N_SPLINE_DOMAINS = N_GEOMETRY_DOMAINS - N_ANALYTIC_DOMAINS

# (index into geometry_domain(), kind_transform, kind_fun, a_has_holes, remove_outside)
KERNEL_PULLPUSH_PIC_CASES = tuple(
    (n % N_ANALYTIC_DOMAINS, kind_transform, kind_fun, n % 3 != 2, n % 2 == 0)
    for n, (kind_transform, kind_fun) in enumerate(TRANSFORMS)
) + tuple(
    (N_ANALYTIC_DOMAINS + n // 3, kind_transform, kind_fun, n % 3 != 2, n % 2 == 1)
    for n, (kind_transform, kind_fun) in enumerate(
        TRANSFORMS[(7 * j + 3 * s) % len(TRANSFORMS)] for s in range(N_SPLINE_DOMAINS) for j in range(3)
    )
)


def kernel_pullpush_pic_args(case):
    """Every third case passes `a` without holes (one row per inside marker)."""
    domain, kind_transform, kind_fun, a_has_holes, remove_outside = case
    markers = logical_markers()
    rng = np.random.default_rng(9)
    n_comp = 1 if kind_fun < 10 else 3
    a = rng.normal(size=(markers.shape[0], n_comp))
    if not a_has_holes:
        a = a[inside_logical_cube(markers)]
    out = rng.random((markers.shape[0], 3))
    return (
        xp.asarray(a),
        xp.asarray(markers),
        kind_transform,
        kind_fun,
        geometry_domain(domain).args_domain,
        xp.asarray(out),
        remove_outside,
    )


# (index into geometry_domain(), kind_transform, kind_fun, is_sparse_meshgrid)
KERNEL_PULLPUSH_CASES = tuple(
    ((3 * n) % N_ANALYTIC_DOMAINS, kind_transform, kind_fun, n % 2 == 1)
    for n, (kind_transform, kind_fun) in enumerate(TRANSFORMS)
) + tuple(
    (N_ANALYTIC_DOMAINS + n // 3, kind_transform, kind_fun, n % 2 == 0)
    for n, (kind_transform, kind_fun) in enumerate(
        TRANSFORMS[(8 * j + 3 * s + 1) % len(TRANSFORMS)] for s in range(N_SPLINE_DOMAINS) for j in range(3)
    )
)


def kernel_pullpush_args(case):
    domain, kind_transform, kind_fun, is_sparse_meshgrid = case
    eta1, eta2, eta3 = evaluation_grid(sparse=is_sparse_meshgrid)
    rng = np.random.default_rng(10)
    n_comp = 1 if kind_fun < 10 else 3
    return (
        xp.asarray(rng.normal(size=(7, 5, 4, n_comp))),
        eta1,
        eta2,
        eta3,
        kind_transform,
        kind_fun,
        geometry_domain(domain).args_domain,
        is_sparse_meshgrid,
        xp.asarray(rng.random((7, 5, 4, 3))),
    )


# ---------------------------------------------------------------- all kernels with a CUDA version

PUSHER_TOLERANCES = {"rtol": 1e-13, "atol": 1e-14}
GEOMETRY_TOLERANCES = {"rtol": 1e-10, "atol": 1e-10}
SPLINE_KINDS = ((0, 0, 0), (1, 0, 1))


def rows_of(index):
    """One thread per row of argument `index`."""
    return lambda args: args[index].shape[0]


def size_of(index, per_thread=1):
    """One thread per `per_thread` entries of argument `index`."""
    return lambda args: args[index].size // per_thread


PARITY_CASES = {
    "push_eta_stage": ParityCases(
        [
            (bc, method, stage)
            for bc in BOUNDARY_CONDITIONS
            for method in ("forward_euler", "rk4")
            for stage in range(ButcherTableau(method).n_stages)
        ],
        push_eta_stage_args,
        **PUSHER_TOLERANCES,
    ),
    "push_v_with_efield": ParityCases(
        BOUNDARY_CONDITIONS,
        lambda bc: (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients(), 0.7),
        **PUSHER_TOLERANCES,
    ),
    "push_vxb_analytic": ParityCases(
        BOUNDARY_CONDITIONS,
        lambda bc: (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients()),
        **PUSHER_TOLERANCES,
    ),
    "push_vxb_implicit": ParityCases(
        BOUNDARY_CONDITIONS,
        lambda bc: (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients()),
        **PUSHER_TOLERANCES,
    ),
    "push_weights_with_efield_lin_va": ParityCases(
        BOUNDARY_CONDITIONS, push_weights_with_efield_lin_va_args, **PUSHER_TOLERANCES
    ),
    "reflect": ParityCases((0, 1, 2), reflect_args, n_threads=size_of(2)),
    # atomics add in another order than the serial loop
    "charge_density_0form": ParityCases(((0, 0, 0), (2, 0, 1)), charge_density_0form_args, rtol=1e-12, atol=1e-13),
    # Cuboid, Colella and HollowTorus (non-diagonal DF) and a 3d spline mapping, by index into geometry_domain();
    # atomics add in another order than the serial loop, and the entries reach 1e4 (atol for cancelling sums)
    "linear_vlasov_ampere": ParityCases((0, 2, 5, 12), linear_vlasov_ampere_args, rtol=1e-12, atol=1e-10),
    "eval_spline_mpi_markers": ParityCases(SPLINE_KINDS, eval_spline_mpi_markers_args),
    "eval_spline_mpi_matrix": ParityCases(SPLINE_KINDS, eval_spline_mpi_grid_args(sparse=False), n_threads=size_of(-1)),
    "eval_spline_mpi_sparse_meshgrid": ParityCases(
        SPLINE_KINDS, eval_spline_mpi_grid_args(sparse=True), n_threads=size_of(-1)
    ),
    "kernel_evaluate_pic": ParityCases(KERNEL_EVALUATE_PIC_CASES, kernel_evaluate_pic_args, **GEOMETRY_TOLERANCES),
    # one thread per grid point: the first three axes of mat_f, shape (n1, n2, n3, 3, 3)
    "kernel_evaluate": ParityCases(
        KERNEL_EVALUATE_CASES, kernel_evaluate_args, n_threads=size_of(5, per_thread=9), **GEOMETRY_TOLERANCES
    ),
    # one thread per marker row; the first array, a, has fewer rows when it has no holes
    "kernel_pullpush_pic": ParityCases(
        KERNEL_PULLPUSH_PIC_CASES, kernel_pullpush_pic_args, n_threads=rows_of(1), **GEOMETRY_TOLERANCES
    ),
    # one thread per grid point: the first three axes of out, shape (n1, n2, n3, 3)
    "kernel_pullpush": ParityCases(
        KERNEL_PULLPUSH_CASES, kernel_pullpush_args, n_threads=size_of(-1, per_thread=3), **GEOMETRY_TOLERANCES
    ),
}
