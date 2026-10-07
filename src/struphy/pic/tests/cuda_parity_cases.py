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

from struphy.geometry.domains import Cuboid
from struphy.ode.utils import ButcherTableau
from struphy.pic.tests.kernel_test_args import (
    BOUNDARY_CONDITIONS,
    N_MARKERS,
    butcher_arguments,
    derham_arguments,
    evaluation_grid,
    marker_arguments,
    spline_coefficients,
    spline_evaluation_arguments,
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


# ---------------------------------------------------------------- spline evaluation


def eval_spline_mpi_markers_args(kind):
    """Spline values at 129 markers, one of them flagged."""
    markers = np.random.default_rng(5).random((129, 3))
    markers[0, 0] = -1.0  # not on the process domain: skipped
    return (xp.asarray(markers), *spline_evaluation_arguments(kind), xp.zeros(129))


def eval_spline_mpi_grid_args(sparse):
    """Spline values on a 7 x 5 x 4 grid (full or sparse meshgrid), one point flagged."""
    return lambda kind: (*evaluation_grid(sparse=sparse), *spline_evaluation_arguments(kind), xp.zeros((7, 5, 4)))


# ---------------------------------------------------------------- all kernels with a CUDA version

PUSHER_TOLERANCES = {"rtol": 1e-13, "atol": 1e-14}
SPLINE_KINDS = ((0, 0, 0), (1, 0, 1))


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
    "eval_spline_mpi_markers": ParityCases(SPLINE_KINDS, eval_spline_mpi_markers_args),
    "eval_spline_mpi_matrix": ParityCases(SPLINE_KINDS, eval_spline_mpi_grid_args(sparse=False), n_threads=size_of(-1)),
    "eval_spline_mpi_sparse_meshgrid": ParityCases(
        SPLINE_KINDS, eval_spline_mpi_grid_args(sparse=True), n_threads=size_of(-1)
    ),
}
