"""Inputs for the pyccel/CUDA parity tests: for every kernel with a CUDA version, the cases it is checked on.

Test-only: ``test_cuda_parity.py`` (on a GPU) and ``test_cuda_emulation.py`` (without one) run each case on both
kernel versions and compare the arrays of the arguments the kernel declares as outputs (``OUTPUTS`` in its folder's
``__init__.py``); the other arrays must stay unchanged. ``build(case)`` returns the kernel's positional arguments on
the active backend; a kernel that is ported to CUDA gets an entry here.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cunumpy as xp
import numpy as np

from struphy.geometry.base import inside_logical_cube
from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.local_projectors_args_cuda import CudaLocalProjectorsArguments
from struphy.kernel_arguments.pusher_args_cuda import CudaDerhamArguments, CudaDomainArguments, CudaMarkerArguments
from struphy.kernel_arguments.spline_args_cuda import CudaSplineArguments
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
    """Launch size, if it is not one thread per row (first axis) of the first array."""


# pyccel argument class name -> the fields of its CUDA struct, the data of an argument object (the pyccel scratch
# arrays bn1, ..., bd3 of DerhamArguments are not fields)
STRUCT_FIELDS = {
    cls.__name__.removeprefix("Cuda"): tuple(field.name for field in cls.struct.fields)
    for cls in (
        CudaMarkerArguments,
        CudaDerhamArguments,
        CudaDomainArguments,
        CudaLocalProjectorsArguments,
        CudaSplineArguments,
    )
}


def argument_arrays(args, indices=None):
    """Host copies of the arrays among `args` (or among the arguments `indices`), by name.

    An array argument is named ``"argument <i>"``; a pyccel argument object contributes the arrays among its struct
    fields, ``"argument <i>.<field>"`` (the names of ``cunumpy.kernel_testing.assert_kernels_agree``).
    """
    found = {}
    for i in range(len(args)) if indices is None else (i % len(args) for i in indices):
        value = args[i]
        if isinstance(value, np.ndarray):
            found[f"argument {i}"] = value.copy()
        for field in STRUCT_FIELDS.get(type(value).__name__, ()):
            item = getattr(value, field, None)
            if isinstance(item, np.ndarray):
                found[f"argument {i}.{field}"] = item.copy()
    return found


# ---------------------------------------------------------------- pushers


def push_eta_stage_args(case):
    from struphy.pic.pushing.kernels.push_eta_stage import push_eta_stage

    bc, method, stage = case
    args = (*marker_arguments(bc), *butcher_arguments(method))
    # reach this stage independently on each backend (one thread per marker, as Pusher launches it)
    for previous in range(stage):
        push_eta_stage(0.2, previous, *args, n_threads=args[0].n_markers)
    return (0.2, stage, *args)


def push_weights_with_efield_lin_va_args(bc):
    f0_values = xp.asarray(np.random.default_rng(13).random(N_MARKERS))
    return (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients(), f0_values, 1.3, 0.8)


def reflect_args(axis):
    """Reverse one logical velocity component of 10 markers in a Cuboid."""
    markers = np.random.default_rng(32).random((8, 140))
    outside_inds = np.arange(139, 129, -1, dtype=np.int64)
    return (xp.asarray(markers), Cuboid(r1=2.0, r2=3.0, r3=4.0).args_domain, xp.asarray(outside_inds), axis)


# guiding-centre evaluation kernels: every mapping, the evaluation point at the new position (alpha = 1), the old one
# (alpha = 0) and in between; the output column alternates. (index into geometry_domain(), alpha, output column)
BSTAR_PARALLEL_ALPHAS = ((1.0,) * 6, (0.0,) * 6, (0.3, 0.6, 0.9, 0.5, 0.0, 0.0))
BSTAR_PARALLEL_3FORM_CASES = tuple(
    (domain, BSTAR_PARALLEL_ALPHAS[domain % 3], 19 if domain % 2 == 0 else 24) for domain in range(N_GEOMETRY_DOMAINS)
)


def bstar_parallel_3form_args(case):
    """Markers with shifts in [0, 1), so eta + shift wraps around, a negative eta_n in marker 0 and a hole in
    marker 2."""
    domain, alpha, output_column = case
    args_markers, _ = marker_arguments((0, 0, 0))
    args_markers.markers[8, 0] = -0.75  # eta_n[0] = mod(-0.75, 1.0) = 0.25 (not 0, where some Jacobians are singular)
    args_markers.markers[0, 2] = -1.0  # a hole: pyccel tests markers[0, ip] == -1, not valid_mks
    return (
        xp.asarray(alpha),
        xp.asarray([output_column], dtype=np.int64),
        args_markers,
        geometry_domain(domain).args_domain,
        derham_arguments(),
        0.37,
        *spline_coefficients(n=2, seed=23),
    )


# ---------------------------------------------------------------- guiding-centre (5D) kernels

# Cuboid, Colella, HollowTorus, ShafranovDshapedCylinder (non-diagonal DF) and a 3d spline mapping, by index into
# geometry_domain()
GC_DOMAINS = (0, 2, 5, 9, 12)
# IGAPolarCylinder: a mapping with a pole at eta1 = 0, for the regular position update (args_markers.pole)
POLE_DOMAIN = 10


def offset_coefficients(n, seed, offset):
    """`n` random coefficient arrays (as :func:`spline_coefficients`) plus `offset`: fields such as B, b_0 and
    B . b keep away from zero, so that B*_parallel, by which the guiding-centre kernels divide, is well conditioned."""
    return tuple(c + offset for c in spline_coefficients(n=n, seed=seed))


# time step of the guiding-centre pushers: the random test fields give logical velocities of order 10, so markers
# move by about 0.05 per step; with larger steps the later Runge-Kutta stages are evaluated far from the marker, where
# round-off differences of the earlier stages are amplified beyond the tolerances
GC_DT = 0.005


def gc_marker_arguments(bc, domain, pole=False):
    """Markers of :func:`marker_arguments` in domain `domain` of geometry_domain(): marker 0 is a hole, marker 1 a
    boundary particle, marker 3 is refilled (first_pusher_idx == -1, skipped by kernels that check it) and
    marker 4 has no shift (eta^{n+1,k} = eta^n, the zero-distance branch of the discrete gradient). With a pole, the
    markers keep eta1 >= 0.1, away from the singular Jacobian at the pole."""
    args_markers, _ = marker_arguments(bc, pole=pole)
    markers = args_markers.markers
    if pole:
        markers[0, 1:] = 0.1 + 0.8 * markers[0, 1:]
        markers[8, 1:] = markers[0, 1:]
    markers[8, 3] = -1.0
    markers[14:17, 4] = 0.0
    # Hamiltonians at n and (n+1, k) of the discrete-gradient pushers
    markers[18:20, 2:] = xp.asarray(np.random.default_rng(15).normal(size=(2, N_MARKERS - 2)))
    return args_markers, geometry_domain(domain).args_domain


# guiding-centre evaluation: the evaluation point at the new position, the old one and in between (alpha), with and
# without the electric potential, two output columns. (index into BSTAR_PARALLEL_ALPHAS, output column, evaluate_e_field)
DRIFTKINETIC_HAMILTONIAN_CASES = tuple((n % 3, 19 if n % 2 == 0 else 24, n < 3) for n in range(6))


def driftkinetic_hamiltonian_args(case):
    alpha, output_column, evaluate_e_field = case
    args_markers, _ = marker_arguments((0, 0, 0))
    args_markers.markers[8, 0] = -0.75  # eta_n[0] = mod(-0.75, 1.0) = 0.25
    args_markers.markers[0, 2] = -1.0  # a hole: pyccel tests markers[0, ip] == -1, not valid_mks
    return (
        xp.asarray(BSTAR_PARALLEL_ALPHAS[alpha]),
        xp.asarray([output_column], dtype=np.int64),
        args_markers,
        geometry_domain(0).args_domain,
        derham_arguments(),
        0.37,
        *spline_coefficients(n=2, seed=24),
        evaluate_e_field,
    )


# (boundary conditions, index into geometry_domain(), evaluate_e_field)
PUSH_GC_BXESTAR_DG_2ND_CASES = tuple(
    (bc, domain, n % 2 == 0) for n, (bc, domain) in enumerate((bc, d) for bc in BOUNDARY_CONDITIONS for d in GC_DOMAINS)
)


def push_gc_bxEstar_discrete_gradient_2nd_order_args(case):
    bc, domain, evaluate_e_field = case
    return (
        GC_DT,
        0,
        *gc_marker_arguments(bc, domain),
        derham_arguments(),
        0.37,
        *spline_coefficients(n=6, seed=25),  # unit_b1, grad_b_full
        *offset_coefficients(1, 26, 5.0),  # B_dot_b
        *spline_coefficients(n=4, seed=27),  # curl_unit_b_dot_b0, e_field
        evaluate_e_field,
    )


# (boundary conditions, Butcher method, stage, index into geometry_domain(), pole, evaluate_e_field)
GC_STAGE_CASES = tuple(
    (bc, method, stage, GC_DOMAINS[n % len(GC_DOMAINS)], False, n % 2 == 0)
    for n, (bc, method, stage) in enumerate(
        (bc, method, stage)
        for bc in BOUNDARY_CONDITIONS
        for method in ("forward_euler", "rk4")
        for stage in range(ButcherTableau(method).n_stages)
    )
) + tuple(((0, 0, 0), "rk4", stage, POLE_DOMAIN, True, stage % 2 == 0) for stage in range(4))


def push_gc_Bstar_explicit_multistage_args(case):
    from struphy.pic.pushing.kernels.push_gc_Bstar_explicit_multistage import push_gc_Bstar_explicit_multistage

    bc, method, stage, domain, pole, evaluate_e_field = case
    args = (
        *gc_marker_arguments(bc, domain, pole=pole),
        derham_arguments(),
        0.37,
        *spline_coefficients(n=3, seed=28),  # grad_b_full
        *offset_coefficients(3, 29, 3.0),  # b2
        *spline_coefficients(n=3, seed=30),  # curl_unit_b2
        *offset_coefficients(1, 31, 5.0),  # B_dot_b
        *spline_coefficients(n=4, seed=32),  # curl_unit_b_dot_b0, e_field
        evaluate_e_field,
        *butcher_arguments(method)[:3],
    )
    # reach this stage independently on each backend (one thread per marker, as Pusher launches it)
    for previous in range(stage):
        push_gc_Bstar_explicit_multistage(GC_DT, previous, *args, n_threads=args[0].n_markers)
    return (GC_DT, stage, *args)


def gc_coupling_fields(seed):
    """B (2-form), b_0 (1-form), both offset so that B*_parallel = b_0 . (B + epsilon v curl b_0) stays away from zero,
    and curl b_0 (2-form): the nine coefficient arrays of the current-coupling kernels."""
    return (
        *offset_coefficients(3, seed, 3.0),
        *offset_coefficients(3, seed + 1, 1.0),
        *spline_coefficients(3, seed + 2),
    )


def push_gc_cc_J1_Hdiv_args(domain):
    return (
        GC_DT,
        0,
        *gc_marker_arguments((0, 0, 0), domain),
        derham_arguments(),
        0.37,
        *gc_coupling_fields(33),
        *spline_coefficients(n=3, seed=36),  # u
    )


def push_gc_cc_J2_stage_Hdiv_args(case):
    from struphy.pic.pushing.kernels.push_gc_cc_J2_stage_Hdiv import push_gc_cc_J2_stage_Hdiv

    bc, method, stage, domain, pole, _ = case
    args = (
        *gc_marker_arguments(bc, domain, pole=pole),
        derham_arguments(),
        0.37,
        *gc_coupling_fields(37),
        *spline_coefficients(n=3, seed=40),  # u
        *butcher_arguments(method)[:3],
    )
    # reach this stage independently on each backend (one thread per marker, as Pusher launches it)
    for previous in range(stage):
        push_gc_cc_J2_stage_Hdiv(GC_DT, previous, *args, n_threads=args[0].n_markers)
    return (GC_DT, stage, *args)


def gc_mag_density_0form_args(scale):
    args_markers, args_domain = marker_arguments((0, 0, 0))
    return (args_markers, derham_arguments(), args_domain, xp.zeros((18, 20, 16)), scale)


# (basis_u, index into geometry_domain(), first marker column of the filling)
CC_LIN_MHD_5D_D_MF_CASES = tuple(
    (basis_u, domain, 19 if n % 2 == 0 else 21)
    for n, (basis_u, domain) in enumerate((basis_u, domain) for basis_u in (0, 1, 2) for domain in GC_DOMAINS)
)


def cc_lin_mhd_5d_D_mf_args(case):
    basis_u, domain, column = case
    return (
        *gc_marker_arguments((0, 0, 0), domain)[:1],
        derham_arguments(),
        geometry_domain(domain).args_domain,
        0.37,
        0.8,
        *gc_coupling_fields(41),
        basis_u,
        column,
    )


# (basis_u, index into geometry_domain(), first marker column of the filling)
CC_LIN_MHD_5D_CURLB_MF_CASES = tuple(
    (basis_u, domain, 19 if n % 2 == 0 else 20)
    for n, (basis_u, domain) in enumerate((basis_u, domain) for basis_u in (0, 2) for domain in GC_DOMAINS)
)


def cc_lin_mhd_5d_curlb_mf_args(case):
    basis_u, domain, column = case
    return (
        *gc_marker_arguments((0, 0, 0), domain)[:1],
        derham_arguments(),
        geometry_domain(domain).args_domain,
        *(xp.zeros((18, 20, 16)) for _ in range(3)),
        0.37,
        0.8,
        *gc_coupling_fields(44),
        basis_u,
        column,
    )


# (basis_u, index into geometry_domain())
CC_LIN_MHD_5D_GRADB_CASES = tuple((basis_u, domain) for basis_u in (0, 2) for domain in GC_DOMAINS)


def cc_lin_mhd_5d_gradB_args(case):
    basis_u, domain = case
    return (
        *gc_marker_arguments((0, 0, 0), domain)[:1],
        derham_arguments(),
        geometry_domain(domain).args_domain,
        *(xp.zeros((18, 20, 16)) for _ in range(3)),
        0.37,
        0.8,
        *gc_coupling_fields(47),
        *spline_coefficients(n=6, seed=50),  # grad_PB, grad_PBeq
        basis_u,
    )


# (basis_u, first marker column of the filling)
MARKER_OPERATOR_CASES = ((0, 19), (1, 20), (2, 19), (1, 19))


def marker_operator_dot_args(case):
    """The filling data of :class:`~struphy.pic.accumulation.particles_to_grid.MarkerOperator` in the (random) marker
    columns from `column`, applied to random coefficients s of the space `basis_u`."""
    basis_u, column = case
    args_markers, args_domain = marker_arguments((0, 0, 0))
    return (
        args_markers,
        derham_arguments(),
        args_domain,
        *(xp.zeros((18, 20, 16)) for _ in range(3)),
        *spline_coefficients(n=3, seed=53),
        basis_u,
        column,
    )


def eval_magnetic_energy_PBb_args(domain):
    """129 markers in domain `domain` of geometry_domain(): marker 0 is a hole, markers 1 and 2 have positions outside
    [0, 1) (evaluated at mod(eta, 1)); the result goes to the diagnostics column."""
    args_markers, _ = marker_arguments((0, 0, 0))
    markers = args_markers.markers
    markers[0, 1] = 1.3
    markers[1, 2] = -0.4
    return (
        markers,
        derham_arguments(),
        geometry_domain(domain).args_domain,
        args_markers.first_diagnostics_idx,
        args_markers.mu_idx,
        *offset_coefficients(1, 56, 2.0),  # abs_B0
        *spline_coefficients(n=1, seed=57),  # PBb
    )


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


def vlasov_maxwell_args(domain):
    """129 markers (a hole, marker 0, and a boundary particle, marker 1, which is accumulated as in pyccel) in domain
    `domain` of geometry_domain(); zeroed V1 data."""
    args_markers, _ = marker_arguments((0, 0, 0))
    args_markers.markers[0, 0] = -1.0
    return (args_markers, derham_arguments(), geometry_domain(domain).args_domain, *v1_symm_accumulation_data())


# ---------------------------------------------------------------- spline evaluation


def eval_spline_mpi_markers_args(kind):
    """Spline values at 129 markers, one of them flagged."""
    markers = np.random.default_rng(5).random((3, 129))
    markers[0, 0] = -1.0  # not on the process domain: skipped
    return (xp.asarray(markers), *spline_evaluation_arguments(kind), xp.zeros(129))


def eval_spline_mpi_grid_args(sparse):
    """Spline values on a 7 x 5 x 4 grid (full or sparse meshgrid), one point flagged."""
    return lambda kind: (*evaluation_grid(sparse=sparse), *spline_evaluation_arguments(kind), xp.zeros((7, 5, 4)))


def eval_spline_mpi_tensor_product_fixed_args(kind):
    """Spline values on a 5 x 4 x 3 grid from pre-evaluated spans and (random) basis values."""
    _data, args_spline = spline_evaluation_arguments(kind)
    kind, pn, starts = args_spline.kind, args_spline.pn, args_spline.starts
    rng = np.random.default_rng(7)
    degree, kinds = (2, 3, 1), xp.to_numpy(kind)
    # spans of the 8 cells: p <= span <= p + 7
    spans = [xp.asarray(rng.integers(p, p + 8, size=n)) for p, n in zip(degree, (5, 4, 3))]
    bases = [xp.asarray(rng.random((n, p - k + 1))) for p, k, n in zip(degree, kinds, (5, 4, 3))]
    return (*spans, *bases, _data, kind, pn, starts, xp.zeros((5, 4, 3)))


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
    mat_f = np.random.default_rng(6).random((markers.shape[1], 3, 3))
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
    """Every third case passes `a` without holes (one row per inside marker); `a` and `out` have one row per
    marker."""
    domain, kind_transform, kind_fun, a_has_holes, remove_outside = case
    markers = logical_markers()
    rng = np.random.default_rng(9)
    n_comp = 1 if kind_fun < 10 else 3
    a = rng.normal(size=(markers.shape[1], n_comp))
    if not a_has_holes:
        a = a[inside_logical_cube(markers)]
    out = rng.random((markers.shape[1], 3))
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


def kernel_3d_mat_threads(args):
    """One thread per (element, codomain basis function, domain basis function), as the kernel's launch size."""
    from struphy.feec.kernels.kernel_3d_mat import _n_threads

    return _n_threads(args)


def kernel_3d_mat_args(case):
    """Mass-matrix assembly on 3 x 2 x 2 elements with mixed degrees and random quadrature data; `case` is the start
    index of the rows in direction 1 (1: the first row is a ghost row, local index -1)."""
    rng = np.random.default_rng(20 + case)
    ne, nq = (3, 2, 2), (3, 2, 2)
    pi, pj, pads = (2, 1, 1), (1, 2, 1), (2, 2, 2)
    starts = (case, 0, 0)
    spans = [xp.asarray(np.arange(n, dtype=np.int64) + p) for n, p in zip(ne, pi)]
    w = [xp.asarray(rng.random((n, q))) for n, q in zip(ne, nq)]
    bi = [xp.asarray(rng.random((n, p + 1, 1, q))) for n, p, q in zip(ne, pi, nq)]
    bj = [xp.asarray(rng.random((n, p + 1, 1, q))) for n, p, q in zip(ne, pj, nq)]
    mat_fun = xp.asarray(rng.normal(size=tuple(n * q for n, q in zip(ne, nq))))
    rows = tuple(n + p + 2 * pad for n, p, pad in zip(ne, pi, pads))
    data = xp.asarray(rng.normal(size=rows + tuple(2 * pad + 1 for pad in pads)))  # added to, not overwritten
    return (*spans, *pi, *pj, *starts, *pads, *w, *bi, *bj, mat_fun, data)


# ---------------------------------------------------------------- all kernels with a CUDA version

PUSHER_TOLERANCES = {"rtol": 1e-13, "atol": 1e-14}
GEOMETRY_TOLERANCES = {"rtol": 1e-10, "atol": 1e-10}
SPLINE_KINDS = ((0, 0, 0), (1, 0, 1))


def markers_of(index):
    """One thread per marker of argument `index`, a marker argument object."""
    return lambda args: args[index].n_markers


def columns_of(index):
    """One thread per column of argument `index`, an array of markers (one column per marker)."""
    return lambda args: args[index].shape[1]


def size_of(index, per_thread=1):
    """One thread per `per_thread` entries of argument `index`."""
    return lambda args: args[index].size // per_thread


PARITY_CASES = {
    # atomic additions: the summation order over elements differs from pyccel
    "kernel_3d_mat": ParityCases((0, 1), kernel_3d_mat_args, rtol=1e-12, atol=1e-13, n_threads=kernel_3d_mat_threads),
    "push_eta_stage": ParityCases(
        [
            (bc, method, stage)
            for bc in BOUNDARY_CONDITIONS
            for method in ("forward_euler", "rk4")
            for stage in range(ButcherTableau(method).n_stages)
        ],
        push_eta_stage_args,
        n_threads=markers_of(2),
        **PUSHER_TOLERANCES,
    ),
    "push_v_with_efield": ParityCases(
        BOUNDARY_CONDITIONS,
        lambda bc: (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients(), 0.7),
        n_threads=markers_of(2),
        **PUSHER_TOLERANCES,
    ),
    "push_vxb_analytic": ParityCases(
        BOUNDARY_CONDITIONS,
        lambda bc: (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients()),
        n_threads=markers_of(2),
        **PUSHER_TOLERANCES,
    ),
    "push_vxb_implicit": ParityCases(
        BOUNDARY_CONDITIONS,
        lambda bc: (0.2, 0, *marker_arguments(bc), derham_arguments(), *spline_coefficients()),
        n_threads=markers_of(2),
        **PUSHER_TOLERANCES,
    ),
    "push_weights_with_efield_lin_va": ParityCases(
        BOUNDARY_CONDITIONS, push_weights_with_efield_lin_va_args, n_threads=markers_of(2), **PUSHER_TOLERANCES
    ),
    "reflect": ParityCases((0, 1, 2), reflect_args, n_threads=size_of(2)),
    # one thread per marker
    "bstar_parallel_3form": ParityCases(
        BSTAR_PARALLEL_3FORM_CASES,
        bstar_parallel_3form_args,
        n_threads=markers_of(2),
        **GEOMETRY_TOLERANCES,
    ),
    # guiding-centre (5D) pushers and evaluation, one thread per marker
    "driftkinetic_hamiltonian": ParityCases(
        DRIFTKINETIC_HAMILTONIAN_CASES, driftkinetic_hamiltonian_args, n_threads=markers_of(2), **PUSHER_TOLERANCES
    ),
    "push_gc_bxEstar_discrete_gradient_2nd_order": ParityCases(
        PUSH_GC_BXESTAR_DG_2ND_CASES,
        push_gc_bxEstar_discrete_gradient_2nd_order_args,
        n_threads=markers_of(2),
        **GEOMETRY_TOLERANCES,
    ),
    # B*_parallel nearly vanishes for a marker of the spline mapping, whose v_parallel grows to 1e4 in the last RK4
    # stage; reflecting its (vdim = 3) velocity then cancels terms of 1e4 (atol for cancelling sums)
    "push_gc_Bstar_explicit_multistage": ParityCases(
        GC_STAGE_CASES, push_gc_Bstar_explicit_multistage_args, n_threads=markers_of(2), rtol=1e-10, atol=1e-8
    ),
    "push_gc_cc_J1_Hdiv": ParityCases(
        GC_DOMAINS, push_gc_cc_J1_Hdiv_args, n_threads=markers_of(2), **GEOMETRY_TOLERANCES
    ),
    "push_gc_cc_J2_stage_Hdiv": ParityCases(
        GC_STAGE_CASES, push_gc_cc_J2_stage_Hdiv_args, n_threads=markers_of(2), **GEOMETRY_TOLERANCES
    ),
    # guiding-centre (5D) accumulation: atomics add in another order than the serial loop
    "gc_mag_density_0form": ParityCases(
        (1.0, 0.37), gc_mag_density_0form_args, n_threads=markers_of(0), rtol=1e-12, atol=1e-13
    ),
    # writes marker columns only (no atomics)
    "cc_lin_mhd_5d_D_mf": ParityCases(
        CC_LIN_MHD_5D_D_MF_CASES, cc_lin_mhd_5d_D_mf_args, n_threads=markers_of(0), **GEOMETRY_TOLERANCES
    ),
    "cc_lin_mhd_5d_curlb_mf": ParityCases(
        CC_LIN_MHD_5D_CURLB_MF_CASES, cc_lin_mhd_5d_curlb_mf_args, n_threads=markers_of(0), rtol=1e-10, atol=1e-10
    ),
    "cc_lin_mhd_5d_gradB": ParityCases(
        CC_LIN_MHD_5D_GRADB_CASES, cc_lin_mhd_5d_gradB_args, n_threads=markers_of(0), rtol=1e-10, atol=1e-10
    ),
    "marker_operator_asym_dot": ParityCases(
        MARKER_OPERATOR_CASES, marker_operator_dot_args, n_threads=markers_of(0), rtol=1e-12, atol=1e-12
    ),
    "marker_operator_rank1_dot": ParityCases(
        MARKER_OPERATOR_CASES, marker_operator_dot_args, n_threads=markers_of(0), rtol=1e-12, atol=1e-12
    ),
    # one thread per marker (column of the markers)
    "eval_magnetic_energy_PBb": ParityCases(
        (0, 5, 12), eval_magnetic_energy_PBb_args, n_threads=columns_of(0), **PUSHER_TOLERANCES
    ),
    # atomics add in another order than the serial loop
    "charge_density_0form": ParityCases(
        ((0, 0, 0), (2, 0, 1)), charge_density_0form_args, n_threads=markers_of(0), rtol=1e-12, atol=1e-13
    ),
    # Cuboid, Colella and HollowTorus (non-diagonal DF) and a 3d spline mapping, by index into geometry_domain();
    # atomics add in another order than the serial loop, and the entries reach 1e4 (atol for cancelling sums)
    "linear_vlasov_ampere": ParityCases(
        (0, 2, 5, 12), linear_vlasov_ampere_args, n_threads=markers_of(0), rtol=1e-12, atol=1e-10
    ),
    # Cuboid, then Colella, HollowTorus, ShafranovDshapedCylinder (non-diagonal DF) and a 3d spline mapping, by index into
    # geometry_domain(); atomics add in another order than the serial loop, and G^{-1} of the spline mapping makes
    # entries of 6e5 (atol for cancelling sums)
    "vlasov_maxwell": ParityCases(
        (0, 2, 5, 9, 12), vlasov_maxwell_args, n_threads=markers_of(0), rtol=1e-12, atol=1e-8
    ),
    "eval_spline_mpi_markers": ParityCases(SPLINE_KINDS, eval_spline_mpi_markers_args, n_threads=columns_of(0)),
    "eval_spline_mpi_matrix": ParityCases(SPLINE_KINDS, eval_spline_mpi_grid_args(sparse=False), n_threads=size_of(-1)),
    "eval_spline_mpi_sparse_meshgrid": ParityCases(
        SPLINE_KINDS, eval_spline_mpi_grid_args(sparse=True), n_threads=size_of(-1)
    ),
    "eval_spline_mpi_tensor_product_fixed": ParityCases(
        SPLINE_KINDS, eval_spline_mpi_tensor_product_fixed_args, n_threads=size_of(-1)
    ),
    "kernel_evaluate_pic": ParityCases(
        KERNEL_EVALUATE_PIC_CASES, kernel_evaluate_pic_args, n_threads=columns_of(0), **GEOMETRY_TOLERANCES
    ),
    # one thread per grid point: the first three axes of mat_f, shape (n1, n2, n3, 3, 3)
    "kernel_evaluate": ParityCases(
        KERNEL_EVALUATE_CASES, kernel_evaluate_args, n_threads=size_of(5, per_thread=9), **GEOMETRY_TOLERANCES
    ),
    # one thread per marker (column of the markers); the first array, a, has fewer rows when it has no holes
    "kernel_pullpush_pic": ParityCases(
        KERNEL_PULLPUSH_PIC_CASES, kernel_pullpush_pic_args, n_threads=columns_of(1), **GEOMETRY_TOLERANCES
    ),
    # one thread per grid point: the first three axes of out, shape (n1, n2, n3, 3)
    "kernel_pullpush": ParityCases(
        KERNEL_PULLPUSH_CASES, kernel_pullpush_args, n_threads=size_of(-1, per_thread=3), **GEOMETRY_TOLERANCES
    ),
}
