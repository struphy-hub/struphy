"""Marker destinations and execution ordering for configured evaluations."""

from types import SimpleNamespace

import numpy as np
import pytest
from cunumpy import PyccelKernel

from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, MarkerArguments
from struphy.pic import sph_smoothing_kernels
from struphy.pic.pushing import eval_kernels_gc, eval_kernels_sph
from struphy.pic.pushing.kernel_setup import KernelSetup
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator


@pytest.fixture
def marker_args():
    markers = np.zeros((2, 40))
    markers[0, :7] = (0.4, 0.3, 0.2, 2.0, 4.0, 6.0, 2.0)
    markers[0, 8:14] = markers[0, :6]
    markers[0, 11] = 1.0  # Initial parallel velocity.
    markers[1, 0] = -1.0  # A hole must never be written to.
    return MarkerArguments(markers, np.array([True, False]), 1, 3, 6, 7, 8, 14, 17, 18, 4, np.zeros(3, dtype=int))


@pytest.fixture
def derham_args():
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    return DerhamArguments(np.ones(3, dtype=int), knots, knots, knots, np.zeros(3, dtype=int))


def coefficients(value):
    # Constant spline coefficients including padding.
    return np.full((4, 4, 4), float(value))


@pytest.mark.parametrize(
    "kernel_name,values,expected",
    [
        ("driftkinetic_hamiltonian", (2.0, 5.0), (23.25,)),
        ("bstar_parallel_3form", (2.0, 5.0), (17.0,)),
        ("grad_driftkinetic_hamiltonian", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (4.0, 11.0, 18.0)),
        ("bstar_2form", (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (13.0, 17.0, 21.0)),
        ("unit_b_1form", (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    ],
)
def test_guiding_center_destinations(marker_args, derham_args, kernel_name, values, expected):
    fields = tuple(coefficients(value) for value in values)
    args = (derham_args, *fields) if kernel_name == "unit_b_1form" else (derham_args, 2.0, *fields)
    if kernel_name in ("driftkinetic_hamiltonian", "grad_driftkinetic_hamiltonian"):
        args += (True,)
    indices = (30,) if len(expected) == 1 else (30, None, 24)
    setup = KernelSetup(
        kernel=getattr(eval_kernels_gc, kernel_name),
        output_indices=indices,
        args=args,
        alpha=(1.0, 0.0, 0.5, 0.5),
    )
    before = marker_args.markers.copy()
    setup.evaluate(marker_args, Cuboid().args_domain)
    reference = before.copy()
    for component, index in enumerate(indices):
        if index is not None:
            reference[0, index] = expected[component]
    np.testing.assert_allclose(marker_args.markers, reference)


@pytest.mark.parametrize("kernel", [eval_kernels_sph.sph_pressure_coeffs, eval_kernels_sph.sph_mean_velocity_coeffs])
def test_sph_vector_destinations(marker_args, kernel):
    boxes = np.array([[0, -1], [-1, -1]])
    neighbours = np.ones((1, 27), dtype=int)
    neighbours[0, 0] = 0
    setup = KernelSetup(
        kernel=kernel,
        output_indices=(30, None, 24),
        args=(boxes, neighbours, ~marker_args.valid_mks, False, False, False, 120, 0.5, 0.5, 0.5),
    )
    before = marker_args.markers.copy()
    setup.evaluate(marker_args, Cuboid().args_domain)
    reference = before.copy()
    # One particle: density = weight * W(0) = 2 * (1 / h) = 4.
    if kernel is eval_kernels_sph.sph_pressure_coeffs:
        reference[0, 30], reference[0, 24] = 4.0, 2.0 * 4.0 ** (-1.0 / 3.0)
    else:
        reference[0, 30], reference[0, 24] = 1.0, 3.0
    np.testing.assert_allclose(marker_args.markers, reference)


def test_sph_tensor_destinations(marker_args):
    # Second particle at eta1 + 0.2 (inside h = 0.5) with the same velocity coefficients.
    marker_args.markers[1] = marker_args.markers[0]
    marker_args.markers[1, 0] = 0.6
    marker_args.markers[:, 18:21] = (1.0, 2.0, 3.0)
    marker_args.valid_mks[1] = True
    boxes = np.array([[0, 1, -1], [-1, -1, -1]])
    neighbours = np.ones((1, 27), dtype=int)
    neighbours[0, 0] = 0
    # Self-gradient of the linear 1D kernel vanishes; the neighbour contributes -+1 / h**2 = -+4.
    gradient = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]])
    density = 2.0 * (2.0 + 1.2)  # weight * (W(0) + W(0.2))
    indices = (30, 29, 28, 27, None, 25, 24, 23, 22)
    setup = KernelSetup(
        kernel=eval_kernels_sph.sph_viscosity_tensor,
        output_indices=indices,
        args=(boxes, neighbours, ~marker_args.valid_mks, False, False, False, 120, 0.5, 0.5, 0.5, 3.0),
    )
    before = marker_args.markers.copy()
    setup.evaluate(marker_args, Cuboid().args_domain)
    symmetric = (gradient + gradient.T) / 2.0
    tensor = -2.0 * 3.0 * (2.0 / density) * (symmetric - np.eye(3) * np.trace(symmetric) / 3.0)
    reference = before.copy()
    for value, index in zip(tensor.flat, indices):
        if index is not None:
            reference[0, index] = value
            reference[1, index] = -value
    np.testing.assert_allclose(marker_args.markers, reference)


# Each gradient component must vanish where its own coordinate is zero (no self-force, issue #437).
ZERO_GRADIENT_CASES = [
    (n + k, point)
    for n in (100, 110, 120, 340, 350, 360, 670, 680, 690, 700)
    for k in (1, 2, 3)
    for point in ((0.0, 0.0, 0.0), (0.0, 0.1, 0.05), (0.1, 0.0, 0.05), (0.1, 0.05, 0.0))
    if point[k - 1] == 0.0 and (n != 690 or not any(point))
]


@pytest.mark.parametrize("kernel_type,point", ZERO_GRADIENT_CASES)
def test_sph_kernel_gradients_vanish_at_zero(kernel_type, point):
    assert sph_smoothing_kernels.smoothing_kernel(kernel_type, *point, 0.3, 0.25, 0.2) == 0.0


def test_sph_tensor_mapped_domain(marker_args):
    # On a mapped domain the strain uses Cartesian derivatives (grad_eta @ DF^{-1}) and the
    # stored columns are in Piola form det(DF) * tensor @ DF^{-T}.
    # Only the neighbour at eta1 + 0.2 carries velocity coefficients, so the result does not
    # depend on the kernel gradient at zero; its linear 1D kernel gradient is 1 / h**2 = 4.
    marker_args.markers[1] = marker_args.markers[0]
    marker_args.markers[1, 0] = 0.6
    marker_args.markers[1, 18:21] = (1.0, 2.0, 3.0)
    marker_args.valid_mks[1] = True
    boxes = np.array([[0, 1, -1], [-1, -1, -1]])
    neighbours = np.ones((1, 27), dtype=int)
    neighbours[0, 0] = 0
    domain = Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=0.5)
    jac = np.diag([2.0, 3.0, 0.5])
    jac_inv = np.linalg.inv(jac)
    gradient = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]]) @ jac_inv
    density = 2.0 * (2.0 + 1.2)  # weight * (W(0) + W(0.2))
    indices = tuple(range(22, 31))
    setup = KernelSetup(
        kernel=eval_kernels_sph.sph_viscosity_tensor,
        output_indices=indices,
        args=(boxes, neighbours, ~marker_args.valid_mks, False, False, False, 120, 0.5, 0.5, 0.5, 3.0),
    )
    setup.evaluate(marker_args, domain.args_domain)
    symmetric = (gradient + gradient.T) / 2.0
    tensor = -2.0 * 3.0 * (2.0 / density) * (symmetric - np.eye(3) * np.trace(symmetric) / 3.0)
    piola = np.linalg.det(jac) * tensor @ jac_inv.T
    np.testing.assert_allclose(marker_args.markers[0, 22:31], piola.flat)


@pytest.mark.parametrize("indices", [(), (1, 2), (-1,), (True,), (1.5,), (None,), (1, 1, 2)])
def test_invalid_destinations(indices):
    with pytest.raises(ValueError):
        KernelSetup(kernel=lambda: None, output_indices=indices)


def test_registration_and_validation():
    # Use the concrete registration methods without constructing a physical model.
    class Registry:
        init_kernels = Propagator.init_kernels
        eval_kernels = Propagator.eval_kernels
        add_init_kernel = Propagator.add_init_kernel
        add_eval_kernel = Propagator.add_eval_kernel

    registry = Registry()
    initial = KernelSetup(kernel=lambda: None, output_indices=(5,))
    evaluated = KernelSetup(kernel=lambda: None, output_indices=(6, None, 8), alpha=(1.0, 0.5, 0.0))
    registry.add_init_kernel(initial)
    registry.add_eval_kernel(evaluated)
    assert registry.init_kernels == (initial,)
    assert registry.eval_kernels == (evaluated,)
    assert Registry().init_kernels == Registry().eval_kernels == ()
    np.testing.assert_equal(evaluated.sorting_alpha, (1.0, 0.5, 0.0))
    assert evaluated.alpha == (1.0, 0.5, 0.0, 0.0, 0.0, 0.0)
    evaluated.validate_outputs(9)
    with pytest.raises(ValueError, match="output column 8"):
        evaluated.validate_outputs(8)
    with pytest.raises(ValueError, match="initial state"):
        registry.add_init_kernel(evaluated)
    with pytest.raises(TypeError, match="KernelSetup"):
        registry.add_eval_kernel((lambda: None, (), (5,)))


def test_pusher_evaluation_order_and_live_arguments(marker_args):
    events = []
    field = np.array([2.0])

    def initialize(alpha, indices, markers, domain, values):
        events.append("init")
        markers.markers[0, indices[0]] = values[0]

    def evaluate(alpha, indices, markers, domain):
        events.append("eval")
        markers.markers[0, indices[0]] = markers.markers[0, 20] + alpha[0]

    def push(dt, stage, markers, domain):
        events.append("push")
        assert markers.markers[0, 21] == field[0] + 0.5
        markers.markers[0, markers.residual_idx] = 1.0  # Force both iterations.

    def sort(**kwargs):
        # Final sort after the last stage uses current positions (no alpha).
        alpha = kwargs.get("alpha")
        events.append("sort" if alpha is None else f"sort{alpha[0]}")
        if alpha is not None:
            assert alpha in ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))

    particles = SimpleNamespace(
        markers=marker_args.markers,
        args_markers=marker_args,
        n_cols=40,
        vdim=3,
        first_pusher_idx=8,
        first_shift_idx=14,
        residual_idx=17,
        n_mks_loc=1,
        Np=1,
        mpi_rank=0,
        mpi_comm=SimpleNamespace(Allreduce=lambda *args, **kwargs: None),
        sorting_boxes=SimpleNamespace(communicate=True),
        put_particles_in_boxes=lambda: events.append("boxes"),
        mpi_sort_markers=sort,
        finish_kernel_bc=lambda **kwargs: events.append("bc"),
    )
    pusher = Pusher(
        particles,
        PyccelKernel(push),
        (),
        None,
        pushes_eta=True,
        alpha_in_kernel=1.0,
        init_kernels=(KernelSetup(kernel=initialize, output_indices=(20,), args=(field,)),),
        eval_kernels=(KernelSetup(kernel=evaluate, output_indices=(21,), alpha=0.5),),
        n_stages=2,
        maxiter=2,
    )
    for value in (2.0, 4.0):
        field[0] = value
        events.clear()
        pusher(0.1)
        # Markers are sorted on entry, so no sort is needed until the first push moves them.
        first = ["eval", "boxes", "push", "bc", "boxes"]
        later = ["sort0.5", "eval", "boxes", "sort1.0", "push", "bc", "boxes"]
        assert events == ["init", "boxes"] + first + later * 3 + ["sort"]
