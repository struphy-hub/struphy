"""Marker destinations and execution ordering for configured evaluations."""

from types import SimpleNamespace

import numpy as np
import pytest
from cunumpy.kernels import PyccelKernel

from struphy.geometry.domains import Cuboid
from struphy.kernel_arguments.pusher_args_kernels import DerhamArguments, MarkerArguments
from struphy.pic import sph_radial_kernels
from struphy.pic.pushing.kernel_setup import KernelSetup
from struphy.pic.pushing.kernels.bstar_2form import bstar_2form
from struphy.pic.pushing.kernels.bstar_parallel_3form import bstar_parallel_3form
from struphy.pic.pushing.kernels.driftkinetic_hamiltonian import driftkinetic_hamiltonian
from struphy.pic.pushing.kernels.grad_driftkinetic_hamiltonian import grad_driftkinetic_hamiltonian
from struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_1st_order import (
    push_gc_Bstar_discrete_gradient_1st_order,
)
from struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_1st_order_newton import (
    push_gc_Bstar_discrete_gradient_1st_order_newton,
)
from struphy.pic.pushing.kernels.push_gc_Bstar_discrete_gradient_2nd_order import (
    push_gc_Bstar_discrete_gradient_2nd_order,
)
from struphy.pic.pushing.kernels.sph_mean_velocity_coeffs import sph_mean_velocity_coeffs
from struphy.pic.pushing.kernels.sph_pressure_coeffs import sph_pressure_coeffs
from struphy.pic.pushing.kernels.sph_viscosity_tensor import sph_viscosity_tensor
from struphy.pic.pushing.kernels.unit_b_1form import unit_b_1form
from struphy.pic.pushing.pusher import Pusher
from struphy.propagators.base import Propagator


@pytest.fixture
def marker_args():
    markers = np.zeros((40, 2))
    markers[:7, 0] = (0.4, 0.3, 0.2, 2.0, 4.0, 6.0, 2.0)
    markers[8:14, 0] = markers[:6, 0]
    markers[11, 0] = 1.0  # Initial parallel velocity.
    markers[0, 1] = -1.0  # A hole must never be written to.
    return MarkerArguments(
        markers, np.array([True, False]), 1, 3, 6, 7, 8, 14, 17, 18, 4, np.zeros(3, dtype=int), False
    )


@pytest.fixture
def derham_args():
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    return DerhamArguments(np.ones(3, dtype=int), knots, knots, knots, np.zeros(3, dtype=int))


def coefficients(value):
    # Constant spline coefficients including padding.
    return np.full((4, 4, 4), float(value))


@pytest.mark.parametrize(
    "kernel,values,expected",
    [
        (driftkinetic_hamiltonian, (2.0, 5.0), (23.25,)),
        (bstar_parallel_3form, (2.0, 5.0), (17.0,)),
        (grad_driftkinetic_hamiltonian, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (4.0, 11.0, 18.0)),
        (bstar_2form, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (13.0, 17.0, 21.0)),
        (unit_b_1form, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
    ],
)
def test_guiding_center_destinations(marker_args, derham_args, kernel, values, expected):
    fields = tuple(coefficients(value) for value in values)
    args = (derham_args, *fields) if kernel.name == "unit_b_1form" else (derham_args, 2.0, *fields)
    if kernel.name in ("driftkinetic_hamiltonian", "grad_driftkinetic_hamiltonian"):
        args += (True,)
    indices = (30,) if len(expected) == 1 else (30, None, 24)
    setup = KernelSetup(
        kernel=kernel,
        output_indices=indices,
        args=args,
        alpha=(1.0, 0.0, 0.5, 0.5),
    )
    before = marker_args.markers.copy()
    setup.evaluate(marker_args, Cuboid().args_domain)
    reference = before.copy()
    for component, index in enumerate(indices):
        if index is not None:
            reference[index, 0] = expected[component]
    np.testing.assert_allclose(marker_args.markers, reference)


# SPH kernel arguments for a 1D Wendland C2 kernel along x (kernel id 2) of width H on a non-periodic domain,
# with the physical positions in the marker columns X_IDX:X_IDX + 3 (see Particles.sph_kernel_args).
X_IDX = 34
H = 0.5
PROJ_X = np.diag([1.0, 0.0, 0.0])


def wendland_1d(r, h=H):
    """Wendland C2 kernel in 1D (sigma = 5/4) and its radial derivative, written out independently of the kernels."""
    q = r / h
    if q >= 1.0:
        return 0.0, 0.0
    return 1.25 / h * (1 - q) ** 3 * (1 + 3 * q), 1.25 / h**2 * (-12.0 * q * (1 - q) ** 2)


def sph_args(boxes, neighbours, valid_mks):
    return (
        boxes,
        neighbours,
        ~valid_mks,
        False,
        False,
        False,
        np.zeros((3, 3)),
        PROJ_X,
        X_IDX,
        2,
        1,
        H,
        1.0,
    )


def set_positions(markers, domain=None):
    """Physical positions x = F(eta) in the columns X_IDX:X_IDX + 3 (identity for domain=None)."""
    for p in range(markers.shape[1]):
        eta = markers[:3, p]
        markers[X_IDX : X_IDX + 3, p] = eta if domain is None else np.ravel(domain(eta[:, None]))


@pytest.mark.parametrize("kernel", [sph_pressure_coeffs, sph_mean_velocity_coeffs])
def test_sph_vector_destinations(marker_args, kernel):
    boxes = np.array([[0, -1], [-1, -1]])
    neighbours = np.ones((1, 27), dtype=int)
    neighbours[0, 0] = 0
    set_positions(marker_args.markers)
    setup = KernelSetup(
        kernel=kernel,
        output_indices=(30, None, 24),
        args=sph_args(boxes, neighbours, marker_args.valid_mks)
        + ((5.0 / 3.0,) if kernel.name == "sph_pressure_coeffs" else ()),
    )
    before = marker_args.markers.copy()
    setup.evaluate(marker_args, Cuboid().args_domain)
    reference = before.copy()
    # One particle: density = weight * W(0) = 2 * 5 / (4 h) = 5.
    rho = 2.0 * wendland_1d(0.0)[0]
    assert rho == pytest.approx(5.0)
    if kernel.name == "sph_pressure_coeffs":
        reference[30, 0], reference[24, 0] = rho, 2.0 * rho ** (-1.0 / 3.0)
    else:
        # w v / rho with v = (2, 4, 6)
        reference[30, 0], reference[24, 0] = 2.0 * 2.0 / rho, 2.0 * 6.0 / rho
    np.testing.assert_allclose(marker_args.markers, reference)


def _tensor_reference(gradient, density, mu=3.0, weight=2.0):
    """Stored stress coefficients w sigma / rho, sigma = -mu (grad v + grad v^T - 2/3 div v I)."""
    div = np.trace(gradient)
    return -mu * (weight / density) * (gradient + gradient.T - 2.0 / 3.0 * div * np.eye(3))


@pytest.mark.parametrize("domain", [None, Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=0.5)])
def test_sph_tensor_destinations(marker_args, domain):
    # Second particle at eta1 + 0.2 with the same velocity coefficients (1, 2, 3). The kernel only uses the
    # physical positions: on the stretched Cuboid the distance is 0.4 instead of 0.2 (Cartesian derivatives,
    # no Piola transform).
    marker_args.markers[:, 1] = marker_args.markers[:, 0]
    marker_args.markers[0, 1] = 0.6
    marker_args.markers[18:21] = np.array([1.0, 2.0, 3.0])[:, None]
    marker_args.valid_mks[1] = True
    set_positions(marker_args.markers, domain)
    boxes = np.array([[0, 1, -1], [-1, -1, -1]])
    neighbours = np.ones((1, 27), dtype=int)
    neighbours[0, 0] = 0
    r = 0.2 if domain is None else 0.4
    w_r, dw_r = wendland_1d(r)
    # Self-gradient vanishes; the neighbour at +r contributes c_j * dW/dr * (x_0 - x_1) / r = -c_j * dW/dr.
    gradient = np.zeros((3, 3))
    gradient[:, 0] = -np.array([1.0, 2.0, 3.0]) * dw_r
    density = 2.0 * (wendland_1d(0.0)[0] + w_r)
    indices = (30, 29, 28, 27, None, 25, 24, 23, 22)
    setup = KernelSetup(
        kernel=sph_viscosity_tensor,
        output_indices=indices,
        args=sph_args(boxes, neighbours, marker_args.valid_mks) + (3.0,),
    )
    before = marker_args.markers.copy()
    setup.evaluate(marker_args, Cuboid().args_domain if domain is None else domain.args_domain)
    tensor = _tensor_reference(gradient, density)
    reference = before.copy()
    for value, index in zip(tensor.flat, indices):
        if index is not None:
            reference[index, 0] = value
            # the second particle sees the first one at -r: the gradient changes sign
            reference[index, 1] = -value
    np.testing.assert_allclose(marker_args.markers, reference)


KERNEL_IDS = {"gaussian": 1, "wendland_c2": 2, "cubic_spline": 3}


@pytest.mark.mpi_skip
@pytest.mark.parametrize("kernel", KERNEL_IDS)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_sph_radial_kernel_normalization(kernel, dim):
    """The kernel integrates to one over R^d (radial quadrature), vanishes outside the support and has zero slope
    at the origin (no self-force)."""
    k = KERNEL_IDS[kernel]
    h = 0.37
    # Gauss-Legendre in r on [0, h], with the surface of the unit sphere
    nodes, weights = np.polynomial.legendre.leggauss(400)
    r = 0.5 * h * (nodes + 1.0)
    wq = 0.5 * h * weights
    surface = {1: 2.0, 2: 2.0 * np.pi, 3: 4.0 * np.pi}[dim]
    vals = np.array([sph_radial_kernels.kernel_value(k, dim, ri, h) for ri in r])
    integral = surface * np.sum(wq * vals * r ** (dim - 1))
    assert integral == pytest.approx(1.0, abs=1e-6)

    # (the truncated Gaussian jumps by e^-9 at r = h; the SPH sums skip r >= h)
    for ri in (1.0001 * h, 2.0 * h):
        assert sph_radial_kernels.kernel_value(k, dim, ri, h) == 0.0
        assert sph_radial_kernels.kernel_dr(k, dim, ri, h) == 0.0
    assert sph_radial_kernels.kernel_dr(k, dim, 0.0, h) == 0.0

    # the radial derivative is the derivative of the value (central differences)
    eps = 1e-6
    for ri in (0.1 * h, 0.45 * h, 0.55 * h, 0.9 * h):
        fd = (
            sph_radial_kernels.kernel_value(k, dim, ri + eps, h) - sph_radial_kernels.kernel_value(k, dim, ri - eps, h)
        ) / (2 * eps)
        assert sph_radial_kernels.kernel_dr(k, dim, ri, h) == pytest.approx(fd, rel=1e-5, abs=1e-6 / h ** (dim + 1))


@pytest.mark.parametrize(
    "kernel,n_fields",
    [
        (push_gc_Bstar_discrete_gradient_1st_order, 6),
        (push_gc_Bstar_discrete_gradient_2nd_order, 14),
        (push_gc_Bstar_discrete_gradient_1st_order_newton, 8),
    ],
)
def test_gc_discrete_gradient_residual_at_zero_velocity(derham_args, kernel, n_fields):
    # A marker at rest (v = mu = 0, no E-field) stays put; the residual must be 0, not NaN (issue #589).
    markers = np.zeros((40, 1))
    markers[:3, 0] = (0.4, 0.3, 0.2)
    markers[8:11, 0] = markers[:3, 0]
    markers[18, 0] = markers[21, 0] = 1.0  # B*_parallel and B* = (0, 0, 1) at time n.
    args_markers = MarkerArguments(
        markers, np.array([True]), 1, 2, 5, 6, 8, 12, 16, 17, 4, np.zeros(3, dtype=int), False
    )
    kernel(0.1, 0, args_markers, Cuboid().args_domain, derham_args, 1.0, *(coefficients(1.0),) * n_fields, False)
    assert markers[16, 0] == 0.0
    np.testing.assert_array_equal(markers[:4, 0], (0.4, 0.3, 0.2, 0.0))


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
        markers.markers[indices[0], 0] = values[0]

    def evaluate(alpha, indices, markers, domain):
        events.append("eval")
        markers.markers[indices[0], 0] = markers.markers[20, 0] + alpha[0]

    def push(dt, stage, markers, domain):
        events.append("push")
        assert markers.markers[21, 0] == field[0] + 0.5
        markers.markers[markers.residual_idx, 0] = 1.0  # Force both iterations.

    def sort(**kwargs):
        # Final sort after the last stage uses current positions (no alpha).
        alpha = kwargs.get("alpha")
        events.append("sort" if alpha is None else f"sort{alpha[0]}")
        if alpha is not None:
            assert alpha in ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))

    particles = SimpleNamespace(
        markers=marker_args.markers,
        n_markers_used=marker_args.markers.shape[1],
        args_markers=marker_args,
        n_markers_max=2,
        n_cols=40,
        vdim=3,
        first_pusher_idx=8,
        first_shift_idx=14,
        residual_idx=17,
        n_mks_loc=1,
        Np=1,
        mpi_rank=0,
        mpi_size=2,
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
