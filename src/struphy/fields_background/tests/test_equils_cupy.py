"""Fluid and MHD equilibria on the CuPy backend, compared with the NumPy backend.

The equilibria are created and evaluated on CuPy (with a GPU) or on cunumpy's fake CuPy (without one, in a
subprocess), which rejects host/device mixing like CuPy, and every evaluation is compared with the NumPy backend.
"""

import contextlib
import importlib.util
import inspect

import cunumpy
import numpy as np
import pytest
from cunumpy.kernel_testing import requires_cupy

from struphy.geometry.tests.test_domain import _cupy_installed, run_fake_cupy_child

# (equilibrium, its parameters, domain, its parameters); domain None for numerical equilibria (own domain).
# Every equilibrium of `equils` is here (`test_all_equilibria_have_cases`); GVEC and DESC need their packages.
EQUIL_CASES = {
    "HomogenSlab": ("HomogenSlab", {}, "Cuboid", {}),
    "ShearedSlab": ("ShearedSlab", {}, "Cuboid", {"r1": 1.0, "r2": 2 * np.pi, "r3": 2 * np.pi * 10.0}),
    "ShearFluid": ("ShearFluid", {}, "Cuboid", {}),
    "ScrewPinch": ("ScrewPinch", {}, "HollowCylinder", {"a1": 0.05, "a2": 1.0, "Lz": 2 * np.pi * 5.0}),
    "ScrewPinch-q_inf": (
        "ScrewPinch",
        {"q0": "inf", "q1": "inf"},
        "HollowCylinder",
        {"a1": 0.05, "a2": 1.0, "Lz": 2 * np.pi * 5.0},
    ),
    "AdhocTorus-q0p0": ("AdhocTorus", {"q_kind": 0, "p_kind": 0}, "HollowTorus", {"a1": 0.05, "a2": 1.0}),
    "AdhocTorus-q1p0": ("AdhocTorus", {"q_kind": 1, "p_kind": 0}, "HollowTorus", {"a1": 0.05, "a2": 1.0}),
    "AdhocTorus-q2p1": ("AdhocTorus", {"q_kind": 2, "p_kind": 1}, "HollowTorus", {"a1": 0.05, "a2": 1.0}),
    "AdhocTorus-Tokamak": ("AdhocTorus", {"q_kind": 1, "p_kind": 0}, "Tokamak", {"num_elements": (6, 16)}),
    "AdhocTorusQPsi": ("AdhocTorusQPsi", {}, "IGAPolarTorus", {"a": 0.361925, "R0": 1.0, "num_elements": (6, 16)}),
    "CircularTokamak": ("CircularTokamak", {}, "HollowTorus", {"a1": 0.05, "a2": 1.0, "R0": 2.0}),
    "EQDSKequilibrium": ("EQDSKequilibrium", {}, "Tokamak", {"num_elements": (6, 16)}),
    "GVECequilibrium": (
        "GVECequilibrium",
        {
            "dat_file": "run_01/CIRCTOK_State_0000_00000000.dat",
            "param_file": "run_01/parameter.ini",
            "num_elements": (6, 6, 4),
        },
        None,
        {},
    ),
    "DESCequilibrium": ("DESCequilibrium", {"num_elements": (6, 6, 6)}, None, {}),
    "ConstantVelocity": ("ConstantVelocity", {}, "Cuboid", {}),
    "HomogenSlabITG": ("HomogenSlabITG", {}, "Cuboid", {}),
    "CurrentSheet": ("CurrentSheet", {}, "Cuboid", {}),
    "GenericCartesianFluidEquilibrium": ("GenericCartesianFluidEquilibrium", {}, "Cuboid", {}),
    "GenericCartesianFluidEquilibriumWithB": ("GenericCartesianFluidEquilibriumWithB", {}, "Cuboid", {}),
}

# equilibria that need an optional package (skipped without it)
NEEDS = {"GVECequilibrium": "gvec", "DESCequilibrium": "desc"}

# equilibria evaluated through their native code on the host (one round trip per call); all others are evaluated on
# the device only, without host/device transfers
HOST_EVALUATION = {"GVECequilibrium", "DESCequilibrium"}

# methods that models and projections call on the logical domain; those an equilibrium does not provide are skipped
METHODS = (
    "absB0",
    "absB3",
    "p0",
    "p3",
    "n0",
    "n3",
    "t0",
    "vth0",
    "b1",
    "b2",
    "bv",
    "b_cart",
    "unit_b1",
    "unit_b2",
    "unit_bv",
    "gradB1",
    "gradB2",
    "gradBv",
    "j1",
    "j2",
    "jv",
    "absJ0",
    "curl_unit_b1",
    "curl_unit_b2",
    "curl_unit_b_dot_b0",
    "u1",
    "u2",
    "uv",
    "a1",
    "a2",
)

# methods of axisymmetric equilibria in (R, Z) (flux and toroidal field function with derivatives)
AXISYMM_DERIVATIVES = ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1))


def _build(case):
    """Create the equilibrium of ``case`` with its domain on the active backend."""
    from struphy import domains, equils

    name, params, dom_name, dom_params = EQUIL_CASES[case]
    equil = getattr(equils, name)(**params)
    if dom_name is None:
        return equil
    if dom_name == "Tokamak":
        domain = domains.Tokamak(equilibrium=equil, **dom_params)
    else:
        domain = getattr(domains, dom_name)(**dom_params)
    equil.domain = domain
    return equil


def _flatten(out):
    """Arrays in ``out`` (an array, or nested tuples/lists of arrays)."""
    if isinstance(out, (tuple, list)):
        return [a for o in out for a in _flatten(o)]
    return [out]


def _evaluations(equil, args):
    """Evaluate all methods of ``equil`` in ``METHODS`` (and psi, g_tor for axisymmetric ones) at ``args``."""
    from struphy.fields_background.base import AxisymmMHDequilibrium

    out = {}
    for name in METHODS:
        method = getattr(equil, name, None)
        if method is None:
            continue
        try:
            out[name] = method(*args)
        except (AssertionError, NotImplementedError) as error:
            # not available for this equilibrium (vector potential, GVEC gradB1, ...)
            out[name] = type(error)
    if isinstance(equil, AxisymmMHDequilibrium):
        R = 1 * equil.psi_axis_RZ[0] + 0.3 * args[0]
        Z = 1 * equil.psi_axis_RZ[1] + 0.2 * args[0] - 0.1
        for dR, dZ in AXISYMM_DERIVATIVES:
            out[f"psi_{dR}{dZ}"] = equil.psi(R, Z, dR=dR, dZ=dZ)
        for dR, dZ in AXISYMM_DERIVATIVES[:3]:
            out[f"g_tor_{dR}{dZ}"] = equil.g_tor(R, Z, dR=dR, dZ=dZ)
    return out


@contextlib.contextmanager
def host_geometry_kernels():
    """Fake CuPy only: run the geometry kernels of :class:`~struphy.geometry.base.Domain` with their pyccel versions.

    The fake CuPy cannot launch CUDA kernels. Here the four geometry entry kernels run their pyccel version on the
    host buffers of the fake device arrays (shared, so the outputs are written in place), with the pyccel
    ``DomainArguments`` built from the same buffers. The equilibria themselves run unchanged on the fake CuPy. The
    CUDA geometry kernels are checked against pyccel in ``pic/tests/test_cuda_parity.py`` (GPU) and
    ``test_cuda_emulation.py`` (CPU emulation).
    """
    import struphy.geometry.base as geometry_base
    from struphy.kernel_arguments.pusher_args_cuda import CudaDomainArguments
    from struphy.kernel_arguments.pusher_args_kernels import DomainArguments

    def host(arg):
        if isinstance(arg, CudaDomainArguments):
            return DomainArguments(arg.kind_map, *(host(getattr(arg, name)) for name, _ in arg.fields[1:]))
        return arg._a if cunumpy.is_gpu(arg) else arg  # the host buffer of a fake CuPy array

    def on_host(kernel):
        def launch(*args, **launch_options):
            return kernel.host_kernel.kernel(*(host(a) for a in args))

        return launch

    names = ("kernel_evaluate", "kernel_evaluate_pic", "kernel_pullpush", "kernel_pullpush_pic")
    kernels = {name: getattr(geometry_base, name) for name in names}
    try:
        for name, kernel in kernels.items():
            setattr(geometry_base, name, on_host(kernel))
        yield
    finally:
        for name, kernel in kernels.items():
            setattr(geometry_base, name, kernel)


def check_equil_on_cupy(case):
    """Create ``case`` on the CuPy backend, evaluate it there and compare with the NumPy backend.

    Every result is a device array (or a scalar for constant parts like ``psi`` derivatives of analytic profiles) and
    agrees with the NumPy result. Evaluated on a meshgrid (three 1d arrays) and at markers (one 2d array), without
    host/device transfers except for the equilibria in ``HOST_EVALUATION``.
    """
    rng = np.random.default_rng(1234)
    e1 = np.sort(rng.uniform(0.1, 0.9, 4))
    e2 = np.sort(rng.uniform(0.0, 1.0, 5))
    e3 = np.sort(rng.uniform(0.0, 1.0, 3))
    markers = np.column_stack([rng.uniform(0.1, 0.9, 7), rng.uniform(0.0, 1.0, 7), rng.uniform(0.0, 1.0, 7)])

    with cunumpy.use_backend("numpy"):
        equil = _build(case)
        ref_grid = _evaluations(equil, (e1, e2, e3))
        ref_markers = _evaluations(equil, (markers,))

    with cunumpy.use_backend("cupy"):
        equil = _build(case)
        grid = tuple(cunumpy.asarray(e) for e in (e1, e2, e3))
        markers_device = cunumpy.asarray(markers)
        if EQUIL_CASES[case][0] in HOST_EVALUATION:
            no_transfers = contextlib.nullcontext()
        else:
            no_transfers = cunumpy.profiling.assert_no_transfers()
        with no_transfers:
            out_grid = _evaluations(equil, grid)
            out_markers = _evaluations(equil, (markers_device,))

    for ref, out, where in ((ref_grid, out_grid, "meshgrid"), (ref_markers, out_markers, "markers")):
        assert ref.keys() == out.keys()
        for name in ref:
            if isinstance(ref[name], type):  # not available on NumPy, not on CuPy either
                assert out[name] is ref[name], (case, where, name)
                continue
            refs, outs = _flatten(ref[name]), _flatten(out[name])
            assert len(refs) == len(outs), (case, where, name)
            for r, o in zip(refs, outs):
                if isinstance(r, np.ndarray) and r.ndim > 0:
                    assert cunumpy.is_gpu(o), (case, where, name, type(o))
                o = cunumpy.to_numpy(o) if cunumpy.is_gpu(o) else np.asarray(o)
                assert np.allclose(o, r, rtol=1e-12, atol=1e-12, equal_nan=True), (case, where, name)


def _check_needs(case, with_gvec=False):
    if EQUIL_CASES[case][0] == "GVECequilibrium" and not with_gvec:
        pytest.skip("GVEC not tested here (with_gvec=False), like the other GVEC tests")
    package = NEEDS.get(EQUIL_CASES[case][0])
    if package is not None and importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is not installed")


def test_all_equilibria_have_cases():
    """Every equilibrium class of ``equils`` is checked here."""
    from struphy import equils
    from struphy.fields_background.base import FluidEquilibrium

    classes = {
        name
        for name, cls in vars(equils).items()
        if inspect.isclass(cls)
        and issubclass(cls, FluidEquilibrium)
        and not inspect.isabstract(cls)
        and cls.__module__ == equils.__name__
    }
    covered = {case[0] for case in EQUIL_CASES.values()}
    assert classes == covered


def test_host_call_numpy_is_plain_call():
    """On the NumPy backend, :func:`cunumpy.host_call` calls the function directly and returns its result unchanged."""
    a = np.linspace(0.0, 1.0, 5)
    with cunumpy.use_backend("numpy"):
        out = cunumpy.host_call(lambda x, y=1.0: x * y, a, y=2.0)
    assert type(out) is np.ndarray
    assert np.array_equal(out, 2 * a)


@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
@pytest.mark.parametrize("case", list(EQUIL_CASES))
def test_equil_fake_cupy(case, with_gvec=False):
    """Without a GPU: :func:`check_equil_on_cupy` with cunumpy's fake CuPy, which rejects host/device mixing.

    Runs in a subprocess because the fake CuPy must be installed before cunumpy is imported.
    """
    _check_needs(case, with_gvec)
    code = (
        "from struphy.fields_background.tests.test_equils_cupy import check_equil_on_cupy, host_geometry_kernels\n"
        f"with host_geometry_kernels(): check_equil_on_cupy({case!r})"
    )
    run_fake_cupy_child(code)


@requires_cupy
@pytest.mark.parametrize("case", list(EQUIL_CASES))
def test_equil_cupy(case, with_gvec=False):
    """On a GPU: the equilibria are created and evaluated on CuPy and agree with NumPy."""
    _check_needs(case, with_gvec)
    check_equil_on_cupy(case)
