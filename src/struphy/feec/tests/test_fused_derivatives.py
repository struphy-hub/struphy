"""The fused gradient with boundary operators (struphy.feec.fused_derivatives) against the composed operators.

On the host the fused operator applies the composed one. Without a GPU, the CUDA kernels run on cunumpy's fake CuPy with
the launches emulated on the CPU (in a subprocess) and must give bit for bit the products of the composed operators.
"""

import numpy as np
import pytest

from struphy.geometry.tests.test_domain import _cupy_installed, run_fake_cupy_child

# (degree, boundary conditions): Dirichlet at both ends, at one end, none (periodic)
CASES = {
    "dirichlet": ((2, 3, 1), (("dirichlet", "dirichlet"), None, None)),
    "mixed": ((1, 2, 3), (None, ("dirichlet", "free"), None)),
    "periodic": ((3, 1, 2), (None, None, None)),
}


def _derham(case):
    from maybempi import MPI

    from struphy import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.topology.grids import TensorProductGrid

    degree, bcs = CASES[case]
    return Derham(
        TensorProductGrid(num_elements=(6, 5, 4)),
        DerhamOptions(degree=degree, bcs=bcs),
        comm=MPI.COMM_WORLD,
        domain=domains.Cuboid(),
    )


def _random(space, seed):
    import cunumpy as xp

    rng = np.random.default_rng(seed)
    v = space.zeros()
    for b in getattr(v, "blocks", [v]):
        b._data[...] = xp.asarray(rng.standard_normal(b._data.shape))
    v.update_ghost_regions()
    return v


def _owned(v):
    import cunumpy as xp

    return xp.to_numpy(v.toarray())


@pytest.mark.parametrize("case", list(CASES))
def test_fuse_recognizes_gradient(case):
    """``derham.grad`` and its transpose are fused; on the host the fused operators give the composed products.

    Without essential boundary conditions the gradient has no boundary operators (only the derivatives), and it is
    returned as it is."""
    from struphy.feec.fused_derivatives import FusedBoundaryGradient, fuse_boundary_gradient

    derham = _derham(case)
    for op in (derham.grad, derham.grad.T):
        fused = fuse_boundary_gradient(op)
        if case == "periodic":
            assert fused is op
            continue
        assert isinstance(fused, FusedBoundaryGradient)
        assert fused.domain is op.domain and fused.codomain is op.codomain
        v = _random(op.domain, 1)
        np.testing.assert_array_equal(_owned(fused.dot(v)), _owned(op.dot(v)))
        assert isinstance(fused.T, FusedBoundaryGradient)
    # other operators are returned as they are
    assert fuse_boundary_gradient(derham.curl) is derham.curl


def check_fused_gradient_on_device():
    """On the CuPy backend: the kernels give bit for bit the products of the composed operators."""
    import cunumpy as xp

    import struphy.feec.fused_derivatives as fd
    from struphy.feec.fused_derivatives import fuse_boundary_gradient

    # count the launches of the two kernels: the checks below must not pass through the composed operators
    kernels, launches = fd._kernels, [0, 0]

    def counted(k, i):
        def launch(*args, **kwargs):
            launches[i] += 1
            return k(*args, **kwargs)

        return launch

    fd._kernels = lambda: tuple(counted(k, i) for i, k in enumerate(kernels()))
    try:
        _check_fused_gradient(xp, fuse_boundary_gradient)
    finally:
        fd._kernels = kernels
    assert launches[0] > 0 and launches[1] > 0, launches


def _check_fused_gradient(xp, fuse_boundary_gradient):
    with xp.use_backend("cupy"):
        for case in CASES:
            derham = _derham(case)
            for op in (derham.grad, derham.grad.T):
                fused = fuse_boundary_gradient(op)
                if fused is op:
                    continue
                v = _random(op.domain, 2)
                expected = _owned(op.dot(v))
                out = op.codomain.zeros()
                result = fused._dot_fused(v, out=out)
                assert result is out
                assert not out.ghost_regions_in_sync
                assert np.abs(expected).max() > 0, case
                np.testing.assert_array_equal(_owned(result), expected, err_msg=case)
                # an input whose ghost regions are not in sync
                v.ghost_regions_in_sync = False
                np.testing.assert_array_equal(_owned(fused._dot_fused(v)), expected, err_msg=case)


def check_fused_gradient_on_fake_cupy():
    """:func:`check_fused_gradient_on_device` with the CUDA kernel launches emulated on the CPU."""
    from cunumpy.kernel_testing import emulated_launches, fake_cupy_active

    from struphy.pic.tests.cuda_emulation import EMULATION_OPTIONS

    assert fake_cupy_active()
    with emulated_launches(options=EMULATION_OPTIONS):
        check_fused_gradient_on_device()


@pytest.mark.skipif(_cupy_installed(), reason="the fake CuPy cannot replace an installed CuPy")
def test_fused_gradient_fake_cupy():
    """Without a GPU: the kernels against the composed operators on cunumpy's fake CuPy (serial subprocess)."""
    run_fake_cupy_child(
        "from struphy.feec.tests.test_fused_derivatives import check_fused_gradient_on_fake_cupy; "
        "check_fused_gradient_on_fake_cupy()"
    )


@pytest.mark.skipif(not _cupy_installed(), reason="needs CuPy and a GPU")
def test_fused_gradient_on_gpu():
    check_fused_gradient_on_device()
