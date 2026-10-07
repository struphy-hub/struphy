import pytest
from cunumpy.kernels import PyccelKernel


@pytest.mark.parametrize("basis_u", [0, 2])
def test_cc_lin_mhd_5d_gradB_uses_full_gradB(basis_u):
    r"""The kernel :func:`~struphy.pic.accumulation.accum_kernels_gc.cc_lin_mhd_5d_gradB` must use the
    full gradient :math:`\nabla B_\parallel = \nabla \tilde B_\parallel + \nabla B_{0,\parallel}` for every
    u-space (``basis_u=0``: H1vec, ``basis_u=2``: Hdiv), see issue #435.

    Only the sum of ``grad_PB`` and ``grad_PBeq`` may enter, so passing a gradient field ``g`` either as
    ``grad_PB`` (with ``grad_PBeq=0``) or as ``grad_PBeq`` (with ``grad_PB=0``) must give the same vector.
    """

    import numpy as np

    from struphy import domains
    from struphy.feec.psydac_derham import Derham
    from struphy.io.options import DerhamOptions
    from struphy.kernel_arguments.pusher_args_kernels import MarkerArguments
    from struphy.pic.accumulation import accum_kernels_gc
    from struphy.topology.grids import TensorProductGrid

    rng = np.random.default_rng(435)
    domain = domains.Cuboid(l1=0.0, r1=2.0, l2=0.0, r2=3.0, l3=0.0, r3=0.5)
    derham = Derham(TensorProductGrid(num_elements=[4, 4, 4]), DerhamOptions(degree=[2, 2, 2]), comm=None)

    def rand(space):
        v = derham.coeff_spaces[space].zeros()
        for blk in v.blocks:
            blk._data[:] = rng.standard_normal(blk._data.shape)
        return v

    b, curl_nb = rand("Hdiv"), rand("Hdiv")
    nb, grad = rand("Hcurl"), rand("Hcurl")
    zero = derham.coeff_spaces["Hcurl"].zeros()

    n_mk = 5
    markers = np.zeros((n_mk, 40))
    markers[:, :3] = rng.uniform(0.05, 0.95, (n_mk, 3))
    markers[:, 3] = rng.standard_normal(n_mk)
    markers[:, 4] = rng.uniform(0.1, 1.0, n_mk)
    markers[:, 5] = rng.uniform(0.5, 1.5, n_mk)
    args_markers = MarkerArguments(
        markers, np.ones(n_mk, dtype=bool), n_mk, 2, 5, 6, 10, 20, 25, 30, 4, np.zeros(3, dtype=int)
    )

    space = "H1vec" if basis_u == 0 else "Hdiv"
    dummy_mat = np.zeros((1, 1, 1, 1, 1, 1))

    def accumulate(grad_PB, grad_PBeq):
        vec = derham.coeff_spaces[space].zeros()
        PyccelKernel(accum_kernels_gc.cc_lin_mhd_5d_gradB)(
            args_markers,
            derham.args_derham,
            domain.args_domain,
            *[dummy_mat] * 6,
            *[blk._data for blk in vec.blocks],
            0.1,
            1.0,
            *[blk._data for blk in b.blocks],
            *[blk._data for blk in nb.blocks],
            *[blk._data for blk in curl_nb.blocks],
            *[blk._data for blk in grad_PB.blocks],
            *[blk._data for blk in grad_PBeq.blocks],
            basis_u,
        )
        return np.concatenate([blk._data.flatten() for blk in vec.blocks])

    vec_pert = accumulate(grad, zero)
    vec_eq = accumulate(zero, grad)

    assert np.max(np.abs(vec_pert)) > 0.0
    assert np.allclose(vec_pert, vec_eq, rtol=1e-12, atol=1e-14 * np.max(np.abs(vec_pert)))


if __name__ == "__main__":
    test_cc_lin_mhd_5d_gradB_uses_full_gradB(0)
    test_cc_lin_mhd_5d_gradB_uses_full_gradB(2)
