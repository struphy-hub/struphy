import time

import pytest
from psydac.ddm.mpi import mpi as MPI

from struphy.utils.arrays import xp


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[1, 2, 1], [2, 1, 2], [3, 4, 3]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [False, True, False], [True, False, False]])
def test_bsplines_span_and_basis(Nel, p, spl_kind):
    """
    Compare Struphy and Psydac bsplines kernels for knot spans and basis values computation.
    Print timings.
    """

    import psydac.core.bsplines_kernels as bsp_psy

    import struphy.bsplines.bsplines_kernels as bsp
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays as cera

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # knot vectors
    tn1, tn2, tn3 = derham.Vh_fem["0"].knots
    td1, td2, td3 = derham.Vh_fem["3"].knots

    # Random points in domain of process
    n_pts = 100
    dom = derham.domain_array[rank]
    eta1s = xp.random.rand(n_pts) * (dom[1] - dom[0]) + dom[0]
    eta2s = xp.random.rand(n_pts) * (dom[4] - dom[3]) + dom[3]
    eta3s = xp.random.rand(n_pts) * (dom[7] - dom[6]) + dom[6]

    # struphy find_span
    t0 = time.time()
    span1s, span2s, span3s = [], [], []
    for eta1, eta2, eta3 in zip(eta1s, eta2s, eta3s):
        span1s += [bsp.find_span(tn1, derham.p[0], eta1)]
        span2s += [bsp.find_span(tn2, derham.p[1], eta2)]
        span3s += [bsp.find_span(tn3, derham.p[2], eta3)]
    t1 = time.time()
    if rank == 0:
        print(f"struphy find_span  : {t1 - t0}")

    # psydac find_span_p
    t0 = time.time()
    span1s_psy, span2s_psy, span3s_psy = [], [], []
    for eta1, eta2, eta3 in zip(eta1s, eta2s, eta3s):
        span1s_psy += [bsp_psy.find_span_p(tn1, derham.p[0], eta1)]
        span2s_psy += [bsp_psy.find_span_p(tn2, derham.p[1], eta2)]
        span3s_psy += [bsp_psy.find_span_p(tn3, derham.p[2], eta3)]
    t1 = time.time()
    if rank == 0:
        print(f"psydac find_span_p : {t1 - t0}")

    assert xp.allclose(span1s, span1s_psy)
    assert xp.allclose(span2s, span2s_psy)
    assert xp.allclose(span3s, span3s_psy)

    # allocate tmps
    bn1 = xp.empty(derham.p[0] + 1, dtype=float)
    bn2 = xp.empty(derham.p[1] + 1, dtype=float)
    bn3 = xp.empty(derham.p[2] + 1, dtype=float)

    bd1 = xp.empty(derham.p[0], dtype=float)
    bd2 = xp.empty(derham.p[1], dtype=float)
    bd3 = xp.empty(derham.p[2], dtype=float)

    # struphy b_splines_slim
    val1s, val2s, val3s = [], [], []
    t0 = time.time()
    for eta1, eta2, eta3, span1, span2, span3 in zip(eta1s, eta2s, eta3s, span1s, span2s, span3s):
        bsp.b_splines_slim(tn1, derham.p[0], eta1, span1, bn1)
        bsp.b_splines_slim(tn2, derham.p[1], eta2, span2, bn2)
        bsp.b_splines_slim(tn3, derham.p[2], eta3, span3, bn3)
        val1s += [bn1]
        val2s += [bn2]
        val3s += [bn3]
    t1 = time.time()
    if rank == 0:
        print(f"bsp.b_splines_slim        : {t1 - t0}")

    # psydac basis_funs_p
    val1s_psy, val2s_psy, val3s_psy = [], [], []
    t0 = time.time()
    for eta1, eta2, eta3, span1, span2, span3 in zip(eta1s, eta2s, eta3s, span1s, span2s, span3s):
        bsp_psy.basis_funs_p(tn1, derham.p[0], eta1, span1, bn1)
        bsp_psy.basis_funs_p(tn2, derham.p[1], eta2, span2, bn2)
        bsp_psy.basis_funs_p(tn3, derham.p[2], eta3, span3, bn3)
        val1s_psy += [bn1]
        val2s_psy += [bn2]
        val3s_psy += [bn3]
    t1 = time.time()
    if rank == 0:
        print(f"bsp_psy.basis_funs_p for N: {t1 - t0}")

    # compare
    for val1, val1_psy in zip(val1s, val1s_psy):
        assert xp.allclose(val1, val1_psy)

    for val2, val2_psy in zip(val2s, val2s_psy):
        assert xp.allclose(val2, val2_psy)

    for val3, val3_psy in zip(val3s, val3s_psy):
        assert xp.allclose(val3, val3_psy)

    # struphy b_d_splines_slim
    val1s_n, val2s_n, val3s_n = [], [], []
    val1s_d, val2s_d, val3s_d = [], [], []
    t0 = time.time()
    for eta1, eta2, eta3, span1, span2, span3 in zip(eta1s, eta2s, eta3s, span1s, span2s, span3s):
        bsp.b_d_splines_slim(tn1, derham.p[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, derham.p[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, derham.p[2], eta3, span3, bn3, bd3)
        val1s_n += [bn1]
        val2s_n += [bn2]
        val3s_n += [bn3]
        val1s_d += [bd1]
        val2s_d += [bd2]
        val3s_d += [bd3]
    t1 = time.time()
    if rank == 0:
        print(f"bsp.b_d_splines_slim      : {t1 - t0}")

    # compare
    for val1, val1_psy in zip(val1s_n, val1s_psy):
        assert xp.allclose(val1, val1_psy)

    for val2, val2_psy in zip(val2s_n, val2s_psy):
        assert xp.allclose(val2, val2_psy)

    for val3, val3_psy in zip(val3s_n, val3s_psy):
        assert xp.allclose(val3, val3_psy)

    # struphy d_splines_slim
    span1s, span2s, span3s = [], [], []
    for eta1, eta2, eta3 in zip(eta1s, eta2s, eta3s):
        span1s += [bsp.find_span(td1, derham.p[0], eta1)]
        span2s += [bsp.find_span(td2, derham.p[1], eta2)]
        span3s += [bsp.find_span(td3, derham.p[2], eta3)]

    val1s, val2s, val3s = [], [], []
    t0 = time.time()
    for eta1, eta2, eta3, span1, span2, span3 in zip(eta1s, eta2s, eta3s, span1s, span2s, span3s):
        bsp.d_splines_slim(td1, derham.p[0], eta1, span1, bd1)
        bsp.d_splines_slim(td2, derham.p[1], eta2, span2, bd2)
        bsp.d_splines_slim(td3, derham.p[2], eta3, span3, bd3)
        val1s += [bd1]
        val2s += [bd2]
        val3s += [bd3]
    t1 = time.time()
    if rank == 0:
        print(f"bsp.d_splines_slim        : {t1 - t0}")

    # psydac basis_funs_p for D-splines
    val1s_psy, val2s_psy, val3s_psy = [], [], []
    t0 = time.time()
    for eta1, eta2, eta3, span1, span2, span3 in zip(eta1s, eta2s, eta3s, span1s, span2s, span3s):
        bsp_psy.basis_funs_p(td1, derham.p[0] - 1, eta1, span1, bd1)
        bsp_psy.basis_funs_p(td2, derham.p[1] - 1, eta2, span2, bd2)
        bsp_psy.basis_funs_p(td3, derham.p[2] - 1, eta3, span3, bd3)
        val1s_psy += [bd1]
        val2s_psy += [bd2]
        val3s_psy += [bd3]
    t1 = time.time()
    if rank == 0:
        print(f"bsp_psy.basis_funs_p for D: {t1 - t0}")

    # compare
    for val1, val1_psy in zip(val1s, val1s_psy):
        assert xp.allclose(val1, val1_psy)

    for val2, val2_psy in zip(val2s, val2s_psy):
        assert xp.allclose(val2, val2_psy)

    for val3, val3_psy in zip(val3s, val3s_psy):
        assert xp.allclose(val3, val3_psy)

    for val1, val1_psy in zip(val1s_d, val1s_psy):
        assert xp.allclose(val1, val1_psy)

    for val2, val2_psy in zip(val2s_d, val2s_psy):
        assert xp.allclose(val2, val2_psy)

    for val3, val3_psy in zip(val3s_d, val3s_psy):
        assert xp.allclose(val3, val3_psy)


if __name__ == "__main__":
    test_bsplines_span_and_basis([8, 9, 10], [3, 4, 3], [False, False, True])
