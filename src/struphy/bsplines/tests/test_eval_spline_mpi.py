from sys import int_info
from time import sleep

import pytest
from psydac.ddm.mpi import mpi as MPI

from struphy.utils.arrays import xp


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[1, 2, 3], [3, 1, 2]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [False, True, False], [True, False, False]])
def test_eval_kernels(Nel, p, spl_kind, n_markers=10):
    """Compares evaluation_kernel_3d with eval_spline_mpi_kernel."""

    from struphy.bsplines import bsplines_kernels as bsp
    from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi_kernel as eval3d_mpi
    from struphy.bsplines.evaluation_kernels_3d import evaluation_kernel_3d as eval3d
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays as cera

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # derham attributes
    tn1, tn2, tn3 = derham.Vh_fem["0"].knots
    indN = derham.indN
    indD = derham.indD

    # Random spline coeffs_loc
    x0, x0_psy = cera(derham.Vh_fem["0"])
    x1, x1_psy = cera(derham.Vh_fem["1"])
    x2, x2_psy = cera(derham.Vh_fem["2"])
    x3, x3_psy = cera(derham.Vh_fem["3"])

    # Random points in domain of process
    dom = derham.domain_array[rank]
    eta1s = xp.random.rand(n_markers) * (dom[1] - dom[0]) + dom[0]
    eta2s = xp.random.rand(n_markers) * (dom[4] - dom[3]) + dom[3]
    eta3s = xp.random.rand(n_markers) * (dom[7] - dom[6]) + dom[6]

    for eta1, eta2, eta3 in zip(eta1s, eta2s, eta3s):
        comm.Barrier()
        sleep(0.02 * (rank + 1))
        print(f"rank {rank} | eta1 = {eta1}")
        print(f"rank {rank} | eta2 = {eta2}")
        print(f"rank {rank} | eta3 = {eta3}\n")
        comm.Barrier()

        # spans (i.e. index for non-vanishing basis functions)
        span1 = bsp.find_span(tn1, derham.p[0], eta1)
        span2 = bsp.find_span(tn2, derham.p[1], eta2)
        span3 = bsp.find_span(tn3, derham.p[2], eta3)

        # non-zero spline values at eta
        bn1 = xp.empty(derham.p[0] + 1, dtype=float)
        bn2 = xp.empty(derham.p[1] + 1, dtype=float)
        bn3 = xp.empty(derham.p[2] + 1, dtype=float)

        bd1 = xp.empty(derham.p[0], dtype=float)
        bd2 = xp.empty(derham.p[1], dtype=float)
        bd3 = xp.empty(derham.p[2], dtype=float)

        bsp.b_d_splines_slim(tn1, derham.p[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, derham.p[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, derham.p[2], eta3, span3, bn3, bd3)

        # Non-vanishing B- and D-spline indices at eta (needed for the non-mpi routines)
        ie1 = span1 - derham.p[0]
        ie2 = span2 - derham.p[1]
        ie3 = span3 - derham.p[2]

        ind_n1 = indN[0][ie1]
        ind_n2 = indN[1][ie2]
        ind_n3 = indN[2][ie3]

        ind_d1 = indD[0][ie1]
        ind_d2 = indD[1][ie2]
        ind_d3 = indD[2][ie3]

        # compare spline evaluation routines in V0
        val = eval3d(*derham.p, bn1, bn2, bn3, ind_n1, ind_n2, ind_n3, x0[0])
        val_mpi = eval3d_mpi(*derham.p, bn1, bn2, bn3, span1, span2, span3, x0_psy._data, xp.array(x0_psy.starts))
        assert xp.allclose(val, val_mpi)

        # compare spline evaluation routines in V1
        val = eval3d(derham.p[0] - 1, derham.p[1], derham.p[2], bd1, bn2, bn3, ind_d1, ind_n2, ind_n3, x1[0])
        val_mpi = eval3d_mpi(
            derham.p[0] - 1,
            derham.p[1],
            derham.p[2],
            bd1,
            bn2,
            bn3,
            span1,
            span2,
            span3,
            x1_psy[0]._data,
            xp.array(x1_psy[0].starts),
        )
        assert xp.allclose(val, val_mpi)

        val = eval3d(derham.p[0], derham.p[1] - 1, derham.p[2], bn1, bd2, bn3, ind_n1, ind_d2, ind_n3, x1[1])
        val_mpi = eval3d_mpi(
            derham.p[0],
            derham.p[1] - 1,
            derham.p[2],
            bn1,
            bd2,
            bn3,
            span1,
            span2,
            span3,
            x1_psy[1]._data,
            xp.array(x1_psy[1].starts),
        )
        assert xp.allclose(val, val_mpi)

        val = eval3d(derham.p[0], derham.p[1], derham.p[2] - 1, bn1, bn2, bd3, ind_n1, ind_n2, ind_d3, x1[2])
        val_mpi = eval3d_mpi(
            derham.p[0],
            derham.p[1],
            derham.p[2] - 1,
            bn1,
            bn2,
            bd3,
            span1,
            span2,
            span3,
            x1_psy[2]._data,
            xp.array(x1_psy[2].starts),
        )
        assert xp.allclose(val, val_mpi)

        # compare spline evaluation routines in V2
        val = eval3d(derham.p[0], derham.p[1] - 1, derham.p[2] - 1, bn1, bd2, bd3, ind_n1, ind_d2, ind_d3, x2[0])
        val_mpi = eval3d_mpi(
            derham.p[0],
            derham.p[1] - 1,
            derham.p[2] - 1,
            bn1,
            bd2,
            bd3,
            span1,
            span2,
            span3,
            x2_psy[0]._data,
            xp.array(x2_psy[0].starts),
        )
        assert xp.allclose(val, val_mpi)

        val = eval3d(derham.p[0] - 1, derham.p[1], derham.p[2] - 1, bd1, bn2, bd3, ind_d1, ind_n2, ind_d3, x2[1])
        val_mpi = eval3d_mpi(
            derham.p[0] - 1,
            derham.p[1],
            derham.p[2] - 1,
            bd1,
            bn2,
            bd3,
            span1,
            span2,
            span3,
            x2_psy[1]._data,
            xp.array(x2_psy[1].starts),
        )
        assert xp.allclose(val, val_mpi)

        val = eval3d(derham.p[0] - 1, derham.p[1] - 1, derham.p[2], bd1, bd2, bn3, ind_d1, ind_d2, ind_n3, x2[2])
        val_mpi = eval3d_mpi(
            derham.p[0] - 1,
            derham.p[1] - 1,
            derham.p[2],
            bd1,
            bd2,
            bn3,
            span1,
            span2,
            span3,
            x2_psy[2]._data,
            xp.array(x2_psy[2].starts),
        )
        assert xp.allclose(val, val_mpi)

        # compare spline evaluation routines in V3
        val = eval3d(derham.p[0] - 1, derham.p[1] - 1, derham.p[2] - 1, bd1, bd2, bd3, ind_d1, ind_d2, ind_d3, x3[0])
        val_mpi = eval3d_mpi(
            derham.p[0] - 1,
            derham.p[1] - 1,
            derham.p[2] - 1,
            bd1,
            bd2,
            bd3,
            span1,
            span2,
            span3,
            x3_psy._data,
            xp.array(x3_psy.starts),
        )
        assert xp.allclose(val, val_mpi)


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[1, 2, 3], [3, 1, 2]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [False, True, False], [True, False, False]])
def test_eval_pointwise(Nel, p, spl_kind, n_markers=10):
    """Compares evaluate_3d with eval_spline_mpi."""

    from struphy.bsplines import bsplines_kernels as bsp
    from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi, evaluate_3d
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays as cera

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # derham attributes
    tn1, tn2, tn3 = derham.Vh_fem["0"].knots

    # Random spline coeffs_loc
    x0, x0_psy = cera(derham.Vh_fem["0"])
    x1, x1_psy = cera(derham.Vh_fem["1"])
    x2, x2_psy = cera(derham.Vh_fem["2"])
    x3, x3_psy = cera(derham.Vh_fem["3"])

    # Random points in domain of process
    dom = derham.domain_array[rank]
    eta1s = xp.random.rand(n_markers) * (dom[1] - dom[0]) + dom[0]
    eta2s = xp.random.rand(n_markers) * (dom[4] - dom[3]) + dom[3]
    eta3s = xp.random.rand(n_markers) * (dom[7] - dom[6]) + dom[6]

    for eta1, eta2, eta3 in zip(eta1s, eta2s, eta3s):
        comm.Barrier()
        sleep(0.02 * (rank + 1))
        print(f"rank {rank} | eta1 = {eta1}")
        print(f"rank {rank} | eta2 = {eta2}")
        print(f"rank {rank} | eta3 = {eta3}\n")
        comm.Barrier()

        # compare spline evaluation routines in V0
        val = evaluate_3d(1, 1, 1, tn1, tn2, tn3, *derham.p, *derham.indN, x0[0], eta1, eta2, eta3)

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x0_psy._data,
            derham.spline_types_pyccel["0"],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)

        # compare spline evaluation routines in V1
        # 1st component
        val = evaluate_3d(
            2,
            1,
            1,
            tn1[1:-1],
            tn2,
            tn3,
            derham.p[0] - 1,
            derham.p[1],
            derham.p[2],
            derham.indD[0],
            derham.indN[1],
            derham.indN[2],
            x1[0],
            eta1,
            eta2,
            eta3,
        )

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x1_psy[0]._data,
            derham.spline_types_pyccel["1"][0],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)

        # 2nd component
        val = evaluate_3d(
            1,
            2,
            1,
            tn1,
            tn2[1:-1],
            tn3,
            derham.p[0],
            derham.p[1] - 1,
            derham.p[2],
            derham.indN[0],
            derham.indD[1],
            derham.indN[2],
            x1[1],
            eta1,
            eta2,
            eta3,
        )

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x1_psy[1]._data,
            derham.spline_types_pyccel["1"][1],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)

        # 3rd component
        val = evaluate_3d(
            1,
            1,
            2,
            tn1,
            tn2,
            tn3[1:-1],
            derham.p[0],
            derham.p[1],
            derham.p[2] - 1,
            derham.indN[0],
            derham.indN[1],
            derham.indD[2],
            x1[2],
            eta1,
            eta2,
            eta3,
        )

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x1_psy[2]._data,
            derham.spline_types_pyccel["1"][2],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)

        # compare spline evaluation routines in V2
        # 1st component
        val = evaluate_3d(
            1,
            2,
            2,
            tn1,
            tn2[1:-1],
            tn3[1:-1],
            derham.p[0],
            derham.p[1] - 1,
            derham.p[2] - 1,
            derham.indN[0],
            derham.indD[1],
            derham.indD[2],
            x2[0],
            eta1,
            eta2,
            eta3,
        )

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x2_psy[0]._data,
            derham.spline_types_pyccel["2"][0],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)

        # 2nd component
        val = evaluate_3d(
            2,
            1,
            2,
            tn1[1:-1],
            tn2,
            tn3[1:-1],
            derham.p[0] - 1,
            derham.p[1],
            derham.p[2] - 1,
            derham.indD[0],
            derham.indN[1],
            derham.indD[2],
            x2[1],
            eta1,
            eta2,
            eta3,
        )

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x2_psy[1]._data,
            derham.spline_types_pyccel["2"][1],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)

        # 3rd component
        val = evaluate_3d(
            2,
            2,
            1,
            tn1[1:-1],
            tn2[1:-1],
            tn3,
            derham.p[0] - 1,
            derham.p[1] - 1,
            derham.p[2],
            derham.indD[0],
            derham.indD[1],
            derham.indN[2],
            x2[2],
            eta1,
            eta2,
            eta3,
        )

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x2_psy[2]._data,
            derham.spline_types_pyccel["2"][2],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)

        # compare spline evaluation routines in V3
        val = evaluate_3d(
            2,
            2,
            2,
            tn1[1:-1],
            tn2[1:-1],
            tn3[1:-1],
            derham.p[0] - 1,
            derham.p[1] - 1,
            derham.p[2] - 1,
            *derham.indD,
            x3[0],
            eta1,
            eta2,
            eta3,
        )

        val_mpi = eval_spline_mpi(
            eta1,
            eta2,
            eta3,
            x3_psy._data,
            derham.spline_types_pyccel["3"],
            xp.array(derham.p),
            tn1,
            tn2,
            tn3,
            xp.array(x0_psy.starts),
        )

        assert xp.allclose(val, val_mpi)


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[1, 2, 3], [3, 1, 2]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [False, True, False], [True, False, False]])
def test_eval_tensor_product(Nel, p, spl_kind, n_markers=10):
    """Compares

    evaluate_tensor_product
    eval_spline_mpi_tensor_product
    eval_spline_mpi_tensor_product_fast

    on random tensor product points.
    """

    import time

    from struphy.bsplines.evaluation_kernels_3d import (
        eval_spline_mpi_tensor_product,
        eval_spline_mpi_tensor_product_fast,
        evaluate_tensor_product,
    )
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays as cera

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # derham attributes
    tn1, tn2, tn3 = derham.Vh_fem["0"].knots

    # Random spline coeffs_loc
    x0, x0_psy = cera(derham.Vh_fem["0"])
    x3, x3_psy = cera(derham.Vh_fem["3"])

    # Random points in domain of process
    dom = derham.domain_array[rank]
    eta1s = xp.random.rand(n_markers) * (dom[1] - dom[0]) + dom[0]
    eta2s = xp.random.rand(n_markers + 1) * (dom[4] - dom[3]) + dom[3]
    eta3s = xp.random.rand(n_markers + 2) * (dom[7] - dom[6]) + dom[6]

    vals = xp.zeros((n_markers, n_markers + 1, n_markers + 2), dtype=float)
    vals_mpi = xp.zeros((n_markers, n_markers + 1, n_markers + 2), dtype=float)
    vals_mpi_fast = xp.zeros((n_markers, n_markers + 1, n_markers + 2), dtype=float)

    comm.Barrier()
    sleep(0.02 * (rank + 1))
    print(f"rank {rank} | eta1 = {eta1s}")
    print(f"rank {rank} | eta2 = {eta2s}")
    print(f"rank {rank} | eta3 = {eta3s}\n")
    comm.Barrier()

    # compare spline evaluation routines in V0
    t0 = time.time()
    evaluate_tensor_product(tn1, tn2, tn3, *derham.p, *derham.indN, x0[0], eta1s, eta2s, eta3s, vals, 0)
    t1 = time.time()
    if rank == 0:
        print("V0 evaluate_tensor_product:".ljust(40), t1 - t0)

    t0 = time.time()
    eval_spline_mpi_tensor_product(
        eta1s,
        eta2s,
        eta3s,
        x0_psy._data,
        derham.spline_types_pyccel["0"],
        xp.array(derham.p),
        tn1,
        tn2,
        tn3,
        xp.array(x0_psy.starts),
        vals_mpi,
    )
    t1 = time.time()
    if rank == 0:
        print("V0 eval_spline_mpi_tensor_product:".ljust(40), t1 - t0)

    t0 = time.time()
    eval_spline_mpi_tensor_product_fast(
        eta1s,
        eta2s,
        eta3s,
        x0_psy._data,
        derham.spline_types_pyccel["0"],
        xp.array(derham.p),
        tn1,
        tn2,
        tn3,
        xp.array(x0_psy.starts),
        vals_mpi_fast,
    )
    t1 = time.time()
    if rank == 0:
        print("v0 eval_spline_mpi_tensor_product_fast:".ljust(40), t1 - t0)

    assert xp.allclose(vals, vals_mpi)
    assert xp.allclose(vals, vals_mpi_fast)

    # compare spline evaluation routines in V3
    t0 = time.time()
    evaluate_tensor_product(
        tn1[1:-1],
        tn2[1:-1],
        tn3[1:-1],
        derham.p[0] - 1,
        derham.p[1] - 1,
        derham.p[2] - 1,
        *derham.indD,
        x3[0],
        eta1s,
        eta2s,
        eta3s,
        vals,
        3,
    )
    t1 = time.time()
    if rank == 0:
        print("V3 evaluate_tensor_product:".ljust(40), t1 - t0)

    t0 = time.time()
    eval_spline_mpi_tensor_product(
        eta1s,
        eta2s,
        eta3s,
        x3_psy._data,
        derham.spline_types_pyccel["3"],
        xp.array(derham.p),
        tn1,
        tn2,
        tn3,
        xp.array(x0_psy.starts),
        vals_mpi,
    )
    t1 = time.time()
    if rank == 0:
        print("V3 eval_spline_mpi_tensor_product:".ljust(40), t1 - t0)

    t0 = time.time()
    eval_spline_mpi_tensor_product_fast(
        eta1s,
        eta2s,
        eta3s,
        x3_psy._data,
        derham.spline_types_pyccel["3"],
        xp.array(derham.p),
        tn1,
        tn2,
        tn3,
        xp.array(x0_psy.starts),
        vals_mpi_fast,
    )
    t1 = time.time()
    if rank == 0:
        print("v3 eval_spline_mpi_tensor_product_fast:".ljust(40), t1 - t0)

    assert xp.allclose(vals, vals_mpi)
    assert xp.allclose(vals, vals_mpi_fast)


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[1, 2, 1], [2, 1, 2], [3, 4, 3]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [False, True, False], [True, False, False]])
def test_eval_tensor_product_grid(Nel, p, spl_kind, n_markers=10):
    """Compares

    evaluate_tensor_product
    eval_spline_mpi_tensor_product_fixed

    on histopolation grid of V3.
    """

    import time

    from struphy.bsplines.evaluation_kernels_3d import eval_spline_mpi_tensor_product_fixed, evaluate_tensor_product
    from struphy.feec.basis_projection_ops import prepare_projection_of_basis
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import create_equal_random_arrays as cera

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # derham attributes
    tn1, tn2, tn3 = derham.Vh_fem["0"].knots

    # Random spline coeffs_loc
    x0, x0_psy = cera(derham.Vh_fem["0"])
    x3, x3_psy = cera(derham.Vh_fem["3"])

    # Histopolation grids
    spaces = derham.Vh_fem["3"].spaces
    ptsG, wtsG, spans, bases, subs = prepare_projection_of_basis(
        spaces, spaces, derham.Vh["3"].starts, derham.Vh["3"].ends
    )
    eta1s = ptsG[0].flatten()
    eta2s = ptsG[1].flatten()
    eta3s = ptsG[2].flatten()

    spans_f, bns_f, bds_f = derham.prepare_eval_tp_fixed([eta1s, eta2s, eta3s])

    # output arrays
    vals = xp.zeros((eta1s.size, eta2s.size, eta3s.size), dtype=float)
    vals_mpi_fixed = xp.zeros((eta1s.size, eta2s.size, eta3s.size), dtype=float)
    vals_mpi_grid = xp.zeros((eta1s.size, eta2s.size, eta3s.size), dtype=float)

    comm.Barrier()
    sleep(0.02 * (rank + 1))
    print(f"rank {rank} | {eta1s = }")
    print(f"rank {rank} | {eta2s = }")
    print(f"rank {rank} | {eta3s = }\n")
    comm.Barrier()

    # compare spline evaluation routines
    t0 = time.time()
    evaluate_tensor_product(
        tn1[1:-1],
        tn2[1:-1],
        tn3[1:-1],
        derham.p[0] - 1,
        derham.p[1] - 1,
        derham.p[2] - 1,
        *derham.indD,
        x3[0],
        eta1s,
        eta2s,
        eta3s,
        vals,
        3,
    )
    t1 = time.time()
    if rank == 0:
        print("V3 evaluate_tensor_product:".ljust(40), t1 - t0)

    t0 = time.time()
    eval_spline_mpi_tensor_product_fixed(
        *spans_f,
        *bds_f,
        x3_psy._data,
        derham.spline_types_pyccel["3"],
        xp.array(derham.p),
        xp.array(x0_psy.starts),
        vals_mpi_fixed,
    )
    t1 = time.time()
    if rank == 0:
        print("v3 eval_spline_mpi_tensor_product_fixed:".ljust(40), t1 - t0)

    assert xp.allclose(vals, vals_mpi_fixed)

    field = derham.create_spline_function("test", "L2")
    field.vector = x3_psy

    assert xp.allclose(field.vector._data, x3_psy._data)

    t0 = time.time()
    field.eval_tp_fixed_loc(spans_f, bds_f, out=vals_mpi_fixed)
    t1 = time.time()
    if rank == 0:
        print("v3 field.eval_tp_fixed:".ljust(40), t1 - t0)

    assert xp.allclose(vals, vals_mpi_fixed)


if __name__ == "__main__":
    # test_eval_tensor_product([8, 9, 10], [2, 1, 2], [True, False, False], n_markers=10)
    test_eval_tensor_product_grid([8, 9, 10], [2, 1, 2], [False, True, False], n_markers=10)
