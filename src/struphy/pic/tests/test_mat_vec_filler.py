import cunumpy as xp
import pytest


@pytest.mark.parametrize("Nel", [[8, 9, 10]])
@pytest.mark.parametrize("p", [[1, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, False, True], [False, True, False], [True, False, False]])
def test_particle_to_mat_kernels(Nel, p, spl_kind, n_markers=1):
    """This test assumes a single particle and verifies
        a) if the correct indices are non-zero in _data
        b) if there are no NaNs
    for all routines in particle_to_mat_kernels.py
    """

    from time import sleep

    from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.stencil import StencilMatrix, StencilVector

    from struphy.bsplines import bsplines_kernels as bsp
    from struphy.feec.psydac_derham import Derham
    from struphy.pic.accumulation import particle_to_mat_kernels as ptomat

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Psydac discrete Derham sequence
    DR = Derham(Nel, p, spl_kind, comm=comm)

    if rank == 0:
        print(f"\nNel={Nel}, p={p}, spl_kind={spl_kind}\n")

    # DR attributes
    pn = xp.array(DR.p)
    tn1, tn2, tn3 = DR.Vh_fem["0"].knots

    starts1 = {}

    starts1["v0"] = xp.array(DR.Vh["0"].starts)

    comm.Barrier()
    sleep(0.02 * (rank + 1))
    print(f"rank {rank} | starts1['v0']: {starts1['v0']}")
    comm.Barrier()

    # basis identifiers
    basis = {}
    basis["v0"] = "NNN"
    basis["v1"] = ["DNN", "NDN", "NND"]
    basis["v2"] = ["NDD", "DND", "DDN"]
    basis["v3"] = "DDD"

    # only for M1 Mac users
    PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

    # _data of StencilMatrices/Vectors
    mat = {}
    vec = {}

    mat["v0"] = StencilMatrix(DR.Vh["0"], DR.Vh["0"], backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)._data
    vec["v0"] = StencilVector(DR.Vh["0"])._data

    mat["v3"] = StencilMatrix(DR.Vh["3"], DR.Vh["3"], backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)._data
    vec["v3"] = StencilVector(DR.Vh["3"])._data

    mat["v1"] = []
    for i in range(3):
        mat["v1"] += [[]]
        for j in range(3):
            mat["v1"][-1] += [
                StencilMatrix(
                    DR.Vh["1"].spaces[i],
                    DR.Vh["1"].spaces[j],
                    backend=PSYDAC_BACKEND_GPYCCEL,
                    precompiled=True,
                )._data,
            ]

    vec["v1"] = []
    for i in range(3):
        vec["v1"] += [StencilVector(DR.Vh["1"].spaces[i])._data]

    mat["v2"] = []
    for i in range(3):
        mat["v2"] += [[]]
        for j in range(3):
            mat["v2"][-1] += [
                StencilMatrix(
                    DR.Vh["2"].spaces[i],
                    DR.Vh["2"].spaces[j],
                    backend=PSYDAC_BACKEND_GPYCCEL,
                    precompiled=True,
                )._data,
            ]

    vec["v2"] = []
    for i in range(3):
        vec["v2"] += [StencilVector(DR.Vh["2"].spaces[i])._data]

    # Some filling for testing
    fill_mat = xp.reshape(xp.arange(9, dtype=float), (3, 3)) + 1.0
    fill_vec = xp.arange(3, dtype=float) + 1.0

    # Random points in domain of process (VERY IMPORTANT to be in the right domain, otherwise NON-TRACKED errors occur in filler_kernels !!)
    dom = DR.domain_array[rank]
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
        # TODO: understand "Argument must be native int" when passing "pn[0]" here instead of "DR.p[0]"
        span1 = bsp.find_span(tn1, DR.p[0], eta1)
        span2 = bsp.find_span(tn2, DR.p[1], eta2)
        span3 = bsp.find_span(tn3, DR.p[2], eta3)

        # non-zero spline values at eta
        bn1 = xp.empty(DR.p[0] + 1, dtype=float)
        bn2 = xp.empty(DR.p[1] + 1, dtype=float)
        bn3 = xp.empty(DR.p[2] + 1, dtype=float)

        bd1 = xp.empty(DR.p[0], dtype=float)
        bd2 = xp.empty(DR.p[1], dtype=float)
        bd3 = xp.empty(DR.p[2], dtype=float)

        bsp.b_d_splines_slim(tn1, DR.p[0], eta1, span1, bn1, bd1)
        bsp.b_d_splines_slim(tn2, DR.p[1], eta2, span2, bn2, bd2)
        bsp.b_d_splines_slim(tn3, DR.p[2], eta3, span3, bn3, bd3)

        # element index of the particle in each direction
        ie1 = span1 - pn[0]
        ie2 = span2 - pn[1]
        ie3 = span3 - pn[2]

        # global indices of non-vanishing B- and D-splines (no modulo)
        glob_n1 = xp.arange(ie1, ie1 + pn[0] + 1)
        glob_n2 = xp.arange(ie2, ie2 + pn[1] + 1)
        glob_n3 = xp.arange(ie3, ie3 + pn[2] + 1)

        glob_d1 = glob_n1[:-1]
        glob_d2 = glob_n2[:-1]
        glob_d3 = glob_n3[:-1]

        # local row indices in _data of non-vanishing B- and D-splines, as sets for comparison
        rows = [{}, {}, {}]
        rows[0]["N"] = set(glob_n1 - starts1["v0"][0] + pn[0])
        rows[1]["N"] = set(glob_n2 - starts1["v0"][1] + pn[1])
        rows[2]["N"] = set(glob_n3 - starts1["v0"][2] + pn[2])

        rows[0]["D"] = set(glob_d1 - starts1["v0"][0] + pn[0])
        rows[1]["D"] = set(glob_d2 - starts1["v0"][1] + pn[1])
        rows[2]["D"] = set(glob_d3 - starts1["v0"][2] + pn[2])

        comm.Barrier()
        sleep(0.02 * (rank + 1))
        print(f"rank {rank} | particles rows[0]['N']: {rows[0]['N']}, rows[0]['D'] {rows[0]['D']}")
        print(f"rank {rank} | particles rows[1]['N']: {rows[1]['N']}, rows[1]['D'] {rows[1]['D']}")
        print(f"rank {rank} | particles rows[2]['N']: {rows[2]['N']}, rows[2]['D'] {rows[2]['D']}")
        comm.Barrier()

        # local column indices in _data of non-vanishing B- and D-splines, as sets for comparison
        cols = [{}, {}, {}]
        for n in range(3):
            cols[n]["NN"] = set(xp.arange(2 * pn[n] + 1))
            cols[n]["ND"] = set(xp.arange(2 * pn[n]))
            cols[n]["DN"] = set(xp.arange(1, 2 * pn[n] + 1))
            cols[n]["DD"] = set(xp.arange(1, 2 * pn[n]))

        # testing vector-valued spaces
        spaces_vector = ["v1", "v2"]
        symmetries = {
            "diag": [[0, 0], [1, 1], [2, 2]],  # index pairs of block matrix
            "asym": [[0, 1], [0, 2], [1, 2]],
            "symm": [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]],
            "full": [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]],
        }
        mvs = ["mat", "m_v"]

        count = 0
        for space in spaces_vector:
            for symmetry, ind_pairs in symmetries.items():
                args = []
                for ij in ind_pairs:
                    # list of matrix _data arguments for the filler
                    args += [mat[space][ij[0]][ij[1]]]
                    args[-1][:, :] = 0.0  # make sure entries are zero
                for ij in ind_pairs:
                    # list of matrix fillings for the filler
                    args += [fill_mat[ij[0], ij[1]]]

                for mv in mvs:
                    name_b = mv + "_fill_b_" + space + "_" + symmetry
                    name = mv + "_fill_" + space + "_" + symmetry

                    fun_b = getattr(ptomat, name_b)
                    fun = getattr(ptomat, name)

                    # add further arguments if vector needs to be filled
                    if mv == "m_v":
                        for i in range(3):
                            args += [vec[space][i]]
                            args[-1][:] = 0.0  # make sure entries are zero
                        for i in range(3):
                            args += [fill_vec[i]]

                    # test with basis evaluation (_b)
                    if rank == 0:
                        print(f"\nTesting {name_b} ...")

                    fun_b(DR.args_derham, eta1, eta2, eta3, *args)

                    for n, ij in enumerate(ind_pairs):
                        assert_mat(
                            args[n],
                            rows,
                            cols,
                            basis[space][ij[0]],
                            basis[space][ij[1]],
                            rank,
                            verbose=False,
                        )  # assertion test of mat
                    if mv == "m_v":
                        for i in range(3):
                            # assertion test of vec
                            assert_vec(args[-6 + i], rows, basis[space][i], rank)

                    count += 1

                    # test without basis evaluation
                    if rank == 0:
                        print(f"\nTesting {name} ...")

                    fun(DR.args_derham, span1, span2, span3, *args)

                    for n, ij in enumerate(ind_pairs):
                        assert_mat(
                            args[n],
                            rows,
                            cols,
                            basis[space][ij[0]],
                            basis[space][ij[1]],
                            rank,
                            verbose=False,
                        )  # assertion test of mat
                    if mv == "m_v":
                        for i in range(3):
                            # assertion test of vec
                            assert_vec(args[-6 + i], rows, basis[space][i], rank)

                    count += 1

                    comm.Barrier()

        # testing salar spaces
        if rank == 0:
            print("\nTesting mat_fill_b_v0 ...")
        ptomat.mat_fill_b_v0(DR.args_derham, eta1, eta2, eta3, mat["v0"], fill_mat[0, 0])
        assert_mat(mat["v0"], rows, cols, basis["v0"], basis["v0"], rank)  # assertion test of mat
        count += 1
        comm.Barrier()

        if rank == 0:
            print("\nTesting m_v_fill_b_v0 ...")
        ptomat.m_v_fill_b_v0(DR.args_derham, eta1, eta2, eta3, mat["v0"], fill_mat[0, 0], vec["v0"], fill_vec[0])
        assert_mat(mat["v0"], rows, cols, basis["v0"], basis["v0"], rank)  # assertion test of mat
        assert_vec(vec["v0"], rows, basis["v0"], rank)  # assertion test of vec
        count += 1
        comm.Barrier()

        if rank == 0:
            print("\nTesting mat_fill_b_v3 ...")
        ptomat.mat_fill_b_v3(DR.args_derham, eta1, eta2, eta3, mat["v3"], fill_mat[0, 0])
        assert_mat(mat["v3"], rows, cols, basis["v3"], basis["v3"], rank)  # assertion test of mat
        count += 1
        comm.Barrier()

        if rank == 0:
            print("\nTesting m_v_fill_b_v3 ...")
        ptomat.m_v_fill_b_v3(DR.args_derham, eta1, eta2, eta3, mat["v3"], fill_mat[0, 0], vec["v3"], fill_vec[0])
        assert_mat(mat["v3"], rows, cols, basis["v3"], basis["v3"], rank)  # assertion test of mat
        assert_vec(vec["v3"], rows, basis["v3"], rank)  # assertion test of vec
        count += 1
        comm.Barrier()

        if rank == 0:
            print("\nTesting mat_fill_v0 ...")
        ptomat.mat_fill_v0(DR.args_derham, span1, span2, span3, mat["v0"], fill_mat[0, 0])
        assert_mat(mat["v0"], rows, cols, basis["v0"], basis["v0"], rank)  # assertion test of mat
        count += 1
        comm.Barrier()

        if rank == 0:
            print("\nTesting m_v_fill_v0 ...")
        ptomat.m_v_fill_v0(DR.args_derham, span1, span2, span3, mat["v0"], fill_mat[0, 0], vec["v0"], fill_vec[0])
        assert_mat(mat["v0"], rows, cols, basis["v0"], basis["v0"], rank)  # assertion test of mat
        assert_vec(vec["v0"], rows, basis["v0"], rank)  # assertion test of vec
        count += 1
        comm.Barrier()

        if rank == 0:
            print("\nTesting mat_fill_v3 ...")
        ptomat.mat_fill_v3(DR.args_derham, span1, span2, span3, mat["v3"], fill_mat[0, 0])
        assert_mat(mat["v3"], rows, cols, basis["v3"], basis["v3"], rank)  # assertion test of mat
        count += 1
        comm.Barrier()

        if rank == 0:
            print("\nTesting m_v_fill_v3 ...")
        ptomat.m_v_fill_v3(DR.args_derham, span1, span2, span3, mat["v3"], fill_mat[0, 0], vec["v3"], fill_vec[0])
        assert_mat(mat["v3"], rows, cols, basis["v3"], basis["v3"], rank)  # assertion test of mat
        assert_vec(vec["v3"], rows, basis["v3"], rank)  # assertion test of vec
        count += 1
        comm.Barrier()

        if rank == 0:
            print(f"\n{count}/40 particle_to_mat_kernels routines tested.")


def assert_mat(mat, rows, cols, row_str, col_str, rank, verbose=False):
    """Check whether the non-zero values in mat are at the indices specified by rows and cols.
    Sets mat to zero after assertion is passed.

    Parameters
    ----------
        mat : array[float]
            6d array, the _data attribute of a StencilMatrix.

        rows : list[dict]
            3-list, each dict has the two keys "N" and "D", holding a set of row indices of p + 1 resp. p non-zero splines.

        cols : list[dict]
            3-list, each dict has four keys "NN", "ND", "DN" or "DD", holding the column indices of non-zero _data entries
            depending on the combination of basis functions in each direction.

        row_str : str
            String of length 3 specifying the codomain of mat, e.g. "DNN" for the first component of V1.

        col_str : str
            String of length 3 specifying the domain of mat, e.g. "DNN" for the first component of V1.

        rank : int
            Mpi rank of process.

        verbose : bool
            Show additional screen output.
    """
    assert len(mat.shape) == 6
    # assert non NaN
    assert ~xp.isnan(mat).any()

    atol = 1e-14

    if verbose:
        print(f"\n({row_str}) ({col_str})")
        print(f"rank {rank} | ind_row1: {set(xp.where(mat > atol)[0])}")
        print(f"rank {rank} | ind_row2: {set(xp.where(mat > atol)[1])}")
        print(f"rank {rank} | ind_row3: {set(xp.where(mat > atol)[2])}")
        print(f"rank {rank} | ind_col1: {set(xp.where(mat > atol)[3])}")
        print(f"rank {rank} | ind_col2: {set(xp.where(mat > atol)[4])}")
        print(f"rank {rank} | ind_col3: {set(xp.where(mat > atol)[5])}")

    # check if correct indices are non-zero
    for n, (r, c) in enumerate(zip(row_str, col_str)):
        assert set(xp.where(mat > atol)[n]) == rows[n][r]
        assert set(xp.where(mat > atol)[n + 3]) == cols[n][r + c]

    # Set matrix back to zero
    mat[:, :] = 0.0

    print(f"rank {rank} | Matrix index assertion passed for ({row_str}) ({col_str}).")


def assert_vec(vec, rows, row_str, rank, verbose=False):
    """Check whether the non-zero values in vec are at the indices specified by rows.
    Sets vec to zero after assertion is passed.

    Parameters
    ----------
        vec : array[float]
            3d array, the _data attribute of a StencilVector.

        rows : list[dict]
            3-list, each dict has the two keys "N" and "D", holding a set of row indices of p + 1 resp. p non-zero splines.

        row_str : str
            String of length 3 specifying the codomain of mat, e.g. "DNN" for the first component of V1.

        rank : int
            Mpi rank of process.

        verbose : bool
            Show additional screen output.
    """
    assert len(vec.shape) == 3
    # assert non Nan
    assert ~xp.isnan(vec).any()

    atol = 1e-14

    if verbose:
        print(f"\n({row_str})")
        print(f"rank {rank} | ind_row1: {set(xp.where(vec > atol)[0])}")
        print(f"rank {rank} | ind_row2: {set(xp.where(vec > atol)[1])}")
        print(f"rank {rank} | ind_row3: {set(xp.where(vec > atol)[2])}")

    # check if correct indices are non-zero
    for n, r in enumerate(row_str):
        assert set(xp.where(vec > atol)[n]) == rows[n][r]

    # Set vector back to zero
    vec[:] = 0.0

    print(f"rank {rank} | Vector index assertion passed for ({row_str}).")


if __name__ == "__main__":
    test_particle_to_mat_kernels([8, 9, 10], [2, 3, 4], [True, False, False], n_markers=1)
