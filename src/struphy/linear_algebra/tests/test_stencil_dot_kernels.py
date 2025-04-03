import pytest


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [12])
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("spl_kind", [False, True])
@pytest.mark.parametrize("domain_ind", ["N", "D"])
@pytest.mark.parametrize("codomain_ind", ["N", "D"])
def test_1d(Nel, p, spl_kind, domain_ind, codomain_ind):
    """Compares the matrix-vector product obtained from the Stencil .dot method
    with

    a) the result from kernel in struphy.linear_algebra.stencil_dot_kernels.matvec_1d_kernel
    b) the result from Stencil .dot with precompiled=True"""

    import numpy as np
    from mpi4py import MPI
    from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
    from psydac.linalg.stencil import StencilMatrix, StencilVector

    from struphy.feec.psydac_derham import Derham
    from struphy.linear_algebra.stencil_dot_kernels import matvec_1d_kernel

    # only for M1 Mac users
    PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    if rank == 0:
        print("\nParameters:")
        print("Nel=", Nel)
        print("p=", p)
        print("spl_kind=", spl_kind)
        print("domain_ind=", domain_ind)
        print("codomain_ind=", codomain_ind)

    # Psydac discrete Derham sequence
    derham = Derham([Nel] * 3, [p] * 3, [spl_kind] * 3, comm=comm)
    V0 = derham.Vh["0"]

    V0_fem = derham.Vh_fem["0"]
    V3_fem = derham.Vh_fem["3"]

    # test 1d matvec
    spaces_1d = {}
    spaces_1d["N"] = V0_fem.spaces[0]
    spaces_1d["D"] = V3_fem.spaces[0]

    domain = spaces_1d[domain_ind]
    codomain = spaces_1d[codomain_ind]

    mat = StencilMatrix(domain.coeff_space, codomain.coeff_space)
    mat_pre = StencilMatrix(domain.coeff_space, codomain.coeff_space, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)
    x = StencilVector(domain.coeff_space)
    out_ker = StencilVector(codomain.coeff_space)

    s_out = int(mat.codomain.starts[0])
    e_out = int(mat.codomain.ends[0])
    p_out = int(mat.codomain.pads[0])
    s_in = int(mat.domain.starts[0])
    e_in = int(mat.domain.ends[0])
    p_in = int(mat.domain.pads[0])

    npts = codomain.coeff_space.npts[0]

    # matrix
    for i in range(s_out, e_out + 1):
        i_loc = i - s_out
        for d1 in range(2 * p_in + 1):
            m = i - p_in + d1  # global column index
            if spl_kind:
                mat._data[p_out + i_loc, d1] = m - i
                mat_pre._data[p_out + i_loc, d1] = m - i
            else:
                if m >= 0 and m < npts:
                    mat._data[p_out + i_loc, d1] = m - i
                    mat_pre._data[p_out + i_loc, d1] = m - i

    # random vector
    # np.random.seed(123)
    x[s_in : e_in + 1] = np.random.rand(domain.coeff_space.npts[0])

    if rank == 0:
        print(f"spl_kind={spl_kind}")
        print("\nx=", x._data)
        print("update ghost regions:")

    # very important: update vectors after changing _data !!
    x.update_ghost_regions()

    if rank == 0:
        print("x=", x._data)

    # stencil .dot
    out = mat.dot(x)

    # kernel matvec
    add = int(e_in >= e_out)
    matvec_1d_kernel(mat._data, x._data, out_ker._data, s_in, p_in, add, s_out, e_out, p_out)

    # precompiled .dot
    out_pre = mat_pre.dot(x)

    if rank == 0:
        print("domain degree:  ", domain.degree)
        print("codomain degree:", codomain.degree)
        print(f"rank {rank} | domain.starts = ", mat.domain.starts)
        print(f"rank {rank} | domain.ends = ", mat.domain.ends)
        print(f"rank {rank} | domain.pads = ", mat.domain.pads)
        print(f"rank {rank} | codomain.starts = ", mat.codomain.starts)
        print(f"rank {rank} | codomain.ends = ", mat.codomain.ends)
        print(f"rank {rank} | codomain.pads = ", mat.codomain.pads)
        print(f"rank {rank} | add = ", add)
        print("\nmat=", mat._data)
        print("\nmat.toarray=\n", mat.toarray())
        print("\nout=    ", out._data)
        print("\nout_ker=", out_ker._data)
        print("\nout_pre=", out_pre._data)

    assert np.allclose(out_ker._data, out._data)
    assert np.allclose(out_pre._data, out._data)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("Nel", [[12, 16, 20]])
@pytest.mark.parametrize("p", [[1, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[True, False, False]])
@pytest.mark.parametrize("domain_ind", ["NNN", "DNN", "NDN", "NND", "NDD", "DND", "DDN", "DDD"])
@pytest.mark.parametrize("codomain_ind", ["NNN", "DNN", "NDN", "NND", "NDD", "DND", "DDN", "DDD"])
def test_3d(Nel, p, spl_kind, domain_ind, codomain_ind):
    """Compares the matrix-vector product obtained from the Stencil .dot method
    with

    a) the result from kernel in struphy.linear_algebra.stencil_dot_kernels.matvec_1d_kernel
    b) the result from Stencil .dot with precompiled=True"""

    import numpy as np
    from mpi4py import MPI
    from psydac.api.settings import PSYDAC_BACKEND_GPYCCEL
    from psydac.linalg.stencil import StencilMatrix, StencilVector

    from struphy.feec.psydac_derham import Derham
    from struphy.linear_algebra.stencil_dot_kernels import matvec_3d_kernel

    # only for M1 Mac users
    PSYDAC_BACKEND_GPYCCEL["flags"] = "-O3 -march=native -mtune=native -ffast-math -ffree-line-length-none"

    comm = MPI.COMM_WORLD
    assert comm.size >= 2
    rank = comm.Get_rank()

    if rank == 0:
        print("\nParameters:")
        print("Nel=", Nel)
        print("p=", p)
        print("spl_kind=", spl_kind)
        print("domain_ind=", domain_ind)
        print("codomain_ind=", codomain_ind)

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    spaces_3d = {}
    spaces_3d["NNN"] = derham.Vh_fem["0"]
    spaces_3d["DNN"] = derham.Vh_fem["1"].spaces[0]
    spaces_3d["NDN"] = derham.Vh_fem["1"].spaces[1]
    spaces_3d["NND"] = derham.Vh_fem["1"].spaces[2]
    spaces_3d["NDD"] = derham.Vh_fem["2"].spaces[0]
    spaces_3d["DND"] = derham.Vh_fem["2"].spaces[1]
    spaces_3d["DDN"] = derham.Vh_fem["2"].spaces[2]
    spaces_3d["DDD"] = derham.Vh_fem["3"]

    domain = spaces_3d[domain_ind]
    codomain = spaces_3d[codomain_ind]

    mat = StencilMatrix(domain.coeff_space, codomain.coeff_space)
    mat_pre = StencilMatrix(domain.coeff_space, codomain.coeff_space, backend=PSYDAC_BACKEND_GPYCCEL, precompiled=True)
    x = StencilVector(domain.coeff_space)
    out_ker = StencilVector(codomain.coeff_space)

    s_out = np.array(mat.codomain.starts)
    e_out = np.array(mat.codomain.ends)
    p_out = np.array(mat.codomain.pads)
    s_in = np.array(mat.domain.starts)
    e_in = np.array(mat.domain.ends)
    p_in = np.array(mat.domain.pads)

    # random matrix
    np.random.seed(123)
    tmp1 = np.random.rand(*codomain.coeff_space.npts, *[2 * q + 1 for q in p])
    mat[
        s_out[0] : e_out[0] + 1,
        s_out[1] : e_out[1] + 1,
        s_out[2] : e_out[2] + 1,
    ] = tmp1[
        s_out[0] : e_out[0] + 1,
        s_out[1] : e_out[1] + 1,
        s_out[2] : e_out[2] + 1,
    ]
    mat_pre[
        s_out[0] : e_out[0] + 1,
        s_out[1] : e_out[1] + 1,
        s_out[2] : e_out[2] + 1,
    ] = tmp1[
        s_out[0] : e_out[0] + 1,
        s_out[1] : e_out[1] + 1,
        s_out[2] : e_out[2] + 1,
    ]

    # random vector
    tmp2 = np.random.rand(*domain.coeff_space.npts)
    x[
        s_in[0] : e_in[0] + 1,
        s_in[1] : e_in[1] + 1,
        s_in[2] : e_in[2] + 1,
    ] = tmp2[
        s_in[0] : e_in[0] + 1,
        s_in[1] : e_in[1] + 1,
        s_in[2] : e_in[2] + 1,
    ]

    # very important: update vectors after changing _data !!
    x.update_ghost_regions()

    # stencil .dot
    out = mat.dot(x)

    # kernel matvec
    add = [int(end_in >= end_out) for end_in, end_out in zip(mat.domain.ends, mat.codomain.ends)]
    add = np.array(add)
    matvec_3d_kernel(mat._data, x._data, out_ker._data, s_in, p_in, add, s_out, e_out, p_out)

    # precompiled .dot
    out_pre = mat_pre.dot(x)

    if rank == 0:
        print("domain degree:  ", domain.degree)
        print("codomain degree:", codomain.degree)
        print(f"rank {rank} | domain.starts = ", s_in)
        print(f"rank {rank} | domain.ends = ", e_in)
        print(f"rank {rank} | domain.pads = ", p_in)
        print(f"rank {rank} | codomain.starts = ", s_out)
        print(f"rank {rank} | codomain.ends = ", e_out)
        print(f"rank {rank} | codomain.pads = ", p_out)
        print(f"rank {rank} | add = ", add)
        print("\nmat=", mat._data[:, p_out[1], p_out[2], :, 0, 0])
        print("\nout[0]=    ", out._data[:, p_out[1], p_out[2]])
        print("\nout_ker[0]=", out_ker._data[:, p_out[1], p_out[2]])
        print("\nout_pre[0]=", out_pre._data[:, p_out[1], p_out[2]])
        print("\nout[1]=    ", out._data[p_out[0], :, p_out[2]])
        print("\nout_ker[1]=", out_ker._data[p_out[0], :, p_out[2]])
        print("\nout_pre[1]=", out_pre._data[p_out[0], :, p_out[2]])
        print("\nout[2]=    ", out._data[p_out[0], p_out[1], :])
        print("\nout_ker[2]=", out_ker._data[p_out[0], p_out[1], :])
        print("\nout_pre[2]=", out_pre._data[p_out[0], p_out[1], :])

    assert np.allclose(
        out_ker[s_out[0] : e_out[0] + 1, s_out[1] : e_out[1] + 1, s_out[2] : e_out[2] + 1],
        out[s_out[0] : e_out[0] + 1, s_out[1] : e_out[1] + 1, s_out[2] : e_out[2] + 1],
    )

    assert np.allclose(
        out_pre[s_out[0] : e_out[0] + 1, s_out[1] : e_out[1] + 1, s_out[2] : e_out[2] + 1],
        out[s_out[0] : e_out[0] + 1, s_out[1] : e_out[1] + 1, s_out[2] : e_out[2] + 1],
    )


if __name__ == "__main__":
    test_1d(10, 1, False, "N", "N")
    test_1d(10, 2, False, "N", "N")
    test_1d(10, 1, True, "N", "N")
    test_1d(10, 2, True, "N", "N")
    test_1d(10, 1, False, "D", "N")
    test_1d(10, 2, False, "D", "N")
    test_1d(10, 1, True, "D", "N")
    test_1d(10, 2, True, "D", "N")
    test_1d(10, 1, False, "N", "D")
    test_1d(10, 2, False, "N", "D")
    test_1d(10, 1, True, "N", "D")
    test_1d(10, 2, True, "N", "D")
    test_1d(10, 1, False, "D", "D")
    test_1d(10, 2, False, "D", "D")
    test_1d(10, 1, True, "D", "D")
    test_1d(10, 2, True, "D", "D")

    test_3d([12, 16, 20], [1, 2, 3], [False, True, True], "NNN", "DNN")
    test_3d([12, 16, 20], [1, 2, 3], [False, True, True], "NDN", "NND")
    test_3d([12, 16, 20], [1, 2, 3], [False, True, True], "NDD", "DND")
    test_3d([12, 16, 20], [1, 2, 3], [False, True, True], "DDN", "DDD")
