import pytest


@pytest.mark.parametrize("Nel", [[8, 8, 12]])
@pytest.mark.parametrize("p", [[1, 2, 3]])
@pytest.mark.parametrize("spl_kind", [[False, False, True]])
def test_psydac_derham(Nel, p, spl_kind):
    """Remark: p=even projectors yield slightly different results, pass with atol=1e-3."""

    import cunumpy as xp
    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.block import BlockVector
    from psydac.linalg.stencil import StencilVector

    
    from struphy.feec.psydac_derham import Derham
    from struphy.feec.utilities import compare_arrays

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Nel=", Nel)
    print("p=", p)
    print("spl_kind=", spl_kind)

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

    # Struphy Derham (deprecated)
    nq_el = [4, 4, 4]
    spaces = [
        Spline_space_1d(Nel_i, p_i, spl_kind_i, nq_el_i)
        for Nel_i, p_i, spl_kind_i, nq_el_i in zip(Nel, p, spl_kind, nq_el)
    ]

    spaces[0].set_projectors(p[0] + 1)
    spaces[1].set_projectors(p[1] + 1)
    spaces[2].set_projectors(p[2] + 1)

    DR_STR = Tensor_spline_space(spaces)
    DR_STR.set_projectors("tensor")

    # Space dimensions
    N0_tot = DR_STR.Ntot_0form
    N1_tot = DR_STR.Ntot_1form
    N2_tot = DR_STR.Ntot_2form
    N3_tot = DR_STR.Ntot_3form

    # Random vectors for testing
    xp.random.seed(1981)
    x0 = xp.random.rand(N0_tot)
    x1 = xp.random.rand(xp.sum(N1_tot))
    x2 = xp.random.rand(xp.sum(N2_tot))
    x3 = xp.random.rand(N3_tot)

    ############################
    ### TEST STENCIL VECTORS ###
    ############################
    # Stencil vectors for Psydac:
    x0_PSY = StencilVector(derham.Vh["0"])
    print(f"rank {rank} | 0-form StencilVector:")
    print(f"rank {rank} | starts:", x0_PSY.starts)
    print(f"rank {rank} | ends  :", x0_PSY.ends)
    print(f"rank {rank} | pads  :", x0_PSY.pads)
    print(f"rank {rank} | shape (=dim):", x0_PSY.shape)
    print(f"rank {rank} | [:].shape (=shape):", x0_PSY[:].shape)

    s0 = x0_PSY.starts
    e0 = x0_PSY.ends

    # Assign from start to end index + 1
    x0_PSY[s0[0] : e0[0] + 1, s0[1] : e0[1] + 1, s0[2] : e0[2] + 1] = DR_STR.extract_0(x0)[
        s0[0] : e0[0] + 1,
        s0[1] : e0[1] + 1,
        s0[2] : e0[2] + 1,
    ]

    # Block of StencilVecttors
    x1_PSY = BlockVector(derham.Vh["1"])
    print(f"rank {rank} | \n1-form StencilVector:")
    print(f"rank {rank} | starts:", [component.starts for component in x1_PSY])
    print(f"rank {rank} | ends  :", [component.ends for component in x1_PSY])
    print(f"rank {rank} | pads  :", [component.pads for component in x1_PSY])
    print(f"rank {rank} | shape (=dim):", [component.shape for component in x1_PSY])
    print(f"rank {rank} | [:].shape (=shape):", [component[:].shape for component in x1_PSY])

    s11, s12, s13 = [component.starts for component in x1_PSY]
    e11, e12, e13 = [component.ends for component in x1_PSY]

    x11, x12, x13 = DR_STR.extract_1(x1)
    x1_PSY[0][s11[0] : e11[0] + 1, s11[1] : e11[1] + 1, s11[2] : e11[2] + 1] = x11[
        s11[0] : e11[0] + 1,
        s11[1] : e11[1] + 1,
        s11[2] : e11[2] + 1,
    ]
    x1_PSY[1][s12[0] : e12[0] + 1, s12[1] : e12[1] + 1, s12[2] : e12[2] + 1] = x12[
        s12[0] : e12[0] + 1,
        s12[1] : e12[1] + 1,
        s12[2] : e12[2] + 1,
    ]
    x1_PSY[2][s13[0] : e13[0] + 1, s13[1] : e13[1] + 1, s13[2] : e13[2] + 1] = x13[
        s13[0] : e13[0] + 1,
        s13[1] : e13[1] + 1,
        s13[2] : e13[2] + 1,
    ]

    x2_PSY = BlockVector(derham.Vh["2"])
    print(f"rank {rank} | \n2-form StencilVector:")
    print(f"rank {rank} | starts:", [component.starts for component in x2_PSY])
    print(f"rank {rank} | ends  :", [component.ends for component in x2_PSY])
    print(f"rank {rank} | pads  :", [component.pads for component in x2_PSY])
    print(f"rank {rank} | shape (=dim):", [component.shape for component in x2_PSY])
    print(f"rank {rank} | [:].shape (=shape):", [component[:].shape for component in x2_PSY])

    s21, s22, s23 = [component.starts for component in x2_PSY]
    e21, e22, e23 = [component.ends for component in x2_PSY]

    x21, x22, x23 = DR_STR.extract_2(x2)
    x2_PSY[0][s21[0] : e21[0] + 1, s21[1] : e21[1] + 1, s21[2] : e21[2] + 1] = x21[
        s21[0] : e21[0] + 1,
        s21[1] : e21[1] + 1,
        s21[2] : e21[2] + 1,
    ]
    x2_PSY[1][s22[0] : e22[0] + 1, s22[1] : e22[1] + 1, s22[2] : e22[2] + 1] = x22[
        s22[0] : e22[0] + 1,
        s22[1] : e22[1] + 1,
        s22[2] : e22[2] + 1,
    ]
    x2_PSY[2][s23[0] : e23[0] + 1, s23[1] : e23[1] + 1, s23[2] : e23[2] + 1] = x23[
        s23[0] : e23[0] + 1,
        s23[1] : e23[1] + 1,
        s23[2] : e23[2] + 1,
    ]

    x3_PSY = StencilVector(derham.Vh["3"])
    print(f"rank {rank} | \n3-form StencilVector:")
    print(f"rank {rank} | starts:", x3_PSY.starts)
    print(f"rank {rank} | ends  :", x3_PSY.ends)
    print(f"rank {rank} | pads  :", x3_PSY.pads)
    print(f"rank {rank} | shape (=dim):", x3_PSY.shape)
    print(f"rank {rank} | [:].shape (=shape):", x3_PSY[:].shape)

    s3 = x3_PSY.starts
    e3 = x3_PSY.ends

    x3_PSY[s3[0] : e3[0] + 1, s3[1] : e3[1] + 1, s3[2] : e3[2] + 1] = DR_STR.extract_3(x3)[
        s3[0] : e3[0] + 1,
        s3[1] : e3[1] + 1,
        s3[2] : e3[2] + 1,
    ]

    ########################
    ### TEST DERIVATIVES ###
    ########################
    # Struphy derivative operators
    grad_STR = DR_STR.G0
    curl_STR = DR_STR.C0
    div_STR = DR_STR.D0

    if rank == 0:
        print("\nStruphy derivatives operators type:")
        print(type(grad_STR), type(curl_STR), type(div_STR))

        print("\nPsydac derivatives operators type:")
        print(type(derham.grad), type(derham.curl), type(derham.div))

    # compare derivatives
    d1_STR = grad_STR.dot(x0)
    d1_PSY = derham.grad.dot(x0_PSY)

    d2_STR = curl_STR.dot(x1)
    d2_PSY = derham.curl.dot(x1_PSY)

    d3_STR = div_STR.dot(x2)
    d3_PSY = derham.div.dot(x2_PSY)

    if rank == 0:
        print("\nCompare grad:")
    compare_arrays(d1_PSY, DR_STR.extract_1(d1_STR), rank)
    comm.Barrier()
    if rank == 0:
        print("\nCompare curl:")
    compare_arrays(d2_PSY, DR_STR.extract_2(d2_STR), rank)
    comm.Barrier()
    if rank == 0:
        print("\nCompare div:")
    compare_arrays(d3_PSY, DR_STR.extract_3(d3_STR), rank)
    comm.Barrier()

    zero2_STR = curl_STR.dot(d1_STR)
    zero2_PSY = derham.curl.dot(d1_PSY)

    assert xp.allclose(zero2_STR, xp.zeros_like(zero2_STR))
    if rank == 0:
        print("\nCompare curl of grad:")
    compare_arrays(zero2_PSY, DR_STR.extract_2(zero2_STR), rank)
    comm.Barrier()

    zero3_STR = div_STR.dot(d2_STR)
    zero3_PSY = derham.div.dot(d2_PSY)

    assert xp.allclose(zero3_STR, xp.zeros_like(zero3_STR))
    if rank == 0:
        print("\nCompare div of curl:")
    compare_arrays(zero3_PSY, DR_STR.extract_3(zero3_STR), rank)
    comm.Barrier()

    #######################
    ### TEST PROJECTORS ###
    #######################
    # Struphy projectors
    DR_STR.set_projectors()
    PI = DR_STR.projectors.PI  # callable as input
    PI_mat = DR_STR.projectors.PI_mat  # dofs as input (as 3d array)
    print("\nStruphy projectors type:")
    print(type(PI), type(PI_mat))

    # compare projectors
    def f(eta1, eta2, eta3):
        return xp.sin(4 * xp.pi * eta1) * xp.cos(2 * xp.pi * eta2) + xp.exp(xp.cos(2 * xp.pi * eta3))

    fh0_STR = PI("0", f)
    fh0_PSY = derham.P["0"](f)

    if rank == 0:
        print("\nCompare P0:")
    compare_arrays(fh0_PSY, fh0_STR, rank)
    comm.Barrier()

    fh11_STR = PI("11", f)
    fh12_STR = PI("12", f)
    fh13_STR = PI("13", f)
    fh1_STR = (fh11_STR, fh12_STR, fh13_STR)
    fh1_PSY = derham.P["1"]((f, f, f))

    if rank == 0:
        print("\nCompare P1:")
    compare_arrays(fh1_PSY, fh1_STR, rank, atol=1e-5)
    comm.Barrier()

    fh21_STR = PI("21", f)
    fh22_STR = PI("22", f)
    fh23_STR = PI("23", f)
    fh2_STR = (fh21_STR, fh22_STR, fh23_STR)
    fh2_PSY = derham.P["2"]((f, f, f))

    if rank == 0:
        print("\nCompare P2:")
    compare_arrays(fh2_PSY, fh2_STR, rank, atol=1e-5)
    comm.Barrier()

    fh3_STR = PI("3", f)
    fh3_PSY = derham.P["3"](f)

    if rank == 0:
        print("\nCompare P3:")
    compare_arrays(fh3_PSY, fh3_STR, rank, atol=1e-5)
    comm.Barrier()


if __name__ == "__main__":
    test_psydac_derham([8, 8, 12], [1, 2, 3], [False, False, True])
