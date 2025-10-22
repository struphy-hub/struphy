import pytest


@pytest.mark.parametrize("Nel", [[32, 1, 1], [1, 32, 1], [1, 1, 32], [31, 32, 1], [32, 1, 31], [1, 31, 32]])
@pytest.mark.parametrize("p", [[1, 1, 1]])
@pytest.mark.parametrize("spl_kind", [[True, True, True]])
def test_lowdim_derham(Nel, p, spl_kind, do_plot=False):
    """Test Nel=1 in various directions."""

    import cunumpy as xp
    from matplotlib import pyplot as plt
    from psydac.ddm.mpi import mpi as MPI
    from psydac.linalg.block import BlockVector
    from psydac.linalg.stencil import StencilVector

    from struphy.feec.psydac_derham import Derham

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Nel=", Nel)
    print("p=", p)
    print("spl_kind=", spl_kind)

    # Psydac discrete Derham sequence
    derham = Derham(Nel, p, spl_kind, comm=comm)

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

    x3_PSY = StencilVector(derham.Vh["3"])
    print(f"rank {rank} | \n3-form StencilVector:")
    print(f"rank {rank} | starts:", x3_PSY.starts)
    print(f"rank {rank} | ends  :", x3_PSY.ends)
    print(f"rank {rank} | pads  :", x3_PSY.pads)
    print(f"rank {rank} | shape (=dim):", x3_PSY.shape)
    print(f"rank {rank} | [:].shape (=shape):", x3_PSY[:].shape)

    # Block of StencilVecttors
    x1_PSY = BlockVector(derham.Vh["1"])
    print(f"rank {rank} | \n1-form StencilVector:")
    print(f"rank {rank} | starts:", [component.starts for component in x1_PSY])
    print(f"rank {rank} | ends  :", [component.ends for component in x1_PSY])
    print(f"rank {rank} | pads  :", [component.pads for component in x1_PSY])
    print(f"rank {rank} | shape (=dim):", [component.shape for component in x1_PSY])
    print(f"rank {rank} | [:].shape (=shape):", [component[:].shape for component in x1_PSY])

    x2_PSY = BlockVector(derham.Vh["2"])
    print(f"rank {rank} | \n2-form StencilVector:")
    print(f"rank {rank} | starts:", [component.starts for component in x2_PSY])
    print(f"rank {rank} | ends  :", [component.ends for component in x2_PSY])
    print(f"rank {rank} | pads  :", [component.pads for component in x2_PSY])
    print(f"rank {rank} | shape (=dim):", [component.shape for component in x2_PSY])
    print(f"rank {rank} | [:].shape (=shape):", [component[:].shape for component in x2_PSY])

    xv_PSY = BlockVector(derham.Vh["v"])
    print(f"rank {rank} | \nVector StencilVector:")
    print(f"rank {rank} | starts:", [component.starts for component in xv_PSY])
    print(f"rank {rank} | ends  :", [component.ends for component in xv_PSY])
    print(f"rank {rank} | pads  :", [component.pads for component in xv_PSY])
    print(f"rank {rank} | shape (=dim):", [component.shape for component in xv_PSY])
    print(f"rank {rank} | [:].shape (=shape):", [component[:].shape for component in xv_PSY])

    #################################
    ### TEST COMMUTING PROJECTORS ###
    #################################
    def fun(eta):
        return xp.cos(2 * xp.pi * eta)

    def dfun(eta):
        return -2 * xp.pi * xp.sin(2 * xp.pi * eta)

    # evaluation points and gradient
    e1 = 0.0
    e2 = 0.0
    e3 = 0.0
    if Nel[0] > 1:
        e1 = xp.linspace(0.0, 1.0, 100)
        e = e1
        c = 0

        def f(x, y, z):
            return fun(x)

        def dfx(x, y, z):
            return dfun(x)

        def dfy(x, y, z):
            return xp.zeros_like(x)

        def dfz(x, y, z):
            return xp.zeros_like(x)
    elif Nel[1] > 1:
        e2 = xp.linspace(0.0, 1.0, 100)
        e = e2
        c = 1

        def f(x, y, z):
            return fun(y)

        def dfx(x, y, z):
            return xp.zeros_like(y)

        def dfy(x, y, z):
            return dfun(y)

        def dfz(x, y, z):
            return xp.zeros_like(y)
    elif Nel[2] > 1:
        e3 = xp.linspace(0.0, 1.0, 100)
        e = e3
        c = 2

        def f(x, y, z):
            return fun(z)

        def dfx(x, y, z):
            return xp.zeros_like(z)

        def dfy(x, y, z):
            return xp.zeros_like(z)

        def dfz(x, y, z):
            return dfun(z)

    def curl_f_1(x, y, z):
        return dfy(x, y, z) - dfz(x, y, z)

    def curl_f_2(x, y, z):
        return dfz(x, y, z) - dfx(x, y, z)

    def curl_f_3(x, y, z):
        return dfx(x, y, z) - dfy(x, y, z)

    def div_f(x, y, z):
        return dfx(x, y, z) + dfy(x, y, z) + dfz(x, y, z)

    grad_f = (dfx, dfy, dfz)
    curl_f = (curl_f_1, curl_f_2, curl_f_3)
    proj_of_grad_f = derham.P["1"](grad_f)
    proj_of_curl_fff = derham.P["2"](curl_f)
    proj_of_div_fff = derham.P["3"](div_f)

    ##########
    # 0-form #
    ##########
    f0_h = derham.P["0"](f)

    field_f0 = derham.create_spline_function("f0", "H1")
    field_f0.vector = f0_h
    field_f0_vals = field_f0(e1, e2, e3, squeeze_out=True)

    # a) projection error
    err_f0 = xp.max(xp.abs(f(e1, e2, e3) - field_f0_vals))
    print(f"\n{err_f0 = }")
    assert err_f0 < 1e-2

    # b) commuting property
    df0_h = derham.grad.dot(f0_h)
    assert xp.allclose(df0_h.toarray(), proj_of_grad_f.toarray())

    # c) derivative error
    field_df0 = derham.create_spline_function("df0", "Hcurl")
    field_df0.vector = df0_h
    field_df0_vals = field_df0(e1, e2, e3, squeeze_out=True)

    err_df0 = [xp.max(xp.abs(exact(e1, e2, e3) - field_v)) for exact, field_v in zip(grad_f, field_df0_vals)]
    print(f"{err_df0 = }")
    assert xp.max(err_df0) < 0.64

    # d) plotting
    plt.figure(figsize=(8, 12))
    plt.subplot(2, 1, 1)
    plt.plot(e, f(e1, e2, e3), "o")
    plt.plot(e, field_f0_vals)
    plt.title("fun")
    plt.xlabel(f"eta{c + 1}")

    plt.subplot(2, 1, 2)
    plt.plot(e, grad_f[c](e1, e2, e3), "o")
    plt.plot(e, field_df0_vals[c])
    plt.title(f"grad comp {c + 1}")

    plt.subplots_adjust(wspace=1.0, hspace=0.4)

    ##########
    # 1-form #
    ##########
    f1_h = derham.P["1"]((f, f, f))

    field_f1 = derham.create_spline_function("f1", "Hcurl")
    field_f1.vector = f1_h
    field_f1_vals = field_f1(e1, e2, e3, squeeze_out=True)

    # a) projection error
    err_f1 = [xp.max(xp.abs(exact(e1, e2, e3) - field_v)) for exact, field_v in zip([f, f, f], field_f1_vals)]
    print(f"{err_f1 = }")
    assert xp.max(err_f1) < 0.09

    # b) commuting property
    df1_h = derham.curl.dot(f1_h)
    assert xp.allclose(df1_h.toarray(), proj_of_curl_fff.toarray())

    # c) derivative error
    field_df1 = derham.create_spline_function("df1", "Hdiv")
    field_df1.vector = df1_h
    field_df1_vals = field_df1(e1, e2, e3, squeeze_out=True)

    err_df1 = [xp.max(xp.abs(exact(e1, e2, e3) - field_v)) for exact, field_v in zip(curl_f, field_df1_vals)]
    print(f"{err_df1 = }")
    assert xp.max(err_df1) < 0.64

    # d) plotting
    plt.figure(figsize=(8, 12))
    plt.subplot(3, 1, 1)
    plt.plot(e, f(e1, e2, e3), "o")
    plt.plot(e, field_f1_vals[c])
    plt.title("all components fun")
    plt.xlabel(f"eta{c + 1}")

    plt.subplot(3, 1, 2)
    plt.plot(e, curl_f[(c + 1) % 3](e1, e2, e3), "o")
    plt.plot(e, field_df1_vals[(c + 1) % 3])
    plt.title(f"curl comp {(c + 1) % 3}")

    plt.subplot(3, 1, 3)
    plt.plot(e, curl_f[(c + 2) % 3](e1, e2, e3), "o")
    plt.plot(e, field_df1_vals[(c + 2) % 3])
    plt.title(f"curl comp {(c + 2) % 3}")

    plt.subplots_adjust(wspace=1.0, hspace=0.4)

    ##########
    # 2-form #
    ##########
    f2_h = derham.P["2"]((f, f, f))

    field_f2 = derham.create_spline_function("f2", "Hdiv")
    field_f2.vector = f2_h
    field_f2_vals = field_f2(e1, e2, e3, squeeze_out=True)

    # a) projection error
    err_f2 = [xp.max(xp.abs(exact(e1, e2, e3) - field_v)) for exact, field_v in zip([f, f, f], field_f2_vals)]
    print(f"{err_f2 = }")
    assert xp.max(err_f2) < 0.09

    # b) commuting property
    df2_h = derham.div.dot(f2_h)
    assert xp.allclose(df2_h.toarray(), proj_of_div_fff.toarray())

    # c) derivative error
    field_df2 = derham.create_spline_function("df2", "L2")
    field_df2.vector = df2_h
    field_df2_vals = field_df2(e1, e2, e3, squeeze_out=True)

    err_df2 = xp.max(xp.abs(div_f(e1, e2, e3) - field_df2_vals))
    print(f"{err_df2 = }")
    assert xp.max(err_df2) < 0.64

    # d) plotting
    plt.figure(figsize=(8, 12))
    plt.subplot(2, 1, 1)
    plt.plot(e, f(e1, e2, e3), "o")
    plt.plot(e, field_f2_vals[c])
    plt.title("all components fun")
    plt.xlabel(f"eta{c + 1}")

    plt.subplot(2, 1, 2)
    plt.plot(e, div_f(e1, e2, e3), "o")
    plt.plot(e, field_df2_vals)
    plt.title(f"div")

    plt.subplots_adjust(wspace=1.0, hspace=0.4)

    ##########
    # 3-form #
    ##########
    f3_h = derham.P["3"](f)

    field_f3 = derham.create_spline_function("f3", "L2")
    field_f3.vector = f3_h
    field_f3_vals = field_f3(e1, e2, e3, squeeze_out=True)

    # a) projection error
    err_f3 = xp.max(xp.abs(f(e1, e2, e3) - field_f3_vals))
    print(f"{err_f3 = }")
    assert err_f3 < 0.09

    # d) plotting
    plt.figure(figsize=(8, 12))
    plt.subplot(2, 1, 1)
    plt.plot(e, f(e1, e2, e3), "o")
    plt.plot(e, field_f3_vals)
    plt.title("fun")
    plt.xlabel(f"eta{c + 1}")

    plt.subplots_adjust(wspace=1.0, hspace=0.4)

    if do_plot:
        plt.show()


if __name__ == "__main__":
    test_lowdim_derham([32, 1, 1], [1, 1, 1], [True, True, True], do_plot=False)
    test_lowdim_derham([1, 32, 1], [1, 1, 1], [True, True, True], do_plot=False)
    test_lowdim_derham([1, 1, 32], [1, 1, 1], [True, True, True], do_plot=False)
