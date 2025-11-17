def test_polar_splines_2D(plot=False):
    """
    TODO
    """

    import sys

    sys.path.append("..")

    import cunumpy as xp
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    
    from struphy.geometry import domains

    # parameters
    # number of elements (number of elements in angular direction must be a multiple of 3)
    Nel = [1, 24]
    p = [3, 3]  # splines degrees
    # kind of splines (for polar domains always [False, True] which means [clamped, periodic])
    spl_kind = [False, True]
    # number of quadrature points per element for integrations
    nq_el = [6, 6]
    # boundary conditions in radial direction (for polar domain always 'f' at eta1 = 0 (pole))
    bc = ["f", "d"]
    # minor radius
    a = 1.0
    # major radius (length or cylinder = 2*pi*R0 in case of spline_cyl)
    R0 = 3.0
    # meaning of angular coordinate in case of spline_tours ('straight' or 'equal arc')
    chi = "equal arc"

    # create domain
    dom_type = "IGAPolarCylinder"
    dom_params = {"a": a, "Lz": R0, "Nel": Nel, "p": p}
    domain_class = getattr(domains, dom_type)
    domain = domain_class(**dom_params)

    # plot the control points and the grid
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    el_b_1 = xp.linspace(0.0, 1.0, Nel[0] + 1)
    el_b_2 = xp.linspace(0.0, 1.0, Nel[1] + 1)

    grid_x = domain(el_b_1, el_b_2, 0.0, squeeze_out=True)[0]
    grid_y = domain(el_b_1, el_b_2, 0.0, squeeze_out=True)[1]

    for i in range(el_b_1.size):
        plt.plot(grid_x[i, :], grid_y[i, :], "k", linewidth=0.5)

    for j in range(el_b_2.size):
        plt.plot(grid_x[:, j], grid_y[:, j], "r", linewidth=0.5)

    plt.scatter(domain.cx[:, :, 0].flatten(), domain.cy[:, :, 0].flatten(), s=2, color="b")

    plt.axis("square")
    plt.xlabel("R [m]")
    plt.ylabel("y [m]")

    plt.title("Control points and grid for Nel = " + str(Nel) + " and p = " + str(p), pad=10)

    if plot:
        plt.show()

    # set up 1D spline spaces in radial and angular direction and 2D tensor-product space
    space_1d_1 = Spline_space_1d(Nel[0], p[0], spl_kind[0], nq_el[0], bc)
    space_1d_2 = Spline_space_1d(Nel[1], p[1], spl_kind[1], nq_el[1])

    space_2d = Tensor_spline_space([space_1d_1, space_1d_2], 1, domain.cx[:, :, 0], domain.cy[:, :, 0])

    print(space_2d.bc)

    # print dimension of spaces
    print(
        "dimension of space V0 : ",
        space_2d.E0.shape[1],
        "dimension of polar space bar(V0) : ",
        space_2d.E0.shape[0],
        "dimension of polar space bar(V0)_0 : ",
        space_2d.E0_0.shape[0],
    )
    print(
        "dimension of space V1 : ",
        space_2d.E1.shape[1],
        "dimension of polar space bar(V1) : ",
        space_2d.E1.shape[0],
        "dimension of polar space bar(V1)_0 : ",
        space_2d.E1_0.shape[0],
    )
    print(
        "dimension of space V2 : ",
        space_2d.E2.shape[1],
        "dimension of polar space bar(V2) : ",
        space_2d.E2.shape[0],
        "dimension of polar space bar(V2)_0 : ",
        space_2d.E2_0.shape[0],
    )
    print(
        "dimension of space V3 : ",
        space_2d.E3.shape[1],
        "dimension of polar space bar(V3) : ",
        space_2d.E3.shape[0],
        "dimension of polar space bar(V3)_0 : ",
        space_2d.E3_0.shape[0],
    )

    # plot three new polar splines in V0
    etaplot = [xp.linspace(0.0, 1.0, 200), xp.linspace(0.0, 1.0, 200)]
    xplot = [
        domain(etaplot[0], etaplot[1], 0.0, squeeze_out=True)[0],
        domain(etaplot[0], etaplot[1], 0.0, squeeze_out=True)[1],
    ]

    fig = plt.figure()
    fig.set_figheight(6)
    fig.set_figwidth(14)

    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    # coeffs in polar basis
    c0_pol1 = xp.zeros(space_2d.E0.shape[0], dtype=float)
    c0_pol2 = xp.zeros(space_2d.E0.shape[0], dtype=float)
    c0_pol3 = xp.zeros(space_2d.E0.shape[0], dtype=float)

    c0_pol1[0] = 1.0
    c0_pol2[1] = 1.0
    c0_pol3[2] = 1.0

    ax1.plot_surface(
        xplot[0],
        xplot[1],
        space_2d.evaluate_NN(etaplot[0], etaplot[1], xp.array([0.0]), c0_pol1, "V0")[:, :, 0],
        cmap="jet",
    )
    ax1.set_xlabel("R [m]", labelpad=5)
    ax1.set_ylabel("y [m]")
    ax1.set_title("1st polar spline in V0")

    ax2.plot_surface(
        xplot[0],
        xplot[1],
        space_2d.evaluate_NN(etaplot[0], etaplot[1], xp.array([0.0]), c0_pol2, "V0")[:, :, 0],
        cmap="jet",
    )
    ax2.set_xlabel("R [m]", labelpad=5)
    ax2.set_ylabel("y [m]")
    ax2.set_title("2nd polar spline in V0")

    ax3.plot_surface(
        xplot[0],
        xplot[1],
        space_2d.evaluate_NN(etaplot[0], etaplot[1], xp.array([0.0]), c0_pol3, "V0")[:, :, 0],
        cmap="jet",
    )
    ax3.set_xlabel("R [m]", labelpad=5)
    ax3.set_ylabel("y [m]")
    ax3.set_title("3rd polar spline in V0")

    if plot:
        plt.show()


if __name__ == "__main__":
    test_polar_splines_2D(plot=True)
