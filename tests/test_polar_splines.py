def test_polar_splines_2D(plot=False):
    
    import sys
    sys.path.append('..')

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    import hylife.geometry.domain_3d as dom
    import hylife.geometry.polar_splines as pol
    
    import hylife.utilitis_FEEC.spline_space as spl

    from hylife.utilitis_FEEC.projectors import projectors_global as proj
    
    # parameters
    Nel        = [4, 6]            # number of elements (number of elements in angular direction must be a multiple of 3)
    p          = [3, 3]            # splines degrees 
    spl_kind   = [False, True]     # kind of splines (for polar domains always [False, True] which means [clamped, periodic])
    nq_el      = [10, 10]          # number of quadrature points per element for integrations
    bc         = ['f', 'f']        # boundary conditions in radial direction (for polar domain always 'f' at eta1 = 0 (pole))
    
    # set up 1D spline spaces in radial and angular direction and 2D tensor-product space 
    space_1d = [spl.spline_space_1d(Nel, p, spl_kind, nq_el) for Nel, p, spl_kind, nq_el in zip(Nel, p, spl_kind, nq_el)]
    space_2d =  spl.tensor_spline_space(space_1d)
    
    # geometry
    geometry   = 'spline cylinder'
    a          = 1.0
    R0         = 10.0
    params_map = [2*np.pi*R0]

    # mapping to be interpolated
    X = lambda eta1, eta2 : a*eta1*np.cos(2*np.pi*eta2)
    Y = lambda eta1, eta2 : a*eta1*np.sin(2*np.pi*eta2)
    
    #kappa = 0.2
    #delta = 0.2
    #X = lambda eta1, eta2 : (1 - kappa)*eta1*np.cos(2*np.pi*eta2) - delta*eta1**2
    #Y = lambda eta1, eta2 : (1 + kappa)*eta1*np.sin(2*np.pi*eta2)
    
    #eps = 0.3
    #e = 1.4
    #X = lambda eta1, eta2 : 1/eps*(1 - np.sqrt(1 + eps*(eps + 2*eta1*np.cos(2*np.pi*eta2))))
    #Y = lambda eta1, eta2 : e*1/np.sqrt(1 - eps**2/4)*eta1*np.sin(2*np.pi*eta2)/(1 + eps*X(eta1, eta2))

    # interpolate mapping (apply Pi_0)
    space_1d[0].set_projectors()
    space_1d[1].set_projectors()
    space_2d.set_projectors()

    cx = space_2d.projectors.PI_0(X)
    cy = space_2d.projectors.PI_0(Y)

    # create domain                              
    domain = dom.domain(geometry, params_map, Nel, p, spl_kind, cx, cy)
    
    # plot the control points and the grid
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    
    grid_x = domain.evaluate(space_2d.el_b[0], space_2d.el_b[1], np.array([0.]), 'x')[:, :, 0]
    grid_y = domain.evaluate(space_2d.el_b[0], space_2d.el_b[1], np.array([0.]), 'y')[:, :, 0]

    for i in range(space_2d.el_b[0].size):
        plt.plot(grid_x[i, :], grid_y[i, :], 'k')

    for j in range(space_2d.el_b[1].size):
        plt.plot(grid_x[:, j], grid_y[:, j], 'k')

    plt.scatter(domain.cx[:, :, 0].flatten(), domain.cy[:, :, 0].flatten(), s=30, color='r')

    plt.axis('square')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')

    # create polar splines by passing the 2D tensor-product space and the control points c^x_ij, c^y_ij 
    polar_splines = pol.polar_splines_2D(space_2d, domain.cx[:, :, 0], domain.cy[:, :, 0])

    # set extraction operators for polar splines in spaces V0, V1, V2 and V3
    space_2d.set_polar_splines(cx, cy)
    
    # print dimension of spaces
    print('dimension of space V0 : ', space_2d.E0.shape[1], 'dimension of polar space bar(V0) : ', space_2d.E0.shape[0])
    print('dimension of space V1 : ', space_2d.E1.shape[1], 'dimension of polar space bar(V1) : ', space_2d.E1.shape[0])
    print('dimension of space V2 : ', space_2d.E2.shape[1], 'dimension of polar space bar(V2) : ', space_2d.E2.shape[0])
    print('dimension of space V3 : ', space_2d.E3.shape[1], 'dimension of polar space bar(V3) : ', space_2d.E3.shape[0])
    
    # plot three new polar splines in V0
    etaplot = [np.linspace(0., 1., 200), np.linspace(0., 1., 200), np.linspace(0., 1., 1)]
    xplot   = [domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'x')[:, :, 0], domain.evaluate(etaplot[0], etaplot[1], etaplot[2], 'y')[:, :, 0]]
    
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    
    # coeffs in polar basis
    c0_pol1 = np.zeros(space_2d.E0.shape[0], dtype=float)
    c0_pol2 = np.zeros(space_2d.E0.shape[0], dtype=float)
    c0_pol3 = np.zeros(space_2d.E0.shape[0], dtype=float)
    
    c0_pol1[0] = 1.
    c0_pol2[1] = 1.
    c0_pol3[2] = 1.
    
    ax1.plot_surface(xplot[0], xplot[1], space_2d.evaluate_NN(etaplot[0], etaplot[1], c0_pol1, 'V0'), cmap='jet')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    
    ax2.plot_surface(xplot[0], xplot[1], space_2d.evaluate_NN(etaplot[0], etaplot[1], c0_pol2, 'V0'), cmap='jet')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    
    ax3.plot_surface(xplot[0], xplot[1], space_2d.evaluate_NN(etaplot[0], etaplot[1], c0_pol3, 'V0'), cmap='jet')
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    
    if plot: 
        plt.show()
    
    
if __name__ == '__main__':
    test_polar_splines_2D(plot=True)