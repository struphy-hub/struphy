from numpy import NaN


def test_interpolation_1d(plot=False, p_range=7, N_range=10):

    import sys
    sys.path.append('..')

    import numpy as np

    import struphy.geometry.domain_3d as dom
    import struphy.feec.basics.spline_evaluation_1d as eva

    import matplotlib.pyplot as plt

    # function to interpolate
    fun = lambda eta : np.exp(2.*eta) - 2.*np.cos(2*np.pi*eta/.2)

    # plot points
    eta_plot = np.linspace(0., 1., 1000)
    fh_plot = np.zeros_like(eta_plot)
    plt.figure()

    print('1d convergence test:')
    for p in range(1, p_range):

        err   = [NaN]
        order = []

        for Nel in [2**n for n in range(5, N_range)]:

            # interpolation points
            x_grid = np.linspace(0., 1., Nel + 1)
            
            # call spline interpolation
            coeff, T, indN = dom.spline_interpolation_nd([p], [x_grid], fun(x_grid))

            # evaluate spline interpolant at plot points (need to use low-level evaluation routine, not spline_space class)
            eva.evaluate_vector(T[0], p, indN[0], coeff, eta_plot, fh_plot, 0)

            # error
            err.append(np.max(np.abs(fh_plot - fun(eta_plot))))
            order.append(np.log2(err[-2]/err[-1]))

            if True:
                print('p: {0:2d}, Nel: {1:4d},   error: {2:8.6f},    order: {3:4.2f}'.format(p, Nel, err[-1], order[-1]))

            if p<5 and Nel==2**5:
                plt.subplot(2, 2, p)
                plt.plot(eta_plot, fun(eta_plot), 'r', label='fun')
                plt.plot(eta_plot, fh_plot, 'b--', label='spline')
                plt.title('p: {0:2d}, Nel: {1:4d}'.format(p, Nel))
                plt.legend()
                plt.autoscale(enable=True, axis='x', tight=True)

        # check order of convergence
        assert order[-1] > (p+1) - .1
        print()

    if plot:
        plt.show()



def test_interpolation_2d(plot=False, p_range=7, N_range=8):

    import sys
    sys.path.append('..')

    import numpy as np

    import struphy.geometry.domain_3d as dom
    import struphy.feec.basics.spline_evaluation_2d as eva

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # function to interpolate
    fun = lambda eta1, eta2 : np.cos(2*np.pi*eta2/.2) * ( np.exp(2.*eta1) - 2.*np.cos(2*np.pi*eta1/.2) )

    # plot points
    eta_plot = [np.linspace(0., 1., 500), 
                np.linspace(0., 1., 500)]
    pp1, pp2 = np.meshgrid(eta_plot[0], eta_plot[1], indexing='ij')
    fh_plot = np.zeros_like(pp1)
    fig, axs = plt.subplots(2, 2)

    print('2d convergence test:')
    for p in range(1, p_range):

        err   = [NaN]
        order = []

        for Nel in [2**n for n in range(5, N_range)]:

            # interpolation points
            grids_1d = [np.linspace(0., 1., Nel + 1), 
                        np.linspace(0., 1., Nel + 2)]

            ee1, ee2 = np.meshgrid(grids_1d[0], grids_1d[1], indexing='ij')

            # call spline interpolation
            coeff, T, indN = dom.spline_interpolation_nd([p, p], grids_1d, fun(ee1, ee2))

            # evaluate spline interpolant at plot points (need to use low-level evaluation routine, not spline_space class)
            eva.evaluate_tensor_product(T[0], T[1], p, p, indN[0], indN[1], coeff, eta_plot[0], eta_plot[1], fh_plot, 0)

            # error
            err.append(np.max(np.abs(fh_plot - fun(pp1, pp2))))
            order.append(np.log2(err[-2]/err[-1]))

            if True:
                print('p: {0:2d}, Nel: {1:4d},   error: {2:8.6f},    order: {3:4.2f}'.format(p, Nel, err[-1], order[-1]))

            if p<5 and Nel==2**5:
                plt.subplot(2, 2, p)
                axs.flatten()[p-1].plot(pp1[:, 250], fun(pp1[:, 250], pp2[:, 250]), 'r', label='fun')
                axs.flatten()[p-1].plot(pp1[:, 250], fh_plot[:, 250], 'b--', label='spline')
                axs.flatten()[p-1].set_title('p: {0:2d}, Nel: {1:4d}'.format(p, Nel))
                axs.flatten()[p-1].legend()
                axs.flatten()[p-1].autoscale(enable=True, axis='x', tight=True)

        # check order of convergence
        assert order[-1] > (p+1) - .6

        print()

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d') 
    #ax.plot_surface(pp1, pp2, fun(pp1, pp2))
    #plt.title('fun')

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d') 
    #ax.plot_surface(pp1, pp2, fh_plot)
    #plt.title('spline')

    if plot:
        plt.show()



def test_interpolation_3d(plot=False, p_range=7, N_range=6):

    import sys
    sys.path.append('..')

    import numpy as np

    import struphy.geometry.domain_3d as dom
    import struphy.feec.basics.spline_evaluation_3d as eva

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # function to interpolate
    fun = lambda eta1, eta2, eta3 : np.sin(2*np.pi*eta3/.5) * np.cos(2*np.pi*eta2) * ( np.exp(2*eta1) - 2.*np.cos(2*np.pi*eta1/.4) )

    # plot points
    eta_plot = [np.linspace(0., 1., 150), 
                np.linspace(0., 1., 150),
                np.linspace(0., 1., 150)]
    pp1, pp2, pp3 = np.meshgrid(eta_plot[0], eta_plot[1], eta_plot[2], indexing='ij')
    fh_plot = np.zeros_like(pp1)
    fig, axs = plt.subplots(2, 2)

    print('3d convergence test:')
    for p in range(1, p_range):

        err   = [NaN]
        order = []

        for Nel in [2**n for n in range(5, N_range)]:

            # interpolation points
            grids_1d = [np.linspace(0., 1., Nel + 1), 
                        np.linspace(0., 1., Nel + 2),
                        np.linspace(0., 1., Nel + 3)]

            ee1, ee2, ee3 = np.meshgrid(grids_1d[0], grids_1d[1], grids_1d[2], indexing='ij')

            # call spline interpolation
            coeff, T, indN = dom.spline_interpolation_nd([p, p, p], grids_1d, fun(ee1, ee2, ee3))

            # evaluate spline interpolant at plot points (need to use low-level evaluation routine, not spline_space class)
            eva.evaluate_tensor_product(T[0], T[1], T[2], p, p, p, indN[0], indN[1], indN[2],
                                        coeff, eta_plot[0], eta_plot[1], eta_plot[2], fh_plot, 0)

            # error
            err.append(np.max(np.abs(fh_plot - fun(pp1, pp2, pp3))))
            order.append(np.log2(err[-2]/err[-1]))

            if True:
                print('p: {0:2d}, Nel: {1:4d},   error: {2:8.6f},    order: {3:4.2f}'.format(p, Nel, err[-1], order[-1]))

            if p<5 and Nel==2**5:
                plt.subplot(2, 2, p)
                axs.flatten()[p-1].plot(pp1[:, 40, 40], fun(pp1[:, 40, 40], pp2[:, 40, 40], pp3[:, 40, 40]), 'r', label='fun')
                axs.flatten()[p-1].plot(pp1[:, 40, 40], fh_plot[:, 40, 40], 'b--', label='spline')
                axs.flatten()[p-1].set_title('p: {0:2d}, Nel: {1:4d}'.format(p, Nel))
                axs.flatten()[p-1].legend()
                axs.flatten()[p-1].autoscale(enable=True, axis='x', tight=True)

        # check order of convergence
        #assert order[-1] > (p+1) - .6

        print()

    if plot:
        plt.show()



if __name__ == '__main__':
    test_interpolation_1d(plot=True)
    test_interpolation_2d(plot=True)
    test_interpolation_3d(plot=True)




