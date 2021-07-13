def test_plot_splines(plot=False):

    import sys
    sys.path.append('..')

    import numpy as np
    import hylife.utilitis_FEEC.spline_space as spl
    import matplotlib.pyplot as plt

    eta_v = np.linspace(0, 1, 200)

    Nel = 10
    for p in range(1,7):

        # spline spaces
        Vh_per = spl.spline_space_1d(Nel, p, spl_kind=True)
        Vh_cla = spl.spline_space_1d(Nel, p, spl_kind=False)

        plt.figure()
        plt.subplot(2, 2, 1)
        Vh_per.plot_splines(which='B-splines')

        plt.subplot(2, 2, 2)
        Vh_per.plot_splines(which='M-splines')

        plt.subplot(2, 2, 3)
        Vh_cla.plot_splines(which='B-splines')

        plt.subplot(2, 2, 4)
        Vh_cla.plot_splines(which='M-splines')

        
    if plot:
        plt.show()


if __name__ == '__main__':
    test_plot_splines(plot=True)