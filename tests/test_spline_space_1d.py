def test_evaluation():

    import sys
    sys.path.append('..')

    import numpy as np
    import hylife.utilitis_FEEC.spline_space as spl
    import matplotlib.pyplot as plt

    from hylife.utilitis_FEEC.basics import spline_evaluation_1d as eval


    eta_v = np.linspace(0, 1, 200)

    Nel = 8
    for p in range(1,4):

        # spline spaces
        Vh_per = spl.spline_space_1d(Nel, p, spl_kind=True)
        Vh_cla = spl.spline_space_1d(Nel, p, spl_kind=False)

        # evaluation
        n_per = Vh_per.NbaseN
        d_per = Vh_per.NbaseD 
        n_cla = Vh_cla.NbaseN
        d_cla = Vh_cla.NbaseD 

        plt.figure()
        plt.subplot(2, 2, 1)
        for i in range(n_per):
            coeff = np.zeros(n_per)
            coeff[i] = 1
            S0_per = []
            for eta in eta_v:
                S0_per.append(eval.evaluate_n(Vh_per.T, Vh_per.p, Vh_per.NbaseN, coeff, eta))
            plt.plot(eta_v, S0_per)
            plt.plot(Vh_per.greville, np.zeros(Vh_per.greville.shape), 'ro')
        plt.title('B-splines, periodic, for p={0:2d}, Nel={1:4d}'.format(p, Nel))   

        plt.subplot(2, 2, 2)
        for i in range(d_per):
            coeff = np.zeros(d_per)
            coeff[i] = 1
            S1_per = []
            for eta in eta_v:
                S1_per.append(eval.evaluate_d(Vh_per.t, Vh_per.p - 1, Vh_per.NbaseD, coeff, eta)) 
            plt.plot(eta_v, S1_per)
            plt.plot(Vh_per.greville, np.zeros(Vh_per.greville.shape), 'ro')
        plt.title('M-splines, periodic, for p={0:2d}, Nel={1:4d}'.format(p, Nel))

        plt.subplot(2, 2, 3)
        for i in range(n_cla):
            coeff = np.zeros(n_cla)
            coeff[i] = 1
            S0_cla = []
            for eta in eta_v:
                S0_cla.append(eval.evaluate_n(Vh_cla.T, Vh_cla.p, Vh_cla.NbaseN, coeff, eta))
            plt.plot(eta_v, S0_cla)
            plt.plot(Vh_cla.greville, np.zeros(Vh_cla.greville.shape), 'ro')

        assert S0_cla[-1] == 1.0

        plt.title('B-splines, clamped, for p={0:2d}, Nel={1:4d}'.format(p, Nel)) 

        plt.subplot(2, 2, 4)
        for i in range(d_cla):
            coeff = np.zeros(d_cla)
            coeff[i] = 1
            S1_cla = []
            for eta in eta_v:
                S1_cla.append(eval.evaluate_d(Vh_cla.t, Vh_cla.p - 1, Vh_cla.NbaseD, coeff, eta)) 
            plt.plot(eta_v, S1_cla)
            plt.plot(Vh_cla.greville, np.zeros(Vh_cla.greville.shape), 'ro')

        assert S1_cla[-1] == 1.0*p*Nel

        plt.title('M-splines, clamped, for p={0:2d}, Nel={1:4d}'.format(p, Nel))


    #plt.show()


if __name__ == '__main__':
    test_evaluation()