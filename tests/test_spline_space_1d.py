def test_evaluation():

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
            S0_per = Vh_per.evaluate_N(eta_v, coeff)
            plt.plot(eta_v, S0_per)
            
        plt.plot(Vh_per.greville, np.zeros(Vh_per.greville.shape), 'ro', label='greville')
        plt.plot(Vh_per.el_b, np.zeros(Vh_per.el_b.shape), 'k+', label='breaks')
        plt.title('B-splines, periodic, for p={0:2d}, Nel={1:4d}'.format(p, Nel))
        plt.legend()

        plt.subplot(2, 2, 2)
        for i in range(d_per):
            coeff = np.zeros(d_per)
            coeff[i] = 1
            S1_per = Vh_per.evaluate_D(eta_v, coeff)
            plt.plot(eta_v, S1_per)
            
        plt.plot(Vh_per.greville, np.zeros(Vh_per.greville.shape), 'ro', label='greville')
        plt.plot(Vh_per.el_b, np.zeros(Vh_per.el_b.shape), 'k+', label='breaks')
        plt.title('M-splines, periodic, for p={0:2d}, Nel={1:4d}'.format(p, Nel))
        plt.legend()

        plt.subplot(2, 2, 3)
        for i in range(n_cla):
            coeff = np.zeros(n_cla)
            coeff[i] = 1
            S0_cla = Vh_cla.evaluate_N(eta_v, coeff)
            plt.plot(eta_v, S0_cla)
            
        plt.plot(Vh_cla.greville, np.zeros(Vh_cla.greville.shape), 'ro', label='greville')
        plt.plot(Vh_cla.el_b, np.zeros(Vh_cla.el_b.shape), 'k+', label='breaks')

        assert S0_cla[-1] == 1.0

        plt.title('B-splines, clamped, for p={0:2d}, Nel={1:4d}'.format(p, Nel))
        plt.legend()

        plt.subplot(2, 2, 4)
        for i in range(d_cla):
            coeff = np.zeros(d_cla)
            coeff[i] = 1
            S1_cla = Vh_cla.evaluate_D(eta_v, coeff) 
            plt.plot(eta_v, S1_cla)
            
        plt.plot(Vh_cla.greville, np.zeros(Vh_cla.greville.shape), 'ro', label='greville')
        plt.plot(Vh_cla.el_b, np.zeros(Vh_cla.el_b.shape), 'k+', label='breaks')

        assert np.allclose(S1_cla[-1], 1.0*p*Nel)

        plt.title('M-splines, clamped, for p={0:2d}, Nel={1:4d}'.format(p, Nel))
        plt.legend()
        

    #plt.show()


if __name__ == '__main__':
    test_evaluation()