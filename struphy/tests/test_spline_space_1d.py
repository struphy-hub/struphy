def test_plot_splines(plot=False):

    import numpy as np
    from struphy.feec.spline_space import Spline_space_1d
    import matplotlib.pyplot as plt

    eta_v = np.linspace(0, 1, 200)

    Nel = 8
    for p in range(1,5):

        # spline spaces
        Vh_per = Spline_space_1d(Nel, p, spl_kind=True)
        Vh_cla = Spline_space_1d(Nel, p, spl_kind=False)

        plt.figure()
        plt.subplot(2, 2, 1)
        Vh_per.plot_splines(which='N')

        plt.subplot(2, 2, 3)
        Vh_per.plot_splines(which='D')

        plt.subplot(2, 2, 2)
        Vh_cla.plot_splines(which='N')

        plt.subplot(2, 2, 4)
        Vh_cla.plot_splines(which='D')

        
    if plot:
        plt.show()


def test_indices():

    import numpy as np
    from struphy.feec.spline_space import Spline_space_1d
    from struphy.feec.bsplines import find_span, basis_funs

    Nel = 10
    nq = 2

    for p in range(1, 4):

        space = Spline_space_1d(Nel, p, True)
        space.set_projectors(nq=nq)

        pts = space.projectors.ptsG

        print('\nDegree:', p)
        print('\nNumber of elements:', Nel)
        print('\nElement boundaries:  ', space.el_b)
        print('\nGreville points:     ', space.greville)
        print('\nHistop. quad points: ', space.projectors.n_quad)
        print('\nInterpolation points:', space.projectors.x_int)
        print('\nHistopol. boundaries:', space.projectors.x_his)
        print('\nGauss-Leg. quad pts:\n', pts)
        print('\nCollocation Greville:\n', space.projectors.N_int.toarray())
        print('')

        print('ie, iq, span, basis, inds:')
        # elements
        for ie in range(pts.shape[0]):
            # quadrature points in element
            for iq in range(pts.shape[1]):
                pt = pts[ie, iq]
                # B-splines
                span  = find_span(space.T, p, pt)
                basis = basis_funs(space.T, p, pt, span) 
                inds  = [span - p + i for i in range(p + 1)]
                # M-splines
                span_D  = find_span(space.t, p - 1, pt)
                basis_D = basis_funs(space.t, p - 1, pt, span_D) 
                inds_D  = [span_D - (p - 1) + i for i in range(p)]
                print(ie, iq, span, basis, inds)
                print(ie, iq, span_D, basis_D, inds_D)
                print('')

        print('knot span indices as matrix:')
        print(space.projectors.span_ptsG_N)
        print(space.projectors.span_ptsG_D)
        print('')

        print('Shapes of basis matrices:')
        print(space.projectors.basis_ptsG_N.shape)
        print(space.projectors.basis_ptsG_D.shape)
        print('')

        print('Values of bases at first quadrature point:')
        print(space.projectors.basis_ptsG_N[0, 0, :])
        print(space.projectors.basis_ptsG_D[0, 0, :])
        print('')


 

if __name__ == '__main__':
    test_plot_splines(plot=True)
    test_indices()