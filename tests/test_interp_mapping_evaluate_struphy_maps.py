import sys
sys.path.append('..')

import numpy as np
#import matplotlib.pyplot as plt

from hylife.geometry import domain_3d 
#import hylife.utilitis_FEEC.bsplines as bsp
import hylife.utilitis_FEEC.spline_space as spl
import hylife.utilitis_FEEC.basics.spline_evaluation_2d as eval_2d
import hylife.utilitis_FEEC.basics.spline_evaluation_3d as eval_3d

# evaluation points
eta1_v = np.linspace(0, 1, 10)
eta2_v = np.linspace(0, 1, 10)
eta3_v = np.linspace(0, 1, 10)

ee1, ee2 = np.meshgrid(eta1_v, eta2_v, indexing='ij')

for kind_map in ['cuboid', 'hollow cylinder', 'colella', 'orthogonal', 'hollow torus']:
    print(kind_map)

    domain = domain_3d.domain(kind_map)

    # 2d mapping from struphy domain
    X = lambda eta1, eta2 : domain.evaluate_12(eta1, eta2, .5, 'x')
    Y = lambda eta1, eta2 : domain.evaluate_12(eta1, eta2, .5, 'y')

    # 2d interpolation, test convergence:
    if kind_map=='cuboid':
        bc_2d = [False, False] 
        #bc_3d = bc_2d.append(False) 
        #print(bc_2d)
        #print(bc_3d)
    elif kind_map=='hollow cylinder':
        bc_2d = [False, True]
    elif kind_map=='colella':
        bc_2d = [False, False]
    elif kind_map=='orthogonal':
        bc_2d = [False, False]
    elif kind_map=='hollow torus':
        bc_2d = [False, True]

    for p1 in range(1, 6):
        p = [p1, 3]
        for Nel1 in [2**power for power in range(3, 8)]:
            Nel = [Nel1, 8]

            spaces_map = [spl.spline_space_1d(Nel_i, p_i, bc_2d_i) for Nel_i, p_i, bc_2d_i in zip(Nel, p, bc_2d)]

            cx, cy = domain_3d.interp_mapping(Nel, p, bc_2d, X, Y) # TODO: pass spline space object to interp_mapping

            X_h = np.empty(ee1.shape) # interpolated mapping evaluated at ee1, ee2
            Y_h = np.empty(ee1.shape)
            eval_2d.evaluate_tensor_product(
                spaces_map[0].T, spaces_map[1].T,
                spaces_map[0].p, spaces_map[1].p,
                spaces_map[0].NbaseN, spaces_map[1].NbaseN,
                cx, eta1_v, eta2_v, X_h, 0
                )
            eval_2d.evaluate_tensor_product(
                spaces_map[0].T, spaces_map[1].T,
                spaces_map[0].p, spaces_map[1].p,
                spaces_map[0].NbaseN, spaces_map[1].NbaseN,
                cy, eta1_v, eta2_v, Y_h, 0
                )

            # max-error
            err_x = np.max(np.max(X_h - X(ee1, ee2)))
            err_y = np.max(np.max(Y_h - Y(ee1, ee2)))
            print('p1: {0:2d}, Nel1: {1:4d}, max(fx_h - fx): {2:10.8f}, max(fy_h - fy): {3:10.8f}'.format(p1, Nel1, err_x, err_y))
            #print(X(ee1, ee2)[:, 0], X_h[:, 0])
            #print(X(ee1, ee2).shape)
            #input()

#plt.plot(eta1_v, X(ee1, ee2)[:, 0])