import sys
sys.path.append('..')
import numpy as np

from hylife.geometry import domain_3d 

eta1_vec = np.linspace(0, 1, 10)
eta2_vec = np.linspace(0, 1, 10)
eta3_vec = np.linspace(0, 1, 10)

# kind_map = 10 : cuboid. params_map = [lx, ly, lz].
# kind_map = 11 : hollow cylinder. params_map = [a1, a2, lz].
# kind_map = 12 : colella. params_map = [lx, ly, alpha, lz].
# kind_map = 13 : othogonal. params_map = [ly, ly, alpha, lz].
# kind_map = 14 : hollow torus. params_map = [a1, a2, r0].

for kind_map in ['cuboid', 'hollow cylinder', 'colella', 'orthogonal', 'hollow torus']:
    print(kind_map)

    domain = domain_3d.domain(kind_map)

    # 2d mapping, components:
    X_2d = lambda eta1, eta2 : domain.evaluate(eta1, eta2, .5, 'x')
    Y_2d = lambda eta1, eta2 : domain.evaluate(eta1, eta2, .5, 'y')

    # 2d interpolation, test convergence:

    if kind_map=='cuboid':
        bc_2d = [False, False] 
        bc_3d = bc_2d.append(False) 
        print(bc_2d)
        print(bc_3d)

    for p1 in range(1, 6):
        print(p1)
        for power in range(3, 8):
            print(str(2**power))

            cx, cy = domain_3d.interp_mapping([2**power, 8], [p1, 1], bc_2d, X_2d, Y_2d)

