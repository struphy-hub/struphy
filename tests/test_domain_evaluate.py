import sys
sys.path.append('..')
import numpy as np

import hylife.geometry.domain_3d

# cuboid, default params_map = [2., 3., 4.].
domain = domain_3d.domain(10, params_map=None, Nel=None, p=None, spl_kind=None, cx=None, cy=None, cz=None)

kind_map = 11 # hollow cylinder. params_map = [a1, a2, lz].
kind_map = 12 # colella. params_map = [lx, ly, alpha, lz].
kind_map = 13 # othogonal. params_map = [ly, ly, alpha, lz].
kind_map = 14 # hollow torus. params_map = [a1, a2, r0].