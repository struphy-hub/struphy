import sys
sys.path.append('..')
import numpy as np

import hylife.geometry.domain_3d

# define 2d spline space for 2d mapping
Nel = [8, 8]
p   = [1, 1]
spl_kind = [True, True]
#X = lambda eta1, eta2:  

#cx, cy = domain_3d.interp_mapping(Nel, p, spl_kind, X, Y)