from pyccel.decorators import pure, stack_array

from numpy import zeros
from numpy import sin, cos, tan, pi, sqrt, arctan2, arcsin, arctan

import struphy.bsplines.bsplines_kernels as bsplines_kernels 
import struphy.bsplines.evaluation_kernels_2d as evaluation_kernels_2d
import struphy.bsplines.evaluation_kernels_3d as evaluation_kernels_3d


#==============================================================================
# Base class
#==============================================================================

#class Mapping(ABC):
#
#    @abstractmethod
#    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):
#        """
#        Evaluate mapping.
#        """
#    
#    @abstractmethod
#    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):
#        """
#        Evaluate Jacobian of mapping.
#        """
#
#    @stack_array('df_mat')
#    def eval_det_df(self, eta1: float, eta2: float, eta3: float) -> float :
#
#        df_mat = empty((3, 3), dtype=float)
#        df(eta1, eta2, eta3, kind_map, params, t1, t2,
#           t3, p, ind1, ind2, ind3, cx, cy, cz, df_mat)
#
#        return linalg_kernels.det(df_mat)
#
#    @stack_array('df_mat')
#    def eval_df_inv(self, eta1: float, eta2: float, eta3: float, dfinv_out: 'float[:,:]'):
#
#        df_mat = empty((3, 3), dtype=float)
#        self.eval_df(eta1, eta2, eta3, df_mat)
#        linalg_kernels.matrix_inv(df_mat, dfinv_out)

#==============================================================================
# Derived classes
#==============================================================================

class CuboidMapping:

    def __init__(self,
                 l1: float, r1: float,
                 l2: float, r2: float,
                 l3: float, r3: float):

        self._l1 = l1; self._r1 = r1
        self._l2 = l2; self._r2 = r2
        self._l3 = l3; self._r3 = r3

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        # value =  begin + (end - begin) * eta
        f_out[0] = self._l1 + (self._r1 - self._l1) * eta1
        f_out[1] = self._l2 + (self._r2 - self._l2) * eta2
        f_out[2] = self._l3 + (self._r3 - self._l3) * eta3

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        df_out[0, 0] = self._r1 - self._l1
        df_out[0, 1] = 0.
        df_out[0, 2] = 0.
        df_out[1, 0] = 0.
        df_out[1, 1] = self._r2 - self._l2
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = self._r3 - self._l3


#==============================================================================
class OrthogonalMapping:

    def __init__(self, lx: float, ly: float, alpha: float, lz: float):

        self._lx = lx
        self._ly = ly
        self._lz = lz
        self._alpha = alpha

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        f_out[0] = self._lx * (eta1 + self._alpha * sin(2*pi*eta1))
        f_out[1] = self._ly * (eta2 + self._alpha * sin(2*pi*eta2))
        f_out[2] = self._lz *  eta3

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        df_out[0, 0] = self._lx * (1 + self._alpha * cos(2*pi*eta1) * 2*pi)
        df_out[0, 1] = 0.
        df_out[0, 2] = 0.
        df_out[1, 0] = 0.
        df_out[1, 1] = self._ly * (1 + self._alpha * cos(2*pi*eta2) * 2*pi)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = self._lz


#==============================================================================
class CollelaMapping:

    def __init__(self, lx: float, ly: float, alpha: float, lz: float):

        self._lx = lx
        self._ly = ly
        self._lz = lz
        self._alpha = alpha

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        f_out[0] = self._lx * (eta1 + self._alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        f_out[1] = self._ly * (eta2 + self._alpha * sin(2*pi*eta1) * sin(2*pi*eta2))
        f_out[2] = self._lz *  eta3

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        df_out[0, 0] = self._lx * (1 + self._alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi)
        df_out[0, 1] = self._lx *      self._alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi
        df_out[0, 2] = 0.
        df_out[1, 0] = self._ly *      self._alpha * cos(2*pi*eta1) * sin(2*pi*eta2) * 2*pi
        df_out[1, 1] = self._ly * (1 + self._alpha * sin(2*pi*eta1) * cos(2*pi*eta2) * 2*pi)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = self._lz


#==============================================================================
class HollowCylinderMapping:

    def __init__(self, a1: float, a2: float, lz: float):

        self._a1 = a1
        self._a2 = a2
        self._lz = lz

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        da = self._a2 - self._a1

        f_out[0] = (self._a1 + eta1 * da) * cos(2*pi*eta2)
        f_out[1] = (self._a1 + eta1 * da) * sin(2*pi*eta2)
        f_out[2] =  self._lz * eta3

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        da = self._a2 - self._a1

        df_out[0, 0] = da * cos(2*pi*eta2)
        df_out[0, 1] = -2*pi * (self._a1 + eta1 * da) * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = da * sin(2*pi*eta2)
        df_out[1, 1] = 2*pi * (self._a1 + eta1 * da) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = self._lz


#==============================================================================
class PoweredEllipseMapping:

    def __init__(self, rx: float, ry: float, lz: float, s: float):

        self._rx = rx
        self._ry = ry
        self._lz = lz
        self._s  = s

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        f_out[0] = self._rx * eta1**self._s * cos(2*pi*eta2)
        f_out[1] = self._ry * eta1**self._s * sin(2*pi*eta2)
        f_out[2] = self._lz * eta3

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        df_out[0, 0] =           eta1**(self._s-1) * self._rx * cos(2*pi*eta2)
        df_out[0, 1] = (-2*pi) * eta1** self._s    * self._rx * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] =           eta1**(self._s-1) * self._ry * sin(2*pi*eta2)
        df_out[1, 1] = ( 2*pi) * eta1** self._s    * self._ry * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = self._lz


#==============================================================================
class HollowTorusMapping:

    def __init__(self, a1: float, a2: float, r0: float, tor_period: float):

        self._a1  = a1
        self._a2  = a2
        self._r0  = r0
        self._tor_period = tor_period

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        a1 = self._a1
        a2 = self._a2
        r0 = self._r0
        tor_period = self._tor_period

        da = self._a2 - self._a1

        f_out[0] =  ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * cos(2*pi*eta3 / tor_period)
        f_out[1] = -((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3 / tor_period)
        f_out[2] =   (a1 + eta1 * da) * sin(2*pi*eta2) 


    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        a1 = self._a1
        a2 = self._a2
        r0 = self._r0
        tor_period = self._tor_period

        da = a2 - a1

        df_out[0, 0] = da * cos(2*pi*eta2) * cos(2*pi*eta3 / tor_period)
        df_out[0, 1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * cos(2*pi*eta3 / tor_period)
        df_out[0, 2] = -2*pi / tor_period * ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * sin(2*pi*eta3 / tor_period) 
        df_out[1, 0] = da * cos(2*pi*eta2) * (-1) * sin(2*pi*eta3 / tor_period)
        df_out[1, 1] = -2*pi * (a1 + eta1 * da) * sin(2*pi*eta2) * (-1) * sin(2*pi*eta3 / tor_period)
        df_out[1, 2] = ((a1 + eta1 * da) * cos(2*pi*eta2) + r0) * (-1) * cos(2*pi*eta3 / tor_period) * 2*pi / tor_period
        df_out[2, 0] = da * sin(2*pi*eta2)
        df_out[2, 1] = (a1 + eta1 * da) * cos(2*pi*eta2) * 2*pi
        df_out[2, 2] = 0.


#==============================================================================
class HollowTorusMappingSFL:

    def __init__(self, a1: float, a2: float, r0: float, tor_period: float):

        self._a1  = a1
        self._a2  = a2
        self._r0  = r0
        self._tor_period = tor_period

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        a1 = self._a1
        a2 = self._a2
        r0 = self._r0
        tor_period = self._tor_period

        da    = a2 - a1
        r     = a1 + eta1 * da
        theta = 2 * arctan(sqrt((1 + r/self._r0) / (1 - r/r0)) * tan(pi*eta2))

        f_out[0] =  (r * cos(theta) + r0) * cos(2*pi*eta3 / tor_period)
        f_out[1] = -(r * cos(theta) + r0) * sin(2*pi*eta3 / tor_period)
        f_out[2] =   r * sin(theta)

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        a1 = self._a1
        a2 = self._a2
        r0 = self._r0
        tor_period = self._tor_period

        da    = a2 - a1
        r     = a1 + da*eta1
        eps   =  r/r0
        eps_p = da/r0
        tpe   = tan(pi*eta2)
        tpe_p = pi/cos(pi*eta2)**2
        g     = sqrt((1 + eps)/(1 - eps))
        g_p   = 1/(2*g) * (eps_p*(1 - eps) + (1 + eps)*eps_p)/(1 - eps)**2

        theta        = 2*arctan(g*tpe)
        dtheta_deta1 = 2/(1 + (g*tpe)**2)*g_p*tpe
        dtheta_deta2 = 2/(1 + (g*tpe)**2)*g*tpe_p

        df_out[0, 0] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * cos(2*pi*eta3 / tor_period)
        df_out[0, 1] = -r * sin(theta) * dtheta_deta2 * cos(2*pi*eta3 / tor_period)
        df_out[0, 2] = -2*pi / tor_period * (r * cos(theta) + r0) * sin(2*pi*eta3 / tor_period)

        df_out[1, 0] = (da * cos(theta) - r * sin(theta) * dtheta_deta1) * (-1) * sin(2*pi*eta3 / tor_period)
        df_out[1, 1] = -r * sin(theta) * dtheta_deta2 * (-1) * sin(2*pi*eta3 / tor_period)
        df_out[1, 2] = 2*pi / tor_period * (r * cos(theta) + r0) * (-1) * cos(2*pi*eta3 / tor_period)

        df_out[2, 0] = (da * sin(theta) + r * cos(theta) * dtheta_deta1)
        df_out[2, 1] = r * cos(theta) * dtheta_deta2
        df_out[2, 2] = 0.


#==============================================================================
class ShafranovShiftMapping:

    def __init__(self, rx: float, ry: float, lz: float, de: float):

        self._rx = rx
        self._ry = ry
        self._lz = lz
        self._de = de

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        rx = self._rx
        ry = self._ry
        lz = self._lz
        de = self._de

        f_out[0] = (eta1 * rx) * cos(2*pi*eta2) + (1-eta1**2) * rx * de
        f_out[1] = (eta1 * ry) * sin(2*pi*eta2)
        f_out[2] = (eta3 * lz)

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        rx = self._rx
        ry = self._ry
        lz = self._lz
        de = self._de

        df_out[0, 0] = rx * cos(2*pi*eta2) - 2 * eta1 * rx * de
        df_out[0, 1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = ry * sin(2*pi*eta2)
        df_out[1, 1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz


#==============================================================================
class ShafranovShiftSqrtMapping:

    def __init__(self, rx: float, ry: float, lz: float, de: float):

        self._rx = rx
        self._ry = ry
        self._lz = lz
        self._de = de

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        rx = self._rx
        ry = self._ry
        lz = self._lz
        de = self._de

        f_out[0] = (eta1 * rx) * cos(2*pi*eta2) + (1-sqrt(eta1)) * rx * de
        f_out[1] = (eta1 * ry) * sin(2*pi*eta2)
        f_out[2] = (eta3 * lz)

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        rx = self._rx
        ry = self._ry
        lz = self._lz
        de = self._de

        df_out[0, 0] = rx * cos(2*pi*eta2) - 0.5 / sqrt(eta1) * rx * de
        df_out[0, 1] = -2*pi * (eta1 * rx) * sin(2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = ry * sin(2*pi*eta2)
        df_out[1, 1] =  2*pi * (eta1 * ry) * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz


#==============================================================================
class ShafranovDShapedMapping:

    def __init__(self, r0: float, lz: float,
                 dx: float, dy: float, dg: float, eg: float, kg: float):

        self._r0 = r0
        self._lz = lz
        self._dx = dx
        self._dy = dy
        self._dg = dg
        self._eg = eg
        self._kg = kg

    @pure
    def eval_f(self, eta1: float, eta2: float, eta3: float, f_out: 'float[:]'):

        r0 = self._r0
        lz = self._lz
        dx = self._dx
        dy = self._dy
        dg = self._dg
        eg = self._eg
        kg = self._kg

        f_out[0] = r0 * (1 + (1 - eta1**2) * dx + eg *
                              eta1 * cos(2*pi*eta2 + arcsin(dg)*eta1*sin(2*pi*eta2)))
        f_out[1] = r0 * ((1 - eta1**2) * dy + eg * kg * eta1 * sin(2*pi*eta2))
        f_out[2] = (eta3 * lz)

    @pure
    def eval_df(self, eta1: float, eta2: float, eta3: float, df_out: 'float[:,:]'):

        r0 = self._r0
        lz = self._lz
        dx = self._dx
        dy = self._dy
        dg = self._dg
        eg = self._eg
        kg = self._kg

        df_out[0, 0] = r0 * (- 2 * dx * eta1 - eg * eta1 * sin(2*pi*eta2) * arcsin(dg) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2) + eg * cos(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2))
        df_out[0, 1] = - r0 * eg * eta1 * (2*pi*eta1 * cos(2*pi*eta2) * arcsin(dg) + 2*pi) * sin(eta1 * sin(2*pi*eta2) * arcsin(dg) + 2*pi*eta2)
        df_out[0, 2] = 0.
        df_out[1, 0] = r0 * (- 2 * dy * eta1 + eg * kg * sin(2*pi*eta2))
        df_out[1, 1] = 2 * pi * r0 * eg * eta1 * kg * cos(2*pi*eta2)
        df_out[1, 2] = 0.
        df_out[2, 0] = 0.
        df_out[2, 1] = 0.
        df_out[2, 2] = lz
