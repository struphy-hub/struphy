from struphy.geometry.base import Domain, interp_mapping
from struphy.geometry.angular_coordinates_torus import theta

import numpy as np
import h5py


class Cuboid(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= l_1 + (r_1 - l_1)\,\eta_1\,, 

        y &= l_2 + (r_2 - l_2)\,\eta_2\,, 

        z &= l_3 + (r_3 - l_3)\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/cuboid.png'''

    def __init__(self, params_map=None):

        self._kind_map = 10

        if params_map is None:
            params = {'l1': 0., 'r1': 1., 'l2': 0.,
                      'r2': 1., 'l3': 0., 'r3': 1.}
        else:
            params = params_map

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': 'l1 + (r1 - l1)*x1',
                                           'y': 'l2 + (r2 - l2)*x2',
                                           'z': 'l3 + (r3 - l3)*x3'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = False

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class HollowCylinder(Domain):
    r'''
    .. math::

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2) + R_0\,, 

        y &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin(2\pi\,\eta_2)\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/hollow_cylinder.png'''

    def __init__(self, params_map=None):

        self._kind_map = 11

        if params_map is None:
            params = {'a1': 0.2, 'a2': 1., 'R0': 3., 'Lz': 2*np.pi*3.}
        else:
            params = params_map

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': '(a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0',
                                           'y': '(a1 + (a2 - a1)*x1)*sin(2*pi*x2)',
                                           'z': 'Lz*x3'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        if self.params_map[0] == 0.:
            self._pole = True
        else:
            self._pole = False

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class Colella(Domain):
    r'''
    .. math::
    
        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= L_x\,\left[\,\eta_1 + \alpha\sin(2\pi\,\eta_1)\sin(2\pi\,\eta_2)\,\right]\,, 

        y &= L_y\,\left[\,\eta_2 + \alpha\sin(2\pi\,\eta_2)\sin(2\pi\,\eta_1)\,\right]\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/colella.png'''

    def __init__(self, params_map=None):

        self._kind_map = 12

        if params_map is None:
            params = {'Lx': 1., 'Ly': 1., 'alpha': 0.1, 'Lz': 1.}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                            'y': 'Ly*(x2 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                            'z': 'Lz*x3'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = False

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map
    
    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class Orthogonal(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= L_x\,\left[\,\eta_1 + \alpha\sin(2\pi\,\eta_1)\,\right]\,, 

        y &= L_y\,\left[\,\eta_2 + \alpha\sin(2\pi\,\eta_2)\,\right]\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/orthogonal.png'''

    def __init__(self, params_map=None):

        self._kind_map = 13

        if params_map is None:
            params = {'Lx': 1., 'Ly': 1., 'alpha': 0.1, 'Lz': 1.}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1))',
                                            'y': 'Ly*(x2 + alpha*sin(2*pi*x2))',
                                            'z': 'Lz*x3'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = False

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class HollowTorus(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2)+R_0\rbrace\cos(2\pi\,\eta_3)\,, 

        y &=  \,\,\,\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin(2\pi\,\eta_2)\,, 

        z &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2)+R_0\rbrace\sin(2\pi\,\eta_3)\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/hollow_torus.png'''

    def __init__(self, params_map=None):

        self._kind_map = 14

        if params_map is None:
            params = {'a1': 0.2, 'a2': 1., 'R0': 3.}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * cos(2*pi*x3)',
                                            'y': '( a1 + (a2 - a1)*x1)*sin(2*pi*x2)',
                                            'z': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * sin(2*pi*x3)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = True
        if self.params_map[0] == 0.:
            self._pole = True
        else:
            self._pole = False

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class EllipticCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= x_0+r_x\,\eta_1\cos(2\pi\,\eta_2)\,, 

        y &= y_0+r_y\,\eta_1\sin(2\pi\,\eta_2)\,, 
        
        z &= z_0+L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/elliptic_cyl.png'''

    def __init__(self, params_map=None):

        self._kind_map = 15

        if params_map is None:
            params = {'x0': 0., 'y0': 0., 'z0': 0.,
                        'rx': 1., 'ry': 2., 'Lz': 1.}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2)',
                                            'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                            'z': 'z0 + (x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = True

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class RotatedEllipticCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= x_0 + r_1\,\eta_1\cos(2\pi\,th)\cos(2\pi\,\eta_2) - r_2\,\eta_1\sin(2\pi\,th)\sin(2\pi\,\eta_2)\,, 

        y &= y_0 + r_1\,\eta_1\sin(2\pi\,th)\cos(2\pi\,\eta_2) + r_2\,\eta_1\cos(2\pi\,th)\sin(2\pi\,\eta_2)\,, 

        z &= z_0 + L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/rot_elliptic_cyl.png'''

    def __init__(self, params_map=None):

        self._kind_map = 16

        if params_map is None:
            params = {'x0': 0., 'y0': 0., 'z0': 0.,
                        'r1': 1., 'r2': 2., 'Lz': 1., 'th': 0.2}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'x0 + (x1*r1) * cos(2*pi*th) * cos(2*pi*x2) - (x1*r2) * sin(2*pi*th) * sin(2*pi*x2)',
                                            'y': 'y0 + (x1*r1) * sin(2*pi*th) * cos(2*pi*x2) + (x1*r2) * cos(2*pi*th) * sin(2*pi*x2)',
                                            'z': 'z0 + (x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = True

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class PoweredEllipticCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= x_0+r_x\,\eta_1^s\cos(2\pi\,\eta_2)\,, 

        y &= y_0+r_y\,\eta_1^s\sin(2\pi\,\eta_2)\,, 

        z &= z_0+L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/pow_elliptic_cyl.png'''

    def __init__(self, params_map=None):

        self._kind_map = 17
            
        if params_map is None:
            params = {'x0': 0., 'y0': 0., 'z0': 0.,
                        'rx': 1., 'ry': 1., 'Lz': 1., 's': 0.5}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'x0 + (x1**s) * rx * cos(2*pi*x2)',
                                            'y': 'y0 + (x1**s) * ry * sin(2*pi*x2)',
                                            'z': 'z0 + (x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = True

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class ShafranovShiftCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= x_0+r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\eta_1^2)r_x\Delta\,, 

        y &= y_0+r_y\,\eta_1\sin(2\pi\,\eta_2)\,, 

        z &= z_0+L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/shafranov_shift.png'''

    def __init__(self, params_map=None):

        self._kind_map = 18

        if params_map is None:
            params = {'x0': 0., 'y0': 0., 'z0' : 0., 'rx' : 1., 'ry' : 1., 'Lz' : 1., 'delta' : 0.2}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2) + (1-x1**2) * rx * delta',
                                            'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                            'z': 'z0 + (x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = True

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class ShafranovSqrtCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= x_0+r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\sqrt \eta_1)r_x\Delta\,, 

        y &= y_0+r_y\,\eta_1\sin(2\pi\,\eta_2)\,, 

        z &= z_0+L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/shafranov_sqrt.png'''

    def __init__(self, params_map=None):

        self._kind_map = 19

        if params_map is None:
            params = {'x0': 0., 'y0': 0., 'z0': 0.,
                        'rx': 1., 'ry': 1., 'Lz': 1., 'delta': 0.2}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'x0 + (x1*rx) * cos(2*pi*x2) + (1-sqrt(x1)) * rx * delta',
                                            'y': 'y0 + (x1*ry) * sin(2*pi*x2)',
                                            'z': 'z0 + (x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = True

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class ShafranovDshapedCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= x_0+R_0\left[1 + (1 - \eta_1^2)\Delta_x + \eta_1\epsilon\cos(2\pi\,\eta_2 + \arcsin(\delta)\eta_1\sin(2\pi\,\eta_2)) \right]\,, 

        y &= y_0+R_0\left[    (1 - \eta_1^2)\Delta_y + \eta_1\epsilon\kappa\sin(2\pi\,\eta_2)\right]\,, 

        z &= z_0+L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/shafranov_dshaped.png'''

    def __init__(self, params_map=None):

        self._kind_map = 20

        if params_map is None:
            params = {'x0': 0., 'y0': 0., 'z0': 0., 'R0': 2., 'Lz': 1., 'delta_x': 0.1,
                        'delta_y': 0., 'delta_gs': 0.33, 'epsilon_gs': 0.32, 'kappa_gs': 1.7}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'x0 + R0 * ( 1 + (1 - x1**2) * delta_x + x1 * epsilon_gs * cos(2*pi*x2 + asin(delta_gs)*x1*sin(2*pi*x2)) )',
                                            'y': 'y0 + R0 * (     (1 - x1**2) * delta_y + x1 * epsilon_gs * kappa_gs * sin(2*pi*x2) )',
                                            'z': 'z0 + (x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = True

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class ShafranovNonAxisSymmCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= x_0+R_0(1 + \chi \cos(2\pi\eta_3))\left[1 + (1 - \eta_1^2)\Delta_x + \eta_1\epsilon\cos(2\pi\,\eta_2 + \arcsin(\delta)\eta_1\sin(2\pi\,\eta_2)) \right]\,, 

        y &= y_0+R_0(1 - \chi \cos(2\pi\eta_3))\left[    (1 - \eta_1^2)\Delta_y + \eta_1\epsilon\kappa\sin(2\pi\,\eta_2)\right]\,, 

        z &= z_0+L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/shafranov_nonsymm.png'''

    def __init__(self, params_map=None):

        self._kind_map = 21

        if params_map is None:
            params = {'x0': 0., 'y0': 0., 'z0': 0., 'R0': 2., 'Lz': 1., 'delta_x': 0.1,
                        'delta_y': 0., 'delta_gs': 0.33, 'epsilon_gs': 0.32, 'kappa_gs': 1.7, 'xi': 0.2}
        else:
            params = params_map

        self.PsydacMapping._expressions = {'x': 'x0 + R0 * (1 + xi * cos(2*pi*x3)) * ( 1 + (1 - x1**2) * delta_x + x1 * epsilon_gs * cos(2*pi*x2 + asin(delta_gs)*x1*sin(2*pi*x2)) )',
                                            'y': 'y0 + R0 * (1 - xi * cos(2*pi*x3)) * (     (1 - x1**2) * delta_y + x1 * epsilon_gs * kappa_gs * sin(2*pi*x2) )',
                                            'z': 'z0 + (x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **params)

        self._params_map = np.array(list(params.values()))
        self._periodic_eta3 = False
        self._pole = True

        super().__init__()

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class Spline(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ijk} c^x_{ijk} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,, 

        y &= \sum_{ijk} c^y_{ijk} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,, 

        z &= \sum_{ijk} c^z_{ijk} N_i(\eta_1) N_j(\eta_2) N_k(\eta_3)\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/xxx.png'''

    def __init__(self, params_map=None):

        # TODO: choose correct params_map
        self._kind_map = 0

        if params_map is None:
            self._params_map = {'Nel': [5, 6, 7], 'p': [2, 3, 4], 'spl_kind': [False, True, False], 'file': None}

            # simple cylinder for testing
            def X(eta1, eta2, eta3): return eta1*np.cos(2*np.pi*eta2) 
            def Y(eta1, eta2, eta3): return eta1*np.sin(2*np.pi*eta2)
            def Z(eta1, eta2, eta3): return 4*eta3

            # project on arbitrary spline space for testing
            self._cx, self._cy, self._cz = interp_mapping(self._params_map['Nel'], self._params_map['p'], self._params_map['spl_kind'], X, Y, Z)

        else:
            self._params_map = params_map
            with h5py.File(params_map['file'], 'r') as handle:

                # print(f'Available keys: {tuple(handle.keys())}')
                self._cx = handle['cx'][:]
                self._cy = handle['cy'][:]
                self._cz = handle['cz'][:]

        _float_params = np.array([])

        self._periodic_eta3 = self._params_map['spl_kind'][-1] # Set by the user.

        if np.all(self.cx[0, :, 0] == self.cx[0, 0, 0]):
            self._pole = True
        else:
            self._pole = False

        super().__init__()

        # reset params_map numpy array
        self._params_map = _float_params

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return None

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class PoloidalSplineStraight(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ij} c^x_{ij} N_i(\eta_1) N_j(\eta_2) \,, 

        y &= \sum_{ij} c^y_{ij} N_i(\eta_1) N_j(\eta_2) \,, 

        z &= L_z\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/xxx.png'''

    def __init__(self, params_map=None):

        self._kind_map = 1

        if params_map is None:
            raise ValueError('Must pass parameters!')
        else:
            with h5py.File(params_map['file'], 'r') as handle:
                self._cx = handle['cx'][:]
                self._cy = handle['cy'][:]

        _float_params = np.array([params_map['Lz']])

        self._periodic_eta3 = False

        assert self._cx.ndim == 2 and self._cy.ndim == 2

        if np.all(self._cx[0, :] == self._cx[0, 0]):
            self._pole = True
        else:
            self._pole = False

        self._cx = self._cx[:, :, None]
        self._cy = self._cy[:, :, None]
        self._cz = np.zeros((1, 1, 1), dtype=float)

        super().__init__()

        # reset params_map numpy array
        self._params_map = _float_params

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return None

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class PoloidalSplineToroidal(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \cos(2\pi\eta_3)  \,, 

        y &= \sum_{ij} c^{y}_{ij} N_i(\eta_1) N_j(\eta_2) \,, 

        z &= \sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \sin(2\pi\eta_3) \,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/xxx.png'''

    def __init__(self, params_map=None):

        self._kind_map = 2

        if params_map is None:
            raise ValueError('Must pass parameters!')
        else:
            with h5py.File(params_map['file'], 'r') as handle:
                self._cx = handle['cx'][:]
                self._cy = handle['cy'][:]

        _float_params = np.array([])

        self._periodic_eta3 = True

        assert self._cx.ndim == 2 and self._cy.ndim == 2

        if np.all(self._cx[0, :] == self._cx[0, 0]):
            self._pole = True
        else:
            self._pole = False

        self._cx = self._cx[:, :, None]
        self._cy = self._cy[:, :, None]
        self._cz = np.zeros((1, 1, 1), dtype=float)

        super().__init__()

        # reset params_map numpy array
        self._params_map = _float_params

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return None

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class PoloidalSplineCylinder(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ij} c^x_{ij} N_i(\eta_1) N_j(\eta_2) \quad \approx \quad a\,\eta_1\cos(2\pi\eta_2) + R_0 \,, 

        y &= \sum_{ij} c^y_{ij} N_i(\eta_1) N_j(\eta_2) \quad \approx \quad a\,\eta_1\sin(2\pi\eta_2)\,, 

        z &= 2\pi R_0\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/xxx.png'''

    def __init__(self, params_map=None):

        self._kind_map = 1

        if params_map is None:
            self._params_map = {'a': 1., 'R0': 3., 'Nel': [
                8, 24], 'p': [2, 2], 'spl_kind': [False, True]}
        else:
            self._params_map = params_map

        # parameter values needed or evaluation
        _float_params = np.array([2*np.pi*self.params_map['R0']])

        self._periodic_eta3 = False

        def X(s, chi): return self.params_map['a']*s*np.cos(2*np.pi*chi) + self.params_map['R0']
        def Y(s, chi): return self.params_map['a']*s*np.sin(2*np.pi*chi)

        self._cx, self._cy = interp_mapping(
            self.params_map['Nel'], self.params_map['p'], self.params_map['spl_kind'], X, Y)

        # make sure that control points at pole are all the same
        self._cx[0] = self.params_map['R0']
        self._cy[0] = 0.

        self._pole = True

        self._cx = self.cx[:, :, None]
        self._cy = self.cy[:, :, None]
        self._cz = np.zeros((1, 1, 1), dtype=float)
        
        super().__init__()
        
        # reset params_map numpy array
        self._params_map = _float_params

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return None

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


class PoloidalSplineTorus(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \cos(2\pi\eta_3) \quad \approx \quad [a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0]\cos(2\pi\eta_3) \,, 

        y &= \sum_{ij} c^{y}_{ij} N_i(\eta_1) N_j(\eta_2) \quad \approx \quad a\,\eta_1\sin(2\pi\theta(\eta_1, \eta_2))\,, 

        z &= \sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \sin(2\pi\eta_3) \quad \approx \quad [a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0]\sin(2\pi\eta_3)\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/xxx.png'''

    def __init__(self, params_map=None):

        self._kind_map = 2

        if params_map is None:
            self._params_map = {'a': 1., 'R0': 3., 'Nel': [8, 24], 'p': [
                2, 2], 'spl_kind': [False, True], 'coordinates': 'straight'}
        else:
            self._params_map = params_map

        # parameter values needed or evaluation
        _float_params = np.array([])

        self._periodic_eta3 = True

        def R(s, chi): return self._params_map['a']*s*np.cos(theta(
            s, chi, self._params_map['a'], self._params_map['R0'], self._params_map['coordinates'])) + self._params_map['R0']
        def Y(s, chi): return self._params_map['a']*s*np.sin(theta(
            s, chi, self._params_map['a'], self._params_map['R0'], self._params_map['coordinates']))

        self._cx, self._cy = interp_mapping(
            self._params_map['Nel'], self._params_map['p'], self._params_map['spl_kind'], R, Y)

        # make sure that control points at pole are all the same
        self._cx[0] = self._params_map['R0']
        self._cy[0] = 0.

        self._pole = True

        self._cx = self._cx[:, :, None]
        self._cy = self._cy[:, :, None]
        self._cz = np.zeros((1, 1, 1), dtype=float)

        super().__init__()

        # reset params_map numpy array
        self._params_map = _float_params

    @property
    def kind_map(self):
        return self._kind_map

    @property
    def params_map(self):
        return self._params_map

    @property
    def F_psy(self):
        return None

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3


