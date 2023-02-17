from struphy.geometry.base import Domain, Spline, PoloidalSplineStraight, PoloidalSplineTorus
import numpy as np


class EQDSKTorus(PoloidalSplineTorus):
    '''Mappings constructed via field line tracing from EQDSK data.
    
    .. image:: ../pics/mappings/eqdsk_raw.png

    |

    .. image:: ../pics/mappings/eqdsk.png'''

    def __init__(self, **params):

        from struphy.fields_background.mhd_equil.equils import EQDSKequilibrium

        eqdsk = EQDSKequilibrium(**params)

        new_params = {}
        new_params['cx'] = eqdsk.domain.cx[:, :, 0].squeeze()
        new_params['cy'] = eqdsk.domain.cy[:, :, 0].squeeze()
        new_params['Nel'] = eqdsk.domain.params_map['Nel']
        new_params['p'] = eqdsk.domain.params_map['p']
        new_params['spl_kind'] = eqdsk.domain.params_map['spl_kind']
        new_params['tor_period'] = eqdsk.domain.params_map['tor_period']

        super().__init__(**new_params)


class GVECunit(Spline):
    '''The mapping "f_unit" from `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_, 
    computed by the GVEC MHD equilibirum code.
    
    .. image:: ../pics/mappings/gvec.png'''

    def __init__(self, **params):

        from struphy.fields_background.mhd_equil.equils import GVECequilibrium

        gvec = GVECequilibrium(**params)

        new_params = {}
        new_params['cx'] = gvec.domain.cx
        new_params['cy'] = gvec.domain.cy 
        new_params['cz'] = gvec.domain.cz
        new_params['Nel'] = gvec.domain.params_map['Nel']
        new_params['p'] = gvec.domain.params_map['p']
        new_params['spl_kind'] = gvec.domain.params_map['spl_kind']

        super().__init__(**new_params)


class IGAPolarCylinder(PoloidalSplineStraight):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ij} c^x_{ij} N_i(\eta_1) N_j(\eta_2) \quad \approx \quad a\,\eta_1\cos(2\pi\eta_2)\,, 

        y &= \sum_{ij} c^y_{ij} N_i(\eta_1) N_j(\eta_2) \quad \approx \quad a\,\eta_1\sin(2\pi\eta_2)\,, 

        z &= L_z\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/iga_cylinder.png'''

    def __init__(self, **params):

        from struphy.geometry.base import interp_mapping
        
        # set default 
        params_default = {'Nel': [8, 24], 'p': [2, 3], 'a': 1., 'Lz': 4.}
        
        params_map = Domain.prepare_params_map(params, params_default, return_numpy=False)
        
        # get control points
        def X(eta1, eta2): return params_map['a'] * eta1 * np.cos(2*np.pi * eta2) 
        def Y(eta1, eta2): return params_map['a'] * eta1 * np.sin(2*np.pi * eta2)

        cx, cy = interp_mapping(params_map['Nel'], params_map['p'], [False, True], X, Y)

        # make sure that control points at pole are all the same (eta1=0 there)
        cx[0] = 0.
        cy[0] = 0.

        # add control points to parameters dictionary
        params_map['cx'] = cx
        params_map['cy'] = cy
        
        # add spline types to parameters dictionary
        params_map['spl_kind'] = [False, True]
        
        # remove "a" temporarily from params_map dictionary (is not a parameter of PoloidalSplineStraight)
        a = params_map['a']
        params_map.pop('a')           

        # init base class
        super().__init__(**params_map)
        
        self._params_map['a'] = a


class IGAPolarTorus(PoloidalSplineTorus):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \cos(2\pi\eta_3) \quad \approx \quad [a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0]\cos(2\pi\eta_3) \,, 

        y &= \sum_{ij} c^{y}_{ij} N_i(\eta_1) N_j(\eta_2) \quad \approx \quad a\,\eta_1\sin(2\pi\theta(\eta_1, \eta_2))\,, 

        z &= \sum_{ij} c^{R}_{ij} N_i(\eta_1) N_j(\eta_2) \sin(-2\pi\eta_3) \quad \approx \quad [a\,\eta_1\cos(2\pi\theta(\eta_1, \eta_2)) + R_0]\sin(- 2\pi\eta_3)\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/iga_torus.png'''

    def __init__(self, **params):

        from struphy.geometry.base import interp_mapping
        
        # set default 
        params_default = {'Nel': [8, 24], 'p': [2, 3], 'a': 1., 'R0': 3., 'sfl': False, 'tor_period' : 3}
        
        params_map = Domain.prepare_params_map(params, params_default, return_numpy=False)
        
        # get control points
        if params_map['sfl']:
            def theta(eta1, eta2):
                return 2*np.arctan(np.sqrt((1 + params_map['a'] * eta1 / params_map['R0'])/(1 - params_map['a'] * eta1 / params_map['R0'])) * np.tan(np.pi*eta2))
        else:
            def theta(eta1, eta2):
                return 2*np.pi*eta2

        def R(eta1, eta2): return params_map['a'] * eta1 * np.cos(theta(eta1, eta2)) + params_map['R0']
        def Z(eta1, eta2): return params_map['a'] * eta1 * np.sin(theta(eta1, eta2))
        
        cx, cy = interp_mapping(params_map['Nel'], params_map['p'], [False, True], R, Z)

        # make sure that control points at pole are all the same (eta1=0 there)
        cx[0] = params_map['R0']
        cy[0] = 0.
        
        # add control points to parameters dictionary
        params_map['cx'] = cx
        params_map['cy'] = cy
        
        # add spline types to parameters dictionary
        params_map['spl_kind'] = [False, True]
        
        # remove "a", "R0" and "sfl" temporarily from params_map dictionary (is not a parameter of PoloidalSplineTorus)
        a   = params_map['a']
        R0  = params_map['R0']
        sfl = params_map['sfl']
        
        params_map.pop('a')
        params_map.pop('R0')
        params_map.pop('sfl')

        # init base class
        super().__init__(**params_map)
        
        self._params_map['a']   = a
        self._params_map['R0']  = R0
        self._params_map['sfl'] = sfl


class Cuboid(Domain):
    r'''
    .. math:: 

        F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= l_1 + (r_1 - l_1)\,\eta_1\,, 

        y &= l_2 + (r_2 - l_2)\,\eta_2\,, 

        z &= l_3 + (r_3 - l_3)\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/cuboid.png'''

    def __init__(self, **params):

        self._kind_map = 10
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'l1': 0., 'r1': 2., 'l2': 0.,
                          'r2': 3., 'l3': 0., 'r3': 6.}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': 'l1 + (r1 - l1)*x1',
                                           'y': 'l2 + (r2 - l2)*x2',
                                           'z': 'l3 + (r3 - l3)*x3'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
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
    def params_numpy(self):
        return self._params_numpy

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

    def __init__(self, **params):

        self._kind_map = 11
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'Lx': 2., 'Ly': 3., 'alpha': 0.1, 'Lz': 6.}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1))',
                                           'y': 'Ly*(x2 + alpha*sin(2*pi*x2))',
                                           'z': 'Lz*x3'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
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
    def params_numpy(self):
        return self._params_numpy

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

    def __init__(self, **params):

        self._kind_map = 12
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'Lx' : 2., 'Ly' : 3., 'alpha' : 0.1, 'Lz' : 6.}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)
        
        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': 'Lx*(x1 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                           'y': 'Ly*(x2 + alpha*sin(2*pi*x1)*sin(2*pi*x2))',
                                           'z': 'Lz*x3'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
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
    def params_numpy(self):
        return self._params_numpy
    
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
        x &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2)\,, 

        y &= \left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin(2\pi\,\eta_2)\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/hollow_cylinder.png'''

    def __init__(self, **params):

        self._kind_map = 20
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'a1': 0.2, 'a2': 1., 'Lz': 4.}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': '(a1 + (a2 - a1)*x1)*cos(2*pi*x2)',
                                           'y': '(a1 + (a2 - a1)*x1)*sin(2*pi*x2)',
                                           'z': 'Lz*x3'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
        self._periodic_eta3 = False
        
        if self.params_map['a1'] == 0.:
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
    def params_numpy(self):
        return self._params_numpy

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
        x &= r_x\,\eta_1^s\cos(2\pi\,\eta_2)\,, 

        y &= r_y\,\eta_1^s\sin(2\pi\,\eta_2)\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/pow_elliptic_cyl.png'''

    def __init__(self, **params):

        self._kind_map = 21
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'rx': 1., 'ry': 2., 'Lz': 6., 's': 0.5}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': '(x1**s) * rx * cos(2*pi*x2)',
                                           'y': '(x1**s) * ry * sin(2*pi*x2)',
                                           'z': '(x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
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
    def params_numpy(self):
        return self._params_numpy

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
        x &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2)+R_0\rbrace\cos(2\pi\,\eta_3 / n)\,, 

        y &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos(2\pi\,\eta_2)+R_0\rbrace\sin(-2\pi\,\eta_3 / n) \,, 

        z &= \,\,\,\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin(2\pi\,\eta_2) \,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/hollow_torus.png'''

    def __init__(self, **params):

        self._kind_map = 22
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'a1': 0.2, 'a2': 1., 'R0': 3., 'tor_period': 3}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * cos(2*pi*x3 / tor_period)',
                                           'y': '((a1 + (a2 - a1)*x1)*cos(2*pi*x2) + R0) * sin(-2*pi*x3 / tor_period)',
                                           'z': '( a1 + (a2 - a1)*x1)*sin(2*pi*x2)'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
        self._periodic_eta3 = True
        
        if self.params_map['a1'] == 0.:
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
    def params_numpy(self):
        return self._params_numpy

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3
    
    
class HollowTorusStraightFieldLine(Domain):
    r'''
    .. math:: 

        &F: (\eta_1, \eta_2, \eta_3) \mapsto (x, y, z) \textnormal{ as } \left\{\begin{aligned}
        x &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos\left[\theta(\eta_1,\eta_2)\right]+R_0\rbrace\cos(2\pi\,\eta_3 / n)\,, 

        y &=  \,\,\,\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\sin\left[\theta(\eta_1,\eta_2)\right]\,, 

        z &= \lbrace\left[\,a_1 + (a_2-a_1)\,\eta_1\,\right]\cos\left[\theta(\eta_1,\eta_2)\right]+R_0\rbrace\sin(-2\pi\,\eta_3 / n)\,,
        \end{aligned}\right.
        
        &\theta(\eta_1,\eta_2) = 2\arctan\left[\sqrt{\frac{1 + \epsilon(\eta_1)}{1 - \epsilon(\eta_1)}}\,\tan\left(\pi\,\eta_2\right)\right]\,,
        
        &\epsilon(\eta_1) = \frac{a_1 + (a_2-a_1)\,\eta_1}{R_0}\,.

    .. image:: ../pics/mappings/hollow_torus_sfl.png'''

    def __init__(self, **params):

        self._kind_map = 23
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'a1': 0.2, 'a2': 1., 'R0': 3., 'tor_period': 3}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self._F_psy = None # TODO

        # periodicity in eta3-direction and pole at eta1=0
        self._periodic_eta3 = True
        
        if self.params_map['a1'] == 0.:
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
    def params_numpy(self):
        return self._params_numpy

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
        x &= r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\eta_1^2)r_x\Delta\,, 

        y &= r_y\,\eta_1\sin(2\pi\,\eta_2)\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/shafranov_shift.png'''

    def __init__(self, **params):

        self._kind_map = 30
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'rx': 1., 'ry': 1., 'Lz': 4., 'delta': 0.2}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': '(x1*rx) * cos(2*pi*x2) + (1-x1**2) * rx * delta',
                                           'y': '(x1*ry) * sin(2*pi*x2)',
                                           'z': 'x3*Lz'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
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
    def params_numpy(self):
        return self._params_numpy

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
        x &= r_x\,\eta_1\cos(2\pi\,\eta_2)+(1-\sqrt \eta_1)r_x\Delta\,, 

        y &= r_y\,\eta_1\sin(2\pi\,\eta_2)\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/shafranov_sqrt.png'''

    def __init__(self, **params):

        self._kind_map = 31
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'rx': 1., 'ry': 1., 'Lz': 4., 'delta': 0.2}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': '(x1*rx) * cos(2*pi*x2) + (1-sqrt(x1)) * rx * delta',
                                           'y': '(x1*ry) * sin(2*pi*x2)',
                                           'z': '(x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
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
    def params_numpy(self):
        return self._params_numpy

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
        x &= R_0\left[1 + (1 - \eta_1^2)\Delta_x + \eta_1\epsilon\cos(2\pi\,\eta_2 + \arcsin(\delta)\eta_1\sin(2\pi\,\eta_2)) \right]\,, 

        y &= R_0\left[    (1 - \eta_1^2)\Delta_y + \eta_1\epsilon\kappa\sin(2\pi\,\eta_2)\right]\,, 

        z &= L_z\,\eta_3\,.
        \end{aligned}\right.

    .. image:: ../pics/mappings/shafranov_dshaped.png'''

    def __init__(self, **params):

        self._kind_map = 32
        
        # set default parameters and remove wrong/not needed keys
        params_default = {'R0': 2., 'Lz': 3., 'delta_x': 0.1, 'delta_y': 0.,
                          'delta_gs': 0.33, 'epsilon_gs': 0.32, 'kappa_gs': 1.7}
        
        self._params_map, self._params_numpy = Domain.prepare_params_map(params, params_default)

        # create interface to Psydac mappings
        self.PsydacMapping._expressions = {'x': 'R0 * ( 1 + (1 - x1**2) * delta_x + x1 * epsilon_gs * cos(2*pi*x2 + asin(delta_gs)*x1*sin(2*pi*x2)) )',
                                           'y': 'R0 * (     (1 - x1**2) * delta_y + x1 * epsilon_gs * kappa_gs * sin(2*pi*x2) )',
                                           'z': '(x3*Lz)'}
        self._F_psy = self.PsydacMapping('F', **self._params_map)

        # periodicity in eta3-direction and pole at eta1=0
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
    def params_numpy(self):
        return self._params_numpy

    @property
    def F_psy(self):
        return self._F_psy

    @property
    def pole(self):
        return self._pole

    @property
    def periodic_eta3(self):
        return self._periodic_eta3
