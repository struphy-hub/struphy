import numpy as np
import h5py

from struphy.fields_background.mhd_equil.base import NumericalMHDequilibrium
from struphy.geometry.base import interp_mapping
from struphy.geometry.domains import Spline

from gvec_to_python.reader.gvec_reader import create_GVEC_json
from gvec_to_python import GVEC


class EQDSKequilibrium(NumericalMHDequilibrium):

    def __init__(self, eqdsk_file):

        self._domain = 99

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    def b2_1(self, eta1, eta2, eta3, squeeze_out=True):
        """First 2-form component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    def b2_2(self, eta1, eta2, eta3, squeeze_out=True):
        """Second 2-form component (eta2) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    def b2_3(self, eta1, eta2, eta3, squeeze_out=True):
        """Third 2-form component (eta3) of magnetic field on logical cube [0, 1]^3.
        """
        pass

    def j2_1(self, eta1, eta2, eta3, squeeze_out=True):
        """First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def j2_2(self, eta1, eta2, eta3, squeeze_out=True):
        """Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def j2_3(self, eta1, eta2, eta3, squeeze_out=True):
        """Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        """
        pass

    def p0(self, eta1, eta2, eta3, squeeze_out=True):
        """0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        pass

    def n0(self, eta1, eta2, eta3, squeeze_out=True):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        pass


class GVECequilibrium(NumericalMHDequilibrium):
    '''Interface to `gvec_to_python <https://gitlab.mpcdf.mpg.de/spossann/gvec_to_python>`_.
    
    Parameters
    ----------
    params: dict
        Parameters that characterize the MHD equilibrium.

        * dat_file : str
            Absolute path to .dat file (with extension).    
        * json_file : str
            Either absolute path to existing .json file (with extension), or filename (no path, with extension)
            of the .json file to be created and stored in <struphy_path>/fields_background/mhd_equil/gvec/json/.
        * use_pest : bool
            Whether to use straigh-field line coordinates (PEST).
        * Nel : tuple[int]
            Number of cells in each direction used for interpolation of the mapping.   
        * p : tuple[int]
            Spline degree in each direction used for interpolation of the mapping.'''

    def __init__(self, params=None):

        # set default parameters
        if params is None:
            params = {'rel_path': True,
                      'dat_file': '/ellipstell_v2/newBC_E1D6_M6N6/GVEC_ELLIPSTELL_V2_State_0000_00200000.dat',
                      'json_file': None,
                      'use_pest': False,
                      'Nel': (16, 16, 16),
                      'p': (3, 3, 3),}
        # or check if given parameter dictionary is complete
        else:
            assert 'rel_path' in params
            assert 'dat_file' in params
            assert 'json_file' in params
            assert 'use_pest' in params
            assert 'Nel' in params
            assert 'p' in params

        if params['dat_file'] is None:

            assert params['json_file'] is not None
            assert params['json_file'][-5:] == '.json'

            if params['rel_path']:
                import struphy as _
                json_file = _.__path__[0] + '/fields_background/mhd_equil/gvec' + params['json_file']
            else:
                json_file = params['json_file']

        else:

            assert params['dat_file'][-4:] == '.dat'

            if params['rel_path']:
                import struphy as _
                dat_file = _.__path__[0] + '/fields_background/mhd_equil/gvec' + params['dat_file']
            else:
                dat_file = params['dat_file']

            json_file = dat_file[:-4] + '.json'
            create_GVEC_json(dat_file, json_file)

        if params['use_pest']:
            mapping = 'unit_pest'
        else:
            mapping = 'unit'

        # gvec object
        self._gvec = GVEC(json_file, mapping=mapping, unit_tor_domain="full", use_pyccel=True)

        # project mapping to splines 
        spl_kind = (False, True, True)
        X = lambda e1, e2, e3: self.gvec.f(e1, e2, e3)[0] 
        Y = lambda e1, e2, e3: self.gvec.f(e1, e2, e3)[1]
        Z = lambda e1, e2, e3: self.gvec.f(e1, e2, e3)[2]
        
        cx, cy, cz = interp_mapping(params['Nel'], params['p'], spl_kind, X, Y, Z) 
        
        # save coeffs in hdf5 file
        _path = _.__path__[0] + '/fields_background/mhd_equil/gvec/output/map_coefs.hdf5'
        _file = h5py.File(_path, 'w')
        _file.create_dataset('cx', data=cx, chunks=True)
        _file.create_dataset('cy', data=cy, chunks=True)
        _file.create_dataset('cz', data=cz, chunks=True)

        # struphy domain object
        params_map = {'file': _path, 'Nel': params['Nel'], 'p': params['p'], 'spl_kind': spl_kind}
        self._domain = Spline(params_map)

        self._params = params

    @property
    def domain(self):
        """ Domain object that characterizes the mapping from the logical to the physical domain.
        """
        return self._domain

    @property
    def gvec(self):
        """ GVEC object.
        """
        return self._gvec

    @property
    def params(self):
        '''Parameters describing the equilibrium.'''
        return self._params

    def b2_1(self, eta1, eta2, eta3, squeeze_out=True):
        """First 2-form component (eta1) of magnetic field on logical cube [0, 1]^3.
        """
        return self.gvec.b2(eta1, eta2, eta3)[0]

    def b2_2(self, eta1, eta2, eta3, squeeze_out=True):
        """Second 2-form component (eta2) of magnetic field on logical cube [0, 1]^3.
        """
        return self.gvec.b2(eta1, eta2, eta3)[1]

    def b2_3(self, eta1, eta2, eta3, squeeze_out=True):
        """Third 2-form component (eta3) of magnetic field on logical cube [0, 1]^3.
        """
        return self.gvec.b2(eta1, eta2, eta3)[2]

    def j2_1(self, eta1, eta2, eta3, squeeze_out=True):
        """First 2-form component (eta1) of current (=curl B) on logical cube [0, 1]^3.
        """
        return self.gvec.j2(eta1, eta2, eta3)[0]

    def j2_2(self, eta1, eta2, eta3, squeeze_out=True):
        """Second 2-form component (eta2) of current (=curl B) on logical cube [0, 1]^3.
        """
        return self.gvec.j2(eta1, eta2, eta3)[0]

    def j2_3(self, eta1, eta2, eta3, squeeze_out=True):
        """Third 2-form component (eta3) of current (=curl B) on logical cube [0, 1]^3.
        """
        return self.gvec.j2(eta1, eta2, eta3)[0]

    def p0(self, eta1, eta2, eta3, squeeze_out=True):
        """0-form equilibrium pressure on logical cube [0, 1]^3.
        """
        return self.gvec.p0(eta1, eta2, eta3)

    def n0(self, eta1, eta2, eta3, squeeze_out=True):
        """0-form equilibrium density on logical cube [0, 1]^3.
        """
        # TODO: which density to set?
        return self.gvec.p0(eta1, eta2, eta3)


