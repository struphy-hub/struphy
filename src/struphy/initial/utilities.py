import os
import yaml
import h5py
import struphy

from struphy.io.output_handling import DataContainer
from struphy.fields_background.mhd_equil.equils import set_defaults


class InitFromOutput:
    r'''Assemble FEEC coefficients array form output files.

    Note
    ----
    In the parameter .yml, use the following in the section ``fluid/<species>``:: or ``em_fields``::

        init :
            type : InitFromOutput 
            InitFromOutput :
                path : 'sim_1'
                path_abs : null
                comps :
                    n3 : False               # components to be initialized 
                    u2 : [True, True, True]  # components to be initialized 
                    p3 : True                # components to be initialized 
    '''

    def __init__(self, derham, name, species, **params):

        libpath = struphy.__path__[0]

        with open(os.path.join(libpath, 'state.yml')) as f:
            state = yaml.load(f, Loader=yaml.FullLoader)

        o_path = state['o_path']

        params_default = {'path': 'sim_1',
                          'path_abs': None,
                          'comps': {'n3': [True],
                                    'u2': [True, True, True],
                                    'p3': [True]}}

        params = set_defaults(params, params_default)

        # absolute path of output data
        if params['path_abs'] is None:
            data_path = os.path.join(o_path, params['path'])
        else:
            data_path = params['path_abs']

        data = DataContainer(data_path, comm=derham.comm)

        if species is None:
            key = 'restart/' + name
        else:
            key = 'restart/' + species + '_' + name

        if isinstance(data.file[key], h5py.Dataset):
            self._vector = data.file[key][-1]

        else:
            self._vector = []

            for n in range(3):
                self._vector += [data.file[key + '/' + str(n + 1)][-1]]

        data.file.close()

    @property
    def vector(self):
        """ vectors from output data
        """

        return self._vector
