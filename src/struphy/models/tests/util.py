import copy
import os
import struphy
from struphy.main import main
from struphy.io.setup import descend_options_dict

libpath = struphy.__path__[0]


def call_model(key, val, map_and_equil, Tend=None):
    '''Does the options testing of one model.

    Parameters
    ----------
    key : str
        Model name.

    val : struphy.model.base.StruphyModel
        Model class.

    map_and_equil : tuple[str]
        Name of mapping and MHD equilibirum.
        
    Tend : float
        End time of simulation other than default.
    '''

    d_opts = {'em_fields': [], 'fluid': {}, 'kinetic': {}}

    parameters = val.generate_default_parameter_file(save=False)
    # set mapping and mhd equilibirum
    parameters['geometry']['type'] = map_and_equil[0]
    parameters['geometry'][map_and_equil[0]] = {}

    parameters['mhd_equilibrium']['type'] = map_and_equil[1]
    parameters['mhd_equilibrium'][map_and_equil[1]] = {}

    # find out the em_fields options of the model
    if 'em_fields' in parameters:
        if 'options' in parameters['em_fields']:
            # create the default options parameters
            d_default = parameters['em_fields']['options']

            # create a list of parameter dicts for the different options
            descend_options_dict(val.options()['em_fields']['options'],
                                 d_opts['em_fields'], d_default=d_default)

    for name in val.species()['fluid']:
        # find out the fluid options of the model
        if 'options' in parameters['fluid'][name]:

            # create the default options parameters
            d_default = parameters['fluid'][name]['options']

            d_opts['fluid'][name] = []

            # create a list of parameter dicts for the different options
            descend_options_dict(val.options()['fluid'][name]['options'],
                                 d_opts['fluid'][name], d_default=d_default)

    for name in val.species()['kinetic']:
        # find out the kinetic options of the model
        if 'options' in parameters['kinetic'][name]:

            # create the default options parameters
            d_default = parameters['kinetic'][name]['options']

            d_opts['kinetic'][name] = []

            # create a list of parameter dicts for the different options
            descend_options_dict(val.options()['kinetic'][name]['options'],
                                 d_opts['kinetic'][name], d_default=d_default)

    path_out = os.path.join(libpath, 'io/out/test_' + key)

    # store default options
    test_list = []
    if 'options' in val.options()['em_fields']:
        test_list += [parameters['em_fields']['options']]
    if 'fluid' in parameters:
        for species in parameters['fluid']:
            if 'options' in val.options()['fluid'][species]:
                test_list += [parameters['fluid'][species]['options']]
    if 'kinetic' in parameters:
        for species in parameters['kinetic']:
            if 'options' in val.options()['kinetic'][species]:
                test_list += [parameters['kinetic'][species]['options']]

    params_default = copy.deepcopy(parameters)

    if Tend is not None:
        parameters['time']['Tend'] = Tend
        main(key, parameters, path_out, save_step=int(Tend/parameters['time']['dt']))
        return
    else:
        # run with default
        main(key, parameters, path_out)
    

    # run available options (if present)
    if len(d_opts['em_fields']) > 0:
        for opts_dict in d_opts['em_fields']:
            parameters = copy.deepcopy(params_default)
            for opt in opts_dict:

                parameters['em_fields']['options'] = opt

                # test only if not aready tested
                if any([opt == i for i in test_list]):
                    continue
                else:
                    test_list += [opt]
                    main(key, parameters, path_out)

    if len(d_opts['fluid']) > 0:
        for species, opts_dicts in d_opts['fluid'].items():
            for opts_dict in opts_dicts:
                parameters = copy.deepcopy(params_default)
                for opt in opts_dict:

                    parameters['fluid'][species]['options'] = opt

                    # test only if not aready tested
                    if any([opt == i for i in test_list]):
                        continue
                    else:
                        test_list += [opt]
                        main(key, parameters, path_out)

    if len(d_opts['kinetic']) > 0:
        for species, opts_dicts in d_opts['kinetic'].items():
            for opts_dict in opts_dicts:
                parameters = copy.deepcopy(params_default)
                for opt in opts_dict:

                    parameters['kinetic'][species]['options'] = opt

                    # test only if not aready tested
                    if any([opt == i for i in test_list]):
                        continue
                    else:
                        test_list += [opt]
                        main(key, parameters, path_out)
