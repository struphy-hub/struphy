import copy
import os

import yaml

import struphy
from struphy.console.main import recursive_get_files
from struphy.io.setup import descend_options_dict
from struphy.main import main
from struphy.models.base import StruphyModel

libpath = struphy.__path__[0]


def call_model(
    model_name: str,
    model: StruphyModel,
    map_and_equil: tuple,
    *,
    Tend: float = None,
    verbose: bool = True,
    comm=None,
    num_clones: int = 1,
    verification: bool = False,
    show_plots: bool = False,
):
    """Does testing of one model, either all options or verification.

    Parameters
    ----------
    model_name : str
        Model name.

    model : StruphyModel
        Instance of model base class.

    map_and_equil : tuple[str]
        Name of mapping and MHD equilibirum.

    num_clones : int
        Number of domain clones.

    Tend : float
        End time of simulation other than default.

    verbose : bool
        Show info on screen.

    verification : bool
        Do verifiaction runs.

    show_plots: bool
        Show plots of verification tests.
    """

    rank = comm.Get_rank()

    if verification:
        ver_path = os.path.join(libpath, "io", "inp", "verification")
        yml_files = recursive_get_files(ver_path, contains=(model_name,))
        if len(yml_files) == 0:
            if rank == 0:
                print(f"\nVerification run not started: no .yml files for model {model_name} in {ver_path}.")
            return
        params_list = []
        paths_out = []
        py_scripts = []
        if rank == 0:
            print("\nThe following verification tests will be run:")
        for n, file in enumerate(yml_files):
            ref = file.split("_")[0]
            if ref != model_name:
                continue
            if rank == 0:
                print(file)
            with open(os.path.join(ver_path, file)) as tmp:
                params_list += [yaml.load(tmp, Loader=yaml.FullLoader)]
            paths_out += [os.path.join(libpath, "io", "out", "verification", model_name, f"{n + 1}")]

        # python scripts for data verification after the run below
        from struphy.models.tests import verification as verif

        tname = file.split(".")[0]
        try:
            py_scripts += [getattr(verif, tname)]
        except:
            if rank == 0:
                print(f"A Python script for {model_name} is missing in models/tests/verification.py, exiting ...")
            exit()
    else:
        params = model.generate_default_parameter_file(save=False)
        params["geometry"]["type"] = map_and_equil[0]
        params["geometry"][map_and_equil[0]] = {}
        params["fluid_background"][map_and_equil[1]] = {}
        params_list = [params]
        paths_out = [os.path.join(libpath, "io/out/test_" + model_name)]
        py_scripts = [None]

    # run model
    for parameters, path_out, py_script in zip(params_list, paths_out, py_scripts):
        if Tend is not None:
            parameters["time"]["Tend"] = Tend
            if rank == 0:
                print_test_params(parameters)
            main(
                model_name,
                parameters,
                path_out,
                save_step=int(
                    Tend / parameters["time"]["dt"],
                ),
                num_clones=num_clones,
                verbose=verbose,
            )
            return
        else:
            # run with default
            if rank == 0:
                print_test_params(parameters)
            main(
                model_name,
                parameters,
                path_out,
                num_clones=num_clones,
                verbose=verbose,
            )

        # run the verification script on the output data
        if verification:
            py_script(
                path_out,
                rank,
                show_plots=show_plots,
            )

    # run available options (if present)
    if not verification:
        d_opts, test_list = find_model_options(model, parameters)
        params_default = copy.deepcopy(parameters)

        if len(d_opts["em_fields"]) > 0:
            for opts_dict in d_opts["em_fields"]:
                parameters = copy.deepcopy(params_default)
                for opt in opts_dict:
                    parameters["em_fields"]["options"] = opt

                    # test only if not aready tested
                    if any([opt == i for i in test_list]):
                        continue
                    else:
                        test_list += [opt]
                        if rank == 0:
                            print_test_params(parameters)
                        main(
                            model_name,
                            parameters,
                            path_out,
                            num_clones=num_clones,
                            verbose=verbose,
                        )

        if len(d_opts["fluid"]) > 0:
            for species, opts_dicts in d_opts["fluid"].items():
                for opts_dict in opts_dicts:
                    parameters = copy.deepcopy(params_default)
                    for opt in opts_dict:
                        parameters["fluid"][species]["options"] = opt

                        # test only if not aready tested
                        if any([opt == i for i in test_list]):
                            continue
                        else:
                            test_list += [opt]
                            if rank == 0:
                                print_test_params(parameters)
                            main(
                                model_name,
                                parameters,
                                path_out,
                                num_clones=num_clones,
                                verbose=verbose,
                            )

        if len(d_opts["kinetic"]) > 0:
            for species, opts_dicts in d_opts["kinetic"].items():
                for opts_dict in opts_dicts:
                    parameters = copy.deepcopy(params_default)
                    for opt in opts_dict:
                        parameters["kinetic"][species]["options"] = opt

                        # test only if not aready tested
                        if any([opt == i for i in test_list]):
                            continue
                        else:
                            test_list += [opt]
                            if rank == 0:
                                print_test_params(parameters)
                            main(
                                model_name,
                                parameters,
                                path_out,
                                num_clones=num_clones,
                                verbose=verbose,
                            )


def print_test_params(parameters):
    print("\nOptions of this test run:")
    for k, v in parameters.items():
        if k == "em_fields":
            if "options" in v:
                print("\nem_fields:")
                for kk, vv in v["options"].items():
                    print(" " * 4, kk)
                    print(" " * 8, vv)
        elif k in ("fluid", "kinetic"):
            print(f"\n{k}:")
            for kk, vv in v.items():
                if "options" in vv:
                    for kkk, vvv in vv["options"].items():
                        print(" " * 4, kkk)
                        print(" " * 8, vvv)


def find_model_options(
    model: StruphyModel,
    parameters: dict,
):
    """Find all options of a model and store them in d_opts.
    The default options are also stored in test_list."""

    d_opts = {"em_fields": [], "fluid": {}, "kinetic": {}}
    # find out the em_fields options of the model
    if "em_fields" in parameters:
        if "options" in parameters["em_fields"]:
            # create the default options parameters
            d_default = parameters["em_fields"]["options"]

            # create a list of parameter dicts for the different options
            descend_options_dict(
                model.options()["em_fields"]["options"],
                d_opts["em_fields"],
                d_default=d_default,
            )

    for name in model.species()["fluid"]:
        # find out the fluid options of the model
        if "options" in parameters["fluid"][name]:
            # create the default options parameters
            d_default = parameters["fluid"][name]["options"]

            d_opts["fluid"][name] = []

            # create a list of parameter dicts for the different options
            descend_options_dict(
                model.options()["fluid"][name]["options"],
                d_opts["fluid"][name],
                d_default=d_default,
            )

    for name in model.species()["kinetic"]:
        # find out the kinetic options of the model
        if "options" in parameters["kinetic"][name]:
            # create the default options parameters
            d_default = parameters["kinetic"][name]["options"]

            d_opts["kinetic"][name] = []

            # create a list of parameter dicts for the different options
            descend_options_dict(
                model.options()["kinetic"][name]["options"],
                d_opts["kinetic"][name],
                d_default=d_default,
            )

    # store default options
    test_list = []
    if "options" in model.options()["em_fields"]:
        test_list += [parameters["em_fields"]["options"]]
    if "fluid" in parameters:
        for species in parameters["fluid"]:
            if "options" in model.options()["fluid"][species]:
                test_list += [parameters["fluid"][species]["options"]]
    if "kinetic" in parameters:
        for species in parameters["kinetic"]:
            if "options" in model.options()["kinetic"][species]:
                test_list += [parameters["kinetic"][species]["options"]]

    return d_opts, test_list
