import argparse
import copy
import os

import standard_geometries as standard_geometries
import standard_mhd_equilibrium as standard_mhd_equils
import yaml

import struphy
import struphy.utils.utils as utils

# Parameters
libpath = struphy.__path__[0]

state = utils.read_state()
i_path = state["i_path"]
o_path = state["o_path"]


# Read the original YAML files
def load_yaml_files(file_paths):
    return {
        os.path.basename(path).replace(".yml", ""): yaml.safe_load(open(path))
        for path in file_paths
    }


standard_params_dir = os.path.join(libpath, "io/inp/standard_parameters")

yaml_files = [
    f"{standard_params_dir}/likwid_config.yml",
    f"{standard_params_dir}/params_Vlasov.yml",
    f"{standard_params_dir}/params_Maxwell.yml",
    f"{standard_params_dir}/params_LinearMHD.yml",
    f"{standard_params_dir}/params_ShearAlfven.yml",
    f"{standard_params_dir}/params_LinearMHDVlasovCC.yml",
    f"{standard_params_dir}/params_LinearMHDDriftkineticCC.yml",
    f"{standard_params_dir}/params_GuidingCenter.yml",
]

yaml_data = load_yaml_files(yaml_files)


def save_parameter_file(parameters, filename):
    with open(filename, "w") as file:
        yaml.safe_dump(parameters, file, default_flow_style=False)
    print(f"Updated '{filename}'")


def apply_modifications(parameters, modifications):
    for key, value in modifications.items():
        keys = key.split(".")
        current_dict = parameters
        for k in keys[:-1]:
            current_dict = current_dict.setdefault(k, {})
        current_dict[keys[-1]] = value


def generate_parameter_files(
    base_parameters, modifications_list, filename_template, params_prefix="params_",
):
    for modifications in modifications_list:
        parameters = copy.deepcopy(base_parameters)
        apply_modifications(parameters, modifications)

        # Prepare filename variables
        filename_vars = {}
        for k, v in modifications.items():
            if isinstance(v, list):
                for i, vi in enumerate(v):
                    filename_vars[f"{k.replace('.', '_')}_{i}"] = vi
            else:
                filename_vars[k.replace(".", "_")] = v
        filename = filename_template.format(**filename_vars)

        # Set the projectname in the parameter file
        param_filename = os.path.basename(filename)
        projectname = param_filename.replace(params_prefix, "").replace(".yml", "")
        parameters["setup"]["projectname"] = projectname

        # Save the simulation name
        save_parameter_file(parameters, filename)


def mpi_scan(
    base_parameters,
    mpi_values,
    filename_template,
    extra_modifications=None,
    params_prefix="params_",
):
    """
    Generate parameter files for different MPI values.

    Parameters:
    - base_parameters: dict, base parameters of the model
    - mpi_values: list, list of MPI values to scan
    - filename_template: str, template for the output filename
    - extra_modifications: dict, additional modifications to apply
    """
    modifications_list = []
    for mpi in mpi_values:
        modifications = {"setup.mpi": mpi, "grid.Nclones": 1}
        if extra_modifications:
            modifications.update(extra_modifications)
        modifications_list.append(modifications)
    generate_parameter_files(
        base_parameters, modifications_list, filename_template, params_prefix,
    )

def mpi_nclones_scan(
    base_parameters,
    mpi_values,
    filename_template,
    extra_modifications=None,
    params_prefix="params_",
):
    """
    Generate parameter files for different MPI values.

    Parameters:
    - base_parameters: dict, base parameters of the model
    - mpi_values: list, list of MPI values to scan
    - filename_template: str, template for the output filename
    - extra_modifications: dict, additional modifications to apply
    """
    modifications_list = []
    for mpi in mpi_values:
        Nclones = int(mpi / 72)
        modifications = {"setup.mpi": mpi, "grid.Nclones": Nclones}
        if extra_modifications:
            modifications.update(extra_modifications)
        modifications_list.append(modifications)
    generate_parameter_files(
        base_parameters, modifications_list, filename_template, params_prefix,
    )



def generate_ptest_params():

    parser = argparse.ArgumentParser(
        description="Generate parameters for performance testing simulations.",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="params_",
        help="Path matching the params to run",
    )

    args = parser.parse_args()

    params_prefix = args.prefix

    # mpi_values = [8, 16, 32, 48, 64, 72]
    mpi_values = [72 * _i for _i in [1,2,4,8,16,32,64]]
    # Save LIKWID configuration
    for mpi_val in mpi_values:
        parameter_file = f"{i_path}/likwid_config_mpi{mpi_val}.yml"
        nperdomain = min(int(mpi_val / 2), 36)
        for param in yaml_data["likwid_config"]["likwid-mpirun"]["options"]:
            if type(param) == dict:
                if "-nperdomain" in param.keys():
                    param["-nperdomain"] = f"S:{nperdomain}"
        save_parameter_file(yaml_data["likwid_config"], parameter_file)

    # Define models and their parameters
    for model in [
                # "Maxwell",
                # "Vlasov",
                'LinearMHDDriftkineticCC',
                'LinearMHDVlasovCC',
                ]:
        # base_params = yaml_data[f"params_{model}"]
        base_params = copy.deepcopy(yaml_data[f"params_{model}"])
        base_params["setup"] = {}
        base_params["setup"]["model"] = model
        # MPI values to scan

        filename_template = f"{i_path}/{params_prefix}{model}_Nclones{{grid_Nclones}}_mpi{{setup_mpi}}.yml"
        extra_modifications = None
        mpi_scan(
            base_params,
            mpi_values,
            filename_template,
            extra_modifications,
            params_prefix,
        )
        
        mpi_nclones_scan(
            base_params,
            mpi_values,
            filename_template,
            extra_modifications,
            params_prefix,
        )
    # grid value scan
    # mpi_values = [72]
    # grid_values = [[16, 16, 16], [32, 32, 32], [64, 64, 64]]

    # Define models and their parameters
    # for model in ["Vlasov", 'Maxwell']:
    #     base_params = yaml_data[f"params_{model}"]

    #     # Filename template including mpi and grid values
    #     filename_template = f"{i_path}/params_{model}_grid{{grid_Nel_0}}x{{grid_Nel_1}}x{{grid_Nel_2}}_mpi{{mpi}}.yml"

    #     for grid in grid_values:
    #         extra_modifications = {'grid.Nel': grid}
    #         mpi_scan(base_params, mpi_values, filename_template, extra_modifications)


if __name__ == "__main__":
    generate_ptest_params()
    