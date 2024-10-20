import struphy
import copy
import yaml
import os

import standard_geometries as standard_geometries
import standard_mhd_equilibrium as standard_mhd_equils

# Parameters
libpath = struphy.__path__[0]
with open(os.path.join(libpath, "state.yml")) as f:
    state = yaml.load(f, Loader=yaml.FullLoader)
i_path = state["i_path"]
o_path = state["o_path"]


# Read the original YAML files
def load_yaml_files(file_paths):
    return {
        os.path.basename(path).replace(".yml", ""): yaml.safe_load(open(path))
        for path in file_paths
    }


standard_params_dir = os.path.join(
    libpath, "io/inp/standard_parameters"
)

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


# Utility functions
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


def generate_parameter_files(base_parameters, modifications_list, filename_template):
    for modifications in modifications_list:
        parameters = copy.deepcopy(base_parameters)
        apply_modifications(parameters, modifications)
        filename_vars = {k.replace(".", "_"): v for k, v in modifications.items()}
        filename = filename_template.format(**filename_vars)
        save_parameter_file(parameters, filename)


# New function for MPI scan
def mpi_scan(base_parameters, mpi_values, filename_template, extra_modifications=None):
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
        modifications = {"mpi": mpi}
        if extra_modifications:
            modifications.update(extra_modifications)
        modifications_list.append(modifications)
    generate_parameter_files(base_parameters, modifications_list, filename_template)


mpi_values = [8, 16, 32, 48, 64, 72]
# Save LIKWID configuration
for mpi_val in mpi_values:
    parameter_file = f"{i_path}/likwid_config_mpi{mpi_val}.yml"
    nperdomain = min(int(mpi_val / 2), 36)
    for param in yaml_data["likwid_config"]['likwid-mpirun']['options']:
        if type(param) == dict:
            if '-nperdomain' in param.keys():
                param['-nperdomain'] = f"S:{nperdomain}"
    save_parameter_file(yaml_data["likwid_config"], parameter_file)
# Define models and their parameters
for model in ["Vlasov", 'Maxwell']: #, 'LinearMHDDriftkineticCC']:
    base_params = yaml_data[f"params_{model}"]

    # MPI values to scan
    
    filename_template = f"{i_path}/params_{model}_mpi{{mpi}}.yml"
    extra_modifications = None  # model_info.get('extra_modifications', None)
    mpi_scan(base_params, mpi_values, filename_template, extra_modifications)
