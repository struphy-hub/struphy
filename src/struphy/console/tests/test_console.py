import os
import pickle
import sys
from unittest import mock
from unittest.mock import patch  # , MagicMock, mock_open

import pytest

# from psydac.ddm.mpi import mpi as MPI
import struphy
import struphy as struphy_lib
from struphy.console.compile import struphy_compile
from struphy.console.main import struphy
from struphy.console.params import struphy_params

# from struphy.console.profile import struphy_profile
# from struphy.console.test import struphy_test
from struphy.utils.utils import read_state, subp_run

libpath = struphy_lib.__path__[0]
state = read_state()

# Create models_list if it doesn't exist
if not os.path.isfile(os.path.join(libpath, "models", "models_list")):
    cmd = ["struphy", "--refresh-models"]
    subp_run(cmd)

with open(os.path.join(libpath, "models", "models_list"), "rb") as fp:
    struphy_models = pickle.load(fp)


def is_sublist(main_list, sub_list):
    """
    Check if sub_list is a sublist of main_list.
    """
    sub_len = len(sub_list)
    return any(main_list[i : i + sub_len] == sub_list for i in range(len(main_list) - sub_len + 1))


def split_command(command):
    """
    Split a command string into a list of arguments.
    """
    # only works if there are no real spaces in the element.
    # Could be improved by not splitting if the space is '\ ' with regex
    spl = []
    for element in command:
        spl.extend(element.split())
    return spl


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "args",
    [
        # Test cases for 'compile' sub-command with options
        ["compile"],
        ["compile", "-y"],
        ["compile", "--language", "fortran"],
        ["compile", "--compiler", "intel"],
        ["compile", "--omp-pic"],
        ["compile", "--verbose"],
        ["compile", "--delete"],
        # Test cases for 'params' sub-command
        ["params", "Maxwell"],
        ["params", "Vlasov"],
        # ["params", "Maxwell", "-f", "params_Maxwell.yml"],
        # Test cases for 'profile' sub-command
        ["profile", "sim_1"],
        ["profile", "sim_2", "--replace"],
        ["profile", "sim_3", "--n-lines", "10"],
        ["profile", "sim_1", "--savefig", "profile_output.png"],
        # Test cases for 'test' sub-command
        ["test", "models"],
        ["test", "unit"],
        ["test", "Maxwell"],
        ["test", "hybrid", "--mpi", "8"],
    ],
)
def test_main(args):
    # Mock the func call (don't execute it)
    with (
        patch("struphy.console.compile.struphy_compile") as mock_compile,
        patch("struphy.console.params.struphy_params") as mock_params,
        patch("struphy.console.profile.struphy_profile") as mock_profile,
        patch("struphy.console.test.struphy_test") as mock_test,
    ):
        funcs = {
            "compile": mock_compile,
            "params": mock_params,
            "profile": mock_profile,
            "test": mock_test,
        }

        # Set sys args
        sys.argv = ["struphy"] + args

        # Call struphy
        try:
            struphy()
        except SystemExit:
            pass  # Ignore the exit in tests

        for func_name, func in funcs.items():
            if args[0] == func_name:
                func.assert_called_once()
            else:
                func.assert_not_called()


def run_struphy(args):
    with mock.patch.object(sys, "argv", ["struphy"] + args):
        struphy()


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "args_expected",
    [
        [["--version"], [""]],
        [["--short-help"], ["available commands"]],
        [["--fluid"], ["Fluid models"]],
        [["--kinetic"], ["Kinetic models"]],
        [["--hybrid"], ["Hybrid models"]],
        [["--toy"], ["Toy models"]],
        [["--refresh-models"], ["Collecting available models"]],
    ],
)
def test_main_options(args_expected, capsys):
    args = args_expected[0]

    with pytest.raises(SystemExit):
        run_struphy(args)

    # Capture the output
    captured = capsys.readouterr()

    # Assert that output was printed
    assert captured.out != ""

    for expected in args_expected[1]:
        assert expected in captured.out


@pytest.mark.mpi_skip
@pytest.mark.parametrize("language", ["c", "fortran"])
@pytest.mark.parametrize("compiler", ["gnu", "intel"])
@pytest.mark.parametrize("compiler_config", [None])
@pytest.mark.parametrize("omp_pic", [True, False])
@pytest.mark.parametrize("omp_feec", [True, False])
@pytest.mark.parametrize("delete", [True, False])
@pytest.mark.parametrize("status", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("dependencies", [True, False])
@pytest.mark.parametrize("time_execution", [True, False])
@pytest.mark.parametrize("yes", [True])
def test_struphy_compile(
    language,
    compiler,
    compiler_config,
    omp_pic,
    omp_feec,
    delete,
    status,
    verbose,
    dependencies,
    time_execution,
    yes,
):
    # Save the original os.remove
    os_remove = os.remove

    def mock_remove(path):
        # Mock `os.remove` except when called for _tmp.py files
        # Otherwise, we will not remove all the *_tmp.py files
        # We can not use the real os.remove becuase then
        # the state and all compiled files will be removed
        print(f"{path =}")
        if "_tmp.py" in path:
            print("Not mock remove")
            os_remove(path)
        else:
            print("Mock remove")
            return

    # Patch utils.save_state
    with (
        patch("struphy.utils.utils.save_state") as mock_save_state,
        patch("subprocess.run") as mock_subprocess_run,
        patch("os.remove", side_effect=mock_remove) as mock_os_remove,
    ):
        # Call the function with parametrized inputs
        struphy_compile(
            language=language,
            compiler=compiler,
            compiler_config=compiler_config,
            omp_pic=omp_pic,
            omp_feec=omp_feec,
            delete=delete,
            status=status,
            verbose=verbose,
            dependencies=dependencies,
            time_execution=time_execution,
            yes=yes,
        )
        print(f"{language =}")
        print(f"{compiler =}")
        print(f"{omp_pic =}")
        print(f"{omp_feec =}")
        print(f"{delete =}")
        print(f"{status} = ")
        print(f"{verbose =}")
        print(f"{dependencies =}")
        print(f"{time_execution =}")
        print(f"{yes =}")
        print(f"{mock_save_state.call_count =}")
        print(f"{mock_subprocess_run.call_count =}")
        print(f"{mock_os_remove.call_count =}")

        if delete:
            print("if delete")
            mock_subprocess_run.assert_called()
            # mock_save_state.assert_called()

        elif status:
            print("elif status")
            # If only status is True (without delete), subprocess.run should not be called
            mock_subprocess_run.assert_not_called()
            mock_save_state.assert_called()

        elif dependencies:
            print("elif dependencies")
            # For dependencies=True, subprocess.run should not be called
            mock_subprocess_run.assert_not_called()
            # mock_save_state.assert_not_called()

        else:
            print("else")
            # Normal compilation case
            mock_subprocess_run.assert_called()
            mock_save_state.assert_called()


@pytest.mark.mpi_skip
@pytest.mark.parametrize("model", ["Maxwell"])
@pytest.mark.parametrize("yes", [True])
def test_struphy_params(model, yes):
    struphy_params(model, yes=yes)


# # TODO: Not working, too much stuff too patch
# @pytest.mark.mpi_skip
# matplotlib.use("Agg")
# @pytest.mark.parametrize("dirs", [["output1"], ["output2"], ["output1", "output2"]])
# @pytest.mark.parametrize("replace", [True, False])
# @pytest.mark.parametrize("all", [True, False])
# @pytest.mark.parametrize("n_lines", [10, 20])
# @pytest.mark.parametrize("print_callers", [True, False])
# @pytest.mark.parametrize("savefig", [None, "profile_output.png"])
# def test_struphy_profile(dirs, replace, all, n_lines, print_callers, savefig):

#     # Retrieve `o_path` from the actual state file
#     o_path = read_state()["o_path"]
#     abs_paths = [os.path.join(o_path, d) for d in dirs]

#     with (
#         patch(
#             "struphy.post_processing.cprofile_analyser.get_cprofile_data",
#         ) as mock_get_cprofile_data,
#         patch(
#             "struphy.post_processing.cprofile_analyser.replace_keys",
#         ) as mock_replace_keys,
#         patch("builtins.open", new_callable=MagicMock) as mock_open,
#         patch(
#             "pickle.load",
#             return_value={"main.py:1(main)": {"cumtime": 1.0}},
#         ) as mock_pickle_load,
#         patch("matplotlib.pyplot.subplots") as mock_subplots,
#     ):

#         # Mocking the plt figure and axis for `subplots`
#         mock_fig, mock_ax = MagicMock(), MagicMock()
#         mock_subplots.return_value = (mock_fig, mock_ax)

#         # Call the function with parameterized arguments
#         struphy_profile(
#             dirs=dirs,
#             replace=replace,
#             all=all,
#             n_lines=n_lines,
#             print_callers=print_callers,
#             savefig=savefig,
#         )

#         for path in abs_paths:
#             mock_get_cprofile_data.assert_any_call(path, print_callers)

#         for path in abs_paths:
#             profile_dict_path = os.path.join(path, "profile_dict.sav")
#             meta_path = os.path.join(path, "meta.txt")
#             params_path = os.path.join(path, "parameters.yml")

#             mock_open.assert_any_call(profile_dict_path, "rb")
#             mock_open.assert_any_call(meta_path, "r")
#             mock_open.assert_any_call(params_path, "r")

#         if replace:
#             mock_replace_keys.assert_called()

#         if savefig:
#             # If savefig is provided, check the savefig call
#             save_path = os.path.join(o_path, savefig)
#             mock_fig.savefig.assert_called_once_with(save_path)
#         else:
#             mock_fig.show.assert_called_once()


if __name__ == "__main__":
    # Set test parameters
    model = "Maxwell"
    input_abs = os.path.join(libpath, "io/inp/parameters.yml")
    output_abs = os.path.join(libpath, "io/out/sim_1")
    batch_abs = os.path.join(libpath, "io/batch/batch_cobra.sh")
    runtime = 300
    save_step = 300
    restart = True
    cprofile = False
    likwid = False
    mpi = 2
