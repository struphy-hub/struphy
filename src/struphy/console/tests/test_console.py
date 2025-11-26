import os
import pickle
import sys
from unittest import mock
from unittest.mock import patch

import pytest

import struphy as struphy_lib
from struphy.console.compile import struphy_compile
from struphy.console.main import struphy
from struphy.console.params import struphy_params
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
@pytest.mark.parametrize("model", ["Maxwell"])
@pytest.mark.parametrize("yes", [True])
def test_struphy_params(model, yes):
    struphy_params(model, yes=yes)


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
