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
from struphy.console.pproc import struphy_pproc

# from struphy.console.profile import struphy_profile
from struphy.console.run import struphy_run, subp_run

# from struphy.console.test import struphy_test
# from struphy.console.units import struphy_units
from struphy.utils.utils import read_state

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
        # Test cases for 'run' sub-command with different models and options
        ["run", "Maxwell"],
        ["run", "Vlasov"],
        ["run", "Maxwell", "--output", "sim_2"],
        # ["run", "Maxwell", "--batch", "batch_cobra.sh"],
        ["run", "Maxwell", "--mpi", "4"],
        ["run", "Vlasov", "--restart"],
        # Test cases for 'compile' sub-command with options
        ["compile"],
        ["compile", "-y"],
        ["compile", "--language", "fortran"],
        ["compile", "--compiler", "intel"],
        ["compile", "--omp-pic"],
        ["compile", "--verbose"],
        ["compile", "--delete"],
        # Test cases for 'units' sub-command
        ["units", "Maxwell"],
        # ["units", "Vlasov", "--input", "params.yml"],
        # ["units", "Maxwell", "--input-abs", "/params.yml"],
        # Test cases for 'params' sub-command
        ["params", "Maxwell"],
        ["params", "Vlasov", "--options"],
        # ["params", "Maxwell", "-f", "params_Maxwell.yml"],
        # Test cases for 'profile' sub-command
        ["profile", "sim_1"],
        ["profile", "sim_2", "--replace"],
        ["profile", "sim_3", "--n-lines", "10"],
        ["profile", "sim_1", "--savefig", "profile_output.png"],
        # Test cases for 'pproc' sub-command
        ["pproc", "-d", "sim_1"],
        ["pproc", "--dir-abs", "/absolute/path/to/sim_1"],
        ["pproc", "--step", "5"],
        ["pproc", "--physical"],
        # Test cases for 'test' sub-command
        ["test", "models"],
        ["test", "unit"],
        ["test", "Maxwell", "--Tend", "1.0"],
        ["test", "hybrid", "--mpi", "8"],
    ],
)
def test_main(args):
    # Mock the func call (don't execute it)
    with (
        patch("struphy.console.run.struphy_run") as mock_subprocess_run,
        patch("struphy.console.compile.struphy_compile") as mock_compile,
        patch("struphy.console.units.struphy_units") as mock_units,
        patch("struphy.console.params.struphy_params") as mock_params,
        patch("struphy.console.profile.struphy_profile") as mock_profile,
        patch("struphy.console.pproc.struphy_pproc") as mock_pproc,
        patch("struphy.console.test.struphy_test") as mock_test,
    ):
        funcs = {
            "run": mock_subprocess_run,
            "compile": mock_compile,
            "units": mock_units,
            "params": mock_params,
            "profile": mock_profile,
            "pproc": mock_pproc,
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
                if func_name == "pproc":
                    pass
                else:
                    func.assert_called_once()
            else:
                func.assert_not_called()


@pytest.mark.mpi_skip
@pytest.mark.parametrize("model", ["Maxwell", "Vlasov"])
@pytest.mark.parametrize("input_abs", [os.path.join(libpath, "io/inp/parameters.yml")])
@pytest.mark.parametrize("output_abs", [os.path.join(libpath, "io/out/sim_1")])
@pytest.mark.parametrize(
    "batch_abs",
    [None, os.path.join(libpath, "io/batch/batch_cobra.sh")],
)
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("cprofile", [False, True])
@pytest.mark.parametrize("likwid", [False, True])
@pytest.mark.parametrize("runtime", [1, 300])
@pytest.mark.parametrize("save_step", [1, 300])
@pytest.mark.parametrize("mpi", [1, 2])
def test_struphy_run(
    model,
    input_abs,
    output_abs,
    batch_abs,
    runtime,
    save_step,
    restart,
    cprofile,
    likwid,
    mpi,
):
    """Test for `struphy run`"""

    with patch("subprocess.run") as mock_subprocess_run:
        # Assert the batch file exists (if provided)
        if batch_abs is not None:
            assert os.path.exists(batch_abs), f"Batch file does not exist: {batch_abs}"

        run_command = struphy_run(
            model,
            input_abs=input_abs,
            output_abs=output_abs,
            batch_abs=batch_abs,
            runtime=runtime,
            save_step=save_step,
            restart=restart,
            cprofile=cprofile,
            likwid=likwid,
            mpi=mpi,
        )

        # Assert that the batch script was copied if batch_abs was not None
        batch_abs_new = os.path.join(output_abs, "batch_script.sh")
        if batch_abs is not None:
            assert os.path.isfile(
                batch_abs_new,
            ), f"Batch script was not created: {batch_abs_new}"

        mock_subprocess_run.assert_called_once()
        subprocess_call = mock_subprocess_run.call_args[0][0]

        if batch_abs is not None:
            assert subprocess_call == ["sbatch", "batch_script.sh"]

            # This is only true if likwid == False, but is taken care of below
            mpirun_command = ["srun", "python3"]
            main = os.path.join(libpath, "main.py")
        else:
            mpirun_command = ["mpirun", "-n", str(mpi), "python3"]
            main = "main.py"

        run_command = split_command(run_command)

        assert is_sublist(run_command, ["--runtime", str(runtime)])
        assert is_sublist(run_command, ["-s", str(save_step)])
        if likwid:
            assert is_sublist(
                run_command,
                ["likwid-mpirun", "-n", str(mpi), "-g", "MEM_DP", "-mpi", "openmpi"],
            )
            assert os.path.join(libpath, "main.py") in run_command
        else:
            assert is_sublist(run_command, mpirun_command)
            assert is_sublist(run_command, [model])
        if restart:
            assert is_sublist(run_command, ["-r"])
        if cprofile:
            assert is_sublist(run_command, ["python3", "-m", "cProfile"])


def run_struphy(args):
    with mock.patch.object(sys, "argv", ["struphy"] + args):
        struphy()


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "args_expected",
    [
        [["--version"], [""]],
        [["--path"], ["Struphy installation path"]],
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
        print(f"{path = }")
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
        print(f"{language = }")
        print(f"{compiler = }")
        print(f"{omp_pic = }")
        print(f"{omp_feec = }")
        print(f"{delete = }")
        print(f"{status} = ")
        print(f"{verbose = }")
        print(f"{dependencies = }")
        print(f"{time_execution = }")
        print(f"{yes = }")
        print(f"{mock_save_state.call_count = }")
        print(f"{mock_subprocess_run.call_count = }")
        print(f"{mock_os_remove.call_count = }")

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
@pytest.mark.parametrize("file", ["params_Maxwell.yml", "params_Maxwel2.yml"])
@pytest.mark.parametrize("yes", [True])
@pytest.mark.parametrize("options", [True, False])
def test_struphy_params(tmp_path, model, file, yes, options):
    file_path = os.path.join(tmp_path, file)
    struphy_params(model, str(file_path), yes=yes, options=options)


@pytest.mark.mpi_skip
@pytest.mark.parametrize("dir", ["simulation_output", "custom_output"])
@pytest.mark.parametrize("dir_abs", [None, "/custom/path/simulation_output"])
@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("celldivide", [1, 2])
@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("guiding_center", [False, True])
@pytest.mark.parametrize("classify", [False, True])
def test_struphy_pproc(
    dir,
    dir_abs,
    step,
    celldivide,
    physical,
    guiding_center,
    classify,
):
    with patch("subprocess.run") as mock_subprocess_run:
        struphy_pproc(
            dirs=[dir],
            dir_abs=dir_abs,
            step=step,
            celldivide=celldivide,
            physical=physical,
            guiding_center=guiding_center,
            classify=classify,
        )

        # Construct the expected directory path
        # Retrieve `o_path` from the actual state file
        o_path = read_state()["o_path"]

        if dir_abs is None:
            expected_dir_abs = os.path.join(o_path, dir)
        else:
            expected_dir_abs = dir_abs

        # Build the expected command
        command = [
            "python3",
            "post_processing/pproc_struphy.py",
            expected_dir_abs,
            "-s",
            str(step),
            "--celldivide",
            str(celldivide),
        ]
        if physical:
            command.append("--physical")
        if guiding_center:
            command.append("--guiding-center")
        if classify:
            command.append("--classify")

        mock_subprocess_run.assert_called_once_with(command, cwd=libpath, check=True)


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

# TODO: Fix error occuring when state is None in the CI
# For now, I 've commented out test_struphy_units
# it works locally, but I get errors in the CI,
# maybe the state is altered in some other test
# TODO: Parametrize all models here
# @pytest.mark.parametrize("model", struphy_models)
# @pytest.mark.parametrize("input", [None])  # , "parameters.yml"])
# # , "src/struphy/io/inp/parameters.yml"])
# @pytest.mark.parametrize("input_abs", [None])
# def test_struphy_units(model, input, input_abs):

#     # TODO: Fix this: AttributeError: type object 'KineticBackground' has no attribute 'generate_default_parameter_file'
#     if model == "KineticBackground":
#         return
#     i_path = read_state()["i_path"]
#     expected_input_abs = (
#         input_abs if input_abs else os.path.join(i_path, input) if input else None
#     )

#     # Redirect stdout to capture print output
#     captured_output = StringIO()
#     sys.stdout = captured_output

#     # Call the function with parameterized arguments
#     struphy_units(model=model, input=input, input_abs=input_abs)

#     # Read stdout
#     sys.stdout = sys.__stdout__
#     output = captured_output.getvalue()
#     assert "UNITS:" in output, f"'UNITS:' not found in output: {output}"
#     if model == "Maxwell":
#         assert "Unit of length" in output
#     # TODO: Add model specific units here


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

    test_struphy_run(
        model=model,
        input_abs=input_abs,
        output_abs=output_abs,
        batch_abs=batch_abs,
        runtime=runtime,
        save_step=save_step,
        restart=restart,
        cprofile=cprofile,
        likwid=likwid,
        mpi=mpi,
    )
    print("Test passed")
