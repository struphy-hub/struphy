import pytest
import os
from unittest.mock import patch, mock_open
import struphy
from struphy.console.run import struphy_run

libpath = struphy.__path__[0]


def is_sublist(main_list, sub_list):
    """
    Check if sub_list is a sublist of main_list.
    """
    sub_len = len(sub_list)
    return any(main_list[i:i + sub_len] == sub_list for i in range(len(main_list) - sub_len + 1))


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


@pytest.mark.parametrize('model', ['Maxwell', 'Vlasov'])
@pytest.mark.parametrize('input_abs', [os.path.join(libpath, 'io/inp/parameters.yml')])
@pytest.mark.parametrize('output_abs', [os.path.join(libpath, 'io/out/')])
@pytest.mark.parametrize('batch_abs', [None, os.path.join(libpath, 'io/batch/batch_cobra.sh')])
@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('cprofile', [False, True])
@pytest.mark.parametrize('likwid', [False, True])
@pytest.mark.parametrize('runtime', [1, 300])
@pytest.mark.parametrize('save_step', [1, 300])
@pytest.mark.parametrize('mpi', [1, 2])
@patch('subprocess.run')
def test_struphy_run(mock_run, model, input_abs, output_abs, batch_abs, runtime, save_step, restart, cprofile, likwid, mpi):

    struphy_run(model, input_abs=input_abs, output_abs=output_abs, batch_abs=batch_abs,
                runtime=runtime, save_step=save_step, restart=restart, cprofile=cprofile, likwid=likwid, mpi=mpi)

    mock_run.assert_called_once()
    command = mock_run.call_args[0][0]

    if batch_abs is not None:
        assert command == ['sbatch', 'batch_script.sh']

        batch_script_path = os.path.join(output_abs, 'batch_script.sh')

        # Check that the batch script exists
        assert os.path.exists(
            batch_script_path), f"{batch_script_path} should exist"

        # Read the content of the batch script
        with open(batch_script_path, 'r') as f:
            lines = f.readlines()

        batch_run_command = lines[-1].strip().split(' ')

        # Remove batch script
        os.remove(batch_script_path)
        assert not os.path.exists(
            batch_script_path), f"{batch_script_path} should have been deleted"
        
        # Update the command to the command fron the batch script
        command = batch_run_command

        # This is only true if likwid == False, but is taken care of below
        mpirun_command = ['srun', 'python3']
        main = os.path.join(libpath, 'main.py')

        assert '>' in command
    else:
        mpirun_command = ['mpirun', '-n', str(mpi), 'python3']
        main = 'main.py'

    command = split_command(command)

    assert is_sublist(command, ['--runtime', str(runtime)])
    assert is_sublist(command, ['-s', str(save_step)])
    if likwid:
        assert is_sublist(
            command, ['likwid-mpirun', '-n', str(mpi), '-g', 'MEM_DP', '-stats'])
        assert os.path.join(libpath, 'main.py') in command
    else:
        assert is_sublist(command, mpirun_command)
        assert is_sublist(command, [main, model])
    if restart:
        assert is_sublist(command, ['-r'])
    if cprofile:
        assert is_sublist(command, ['python3', '-m', 'cProfile'])