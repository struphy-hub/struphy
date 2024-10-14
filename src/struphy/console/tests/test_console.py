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
    return any(main_list[i:i + sub_len] ==
               sub_list for i in range(len(main_list) - sub_len + 1))


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
@pytest.mark.parametrize('output_abs', [os.path.join(libpath, 'io/out/sim_1')])
@pytest.mark.parametrize('batch_abs',[None, os.path.join(libpath, 'io/batch/batch_cobra.sh')])
@pytest.mark.parametrize('restart', [False, True])
@pytest.mark.parametrize('cprofile', [False, True])
@pytest.mark.parametrize('likwid', [False, True])
@pytest.mark.parametrize('runtime', [1, 300])
@pytest.mark.parametrize('save_step', [1, 300])
@pytest.mark.parametrize('mpi', [1, 2])
@patch('subprocess.run')
def test_struphy_run(
        mock_run,
        model,
        input_abs,
        output_abs,
        batch_abs,
        runtime,
        save_step,
        restart,
        cprofile,
        likwid,
        mpi):

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
        mpi=mpi)

    mock_run.assert_called_once()
    subprocess_call = mock_run.call_args[0][0]

    if batch_abs is not None:
        assert subprocess_call == ['sbatch', 'batch_script.sh']

        # This is only true if likwid == False, but is taken care of below
        mpirun_command = ['srun', 'python3']
        main = os.path.join(libpath, 'main.py')
    else:
        mpirun_command = ['mpirun', '-n', str(mpi), 'python3']
        main = 'main.py'

    run_command = split_command(run_command)

    assert is_sublist(run_command, ['--runtime', str(runtime)])
    assert is_sublist(run_command, ['-s', str(save_step)])
    if likwid:
        assert is_sublist(
            run_command, [
                'likwid-mpirun', '-n', str(mpi), '-g', 'MEM_DP', '-stats'])
        assert os.path.join(libpath, 'main.py') in run_command
    else:
        assert is_sublist(run_command, mpirun_command)
        assert is_sublist(run_command, [main, model])
    if restart:
        assert is_sublist(run_command, ['-r'])
    if cprofile:
        assert is_sublist(run_command, ['python3', '-m', 'cProfile'])
