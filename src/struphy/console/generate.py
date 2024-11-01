from struphy.console.utils import generate_batch_script, save_batch_script
def struphy_generate(kind, template, likwid = False, **slurm_args):
    print(f'Generate batch script for {template}')
    sbatch_params = {
        'raven': {
            'ntasks-per-node': 8,
            'module-setup': "module load anaconda/3/2023.03 gcc/12 openmpi/4.1 likwid/5.2",
            'likwid': likwid,
        },
        'cobra': {
            'ntasks-per-node': 72,
            'likwid': likwid,
        },
        'viper': {
            'ntasks-per-node': 4,
            'likwid': likwid,
        },
    }
    params = {}
    if template is not None:
        params = {**sbatch_params[template]}
    if slurm_args['slurm_args'] is not None:
        params = {**params, **slurm_args['slurm_args']}
    batch_script = generate_batch_script(chars_until_comment=60,**params)
    print(batch_script)
    print('done')
    