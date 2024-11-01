from struphy.console.utils import generate_batch_script, save_batch_script
def struphy_generate(kind, template, likwid = False, **slurm_args):
    print('Generate batch script for {template}')
    sbatch_params = {
        'raven': {
            'ntasks_per_node': 8,
            'module_setup': "module load anaconda/3/2023.03 gcc/12 openmpi/4.1 likwid/5.2",
            'likwid': likwid,
        },
        'cobra': {
            'ntasks_per_node': 72,
            'likwid': likwid,
        },
        'viper': {
            'ntasks_per_node': 4,
            'likwid': likwid,
        },
    }
    params = {**sbatch_params[template], **slurm_args['slurm_args']}
    print(params)

    
    batch_script = generate_batch_script(**params)
    print(batch_script)
    