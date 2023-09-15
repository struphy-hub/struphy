import os
import subprocess
import struphy

libpath = struphy.__path__[0]
i_path = os.path.join(libpath, 'io', 'inp')
o_path = os.path.join(libpath, 'io', 'out')


def tutorial_02():
    subprocess.run(['struphy', 'run', 'LinearMHDVlasovCC',
                    '--input-abs', os.path.join(i_path,
                                                'tutorials', 'params_02.yml'),
                    '--output-abs', os.path.join(o_path, 'tutorial_02'),
                    '--mpi', '2'], check=True)


def tutorial_03():
    subprocess.run(['struphy', 'run', 'LinearMHD',
                    '--input-abs', os.path.join(i_path,
                                                'tutorials', 'params_03.yml'),
                    '--output-abs', os.path.join(o_path, 'tutorial_03'),
                    '--mpi', '2'], check=True)

    subprocess.run(['struphy', 'pproc', '--dir-abs',
                   os.path.join(o_path, 'tutorial_03')], check=True)


def tutorial_04():
    subprocess.run(['struphy', 'run', 'Maxwell',
                    '--input-abs', os.path.join(i_path,
                                                'tutorials', 'params_04a.yml'),
                    '--output-abs', os.path.join(o_path, 'tutorial_04a'),
                    '--mpi', '2'], check=True)

    subprocess.run(['struphy', 'pproc', '-d',
                   os.path.join(o_path, 'tutorial_04a')], check=True)

    subprocess.run(['struphy', 'run', 'LinearMHD',
                    '--input-abs', os.path.join(i_path,
                                                'tutorials', 'params_04b.yml'),
                    '--output-abs', os.path.join(o_path, 'tutorial_04b'),
                    '--mpi', '2'], check=True)

    subprocess.run(['struphy', 'pproc', '-d',
                    os.path.join(o_path, 'tutorial_04b')], check=True)


def tutorial_05():
    subprocess.run(['struphy', 'run', 'Vlasov',
                    '--input-abs', os.path.join(i_path, 'tutorials',
                                                'params_05a.yml'),
                    '--output-abs', os.path.join(o_path, 'tutorial_05a')], check=True)

    subprocess.run(['struphy', 'pproc', '-d',
                   os.path.join(o_path, 'tutorial_05a')], check=True)

    subprocess.run(['struphy', 'run', 'DriftKinetic',
                    '--input-abs', os.path.join(i_path, 'tutorials',
                                                'params_05b.yml'),
                    '--output-abs', os.path.join(o_path, 'tutorial_05b')], check=True)

    subprocess.run(['struphy', 'pproc', '-d',
                   os.path.join(o_path, 'tutorial_05b')], check=True)
