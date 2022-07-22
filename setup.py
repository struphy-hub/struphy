from setuptools import setup, find_packages

with open('struphy/version.py') as f:  
    exec(f.read())

setup(
    name="struphy",
    version=__version__, # auto-detected from struphy/version.py
    packages=find_packages(),
    package_data={
        'struphy.mhd_equil': [
            'gvec/*.dat',
            'gvec/*.json',
            'gvec/*.hdf5',
        ],
        'struphy.io': ['batch/*.sh'],
        'struphy.io.inp': ['parameters.yml',
                           'tests/*.yml',
                           'examples/*.yml',
                           ],
        'struphy': ['compile_struphy.mk'],
    },
    # list of executable(s) that come with the package (if applicable)s
    scripts=['scripts/struphy',
             'scripts/example_psydac_parallel',
             'scripts/example_maxwell_serial',
             'scripts/example_maxwell_mpi_3',
             'scripts/example_linearmhd_mpi_4'
             ],
    # list of package dependencies (if necessary)
    install_requires=[
        'h5py',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pyccel',
        'PyYAML',
        'scipy',
        'sympy<1.7,>=1.2',
        'sympde',
        'pytest',
        'pytest-mpi',
        'mistune<2',
        'sphinx==4.2.0',
        'sphinxcontrib-napoleon',
        'sphinx-rtd-theme',
        'm2r2==0.3.1',
        'vtk',
        'docutils==0.15',
        'wheel',
        'tqdm',
        'psydac',
        'gvec_to_python',
    ],
    # more information, necessary for an upload to PyPI
    author="Max Planck Institute for Plasma Physics, Garching, Germany",
    author_email="stefan.possanner@ipp.mpg.de, florian.holderied@ipp.mpg.de, xin.wang@ipp.mpg.de",
    description="Multi-model plasma physics package",
    license="not yet licensed.",
    keywords="plasma, partial differential equations, energetic particles",
    url="https://clapp.pages.mpcdf.de/hylife/",   # project home page, if any
)
