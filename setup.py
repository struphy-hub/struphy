from setuptools import setup, find_packages

with open('struphy/version.py') as f:  
    exec(f.read())

setup(
    name="struphy",
    version=__version__, # auto-detected from struphy/version.py
    packages=find_packages(),
    package_data={
        'struphy.fields_background.mhd_equil.eqdsk': [
            'data/*.high',
        ],
        'struphy.fields_background.mhd_equil.gvec': [
            'ellipstell_v2/newBC_E1D6_M6N6/*.dat',
            'ellipstell_v2/newBC_E1D6_M6N6/*.ini',
            'ellipstell_v2/newBC_E4D6_M6N6/*.dat',
            'ellipstell_v2/newBC_E4D6_M6N6/*.ini',
            'ellipstell_v2/oldBC_E40D5M6N6/*.dat',
            'ellipstell_v2/oldBC_E40D5M6N6/*.ini',
        ],
        'struphy.io': ['batch/*.sh'],
        'struphy.io.inp': ['parameters.yml',
                           'tests/*.yml',
                           'examples/*.yml',
                           ],
        'struphy': ['compile_struphy.mk'],
    },
    # list of executable(s) that come with the package (if applicable)s
    scripts=['bin/struphy',
             'bin/code_tests',
             'bin/example_maxwell_serial',
             'bin/example_maxwell_mpi_3',
             'bin/example_linearmhd_mpi_4',
             'bin/example_orbits_tokamak_mpi_2',
             'bin/example_gc_orbits_tokamak_mpi_2',
             'bin/example_TAE_tokamak_mpi_2',
             'bin/example_hybridmhdvlasovcc_mpi_2'
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
    author="Max Planck Institute for Plasma Physics",
    author_email="stefan.possanner@ipp.mpg.de, florian.holderied@ipp.mpg.de, xin.wang@ipp.mpg.de",
    description="Multi-model plasma physics package",
    license="MIT",
    keywords="plasma physics, fusion, numerical modeling, partial differential equations, energetic particles",
    url="https://struphy.pages.mpcdf.de/struphy/",   # project home page, if any
)