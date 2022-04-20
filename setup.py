import setuptools  # this is the "magic" import
from setuptools import setup
#from numpy.distutils.core import setup, Extension

setup(
    name="struphy",
    version="1.7",
    packages=setuptools.find_packages(),
    package_data={
        'struphy.mhd_equil': [
            'gvec/*.dat',
            'gvec/*.json',
            'gvec/*.hdf5',
        ],
        'struphy.io': ['batch/*.sh'],
        'struphy.io.inp': [
            'lin_mhd/*.yml',
            'lin_mhd_MF/*.yml',
            'lin_mhd_psydac/*.yml',
            'cc_lin_mhd_6d/*.yml',
            'cc_lin_mhd_6d_MF/*.yml',
            'pc_lin_mhd_6d_MF_full/*.yml',
            'pc_lin_mhd_6d_MF_perp/*.yml',
            'kinetic_extended/*.yml',
            'lin_Vlasov_Maxwell/*yml',
            'maxwell/*yml',
            'maxwell_psydac/*yml',
            'cold_plasma/*yml',
            'cc_cold_plasma_6d/*yml',
            'inverse_mass_test/*yml',
        ],
        'struphy': ['compile_struphy.mk'],
    },
    # list of executable(s) that come with the package (if applicable)s
    scripts=['scripts/struphy',
             'scripts/example_lin_mhd_1d_fft',
             'scripts/example_maxwell_1d_fft',
             'scripts/example_psydac_parallel',
             'scripts/example_cold_plasma_1d_fft',
             'scripts/example_lin_Vlasov_Maxwell'
             ],
    # list of package dependencies (if necessary)
    install_requires=[
        'h5py',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pyccel==0.10.1',
        'PyYAML',
        'scipy',
        'sympy<1.7,>=1.2',
        'sympde',
        'pytest',
        'mistune<2',
        'sphinx==4.2.0',
        'sphinxcontrib-napoleon',
        'sphinx-rtd-theme',
        'm2r2==0.3.1',
        'vtk',
        'docutils==0.15',
        'wheel',
    ],
    # more information, necessary for an upload to PyPI
    author="Stefan Possanner",
    author_email="spossann@ipp.mpg.de",
    description="Multi-model plasma physics package for the simulation energetic particles",
    license="GNU",
    keywords="plasma, energetic particles, particle-in-cell, discrete differential forms",
    url="https://clapp.pages.mpcdf.de/hylife/",   # project home page, if any
)
