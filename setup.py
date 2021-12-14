import setuptools  # this is the "magic" import
from setuptools import setup
#from numpy.distutils.core import setup, Extension

setup(
    name="struphy",
    version="1.7",
    packages=setuptools.find_packages(),
    # packages = ['struphy', 
    #             'struphy.diagnostics', 
    #             'struphy.models.dispersion_relations', 
    #             'struphy.feec', 
    #             'struphy.feec.basics', 
    #             'struphy.feec.projectors',
    #             'struphy.geometry',
    #             'struphy.io',
    #             'struphy.models', 
    #             'struphy.models.codes', 
    #             'struphy.models.dispersion_relations', 
    #             'struphy.models.distribution_functions',
    #             'struphy.models.MHD_equil',
    #             ],
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
            'cc_lin_mhd_6d/*.yml',
            'cc_lin_mhd_6d_MF/*.yml',
            'pc_lin_mhd_6d_MF_full/*.yml',
            'pc_lin_mhd_6d_MF_perp/*.yml',
            'massless_extended/*.yml',
        ],
        'struphy': ['compile_struphy.mk'],
    },
    # list of executable(s) that come with the package (if applicable)s
    scripts=['scripts/struphy'],
    # list of package dependencies (if necessary)
    install_requires=[
        'h5py',
        'matplotlib',
        'mpi4py',
        'numba',
        'numpy<1.21,>=1.17',
        'pyccel==0.10.1',
        'PyYAML',
        'scipy',
        'sympy',
        'pytest',
        'mistune<2',
        'sphinx==4.2.0',
        'sphinxcontrib-napoleon',
        'sphinx-rtd-theme', 
        'm2r2==0.3.1',
        'vtk',
        'docutils==0.15',
        'pandas',
    ],
    # more information, necessary for an upload to PyPI
    author="Stefan Possanner",
    author_email="spossann@ipp.mpg.de",
    description="Multi-model plasma physics package for the simulation energetic particles",
    license="GNU",
    keywords="plasma, energetic particles, particle-in-cell, discrete differential forms",
    url="https://clapp.pages.mpcdf.de/hylife/",   # project home page, if any
)
