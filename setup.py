import setuptools  # this is the "magic" import
from numpy.distutils.core import setup, Extension

setup(
    name="struphy",
    version="0.9",
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
    package_data={'struphy.models.mhd_equil': ['gvec/*.dat'],
                  'struphy.io': ['batch/*.sh'],
                  'struphy.io.inp': ['cc_lin_mhd_6d/*.yml',
                                    'massless_extended/*.yml',
                                    'pc_lin_mhd_6d/*.yml',],
                  'struphy': ['compile_struphy.mk'],
                  },
    # data_files=[('struphy_config/parameters', ['struphy/models/codes/yaml/code_1_input.yml',
    #                                            'struphy/models/codes/yaml/code_2_input.yml']),
                #('struphy_config/I_O', ['struphy/models/MHD_equilibria/gvec/*.dat',]),
                # ],
    # list of executable(s) that come with the package (if applicable)
    scripts=['scripts/struphy'],
    # list of package dependencies (if necessary)
    install_requires=['numpy', 'pyccel', 'numba', 'h5py', 'pyyaml', 'matplotlib'],
    # python extensions (pyccel)
    # ext_package='struphy',
    # ext_modules=[Extension(name = 'feec.projectors.kernels',
    #                        extra_compile_args = ['-O3'], 
    #                        sources = ['struphy/feec/projectors/__pyccel__/kernels.f90',
    #                                   'struphy/feec/projectors/__pyccel__/bind_c_kernels.f90'],
    #                        #f2py_options = ['--quiet'],
    #                        )],
    # other files:
    #data_files=[('bitmaps', ['bm/b1.gif', 'bm/b2.gif']),
    #            ('config', ['cfg/data.cfg'])],
    # more information, necessary for an upload to PyPI
    author="John Doe",
    author_email="john.doe@example.mpg.de",
    description="example package that prints hello world",
    license="PSF",
    keywords="hello world example",
    url="https://example.mpg.de/helloworld/",   # project home page, if any
)
