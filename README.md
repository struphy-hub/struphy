# Welcome to STRUPHY!

*STRUPHY* (STRUcture-Preserving HYbrid codes) is a multi-model plasma physics package for 
the simulation energetic particles (EPs) in ambient plasma.

The package is developed at [Max Planck Institute for Plasma Physics](https://www.ipp.mpg.de/) 
in the division [NMPP (Numerical Methods for Plasma Physics)](https://www.ipp.mpg.de/ippcms/de/for/bereiche/numerik)
and includes

1. initial-value solvers for kinetic-fluid hybrid models 
2. MHD eigenvalue solver for axis-symmetric equilibria
3. interface to the [MHD equilibrium code GVEC](https://gitlab.mpcdf.mpg.de/gvec-group/gvec)
4. interface to `eqdesk` file format
5. dispersion relation solvers for MHD, hybrid models and Vlasov-Maxwell (all in slab)

*STRUPHY* computational kernels are pre-compiled with [pyccel](https://github.com/pyccel/pyccel) to reach ***FORTRAN*** performance. *STRUPHY* can be run on clusters with OpenMP/MPI hybrid parallelization.

Contact:

* Stefan Possanner [spossann@ipp.mpg.de](spossann@ipp.mpg.de)
* Florian Holderied [floho@ipp.mpg.de](floho@ipp.mpg.de)

## License

Not yet published.

## Installation 
    
You can clone the full repository and checkout the `devel` branch via::

    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git

After that you can install *STRUPHY* to the Python user install directory for your platform, typically ~/.local/::

    pip install --user .

However, for developers we recommend the creation of a virtual environment and installation therein::

    python3 -m pip install --user virtualenv
    python3 -m venv <env_name>
    source <env_name>/bin/activate
    pip3 install .
    
For quick help type

    struphy -h

Compilation of kernels:

    struphy -c

The default code `lin_mhd` is run via

    cd ..
    struphy

We recommend the change of directory such that the compiled versions of files is used.

## Requirements

*STRUPHY* has been tested on Debian `linux-x86_64` systems; it requires

* Python 3 
* pip3
* Fortran compiler (gcc/gfortran))

as well as the following Ubuntu packages (`apt-get install`):

    libblas-dev 
    liblapack-dev
    libopenmpi-dev
    openmpi-bin
    libomp-dev 
    libomp5
    tree

and finally the Python packages (automatically installed via `pip install .`):

    h5py
    matplotlib
    mpi4py
    numpy<1.21,>=1.17
    pyccel==0.10.1
    PyYAML
    scipy
    sympy
    vtk
    pandas
    pytest
    sphinx
    sphinxcontrib-napoleon
    sphinx-rtd-theme 
    m2r2
    docutils==0.15



## Documentation

* [STRUPHY userguide](https://clapp.pages.mpcdf.de/hylife/)

* [Wiki for developers](https://gitlab.mpcdf.mpg.de/clapp/hylife/-/wikis/home)


## Examples

* [MHD waves in a slab (textbook example)](https://clapp.pages.mpcdf.de/hylife/sections/examples.html#mhd-dispersion-relation-slab)


## Contributors

* Florian Holderied (since 2019)
* Stefan Possanner (since 2019)
* Xin Wang (since 2019)
* Benedikt Aigner (since 2021)
* Tin Kei Cheng (since 2021)
* Yingzhe Li (since 2021)
* Byung Kyu Na (since 2021)

The project benfits from the constant advice of Yaman Güclü and Florian Hindenlang.


## References

If you use *STRUPHY* please cite at least one of the following works:

* F. Holderied, S. Possanner, X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell", [J. Comp. Phys. 433 (2021) 110143](https://www.sciencedirect.com/science/article/pii/S0021999121000358?via%3Dihub)

* F. Holderied, S. Possanner, "Magneto-hydrodynamic eigenvalue solver for axisymmetric equilibria based on smooth polar splines", [IPP pinboard](https://users.euro-fusion.org/auth)



