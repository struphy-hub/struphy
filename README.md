# Welcome to STRUPHY

__A Python package for 
simulating energetic particles in plasma fluids.__

STRUPHY stands for STRUcture-Preserving HYbrid codes. The package is developed since 2019 at [Max Planck Institute for Plasma Physics](https://www.ipp.mpg.de/) 
in the division [NMPP (Numerical Methods for Plasma Physics)](https://www.ipp.mpg.de/ippcms/de/for/bereiche/numerik).

Physics features:

* Initial-value solvers for [several kinetic-fluid hybrid models](https://clapp.pages.mpcdf.de/hylife/sections/models.html) 
* MHD eigenvalue solver for axis-symmetric equilibria
* Interface to the [MHD equilibrium code GVEC](https://gitlab.mpcdf.mpg.de/gvec-group/gvec) and to `eqdesk` equilibrium files
* Dispersion relation solvers for MHD, hybrid models and Vlasov-Maxwell (all in slab)

Algorithmic features:

* Discrete differential forms based on high-order B-spline finite elements
* 3d mapped domains with polar singularity (IGA approach with spline mappings available)
* Particle-in-cell method for kinetic species
* Exact conservation laws

Code features:

* Computational kernels pre-compiled with [Pyccel](https://github.com/pyccel/pyccel) to achieve near-Fortran performance
* MPI/OpenMP parallelization of particles (kinetic species)
* In development: MPI paralleization of field solvers via the integration of [Psydac](https://github.com/pyccel/psydac)

Documentation:

* [Struphy user- and developer's guide](https://clapp.pages.mpcdf.de/hylife/)

Key references:

* F. Holderied, S. Possanner, X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell", [J. Comp. Phys. 433 (2021) 110143](https://www.sciencedirect.com/science/article/pii/S0021999121000358?via%3Dihub)

* F. Holderied, S. Possanner, "Magneto-hydrodynamic eigenvalue solver for axisymmetric equilibria based on smooth polar splines", [IPP pinboard](https://users.euro-fusion.org/auth)

Contributors:

* Florian Holderied (since 2019)
* Stefan Possanner (since 2019)
* Xin Wang (since 2019)
* Benedikt Aigner (since 2021)
* Tin Kei Cheng (since 2021)
* Yingzhe Li (since 2021)
* Byung Kyu Na (since 2021)
* Dominik Bell (since 2022)

The project benefits from the constant advice of Yaman Güclü, Said Hadjout and Florian Hindenlang.

Contact:

* Florian Holderied [floho@ipp.mpg.de](floho@ipp.mpg.de)
* Stefan Possanner [spossann@ipp.mpg.de](spossann@ipp.mpg.de)
* Xin Wang [xin.wang@ipp.mpg.de](xin.wang@ipp.mpg.de)

## License

Not yet published.

## Requirements

*STRUPHY* has been tested on Debian `linux-x86_64` systems; it requires

* Python 3 
* pip3
* Fortran compiler (gcc/gfortran)
* openmpi

as well as the following Ubuntu packages (`apt-get install`):

    libblas-dev 
    liblapack-dev

Necessary Python packages will be automatically installed with `pip install .` (list of packages in `setup.py`).


## Mac with M1 chip

Numba must be installed from source::

    git clone https://github.com/numba/llvmlite.git
    cd llvmlite; python setup.py install
    git clone git://github.com/numba/numba.git
    cd numba
    python setup.py build_ext --inplace 
    python setup.py install

Installation of `h5py` when using homebrew::

    HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.13.0 
    pip install h5py


## Installation 
    
Clone and checkout the `devel` branch::

    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git struphy
    cd struphy

User specific install::

    pip install --user .

For developers (path search happens in cloned repo first)::

    pip install -e .

Virtual environment install (recommended if not on computing cluster)::

    python3 -m pip install --user virtualenv
    python3 -m venv <env_name>
    source <env_name>/bin/activate
    pip install .

Next, install the submodules `gvec_to_python` and `psydac` ::

    git submodule init
    git submodule update
    cd psydac
    git pull origin devel
    export CC="mpicc"
    export HDF5_MPI="ON"
    export HDF5_DIR=/path/to/hdf5/openmpi
    python3 -m pip install -r requirements.txt
    python3 -m pip install -r requirements_extra.txt --no-build-isolation
    pip install .
    cd ..
    cd gvec_to_python
    python3 -m pip install . -r requirements.txt
    pip install sympy==1.6.1 
    cd ..
    
Quick help:

    struphy












