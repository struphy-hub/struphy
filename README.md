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
    
    git clone -b devel git@gitlab.mpcdf.mpg.de:clapp/hylife.git
    pip install .
    
For quick help type

    struphy -h

Compilation of kernels:

    struphy -c

## Requirements

*STRUPHY* has been tested on Debian `linux-x86_64` systems; it requires

* Python 3 
* libopenmpi-dev (`apt-get install libopenmpi-dev`)
* requirements for [pyccel](https://github.com/pyccel/pyccel)

Python packages (automatically installed via `pip install .`):

    h5py
    matplotlib
    mpi4py
    numpy<1.21,>=1.17
    pyccel==0.10.1
    PyYAML
    scipy
    sympy
    pytest
    treelib

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



