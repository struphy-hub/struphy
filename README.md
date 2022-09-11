# STRUPHY - a Python package for energetic particles in plasma 

STRUPHY stands for STRUcture-Preserving HYbrid codes. 

The package is developed since 2019 at [Max Planck Institute for Plasma Physics](https://www.ipp.mpg.de/) 
in the division [NMPP (Numerical Methods for Plasma Physics)](https://www.ipp.mpg.de/ippcms/de/for/bereiche/numerik).

## What you can do with STRUPHY

* Solve a variety of [plasma physics PDEs](https://clapp.pages.mpcdf.de/hylife/sections/models.html) with fluid and/or kinetic components 
* Prescribe curved geometries via [mapped domains](https://clapp.pages.mpcdf.de/hylife/sections/domains.html) 
* [Post process](https://clapp.pages.mpcdf.de/hylife/sections/userguide.html#post-processing) data and generate `vtk` files
* Seamlessly [add](https://clapp.pages.mpcdf.de/hylife/sections/developers.html#how-to-add) your own model/mapping/physics feature
* Contribute to this open source project! 

## Algorithmic features

* Particle-in-cell method for kinetic species
* Discrete differential forms based on high-order B-spline finite elements ([Psydac library](https://github.com/pyccel/psydac)) for fields/fluids
* Exact conservation laws
* Polar splines to treat a polar singularity 
* Kernels are pre-compiled with [Pyccel](https://github.com/pyccel/pyccel) to achieve near-Fortran performance
* MPI/OpenMP hybrid parallelization  

## Installation

* [Struphy documentation](https://clapp.pages.mpcdf.de/hylife/)

## Key references

* F. Holderied, S. Possanner, X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell", [J. Comp. Phys. 433 (2021) 110143](https://www.sciencedirect.com/science/article/pii/S0021999121000358?via%3Dihub)

* F. Holderied, S. Possanner, "Magneto-hydrodynamic eigenvalue solver for axis-symmetric equilibria based on smooth polar splines", [J. Comp. Phys. 464 (2022) 111329](https://www.sciencedirect.com/science/article/pii/S0021999122003916?via%3Dihub)

* F. Holderied, "STRUPHY: a structure-preserving hybrid MHD-kinetic code for the interaction of energetic particles with Alfvén waves in magnetized plasmas", [PhD thesis (2022)](https://mediatum.ub.tum.de/?id=1656539)

## License

Not yet published.

## Contact

* Florian Holderied [floho@ipp.mpg.de](floho@ipp.mpg.de)
* Stefan Possanner [spossann@ipp.mpg.de](spossann@ipp.mpg.de)
* Xin Wang [xin.wang@ipp.mpg.de](xin.wang@ipp.mpg.de)