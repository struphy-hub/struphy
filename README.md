# Welcome to STRUPHY

__A Python package for 
simulating energetic particles in plasma fluids.__

STRUPHY stands for STRUcture-Preserving HYbrid codes. 

The package is developed since 2019 at [Max Planck Institute for Plasma Physics](https://www.ipp.mpg.de/) 
in the division [NMPP (Numerical Methods for Plasma Physics)](https://www.ipp.mpg.de/ippcms/de/for/bereiche/numerik).

## What you can do with STRUPHY

* Solve [PDEs for plasma physics](https://clapp.pages.mpcdf.de/hylife/sections/models.html) with fluid and/or kinetic components 
* Use mapped domains in curved geometries
* Run on HPC clusters (CPUs)
* Post process data and generate `.vtk` files
* Use Python interfaces to quickly add your own model/mapping/physics feature

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

* F. Holderied, S. Possanner, "Magneto-hydrodynamic eigenvalue solver for axis-symmetric equilibria based on smooth polar splines", [IPP pinboard](https://users.euro-fusion.org/auth)

## Contributors

### Current

* Florian Holderied (since 2019)
* Stefan Possanner (since 2019)
* Xin Wang (since 2019)
* Yingzhe Li (since 2021)
* Byung Kyu Na (since 2021)
* Tin Kei Cheng (since 2021)
* Dominik Bell (since 2022)

### Previous

* Benedikt Aigner (2021-2022)

The project benefits from the constant advice of Yaman Güclü, Said Hadjout and Florian Hindenlang.

## License

Not yet published.

## Contact

* Florian Holderied [floho@ipp.mpg.de](floho@ipp.mpg.de)
* Stefan Possanner [spossann@ipp.mpg.de](spossann@ipp.mpg.de)
* Xin Wang [xin.wang@ipp.mpg.de](xin.wang@ipp.mpg.de)











