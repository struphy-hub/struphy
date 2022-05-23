# Welcome to STRUPHY

__A Python package for 
simulating energetic particles in plasma fluids.__

STRUPHY stands for STRUcture-Preserving HYbrid codes. The package is developed since 2019 at [Max Planck Institute for Plasma Physics](https://www.ipp.mpg.de/) 
in the division [NMPP (Numerical Methods for Plasma Physics)](https://www.ipp.mpg.de/ippcms/de/for/bereiche/numerik).

## What's new in version 1.9.0

* Tear-down and rebuild has begun: only models based on the `StruphyModel` base class can be executed (just `maxwell` at the moment), no backward compatibility
* Added propagator base class `struphy.models.codes.propagators.Propagator`
* Added MHD equilibirum base class `struphy.fields_equil.mhd_equil.mhd_equils.EquilibriumMHD` and 4 subclasses `EquilibriumMHDSlab`, `EquilibriumMHDShearedSlab`, `EquilibriumMHDCylinder` and `EquilibriumMHDTorus`.
* Removed pyccel requirement in the setup file, always install newest version
* Improved continuous integration: use dedicated MPCDF runner for struphy (thanks to Flo Hindenlang), linting, code tests, `.whl` file available as artifact
* Improved documentation: 
    * Install from wheel (no source code needed)
    * workflow for adding code
    * detailed explanation of how to add new models, propagators
    * Section Continuous Integration shows how to add tests
    * Added new top-level section Toolkit 

## What you can do with STRUPHY

* Solve a variety of [PDEs for plasma physics](https://clapp.pages.mpcdf.de/hylife/sections/models.html) 
* Specify your own geometry (mapping) and initial conditions
* Use input from Tokamak/Stellarator equilibrium solvers
* Run on HPC clusters
* Post process data and generate `vtk` files
* Seamlessly add a new model PDE using the `StruphyModel` base class

## Algorithmic features

* Discrete differential forms based on high-order B-spline finite elements using the [Psydac library](https://github.com/pyccel/psydac)
* 3d mapped domains with polar singularity (IGA approach with spline mappings available)
* Particle-in-cell method for kinetic species
* Exact conservation laws
* Heavy kernels pre-compiled with [Pyccel](https://github.com/pyccel/pyccel) to achieve near-Fortran performance
* MPI/OpenMP parallelization 

## Installation

* [Struphy user guide](https://clapp.pages.mpcdf.de/hylife/)

## Key references

* F. Holderied, S. Possanner, X. Wang, "MHD-kinetic hybrid code based on structure-preserving finite elements with particles-in-cell", [J. Comp. Phys. 433 (2021) 110143](https://www.sciencedirect.com/science/article/pii/S0021999121000358?via%3Dihub)

* F. Holderied, S. Possanner, "Magneto-hydrodynamic eigenvalue solver for axis-symmetric equilibria based on smooth polar splines", [IPP pinboard](https://users.euro-fusion.org/auth)

## Contributors

* Florian Holderied (since 2019)
* Stefan Possanner (since 2019)
* Xin Wang (since 2019)
* Benedikt Aigner (since 2021)
* Tin Kei Cheng (since 2021)
* Yingzhe Li (since 2021)
* Byung Kyu Na (since 2021)
* Dominik Bell (since 2022)

The project benefits from the constant advice of Yaman Güclü, Said Hadjout and Florian Hindenlang.

## License

Not yet published.

## Contact

* Florian Holderied [floho@ipp.mpg.de](floho@ipp.mpg.de)
* Stefan Possanner [spossann@ipp.mpg.de](spossann@ipp.mpg.de)
* Xin Wang [xin.wang@ipp.mpg.de](xin.wang@ipp.mpg.de)











