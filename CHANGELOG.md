## Version 1.9.3

* Make spline maps work, !188
* Bug fix in `send_recv_determine_mtbs` v2, !193
* New model: Linearized Vlasov Maxwell, !197
* Improved hdf5 file structure, !198
* Add new pyccel decorators, !199
* Added holes attribute to Particles6D class, !200
* Added GEMPIC doc, first jupyter notebook on derham, !201


## Version 1.9.2

* Use of mappings is now object-oriented, similar to other features in Struphy, see !181
* New Accumulator routines for pressure coupling scheme, see !182
* Improve handling of homogeneous Dirichlet boundary conditions, see !185
* Add particle pusher wrapper class and pusher unit tests, !186
* Added docker install mode, !187


## Version 1.9.1

* New ideal MHD equilibrium base class and four example subclasses
* Fixed eigenvalue solver for 2D axisymmetric ideal MHD equilibria
* Initialized new structure for particles object, added attribute `neighbours` for 26 geometric neighbours and internal function `_get_neighbours`
* Particle send/receive for domain decomposition with example
* Adapted `mat_vec_filler` and `filler_kernel` to stencil format
* Routines for assembling weighted mass matrices in arbitrary dimensions and with arbitrary combination of spaces
* New `Accumulator` class with ghost region sender to include corner blocks
* Cleaned spline evaluation functions and mappings_3d
* Improved and cleaned up `test_psydac_basics.py`
* New model linear MHD implemented
* MHD dispersion relations and example binary for running a linear MHD simulation
* Make ci exit on bash errors
* Add option to choose different iterative solvers in SchurSolver class
* Removal of mass matrices from Derham class
* Model specific analytical initial conditions
* Added homogeneous Dirichlet boundary conditions to model LinearMHD
* Enable Paraview


## Version 1.9.0

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