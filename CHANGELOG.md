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