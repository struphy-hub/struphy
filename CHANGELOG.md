## Version 1.9.9

* Add `dims_mask` to grid parameters; lets you choose spatial directions to decompose with MPI, e.g. `dims_mask : [True, True, False]` does not decompose the third direction. !321
* Improved batch run mode; for batch script runs, the batch script is now first copied to the output folder and submitted in this output folder. With this, the files `sim.out`, `sim.err` and `struphy.out` are written in the output folder, and also the batch script information is not lost. !322
* A new function `print_plasma_params` is implemented; print volume averaged plasma parameters(characteristics) for each species of the model. (When there is at least one species). !323

    ```
    Global parameters:
        - plasma volume
        - transit length
        - magnetic field

    Species dependent parameters:
        - mass
        - charge
        - density
        - pressure
        - thermal energy
        - thermal speed
        - thermal frequency
        - cyclotron frequency
        - larmour radius
        - rho/L
        - plasma frequency

    in case of MHD species
        - alfven speed
        - alfven frequency
    ```
* Time resolution flag in output data and post processing; new flag `-s, --save-step` in struphy run command to enable skipping in output data, e.g. `struphy run Maxwell -s 4` saves output data every fourth time step. New flag `-s, --step` to enable skipping in creation of post-processing data, e.g. `struphy pproc -d sim_1 -s 2` creates post-processing data (grids, evaluated fields, vtk files, etc.) for every second step in the hdf5 files. !326
* Reform Particles Class; new abstract methods:
`draw_markers`, `svol` and `s0`. !327  


## Version 1.9.8

* Additional examples `struphy example orbits_tokamak` and `struphy example gc_orbits_tokamak` using `EQDSKequilibrium` and geometry `Tokamak`. !297
* Two discrete gradients methods for particle pushing: `discrete_gradients` and `discrete_gradients_faster`. !298
* Added setters for `self.domain`, `self.derham`, `self_mass_ops` and `self.basis_ops` to the `Propagator` base class. !299
* Input to each propagator is `(var1, var2, ... , **params)` where everything other than updated variables must get a keyword. !299
* Added new iterative solver classes `ConjugateGradient`, `PConjugateGradient`, `BiConjugateGradientStab` and `PBiConjugateGradientStab` which create temporary vectors needed in the iteration loop at the `__init__()` stage and use in-place operations in the iteration loop. !300
* Remove all temporary vectors in `ShurSolver` and propagators `Maxwell`, `ShearAlfvén` and `Magnetosonic` and use in-place updates where possible. !300
* Enhanced field line tracing and MHD inheritance structure for axisymmetric MHD equilibria: different parametrizations for the angular coordinate are now possible (e.g. straight field line coordinates). !301
* New domain class `Tokamak` which accepts as an input any instance of AxisymmMHDequilibrium and it performs the desired field line tracing at the initialization step. !301
* New propagators `CurrentCoupling5DCurrent1`, `CurrentCoupling5DCurrent2`. !302
* Helper function `setup_domain_mhd()` for creation of the domain and MHD equilibrium form simulation parmeters. !303
* Added notebook for 1d mass and collocation matrices. !304
* Restart of simulations are now possible: e.g. `struphy run Maxwell -o sim_1 -r` will restart the simulatinn in `sim_1` if the simulation time wants to be increased for instance. !307
* Improve Basis projection operators, new doc style. !308
* `WeightedMassOperator` now has a single flag `weights_info` in `__init__()` which allows for three different kinds of initializations as described in the docstring: `None` for all blocks, a string for symmetry block matirces, or callables for identifying zero blocks. !314
* Added kernels for the `StencilMatrix` methods `.dot` and `.transpose` that can be pre-compiled. !315
* New Python based command line interface using `argparse` which replaces the old binary file `struphy`. !316
* Improved profiling `struphy profile`. !317

## Version 1.9.7

* Three types of kinetic species possible (full_f, control_variate, delta_f). Model using control variate must implement the background function in the accumulation. Delta-f models are initialized as full-f and then lets the developer overwrite/extend the set_initial_conditions() function of the base class. !269 !283
* New model `LinearMHDVlasovCC` (hybrid linear MHD + 6d full-orbit energetic ions) available, including option to run with control variate !270, !273, !279
* Example command line `struphy example linearmhdvlasov_cc` for model `LinearMHDVlasovCC`, including plots of the energy evolution (comparison to analytical growth rate) and distribution function snapshots !274
* New propagators `CurrentCoupling6DCurrent` and `CurrentCoupling6DDensity` !273
* New particles base class `Particles`. `Particles6D` and `Particles5D` are child classes thereof !273
* Model drafts `LinearMHDDriftkineticCC`, `ColdPlasma`, `ColdPlasmaVlasov` and `Hybrid_fA` !271, !286
* New MHD equilibrium class `GVECequilibrium` for loading 3D GVEC equilibria !277
* Drafts for propagators in model `LinearMHDDriftkineticCC`: `StepPushDriftkinetic1`, `StepPushDriftkinetic2` and `CurrentCoupling5DCurrent1` !278
* Method `accumulate` in class `Accumulator` can now handle analytical control variates !279
* New domain base classes `PoloidalSpline`, `PoloidalSplineStraight` and `PoloidalSplineTorus` and new domains `IGAPolarCylinder`, `IGAPolarTorus`, `GVECunit` !281
* Toroidal mappings can be split into a half, third, quarter etc. !281
* Renamed `plot_equil` to `show` and improved the plotting experience of MHD equilbria. !281
* New command line interface `struphy units <MODEL>` for printing the physical units of the model unknowns !282
* Added post-processing for control_variate and delta_f methods, saved are both the delta and full distribution functions binnings !283
* The send/receive of ghost regions is now done with the Psydac method `exchange_assembly_data()` for stencil objects instead of STRUPHY's own implementation !287
* Kernels in `mass_kernels.py` have been updated due to the new quadrature grid decomposition !287
* Class `WeightedMassOperator` now also accepts numpy arrays as weights. These numpy arrays are the weight values at the quadrature points !287
* New mapping `EQDSKTORUS` and `EQDSKequilibirum(CartesianMHDequilibrium)` with Cartesian MHD variables. Extensive plotting can be shown at instantiation (show=True) !289
* print the model units at the beginning of a simulation (same as command `struphy unit <MODEL>`) !293
* Default parameter seetings in domains and MHD equilibria classes have been improved !294

## Version 1.9.6

* Implementation of the DriftKinetic particle pusher with unperturbed magnetic field !251
* Unifying pressure coupling schemes !252
* Change of StruphyModel base class: there are now three categories of unknowns: `self.em_fields`, `self.fluid` and `self.kinetic` !253
* Add polar projectors, polar basis projection operators and polar unit tests !254, !255, !256
* Substep efield weights for linearized Vlasov Maxwell !257
* Electric Background: implemented a base class and analytical class for electric backgrounds; base class gives electric potential phi as 0-form or 3-form, and electric field as 1-form !258
* TAE example tokamak: added complete example for model LinearMHD that is initialized with a TAE eigenfunction obtained from the MHD eigenvalue solver !260
* Linear Vlasov Maxwell other substeps !261
* Current coupling dispersion relation and MHD continuous spectra in cylinder and slab !263
* Add new model--hybrid_fA, rotation sub-step, get density from particles (hat...) !264
* Enable numerical MHD equilibria !265
* Pure python analytical kinetic functions and unification of domain methods !266

## Version 1.9.5

* Generalized eta pusher for arbitrary s-stage explicit Runge-Kutta methods and implicit vxb pusher !234
* Loading of MHD eigenfunction as initial condition for axisymmetric equilibria !235
* Separate propagators into "fields", "markers" and "coupling" !238
* Finish polar extraction and linear operators !239
* adding draft of guiding center equations with explicit solver !240
* Polar mass matrices and projectors with units tests !242
* Preconditioner for polar mass matrices !243
* Make pyccel work on Cobra !245
* Draft of guiding center models with implicit scheme !246
* Added --no-build-isolation to psydac install !249

## Version 1.9.4

* new model PC_LinearMHD_Vlasov_full is implemented !212
* Added particle orbit post processing tool post_process_markers() to post_processing_tools.py, extended the method show() in the domain base class !213
* Kinetic boundary conditions implemented !215
* Banana orbit example binary !216
* Step magnetosonic hcurl added !217
* Step pressurecoupling h1vec added !219
* Add __call__ routine to Field object !220
* Marker binning in arbitrary directions implemented !221
* Added Ohm propagator for cold plasma !224
* PC dispersion relation added !226
* Unify filler kernels !227
* Enable p=1 spline degree !228
* PC_LinearMHD_Vlasov model (only perp U) is implemented !229

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