## Version 2.3.0

### Core changes

* Refactor FEEC initialization; FEEC variables are initialized as `background + perturbation`. Details can be found in a [new section of the documentation](https://struphy.pages.mpcdf.de/struphy/sections/subsections/initial_conditions.html). !490

* Added `bckgr_params` and `pert_params` to `Field.create_field()`; new types `LogicalConst` and `MHD` for background params keyword. !490

* Remove `electric_equilibrium`. !490

* Moments of `Maxwellians` can be set from `MHDequilibrium`. In the background parameters for `Maxwellians`, there is the new option `mhd` (instead of a constant value) for the moments. The perturbation is added on top of the `MHDequilibrium`. The broadcasting has been taken care of in the `Maxwellian` base class. Unit test have been added for `Maxwellian6D` and `Maxwellian5D` !486

* `self.f_backgr` is now always (!) defined in the `Particles` base class. !486

* The signature of `Maxwellians` has been changed to look like `def __init__(self, maxw_params=None, pert_params=None, mhd_equil=None)`. Added attribute `moment_factors` too. !486

* Initialization of distribution function in the `paramaters.yml` file is now done using the keywords `background` and `perturbation` for each particles species. All functions that were possible in the `fluid` and `em_fields` initialization (can be found under `struphy.initial.perturbations`) are available here. !478

* Update psydac to version 0.1.12 : add the possibility to use `recycle` flag in linear solvers and use it in Newton algorithm. !482

* `Maxwellain5DUniform` is renamed to `Maxwellian5D` and fixed to be analogous to `Maxwellian6D`. New perturbation function `ITPA_density` is implemented so then `Maxwellian6DITPA` and `Maxwellian5DITPA` are removed. !484

* Add `time_state` to `StruphyModel` and `Propagator`. !477

* Add a new preconditioner `MassMatrixDiagonalPreconditioner` which relies on the methodoloy described [here](https://www.sciencedirect.com/science/article/abs/pii/S0898122120304715?via%3Dihub). !475

* Add `StencilDiagonalMatrix` and the method `diagonal` to `StencilMatrix` and `BlockLinearOperator` to our psydac. !474

* Add a `diagonal` method to `StencilMatrixFreeMassOperator` using a kernel to compute efficiently the diagonal. Add a `matrix_free` option to `WeightedMassOperators` and test it in test_mass_matrices.py !472


### Model specific changes 

* Remove temporaries from Variational Propagators. !488

* Use a "hand made" preconditioning for the solvers of the system in the Newton iteration of `Variational` models by taking advantage that the Jacobian is mostly a mass matrix for which we already have good preconditioners. This simple change allow for roughly 80% speedup as most of the time (90%) was spent on GMRES solve. !487

* Rename toy model `DriftKinetic` to `GuidingCenter`. !486

* Added newton solver for both `VariationalEntropyEvolve` and `VariationalMagFieldEvolve`. The default solver is Newton but Picard can still be use by passing `'non_linear_solver' = 'Picard'` in the parameters of the propagator. !485

* The propagator `MagnetosonicCurruntCoupling5D` has been fixed. A new strategy based on accumulation instead of projection is used. !465

* Updated the `VariationalDensityAdvection` propagator to use a Newton algorithm to solve the non-linear system instead of the previous Picard iteration. !482

* Add new model `VlasovAmpereOneSpecies`. It consists of the two propagators `PushEta` and `VlasovAmpere`. The latter propagator was formally called `VlasovMaxwell`, but was now renamed. Added parameters and diagnostics for linear Landau damping with VlasovAmpereOneSpecies to [struphy-simulations](https://gitlab.mpcdf.mpg.de/struphy/struphy-simulations/-/tree/main/VlasovAmpere?ref_type=heads). !480

* Renamed kinetic model `VlasovMaxwell` to `VlasovMaxwellOneSpecies`. !480

* Commented model `VlasovMasslessElectrons` (not working). !480

* Rename `epsilon_unit` to `epsilon` (same for `alpha`). !480

* Divided `ImplicitDiffusion` equation by $\Delta t$ to avoid needing 'dt=1'. in parameters. !477

* Added new field propagator `TimeDependentSource`. Added a time dependent source to the toy model `Poisson`. !477

* Fixed and enhanced Poisson tests; 1D convergence is proved for Cuboid and Orthogonal mappings with all three possible boundary conditions. !477

* Implement a newton algorithm for the nonlinear solve in `VariationalMomentumAdvection`. This would probably help improving the stability of the algorithm. The new `BracketOperator` was needed for this. !476

* In variational models, parameters can now be passed separately for each solver. !473

* The signature of `ImplicitDiffusion` now includes two matrices `A1_mat` and `A2_mat` and three parameters `sigma_1`, `sigma_2` and `sigma_3`, the rhs is called rho. !467


### Documentation, tutorials, testing, etc.

* New section "Userguide: Boundary conditions". !491

* New sections "Userguide: Initial conditions" and "Python modules: Fluid backgrounds". !490

* Updated all docstrings in `WeightedMassOperators` to the correct Struphy notation. !476

* Changed the section name "Inventory" to "Python modules" and included all available modules in the doc (many were not included until now). !476

* Speed-up CI pipeline. Run only `Ubuntu:latest` and `default`, both only with C-compilation. More thorough testing is done on the week-end with a cron-job ("scheduled" pipeline). !471



