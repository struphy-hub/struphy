# Changelog


## Struphy 3.0.3 - 2026-02-23

* [PyPI](https://pypi.org/project/struphy/3.0.3)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.3)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.2...v3.0.3)

### Headlines

1. New class `Simulation` inherits the generic `SimulationBase`, both are in the new folder `struphy/simulations/`. The most import methods are:
    * `Simulation.run()`
    * `Simulation.pproc()` 
    * `Simulation.load_plotting_data()`
    * `Simulation.spawn_sister()` (my new favorite!)
    
These are tested in the new tutorials: https://github.com/struphy-hub/struphy-tutorials/tree/use-species-properties

The file `main.py` has been deleted.

The `Simulation` takes a model as input. Other API classes are passed as well (see tutorials). 
The model is viewed as everything related to the PDE, i.e. its variables, initial conditions etc. The simulation deals with the rest (geometry, derham, environment etc.)

Some important changes to the logic: The model does not have access to `derham`, `mass_ops` etc. anymore, these can be called from `Propagator` when needed. Solves that need to happen before the time stepping (like initial Poisson solves) are moved to `model.allocate_helpers()`.

2. The default launch file has been improved. See Tutorial 2 or test with `struphy params MODEL`. The files in the submodule struphy-parametershave been adapted.

3. Several new classes have been introduced for post processing and plotting data, see `post_processing_tools.py`. The most important ones are `PostProcessor` and `PlottingData`. Dictionaries in the plotting data have been replaced by classes. Many classes now feature the `__repr__` dunder for customized printing.


**API changes:**

New classes exposed: `Simulation`, `PostProcessor` and `PlottingData`.


### User news

* Add `set_zero_velocity` argument into `LoadingParameters`, enforcing velocities of all particles along specified axis to always be zero: https://github.com/struphy-hub/struphy/pull/176 
* New model `ViscousEulerSPH` replaces `EulerSPH`. The evaluation of the viscosity tensor has been implemented and tested for SPH methods. Unit tests for evaluation of the fluid velocity and its gradients (needed in the viscosity tensor) have been improved: https://github.com/struphy-hub/struphy/pull/160

### Developer news

* Use `pyccel 2.1`: https://github.com/struphy-hub/struphy/pull/153
* Added three submodules: `struphy-parameter-files`, `struphy-tutorials` and`feectools`. The Struphy repo should be cloned with `git clone --recurse-submodules https://github.com/struphy-hub/struphy.git` to init and update the submodules. Also, run `git submodule update` regularly to get updates from the submodules. See https://github.com/struphy-hub/struphy/pull/154
* Introduced class `options.LiteralOptions` for parsing literals. Moved `Units` to `physics.py`: https://github.com/struphy-hub/struphy/pull/167


### Bug fixes

* Use `struphy.io.options.Units` in equils. This enables the use of GVEC, EQDSK and DESC in the new framework: https://github.com/struphy-hub/struphy/pull/158


## Struphy 3.0.2 - 2026-02-06

* [PyPI](https://pypi.org/project/struphy/3.0.2)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.2)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.1...v3.0.2)

### Headlines

* Added a public API. This allows imports like `from struphy import equils`: https://github.com/struphy-hub/struphy/pull/168
* New default compile language is Fortran: https://github.com/struphy-hub/struphy/pull/158
* Moved each model to its own file. Calling sub-processes must be avoided in the future because of incompatibility with MPI: https://github.com/struphy-hub/struphy/pull/152

### User news

* Added binning of higher order moments (current density, energy tensor) of f and delta f: https://github.com/struphy-hub/struphy/pull/162 

### Developer news

* Use `pyccel 2.1`: https://github.com/struphy-hub/struphy/pull/153
* Added three submodules: `struphy-parameter-files`, `struphy-tutorials` and`feectools`. The Struphy repo should be cloned with `git clone --recurse-submodules https://github.com/struphy-hub/struphy.git` to init and update the submodules. Also, run `git submodule update` regularly to get updates from the submodules. See https://github.com/struphy-hub/struphy/pull/154
* Introduced class `options.LiteralOptions` for parsing literals. Moved `Units` to `physics.py`: https://github.com/struphy-hub/struphy/pull/167


### Bug fixes

* Use `struphy.io.options.Units` in equils. This enables the use of GVEC, EQDSK and DESC in the new framework: https://github.com/struphy-hub/struphy/pull/158



## Struphy 3.0.1 - 2025-12-11

* [PyPI](https://pypi.org/project/struphy/3.0.1)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.1)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v3.0.0...v3.0.1)

### Headlines

`Psydac` is now installed via `pip` from our fork renamed to `feectools`, which is published on PyPI. This avoids installing from a `.whl` file.
Functionality remains unchanged, but future releases of Struphy are much easier and quicker. We have kept the option to install the upstream `psydac`, once it is also on PyPI.
See https://github.com/struphy-hub/struphy/pull/147.

### User news

None

### Developer news

* Removed legacy code (eigenvalue solver): https://github.com/struphy-hub/struphy/pull/129
* Add context manager to h5py.File() calls: https://github.com/struphy-hub/struphy/pull/135
* Fix undefined variables: https://github.com/struphy-hub/struphy/pull/141

### Bug fixes

* Fix setter in DESCequilibirum, update quickstart guide: https://github.com/struphy-hub/struphy/pull/132
* Set defaults for given_in_basis: "0" for scalar and "v" for vector-valued: https://github.com/struphy-hub/struphy/pull/136
* Fix the restart function: https://github.com/struphy-hub/struphy/pull/143


## Struphy 3.0.0 - 2025-11-13

* [PyPI](https://pypi.org/project/struphy/3.0.0)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v2.5.0...v3.0.0)

### Headlines

Struphy 3 represents a major refactoring with breaking changes with respect to Struphy 2, in particular:

* The `.yml` parameter files cannot be used anymore. Simulation parameters have to be transferred to the new `.py` launch files that are generated from `struphy params MODEL`. See the [Struphy README](https://github.com/struphy-hub/struphy) for a quick introduction.
* The console command `struphy run ...` has been deprecated. The new way to launch simulations is by executing the `.py` launch file, for instance with `python params_MODEL.py`.
* Other deprecated console commands are `struphy pproc` and `struphy units`. Post-processing is now done through the API via `main.pproc()`.
* The Struphy repo has moved to [Github](https://github.com/struphy-hub/struphy). The [old Gitlab repo](https://gitlab.mpcdf.mpg.de/struphy/struphy) will persist but not be maintained any longer. Issues, discussion and PRs will solely take place on the new Github repo.

### User news

* Please consult the [Struphy README](https://github.com/struphy-hub/struphy) and links therein to get familiar with the new workflows. 
* New tutorials can be found on [mybinder](https://mybinder.org/v2/gh/struphy-hub/struphy-tutorials/main).

### Developer news

Struphy has been refactored with the following principles in mind:

* get rid of console commands and increase the use of the Struphy API wherever possible
* become even more object-oriented
* use `Classes` instead of `dicts` wherever possible
* use `Literals` to show options for string arguments

In Struphy 3, models feature the following important objects:

* `ParticleSpecies`, `FieldSpecies`, `FluidSpecies`

Each species is a collection of Variables:

* `PICVariable`, `FEECVariable`, `SPHVariable`

These variables are updated by `Propagators`. All options for a simluation can be set in the new `.py` launch file.

### Bug fixes

* Incorporate psydac updates: https://github.com/struphy-hub/struphy/pull/109
* Auto install Psydac on first Struphy import: https://github.com/struphy-hub/struphy/pull/118
* Remove MPI Barrier responsible for deadlock: https://github.com/struphy-hub/struphy/pull/121 


## Struphy 2.6.0 - 2025-11-12

* [PyPI](https://pypi.org/project/struphy/2.6.0)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v2.6.0)
* [Diff to previous release](https://github.com/struphy-hub/struphy/compare/v2.5.0...v2.6.0)

### Headlines

* This is a test run for the relaease of Struphy 3.0 from the new Github repo


## Struphy 2.5.0 and prior releases

* See [Gitlab](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/releases)
