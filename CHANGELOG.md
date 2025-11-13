# Changelog

## Struphy 3.0.0 - 2025-11-13

* [PyPI](https://pypi.org/project/struphy/3.0.0)
* [Github pages](https://struphy-hub.github.io/struphy/index.html)
* [Github release](https://github.com/struphy-hub/struphy/releases/tag/v3.0.0)

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

### Headlines

* This is a test run for the relaease of Struphy 3.0 from the new Github repo



## Struphy 2.5.0 and prior releases

* See [Gitlab](https://gitlab.mpcdf.mpg.de/struphy/struphy/-/releases)
