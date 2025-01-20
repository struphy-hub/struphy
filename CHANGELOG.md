## Version 2.4.0

### Headlines

* Remove `python<3.12` requirement and `numpy<2` requirements !600

* Use optional dependencies: unit testing can be enabled with `pip install .[test]` (or `pip install struphy[test]` from PyPI); usual development (testing + linting + formatting) is enabled by `pip install .[dev]`; building the doc is enabled by `pip install .[doc]`. These can be also combined, e.g. you get the full version (as until now) via `pip install .[dev, doc]`. !609

* Use new psydac fork https://github.com/max-models/psydac-for-struphy !563


### User news

* Reduce memory consumption at mpi sort markers and draw markers within the process domain !599

* Possible first guess in the solve of Interpolation/Histpolation matrix (used in polar splines mainly) !598

* Speedup and linearization on variational propagators !597

* Removes assertion that `Np` should be in params file !588

* New MHD tutorial notebook with slab dispersion relation !603

* Added `-v (--verbose)` flag to struphy run command and to StruphyModel base class; by default the major outputs of the model initialization are now suppressed (see the model tests for instance). !605

* New toy model `PressurlessSPH` : first try to sph models. New Particle class `HydroParticles`, New background `FluidEquilibrium`. Added the possibility to pass `moments: degenerate` to the loading of the particles. In this case the velocity will be initialized as a function of the position without any randomness. !579

* Basis Projection Operators with **local projectors**, based on quasi inter-/histopolation !562

* Added the linearized Vlasov-Maxwell model (same as linearized Vlasov-Ampère but with Maxwell step). Added background magnetic field to `VlasovAmpereOneSpecies` and `LinearVlasovAmpereOneSpecies`. Updated Documentation of all kinetic models with Maxwell/Ampere equation + Vlasov equation to have a consistent normalization !601


### Developer news

* Set ruff as the default option (which is used in the CI) for code formatting !592

* Check OpenMP pragma formatting with `struphy lint` !604

* Add a job that tests `make html` to the CI !606

* Remove `pytest-monitor` package and its use in console. This gets rid of the annoying pymon error when locally running parallel unit test. Also added the function `subp_run` which launches a subprocess and prints the command on screen. !605

* Add tests for console commands !570

* Scheduled CI pipelines can be started from Gitlab by clicking Pipelines --> Run Pipeline !600


### Bug fixes

* Fix DESC speedup - troubleshoot why it takes a LinearMHD simulation too long to ramp up when using a DESC equilibrium on many processes. !594

* Resolve "Linting of OpenMP pragmas" !587

* Fix libpython error: the error occured since the kernels in `psydac-for-struphy` was pyccelized without the --libdir LIBDIR flag. This meant that LD_LIBRARY_PATH had to be manually set. !610

