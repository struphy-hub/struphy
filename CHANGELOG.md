## Version 2.4.2

### Headlines

* enable Python 3.13 and `mpi4py >= 4.0` !652
* use new, smaller version of `psydac` from branch `max-model/psydac-for-struphy/devel-tiny` which does not depend on `sympde` !652
* added regular testing on macOS (with arm M4 chips) !655


### Other user news

* enable initialization of noise for kinetic backgrounds !644
* new option `--no-vtk` for `struphy pproc` !648
* `README.md` has been updated to contain a quick install and quick test, as well as a link to the new mailing list !649


### Developer news

* replaced `anaconda` with `waterboa` when loading modules in the mpcdf images of the CI !641
* added domain cloning to the model verification tests !640
* pyccelize kernels with OpenMP in Fortran !623
* added the job compile_timings to the CI pipeline which summarizes the compile time for C and fortran !642
* added unit test for console commands !570
* re-factoring of model testing: New function `wrapper_for_testing` in tests/util.py unifies the testing of all four model classes and will simplify refactoring in the future !649 
* new functions  `init_derham`, `_discretize_derham`, `_discretize_space` and class `DiscreteDerham`. These allow for the use of `devel-tiny` psydac branch. The full (old) psydac-for-struphy is used if `dev0` is absent from the psydac version number !652
* renamed `vector_space` to `coeff_space` and `ProductFemSpace` to `MultipatchFemSpace` everywhre; this corresponds to PR https://github.com/pyccel/psydac/pull/468 !653
* re-factoring of `.gitlab-ci.yml`, in particular making more use of templates and `!reference` !655


### Bug fixes

* a bug in the SPH pressure evaluation kernel has been corrected - the formulas contained an unnecessary multiplication by the weights and the wrong metric coefficient !649


