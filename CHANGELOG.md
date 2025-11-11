## Struphy 2.5.0

* [Struphy on PyPI](https://pypi.org/project/struphy/)
* [Struphy pages](https://struphy-hub.github.io/struphy/index.html)

### Headlines

* Base install and run without `mpi4py` !775
* Addition of a solver for saddle point problems based on the Uzawa algorithm !624
* New convergence tests for SPH density evaluation: all unit tests are done for the available SPH boundary conditions `periodic`, `mirror` (-> Neumann) and `fixed` (-> Dirichlet) !724 
* Integration of [RatGUI](https://rat-gui.com/) for magnetic coil fields !576

### Other user news

* Verification of model `Maxwell` with analytic solution of coaxial cable !733
* Improved auto-sampling of markers (-> importance sampling) !735
* Add `struphy params MODEL --check-file FILE` and the model name in the params file !707
* Store `Domain` and `Equilibrium` input paramaters unchanged !745

### Developer news

* Refactor console module !711
* Use `MarkerArguments` in `accum_kernels` !635
* Format all source files !737
* Code profiling: Improvements of the time traces !713
* Expose documentation and lint reports as artifacts in each MR !747
* Added `ssort` to the linters !762 
* Added a `Pyccelkernel` class with a __call__ method !759
* Added an `xp` module: this helps with importing `numpy`/`cupy` depending on the environment variable `ARRAY_BACKEND` !768

### Bug fixes

* Allow float evaluation in GVEC equilibrium !729
* Remove cyclic dependencies between folders !736
* Post-processing of multiple output folders fixed !744
* Remove deepcopy from `DESCunit` !772
