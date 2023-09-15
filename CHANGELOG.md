## Version 2.0.3

### Core changes

* Removed psydac submodule. !397

* Require `pyccel<1.8.0` to have faster compilation. !397

* Added a draft for `Particles.boundary_transfer()`. This method transfers particles to the opposite side of the hole (around the magentic axis), where the magnitude of the magnetic field is the same. !398

* Console command `struphy examples` has been removed. !399

    The content of the examples can now be found in tutorial notebooks. These notebooks can be seen in the Struphy documentation under /Tutorials.
    The data necessary to run the notebooks is created by running the new console command `struphy tutorials [-n N]`; when the option `-n N` is given, only the simulations for the tutorial number `N` is executed. The output is stored in the install path under `io/out/tutorial_N`.

* Renamed `fourier_1d` -> `power_spectrum_2d` !399

* Initialize skew symmetric S-matrix with `np.zeros((3, 3))` in kernels. Otherwise the diagonal might get filled with unwanted non-zeros by the system. Also fixed minus sign bugs in `gc_discrete_gradient` kernels. !402

* Use newer modules on the mpcdf clusters: `module load gcc/12 openmpi/4 anaconda/3/2023.03 git pandoc`. The module `mpi4py` does not need to be loaded. !402

* Modified initialization sequence of FEEC variables to be more flexible. The init `type` can now be a list; for each entry in the list, parameters have to be given, and different `comps` can be set to `True`. !403

* The FEEC init type `TorusModesCos` has been added. !403

* Speed up marker evaluation a lot by using more `numpy`; also save `.npy` files aside `.txt` for markers. !404

* Moved `setup` and `output_handling` to folder `io/`. !404

* Move `struphy/models/main.py` to `struphy/main.py` !404

* Remove `params_*` files from `io/inp/`. !404


### Model specific changes

* Small adaptions in `LinearVlasovMaxwell` and `DeltaFVlasovMaxwell`. !396 


### Documentation, tutorials, etc.

* Scaling test results of `LinearMHD` model have been added [here](https://struphy.pages.mpcdf.de/struphy/sections/performance_tests.html#linearmhd). !392

* Change doc appearance to Python style. !397

* Included [notebook tutorials](https://struphy.pages.mpcdf.de/struphy/sections/tutorials.html) in doc via Pandoc. !399

### Repo struphy-simulations, new files:

None.
