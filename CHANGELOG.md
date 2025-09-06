## Version 2.4.5

### Headlines

* Enable `pyccel 2.0` (https://github.com/pyccel/pyccel) !678
* New interface to `gvec` Python API (https://gitlab.mpcdf.mpg.de/gvec-group/gvec) !705 and !723

### Other user news

* Rename `Particles` parameter `eps -> bufsize` for more clarity !690
* Particle simulations: save weights together with orbits !690
* Add `mpi_dims_mask` for mesh-less simulations !690
* Added `gravity` and `thermodynamics` (isothermal or polytropic) options to `PushVinSPHpressure` !690
* Add options `gaussian_xy` and `step_function_x` to the `ConstanVelocity` fluid background !690
* Allow for running `struphy run` without specifying the model if `model: MODEL` is specified in the params !708
* Remove the `-d` flag from `struphy pproc`, now just run `struphy pproc sim_1 sim_2 ...` !709
* New variational MHD model with $q = \sqrt{p}$ as a variable !697

### Developer news

* New methods in `Particles`: `n_mks_on_each_proc`, `n_mks_on_clone`, `n_mks_on_each_clone` and `n_mks_global`; they refer to valid markers (not holes or ghosts) !690
* The CI pipeline no longer runs on every push, only on pushes to branches which currently have a MR with devel or master !719

### Bug fixes

* Print user warning and temrinate MPI when out-of-bounds error occurs due to load imbalance in particles comm !690
* Do NOT duplicate marker IDs when working with clones !690
* Enable running `cprofile` without a grid !690
