## Version 2.3.2

Diff to previous release: Merge requests !521 - !564 


### User news

* Bug fixes !525, !546, !551, !557.

* Generate logical output in post-processing, i.e. variables on the computational unit cube `[0, 1]^3`.
The previous default of push-forwared physical output can now be obtained with the `--physical` flag, e.g. `struphy pproc -d sim_1 --physical`. !521

* Additional options in `struphy test GROUP` for testing of model subgroups. Use as `struphy test kinetic` for example. !532

* Restrict to `mpi4py<4.0.0`. !535

* Model `ViscoResistiveMHD`: Added an artificial resistivity term to stabilize the simulations. 
This term is given by $\nabla \times (\eta_a(\mathbf x) \nabla \times \mathbf B)$ where $\eta_a(\mathbf x) = \eta_a |\nabla \times \mathbf B|$. An additional term is also added to the entropy equation to keep energy preservation of the model. !539

* Scale the density in `Desc` equilibrium by a constant temperature (ideal gas law $p = n k_B T$). !542

* In addition to FEEC coefficients of the `StruphyModel.species`, any other FEEC variables can be saved during the simulation and post-processed as well. By adding the staticmethod `StruphyModel.diagnostics_dct`, we can define a new FEEC variable for a specific diagnostics, for example !534

```
    @staticmethod
    def diagnostics_dct():
        dct = {}

        dct['accumulated_magnetization']= 'Hdiv'
        return dct
```

This variable can be accessed through `StruphyModel.pointer['accumulated_magnetization']`.
For instance, we can save the accumulated vector by doing `self._ACC.vectors[0].copy(out=self._accumulated_magnetization)`. In the parameter file, the saving can be controlled via the new top-level key

```
diagnostics:
    accumulated_magnetization: {save_data: true}
```

* **Binomial filter** ("three-point filter") has been added for noise reduction. !534

* **Fourier filter** has been added for noise reduction. !547

* Introduced command-line arguments and integration for running with [**Likwid**](https://github.com/RRZE-HPC/likwid), which allows for measuring hardware counters (timing, performance, bandwidth, vectorization and more) for specific code-blocks. Called via the option `struphy run MODEL --likwid`. The likwid parameters can be included as a YAML file from the console with `--likwid-inp` or `--likwid-input-abs`. Parameter `--likwid-repetitions` is added to the console to run the same simulation multiple times for statistics. !537

* Added **domain cloning** support with appropriate MPI communicators. `Nclones` number of identical clones are created at the start of the simulation. Each clone contains `Np/Nclones` particles, but each clone contains the same fields. Each clone runs on a distinct group of processors, allowing for improved load balancing and simulation scalability without grid size restrictions. Resources: https://doi.org/10.1016/j.parco.2006.03.001. 
The number of clones should be included in the parameters file as `params['grid']['Nclones']`. !537

* Added a background magnetic field to `LinearVlasovAmpere`. !555

* Added the normalization with the rights units in `GVEC` equilibrium and the option to choose the density profile. !564



### Developer news

* Improved code structure of models. The addition of `Propagators` is now done through the static method `StruphyModel.propagators_dct()`, which
appears automatically in the doc. 
Propagator keyword arguments are now passed via the dict `StruphyModel._kwargs[PropagatorX] = ...`, where `PropagatorX` is a propagator class.
A value `None` in this dict inidicates that `PropagatorX` is not used in the model. !522 and !531

* Use pyccelized classes for pusher arguments (available since pyccel 1.12). New classes `DerhamArguments` and `DomainArguments` in new file `pusher_args_kernels.py` (-> will be pyccelized automatically). 
These classes hold the relevant arguments for pusher kernels. Re-factoring of `Pusher` class: new signatures of `__init__` as well as `__call__`.
Re-factoring of `Accumulator` class: new signatures of `__init__` as well as `__call__`. The method `accumulate` has been replaced by `__call__`. !533

* New class `mhd_equil.projected_equils.ProjectedMHDequilibrium`: has attributes that return the Derham spline coeffs of each MHDequilibrium callable. Projections are done with commuting projectors; polar splines extraction and `update_ghost_regions` is automated. Propagators can access these via `self.projected_mhd_equil`, see `PushGuidingCenterBxEstar` for an example. !536

* Introduced pyccel class `pic.pushing.pusher_args_kernels.MarkerArguments`; it holds all info regarding markers that is necessary in kernels, in particular `markers`, `n_markers`, `vdim` and the indices `buffer_idx`, `shift_idx = buffer_idx + 3 + vdim`, `residual_idx = shift_idx + 3` and `first_free_idx = residual_idx + 1`. The buffer_idx has already been used before: it yields the position after the "usual" marker attributes. The columns of each marker array are as follows: !536

```
    0:buffer_idx                -> usual attributes (eta, v, w0, etc.)
    buffer_idx:shift_idx        -> phase space coords at time t^n
    shift_idx:residual_idx      -> eta-shifts due to kinetic boundary conditions
    residual_idx:first_free_idx -> the residual in iterative solvers
    first_free_idx:-1           -> auxiliary positions for saving
    -1                          -> marker ID
```

* Moved optional arguments of a pusher kernel to the constructor, i.e. now we call just `self._pusher(dt)`. This makes for better practice of allocating all arrays in the constructor, and none during `__call__`. !536

* Changed the evaluation logic of the `eval_kernels` in `Pusher`: an MPI sort is performed before the call to each kernel. This allows to specify weights `alpha` for each evaluation: sorting before evaluation is according to `alpha[i]*markers[:, i] + (1 - alpha[i])*markers[:, buffer_idx + i]` for `i=0,1,2`. alpha must be between 0 and 1. The routine `Particles.mpi_sort_markers` has been adapted accordingly. !536

* Added new abstraction methods `Propagator.add_init_kernel` and `Propagator.add_eval_kernel` for easier creation of iterative Pushers. !536

* Use new MPCDF image `gitlab-registry.mpcdf.mpg.de/mpcdf/ci-module-image/gcc_12-openmpi_4_1`. 
Use artifacts instead of cache. !551

* Added magic methods` __imul__`, `__iadd__`, `__isub__` to `WeightedMassOperator` which are used in `SchurSolver.__call__` to do in-place updates, aka no new memory allocation. !557


